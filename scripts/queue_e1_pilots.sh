#!/usr/bin/env bash
# Day GPU queue, 2026-10-21 (DECISIONS.md D94 approved, D97). The R3-3 optimization pilots, each E1a with one change:
#   1. GPU: E1c train (scale_rewards: batch), 300 steps
#   2. GPU: E1c evals, the corrected eval (D67): final_adapter -> final_4bit, checkpoint-150 -> step150_4bit
#   3. GPU: E1d train (learning rate 5e-5), only if >= E1D_MIN_LEFT_S (3 h) remain before GPU_DEADLINE
#   4. GPU: E1d evals, as 2
#   5. CPU: compare_pilots (D94's rho tests, Holm over the two; a pilot that did not run counts as p = 1)
# Every stage is a child whose PID goes to $LOG/<stage>.pid and is waited on with `wait` (never pgrep -f). Every GPU
# stage gets the in-stage temperature guard (scripts/gpu_temp_guard.sh, D96: GPU > 87 C -> its process group is
# stopped for 10 min); GPU temp > 87 C before a stage -> pause 10 min. Training stops cleanly (--stop-at) early enough
# to leave EVAL_RESERVE_S for its two evals; the hard timeout is HARD_EXTRA_S later (the final save + fingerprint).
# A failed stage is logged BLOCKED and the next independent stage goes on (E1d does not depend on E1c; an eval needs
# its own adapter). After every stage the queue commits and pushes its outputs under a flock.
#
# usage:  nohup scripts/queue_e1_pilots.sh > results/queue_e1_pilots/nohup.log 2>&1 &
#         DRY_RUN=1 scripts/queue_e1_pilots.sh     # print the plan, run nothing
set -u
cd "$(dirname "$0")/.."
LOG=${QUEUE_LOG:-results/queue_e1_pilots}
GIT_LOCK=${GIT_LOCK:-.git/queue-commit.lock}
EVAL_RESERVE_S=${EVAL_RESERVE_S:-1200}   # two evals (~2 min each measured, D67) + commits + slack
HARD_EXTRA_S=${HARD_EXTRA_S:-900}        # hard timeout = soft stop + this (the step in flight, save, fingerprint)
E1D_MIN_LEFT_S=${E1D_MIN_LEFT_S:-10800}  # your rule: E1d starts only if >= 3 h remain before the deadline
EVAL_TIMEOUT_S=${EVAL_TIMEOUT_S:-1200}
DRY_RUN=${DRY_RUN:-0}
CO_AUTHOR="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
PY="uv run python"
T=results/train
mkdir -p "$LOG"

ist() { TZ=Asia/Kolkata date -d "@$1" +%H:%M; }
iso() { TZ=Asia/Kolkata date -d "@$1" --iso-8601=seconds; }
log() { echo "[$(TZ=Asia/Kolkata date +%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }
left() { echo $((DEADLINE - $(date +%s))); }
cool() { [ "$DRY_RUN" = 1 ] && return; while t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "${t:-0}" -gt 87 ]; do
           log "GPU at ${t}C > 87C: pausing 600s"; sleep 600; done; log "GPU at ${t}C, memory.used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; }

run_stage() {  # run_stage NAME TIMEOUT_S CMD...: child process, PID file, wait; returns the child's exit code
  local name=$1 limit=$2; shift 2
  if [ "$DRY_RUN" = 1 ]; then log "DRY $name (timeout ${limit}s): $*"; return 0; fi
  if [ "$limit" -lt 300 ]; then log "skip $name: only ${limit}s left"; return 98; fi
  log "start $name (timeout ${limit}s): $*"
  timeout "$limit" "$@" > "$LOG/$name.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$LOG/$name.pid"
  [ "${GUARD:-0}" = 1 ] && { scripts/gpu_temp_guard.sh "$pid" "$LOG/chain.log" > /dev/null 2>&1 & }
  wait "$pid"; local rc=$?
  log "end $name: exit $rc"
  return $rc
}

commit_stage() {  # commit_stage MESSAGE PATH...: add the paths that exist + queue logs, commit, push (under a flock)
  local msg=$1; shift
  if [ "$DRY_RUN" = 1 ]; then log "DRY commit: $msg"; return 0; fi
  local paths=() p
  for p in "$@"; do [ -e "$p" ] && paths+=("$p"); done
  (
    flock -w 600 9 || { echo "git lock timeout"; exit 1; }
    git add -- "${paths[@]}" "$LOG"/*.log
    if git diff --cached --quiet; then echo "nothing new"; exit 0; fi
    git commit -q -m "$msg" -m "$CO_AUTHOR" || exit 1
    for i in 1 2 3; do git push -q origin main && { echo "pushed"; exit 0; }; sleep 30; done
    echo "PUSH FAILED"; exit 1
  ) 9>"$GIT_LOCK" >> "$LOG/git.log" 2>&1
  log "commit '$msg': $(tail -1 "$LOG/git.log")"
}

steps_done() { [ -f "$1/summary.json" ] && $PY -c "import json,sys; print(json.load(open(sys.argv[1]))['steps_completed'])" \
                 "$1/summary.json" 2>/dev/null || echo "?"; }

train_stage() {  # train_stage NAME CONFIG RUN_DIR: train to 300 steps or the soft stop, gzip the tape, commit
  local name=$1 cfg=$2 run=$3 stop rc
  cool
  stop=$((DEADLINE - EVAL_RESERVE_S))
  if [ $((stop - $(date +%s))) -lt 1800 ] && [ "$DRY_RUN" != 1 ]; then
    log "BLOCKED $name: only $(( (stop - $(date +%s)) / 60 )) min before its stop"; return 97; fi
  log "$name: config $cfg, soft stop $(ist "$stop") IST"
  GUARD=1 run_stage "${name}_train" $((stop - $(date +%s) + HARD_EXTRA_S)) $PY -m physcal.train.e1 train --config "$cfg" \
    --stop-at "$(iso "$stop")"; rc=$?
  [ "$rc" != 0 ] && log "BLOCKED? ${name}_train exit $rc (see $LOG/${name}_train.log)"
  [ "$DRY_RUN" != 1 ] && [ -f "$run/rollouts.jsonl" ] && gzip -kfc "$run/rollouts.jsonl" > "$run/rollouts.jsonl.gz"
  [ "$DRY_RUN" != 1 ] && log "$name: $(steps_done "$run") steps completed; $(ls "$run"/permanent_checkpoints 2>/dev/null | tr '\n' ' ')"
  commit_stage "$name (D94): training, $(steps_done "$run") steps (exit $rc)" "$run/summary.json" \
    "$run/train_metrics.jsonl" "$run/memory.jsonl" "$run/events.jsonl" "$run/rollouts.meta.json" "$run/rollouts.jsonl.gz"
  return $rc
}

eval_stages() {  # eval_stages NAME RUN_DIR: the corrected eval (D67) of the final and the step-150 adapter
  local name=$1 run=$2 spec rc
  for spec in "final_adapter final_4bit" "permanent_checkpoints/checkpoint-150 step150_4bit"; do
    set -- $spec
    if [ "$DRY_RUN" != 1 ] && [ ! -d "$run/$1" ]; then log "BLOCKED ${name}_eval_$2: missing $run/$1"; continue; fi
    cool
    GUARD=1 run_stage "${name}_eval_$2" "$EVAL_TIMEOUT_S" $PY -m physcal.train.e1 eval --run "$run" --adapter "$1" \
      --out-name "$2"; rc=$?
    [ "$rc" != 0 ] && log "BLOCKED? ${name}_eval_$2 exit $rc (see $LOG/${name}_eval_$2.log)"
    commit_stage "$name (D94): corrected eval $2, 4-bit unmerged (exit $rc)" "$run/e1_eval/$2"
  done
}

echo $$ > "$LOG/queue.pid"
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-14:00}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "queue start; GPU deadline $(ist "$DEADLINE") IST ($(( $(left) / 60 )) min left); queue PID $$"
export HF_HUB_OFFLINE=1   # every model is local; never resolve anything online (D78)

# ---- 0. inputs that must exist; runs that must not ---------------------------------------------------------------------
for f in results/e1/data/train_prefixes.jsonl results/e1/data/eval_prefixes.jsonl \
         results/sft/qwen2.5-math-1.5b-instruct/merged/config.json "$T/e1a-sigma-only-free-seed0/e1_eval/final_4bit/rows.jsonl"; do
  [ -e "$f" ] || log "BLOCKED: missing $f"
done
for r in e1c e1d; do
  [ -e "$T/$r-sigma-only-free-seed0/summary.json" ] && log "NOTE: $T/$r-sigma-only-free-seed0 already has a summary.json"
done

# ---- 1-2. E1c ------------------------------------------------------------------------------------------------------------
E1C=$T/e1c-sigma-only-free-seed0
train_stage e1c grpo_e1c "$E1C"
eval_stages e1c "$E1C"

# ---- 3-4. E1d (only with >= 3 h left) --------------------------------------------------------------------------------------
E1D=$T/e1d-sigma-only-free-seed0
if [ "$(left)" -ge "$E1D_MIN_LEFT_S" ] || [ "$DRY_RUN" = 1 ]; then
  train_stage e1d grpo_e1d "$E1D"
  eval_stages e1d "$E1D"
else
  log "SKIP e1d: $(( $(left) / 60 )) min left before the deadline, under $((E1D_MIN_LEFT_S / 60)) min (your rule)"
fi

# ---- 5. CPU: compare_pilots (D94) --------------------------------------------------------------------------------------------
run_stage compare_pilots 3600 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
  nice -n 19 $PY -c "from physcal.train.e1_compare import compare_pilots as c; import json; print(json.dumps(c()['decision_rule']))" \
  || log "BLOCKED compare_pilots (see $LOG/compare_pilots.log)"
commit_stage "E1c / E1d pilots (D94): rho vs E1a, Holm, decision rule" results/e1/pilots
log "queue done"

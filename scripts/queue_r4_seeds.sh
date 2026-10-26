#!/usr/bin/env bash
# Night GPU chain, 2026-10-26/27 (DECISIONS.md D118): the R4-3 confirmatory run's first night.
#   1-4. SFT D-pos-v2 seeds 1, 2, then B-v2 seeds 1, 2 (stage R4's recipe and data; only the seed changes).
#   5-10. Confirmatory evals on the frozen R4-5 set (PHYSCAL_CONFIRMATORY=1): dpos2 s0, b2 s0, dpos2 s1, b2 s1,
#      dpos2 s2, b2 s2. An eval needs its seed's complete adapter (750 / 750 steps); otherwise BLOCKED and the next.
#   11. MC_B=1 only (the following night, D119): MC-B generation on A6c, A6, A6b (GPU), then its execution and
#      analysis (CPU, 1 thread, nice 19, after the GPU stages).
#   12. CPU: D118's analysis (physcal.eval.r4_confirm), only once all six confirmatory evals exist (D118 §10).
# Each stage starts only if its estimate (seed 0's measured wall time) + SLACK fits before GPU_DEADLINE; otherwise SKIP
# and the next stage is tried. Trained seeds and finished evals are skipped, so the same script runs the second night.
# The launch cap (D118 §1): host VRAM is read with nothing running in WSL (median of 8 readings, re-read while the
# cap it gives is below parity), cap = min(5.9, total - host - 0.5) GiB, never raised; every stage gets it.
# Every stage is a child whose PID goes to $LOG/<stage>.pid and is waited on with `wait` (never pgrep -f); every GPU
# stage gets the in-stage temperature guard (scripts/gpu_temp_guard.sh) and the cool check before it. After every stage
# the chain commits and pushes under the shared flock. The free-space guard (D111) runs before every stage, fail-closed.
#
# usage:  START=23:00 PID_FILE=results/queue_r4_seeds/waiter.pid GPU_DEADLINE=09:00 \
#           setsid nohup bash scripts/run_at.sh bash scripts/queue_r4_seeds.sh > results/queue_r4_seeds/nohup.log 2>&1 &
#         DRY_RUN=1 QUEUE_LOG=<scratch dir> bash scripts/queue_r4_seeds.sh
set -u
cd "$(dirname "$0")/.."
. scripts/disk_guard.sh   # the free-space guard (D111): disk_guard before every stage
LOG=${QUEUE_LOG:-results/queue_r4_seeds}
GIT_LOCK=${GIT_LOCK:-.git/queue-commit.lock}
TRAIN_EST_dpos2=${TRAIN_EST_dpos2:-5000}  # seed 0: 83 min wall incl. one 10-min temperature pause (D117)
TRAIN_EST_b2=${TRAIN_EST_b2:-4800}        # seed 0: 80 min wall incl. one pause
EVAL_EST_dpos2=${EVAL_EST_dpos2:-6700}    # seed 0's development eval: 111 min (3,060 problems; same size here)
EVAL_EST_b2=${EVAL_EST_b2:-3900}          # seed 0's development eval: 65 min
SLACK_S=${SLACK_S:-900}
CAP_MAX=${CAP_MAX:-5.9}                   # D115 / D118: parity with seed 0, never raised
HOST_TRIES=${HOST_TRIES:-5}
HOST_READ_S=${HOST_READ_S:-2}             # seconds between the 8 readings
HOST_RETRY_S=${HOST_RETRY_S:-60}
DRY_RUN=${DRY_RUN:-0}
MC_B=${MC_B:-0}                           # 1: the MC-B stages (D119; the following night)
MCB_EST=${MCB_EST:-2400}                  # per set: 1,020 short programs, ~1,300-token prompts (estimate, unmeasured)
MCB_SETS=${MCB_SETS:-"a6c_pairs a6_pairs a6b_pairs"}
SFT_ORDER=${SFT_ORDER:-"dpos2:1 dpos2:2 b2:1 b2:2"}
EVAL_ORDER=${EVAL_ORDER:-"dpos2:0 b2:0 dpos2:1 b2:1 dpos2:2 b2:2"}
CO_AUTHOR="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
PY="uv run python"
OUT=results/stage_r4
mkdir -p "$LOG"

ist() { TZ=Asia/Kolkata date -d "@$1" +%H:%M; }
iso() { TZ=Asia/Kolkata date -d "@$1" --iso-8601=seconds; }
log() { echo "[$(TZ=Asia/Kolkata date +%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }
left() { echo $((DEADLINE - $(date +%s))); }
smi() { nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits | head -1 | tr -dc '0-9'; }
cool() { [ "$DRY_RUN" = 1 ] && return; while t=$(smi temperature.gpu); [ "${t:-0}" -gt 87 ]; do
           log "GPU at ${t}C > 87C: pausing 600s"; sleep 600; done; log "GPU at ${t}C, memory.used $(smi memory.used) MiB"; }

host_mib() {  # the steady host use with nothing running in WSL: the median of 8 readings, 2 s apart
  local v=() i
  for i in 1 2 3 4 5 6 7 8; do v+=("$(smi memory.used)"); sleep "$HOST_READ_S"; done
  log "host VRAM readings (MiB): ${v[*]}" >&2
  printf '%s\n' "${v[@]}" | sort -n | sed -n 4,5p | awk '{s += $1} END {printf "%d", s / 2}'
}

cap_of() { awk -v t="$1" -v h="$2" -v m="$CAP_MAX" 'BEGIN {c = (t - h) / 1024 - 0.5; if (c > m) c = m;
  printf "%.1f", int(c * 10) / 10}'; }

launch_cap() {  # cap = min(CAP_MAX, total - host - 0.5 GiB), floored to 0.1 GiB; re-read while it is below parity
  local total host cap i
  if [ "$DRY_RUN" = 1 ]; then echo "$CAP_MAX"; return; fi
  total=$(smi memory.total)
  for ((i = 1; i <= HOST_TRIES; i++)); do
    host=$(host_mib)
    cap=$(cap_of "$total" "$host")
    [ "$cap" = "$CAP_MAX" ] && break
    [ "$i" -lt "$HOST_TRIES" ] && { log "host use ${host} MiB gives ${cap} GiB < ${CAP_MAX}: re-reading in ${HOST_RETRY_S} s" >&2
                                    sleep "$HOST_RETRY_S"; }
  done
  echo "$cap"
}

run_stage() {  # run_stage NAME TIMEOUT_S CMD...: child process, PID file, wait; returns the child's exit code
  local name=$1 limit=$2; shift 2
  disk_guard "$name" || exit 75   # < MIN_FREE_GB on MLStorage (or unreadable): stop the queue (D111)
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

json_get() { $PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d$2)" "$1" 2>/dev/null || echo "?"; }

complete_adapter() {  # complete_adapter DIR: the adapter exists and its training ran every planned step
  [ -f "$1/final_adapter/adapter_model.safetensors" ] || return 1
  $PY -c "import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s['steps'] == s['steps_planned'] else 1)" \
    "$1/summary.json" 2>/dev/null
}

[ "${QUEUE_LIB_ONLY:-0}" = 1 ] && return 0   # tests source the functions above without running the chain

echo $$ > "$LOG/queue.pid"
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-09:00}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "R4 seeds chain start (D118); GPU deadline $(ist "$DEADLINE") IST ($(( $(left) / 60 )) min left); chain PID $$"
export HF_HUB_OFFLINE=1   # every model is local (D78)
for f in results/phase12_sft_data_r4/explicit.jsonl results/phase12_sft_data_r4/supervised.jsonl \
         results/phase12_sft_data_r4/problems.jsonl data/eval_sets/2.0.0+bdbb2279/confirmatory/a6c_pairs.jsonl \
         $OUT/token_budget.json; do
  [ -e "$f" ] || log "BLOCKED: missing $f"
done
CAP=$(launch_cap)
log "launch cap ${CAP} GiB (max ${CAP_MAX}; D118 §1)"

# ---- 1-4. SFT, seeds 1 and 2 ------------------------------------------------------------------------------------------
for item in $SFT_ORDER; do
  arm=${item%%:*} seed=${item##*:}
  dir=$OUT/$arm/seed$seed
  tv=TRAIN_EST_$arm; est=${!tv}
  if [ "$(left)" -lt $((est + SLACK_S)) ] && [ "$DRY_RUN" != 1 ]; then
    log "SKIP train $arm seed $seed: $(( $(left) / 60 )) min left, needs $(( (est + SLACK_S) / 60 )) min"; continue
  fi
  if [ -e "$dir/final_adapter" ]; then log "SKIP train $arm seed $seed: $dir/final_adapter exists"; continue; fi
  cool
  t0=$(date +%s)
  GUARD=1 run_stage "train_${arm}_s$seed" $(( $(left) - 300 )) $PY -m physcal.train.stage_r4 train --arm "$arm" \
    --seed "$seed" --stop-at "$(iso $((DEADLINE - SLACK_S)))" --max-reserved-gib "$CAP"; rc=$?
  [ "$rc" != 0 ] && log "BLOCKED? train_${arm}_s$seed exit $rc (see $LOG/train_${arm}_s$seed.log)"
  [ "$DRY_RUN" != 1 ] && log "train $arm seed $seed: $(json_get "$dir/summary.json" "['steps']") steps of $(json_get "$dir/summary.json" "['steps_planned']"), $(( ($(date +%s) - t0) / 60 )) min"
  commit_stage "R4 confirmatory (D118): SFT $arm seed $seed on the R4-1 data, $(json_get "$dir/summary.json" "['steps']") steps (exit $rc)" \
    "$dir/summary.json" "$dir/events.jsonl" "$dir/final_adapter/physcal_fingerprint.json" "$dir/final_adapter/adapter_config.json"
done

# ---- 5-10. confirmatory evals -----------------------------------------------------------------------------------------
for item in $EVAL_ORDER; do
  arm=${item%%:*} seed=${item##*:}
  dir=$OUT/$arm/seed$seed
  ev=EVAL_EST_$arm; est=${!ev}
  if [ -f "$dir/confirmatory/rollouts.jsonl" ]; then log "SKIP eval $arm seed $seed: already evaluated"; continue; fi
  if [ "$DRY_RUN" != 1 ] && ! complete_adapter "$dir"; then
    log "BLOCKED eval $arm seed $seed: no complete adapter in $dir"; continue; fi
  if [ "$(left)" -lt $((est + SLACK_S)) ] && [ "$DRY_RUN" != 1 ]; then
    log "SKIP eval $arm seed $seed: $(( $(left) / 60 )) min left, needs $(( (est + SLACK_S) / 60 )) min"; continue
  fi
  cool
  lim=$((2 * est)); [ "$(left)" -lt "$lim" ] && lim=$(left)
  GUARD=1 run_stage "eval_${arm}_s$seed" "$lim" env PHYSCAL_CONFIRMATORY=1 $PY -m physcal.train.stage_r4 eval \
    --arm "$arm" --seed "$seed" --confirmatory --max-reserved-gib "$CAP"; rc=$?
  [ "$rc" != 0 ] && log "BLOCKED? eval_${arm}_s$seed exit $rc (see $LOG/eval_${arm}_s$seed.log)"
  [ "$DRY_RUN" != 1 ] && [ -f "$dir/confirmatory/rollouts.jsonl" ] && \
    gzip -kfc "$dir/confirmatory/rollouts.jsonl" > "$dir/confirmatory/rollouts.jsonl.gz"
  commit_stage "R4 confirmatory (D118): $arm seed $seed greedy eval on the frozen R4-5 set (exit $rc)" \
    "$dir/confirmatory/rollouts.jsonl.gz" "$dir/confirmatory/rollouts.meta.json" "$dir/confirmatory/metrics.json" \
    "$dir/confirmatory/summary.md"
done
# ---- 11. MC-B (D119): GPU generation per set, then CPU execution + analysis --------------------------------------------
CPU_ENV="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= nice -n 19"
if [ "$MC_B" = 1 ]; then
  for set in $MCB_SETS; do
    if [ "$(left)" -lt $((MCB_EST + SLACK_S)) ] && [ "$DRY_RUN" != 1 ]; then
      log "SKIP mc_b generate $set: $(( $(left) / 60 )) min left, needs $(( (MCB_EST + SLACK_S) / 60 )) min"; continue
    fi
    cool
    lim=$((2 * MCB_EST)); [ "$(left)" -lt "$lim" ] && lim=$(left)
    GUARD=1 run_stage "mc_b_gen_$set" "$lim" $PY -m physcal.eval.mc_b generate --set "$set" --max-reserved-gib "$CAP"; rc=$?
    [ "$rc" != 0 ] && log "BLOCKED? mc_b_gen_$set exit $rc (see $LOG/mc_b_gen_$set.log)"
    commit_stage "MC-B (D119): $set, one greedy program per problem (exit $rc)" \
      "results/mc_b/$set/generations.jsonl" "results/mc_b/$set/meta.json"
  done
  for set in $MCB_SETS; do
    [ -f "results/mc_b/$set/generations.jsonl" ] || { log "BLOCKED mc_b analysis $set: no generations"; continue; }
    run_stage "mc_b_exec_$set" 3600 $CPU_ENV $PY -m physcal.eval.mc_b execute --set "$set" \
      && run_stage "mc_b_analyze_$set" 3600 $CPU_ENV $PY -m physcal.eval.mc_b analyze --set "$set" \
      || log "BLOCKED mc_b execute/analyze $set (see $LOG/mc_b_exec_$set.log, $LOG/mc_b_analyze_$set.log)"
    commit_stage "MC-B (D119): $set executed in the sandbox and analyzed" "results/mc_b/$set/executions.jsonl" \
      "results/mc_b/$set/results.json" "results/mc_b/$set/summary.md"
  done
fi

# ---- 12. CPU: D118's analysis, once all six confirmatory evals exist ----------------------------------------------------
n_done=0
for item in $EVAL_ORDER; do [ -f "$OUT/${item%%:*}/seed${item##*:}/confirmatory/rollouts.jsonl" ] && n_done=$((n_done + 1)); done
if [ "$n_done" = 6 ] || [ "$DRY_RUN" = 1 ]; then
  run_stage r4_confirm 5400 $CPU_ENV env PHYSCAL_CONFIRMATORY=1 $PY -m physcal.eval.r4_confirm analyze \
    || log "BLOCKED r4_confirm (see $LOG/r4_confirm.log)"
  commit_stage "R4-3 confirmatory analysis (D118): K1-K8, P, the depth split, 3 seeds on the frozen R4-5 set" \
    "$OUT/confirmatory"
else
  log "D118 §10: $n_done of 6 confirmatory evals exist; the analysis waits"
fi
log "R4 seeds chain done"

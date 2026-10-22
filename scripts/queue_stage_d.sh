#!/usr/bin/env bash
# Night GPU chain, 2026-10-22/23 (DECISIONS.md D109): Phase 12 stage D, explicit propagation, after the stage C pilot.
#   0. wait for the stage C pilot's queue (WAIT_PID, scripts/queue_stage_c_pilot.sh) to exit
#   1-2. arms pos -> fmt (your order): GPU train (stage B's settings, D103 cap + OOM check) -> GPU eval (A6 / A6b /
#      A6c, greedy, prompt v5). An arm starts only if its training + eval + SLACK fit before GPU_DEADLINE (estimates:
#      the defaults below, then pos's measured times); otherwise SKIP.
#   3. CPU: stage D's analysis (D109; 1 thread, nice 19)
# Every stage is a child whose PID goes to $LOG/<stage>.pid and is waited on with `wait` (never pgrep -f); every GPU
# stage gets the in-stage temperature guard (scripts/gpu_temp_guard.sh, D96) and the cool check before it. A failed
# stage is logged BLOCKED and the next independent stage goes on (an eval needs its own adapter). After every stage the
# chain commits and pushes under the shared flock.
#
# usage:  WAIT_PID=<pid> GPU_DEADLINE=09:00 setsid nohup bash scripts/queue_stage_d.sh > results/queue_stage_d/nohup.log 2>&1 &
#         DRY_RUN=1 QUEUE_LOG=<scratch dir> bash scripts/queue_stage_d.sh
set -u
cd "$(dirname "$0")/.."
LOG=${QUEUE_LOG:-results/queue_stage_d}
GIT_LOCK=${GIT_LOCK:-.git/queue-commit.lock}
TRAIN_EST_S=${TRAIN_EST_S:-5100}     # stage B's measured 74-81 min per seed (load + 750 steps + fingerprint), +3% tokens
EVAL_EST_S=${EVAL_EST_S:-3900}       # stage B's measured 54 min, + ~16% longer completions (the line)
SLACK_S=${SLACK_S:-900}
DRY_RUN=${DRY_RUN:-0}
CO_AUTHOR="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
PY="uv run python"
OUT=results/stage_d
CPU_ENV="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= nice -n 19"
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

json_get() { $PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d$2)" "$1" 2>/dev/null || echo "?"; }

echo $$ > "$LOG/queue.pid"
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-09:00}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "stage D chain start; GPU deadline $(ist "$DEADLINE") IST ($(( $(left) / 60 )) min left); chain PID $$"
export HF_HUB_OFFLINE=1   # every model is local (D78)

# ---- 0. wait for the stage C pilot --------------------------------------------------------------------------------------
if [ -n "${WAIT_PID:-}" ] && [ "$DRY_RUN" != 1 ]; then
  log "waiting for PID $WAIT_PID (the stage C pilot's queue)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  log "PID $WAIT_PID has exited ($(( $(left) / 60 )) min left)"
fi

# ---- 1-2. arms --------------------------------------------------------------------------------------------------------
for f in results/phase12_sft_data/explicit.jsonl results/phase12_sft_data/problems.jsonl \
         data/eval_sets/2.0.0+bdbb2279/a6c_pairs.jsonl; do
  [ -e "$f" ] || log "BLOCKED: missing $f"
done
train_est=$TRAIN_EST_S eval_est=$EVAL_EST_S
for arm in pos fmt; do
  dir=$OUT/$arm/seed0
  need=$((train_est + eval_est + SLACK_S))
  if [ "$(left)" -lt "$need" ] && [ "$DRY_RUN" != 1 ]; then
    log "SKIP arm $arm: $(( $(left) / 60 )) min left, needs $((need / 60)) min (train $((train_est / 60)) + eval $((eval_est / 60)) + slack)"
    continue
  fi
  cool
  stop=$((DEADLINE - eval_est - SLACK_S))
  t0=$(date +%s)
  lim=$((2 * train_est)); [ $((stop - t0)) -lt "$lim" ] && lim=$((stop - t0))
  GUARD=1 run_stage "stage_d_train_$arm" $((lim + 900)) $PY -m physcal.train.stage_d train --arm "$arm" \
    --stop-at "$(iso "$stop")"; rc=$?
  t1=$(date +%s)
  [ "$rc" != 0 ] && log "BLOCKED? stage_d_train_$arm exit $rc (see $LOG/stage_d_train_$arm.log)"
  [ "$DRY_RUN" != 1 ] && log "arm $arm: $(json_get "$dir/summary.json" "['steps']") steps of $(json_get "$dir/summary.json" "['steps_planned']"), $(( (t1 - t0) / 60 )) min"
  commit_stage "stage D $arm (D109): SFT on the explicit arm, $(json_get "$dir/summary.json" "['steps']") steps (exit $rc)" \
    "$dir/summary.json" "$dir/events.jsonl" "$dir/final_adapter/physcal_fingerprint.json" "$dir/final_adapter/adapter_config.json"
  [ "$rc" = 0 ] && [ "$DRY_RUN" != 1 ] && train_est=$((t1 - t0 + 300))
  if [ "$DRY_RUN" != 1 ] && [ ! -f "$dir/final_adapter/adapter_model.safetensors" ]; then
    log "BLOCKED stage_d_eval_$arm: no adapter"; continue; fi
  cool
  lim=$((2 * eval_est)); [ "$(left)" -lt "$lim" ] && lim=$(left)
  t0=$(date +%s)
  GUARD=1 run_stage "stage_d_eval_$arm" "$lim" $PY -m physcal.train.stage_d eval --arm "$arm"; rc=$?
  t1=$(date +%s)
  [ "$rc" != 0 ] && log "BLOCKED? stage_d_eval_$arm exit $rc (see $LOG/stage_d_eval_$arm.log)"
  [ "$rc" = 0 ] && [ "$DRY_RUN" != 1 ] && eval_est=$((t1 - t0 + 300))
  [ "$DRY_RUN" != 1 ] && [ -f "$dir/eval/rollouts.jsonl" ] && gzip -kfc "$dir/eval/rollouts.jsonl" > "$dir/eval/rollouts.jsonl.gz"
  commit_stage "stage D $arm (D109): greedy eval on A6 / A6b / A6c, 4-bit pinned base, adapter unmerged (exit $rc)" \
    "$dir/eval/rollouts.jsonl.gz" "$dir/eval/rollouts.meta.json" "$dir/eval/metrics.json" "$dir/eval/summary.md"
done

# ---- 3. CPU: stage D's analysis ---------------------------------------------------------------------------------------
run_stage stage_d_analysis 5400 $CPU_ENV $PY -m physcal.eval.stage_d_analysis analyze \
  || log "BLOCKED stage_d_analysis (see $LOG/stage_d_analysis.log)"
commit_stage "Phase 12 stage D (D109): D1-D7, F1-F6, Holm, line diagnostics" "$OUT/results.json" "$OUT/summary.md" \
  "$OUT/pos/results.json" "$OUT/pos/summary.md" "$OUT/fmt/results.json" "$OUT/fmt/summary.md"
log "stage D chain done"

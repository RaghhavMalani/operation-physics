#!/usr/bin/env bash
# Night GPU chain, 2026-10-25/26 (DECISIONS.md D113 / D115): R4-1's D-pos-v2 and its matched implicit control B-v2.
#   1-2. arms dpos2 -> b2 (your order): GPU train (D-pos's recipe on the R4-1 data; the D115 cap + OOM check) -> GPU
#      eval (A6 / A6b / A6c, greedy, prompt v5, D114's budget). An arm starts only if its training + eval + SLACK fit
#      before GPU_DEADLINE (estimates: the per-arm defaults below, scaled from D-pos and stage B); otherwise SKIP.
#   3. CPU: D113's analysis (physcal.eval.r4_analysis; 1 thread, nice 19).
# MC-A (R4-4) is a separate chain (scripts/queue_mc_a.sh) that waits on this chain's PID.
# Every stage is a child whose PID goes to $LOG/<stage>.pid and is waited on with `wait` (never pgrep -f); every GPU
# stage gets the in-stage temperature guard (scripts/gpu_temp_guard.sh, D96) and the cool check before it. A failed
# stage is logged BLOCKED and the next independent stage goes on (an eval needs its own adapter). After every stage the
# chain commits and pushes under the shared flock. The free-space guard (D111) runs before every stage, fail-closed.
#
# usage:  GPU_DEADLINE=09:00 setsid nohup bash scripts/queue_r4.sh > results/queue_r4/nohup.log 2>&1 &
#         DRY_RUN=1 QUEUE_LOG=<scratch dir> bash scripts/queue_r4.sh
set -u
cd "$(dirname "$0")/.."
. scripts/disk_guard.sh   # the free-space guard (D111): disk_guard before every stage
LOG=${QUEUE_LOG:-results/queue_r4}
GIT_LOCK=${GIT_LOCK:-.git/queue-commit.lock}
TRAIN_EST_dpos2=${TRAIN_EST_dpos2:-4800}  # D-pos's measured 67 min + load, x1.06 tokens per row (1,498 vs 1,420)
EVAL_EST_dpos2=${EVAL_EST_dpos2:-6000}    # D-pos's measured 55 min x ~1.5 (the S lines, depth 5) + the 1,280 budget
TRAIN_EST_b2=${TRAIN_EST_b2:-4900}        # stage B's measured 74-81 min
EVAL_EST_b2=${EVAL_EST_b2:-4200}          # stage B's measured 54 min + the 1,280 budget
SLACK_S=${SLACK_S:-900}
DRY_RUN=${DRY_RUN:-0}
CO_AUTHOR="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
PY="uv run python"
OUT=results/stage_r4
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

echo $$ > "$LOG/queue.pid"
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-09:00}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "R4 chain start; GPU deadline $(ist "$DEADLINE") IST ($(( $(left) / 60 )) min left); chain PID $$"
export HF_HUB_OFFLINE=1   # every model is local (D78)

# ---- 1-2. arms --------------------------------------------------------------------------------------------------------
for f in results/phase12_sft_data_r4/explicit.jsonl results/phase12_sft_data_r4/supervised.jsonl \
         results/phase12_sft_data_r4/problems.jsonl data/eval_sets/2.0.0+bdbb2279/a6c_pairs.jsonl; do
  [ -e "$f" ] || log "BLOCKED: missing $f"
done
for arm in dpos2 b2; do
  dir=$OUT/$arm/seed0
  tv=TRAIN_EST_$arm ev=EVAL_EST_$arm
  train_est=${!tv} eval_est=${!ev}
  need=$((train_est + eval_est + SLACK_S))
  if [ "$(left)" -lt "$need" ] && [ "$DRY_RUN" != 1 ]; then
    log "SKIP arm $arm: $(( $(left) / 60 )) min left, needs $((need / 60)) min (train $((train_est / 60)) + eval $((eval_est / 60)) + slack)"
    continue
  fi
  cool
  stop=$((DEADLINE - eval_est - SLACK_S))
  t0=$(date +%s)
  lim=$((2 * train_est)); [ $((stop - t0)) -lt "$lim" ] && lim=$((stop - t0))
  GUARD=1 run_stage "r4_train_$arm" $((lim + 900)) $PY -m physcal.train.stage_r4 train --arm "$arm" \
    --stop-at "$(iso "$stop")"; rc=$?
  t1=$(date +%s)
  [ "$rc" != 0 ] && log "BLOCKED? r4_train_$arm exit $rc (see $LOG/r4_train_$arm.log)"
  [ "$DRY_RUN" != 1 ] && log "arm $arm: $(json_get "$dir/summary.json" "['steps']") steps of $(json_get "$dir/summary.json" "['steps_planned']"), $(( (t1 - t0) / 60 )) min"
  commit_stage "R4 $arm (D113): SFT on the R4-1 data, $(json_get "$dir/summary.json" "['steps']") steps (exit $rc)" \
    "$dir/summary.json" "$dir/events.jsonl" "$dir/final_adapter/physcal_fingerprint.json" "$dir/final_adapter/adapter_config.json"
  if [ "$DRY_RUN" != 1 ] && [ ! -f "$dir/final_adapter/adapter_model.safetensors" ]; then
    log "BLOCKED r4_eval_$arm: no adapter"; continue; fi
  cool
  lim=$((2 * eval_est)); [ "$(left)" -lt "$lim" ] && lim=$(left)
  GUARD=1 run_stage "r4_eval_$arm" "$lim" $PY -m physcal.train.stage_r4 eval --arm "$arm"; rc=$?
  [ "$rc" != 0 ] && log "BLOCKED? r4_eval_$arm exit $rc (see $LOG/r4_eval_$arm.log)"
  [ "$DRY_RUN" != 1 ] && [ -f "$dir/eval/rollouts.jsonl" ] && gzip -kfc "$dir/eval/rollouts.jsonl" > "$dir/eval/rollouts.jsonl.gz"
  commit_stage "R4 $arm (D113): greedy eval on A6 / A6b / A6c, 4-bit pinned base, adapter unmerged (exit $rc)" \
    "$dir/eval/rollouts.jsonl.gz" "$dir/eval/rollouts.meta.json" "$dir/eval/metrics.json" "$dir/eval/summary.md" \
    "$OUT/token_budget.json"
done

# ---- 3. CPU: D113's analysis ------------------------------------------------------------------------------------------
run_stage r4_analysis 5400 $CPU_ENV $PY -m physcal.eval.r4_analysis analyze \
  || log "BLOCKED r4_analysis (see $LOG/r4_analysis.log)"
commit_stage "R4-1 / R4-2 / R4-3 (D113): D-pos-v2 and B-v2, the three sigmas, the stop/go, R1-R8" \
  "$OUT/results.json" "$OUT/summary.md" "$OUT/dpos2/results.json" "$OUT/dpos2/summary.md" \
  "$OUT/b2/results.json" "$OUT/b2/summary.md"
log "R4 chain done"

#!/usr/bin/env bash
# Night GPU chain, 2026-10-25/26 (DECISIONS.md D116): D-pos-v2's eval again after its OOM, then MC-A.
#   0. wait until the R4 chain (WAIT_PID, scripts/queue_r4.sh) has finished its GPU stages (its chain log shows the
#      analysis starting) or has exited
#   1. GPU: D-pos-v2's eval (stage R4 dpos2; the backoff fix and batch 16, D116), if its estimate + SLACK fits
#   2. start the MC-A chain (scripts/queue_mc_a.sh, no wait) on the GPU, and run D113's analysis on the CPU
#      (1 thread, nice 19) beside it; commit + push after each
# Same conventions as scripts/queue_r4.sh: PID files and `wait`, the in-stage temperature guard, BLOCKED-and-continue,
# the free-space guard (D111) before every stage, commits under the shared flock.
#
# usage:  WAIT_PID=<pid> GPU_DEADLINE=09:00 setsid nohup bash scripts/queue_r4_tail.sh > results/queue_r4_tail/nohup.log 2>&1 &
#         DRY_RUN=1 QUEUE_LOG=<scratch dir> bash scripts/queue_r4_tail.sh
set -u
cd "$(dirname "$0")/.."
. scripts/disk_guard.sh   # the free-space guard (D111): disk_guard before every stage
LOG=${QUEUE_LOG:-results/queue_r4_tail}
GIT_LOCK=${GIT_LOCK:-.git/queue-commit.lock}
R4_LOG=${R4_LOG:-results/queue_r4/chain.log}
EVAL_EST_S=${EVAL_EST_S:-9000}   # D-pos's 55 min at 512 tokens / batch 32, x ~2 (the S lines, 1,280, depth 5) x 1.3 (batch 16)
SLACK_S=${SLACK_S:-900}
DRY_RUN=${DRY_RUN:-0}
CO_AUTHOR="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
PY="uv run python"
OUT=results/stage_r4
CPU_ENV="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= nice -n 19"
mkdir -p "$LOG"

ist() { TZ=Asia/Kolkata date -d "@$1" +%H:%M; }
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

echo $$ > "$LOG/queue.pid"
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-09:00}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "R4 tail chain start; GPU deadline $(ist "$DEADLINE") IST ($(( $(left) / 60 )) min left); chain PID $$"
export HF_HUB_OFFLINE=1   # every model is local (D78)

# ---- 0. wait for the R4 chain's GPU stages ----------------------------------------------------------------------------
if [ -n "${WAIT_PID:-}" ] && [ "$DRY_RUN" != 1 ]; then
  log "waiting for PID $WAIT_PID (the R4 chain) to reach its analysis or exit"
  while kill -0 "$WAIT_PID" 2>/dev/null && ! grep -q "start r4_analysis" "$R4_LOG" 2>/dev/null; do sleep 30; done
  log "R4 chain's GPU stages are done ($(( $(left) / 60 )) min left)"
fi

# ---- 1. D-pos-v2's eval -----------------------------------------------------------------------------------------------
dir=$OUT/dpos2/seed0
if [ "$(left)" -lt $((EVAL_EST_S + SLACK_S)) ] && [ "$DRY_RUN" != 1 ]; then
  log "SKIP D-pos-v2 eval: $(( $(left) / 60 )) min left, needs $(( (EVAL_EST_S + SLACK_S) / 60 )) min"
elif [ "$DRY_RUN" != 1 ] && [ ! -f "$dir/final_adapter/adapter_model.safetensors" ]; then
  log "BLOCKED r4_eval_dpos2: no adapter"
else
  cool
  lim=$((2 * EVAL_EST_S)); [ $(( $(left) - SLACK_S )) -lt "$lim" ] && lim=$(( $(left) - SLACK_S ))
  GUARD=1 run_stage r4_eval_dpos2 "$lim" $PY -m physcal.train.stage_r4 eval --arm dpos2; rc=$?
  [ "$rc" != 0 ] && log "BLOCKED? r4_eval_dpos2 exit $rc (see $LOG/r4_eval_dpos2.log)"
  [ "$DRY_RUN" != 1 ] && [ -f "$dir/eval/rollouts.jsonl" ] && gzip -kfc "$dir/eval/rollouts.jsonl" > "$dir/eval/rollouts.jsonl.gz"
  commit_stage "R4 dpos2 (D113 / D116): greedy eval on A6 / A6b / A6c, rerun after the OOM, batch 16 (exit $rc)" \
    "$dir/eval/rollouts.jsonl.gz" "$dir/eval/rollouts.meta.json" "$dir/eval/metrics.json" "$dir/eval/summary.md"
fi

# ---- 2. MC-A on the GPU, D113's analysis on the CPU beside it --------------------------------------------------------
if [ "$DRY_RUN" = 1 ]; then log "DRY start scripts/queue_mc_a.sh"; else
  mkdir -p results/queue_mc_a
  GPU_DEADLINE=${GPU_DEADLINE:-09:00} setsid nohup bash scripts/queue_mc_a.sh >> results/queue_mc_a/nohup.log 2>&1 &
  log "MC-A chain started (PID $!)"
fi
run_stage r4_analysis 5400 $CPU_ENV $PY -m physcal.eval.r4_analysis analyze \
  || log "BLOCKED r4_analysis (see $LOG/r4_analysis.log)"
commit_stage "R4-1 / R4-2 / R4-3 (D113): D-pos-v2 and B-v2, the three sigmas, the stop/go, R1-R8" \
  "$OUT/results.json" "$OUT/summary.md" "$OUT/dpos2/results.json" "$OUT/dpos2/summary.md" \
  "$OUT/b2/results.json" "$OUT/b2/summary.md"
log "R4 tail chain done"

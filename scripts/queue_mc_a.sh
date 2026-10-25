#!/usr/bin/env bash
# Night GPU chain, 2026-10-25/26 (DECISIONS.md D113 §7): R4-4's MC-A on A6c, then A6 and A6b if time remains.
#   0. wait until the R4 chain (WAIT_PID, scripts/queue_r4.sh) has finished its GPU stages (its chain log shows the
#      analysis starting) or has exited
#   1-3. per set: GPU `mc_a run` (B-v2, or stage B seed 0 if B-v2 has no adapter; resumable chunks) -> CPU `mc_a
#      analyze` (1 thread, nice 19) -> gzip -> commit + push. A set starts only if its estimate + SLACK fits before
#      GPU_DEADLINE (the estimate: the model's eval seconds per completion x k x problems, + load); the run's timeout is
#      the time left minus SLACK, and a stopped run keeps its finished chunks.
# Every stage is a child whose PID goes to $LOG/<stage>.pid and is waited on with `wait` (never pgrep -f); every GPU
# stage gets the in-stage temperature guard and the cool check before it; a failed stage is logged BLOCKED and the next
# set goes on. The free-space guard (D111) runs before every stage, fail-closed.
#
# usage:  WAIT_PID=<pid> GPU_DEADLINE=09:00 setsid nohup bash scripts/queue_mc_a.sh > results/queue_mc_a/nohup.log 2>&1 &
#         DRY_RUN=1 QUEUE_LOG=<scratch dir> bash scripts/queue_mc_a.sh
set -u
cd "$(dirname "$0")/.."
. scripts/disk_guard.sh   # the free-space guard (D111): disk_guard before every stage
LOG=${QUEUE_LOG:-results/queue_mc_a}
GIT_LOCK=${GIT_LOCK:-.git/queue-commit.lock}
R4_LOG=${R4_LOG:-results/queue_r4/chain.log}
SLACK_S=${SLACK_S:-900}
LOAD_S=${LOAD_S:-180}
DEFAULT_PER_COMPLETION_S=${DEFAULT_PER_COMPLETION_S:-1.1}   # stage B's measured 54 min / 3,060, + the 1,280 budget
K=8
DRY_RUN=${DRY_RUN:-0}
CO_AUTHOR="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
PY="uv run python"
OUT=results/mc_a
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

per_completion() {  # the MC-A model's measured eval seconds per completion (B-v2's, else stage B seed 0's), or the default
  local f
  for f in results/stage_r4/b2/seed0/eval/metrics.json results/stage_b/seed0/eval/metrics.json; do
    if [ -e "$f" ]; then
      $PY -c "import json,sys; m=json.load(open(sys.argv[1])); n=sum(v.get('n', 0) for v in m['splits'].values()) if 'splits' in m else 3060; print(round(m['generation_seconds'] / max(n, 1), 3))" "$f" 2>/dev/null && return
    fi
  done
  echo "$DEFAULT_PER_COMPLETION_S"
}

echo $$ > "$LOG/queue.pid"
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-09:00}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "MC-A chain start; GPU deadline $(ist "$DEADLINE") IST ($(( $(left) / 60 )) min left); chain PID $$"
export HF_HUB_OFFLINE=1   # every model is local (D78)

# ---- 0. wait for the R4 chain's GPU stages ----------------------------------------------------------------------------
if [ -n "${WAIT_PID:-}" ] && [ "$DRY_RUN" != 1 ]; then
  log "waiting for PID $WAIT_PID (the R4 chain) to reach its analysis or exit"
  while kill -0 "$WAIT_PID" 2>/dev/null && ! grep -q "start r4_analysis" "$R4_LOG" 2>/dev/null; do sleep 30; done
  log "R4 chain's GPU stages are done ($(( $(left) / 60 )) min left)"
fi

# ---- 1-3. sets ---------------------------------------------------------------------------------------------------------
for set in a6c_pairs a6_pairs a6b_pairs; do
  dir=$OUT/$set
  pc=$(per_completion)
  est=$($PY -c "print(int($pc * $K * 1020 + $LOAD_S))" 2>/dev/null || echo 9000)
  if [ "$(left)" -lt $((est + SLACK_S)) ] && [ "$DRY_RUN" != 1 ]; then
    log "SKIP MC-A $set: $(( $(left) / 60 )) min left, needs $(( (est + SLACK_S) / 60 )) min (${pc}s per completion)"
    continue
  fi
  cool
  log "MC-A $set: estimate $((est / 60)) min (${pc}s per completion)"
  GUARD=1 run_stage "mc_a_run_$set" $(( $(left) - SLACK_S )) $PY -m physcal.eval.mc_a run --set "$set"; rc=$?
  [ "$rc" != 0 ] && log "BLOCKED? mc_a_run_$set exit $rc (see $LOG/mc_a_run_$set.log)"
  if [ "$DRY_RUN" != 1 ] && [ ! -s "$dir/runs.jsonl" ]; then log "BLOCKED mc_a_analyze_$set: no runs"; continue; fi
  run_stage "mc_a_analyze_$set" 3600 $CPU_ENV $PY -m physcal.eval.mc_a analyze --set "$set" \
    || log "BLOCKED mc_a_analyze_$set (see $LOG/mc_a_analyze_$set.log)"
  [ "$DRY_RUN" != 1 ] && gzip -kfc "$dir/runs.jsonl" > "$dir/runs.jsonl.gz"
  commit_stage "MC-A on $set (R4-4, D113 §7): k = 8 perturbed greedy runs, raw / robust / oracle (run exit $rc)" \
    "$dir/runs.jsonl.gz" "$dir/meta.json" "$dir/results.json" "$dir/summary.md"
done
log "MC-A chain done"

#!/usr/bin/env bash
# Reruns after tonight's queue (DECISIONS.md D87, D88), started by hand once scripts/queue_x2_x3.sh is done:
#   1. GPU: X3 student (a) on the X2 answers (D88: its first run crashed on a mu = 0 answer while scoring)
#   2. CPU, background: the X3 analysis (supersedes the queue's, which lacked (a)'s X2 numbers)
#   3. GPU: X2 method 2, split credit seed 0's sigma suffixes (D87: the first run spilled into WSL shared memory)
#   4. CPU: the X2 analysis with method 2 (supersedes the queue's, where method 2 counted as p = 1)
# Same guards as the queue: child PIDs waited on, 87 C pause, GPU work by GPU_DEADLINE (09:00 IST tonight), commit + push
# after every stage under the queue's flock, BLOCKED logging.
#
# usage:  GPU_DEADLINE=09:00 nohup scripts/queue_x2_x3_reruns.sh > results/queue_x2_x3/reruns_nohup.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
. scripts/disk_guard.sh   # the free-space guard (D111): disk_guard before every stage
LOG=${QUEUE_LOG:-results/queue_x2_x3}
GIT_LOCK=.git/queue-commit.lock
DRY_RUN=${DRY_RUN:-0}
CO_AUTHOR="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
PY="uv run python"
export HF_HUB_OFFLINE=1
mkdir -p "$LOG"

log() { echo "[$(TZ=Asia/Kolkata date +%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }
left() { echo $((DEADLINE - $(date +%s))); }
minimum() { [ "$1" -lt "$2" ] && echo "$1" || echo "$2"; }
cool() { [ "$DRY_RUN" = 1 ] && return; while t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "${t:-0}" -gt 87 ]; do
           log "GPU at ${t}C > 87C: pausing 600s"; sleep 600; done; log "GPU at ${t}C"; }
run_stage() {
  local name=$1 limit=$2; shift 2
  disk_guard "$name" || exit 75   # < MIN_FREE_GB on MLStorage (or unreadable): stop the queue (D111)
  if [ "$DRY_RUN" = 1 ]; then log "DRY $name (timeout ${limit}s): $*"; return 0; fi
  if [ "$limit" -lt 300 ]; then log "skip $name: only ${limit}s left"; return 98; fi
  log "start $name (timeout ${limit}s): $*"
  timeout "$limit" "$@" > "$LOG/$name.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$LOG/$name.pid"
  wait "$pid"; local rc=$?
  log "end $name: exit $rc"
  return $rc
}
commit_stage() {
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

DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-09:00}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "reruns start (D87, D88); GPU deadline $(TZ=Asia/Kolkata date -d "@$DEADLINE" +%H:%M) IST"

cool
run_stage x3_eval_a_x2_rerun "$(minimum "$(left)" 2400)" $PY -m physcal.train.x3 eval --arm a --set x2 \
  || log "BLOCKED x3_eval_a_x2_rerun (see $LOG/x3_eval_a_x2_rerun.log)"
commit_stage "X3 pilot (D79/D88): student (a) sigma suffixes on the X2 answers, rerun after the mu = 0 fix" results/x3/a/eval_x2

x3_analysis() {
  run_stage x3_analyze_rerun 1800 env CUDA_VISIBLE_DEVICES="" nice -n 10 $PY -m physcal.train.x3 analyze \
    || log "BLOCKED x3_analyze_rerun (see $LOG/x3_analyze_rerun.log)"
  commit_stage "X3 pilot (D79) results, with student (a) on the X2 answers (supersedes the queue's analysis)" \
    results/x3/pilot.json results/x3/pilot.md
}
x3_analysis &
X3A=$!

cool
run_stage x2_suffix_rerun "$(minimum "$(left)" 3000)" $PY -m physcal.eval.x2_pairs suffix \
  || log "BLOCKED x2_suffix_rerun (see $LOG/x2_suffix_rerun.log)"
commit_stage "X2 (D74/D87): split credit seed 0 @287 sigma suffixes on the kept answers, rerun (batch 16, capped)" \
  results/x2/split_seed0_suffix

run_stage x2_analysis_rerun 3600 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" \
  nice -n 10 $PY -m physcal.eval.x2_analysis analyze || log "BLOCKED x2_analysis_rerun (see $LOG/x2_analysis_rerun.log)"
commit_stage "X2 (D74) results with method 2 (supersedes the queue's analysis, where method 2 counted as p = 1)" \
  results/x2/results.json results/x2/summary.md
[ "$DRY_RUN" != 1 ] && wait "$X3A"
log "reruns done"

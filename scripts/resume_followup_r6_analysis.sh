#!/usr/bin/env bash
# D169 (EXPLORATORY, follow-up project, paper #2): resume D167's CPU analysis after the pause of 2026-11-04. R6-3 and
# R6-4 are done and every one of their fits is cached, so they are re-read in minutes; R6-5 runs from the start.
# The queue's analysis stage alone (no GPU stage): its CPU_ENV (1 BLAS thread, nice 19, the GPU hidden; the probes'
# 4 jobs come from configs/followup.yaml), its disk guard, its lock, and its commit + push at the end.
#
# usage:  setsid nohup scripts/resume_followup_r6_analysis.sh >> results/queue_followup_r6/nohup.log 2>&1 < /dev/null &
#         DRY_RUN=1 DISK_GUARD=off QUEUE_LOG=<scratch> bash scripts/resume_followup_r6_analysis.sh   (runs nothing)
set -u
cd "$(dirname "$0")/.."
export QUEUE_LOG=${QUEUE_LOG:-results/queue_followup_r6}
QUEUE_LIB_ONLY=1 . scripts/queue_r4_seeds.sh   # LOG, log, run_stage, commit_stage, PY
XM2_LIB_ONLY=1 . scripts/queue_xm2.sh          # CPU_ENV
R6=${R6_OUT:-results/followup/r6}
ANALYZE_S=${ANALYZE_S:-43200}

exec 8>>"$LOG/queue.lock"
if ! flock -n 8; then
  log "already running: another queue holds $LOG/queue.lock; this copy (PID $$) exits 0"; exit 0
fi
echo $$ > "$LOG/queue.pid"
log "R6 analysis resume (D169): the cached fits are reused; PID $$"
run_stage r6_analyze_resume "$ANALYZE_S" $CPU_ENV $PY -m physcal.eval.followup_r6 analyze \
  || log "BLOCKED r6_analyze_resume (see $LOG/r6_analyze_resume.log)"
commit_stage "Follow-up (D167/D169, exploratory): R6-3 / R6-4 / R6-5's analysis (resumed after the pause)" \
  "$R6/results.json" "$R6/cache"
log "R6 analysis resume done"

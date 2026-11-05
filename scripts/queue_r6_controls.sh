#!/usr/bin/env bash
# D173 (EXPLORATORY, follow-up project, paper #2): the two local controls for R6-5 at 1.5B: the stages of
# `python -m physcal.eval.followup_r6_controls stages`, then its CPU analysis. Same guards as queue_followup_r6.sh:
# D158's VRAM gate and the temperature guard before each stage, the stall watchdog and free-space guard inside it.
#
# usage:  GPU_DEADLINE=13:00 setsid nohup scripts/queue_r6_controls.sh >> results/queue_r6_controls/nohup.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
export QUEUE_LOG=${QUEUE_LOG:-results/queue_r6_controls}
CAP_MAX=${CAP_MAX:-7.5}
QUEUE_LIB_ONLY=1 . scripts/queue_r4_seeds.sh   # LOG, log, run_stage, commit_stage, cool, launch_cap
XM2_LIB_ONLY=1 . scripts/queue_xm2.sh          # CPU_ENV, gpu_stage
R6=${R6_OUT:-results/followup/r6}
MIN_STAGE_S=${MIN_STAGE_S:-900}
ANALYZE_S=${ANALYZE_S:-10800}
CAP_WAIT_S=${CAP_WAIT_S:-3600}
NEED=${MIN_CAP_15B:-3.5}
FR="$PY -m physcal.eval.followup_r6"
FC="$PY -m physcal.eval.followup_r6_controls"

exec 8>>"$LOG/queue.lock"
if ! flock -n 8; then log "already running: another queue holds $LOG/queue.lock"; exit 0; fi
echo $$ > "$LOG/queue.pid"
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-13:00}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "D173 chain start: GPU deadline $(ist "$DEADLINE") IST ($(( $(left) / 60 )) min left); chain PID $$"
export HF_HUB_OFFLINE=1
mapfile -t STAGES < <($FC stages)
[ "${#STAGES[@]}" -gt 0 ] || { log "BLOCKED: no stages"; exit 1; }

DONE=0 N=0
for line in "${STAGES[@]}"; do
  read -r kind set reader <<< "$line"
  N=$((N + 1))
  name="d173_${kind}_${set}_${reader}"
  if [ "$(left)" -lt "$MIN_STAGE_S" ]; then log "SKIP $name: $(( $(left) / 60 )) min left (re-run to resume)"; continue; fi
  CAP=$(launch_cap)
  t_wait=$(date +%s)
  while awk -v c="$CAP" -v m="$NEED" 'BEGIN {exit !(c + 0 < m + 0)}'; do
    if [ $(( $(date +%s) - t_wait )) -ge "$CAP_WAIT_S" ] || [ "$(left)" -lt "$MIN_STAGE_S" ]; then
      log "BLOCKED: launch cap ${CAP} GiB < ${NEED} for $name: Windows holds too much VRAM (close GPU apps, then re-run)"
      exit 1
    fi
    log "launch cap ${CAP} GiB < ${NEED}: Windows holds $(smi memory.used) MiB of VRAM; re-reading in 60 s"
    sleep 60
    CAP=$(HOST_TRIES=1 launch_cap)
  done
  cool
  if gpu_stage "$name" "$(left)" $FR "$kind" --set "$set" --reader "$reader" --cap "$CAP"; then DONE=$((DONE + 1))
  else log "BLOCKED-resumable $name (see $LOG/$name.log)"; fi
  commit_stage "Follow-up (D173, exploratory): control states, $set read by $reader" "$R6/states/$set/${reader}_info.json"
done

if [ "$DONE" = "$N" ]; then
  run_stage d173_analyze "$ANALYZE_S" $CPU_ENV $FC analyze || log "BLOCKED d173_analyze (see $LOG/d173_analyze.log)"
  commit_stage "Follow-up (D173, exploratory): the two controls' analysis" "$R6/controls_d173.json"
else
  log "SKIP d173_analyze: $((N - DONE)) of $N stages incomplete (re-run to resume)"
fi
log "D173 chain done: $DONE of $N stages"

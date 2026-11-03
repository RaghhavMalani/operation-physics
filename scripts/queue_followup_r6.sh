#!/usr/bin/env bash
# D167 (EXPLORATORY, follow-up project, paper #2): R6-3 / R6-4 / R6-5's GPU readouts on the frozen item sets
# (results/followup/r6/items), then their CPU analysis. NOT RUN YET: D167 waits for your go.
#   1. the stages of `python -m physcal.eval.followup_r6 stages`, in that order (the 1.5B readers first, then the 7B's),
#      each resumable: states = every layer at p1 / p2, yes = direct P(yes);
#   2. the analysis (CPU, 4 jobs with the GPU idle, D97), only if every stage finished.
# Before every stage: D158's VRAM gate for its reader (the 7B 6.5 GiB, the 1.5B readers 3.5), then the temperature
# guard (D143). Inside it: the stall watchdog (D132) and the free-space guard (D111). The GPU deadline gates each start.
# Commit + push after every stage. The functions are queue_r4_seeds.sh's and queue_xm2.sh's (library mode).
#
# usage:  GPU_DEADLINE=09:00 setsid nohup scripts/queue_followup_r6.sh >> results/queue_followup_r6/nohup.log 2>&1 < /dev/null &
#         DRY_RUN=1 DISK_GUARD=off QUEUE_LOG=<scratch> R6_OUT=<scratch> bash scripts/queue_followup_r6.sh   (runs nothing)
set -u
cd "$(dirname "$0")/.."
export QUEUE_LOG=${QUEUE_LOG:-results/queue_followup_r6}
CAP_MAX=${CAP_MAX:-7.5}                        # D151: total - host - 0.5 GiB
QUEUE_LIB_ONLY=1 . scripts/queue_r4_seeds.sh   # LOG, log, run_stage, commit_stage, cool, launch_cap
XM2_LIB_ONLY=1 . scripts/queue_xm2.sh          # CPU_ENV, gpu_stage
R6=${R6_OUT:-results/followup/r6}
MIN_STAGE_S=${MIN_STAGE_S:-900}                # a stage needs at least this much time left
ANALYZE_S=${ANALYZE_S:-43200}                  # the CPU analysis: ~20 probe cross-fits and 3 text classifiers
CAP_WAIT_S=${CAP_WAIT_S:-3600}                 # D158: wait this long for Windows to free VRAM, then BLOCKED
FR="$PY -m physcal.eval.followup_r6"
min_cap() { if [ "$1" = 7b ]; then echo "${MIN_CAP_7B:-6.5}"; else echo "${MIN_CAP_15B:-3.5}"; fi; }

exec 8>>"$LOG/queue.lock"
if ! flock -n 8; then
  log "already running: another queue holds $LOG/queue.lock; this copy (PID $$) exits 0"; exit 0
fi
echo $$ > "$LOG/queue.pid"
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-09:00}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "R6 chain start (D167, exploratory): GPU deadline $(ist "$DEADLINE") IST ($(( $(left) / 60 )) min left); chain PID $$"
export HF_HUB_OFFLINE=1
[ -f "$R6/items/manifest.json" ] || { log "BLOCKED: no frozen item sets in $R6/items (run followup_r6 items)"; exit 1; }
mapfile -t STAGES < <($FR stages)              # read first: a stage's child must not consume the list
[ "${#STAGES[@]}" -gt 0 ] || { log "BLOCKED: followup_r6 lists no stages"; exit 1; }

DONE=0 N=0
for line in "${STAGES[@]}"; do
  read -r kind set reader <<< "$line"
  N=$((N + 1))
  name="r6_${kind}_${set}_${reader}"
  if [ "$(left)" -lt "$MIN_STAGE_S" ]; then log "SKIP $name: $(( $(left) / 60 )) min left (re-run to resume)"; continue; fi
  need=$(min_cap "$reader")
  CAP=$(launch_cap)
  t_wait=$(date +%s)
  while awk -v c="$CAP" -v m="$need" 'BEGIN {exit !(c + 0 < m + 0)}'; do   # D158: never start below the need
    if [ $(( $(date +%s) - t_wait )) -ge "$CAP_WAIT_S" ] || [ "$(left)" -lt "$MIN_STAGE_S" ]; then
      log "BLOCKED: launch cap ${CAP} GiB < ${need} for $name: Windows holds too much VRAM (close GPU apps, then re-run)"
      exit 1
    fi
    log "launch cap ${CAP} GiB < ${need}: Windows holds $(smi memory.used) MiB of VRAM; re-reading in 60 s"
    sleep 60
    CAP=$(HOST_TRIES=1 launch_cap)
  done
  cool
  if gpu_stage "$name" "$(left)" $FR "$kind" --set "$set" --reader "$reader" --cap "$CAP"; then DONE=$((DONE + 1))
  else log "BLOCKED-resumable $name (see $LOG/$name.log)"; fi
  commit_stage "Follow-up (D167, exploratory): R6 $kind, $set read by $reader" \
    "$R6/states/$set/${reader}_info.json" "$R6/yes/$set/$reader.jsonl"
done

if [ "$DONE" = "$N" ]; then
  run_stage r6_analyze "$ANALYZE_S" $CPU_ENV $FR analyze || log "BLOCKED r6_analyze (see $LOG/r6_analyze.log)"
  commit_stage "Follow-up (D167, exploratory): R6-3 / R6-4 / R6-5's analysis" "$R6/results.json"
else
  log "SKIP r6_analyze: $((N - DONE)) of $N stages incomplete (re-run to resume)"
fi
log "R6 chain done: $DONE of $N stages; ${GPU_PAUSES:-0} temperature pauses, ${GPU_PAUSED_S:-0}s paused"

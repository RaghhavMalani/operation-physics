#!/usr/bin/env bash
# GPU chain, 2026-10-31 night (DECISIONS.md D151 / D152; EXPLORATORY, a follow-up project, not the paper): item 3,
# the scale check of the epistemic wall. The feasibility runs (7B and the native 1.5B control) are done by hand first.
#   1. plan: D151's stop time for 7B sampling (physcal.eval.followup_scale plan; written to the scale dir's plan.json).
#   2. 7B sampling: chunks of X2's pool until the stop time, resumable (one file per chunk).
#   3. 7B direct P(yes) on the kept answers; 4. 7B hidden states on the kept answers (both resumable).
#   5-7. The native 1.5B control on the 7B's prefix, no further: sampling, P(yes), hidden states, while time is left.
# The functions are queue_r4_seeds.sh's and queue_xm2.sh's (library mode): temperature guard (D143), stall watchdog
# (D132), free-space guard before every stage, GPU_DEADLINE gate, commit + push after every stage.
#
# usage:  GPU_DEADLINE=11:00 setsid nohup scripts/queue_followup.sh >> results/queue_followup/nohup.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
export QUEUE_LOG=${QUEUE_LOG:-results/queue_followup}
CAP_MAX=${CAP_MAX:-7.5}                        # D151: total - host - 0.5 GiB; no R4 parity cap (no R4 comparison)
QUEUE_LIB_ONLY=1 . scripts/queue_r4_seeds.sh   # LOG, log, run_stage, commit_stage, cool, launch_cap
XM2_LIB_ONLY=1 . scripts/queue_xm2.sh          # CPU_ENV, gpu_stage
SC=${SC_OUT:-results/followup/scale}
MIN_STAGE_S=${MIN_STAGE_S:-900}                # a readout or control stage needs at least this much time left
FS="$PY -m physcal.eval.followup_scale"

exec 8>>"$LOG/queue.lock"
if ! flock -n 8; then
  log "already running: another queue holds $LOG/queue.lock; this copy (PID $$) exits 0"; exit 0
fi
echo $$ > "$LOG/queue.pid"
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-11:00}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "follow-up chain start (D151, exploratory); GPU deadline $(ist "$DEADLINE") IST ($(( $(left) / 60 )) min left); chain PID $$"
export HF_HUB_OFFLINE=1
[ -f "$SC/7b/feasibility.json" ] || { log "BLOCKED: no 7B feasibility ($SC/7b/feasibility.json)"; exit 1; }
CAP=$(launch_cap)
log "launch cap ${CAP} GiB (max ${CAP_MAX})"

# ---- 1. plan ------------------------------------------------------------------------------------------------------------
run_stage fs_plan 600 $CPU_ENV $FS plan --deadline "$DEADLINE" || { log "BLOCKED fs_plan (see $LOG/fs_plan.log)"; exit 1; }
T_STOP=$(json_get "$SC/plan.json" "['t_stop']" | cut -d. -f1)
log "plan: $(json_get "$SC/plan.json" "['n_planned']") candidates planned, 7B sampling stops by $(ist "$T_STOP") IST; control planned: $(json_get "$SC/plan.json" "['control_planned']")"
commit_stage "Follow-up (D151, exploratory): the plan for item 3 (stop time from the feasibility numbers)" "$SC/plan.json" "$SC"/*/feasibility.json

n_chunks() { ls "$SC/$1/samples"/chunk_*.jsonl 2>/dev/null | wc -l; }

# ---- 2. 7B sampling -----------------------------------------------------------------------------------------------------
cool
T_LIM=$(( T_STOP - $(date +%s) + 1800 )); [ "$T_LIM" -gt "$(left)" ] && T_LIM=$(left)
gpu_stage fs_sample_7b "$T_LIM" $FS sample --model 7b --cap "$CAP" --stop-at "$T_STOP" \
  || log "fs_sample_7b ended with an error (see $LOG/fs_sample_7b.log); the completed chunks stand (resumable)"
log "7B: $(n_chunks 7b) chunks on disk"
commit_stage "Follow-up (D151, exploratory): 7B samples, $(n_chunks 7b) chunks of X2's pool" "$SC/7b/samples"

# ---- 3-4. 7B readouts ---------------------------------------------------------------------------------------------------
for what in yes hidden; do
  if [ "$(left)" -lt "$MIN_STAGE_S" ]; then log "SKIP fs_${what}_7b: $(( $(left) / 60 )) min left"; continue; fi
  cool
  gpu_stage "fs_${what}_7b" "$(left)" $FS "$what" --model 7b --cap "$CAP" || log "BLOCKED-resumable fs_${what}_7b (see $LOG/fs_${what}_7b.log)"
  commit_stage "Follow-up (D151, exploratory): 7B $what readout on the kept answers" "$SC/7b/yes.jsonl" "$SC/7b/hidden_info.json"
done

# ---- 5-7. the native 1.5B control on the 7B's prefix --------------------------------------------------------------------
LIMIT=$(( $(n_chunks 7b) * 24 ))
if [ ! -f "$SC/1.5b/feasibility.json" ]; then
  log "SKIP control: no 1.5B feasibility"
else
  for what in sample yes hidden; do
    if [ "$(left)" -lt "$MIN_STAGE_S" ]; then log "SKIP fs_${what}_1.5b: $(( $(left) / 60 )) min left"; continue; fi
    cool
    if [ "$what" = sample ]; then
      gpu_stage fs_sample_1.5b "$(left)" $FS sample --model 1.5b --cap "$CAP" --stop-at "$(( DEADLINE - MIN_STAGE_S * 2 ))" --limit "$LIMIT" \
        || log "fs_sample_1.5b ended with an error (see $LOG/fs_sample_1.5b.log)"
      commit_stage "Follow-up (D151, exploratory): native 1.5B control samples, $(n_chunks 1.5b) chunks (7B prefix: $LIMIT)" "$SC/1.5b/samples"
    else
      gpu_stage "fs_${what}_1.5b" "$(left)" $FS "$what" --model 1.5b --cap "$CAP" || log "BLOCKED-resumable fs_${what}_1.5b"
      commit_stage "Follow-up (D151, exploratory): native 1.5B $what readout" "$SC/1.5b/yes.jsonl" "$SC/1.5b/hidden_info.json"
    fi
  done
fi
log "follow-up chain done: $GPU_PAUSES temperature pauses, ${GPU_PAUSED_S}s paused of ${GPU_WALL_S}s GPU-stage time"

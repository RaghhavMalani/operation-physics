#!/usr/bin/env bash
# Local step after the Kaggle run (docs/exploratory/DECISIONS_FOLLOWUP.md D157; EXPLORATORY, a follow-up project, not
# the paper): item 3's readouts and D151's analysis on the 7B samples drawn on Kaggle.
#   1. ingest the downloaded zip (CPU): safe extraction, manifest, provenance, prompts re-rendered with the local
#      tokenizer, every row re-graded; into results/followup/scale/7b/samples (idempotent).
#   2. 7B direct P(yes) on the kept answers; 3. 7B hidden states on the kept answers (GPU, resumable; the local 4-bit
#      copy of the same weights, D157).
#   4. D151's analysis (CPU, after the GPU stages; the probe uses 4 jobs, D97).
# The functions are queue_r4_seeds.sh's and queue_xm2.sh's (library mode): temperature guard (D143), stall watchdog
# (D132), free-space guard before every stage, GPU_DEADLINE gate, commit + push after every stage. A re-run resumes.
#
# usage:  KAGGLE_ZIP=/path/to/followup_7b_scale.zip GPU_DEADLINE=09:00 setsid nohup scripts/queue_followup_kaggle.sh \
#           >> results/queue_followup_kaggle/nohup.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
export QUEUE_LOG=${QUEUE_LOG:-results/queue_followup_kaggle}
CAP_MAX=${CAP_MAX:-7.5}                        # D151: total - host - 0.5 GiB
QUEUE_LIB_ONLY=1 . scripts/queue_r4_seeds.sh   # LOG, log, run_stage, commit_stage, cool, launch_cap
XM2_LIB_ONLY=1 . scripts/queue_xm2.sh          # CPU_ENV, gpu_stage
SC=${SC_OUT:-results/followup/scale}
MIN_STAGE_S=${MIN_STAGE_S:-900}                # a readout stage needs at least this much time left
ANALYZE_S=${ANALYZE_S:-21600}                  # the CPU analysis (29 layers, grouped CV), after the GPU deadline too
MIN_CAP=${MIN_CAP:-6.5}                        # D158: the 4-bit 7B is 5.23 GiB loaded; the longest readouts need ~6.5
CAP_WAIT_S=${CAP_WAIT_S:-3600}                 # D158: wait this long for Windows to free VRAM, then BLOCKED (no stage)
FS="$PY -m physcal.eval.followup_scale"
FK="$PY -m physcal.eval.followup_kaggle"

exec 8>>"$LOG/queue.lock"
if ! flock -n 8; then
  log "already running: another queue holds $LOG/queue.lock; this copy (PID $$) exits 0"; exit 0
fi
echo $$ > "$LOG/queue.pid"
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-09:00}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "follow-up Kaggle chain start (D157, exploratory); GPU deadline $(ist "$DEADLINE") IST ($(( $(left) / 60 )) min left); chain PID $$"
export HF_HUB_OFFLINE=1

# ---- 1. ingest ----------------------------------------------------------------------------------------------------------
if [ -n "${KAGGLE_ZIP:-}" ]; then
  [ -f "$KAGGLE_ZIP" ] || { log "BLOCKED: no zip at $KAGGLE_ZIP"; exit 1; }
  run_stage fk_ingest 3600 $CPU_ENV $FK ingest --zip "$KAGGLE_ZIP" || { log "BLOCKED fk_ingest (see $LOG/fk_ingest.log)"; exit 1; }
  log "ingest: $(json_get "$SC/7b/kaggle/ingest.json" "['n_candidates']") candidates, $(json_get "$SC/7b/kaggle/ingest.json" "['regraded_changed_rows']") rows re-graded differently"
  commit_stage "Follow-up (D157, exploratory): the 7B's Kaggle samples ingested ($(json_get "$SC/7b/kaggle/ingest.json" "['n_candidates']") candidates)" \
    "$SC/7b/samples" "$SC/7b/kaggle"
fi
ls "$SC/7b/samples"/chunk_*.jsonl > /dev/null 2>&1 || { log "BLOCKED: no 7B samples (set KAGGLE_ZIP)"; exit 1; }

# ---- 2-3. 7B readouts on the kept answers -------------------------------------------------------------------------------
CAP=$(launch_cap)
log "launch cap ${CAP} GiB (max ${CAP_MAX}, needed ${MIN_CAP})"
t_wait=$(date +%s)
while awk -v c="$CAP" -v m="$MIN_CAP" 'BEGIN {exit !(c + 0 < m + 0)}'; do   # D158: never start below the need
  if [ $(( $(date +%s) - t_wait )) -ge "$CAP_WAIT_S" ] || [ "$(left)" -lt "$MIN_STAGE_S" ]; then
    log "BLOCKED: launch cap ${CAP} GiB < ${MIN_CAP} after $(( ($(date +%s) - t_wait) / 60 )) min: Windows holds too much VRAM (close GPU apps, then re-run; nothing was started)"
    exit 1
  fi
  log "launch cap ${CAP} GiB < ${MIN_CAP}: Windows holds $(smi memory.used) MiB of VRAM; close GPU apps; re-reading in 60 s"
  sleep 60
  CAP=$(HOST_TRIES=1 launch_cap)
done
log "launch cap ${CAP} GiB: starting the readouts"
DONE=0                                         # readouts that finished (exit 0; a finished one re-runs in seconds)
for what in yes hidden; do
  if [ "$(left)" -lt "$MIN_STAGE_S" ]; then log "SKIP fs_${what}_7b: $(( $(left) / 60 )) min left (re-run to resume)"; continue; fi
  cool
  if gpu_stage "fs_${what}_7b" "$(left)" $FS "$what" --model 7b --cap "$CAP"; then DONE=$((DONE + 1))
  else log "BLOCKED-resumable fs_${what}_7b (see $LOG/fs_${what}_7b.log)"; fi
  commit_stage "Follow-up (D157, exploratory): 7B $what readout on the kept answers (Kaggle samples)" "$SC/7b/yes.jsonl" "$SC/7b/hidden_info.json"
done

# ---- 4. D151's analysis -------------------------------------------------------------------------------------------------
if [ "$DONE" = 2 ]; then
  run_stage fs_analyze "$ANALYZE_S" $CPU_ENV $FS analyze || log "BLOCKED fs_analyze (see $LOG/fs_analyze.log)"
  commit_stage "Follow-up (D157, exploratory): item 3's analysis on the Kaggle samples" "$SC/results.json" "$SC/summary.md"
else
  log "SKIP fs_analyze: a readout is incomplete (re-run to resume)"
fi
log "follow-up Kaggle chain done: $GPU_PAUSES temperature pauses, ${GPU_PAUSED_S}s paused of ${GPU_WALL_S}s GPU-stage time"

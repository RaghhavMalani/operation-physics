#!/usr/bin/env bash
# Local step after a Kaggle run (docs/exploratory/DECISIONS_FOLLOWUP.md D157, D160; EXPLORATORY, a follow-up project,
# not the paper): item 3's readouts and D151's analysis on a model's samples drawn on Kaggle. MODEL = 7b (default) or
# 1.5b (D160's native control).
#   1. ingest the downloaded zip (CPU): safe extraction, manifest, provenance, prompts re-rendered with the local
#      tokenizer, every row re-graded; into results/followup/scale/$MODEL/samples (idempotent).
#   2. D160's random-weights copy of the readout model (CPU; ~15 GB for the 7B), if GPU_STAGES has hidden_random.
#   3. GPU_STAGES in order (resumable; the local 4-bit copy, D157): yes = direct P(yes), hidden = the hidden states,
#      hidden_random = the hidden states of the random-weights copy (the probe's control, D160).
#   4. D151's analysis with D160's controls (CPU, after the GPU stages; the probes use 4 jobs, D97), only if every
#      GPU stage finished.
# The functions are queue_r4_seeds.sh's and queue_xm2.sh's (library mode): temperature guard (D143), stall watchdog
# (D132), free-space guard before every stage, GPU_DEADLINE gate, commit + push after every stage. A re-run resumes.
#
# usage:  [MODEL=1.5b] [KAGGLE_ZIP=/path/to/followup_<model>_scale.zip] [GPU_STAGES="hidden_random"] GPU_DEADLINE=09:00 \
#           setsid nohup scripts/queue_followup_kaggle.sh >> results/queue_followup_kaggle/nohup.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
export QUEUE_LOG=${QUEUE_LOG:-results/queue_followup_kaggle}
CAP_MAX=${CAP_MAX:-7.5}                        # D151: total - host - 0.5 GiB
QUEUE_LIB_ONLY=1 . scripts/queue_r4_seeds.sh   # LOG, log, run_stage, commit_stage, cool, launch_cap
XM2_LIB_ONLY=1 . scripts/queue_xm2.sh          # CPU_ENV, gpu_stage
SC=${SC_OUT:-results/followup/scale}
MODEL=${MODEL:-7b}
GPU_STAGES=${GPU_STAGES:-"yes hidden hidden_random"}
MIN_STAGE_S=${MIN_STAGE_S:-900}                # a readout stage needs at least this much time left
ANALYZE_S=${ANALYZE_S:-21600}                  # the CPU analysis (29 layers, grouped CV), after the GPU deadline too
case $MODEL in                                 # D158 / D160: the VRAM a model's longest readouts need; the random copy
  7b)   MIN_CAP=${MIN_CAP:-6.5}; RANDOM_GB=${RANDOM_GB:-16} ;;   # 5.23 GiB loaded; 15.2 GB of bf16 shards
  1.5b) MIN_CAP=${MIN_CAP:-3.5}; RANDOM_GB=${RANDOM_GB:-4} ;;    # 1.12 GiB loaded, 2.33 at batch 8 (D153); 3.1 GB
  *)    echo "unknown MODEL=$MODEL" >&2; exit 2 ;;
esac
CAP_WAIT_S=${CAP_WAIT_S:-3600}                 # D158: wait this long for Windows to free VRAM, then BLOCKED (no stage)
FS="$PY -m physcal.eval.followup_scale"
FK="$PY -m physcal.eval.followup_kaggle"

exec 8>>"$LOG/queue.lock"
if ! flock -n 8; then
  log "already running: another queue holds $LOG/queue.lock; this copy (PID $$) exits 0"; exit 0
fi
echo $$ > "$LOG/queue.pid"
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-09:00}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "follow-up Kaggle chain start (D157/D160, exploratory): MODEL=$MODEL, GPU_STAGES='$GPU_STAGES'; GPU deadline $(ist "$DEADLINE") IST ($(( $(left) / 60 )) min left); chain PID $$"
export HF_HUB_OFFLINE=1

# ---- 1. ingest ----------------------------------------------------------------------------------------------------------
if [ -n "${KAGGLE_ZIP:-}" ]; then
  [ -f "$KAGGLE_ZIP" ] || { log "BLOCKED: no zip at $KAGGLE_ZIP"; exit 1; }
  run_stage "fk_ingest_$MODEL" 3600 $CPU_ENV $FK ingest --model "$MODEL" --zip "$KAGGLE_ZIP" || { log "BLOCKED fk_ingest_$MODEL (see $LOG/fk_ingest_$MODEL.log)"; exit 1; }
  log "ingest: $(json_get "$SC/$MODEL/kaggle/ingest.json" "['n_candidates']") candidates, $(json_get "$SC/$MODEL/kaggle/ingest.json" "['regraded_changed_rows']") rows re-graded differently"
  commit_stage "Follow-up (D157/D160, exploratory): the $MODEL's Kaggle samples ingested ($(json_get "$SC/$MODEL/kaggle/ingest.json" "['n_candidates']") candidates)" \
    "$SC/$MODEL/samples" "$SC/$MODEL/kaggle"
fi
ls "$SC/$MODEL/samples"/chunk_*.jsonl > /dev/null 2>&1 || { log "BLOCKED: no $MODEL samples (set KAGGLE_ZIP)"; exit 1; }

# ---- 2. D160's random-weights copy (CPU) --------------------------------------------------------------------------------
RANDOM_DIR=$($PY -c "from physcal.eval import followup_scale as F; print(F.random_init_dir('$MODEL', F._config()))")
if [[ " $GPU_STAGES " == *" hidden_random "* ]]; then
  if [ -f "$RANDOM_DIR/physcal_random_init.json" ]; then
    log "random-weights copy present: $RANDOM_DIR ($(json_get "$RANDOM_DIR/physcal_random_init.json" "['content_sha256']"))"
  else  # its size on top of the usual free-space floor (the disk image does not shrink by itself)
    MIN_FREE_GB=$((MIN_FREE_GB + RANDOM_GB)) run_stage "fs_random_$MODEL" 3600 $CPU_ENV $FS random_init --model "$MODEL" \
      || { log "BLOCKED fs_random_$MODEL (see $LOG/fs_random_$MODEL.log)"; exit 1; }
  fi
fi

# ---- 3. the GPU readouts on the kept answers ----------------------------------------------------------------------------
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
DONE=0 N=0                                     # readouts that finished (exit 0; a finished one re-runs in seconds)
for what in $GPU_STAGES; do
  N=$((N + 1))
  case $what in
    yes|hidden) args=("$what") ;;
    hidden_random) args=(hidden --random-init) ;;
    *) log "BLOCKED: unknown GPU stage $what"; exit 1 ;;
  esac
  name="fs_${what}_${MODEL}"
  if [ "$(left)" -lt "$MIN_STAGE_S" ]; then log "SKIP $name: $(( $(left) / 60 )) min left (re-run to resume)"; continue; fi
  cool
  if gpu_stage "$name" "$(left)" $FS "${args[@]}" --model "$MODEL" --cap "$CAP"; then DONE=$((DONE + 1))
  else log "BLOCKED-resumable $name (see $LOG/$name.log)"; fi
  commit_stage "Follow-up (D157/D160, exploratory): $MODEL $what readout on the kept answers (Kaggle samples)" \
    "$SC/$MODEL/yes.jsonl" "$SC/$MODEL/hidden_info.json" "$SC/$MODEL/hidden_random_info.json"
done

# ---- 4. D151's analysis with D160's controls ----------------------------------------------------------------------------
if [ "$DONE" = "$N" ]; then
  run_stage fs_analyze "$ANALYZE_S" $CPU_ENV $FS analyze || log "BLOCKED fs_analyze (see $LOG/fs_analyze.log)"
  commit_stage "Follow-up (D157/D160, exploratory): item 3's analysis ($MODEL readouts)" "$SC/results.json" "$SC/summary.md" \
    "$SC"/*/probe_hidden.json "$SC"/*/probe_hidden_random.json "$SC/x2_sft_probe_refit.json"
else
  log "SKIP fs_analyze: a readout is incomplete (re-run to resume)"
fi
log "follow-up Kaggle chain done: $GPU_PAUSES temperature pauses, ${GPU_PAUSED_S}s paused of ${GPU_WALL_S}s GPU-stage time"

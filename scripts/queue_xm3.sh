#!/usr/bin/env bash
# Day GPU chain, 2026-10-30 (DECISIONS.md D146): the XM pilot's last two stages.
#   1. Eval B-v2 (Llama, seed 0) on the frozen A6c + A6b, resumable (D142): queue_xm2.sh's eval_xm, the path that ran
#      D-pos-v2's eval on 29/30. A run cut off at GPU_DEADLINE stays BLOCKED-but-resumable; the next run continues it.
#   2. The pilot analysis (physcal.eval.xm_pilot; CPU, 1 thread, nice 19), once, when both Llama evals are complete:
#      D137's descriptive items, including D-pos-v2 (explicit) vs B-v2 (implicit) on Llama.
# The functions are queue_r4_seeds.sh's and queue_xm2.sh's (library mode): temperature guard (D143), stall watchdog
# (D132), free-space guard before every stage, GPU_DEADLINE gate, commit + push after every stage, skips for done work.
#
# usage:  GPU_DEADLINE=13:00 setsid nohup scripts/queue_xm3.sh >> results/queue_xm3/nohup.log 2>&1 < /dev/null &
#         DRY_RUN=1 QUEUE_LOG=<scratch> XM_OUT=<scratch> bash scripts/queue_xm3.sh
set -u
cd "$(dirname "$0")/.."
export QUEUE_LOG=${QUEUE_LOG:-results/queue_xm3}
QUEUE_LIB_ONLY=1 . scripts/queue_r4_seeds.sh   # LOG, log, run_stage, commit_stage, cool, launch_cap, complete_adapter, ...
XM2_LIB_ONLY=1 . scripts/queue_xm2.sh          # XM_OUT, CPU_ENV, gpu_stage, eval_xm, the eval settings (D144, D146)
PILOT_TIMEOUT_S=${PILOT_TIMEOUT_S:-7200}       # CPU at 1 thread; D137 set no estimate

exec 8>>"$LOG/queue.lock"
if ! flock -n 8; then
  log "already running: another queue holds $LOG/queue.lock; this copy (PID $$) exits 0"; exit 0
fi
echo $$ > "$LOG/queue.pid"
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-13:00}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "XM B-v2 eval + pilot chain start (D146); GPU deadline $(ist "$DEADLINE") IST ($(( $(left) / 60 )) min left); chain PID $$"
export HF_HUB_OFFLINE=1   # every model is local (D78)
for f in data/eval_sets/2.0.0+bdbb2279/confirmatory/a6c_pairs.jsonl data/eval_sets/2.0.0+bdbb2279/confirmatory/a6b_pairs.jsonl \
         configs/base_llama32_3b_4bit.yaml; do
  [ -e "$f" ] || log "BLOCKED: missing $f"
done
CAP=$(launch_cap)
log "launch cap ${CAP} GiB (max ${CAP_MAX})"

# ---- 1. eval B-v2 (Llama, seed 0), resumable (D142) -------------------------------------------------------------------
eval_xm b2 D146

# ---- 2. the pilot analysis (D137), once, when both Llama evals are complete ---------------------------------------------
PILOT=$XM_OUT/pilot/results.json
if [ -f "$PILOT" ]; then
  log "SKIP xm_pilot: the one-time analysis exists ($PILOT; D137)"
elif { [ -f "$XM_OUT/dpos2/seed0/confirmatory/metrics.json" ] && [ -f "$XM_OUT/b2/seed0/confirmatory/metrics.json" ]; } \
     || [ "$DRY_RUN" = 1 ]; then
  run_stage xm_pilot "$PILOT_TIMEOUT_S" $CPU_ENV env PHYSCAL_CONFIRMATORY=1 $PY -m physcal.eval.xm_pilot analyze \
    || log "BLOCKED xm_pilot (see $LOG/xm_pilot.log)"
  commit_stage "XM pilot (D137, D146): the pilot analysis, D-pos-v2 and B-v2 on Llama-3.2-3B, seed 0" "$XM_OUT/pilot"
else
  log "xm_pilot not run: the B-v2 eval is not complete (resumable tonight; D146)"
fi
log "XM B-v2 eval + pilot chain done: $GPU_PAUSES temperature pauses, ${GPU_PAUSED_S}s paused of ${GPU_WALL_S}s GPU-stage time"

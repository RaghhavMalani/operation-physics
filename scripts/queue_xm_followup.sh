#!/usr/bin/env bash
# The conditional addition to the XM chain (DECISIONS.md D139, your instruction of 2026-10-29). It runs beside the
# chain and never touches it: queue_xm.sh and every module its stages import are unchanged.
#   1. When the Llama D-pos-v2 eval is over (its stage ended in the chain's log, or the chain exited): D137's go rule on
#      D-pos-v2 (physcal.eval.xm_go; CPU, 1 thread, nice 19), once; commit + push. A missing or incomplete eval means
#      the rule cannot be evaluated.
#   2. After the chain exits, under its queue lock: if and only if the rule passed, SFT D-pos-v2 Llama seed 1
#      (physcal.train.stage_xm_seed: stage XM's recipe, only the seed changes), gated to finish by GPU_DEADLINE. Its
#      estimate is the longest measured wall time of tonight's XM SFT stages (temperature pauses included), at least
#      XM_TRAIN_EST; commit + push. Its eval runs on a later night. NO-GO or not evaluable: nothing is added.
# The functions are queue_r4_seeds.sh's (library mode): the temperature guard, the stall watchdog, the free-space
# guard, the deadline gate and the commit under the shared flock.
#
# usage:  CHAIN_PID=<the XM chain's PID> GPU_DEADLINE=10:00 \
#           setsid nohup scripts/queue_xm_followup.sh >> results/queue_xm/followup/nohup.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
CHAIN_LOG_DIR=${CHAIN_LOG_DIR:-results/queue_xm}     # the XM chain's QUEUE_LOG: its chain.log, stage logs and lock
export QUEUE_LOG=${QUEUE_LOG:-$CHAIN_LOG_DIR/followup}
mkdir -p "$QUEUE_LOG"
QUEUE_LIB_ONLY=1 . scripts/queue_r4_seeds.sh       # LOG, log, run_stage, commit_stage, cool, launch_cap, ...
CHAIN_PID=${CHAIN_PID:?the XM chain PID}
XM_OUT=${XM_OUT:-results/stage_xm}
XM_TRAIN_EST=${XM_TRAIN_EST:-8400}                 # queue_xm.sh's; the measured wall time replaces it when longer
NEW_SEED=${NEW_SEED:-1}
POLL_S=${POLL_S:-60}
LOCK_WAIT_S=${LOCK_WAIT_S:-1800}                   # the chain's stage children hold its lock until they exit
CPU_ENV="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= nice -n 19"
GO=$XM_OUT/go_dpos2/go.json
alive() { kill -0 "$CHAIN_PID" 2>/dev/null; }
mtime() { stat -c %Y "$1" 2>/dev/null || echo 0; }

exec 7>>"$LOG/followup.lock"
if ! flock -n 7; then log "already running: another follow-up holds $LOG/followup.lock; this copy exits 0"; exit 0; fi
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-10:00}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
from=$(( $(wc -l < "$CHAIN_LOG_DIR/chain.log" 2>/dev/null || echo 0) + 1 ))
log "XM follow-up start (D139): chain PID $CHAIN_PID, its log from line $from; GPU deadline $(ist "$DEADLINE") IST; PID $$"

# ---- 1. the go rule, once, when the D-pos-v2 eval is over ------------------------------------------------------------
over() { tail -n +"$from" "$CHAIN_LOG_DIR/chain.log" 2>/dev/null | grep -qE "\] (end xm_eval_dpos2:|SKIP eval xm dpos2|BLOCKED eval xm dpos2)"; }
while alive && ! over && [ ! -f "$GO" ]; do sleep "$POLL_S"; done
if [ -f "$GO" ]; then
  log "SKIP xm_go: the one-time go rule exists ($GO)"
else
  run_stage xm_go 3600 $CPU_ENV env PHYSCAL_CONFIRMATORY=1 $PY -m physcal.eval.xm_go analyze \
    || log "go rule not evaluated (see $LOG/xm_go.log): nothing is added (D139)"
  commit_stage "XM pilot (D139): D137's go rule on D-pos-v2 (Llama, seed 0), before B-v2's eval" "$XM_OUT/go_dpos2"
fi
go=$(json_get "$GO" "['go']")
log "go rule: ${go} (True = GO; anything else adds nothing)"

# ---- 2. after the chain: SFT D-pos-v2 seed 1, if and only if GO -------------------------------------------------------
while alive; do sleep "$POLL_S"; done
log "XM chain (PID $CHAIN_PID) has exited"
if [ "$go" != True ]; then log "XM follow-up done: no seed $NEW_SEED stage (go rule: $go)"; exit 0; fi
exec 8>>"$CHAIN_LOG_DIR/queue.lock"
if ! flock -w "$LOCK_WAIT_S" 8; then log "BLOCKED xm_train_dpos2_seed$NEW_SEED: $CHAIN_LOG_DIR/queue.lock still held after ${LOCK_WAIT_S}s"; exit 0; fi
dir=$XM_OUT/dpos2/seed$NEW_SEED
est=$XM_TRAIN_EST
for a in dpos2 b2; do  # tonight's measured wall time per XM SFT stage: its PID file (start) to its log's last write
  p=$CHAIN_LOG_DIR/xm_train_$a
  [ -f "$p.pid" ] && [ -f "$p.log" ] && w=$(( $(mtime "$p.log") - $(mtime "$p.pid") )) && [ "$w" -gt "$est" ] && est=$w
done
if [ -e "$dir/final_adapter" ]; then
  log "SKIP train xm dpos2 seed $NEW_SEED: $dir/final_adapter exists"
elif [ "$(left)" -lt $((est + SLACK_S)) ] && [ "$DRY_RUN" != 1 ]; then
  log "SKIP train xm dpos2 seed $NEW_SEED: $(( $(left) / 60 )) min left, needs $(( (est + SLACK_S) / 60 )) min (measured)"
else
  cool
  CAP=$(launch_cap)
  log "launch cap ${CAP} GiB; estimate $(( est / 60 )) min"
  GUARD=1 run_stage "xm_train_dpos2_seed$NEW_SEED" $(( $(left) - 300 )) $PY -m physcal.train.stage_xm_seed train \
    --arm dpos2 --seed "$NEW_SEED" --stop-at "$(iso $((DEADLINE - SLACK_S)))" --max-reserved-gib "$CAP"; rc=$?
  [ "$rc" != 0 ] && log "BLOCKED? xm_train_dpos2_seed$NEW_SEED exit $rc (see $LOG/xm_train_dpos2_seed$NEW_SEED.log)"
  commit_stage "XM pilot (D137, D139: GO): SFT dpos2 on Llama-3.2-3B, seed $NEW_SEED, $(json_get "$dir/summary.json" "['steps']") steps (exit $rc)" \
    "$dir/summary.json" "$dir/events.jsonl" "$dir/final_adapter/physcal_fingerprint.json" "$dir/final_adapter/adapter_config.json"
fi
log "XM follow-up done"

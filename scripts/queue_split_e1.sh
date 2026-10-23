#!/usr/bin/env bash
# Tonight's GPU queue (prepared 2026-10-14; NOT launched by Claude). Stages, strictly one after another:
#   1. split-credit seed 0 to step 287 (permanent checkpoint), PREREGISTRATION family S
#   2. eval of the step-287 adapter on the frozen 1.1.0 eval sets, then the sanity check (no OOM, parse failure < 5%;
#      recorded in sanity.json) and the family S / N comparisons (CPU)
#   3. E1a (free sigma) to <= 300 steps, then its eval (D52)
#   4. E1b (binned sigma) to <= 300 steps, then its eval
#   5. E1 family E comparison (CPU)
# E1 runs regardless of the stage-2 sanity verdict (your brief replaced "seed 1 only if sane" with E1a/E1b).
# Waiting is PID-based only: every stage runs as a child whose PID is written to $LOG/<stage>.pid and is waited on
# with `wait`/`kill -0` (never pgrep -f). GPU work ends at GPU_DEADLINE (default 08:30 IST, next occurrence); the
# E1 runs share the time left after stage 2 equally. GPU temp > 87 C before an eval -> pause 10 min.
#
# usage:  scripts/queue_split_e1.sh                # run
#         DRY_RUN=1 scripts/queue_split_e1.sh      # print the plan and deadlines, run nothing
#         GPU_DEADLINE="07:30" scripts/queue_split_e1.sh
set -u
cd "$(dirname "$0")/.."
. scripts/disk_guard.sh   # the free-space guard (D111): disk_guard before every stage
LABEL=qwen2.5-math-1.5b-instruct
SC_CFG=grpo_split_credit
SC_RUN=results/train/split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed0
E1A_RUN=results/train/e1a-sigma-only-free-seed0
E1B_RUN=results/train/e1b-sigma-only-bins-seed0
LOG=${QUEUE_LOG:-results/queue_split_e1}
EVAL_RESERVE_S=${EVAL_RESERVE_S:-1800}      # time kept back for the step-287 eval (~15-25 min measured before)
E1_EVAL_RESERVE_S=${E1_EVAL_RESERVE_S:-600} # per E1 eval (400 prefixes x <= 24 greedy tokens)
MIN_STAGE_S=${MIN_STAGE_S:-900}             # do not start a training stage with less than this left
DRY_RUN=${DRY_RUN:-0}
mkdir -p "$LOG"
echo $$ > "$LOG/queue.pid"

ist() { TZ=Asia/Kolkata date -d "@$1" +%H:%M; }
log() { echo "[$(TZ=Asia/Kolkata date +%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }
iso() { TZ=Asia/Kolkata date -d "@$1" --iso-8601=seconds; }
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-08:30}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
cool() { [ "$DRY_RUN" = 1 ] && return; while t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "${t:-0}" -gt 87 ]; do
           log "GPU at ${t}C > 87C: pausing 600s"; sleep 600; done; log "GPU at ${t}C"; }

# run_stage NAME TIMEOUT_S CMD...: child process, PID file, wait; returns the child's exit code
run_stage() {
  local name=$1 limit=$2; shift 2
  disk_guard "$name" || exit 75   # < MIN_FREE_GB on MLStorage (or unreadable): stop the queue (D111)
  if [ "$DRY_RUN" = 1 ]; then log "DRY $name (timeout ${limit}s): $*"; return 0; fi
  if [ "$limit" -lt 300 ]; then log "skip $name: only ${limit}s left before the GPU deadline"; return 98; fi
  log "start $name (timeout ${limit}s): $*"
  timeout "$limit" "$@" > "$LOG/$name.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$LOG/$name.pid"
  wait "$pid"; local rc=$?
  log "end $name: exit $rc"
  return $rc
}

left() { echo $((DEADLINE - $(date +%s))); }
log "queue start; GPU deadline $(ist "$DEADLINE") IST; queue PID $$"

# ---- 1. split-credit seed 0 to step 287 -------------------------------------------------------------------
STOP=$((DEADLINE - EVAL_RESERVE_S))
if [ $((STOP - $(date +%s))) -lt "$MIN_STAGE_S" ]; then
  log "not enough time for split-credit training"; SC_RC=99
else
  run_stage split_train $((STOP - $(date +%s) + 600)) uv run python -m physcal.train.grpo --config "$SC_CFG" \
    --max-steps 287 --stop-at "$(iso "$STOP")"; SC_RC=$?
fi

# ---- 2. eval of the permanent step-287 adapter, sanity check, family S / N ---------------------------------
if [ "$DRY_RUN" = 1 ] || [ -d "$SC_RUN/permanent_checkpoints/checkpoint-287" ]; then
  cool
  run_stage split_eval287 "$(left)" uv run python -m physcal.train.compare --part rl --run "$SC_RUN" --label "$LABEL" \
    --budget 512 --adapter permanent_checkpoints/checkpoint-287 --out-name rl287
  [ "$DRY_RUN" = 1 ] || uv run python scripts/queue_sanity.py "$SC_RUN" "$SC_RC" > "$LOG/sanity.log" 2>&1
  [ "$DRY_RUN" = 1 ] || { cat "$LOG/sanity.log" | tee -a "$LOG/chain.log"; }
  run_stage split_compare 1800 uv run python scripts/phase95_compare.py "$SC_RUN"
else
  log "MISSING $SC_RUN/permanent_checkpoints/checkpoint-287 (training exit $SC_RC): no eval, sanity = FAIL"
  [ "$DRY_RUN" = 1 ] || uv run python scripts/queue_sanity.py "$SC_RUN" "$SC_RC" > "$LOG/sanity.log" 2>&1
fi

# ---- 3./4. E1a then E1b, sharing the time left equally -------------------------------------------------------
e1_stage() {  # e1_stage NAME CONFIG RUN_DIR SHARE_END
  local name=$1 cfg=$2 run=$3 end=$4
  local stop=$((end - E1_EVAL_RESERVE_S))
  if [ $((stop - $(date +%s))) -lt "$MIN_STAGE_S" ]; then log "not enough time for $name"; return; fi
  run_stage "${name}_train" $((stop - $(date +%s) + 600)) uv run python -m physcal.train.e1 train --config "$cfg" \
    --stop-at "$(iso "$stop")"
  cool
  for spec in "final_adapter final" "permanent_checkpoints/checkpoint-150 step150"; do
    set -- $spec
    if [ "$DRY_RUN" = 1 ] || [ -d "$run/$1" ]; then
      run_stage "${name}_eval_$2" "$(left)" uv run python -m physcal.train.e1 eval --run "$run" --adapter "$1" --out-name "$2"
    else
      log "MISSING $run/$1, skipping its eval"
    fi
  done
}
NOW=$(date +%s); HALF=$(( (DEADLINE - NOW) / 2 ))
log "E1 time split: E1a until $(ist $((NOW + HALF))) IST, E1b until $(ist "$DEADLINE") IST"
e1_stage e1a grpo_e1a "$E1A_RUN" $((NOW + HALF))
e1_stage e1b grpo_e1b "$E1B_RUN" "$DEADLINE"

# ---- 5. E1 family E (CPU) ----------------------------------------------------------------------------------
run_stage e1_compare 1800 uv run python -m physcal.train.e1 compare
log "queue done"

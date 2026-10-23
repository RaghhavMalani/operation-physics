#!/usr/bin/env bash
# D65 (exploratory): RL adapters on the 4-bit training base, greedy and k = $SAMPLES sampled, one stage at a time.
#   1. E1a final, greedy                        -> <E1a run>/e1_eval/final_4bit
#   2. split-credit seed 0 @287, greedy         -> <split run>/compare/rl287_4bit
#   3. E1a final, k sampled                     -> <E1a run>/e1_eval/final_4bit_sampled
#   4. split-credit seed 0 @287, k sampled      -> <split run>/compare/rl287_4bit_sampled
# Every stage is a child whose PID goes to $LOG/<stage>.pid and is waited on (never pgrep -f). GPU temp > 87 C
# before a stage -> pause 10 min. A failed stage is logged and the queue moves on. Commits are made by hand afterwards.
#
# usage:  scripts/queue_precision_check.sh             # run
#         DRY_RUN=1 scripts/queue_precision_check.sh   # print the plan, run nothing
set -u
cd "$(dirname "$0")/.."
. scripts/disk_guard.sh   # the free-space guard (D111): disk_guard before every stage
LABEL=qwen2.5-math-1.5b-instruct
SPLIT_RUN=results/train/split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed0
E1A_RUN=results/train/e1a-sigma-only-free-seed0
LOG=${QUEUE_LOG:-results/precision_check/queue}
SAMPLES=${SAMPLES:-4}                 # D62 / D65: k sampled completions per problem
STAGE_TIMEOUT_S=${STAGE_TIMEOUT_S:-7200}
TEMP_LIMIT_C=${TEMP_LIMIT_C:-87}
PAUSE_S=${PAUSE_S:-600}
DRY_RUN=${DRY_RUN:-0}
mkdir -p "$LOG"
echo $$ > "$LOG/queue.pid"

log() { echo "[$(TZ=Asia/Kolkata date +%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }
cool() { [ "$DRY_RUN" = 1 ] && return; while t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "${t:-0}" -gt "$TEMP_LIMIT_C" ]; do
           log "GPU at ${t}C > ${TEMP_LIMIT_C}C: pausing ${PAUSE_S}s"; sleep "$PAUSE_S"; done; log "GPU at ${t}C"; }

run_stage() {  # run_stage NAME CMD...: child process, PID file, wait; logs the child's exit code
  local name=$1; shift
  disk_guard "$name" || exit 75   # < MIN_FREE_GB on MLStorage (or unreadable): stop the queue (D111)
  if [ "$DRY_RUN" = 1 ]; then log "DRY $name: $*"; return 0; fi
  cool
  log "start $name: $*"
  timeout "$STAGE_TIMEOUT_S" "$@" > "$LOG/$name.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$LOG/$name.pid"
  wait "$pid"; local rc=$?
  log "end $name: exit $rc"
  return $rc
}

E1="uv run python -m physcal.train.e1 eval --run $E1A_RUN --adapter final_adapter --loader unsloth_4bit"
CMP="uv run python -m physcal.train.compare --part rl --run $SPLIT_RUN --label $LABEL --budget 512 \
--adapter permanent_checkpoints/checkpoint-287 --loader unsloth_4bit"

log "queue start (pid $$)"
run_stage e1a_4bit_greedy $E1 --out-name final_4bit
run_stage split0_4bit_greedy $CMP --out-name rl287_4bit
run_stage e1a_4bit_sampled $E1 --out-name final_4bit_sampled --samples "$SAMPLES"
run_stage split0_4bit_sampled $CMP --out-name rl287_4bit_sampled --samples "$SAMPLES"
log "queue done"

#!/usr/bin/env bash
# Overnight GPU queue, 2026-10-16/17 (DECISIONS.md D62). Starts only after the previous queue's PID has exited.
#   0. commit whatever the previous queue (E1b, family E) left uncommitted
#   1. split-credit seed 1 to step 287 (permanent checkpoint) -> step-287 eval -> sanity -> phase95_compare (CPU)
#   2. seed 2, the same, ONLY if seed 1's sanity.json says sane
#   3. exploratory sampled evals, k = $SAMPLES at the training settings: split-credit seed 0 @287, E1a final, E1b final
#   4. CPU: the seed-mean family S test (if all 3 seeds have a step-287 eval) and the greedy-vs-sampled report
# Every stage is a child whose PID goes to $LOG/<stage>.pid and is waited on with `wait` (never pgrep -f). GPU work
# ends at GPU_DEADLINE (default 08:30 IST, next occurrence). GPU temp > 87 C before an eval -> pause 10 min (the
# trainers pause themselves). After every stage the queue commits and pushes its outputs (tapes gzipped), under a
# flock so it never races a manual commit.
#
# usage:  scripts/queue_split_seeds.sh             # run
#         DRY_RUN=1 scripts/queue_split_seeds.sh   # print the plan, run nothing
set -u
cd "$(dirname "$0")/.."
LABEL=qwen2.5-math-1.5b-instruct
RUN_PREFIX=results/train/split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed
E1A_RUN=results/train/e1a-sigma-only-free-seed0
E1B_RUN=results/train/e1b-sigma-only-bins-seed0
PREV_PID_FILE=${PREV_PID_FILE:-results/queue_split_e1/queue.pid}
LOG=${QUEUE_LOG:-results/queue_split_seeds}
GIT_LOCK=${GIT_LOCK:-.git/queue-commit.lock}
SAMPLES=${SAMPLES:-4}                       # D62: k sampled completions per problem (exploratory)
SPLIT_SAMPLED_MIN_S=${SPLIT_SAMPLED_MIN_S:-3600}  # D63: start the split-credit sampled eval only with this much time left
E1_SAMPLED_MIN_S=${E1_SAMPLED_MIN_S:-2100}        # D63: same for each E1 sampled eval (otherwise: tomorrow night)
LEFTOVERS=${LEFTOVERS:-1}                   # 0 skips stage 0 (the previous queue's leftovers are already committed)
EVAL_RESERVE_S=${EVAL_RESERVE_S:-1800}      # kept back after each training run for its step-287 eval
MIN_STAGE_S=${MIN_STAGE_S:-900}             # do not start a training stage with less than this left
DRY_RUN=${DRY_RUN:-0}
CO_AUTHOR="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
mkdir -p "$LOG"

ist() { TZ=Asia/Kolkata date -d "@$1" +%H:%M; }
log() { echo "[$(TZ=Asia/Kolkata date +%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }
iso() { TZ=Asia/Kolkata date -d "@$1" --iso-8601=seconds; }
left() { echo $((DEADLINE - $(date +%s))); }
cool() { [ "$DRY_RUN" = 1 ] && return; while t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "${t:-0}" -gt 87 ]; do
           log "GPU at ${t}C > 87C: pausing 600s"; sleep 600; done; log "GPU at ${t}C"; }

run_stage() {  # run_stage NAME TIMEOUT_S CMD...: child process, PID file, wait; returns the child's exit code
  local name=$1 limit=$2; shift 2
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

commit_stage() {  # commit_stage MESSAGE PATH...: gzip run tapes, add the paths that exist + queue logs, commit, push
  local msg=$1; shift
  if [ "$DRY_RUN" = 1 ]; then log "DRY commit: $msg"; return 0; fi
  local paths=() p
  for p in "$@"; do
    case "$p" in  # training tapes: the raw file is git-ignored, the .gz is committed (D42)
      *rollouts.jsonl.gz) [ -f "${p%.gz}" ] && gzip -kf -9 "${p%.gz}" ;;
    esac
    [ -e "$p" ] && paths+=("$p")
  done
  (
    flock -w 600 9 || { echo "git lock timeout"; exit 1; }
    git add -- "${paths[@]}" "$LOG"/*.log
    if git diff --cached --quiet; then echo "nothing new"; exit 0; fi
    git commit -q -m "$msg" -m "$CO_AUTHOR" || exit 1
    for i in 1 2 3; do git push -q origin main && { echo "pushed"; exit 0; }; sleep 30; done
    echo "PUSH FAILED"; exit 1
  ) 9>"$GIT_LOCK" >> "$LOG/git.log" 2>&1
  log "commit '$msg': $(tail -1 "$LOG/git.log")"
}

train_paths() {  # the committed files of a training run
  local r=$1
  echo "$r/rollouts.jsonl.gz $r/rollouts.meta.json $r/curriculum.jsonl $r/memory.jsonl $r/split_credit.jsonl \
        $r/summary.json $r/train_metrics.jsonl $r/events.jsonl"
}

# ---- wait for the previous queue (PID-based) ---------------------------------------------------------------
if [ "$DRY_RUN" != 1 ] && [ -f "$PREV_PID_FILE" ]; then
  PREV=$(cat "$PREV_PID_FILE")
  log "waiting for the previous queue (PID $PREV) to exit"
  PREV_LOG=$(dirname "$PREV_PID_FILE")/chain.log   # "queue done" also ends the wait, in case the PID is ever reused
  while kill -0 "$PREV" 2>/dev/null && ! grep -q "queue done" "$PREV_LOG"; do sleep 30; done
fi
echo $$ > "$LOG/queue.pid"
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-08:30}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "queue start; GPU deadline $(ist "$DEADLINE") IST; queue PID $$"

# ---- 0. leftovers of the previous queue ----------------------------------------------------------------------
[ "$LEFTOVERS" = 1 ] && commit_stage "E1b seed 0 + E1 family E: leftovers of the previous queue" $(train_paths "$E1B_RUN") \
  "$E1B_RUN/e1_eval" results/e1 results/queue_split_e1/chain.log results/queue_split_e1/nohup.log \
  results/queue_split_e1/e1b_train.log results/queue_split_e1/e1b_eval_final.log \
  results/queue_split_e1/e1b_eval_step150.log results/queue_split_e1/e1_compare.log

# ---- 1./2. split-credit seeds 1 and 2 ------------------------------------------------------------------------
seed_stage() {  # seed_stage SEED: train, eval, sanity, per-seed compare; returns 0 iff sane
  local s=$1 run="${RUN_PREFIX}$1"
  local stop=$((DEADLINE - EVAL_RESERVE_S)) rc
  if [ $((stop - $(date +%s))) -lt "$MIN_STAGE_S" ]; then log "not enough time for seed $s training"; return 99; fi
  run_stage "seed${s}_train" $((stop - $(date +%s) + 600)) uv run python -m physcal.train.grpo \
    --config "grpo_split_credit_seed$s" --max-steps 287 --stop-at "$(iso "$stop")"; rc=$?
  commit_stage "Split-credit seed $s: training to step 287 (exit $rc); tape gzipped" $(train_paths "$run")
  if [ "$DRY_RUN" != 1 ] && [ ! -d "$run/permanent_checkpoints/checkpoint-287" ]; then
    log "MISSING $run/permanent_checkpoints/checkpoint-287 (training exit $rc): no eval"
    uv run python scripts/queue_sanity.py "$run" "$rc" > "$LOG/seed${s}_sanity.log" 2>&1
    commit_stage "Split-credit seed $s: no step-287 checkpoint, sanity FAIL" "$run/sanity.json"
    return 1
  fi
  cool
  run_stage "seed${s}_eval287" "$(left)" uv run python -m physcal.train.compare --part rl --run "$run" \
    --label "$LABEL" --budget 512 --adapter permanent_checkpoints/checkpoint-287 --out-name rl287
  if [ "$DRY_RUN" = 1 ]; then log "DRY sanity + phase95_compare for seed $s"; return 0; fi
  uv run python scripts/queue_sanity.py "$run" "$rc" > "$LOG/seed${s}_sanity.log" 2>&1
  tee -a "$LOG/chain.log" < "$LOG/seed${s}_sanity.log"
  run_stage "seed${s}_compare" 1800 uv run python scripts/phase95_compare.py "$run"
  commit_stage "Split-credit seed $s: step-287 eval, sanity, per-seed family S/N" "$run/compare" "$run/sanity.json"
  grep -q '"sane": true' "$run/sanity.json"
}

if seed_stage 1; then
  seed_stage 2
else
  log "seed 1 not sane (or did not finish): seed 2 does NOT run (brief)"
fi

# ---- 3. exploratory sampled evals (D62); each starts only if it can finish before the deadline (D63) ----------
SC0="${RUN_PREFIX}0"
if [ "$(left)" -lt "$SPLIT_SAMPLED_MIN_S" ]; then
  log "skip split0_sampled: $(left)s left < ${SPLIT_SAMPLED_MIN_S}s; moves to tomorrow night (brief)"
else
  cool
  run_stage split0_sampled "$(left)" uv run python -m physcal.train.compare --part rl --run "$SC0" --label "$LABEL" \
    --budget 512 --adapter permanent_checkpoints/checkpoint-287 --out-name rl287_sampled --samples "$SAMPLES"
  commit_stage "Exploratory (D62): split-credit seed 0 @287, $SAMPLES sampled completions per problem" "$SC0/compare/rl287_sampled"
fi
for run in "$E1A_RUN" "$E1B_RUN"; do
  name=$(basename "$run" | cut -d- -f1)
  if [ "$DRY_RUN" != 1 ] && [ ! -d "$run/final_adapter" ]; then log "MISSING $run/final_adapter: no sampled eval"; continue; fi
  if [ "$(left)" -lt "$E1_SAMPLED_MIN_S" ]; then
    log "skip ${name}_sampled: $(left)s left < ${E1_SAMPLED_MIN_S}s; moves to tomorrow night (brief)"; continue
  fi
  cool
  run_stage "${name}_sampled" "$(left)" uv run python -m physcal.train.e1 eval --run "$run" --adapter final_adapter \
    --out-name final_sampled --samples "$SAMPLES"
  commit_stage "Exploratory (D62): $name final, $SAMPLES sampled sigma suffixes per prefix" "$run/e1_eval/final_sampled"
done
log "GPU work done"

# ---- 4. CPU analyses -----------------------------------------------------------------------------------------
if [ "$DRY_RUN" = 1 ] || { [ -f "${RUN_PREFIX}1/compare/rl287/rollouts.jsonl" ] && [ -f "${RUN_PREFIX}2/compare/rl287/rollouts.jsonl" ]; }; then
  run_stage seed_mean 3600 uv run python scripts/phase95_seed_mean.py "${RUN_PREFIX}0" "${RUN_PREFIX}1" "${RUN_PREFIX}2"
  commit_stage "Family S on the seed mean (D62)" results/split_credit_seed_mean
else
  log "not all 3 seeds have a step-287 eval: no seed-mean family S test"
fi
run_stage greedy_vs_sampled 1800 uv run python scripts/sigma_greedy_vs_sampled.py
commit_stage "Exploratory (D62): greedy vs sampled sigma report" results/sigma_greedy_vs_sampled
log "queue done"

#!/usr/bin/env bash
# The Phase 12 stage C pilot (DECISIONS.md D106): run by scripts/queue_stage_b.sh only if results/queue_stage_b/
# STAGE_C_READY exists and >= 3.5 h remain before GPU_DEADLINE (your rule). Stages:
#   1. GPU: GRPO, E1d-style sigma-only RL with verifier 1.3.0's score_sigma, 300 steps (clean stop EVAL_RESERVE_S
#      before the deadline)
#   2. GPU: greedy sigma suffixes on the 3,060 A6 / A6b / A6c prefixes: final adapter, SFT-only control, step 150
#   3. CPU: the analysis (C1-C6, Holm)
# Mechanics as every queue: child PIDs waited on, the in-stage temperature guard, the cool check, BLOCKED + go on,
# commit + push after every stage under the shared flock.
set -u
cd "$(dirname "$0")/.."
LOG=${QUEUE_LOG:-results/queue_stage_c}
GIT_LOCK=${GIT_LOCK:-.git/queue-commit.lock}
EVAL_RESERVE_S=${EVAL_RESERVE_S:-3000}   # three evals of ~12 min (3,060 suffixes at ~4.4/s) + commits
HARD_EXTRA_S=${HARD_EXTRA_S:-900}
EVAL_TIMEOUT_S=${EVAL_TIMEOUT_S:-2400}
DRY_RUN=${DRY_RUN:-0}
CO_AUTHOR="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
PY="uv run python"
OUT=results/stage_c
RUN=$OUT/run
CPU_ENV="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= nice -n 19"
mkdir -p "$LOG"

ist() { TZ=Asia/Kolkata date -d "@$1" +%H:%M; }
iso() { TZ=Asia/Kolkata date -d "@$1" --iso-8601=seconds; }
log() { echo "[$(TZ=Asia/Kolkata date +%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }
left() { echo $((DEADLINE - $(date +%s))); }
cool() { [ "$DRY_RUN" = 1 ] && return; while t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "${t:-0}" -gt 87 ]; do
           log "GPU at ${t}C > 87C: pausing 600s"; sleep 600; done; log "GPU at ${t}C, memory.used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; }

run_stage() {
  local name=$1 limit=$2; shift 2
  if [ "$DRY_RUN" = 1 ]; then log "DRY $name (timeout ${limit}s): $*"; return 0; fi
  if [ "$limit" -lt 300 ]; then log "skip $name: only ${limit}s left"; return 98; fi
  log "start $name (timeout ${limit}s): $*"
  timeout "$limit" "$@" > "$LOG/$name.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$LOG/$name.pid"
  [ "${GUARD:-0}" = 1 ] && { scripts/gpu_temp_guard.sh "$pid" "$LOG/chain.log" > /dev/null 2>&1 & }
  wait "$pid"; local rc=$?
  log "end $name: exit $rc"
  return $rc
}

commit_stage() {
  local msg=$1; shift
  if [ "$DRY_RUN" = 1 ]; then log "DRY commit: $msg"; return 0; fi
  local paths=() p
  for p in "$@"; do [ -e "$p" ] && paths+=("$p"); done
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

DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-09:00}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "stage C pilot queue start; GPU deadline $(ist "$DEADLINE") IST ($(( $(left) / 60 )) min left)"
export HF_HUB_OFFLINE=1
[ -f "$OUT/data/train_prefixes.jsonl" ] || log "BLOCKED: missing $OUT/data/train_prefixes.jsonl"

cool
stop=$((DEADLINE - EVAL_RESERVE_S))
GUARD=1 run_stage stage_c_train $((stop - $(date +%s) + HARD_EXTRA_S)) $PY -m physcal.train.stage_c train \
  --stop-at "$(iso "$stop")"; rc=$?
[ "$rc" != 0 ] && log "BLOCKED? stage_c_train exit $rc (see $LOG/stage_c_train.log)"
[ "$DRY_RUN" != 1 ] && [ -f "$RUN/rollouts.jsonl" ] && gzip -kfc "$RUN/rollouts.jsonl" > "$RUN/rollouts.jsonl.gz"
commit_stage "stage C pilot (D106): training (exit $rc)" "$RUN/summary.json" "$RUN/train_metrics.jsonl" \
  "$RUN/memory.jsonl" "$RUN/events.jsonl" "$RUN/rollouts.meta.json" "$RUN/rollouts.jsonl.gz"

for arm in final sft_control step150; do
  if [ "$arm" != sft_control ] && [ "$DRY_RUN" != 1 ] && [ ! -d "$RUN/$( [ "$arm" = final ] && echo final_adapter || echo permanent_checkpoints/checkpoint-150)" ]; then
    log "BLOCKED stage_c_eval_$arm: no adapter"; continue; fi
  cool
  lim=$EVAL_TIMEOUT_S; [ "$(left)" -lt "$lim" ] && lim=$(left)
  GUARD=1 run_stage "stage_c_eval_$arm" "$lim" $PY -m physcal.train.stage_c eval --arm "$arm"; rc=$?
  [ "$rc" != 0 ] && log "BLOCKED? stage_c_eval_$arm exit $rc"
  d=$OUT/$arm/seed0/eval
  [ "$DRY_RUN" != 1 ] && [ -f "$d/rollouts.jsonl" ] && gzip -kfc "$d/rollouts.jsonl" > "$d/rollouts.jsonl.gz"
  commit_stage "stage C pilot (D106): sigma suffixes on A6 / A6b / A6c, $arm (exit $rc)" "$d/rows.jsonl" \
    "$d/rollouts.jsonl.gz" "$d/rollouts.meta.json" "$OUT/$arm/seed0/summary.json"
done

run_stage stage_c_analysis 5400 $CPU_ENV $PY -m physcal.eval.stage_c_analysis analyze \
  || log "BLOCKED stage_c_analysis (see $LOG/stage_c_analysis.log)"
commit_stage "stage C pilot (D106): C1-C6, Holm" "$OUT/results.json" "$OUT/summary.md" "$OUT/final/results.json" \
  "$OUT/final/summary.md" "$OUT/sft_control/results.json" "$OUT/sft_control/summary.md" "$OUT/step150/results.json" \
  "$OUT/step150/summary.md"
log "stage C pilot queue done"

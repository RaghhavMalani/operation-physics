#!/usr/bin/env bash
# Night GPU queue, 2026-10-21/22, stage (a) (DECISIONS.md D102): E1d's sigma on the X2 pairs.
#   1. GPU: E1d's final adapter writes a greedy sigma suffix on X2's 4,116 kept answers (4-bit base, unmerged)
#   2. GPU: the SFT model's own greedy sigma suffix on the same prefixes (D102's secondary control)
# The CPU analysis (physcal.eval.e1d_x2_analysis) runs after this queue, from the night's chain. Every stage is a child
# whose PID goes to $LOG/<stage>.pid and is waited on with `wait` (never pgrep -f); every stage gets the in-stage
# temperature guard (scripts/gpu_temp_guard.sh, D96) and the cool check before it. A failed stage is logged BLOCKED and
# the next goes on (they are independent). After every stage the queue commits and pushes under the shared flock.
#
# usage:  nohup scripts/queue_e1d_x2.sh > results/queue_e1d_x2/nohup.log 2>&1 &
#         DRY_RUN=1 scripts/queue_e1d_x2.sh
set -u
cd "$(dirname "$0")/.."
. scripts/disk_guard.sh   # the free-space guard (D111): disk_guard before every stage
LOG=${QUEUE_LOG:-results/queue_e1d_x2}
GIT_LOCK=${GIT_LOCK:-.git/queue-commit.lock}
STAGE_TIMEOUT_S=${STAGE_TIMEOUT_S:-3600}   # ~18 min expected (X3 student (a) on the same 4,116 prefixes, D88)
DRY_RUN=${DRY_RUN:-0}
CO_AUTHOR="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
PY="uv run python"
OUT=results/e1d_x2
mkdir -p "$LOG"

ist() { TZ=Asia/Kolkata date -d "@$1" +%H:%M; }
log() { echo "[$(TZ=Asia/Kolkata date +%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }
left() { echo $((DEADLINE - $(date +%s))); }
cool() { [ "$DRY_RUN" = 1 ] && return; while t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "${t:-0}" -gt 87 ]; do
           log "GPU at ${t}C > 87C: pausing 600s"; sleep 600; done; log "GPU at ${t}C, memory.used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; }

run_stage() {  # run_stage NAME TIMEOUT_S CMD...: child process, PID file, wait; returns the child's exit code
  local name=$1 limit=$2; shift 2
  disk_guard "$name" || exit 75   # < MIN_FREE_GB on MLStorage (or unreadable): stop the queue (D111)
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

commit_stage() {  # commit_stage MESSAGE PATH...: add the paths that exist + queue logs, commit, push (under a flock)
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

echo $$ > "$LOG/queue.pid"
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-09:00}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "queue start; GPU deadline $(ist "$DEADLINE") IST ($(( $(left) / 60 )) min left); queue PID $$"
export HF_HUB_OFFLINE=1   # every model is local (D78)

for f in results/train/e1d-sigma-only-free-seed0/final_adapter/adapter_model.safetensors \
         results/sft/qwen2.5-math-1.5b-instruct/merged/config.json results/x2/selected.json; do
  [ -e "$f" ] || log "BLOCKED: missing $f"
done

for which in e1d sft; do
  name=e1d_x2_$which
  [ "$which" = e1d ] && dir=$OUT/e1d_final || dir=$OUT/sft_suffix
  cool
  lim=$STAGE_TIMEOUT_S; [ "$(left)" -lt "$lim" ] && lim=$(left)
  GUARD=1 run_stage "$name" "$lim" $PY -m physcal.eval.e1d_x2 suffix --which "$which"; rc=$?
  [ "$rc" != 0 ] && log "BLOCKED $name: exit $rc (see $LOG/$name.log)"
  [ "$DRY_RUN" != 1 ] && [ -f "$dir/rollouts.jsonl" ] && gzip -kfc "$dir/rollouts.jsonl" > "$dir/rollouts.jsonl.gz"
  commit_stage "$name (D102): greedy sigma suffixes on the 4,116 X2 answers (exit $rc)" "$dir/rows.jsonl" \
    "$dir/rollouts.meta.json" "$dir/rollouts.jsonl.gz" "$dir/excluded.json"
done
log "queue done"

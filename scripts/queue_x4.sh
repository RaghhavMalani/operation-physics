#!/usr/bin/env bash
# Night GPU queue, 2026-10-20/21 (DECISIONS.md D90, amended by D91; X4 = D89 + D91). The arms, in D91's order:
#   1. GPU: X4 sft (primary, recompute): the SFT model checks each of its 4,116 X2 answers
#   2. GPU: X4 direct (D91): the same conversations read right after the generation prompt (no recomputation)
#   3. GPU: X4 perturbed (D91 control): each answer's final mu x 1.25 (sigma, unit, steps unchanged)
#   4. GPU: X4 shuffled (control): the same model, each answer under another problem's text
#   5. GPU: X4 random_init (control): X2's random-init model, the same conversations
#   6. CPU: the X4 analysis (D89 + D91)
# X4_ARMS picks and orders the arms (default: all five). WAIT_PID: wait for that process (the queue running the earlier
# arms) to exit before the first stage. Every GPU stage gets the temperature guard (scripts/gpu_temp_guard.sh: GPU
# > 87 C inside a stage -> its process group is stopped for 10 min). At the end, an incomplete sft (the primary arm)
# is resumed if time is left.
# Before each GPU stage the allocator cap is measured: total VRAM - nvidia-smi memory.used (nothing of ours running)
# - 0.5 GiB, clipped to [4.0, 7.5] (D86/D87, the WSL-spill rule). Each GPU stage gets a soft limit (--minutes: it
# stops cleanly between batches) that keeps time back for the stages after it, and a hard timeout 10 min later. Rows are
# appended after every batch, so a stage that stops early leaves a prefix of the kept order (D89 analyzes the problems
# each arm completed), and a rerun of the same command resumes it.
# Every stage is a child whose PID goes to $LOG/<stage>.pid and is waited on with `wait` (never pgrep -f). GPU work
# ends at GPU_DEADLINE (default 11:30 IST, next occurrence; D91). GPU temp > 87 C before a stage -> pause 10 min. After every
# stage the queue commits and pushes its outputs under a flock. A failed stage is logged BLOCKED; the next stage goes on
# (the arms do not depend on each other).
#
# usage:  nohup scripts/queue_x4.sh > results/queue_x4/nohup.log 2>&1 &
#         X4_ARMS="direct perturbed shuffled random_init" WAIT_PID=<pid> nohup scripts/queue_x4.sh > ... &   (D91)
#         DRY_RUN=1 scripts/queue_x4.sh     # print the plan, run nothing
set -u
cd "$(dirname "$0")/.."
LOG=${QUEUE_LOG:-results/queue_x4}
GIT_LOCK=${GIT_LOCK:-.git/queue-commit.lock}
X4_ARMS=${X4_ARMS:-sft direct perturbed shuffled random_init}  # D91's order
LATER_RESERVE_S=${LATER_RESERVE_S:-1800}         # kept back per later GPU arm (D91: earlier arms have priority)
END_RESERVE_S=${END_RESERVE_S:-900}              # kept back from the last arm (the last commit, slack)
WAIT_PID=${WAIT_PID:-}
HARD_EXTRA_S=600                                 # hard timeout = soft limit + this (a batch in flight, the save)
DRY_RUN=${DRY_RUN:-0}
CO_AUTHOR="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
PY="uv run python"
mkdir -p "$LOG"

ist() { TZ=Asia/Kolkata date -d "@$1" +%H:%M; }
log() { echo "[$(TZ=Asia/Kolkata date +%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }
left() { echo $((DEADLINE - $(date +%s))); }
cool() { [ "$DRY_RUN" = 1 ] && return; while t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "${t:-0}" -gt 87 ]; do
           log "GPU at ${t}C > 87C: pausing 600s"; sleep 600; done; log "GPU at ${t}C"; }

run_stage() {  # run_stage NAME TIMEOUT_S CMD...: child process, PID file, wait; returns the child's exit code
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

rows() { [ -f "results/x4/$1/rows.jsonl" ] && wc -l < "results/x4/$1/rows.jsonl" || echo 0; }

echo $$ > "$LOG/queue.pid"
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-11:30}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "queue start; GPU deadline $(ist "$DEADLINE") IST; queue PID $$; arms: $X4_ARMS"
if [ -n "$WAIT_PID" ] && [ "$DRY_RUN" != 1 ]; then  # PID-based: the earlier queue, checked by its command line too
  cmd=$(tr '\0' ' ' < "/proc/$WAIT_PID/cmdline" 2>/dev/null)
  log "waiting for PID $WAIT_PID to exit: ${cmd:-already gone}"
  while kill -0 "$WAIT_PID" 2>/dev/null && [ "$(tr '\0' ' ' < "/proc/$WAIT_PID/cmdline" 2>/dev/null)" = "$cmd" ]; do
    sleep 30; done
  log "PID $WAIT_PID has exited"
fi
export HF_HUB_OFFLINE=1   # every model is local; never resolve anything online (D78)

# ---- 0. inputs that must exist (made on CPU, 2026-10-20 and before) ----------------------------------------------------
for f in results/x4/x2_reference.json results/x2/selected.json results/x2/random_init_model/config.json \
         results/sft/qwen2.5-math-1.5b-instruct/merged/config.json; do
  [ -e "$f" ] || log "BLOCKED: missing $f"
done
for a in $X4_ARMS; do
  [ "$(rows $a)" != 0 ] && log "NOTE: results/x4/$a already has $(rows $a) rows: that arm resumes"
done

# ---- 1-5. GPU: the arms, in D91's order ---------------------------------------------------------------------------------
arm_stage() {  # arm_stage ARM RESERVE_S: soft limit = time left - reserve; commit the rows whatever happened
  local arm=$1 reserve=$2 cap soft rc
  cool
  cap=$($PY -m physcal.eval.x4_verify cap 2>/dev/null | tail -1); [ -z "$cap" ] && cap=5.75
  [ "$DRY_RUN" = 1 ] && cap=${cap:-5.75}
  soft=$(( $(left) - reserve ))
  log "X4 $arm: cap ${cap} GiB, soft limit $((soft / 60)) min (reserve ${reserve}s), $(rows "$arm") rows so far"
  if [ "$soft" -lt 600 ] && [ "$DRY_RUN" != 1 ]; then log "skip x4_$arm: only ${soft}s before its reserve"; return; fi
  GUARD=1 run_stage "x4_$arm" $((soft + HARD_EXTRA_S)) $PY -m physcal.eval.x4_verify run --arm "$arm" \
    --max-reserved-gib "$cap" --minutes $((soft / 60)); rc=$?
  [ "$rc" != 0 ] && log "BLOCKED? x4_$arm exit $rc (see $LOG/x4_$arm.log); its rows so far are kept: $(rows "$arm")"
  [ "$DRY_RUN" != 1 ] && log "X4 $arm: $(rows "$arm") of 4116 rows"
  commit_stage "X4 (D89/D91): $arm arm, $(rows "$arm") of 4,116 X2 answers verified (exit $rc, cap ${cap} GiB)" \
    "results/x4/$arm"
}
read -r -a arms <<< "$X4_ARMS"
for i in "${!arms[@]}"; do
  arm_stage "${arms[$i]}" $(( (${#arms[@]} - 1 - i) * LATER_RESERVE_S + END_RESERVE_S ))
done
if [[ " $X4_ARMS " != *" sft "* ]] && [ "$DRY_RUN" != 1 ] && [ "$(rows sft)" -lt 4116 ]; then
  log "sft (primary) has $(rows sft) of 4116 rows: resuming it with the time left"
  arm_stage sft "$END_RESERVE_S"
fi

# ---- 4. CPU: the X4 analysis (D89) ---------------------------------------------------------------------------------------
run_stage x4_analysis 3600 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" \
  nice -n 10 $PY -m physcal.eval.x4_analysis analyze || log "BLOCKED x4_analysis (see $LOG/x4_analysis.log)"
commit_stage "X4 (D89/D91) results: self-verification pairwise accuracy on the X2 pairs, direct vs recompute, controls" \
  results/x4/results.json results/x4/summary.md
log "queue done"

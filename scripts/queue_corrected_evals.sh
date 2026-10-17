#!/usr/bin/env bash
# Night queue 2026-10-18 (DECISIONS.md D67 / D68): corrected evals, every RL adapter unmerged on its 4-bit training
# base (the default loader since D68), the SFT-only control on the same base.
#   tier 1, greedy:  SFT-only (frozen sets + E1 prefixes); split credit seeds 0 (rerun), 1, 2 @287; uniform @287;
#                    E1a final (rerun) + step 150; E1b final + step 150;  then the CPU analyses (if the script exists)
#   tier 2:          the FINDINGS #5 and #6 adapters; sampled k = 4 for split seed 0 and E1a (reruns of D65's);
#                    the GPU check of --resume (D66);  then the CPU analyses again
# Every stage is a child whose PID goes to $LOG/<stage>.pid and is waited on (never pgrep -f). A stage starts only if
# its estimated duration x 1.3 fits before GPU_DEADLINE (08:30 IST); its timeout is the time left. GPU temp > 87 C
# before a stage -> pause 10 min. After every stage the queue commits and pushes its outputs, under a flock.
#
# usage:  scripts/queue_corrected_evals.sh             # run
#         DRY_RUN=1 scripts/queue_corrected_evals.sh   # print the plan, run nothing
set -u
cd "$(dirname "$0")/.."
LABEL=qwen2.5-math-1.5b-instruct
T=results/train
SPLIT=$T/split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed
UNI=$T/overnight2-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512
N5=$T/overnight-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-512
E1A=$T/e1a-sigma-only-free-seed0
E1B=$T/e1b-sigma-only-bins-seed0
LOG=${QUEUE_LOG:-results/queue_corrected}
GIT_LOCK=${GIT_LOCK:-.git/queue-commit.lock}
GPU_DEADLINE=${GPU_DEADLINE:-08:30}
SAMPLES=${SAMPLES:-4}
SAFETY=${SAFETY:-1.3}                       # a stage starts only with estimate x SAFETY left
TEMP_LIMIT_C=87
PAUSE_S=600
DRY_RUN=${DRY_RUN:-0}
CO_AUTHOR="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
mkdir -p "$LOG"
echo $$ > "$LOG/queue.pid"
DEADLINE=$(TZ=Asia/Kolkata date -d "$GPU_DEADLINE" +%s)
[ "$DEADLINE" -le "$(date +%s)" ] && DEADLINE=$(TZ=Asia/Kolkata date -d "tomorrow $GPU_DEADLINE" +%s)

log() { echo "[$(TZ=Asia/Kolkata date +%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }
left() { echo $((DEADLINE - $(date +%s))); }
cool() { [ "$DRY_RUN" = 1 ] && return; while t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "${t:-0}" -gt "$TEMP_LIMIT_C" ]; do
           log "GPU at ${t}C > ${TEMP_LIMIT_C}C: pausing ${PAUSE_S}s"; sleep "$PAUSE_S"; done; log "GPU at ${t}C"; }

run_gpu() {  # run_gpu NAME ESTIMATE_S CMD...: skip if it does not fit before the deadline; else child + PID + wait
  local name=$1 est=$2; shift 2
  local need; need=$(python3 -c "print(int($est * $SAFETY))")
  if [ "$DRY_RUN" = 1 ]; then log "DRY $name (est ${est}s): $*"; return 0; fi
  cool
  local l; l=$(left)
  if [ "$l" -lt "$need" ]; then log "SKIP $name: ${l}s left before $GPU_DEADLINE IST, needs ${need}s"; return 98; fi
  log "start $name (est ${est}s, timeout ${l}s): $*"
  timeout "$l" "$@" > "$LOG/$name.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$LOG/$name.pid"
  wait "$pid"; local rc=$?
  log "end $name: exit $rc"
  return $rc
}

run_cpu() {  # run_cpu NAME CMD...
  local name=$1; shift
  if [ "$DRY_RUN" = 1 ]; then log "DRY $name: $*"; return 0; fi
  log "start $name: $*"
  "$@" > "$LOG/$name.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$LOG/$name.pid"
  wait "$pid"; local rc=$?
  log "end $name: exit $rc"
  return $rc
}

commit_stage() {  # commit_stage MESSAGE PATH...: add the paths that exist + queue logs, commit, push
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

CMP="uv run python -m physcal.train.compare --label $LABEL --budget 512"
E1="uv run python -m physcal.train.e1 eval"
analyses() {  # the corrected family S / N / E analyses (CPU), if the script exists yet
  if [ -x scripts/corrected_analyses.sh ]; then
    run_cpu "analyses_$1" scripts/corrected_analyses.sh
    commit_stage "Corrected analyses ($1): families S, N, E on the 4-bit unmerged evals (D67)" results/corrected
  else
    log "skip analyses_$1: scripts/corrected_analyses.sh not there yet"
  fi
}

log "queue start (pid $$), GPU deadline $(TZ=Asia/Kolkata date -d "@$DEADLINE" '+%F %H:%M') IST"

# ------------------------------------------------------------------ tier 1, greedy
run_gpu sft_4bit 620 $CMP --part sft --run $UNI --out-name sft_4bit
commit_stage "Corrected eval (D67): SFT-only on the 4-bit base, frozen 1.1.0 sets, greedy" $UNI/compare/sft_4bit
run_gpu e1_sft_4bit 300 $E1 --sft
commit_stage "Corrected eval (D67): SFT-only sigma suffix on the E1 eval prefixes, 4-bit base" results/e1/sft_4bit
run_gpu split0_rerun 600 $CMP --part rl --run ${SPLIT}0 --adapter permanent_checkpoints/checkpoint-287 --out-name rl287_4bit_rerun
commit_stage "Corrected eval (D67): split credit seed 0 @287, rerun of D65 (determinism check)" ${SPLIT}0/compare/rl287_4bit_rerun
for s in 1 2; do
  run_gpu split${s} 600 $CMP --part rl --run ${SPLIT}${s} --adapter permanent_checkpoints/checkpoint-287 --out-name rl287_4bit
  commit_stage "Corrected eval (D67): split credit seed $s @287, 4-bit unmerged, greedy" ${SPLIT}${s}/compare/rl287_4bit
done
run_gpu uniform287 600 $CMP --part rl --run $UNI --adapter permanent_checkpoints/checkpoint-287 --out-name rl287_4bit
commit_stage "Corrected eval (D67): uniform credit @287, 4-bit unmerged, greedy" $UNI/compare/rl287_4bit
run_gpu e1a_final_rerun 300 $E1 --run $E1A --adapter final_adapter --out-name final_4bit_rerun
commit_stage "Corrected eval (D67): E1a final, rerun of D65 (determinism check)" $E1A/e1_eval/final_4bit_rerun
run_gpu e1a_step150 300 $E1 --run $E1A --adapter permanent_checkpoints/checkpoint-150 --out-name step150_4bit
commit_stage "Corrected eval (D67): E1a step 150, 4-bit unmerged" $E1A/e1_eval/step150_4bit
run_gpu e1b_final 300 $E1 --run $E1B --adapter final_adapter --out-name final_4bit
commit_stage "Corrected eval (D67): E1b final, 4-bit unmerged" $E1B/e1_eval/final_4bit
run_gpu e1b_step150 300 $E1 --run $E1B --adapter permanent_checkpoints/checkpoint-150 --out-name step150_4bit
commit_stage "Corrected eval (D67): E1b step 150, 4-bit unmerged" $E1B/e1_eval/step150_4bit
analyses tier1

# ------------------------------------------------------------------ tier 2, if time remains
run_gpu n5_final 600 $CMP --part rl --run $N5 --adapter final_adapter --out-name rl_4bit
commit_stage "Corrected eval (D67): FINDINGS #5 adapter (overnight run final, step 287), 4-bit unmerged" $N5/compare/rl_4bit
run_gpu n6_final 600 $CMP --part rl --run $UNI --adapter final_adapter --out-name rlfinal_4bit
commit_stage "Corrected eval (D67): FINDINGS #6 adapter (overnight2 final, step 759), 4-bit unmerged" $UNI/compare/rlfinal_4bit
run_gpu split0_sampled_rerun 2100 $CMP --part rl --run ${SPLIT}0 --adapter permanent_checkpoints/checkpoint-287 \
  --out-name rl287_4bit_sampled_rerun --samples "$SAMPLES"
commit_stage "Corrected eval (D67): split credit seed 0 @287, k = $SAMPLES sampled, rerun of D65" ${SPLIT}0/compare/rl287_4bit_sampled_rerun
run_gpu e1a_sampled_rerun 1200 $E1 --run $E1A --adapter final_adapter --out-name final_4bit_sampled_rerun --samples "$SAMPLES"
commit_stage "Corrected eval (D67): E1a final, k = $SAMPLES sampled, rerun of D65" $E1A/e1_eval/final_4bit_sampled_rerun
run_gpu resume_check 900 uv run python scripts/resume_gpu_check.py
commit_stage "D66 GPU resume check (resume-smoke A/B)" results/resume_gpu_check $T/resume-smoke-A $T/resume-smoke-B
analyses final
log "queue done"

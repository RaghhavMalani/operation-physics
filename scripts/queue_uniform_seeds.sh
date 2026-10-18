#!/usr/bin/env bash
# Night GPU queue, 2026-10-18/19 (DECISIONS.md D69, D71/R2-6, D72; D75 records the plan). Prepared, not launched.
#   1. GPU: 4-bit SFT-only completions of the calibration split (D69): the 400 of D40, then the 1,600 of probe_sft
#   2. CPU, in the background while the GPU trains: re-fit the global post-hoc scale and A5 on them (D69)
#   3. GPU: uniform-credit seed 1 to step 287 (records the R2-1 fingerprint) -> corrected eval (4-bit, unmerged,
#      greedy, frozen 1.1.0 sets; runs the fingerprint check) -> sanity
#   4. GPU: uniform-credit seed 2, the same, if a full run still fits the deadline. D76 (your instruction at launch):
#      it no longer waits for seed 1's sanity; a failed stage is logged BLOCKED and independent stages continue
#   5. CPU: the 3-vs-3 robustness analysis (D71: R2-4 / R2-6), with whatever uniform seeds reached step 287
#   6. GPU, only if >= SMOKE_MIN_S remain: the D70 smoke test (SFT adapter loaded and trainable on the 4-bit base)
# Every stage is a child whose PID goes to $LOG/<stage>.pid and is waited on with `wait` (never pgrep -f). GPU work
# ends at GPU_DEADLINE (default 08:30 IST, next occurrence). GPU temp > 87 C before a stage -> pause 10 min (the
# trainer pauses itself too). After every stage the queue commits and pushes its outputs (tapes gzipped), under a
# flock so it never races a manual commit.
#
# usage:  nohup scripts/queue_uniform_seeds.sh > results/queue_uniform_seeds/nohup.log 2>&1 &
#         DRY_RUN=1 scripts/queue_uniform_seeds.sh   # print the plan, run nothing
set -u
cd "$(dirname "$0")/.."
LABEL=qwen2.5-math-1.5b-instruct
RUN_PREFIX=results/train/overnight2-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed
CALIB=results/posthoc_calibration_4bit
LOG=${QUEUE_LOG:-results/queue_uniform_seeds}
GIT_LOCK=${GIT_LOCK:-.git/queue-commit.lock}
EVAL_RESERVE_S=${EVAL_RESERVE_S:-1800}      # kept back after each training run for its step-287 eval + sanity
MIN_TRAIN_S=${MIN_TRAIN_S:-10800}           # start a seed only if this much training time is left (287 steps ~ 2.8-3.2 h)
FINGERPRINT_SLACK_S=${FINGERPRINT_SLACK_S:-1200}  # after --stop-at: final save + the R2-1 fingerprint (~5 min)
CALIB_TIMEOUT_S=${CALIB_TIMEOUT_S:-5400}    # per calibration tape (400: ~8 min, 1,600: ~30 min at 4-bit speed)
SMOKE_MIN_S=${SMOKE_MIN_S:-1500}           # D76: the D70 smoke test (~5-10 min) starts only with this much left
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

train_paths() {  # the committed files of a training run (final_adapter/ is git-ignored: its fingerprint is copied up)
  local r=$1
  echo "$r/rollouts.jsonl.gz $r/rollouts.meta.json $r/curriculum.jsonl $r/memory.jsonl \
        $r/summary.json $r/train_metrics.jsonl $r/events.jsonl $r/physcal_fingerprint.json"
}

echo $$ > "$LOG/queue.pid"
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-08:30}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "queue start; GPU deadline $(ist "$DEADLINE") IST; queue PID $$"

# ---- 1. GPU: 4-bit SFT calibration completions (D69) ---------------------------------------------------------
CALIB_OK=1
for part in set ext; do
  cool
  lim=$(left); [ "$lim" -gt "$CALIB_TIMEOUT_S" ] && lim=$CALIB_TIMEOUT_S
  run_stage "calib_$part" "$lim" uv run python -m physcal.eval.posthoc generate-4bit --part "$part" \
    || { CALIB_OK=0; log "BLOCKED calib_$part (see $LOG/calib_$part.log); the D69 re-fits are skipped, the seeds go on"; }
  commit_stage "D69: 4-bit SFT-only completions, calibration $part (corrected loader, greedy)" "$CALIB/sft_calibration_$part"
done

# ---- 2. CPU (background): both re-fits (D69) ------------------------------------------------------------------
refit() {
  if [ "$CALIB_OK" != 1 ]; then log "skip the D69 re-fits: a calibration stage failed"; return; fi
  if [ "$DRY_RUN" = 1 ]; then log "DRY refit: posthoc refit-4bit; nested_baselines fit --config nested_baselines_4bit"; return; fi
  log "start D69 re-fits (CPU, nice, 4 threads)"
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" nice -n 10 \
    uv run python -m physcal.eval.posthoc refit-4bit > "$LOG/refit_posthoc.log" 2>&1; local a=$?
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" nice -n 10 \
    uv run python -m physcal.eval.nested_baselines fit --config nested_baselines_4bit > "$LOG/refit_a5.log" 2>&1; local b=$?
  log "end D69 re-fits: posthoc exit $a, A5 exit $b"
  commit_stage "D69: global post-hoc scale and A5 re-fitted on 4-bit SFT calibration completions" \
    "$CALIB/posthoc_fit.json" "$CALIB/posthoc_fit_all2000.json" "$CALIB"/sft_eval_scaled_*.jsonl results/a5_baselines_4bit
}
refit &
REFIT_PID=$!
echo "$REFIT_PID" > "$LOG/refit.pid"

# ---- 3./4. uniform-credit seeds 1 and 2 (R2-6) -----------------------------------------------------------------
seed_stage() {  # seed_stage SEED: train, eval, sanity; returns 0 iff sane
  local s=$1 run="${RUN_PREFIX}$1"
  local stop=$((DEADLINE - EVAL_RESERVE_S)) rc
  if [ $((stop - $(date +%s))) -lt "$MIN_TRAIN_S" ]; then
    log "not enough time for uniform seed $s: $((stop - $(date +%s)))s < MIN_TRAIN_S=${MIN_TRAIN_S}s (D71: report 3 vs fewer)"
    return 99
  fi
  cool
  run_stage "uniform${s}_train" $((stop - $(date +%s) + FINGERPRINT_SLACK_S)) uv run python -m physcal.train.grpo \
    --config "grpo_overnight2_seed$s" --max-steps 287 --stop-at "$(iso "$stop")"; rc=$?
  [ "$rc" != 0 ] && log "BLOCKED? uniform seed $s training exit $rc (see $LOG/uniform${s}_train.log)"
  [ "$DRY_RUN" != 1 ] && [ -f "$run/final_adapter/physcal_fingerprint.json" ] && cp "$run/final_adapter/physcal_fingerprint.json" "$run/"
  commit_stage "Uniform-credit seed $s (R2-6): training to step 287 (exit $rc); tape gzipped; R2-1 fingerprint" $(train_paths "$run")
  if [ "$DRY_RUN" != 1 ] && [ ! -d "$run/permanent_checkpoints/checkpoint-287" ]; then
    log "MISSING $run/permanent_checkpoints/checkpoint-287 (training exit $rc): no eval"
    uv run python scripts/queue_sanity.py "$run" "$rc" rl287_4bit > "$LOG/uniform${s}_sanity.log" 2>&1
    commit_stage "Uniform-credit seed $s: no step-287 checkpoint, sanity FAIL" "$run/sanity.json"
    return 1
  fi
  cool
  run_stage "uniform${s}_eval287" "$(left)" uv run python -m physcal.train.compare --part rl --run "$run" \
    --label "$LABEL" --budget 512 --adapter permanent_checkpoints/checkpoint-287 --out-name rl287_4bit \
    || log "BLOCKED uniform seed $s corrected eval (see $LOG/uniform${s}_eval287.log)"
  if [ "$DRY_RUN" = 1 ]; then log "DRY sanity for uniform seed $s"; return 0; fi
  uv run python scripts/queue_sanity.py "$run" "$rc" rl287_4bit > "$LOG/uniform${s}_sanity.log" 2>&1
  tee -a "$LOG/chain.log" < "$LOG/uniform${s}_sanity.log"
  uv run python -c "import json,sys; c=json.load(open(sys.argv[1]))['loader_check'].get('fingerprint',{}); print('R2-1 fingerprint check:', c.get('status'), json.dumps({k: v.get('ratio', v.get('abs_dlog_median', v.get('pass'))) for k, v in c.get('criteria', {}).items()}))" \
    "$run/compare/rl287_4bit/metrics.json" 2>&1 | tee -a "$LOG/chain.log"
  commit_stage "Uniform-credit seed $s: corrected step-287 eval (4-bit, unmerged, greedy; R2-1 check), sanity" \
    "$run/compare/rl287_4bit" "$run/sanity.json"
  grep -q '"sane": true' "$run/sanity.json"
}

seed_stage 1; S1=$?
[ "$S1" != 0 ] && log "uniform seed 1 not sane (code $S1): BLOCKED; seed 2 still runs (D76: it does not depend on seed 1)"
seed_stage 2; S2=$?
log "GPU work done: seed 1 code $S1, seed 2 code $S2"

# ---- 5. CPU: the 3-vs-3 robustness analysis (D71) ----------------------------------------------------------------
if [ "$DRY_RUN" != 1 ]; then log "waiting for the D69 re-fits (PID $REFIT_PID)"; fi
wait "$REFIT_PID"
run_stage robustness_3v3 3600 env CUDA_VISIBLE_DEVICES="" nice -n 10 uv run python scripts/robustness_3v3.py
commit_stage "R2-4 / R2-6 (D71): split vs uniform credit, 3 vs N seeds, robustness analysis (corrected evals)" \
  results/corrected/robustness_3v3

# ---- 6. GPU, if time remains: the D70 smoke test (D76) -------------------------------------------------------------
if [ "$(left)" -ge "$SMOKE_MIN_S" ] || [ "$DRY_RUN" = 1 ]; then
  cool
  lim=$(( $(left) - 120 )); [ "$lim" -gt 1800 ] && lim=1800
  run_stage d70_smoke "$lim" env HF_HUB_OFFLINE=1 uv run python scripts/sft_continue_smoke.py \
    || log "BLOCKED? D70 smoke test exit nonzero (see $LOG/d70_smoke.log)"
  commit_stage "D70 smoke test (D76): SFT adapter loaded and trained by GRPO on the 4-bit base" \
    results/sft_continue_smoke results/train/sft-continue-smoke/summary.json \
    results/train/sft-continue-smoke/train_metrics.jsonl results/train/sft-continue-smoke/events.jsonl
else
  log "skip the D70 smoke test: $(left)s left < SMOKE_MIN_S=${SMOKE_MIN_S}s"
fi
log "queue done"

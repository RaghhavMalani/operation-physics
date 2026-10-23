#!/usr/bin/env bash
# Night GPU queue, 2026-10-19/20 (DECISIONS.md D81; X2 = D74, X3 = D79). Prepared, NOT launched.
#   1. GPU: X2 batch-size probe: the largest batch (sequences per call) that fits, worst case (D81)
#   2. GPU: X2 sampling, k = 8, corrected SFT model, all 1,200 candidates in 6 chunks of 200 (the frozen order). A
#      chunk starts only if its worst-case time (from the probe, x CHUNK_SAFETY) still leaves X2_DOWNSTREAM_S before
#      the deadline; otherwise the sampled prefix is used (D74). Every chunk is its own committed tape.
#   3. CPU: X2 selection (the kept, mixed problems)
#   4. GPU: X2 method 2: split credit seed 0 @287 writes a greedy sigma suffix on every kept answer
#   5. GPU: X2 hidden states at the first "±" token: SFT (4-bit) and random init, fit set and X2 answers
#   6. CPU, in the background: the X2 analysis (D74), while the GPU goes on
#   7. GPU, only if >= X3_MIN_S remain after step 5 (your rule: 2 h): the X3 pilot (D79): student (a) with a wall-clock
#      cap so both fit, its evals (frozen sets, X2 answers), student (b) at (a)'s step count, its evals, the analysis
#   8. GPU, only if >= PIN_MIN_S remain: the D78 pinned-base check (~2 min)
# Every stage is a child whose PID goes to $LOG/<stage>.pid and is waited on with `wait` (never pgrep -f). GPU work
# ends at GPU_DEADLINE (default 08:30 IST, next occurrence). GPU temp > 87 C before a stage -> pause 10 min. After every
# stage the queue commits and pushes its outputs under a flock. A failed stage is logged BLOCKED; stages that do not
# depend on it go on.
#
# usage:  nohup scripts/queue_x2_x3.sh > results/queue_x2_x3/nohup.log 2>&1 &
#         DRY_RUN=1 scripts/queue_x2_x3.sh     # print the plan, run nothing
set -u
cd "$(dirname "$0")/.."
. scripts/disk_guard.sh   # the free-space guard (D111): disk_guard before every stage
LOG=${QUEUE_LOG:-results/queue_x2_x3}
GIT_LOCK=${GIT_LOCK:-.git/queue-commit.lock}
N_CANDIDATES=1200
CHUNK=200                                        # = configs/x2.yaml sampling.chunk_size
CHUNK_SAFETY=${CHUNK_SAFETY:-1.3}                # x the probe's worst-case chunk time
FALLBACK_BATCH=16                                # D74's batch size, if the probe fails
FALLBACK_CHUNK_S=${FALLBACK_CHUNK_S:-2400}       # D62/D65 sampled speed at batch 16: ~1.2 s / completion x 1,600
X2_DOWNSTREAM_S=${X2_DOWNSTREAM_S:-3600}         # kept back after sampling: suffixes (~15 min) + 4 extractions (~30 min)
X3_MIN_S=${X3_MIN_S:-7200}                       # your rule: the X3 pilot only if >= 2 h remain
X3_EVAL_RESERVE_S=${X3_EVAL_RESERVE_S:-3000}     # 4 student evals + slack, kept back from the two trainings
X3_SLACK_S=${X3_SLACK_S:-900}                    # per student: load, save, the R2-1 fingerprint
PIN_MIN_S=${PIN_MIN_S:-600}
DRY_RUN=${DRY_RUN:-0}
CO_AUTHOR="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
PY="uv run python"
mkdir -p "$LOG"

ist() { TZ=Asia/Kolkata date -d "@$1" +%H:%M; }
log() { echo "[$(TZ=Asia/Kolkata date +%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }
left() { echo $((DEADLINE - $(date +%s))); }
minimum() { [ "$1" -lt "$2" ] && echo "$1" || echo "$2"; }
cool() { [ "$DRY_RUN" = 1 ] && return; while t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "${t:-0}" -gt 87 ]; do
           log "GPU at ${t}C > 87C: pausing 600s"; sleep 600; done; log "GPU at ${t}C"; }
jget() { $PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {'d': d}))" "$1" "$2" 2>/dev/null; }

run_stage() {  # run_stage NAME TIMEOUT_S CMD...: child process, PID file, wait; returns the child's exit code
  local name=$1 limit=$2; shift 2
  disk_guard "$name" || exit 75   # < MIN_FREE_GB on MLStorage (or unreadable): stop the queue (D111)
  if [ "$DRY_RUN" = 1 ]; then log "DRY $name (timeout ${limit}s): $*"; return 0; fi
  if [ "$limit" -lt 300 ]; then log "skip $name: only ${limit}s left"; return 98; fi
  log "start $name (timeout ${limit}s): $*"
  timeout "$limit" "$@" > "$LOG/$name.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$LOG/$name.pid"
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
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-08:30}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "queue start; GPU deadline $(ist "$DEADLINE") IST; queue PID $$"
export HF_HUB_OFFLINE=1   # every model is local or cached; never resolve `main` online (D78)

# ---- 0. inputs that must exist (made on CPU, 2026-10-19) --------------------------------------------------------------
for f in results/x2/random_init_model/config.json results/x3/data/rows.jsonl results/a5_baselines_4bit/models.joblib; do
  [ -e "$f" ] || { [ "$DRY_RUN" = 1 ] && log "DRY: missing $f" || { log "BLOCKED: missing $f"; }; }
done

# ---- 1. GPU: batch-size probe (D81) --------------------------------------------------------------------------------
cool
run_stage x2_batch_probe "$(minimum "$(left)" 1800)" $PY scripts/x2_batch_probe.py \
  || log "BLOCKED? x2_batch_probe failed (see $LOG/x2_batch_probe.log): batch ${FALLBACK_BATCH}, ${FALLBACK_CHUNK_S}s per chunk"
BATCH=$(jget results/x2/batch_probe.json "d['chosen_batch_size']"); BATCH=${BATCH:-$FALLBACK_BATCH}
EST=$(jget results/x2/batch_probe.json "int(d['worst_case_seconds_per_chunk'] * $CHUNK_SAFETY) + 1"); EST=${EST:-$FALLBACK_CHUNK_S}
[ "$DRY_RUN" = 1 ] && { BATCH=${BATCH:-64}; EST=${EST:-900}; }
log "X2 sampling: batch $BATCH, worst-case chunk ${EST}s (x $CHUNK_SAFETY)"
commit_stage "X2 (D81): batch-size probe (largest batch that fits, worst case)" results/x2/batch_probe.json

# ---- 2. GPU: X2 sampling in chunks of the frozen order (D74 / D81) --------------------------------------------------
start=0; i=0
while [ "$start" -lt "$N_CANDIDATES" ]; do
  if [ $(( $(left) - X2_DOWNSTREAM_S )) -lt "$EST" ] && [ "$DRY_RUN" != 1 ]; then
    log "stop X2 sampling before chunk $i: $(left)s left, a chunk needs ${EST}s + ${X2_DOWNSTREAM_S}s downstream; prefix = $start candidates (D74)"
    break
  fi
  cool
  dir=results/x2/sft_k8/chunk_$(printf %05d "$start")
  run_stage "x2_sample_chunk$i" "$(minimum $(( $(left) - X2_DOWNSTREAM_S / 2 )) $((3 * EST)))" \
    $PY -m physcal.eval.x2_pairs sample --start "$start" --count "$CHUNK" --batch-size "$BATCH" --chunk "$i"; rc=$?
  if [ "$rc" != 0 ]; then
    log "BLOCKED x2_sample_chunk$i exit $rc (see $LOG/x2_sample_chunk$i.log); sampling stops, prefix = $start candidates"
    [ "$DRY_RUN" != 1 ] && [ -d "$dir" ] && [ ! -f "$dir/rollouts.jsonl" ] && rm -rf "$dir"
    break
  fi
  commit_stage "X2 (D74/D81): k=8 samples of the corrected SFT model, candidates $start-$((start + CHUNK - 1)) (chunk $i, batch $BATCH)" "$dir"
  start=$((start + CHUNK)); i=$((i + 1))
done
SAMPLED=$start
log "X2 sampling done: $SAMPLED of $N_CANDIDATES candidates"

X2_OK=1
if [ "$SAMPLED" = 0 ] && [ "$DRY_RUN" != 1 ]; then X2_OK=0; log "BLOCKED: no X2 chunk; X2 stages 3-6 skipped"; fi

if [ "$X2_OK" = 1 ]; then
  # ---- 3. CPU: selection ------------------------------------------------------------------------------------------
  run_stage x2_select 900 env CUDA_VISIBLE_DEVICES="" nice -n 10 $PY -m physcal.eval.x2_pairs select \
    || log "BLOCKED x2_select (see $LOG/x2_select.log)"
  [ "$DRY_RUN" != 1 ] && tail -1 "$LOG/x2_select.log" | tee -a "$LOG/chain.log"
  commit_stage "X2 (D74): selection of the mixed problems from the sampled prefix" results/x2/selected.json

  # ---- 4. GPU: method 2, split credit's sigma suffixes --------------------------------------------------------------
  cool
  run_stage x2_suffix "$(minimum "$(left)" 3600)" $PY -m physcal.eval.x2_pairs suffix \
    || log "BLOCKED x2_suffix (see $LOG/x2_suffix.log): method 2 counts as p = 1"
  commit_stage "X2 (D74): split credit seed 0 @287 sigma suffixes on the kept answers (corrected loader)" \
    results/x2/split_seed0_suffix

  # ---- 5. GPU: hidden states (methods 5 / 6) ------------------------------------------------------------------------
  for model in sft random_init; do
    for set in fit x2; do
      cool
      run_stage "x2_hidden_${model}_${set}" "$(minimum "$(left)" 3600)" \
        $PY -m physcal.eval.x2_hidden extract --model "$model" --set "$set" \
        || log "BLOCKED x2_hidden_${model}_${set} (see $LOG/x2_hidden_${model}_${set}.log)"
    done
  done
  commit_stage "X2 (D74): hidden states at the first ± token (SFT 4-bit and random init; index and info, arrays local)" \
    results/x2/hidden

  # ---- 6. CPU, background: the X2 analysis ---------------------------------------------------------------------------
  x2_analysis() {
    if [ "$DRY_RUN" = 1 ]; then log "DRY x2_analysis (CPU, nice, 4 jobs)"; return; fi
    log "start x2_analysis (CPU, background)"
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" nice -n 10 \
      $PY -m physcal.eval.x2_analysis analyze > "$LOG/x2_analysis.log" 2>&1; local rc=$?
    log "end x2_analysis: exit $rc"; [ "$rc" != 0 ] && log "BLOCKED x2_analysis (see $LOG/x2_analysis.log)"
    commit_stage "X2 (D74) results: pairwise accuracy of every confidence source on the kept problems" \
      results/x2/results.json results/x2/summary.md
  }
  x2_analysis &
  ANALYSIS_PID=$!
  echo "$ANALYSIS_PID" > "$LOG/x2_analysis.pid"
fi

# ---- 7. GPU: the X3 pilot, only if >= X3_MIN_S remain (D79) -----------------------------------------------------------
x3_paths() { local a=$1; echo "results/x3/$a/summary.json results/x3/$a/events.jsonl results/x3/$a/physcal_fingerprint.json"; }
x3_eval() {  # x3_eval ARM: both eval sets, committed
  local a=$1 s
  for s in eval x2; do
    [ "$s" = x2 ] && [ "$X2_OK" != 1 ] && { log "skip x3_eval_${a}_x2: no X2 samples"; continue; }
    cool
    run_stage "x3_eval_${a}_${s}" "$(minimum "$(left)" 2400)" $PY -m physcal.train.x3 eval --arm "$a" --set "$s" \
      || log "BLOCKED x3_eval_${a}_${s} (see $LOG/x3_eval_${a}_${s}.log)"
    commit_stage "X3 pilot (D79): student ($a) sigma suffixes on the $s set (4-bit, unmerged, R2-1 enforced)" \
      "results/x3/$a/eval_$s"
  done
}
if [ "$(left)" -ge "$X3_MIN_S" ] || [ "$DRY_RUN" = 1 ]; then
  per=$(( ($(left) - X3_EVAL_RESERVE_S - 2 * X3_SLACK_S) / 2 ))
  log "X3 pilot: $(left)s left; student (a) wall-clock cap $((per / 60)) min (2 epochs if they fit)"
  cool
  run_stage x3_train_a $((per + X3_SLACK_S)) $PY -m physcal.train.x3 train --arm a --minutes $((per / 60)); rc=$?
  [ "$rc" != 0 ] && log "BLOCKED? x3_train_a exit $rc (see $LOG/x3_train_a.log)"
  [ "$DRY_RUN" != 1 ] && [ -f results/x3/a/final_adapter/physcal_fingerprint.json ] && \
    cp results/x3/a/final_adapter/physcal_fingerprint.json results/x3/a/
  commit_stage "X3 pilot (D79): student (a), the out-of-fold A5 teacher sigma (exit $rc); R2-1 fingerprint" $(x3_paths a)
  STEPS_A=$(jget results/x3/a/summary.json "d['steps']")
  [ "$DRY_RUN" = 1 ] && STEPS_A=${STEPS_A:-500}
  if [ -n "$STEPS_A" ] && [ "$STEPS_A" -gt 0 ]; then
    x3_eval a
    cool
    lim=$(( $(left) - X3_EVAL_RESERVE_S / 2 ))
    log "X3 student (b): matched to (a)'s $STEPS_A steps; timeout ${lim}s"
    run_stage x3_train_b "$lim" $PY -m physcal.train.x3 train --arm b --max-steps "$STEPS_A" \
      --minutes $(( (lim - X3_SLACK_S) / 60 )); rc=$?
    [ "$rc" != 0 ] && log "BLOCKED? x3_train_b exit $rc (see $LOG/x3_train_b.log)"
    [ "$DRY_RUN" != 1 ] && [ -f results/x3/b/final_adapter/physcal_fingerprint.json ] && \
      cp results/x3/b/final_adapter/physcal_fingerprint.json results/x3/b/
    commit_stage "X3 pilot (D79): student (b), the true-error-label control at (a)'s step count (exit $rc)" $(x3_paths b)
    STEPS_B=$(jget results/x3/b/summary.json "d['steps']")
    [ -n "$STEPS_B" ] && [ "$STEPS_B" != "$STEPS_A" ] && log "NOTE: student (b) ran $STEPS_B steps, (a) $STEPS_A (budget not matched)"
    [ -n "$STEPS_B" ] && x3_eval b
  else
    log "BLOCKED: student (a) has no summary; X3 evals and student (b) skipped"
  fi
  if [ -n "${ANALYSIS_PID:-}" ] && [ "$DRY_RUN" != 1 ]; then log "waiting for the X2 analysis (PID $ANALYSIS_PID) before X3's"; wait "$ANALYSIS_PID"; ANALYSIS_PID=; fi
  run_stage x3_analyze 1800 env CUDA_VISIBLE_DEVICES="" nice -n 10 $PY -m physcal.train.x3 analyze \
    || log "BLOCKED x3_analyze (see $LOG/x3_analyze.log)"
  commit_stage "X3 pilot (D79) results: student sigma vs the teacher and the control, frozen sets and X2 pairs" \
    results/x3/pilot.json results/x3/pilot.md
else
  log "skip the X3 pilot: $(left)s left < X3_MIN_S=${X3_MIN_S}s (your rule: only if >= 2 h remain)"
fi

# ---- 8. GPU, if time remains: the D78 pinned-base check ------------------------------------------------------------------
if [ "$(left)" -ge "$PIN_MIN_S" ] || [ "$DRY_RUN" = 1 ]; then
  cool
  run_stage pin_smoke "$(minimum $(( $(left) - 60 )) 900)" $PY scripts/pinned_base_smoke.py \
    || log "BLOCKED? D78 pinned-base check exit nonzero (see $LOG/pin_smoke.log)"
  commit_stage "D78 GPU check: the pinned 4-bit base loads from its snapshot dir; an adapter on it names that dir" \
    results/pinned_base_smoke/report.json
else
  log "skip the D78 check: $(left)s left < PIN_MIN_S=${PIN_MIN_S}s"
fi

if [ -n "${ANALYSIS_PID:-}" ] && [ "$DRY_RUN" != 1 ]; then log "waiting for the X2 analysis (PID $ANALYSIS_PID)"; wait "$ANALYSIS_PID"; fi
log "queue done"

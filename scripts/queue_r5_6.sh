#!/usr/bin/env bash
# GPU chain, 2026-10-30/31 (DECISIONS.md D147): R5-6 part J, the frozen J set.
#   1-6. Evals of D-pos-v2 and B-v2, seeds 0-2, on the frozen J files (stage_r4 eval --r5-6; D118's confirmatory eval
#        otherwise), resumable (D142): a cut-off attempt is retried from its partial file, up to EVAL_ATTEMPTS while
#        time is left; one still unfinished at GPU_DEADLINE is BLOCKED-but-resumable for the next run.
#   7. X1 on the J files (physcal.eval.x1_a6c r5_6; CPU, 1 thread, nice 19), once.
#   8. D147's analysis (physcal.eval.r5_6; CPU, 1 thread, nice 19), once, when all six evals are complete (D147 §11).
# The functions are queue_r4_seeds.sh's and queue_xm2.sh's (library mode): temperature guard (D143), stall watchdog
# (D132), free-space guard before every stage, GPU_DEADLINE gate, commit + push after every stage, skips for done work.
#
# usage:  GPU_DEADLINE=10:00 setsid nohup scripts/queue_r5_6.sh >> results/queue_r5_6/nohup.log 2>&1 < /dev/null &
#         DRY_RUN=1 QUEUE_LOG=<scratch> QUEUE_OUT=<scratch> R56_OUT=<scratch> X1_OUT=<scratch> bash scripts/queue_r5_6.sh
set -u
cd "$(dirname "$0")/.."
export QUEUE_LOG=${QUEUE_LOG:-results/queue_r5_6}
QUEUE_LIB_ONLY=1 . scripts/queue_r4_seeds.sh   # LOG, log, run_stage, commit_stage, cool, launch_cap, complete_adapter, OUT
XM2_LIB_ONLY=1 . scripts/queue_xm2.sh          # CPU_ENV, gpu_stage, partial_batches, EVAL_ATTEMPTS, EVAL_MIN_S
R56_ORDER=${R56_ORDER:-"dpos2:0 b2:0 dpos2:1 b2:1 dpos2:2 b2:2"}   # D147 §13 (D118's order)
R56_BATCHES=${R56_BATCHES:-102}                # 1,632 problems at batch 16
R56_OUT=${R56_OUT:-results/r5_6}               # configs/r5_6.yaml output_dir; tests: an empty dir
X1_OUT=${X1_OUT:-results/x1_text_sigma/r5_6}   # X1's J predictions; tests: an empty dir
CPU_TIMEOUT_S=${CPU_TIMEOUT_S:-7200}

eval_r56() {  # eval_r56 ARM SEED: the arm's resumable eval on the frozen J files (D142, D147), up to EVAL_ATTEMPTS
  local arm=$1 seed=$2 dir=$OUT/$1/seed$2 ev=$OUT/$1/seed$2/r5_6 attempt=0 rc
  if [ -f "$ev/metrics.json" ]; then log "SKIP eval r5_6 $arm seed $seed: already evaluated ($ev/metrics.json)"; fi
  while [ ! -f "$ev/metrics.json" ] && [ "$attempt" -lt "$EVAL_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    if [ "$DRY_RUN" != 1 ] && ! complete_adapter "$dir"; then log "BLOCKED eval r5_6 $arm seed $seed: no complete adapter in $dir"; break; fi
    if [ "$(left)" -lt "$EVAL_MIN_S" ] && [ "$DRY_RUN" != 1 ]; then
      log "SKIP eval r5_6 $arm seed $seed attempt $attempt: $(( $(left) / 60 )) min left, needs $(( EVAL_MIN_S / 60 ))"; break
    fi
    log "eval r5_6 $arm seed $seed attempt $attempt of $EVAL_ATTEMPTS: $(partial_batches "$ev") of $R56_BATCHES batches already on disk"
    cool
    gpu_stage "r56_eval_${arm}_s${seed}_a$attempt" "$(left)" env PHYSCAL_CONFIRMATORY=1 $PY -m physcal.train.stage_r4 eval \
      --arm "$arm" --seed "$seed" --r5-6 --max-reserved-gib "$CAP"; rc=$?
    [ "$DRY_RUN" = 1 ] && break
    if [ -f "$ev/metrics.json" ]; then
      gzip -kfc "$ev/rollouts.jsonl" > "$ev/rollouts.jsonl.gz"
      commit_stage "R5-6 J (D147): $arm seed $seed greedy eval on the frozen J set, attempt $attempt (exit $rc)" \
        "$ev/rollouts.jsonl.gz" "$ev/rollouts.meta.json" "$ev/metrics.json" "$ev/summary.md" "$ev"/*.png
    else
      log "BLOCKED-resumable r56_eval_${arm}_s${seed} attempt $attempt (exit $rc): $(partial_batches "$ev") of $R56_BATCHES batches on disk; the next attempt or run continues it (D142)"
      commit_stage "R5-6 J (D147): $arm seed $seed eval attempt $attempt cut off (exit $rc), $(partial_batches "$ev") of $R56_BATCHES batches on disk, resumable"
    fi
  done
  return 0
}

[ "${R56_LIB_ONLY:-0}" = 1 ] && return 0   # tests source the functions above without running the chain

exec 8>>"$LOG/queue.lock"
if ! flock -n 8; then
  log "already running: another queue holds $LOG/queue.lock; this copy (PID $$) exits 0"; exit 0
fi
echo $$ > "$LOG/queue.pid"
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-10:00}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "R5-6 J chain start (D147); GPU deadline $(ist "$DEADLINE") IST ($(( $(left) / 60 )) min left); chain PID $$"
export HF_HUB_OFFLINE=1   # every model is local (D78)
for f in data/eval_sets/2.0.0+bdbb2279/confirmatory/r5_6/a6c_pairs.jsonl data/eval_sets/2.0.0+bdbb2279/confirmatory/r5_6/a6b_pairs.jsonl \
         data/jacobian_support/r4_1.json $OUT/token_budget.json; do
  [ -e "$f" ] || log "BLOCKED: missing $f"
done
CAP=$(launch_cap)
log "launch cap ${CAP} GiB (max ${CAP_MAX})"

# ---- 1-6. the six J evals, resumable (D142) ----------------------------------------------------------------------------
for item in $R56_ORDER; do eval_r56 "${item%%:*}" "${item##*:}"; done

# ---- 7. X1 on the J files (D147 §7), once -------------------------------------------------------------------------------
if [ -f "$X1_OUT/predictions.jsonl" ]; then
  log "SKIP x1_r5_6: $X1_OUT/predictions.jsonl exists"
else
  run_stage x1_r5_6 "$CPU_TIMEOUT_S" $CPU_ENV env PHYSCAL_CONFIRMATORY=1 $PY -m physcal.eval.x1_a6c r5_6 \
    || log "BLOCKED x1_r5_6 (see $LOG/x1_r5_6.log): X1's tests enter Holm as missing (D147 §7)"
  commit_stage "R5-6 J (D147 §7): X1 (D99's refit) on the frozen J set" "$X1_OUT"
fi

# ---- 8. D147's analysis, once, when all six evals are complete ---------------------------------------------------------
n_done=0
for item in $R56_ORDER; do [ -f "$OUT/${item%%:*}/seed${item##*:}/r5_6/metrics.json" ] && n_done=$((n_done + 1)); done
if [ -f "$R56_OUT/results.json" ]; then
  log "SKIP r5_6 analysis: D147's one-time analysis exists ($R56_OUT/results.json; D147 §11)"
elif [ "$n_done" = 6 ] || [ "$DRY_RUN" = 1 ]; then
  run_stage r5_6_analysis "$CPU_TIMEOUT_S" $CPU_ENV env PHYSCAL_CONFIRMATORY=1 $PY -m physcal.eval.r5_6 analyze \
    || log "BLOCKED r5_6_analysis (see $LOG/r5_6_analysis.log)"
  commit_stage "R5-6 J (D147): the one-time analysis, P_J and J1-J6, D-pos-v2 and B-v2 x 3 seeds on the frozen J set" "$R56_OUT"
else
  log "D147 §11: $n_done of 6 J evals exist; the analysis waits (the next run continues the evals)"
fi
log "R5-6 J chain done: $GPU_PAUSES temperature pauses, ${GPU_PAUSED_S}s paused of ${GPU_WALL_S}s GPU-stage time"

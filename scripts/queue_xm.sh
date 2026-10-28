#!/usr/bin/env bash
# Night GPU chain (DECISIONS.md D137, pre-registered): the cross-model pilot on Llama-3.2-3B-Instruct (R5-5), then
# MC-B A6b (D132).
#   1. SFT D-pos-v2 (Llama, seed 0)   2. its eval on the frozen A6c + A6b (PHYSCAL_CONFIRMATORY=1)
#   3. SFT B-v2 (Llama, seed 0)       4. its eval
#   5. MC-B generation on A6b (GPU), then its execution and analysis (CPU, 1 thread, nice 19)
#   6. CPU: the pilot analysis (physcal.eval.xm_pilot), once, when both Llama evals exist
# The functions are queue_r4_seeds.sh's (library mode): every GPU stage gets the temperature guard and the stall
# watchdog (D132), the free-space guard runs before every stage, every stage starts only if its estimate + SLACK fits
# before GPU_DEADLINE, and the chain commits and pushes after every stage. Finished stages are skipped.
#
# usage:  GPU_DEADLINE=10:00 setsid nohup scripts/queue_xm.sh >> results/queue_xm/nohup.log 2>&1 < /dev/null &
#         DRY_RUN=1 QUEUE_LOG=<scratch> XM_OUT=<scratch> MCB_OUT=<scratch> bash scripts/queue_xm.sh
set -u
cd "$(dirname "$0")/.."
export QUEUE_LOG=${QUEUE_LOG:-results/queue_xm}
QUEUE_LIB_ONLY=1 . scripts/queue_r4_seeds.sh   # LOG, log, run_stage, commit_stage, cool, launch_cap, mcb_generated, ...
XM_OUT=${XM_OUT:-results/stage_xm}             # configs/stage_xm.yaml output_dir; tests: an empty dir
XM_ARMS=${XM_ARMS:-"dpos2 b2"}                 # D137's order
XM_TRAIN_EST=${XM_TRAIN_EST:-8400}             # 750 steps x 9.4-10.2 s (D136, the D137 smoke) + one 10-min pause
XM_EVAL_EST_dpos2=${XM_EVAL_EST_dpos2:-9000}   # 2,040 problems at batch 6 (the smoke: 2.65 s per problem at ~150 tokens; trained ~2.5x longer)
XM_EVAL_EST_b2=${XM_EVAL_EST_b2:-7000}         # B-v2's completions are shorter (Qwen: 241 vs 386 tokens)
XM_MCB_SETS=${XM_MCB_SETS:-a6b_pairs}          # D132: A6c and A6 are done
CPU_ENV="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= nice -n 19"

exec 8>>"$LOG/queue.lock"
if ! flock -n 8; then
  log "already running: another queue holds $LOG/queue.lock; this copy (PID $$) exits 0"; exit 0
fi
echo $$ > "$LOG/queue.pid"
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-10:00}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "XM chain start (D137); GPU deadline $(ist "$DEADLINE") IST ($(( $(left) / 60 )) min left); chain PID $$"
export HF_HUB_OFFLINE=1   # every model is local (D78)
for f in results/phase12_sft_data_r4/explicit.jsonl results/phase12_sft_data_r4/supervised.jsonl \
         data/eval_sets/2.0.0+bdbb2279/confirmatory/a6c_pairs.jsonl configs/base_llama32_3b_4bit.yaml; do
  [ -e "$f" ] || log "BLOCKED: missing $f"
done
CAP=$(launch_cap)
log "launch cap ${CAP} GiB (max ${CAP_MAX})"

# ---- 1-4. SFT and eval per arm ----------------------------------------------------------------------------------------
for arm in $XM_ARMS; do
  dir=$XM_OUT/$arm/seed0
  if [ -e "$dir/final_adapter" ]; then
    log "SKIP train xm $arm: $dir/final_adapter exists"
  elif [ "$(left)" -lt $((XM_TRAIN_EST + SLACK_S)) ] && [ "$DRY_RUN" != 1 ]; then
    log "SKIP train xm $arm: $(( $(left) / 60 )) min left, needs $(( (XM_TRAIN_EST + SLACK_S) / 60 )) min"
  else
    cool
    t0=$(date +%s)
    GUARD=1 run_stage "xm_train_$arm" $(( $(left) - 300 )) $PY -m physcal.train.stage_xm train --arm "$arm" \
      --stop-at "$(iso $((DEADLINE - SLACK_S)))" --max-reserved-gib "$CAP"; rc=$?
    [ "$rc" != 0 ] && log "BLOCKED? xm_train_$arm exit $rc (see $LOG/xm_train_$arm.log)"
    [ "$DRY_RUN" != 1 ] && log "train xm $arm: $(json_get "$dir/summary.json" "['steps']") steps of $(json_get "$dir/summary.json" "['steps_planned']"), $(( ($(date +%s) - t0) / 60 )) min"
    commit_stage "XM pilot (D137): SFT $arm on Llama-3.2-3B, seed 0, R4-1 data, $(json_get "$dir/summary.json" "['steps']") steps (exit $rc)" \
      "$dir/summary.json" "$dir/events.jsonl" "$dir/final_adapter/physcal_fingerprint.json" "$dir/final_adapter/adapter_config.json"
  fi
  ev=XM_EVAL_EST_$arm; est=${!ev}
  if [ -f "$dir/confirmatory/rollouts.jsonl" ]; then log "SKIP eval xm $arm: already evaluated"; continue; fi
  if [ "$DRY_RUN" != 1 ] && ! complete_adapter "$dir"; then log "BLOCKED eval xm $arm: no complete adapter in $dir"; continue; fi
  if [ "$(left)" -lt $((est + SLACK_S)) ] && [ "$DRY_RUN" != 1 ]; then
    log "SKIP eval xm $arm: $(( $(left) / 60 )) min left, needs $(( (est + SLACK_S) / 60 )) min"; continue
  fi
  cool
  lim=$((2 * est)); [ "$(left)" -lt "$lim" ] && lim=$(left)
  GUARD=1 run_stage "xm_eval_$arm" "$lim" env PHYSCAL_CONFIRMATORY=1 $PY -m physcal.train.stage_xm eval --arm "$arm" \
    --max-reserved-gib "$CAP"; rc=$?
  [ "$rc" != 0 ] && log "BLOCKED? xm_eval_$arm exit $rc (see $LOG/xm_eval_$arm.log)"
  [ "$DRY_RUN" != 1 ] && [ -f "$dir/confirmatory/rollouts.jsonl" ] && \
    gzip -kfc "$dir/confirmatory/rollouts.jsonl" > "$dir/confirmatory/rollouts.jsonl.gz"
  commit_stage "XM pilot (D137): $arm (Llama, seed 0) greedy eval on the frozen A6c + A6b (exit $rc)" \
    "$dir/confirmatory/rollouts.jsonl.gz" "$dir/confirmatory/rollouts.meta.json" "$dir/confirmatory/metrics.json" \
    "$dir/confirmatory/summary.md" "$dir/confirmatory"/*.png
done

# ---- 5. MC-B A6b (D132): GPU generation, then CPU execution and analysis ---------------------------------------------
for set in $XM_MCB_SETS; do
  if mcb_generated "$MCB_OUT/$set"; then
    log "SKIP mc_b generate $set: already generated"
  elif [ "$(left)" -lt $((MCB_EST + SLACK_S)) ] && [ "$DRY_RUN" != 1 ]; then
    log "SKIP mc_b generate $set: $(( $(left) / 60 )) min left, needs $(( (MCB_EST + SLACK_S) / 60 )) min"
  else
    cool
    lim=$((2 * MCB_EST)); [ "$(left)" -lt "$lim" ] && lim=$(left)
    GUARD=1 run_stage "mc_b_gen_$set" "$lim" $PY -m physcal.eval.mc_b generate --set "$set" --max-reserved-gib "$CAP"; rc=$?
    [ "$rc" != 0 ] && log "BLOCKED? mc_b_gen_$set exit $rc (see $LOG/mc_b_gen_$set.log)"
    commit_stage "MC-B (D119, D132): $set, one greedy program per problem (exit $rc)" \
      "$MCB_OUT/$set/generations.jsonl" "$MCB_OUT/$set/meta.json"
  fi
  if [ -f "$MCB_OUT/$set/results.json" ]; then log "SKIP mc_b analysis $set: already analyzed"; continue; fi
  [ -f "$MCB_OUT/$set/generations.jsonl" ] || { log "BLOCKED mc_b analysis $set: no generations"; continue; }
  mcb_generated "$MCB_OUT/$set" || { log "BLOCKED mc_b analysis $set: generation incomplete (it resumes on the next run; D127)"; continue; }
  run_stage "mc_b_exec_$set" 3600 $CPU_ENV $PY -m physcal.eval.mc_b execute --set "$set" \
    && run_stage "mc_b_analyze_$set" 3600 $CPU_ENV $PY -m physcal.eval.mc_b analyze --set "$set" \
    || log "BLOCKED mc_b execute/analyze $set (see $LOG/mc_b_exec_$set.log, $LOG/mc_b_analyze_$set.log)"
  commit_stage "MC-B (D119, D132): $set executed in the sandbox and analyzed" "$MCB_OUT/$set/executions.jsonl" \
    "$MCB_OUT/$set/results.json" "$MCB_OUT/$set/summary.md"
done

# ---- 6. CPU: the pilot analysis (D137), once, when both Llama evals exist ---------------------------------------------
n_done=0
for arm in $XM_ARMS; do [ -f "$XM_OUT/$arm/seed0/confirmatory/rollouts.jsonl" ] && n_done=$((n_done + 1)); done
if [ -f "$XM_OUT/pilot/results.json" ]; then
  log "SKIP xm_pilot: the one-time analysis exists ($XM_OUT/pilot/results.json; D137)"
elif [ "$n_done" = 2 ] || [ "$DRY_RUN" = 1 ]; then
  run_stage xm_pilot 3600 $CPU_ENV env PHYSCAL_CONFIRMATORY=1 $PY -m physcal.eval.xm_pilot analyze \
    || log "BLOCKED xm_pilot (see $LOG/xm_pilot.log)"
  commit_stage "XM pilot (D137): the pre-registered analysis (go rule K1 and C2 and C3 on D-pos-v2)" "$XM_OUT/pilot"
else
  log "D137: $n_done of 2 Llama evals exist; the pilot analysis waits"
fi
log "XM chain done"

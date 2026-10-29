#!/usr/bin/env bash
# Night GPU chain, 2026-10-29/30 (DECISIONS.md D144): the XM pilot's second night, after D140's recovery.
#   1. Eval D-pos-v2 (Llama, seed 0) on the frozen A6c + A6b, resumable (D142).
#      - A cut-off attempt (timeout, stall kill, crash) is retried and continues from its partial file, up to
#        EVAL_ATTEMPTS attempts while time is left.
#      - One still unfinished at the deadline is BLOCKED-but-resumable: the next run continues it.
#   2. D137's go rule on D-pos-v2 (physcal.eval.xm_go; CPU, 1 thread, nice 19), once, when that eval is complete (D139).
#   3. MC-B A6b (D132): GPU generation, then execution and analysis (CPU, 1 thread, nice 19).
#   4. SFT B-v2 (Llama, seed 0) from scratch, only if it can finish by GPU_DEADLINE.
#      - **Start gate:** its measured active time, stretched by tonight's measured pause share.
#      - **Projection check:** every PROJ_EVERY_S after the first PROJ_FIRST_S, from its own step rate, pauses
#        included. A run that cannot finish is stopped and its directory set aside.
# Not tonight, whatever the go rule says (your instruction): no seed-1 stage and no B-v2 eval.
# The functions are queue_r4_seeds.sh's (library mode):
#   - every GPU stage gets the temperature guard (D143 hysteresis) and the stall watchdog (D132);
#   - the free-space guard runs before every stage;
#   - every stage is gated on GPU_DEADLINE;
#   - the chain commits and pushes after every stage, and finished stages are skipped.
#
# usage:  GPU_DEADLINE=13:00 setsid nohup scripts/queue_xm2.sh >> results/queue_xm2/nohup.log 2>&1 < /dev/null &
#         DRY_RUN=1 QUEUE_LOG=<scratch> XM_OUT=<scratch> MCB_OUT=<scratch> bash scripts/queue_xm2.sh
set -u
cd "$(dirname "$0")/.."
export QUEUE_LOG=${QUEUE_LOG:-results/queue_xm2}
QUEUE_LIB_ONLY=1 . scripts/queue_r4_seeds.sh   # LOG, log, run_stage, commit_stage, cool, launch_cap, mcb_generated, ...
XM_OUT=${XM_OUT:-results/stage_xm}             # configs/stage_xm.yaml output_dir; tests: an empty dir
EVAL_ATTEMPTS=${EVAL_ATTEMPTS:-3}              # the overnight rule: 3 genuine attempts, then BLOCKED
EVAL_MIN_S=${EVAL_MIN_S:-1800}                 # an attempt needs ~7 min of load + reload check before its first batch
EVAL_BATCHES=${EVAL_BATCHES:-340}              # 2,040 problems at batch 6 (configs/stage_xm.yaml)
XM_MCB_SETS=${XM_MCB_SETS:-a6b_pairs}          # D132: A6c and A6 are done
B2_ACTIVE_S=${B2_ACTIVE_S:-8800}               # D144: B-v2 on 28/29 Oct: 11.0 s / step x 750 + fingerprint + load
B2_STEPS=${B2_STEPS:-750}
B2_TAIL_S=${B2_TAIL_S:-600}                    # after the last step: R2-1's fingerprint (345 s on 28 Oct) + saving
PROJ_FIRST_S=${PROJ_FIRST_S:-1800}
PROJ_EVERY_S=${PROJ_EVERY_S:-900}
PROJ_MIN_STEPS=${PROJ_MIN_STEPS:-30}
PROJ_GRACE_S=${PROJ_GRACE_S:-30}
CPU_ENV="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= nice -n 19"
GPU_WALL_S=0 GPU_PAUSED_S=0 GPU_PAUSES=0

gpu_stage() {  # gpu_stage NAME TIMEOUT CMD...: a guarded run_stage; adds its wall time and temperature pauses to tonight's
  local name=$1 t0 rc pid p n w
  t0=$(date +%s)
  GUARD=1 run_stage "$@"; rc=$?
  [ "$DRY_RUN" = 1 ] && return $rc
  pid=$(cat "$LOG/$name.pid" 2>/dev/null)
  p=$(grep -oE "guard\($pid\): stage continued after [0-9]+s" "$LOG/chain.log" | grep -oE "[0-9]+s$" | tr -d s \
      | awk '{s += $1} END {print s + 0}')
  n=$(grep -cE "guard\($pid\): stage continued after" "$LOG/chain.log")
  w=$(( $(date +%s) - t0 ))
  GPU_WALL_S=$((GPU_WALL_S + w)) GPU_PAUSED_S=$((GPU_PAUSED_S + p)) GPU_PAUSES=$((GPU_PAUSES + n))
  log "$name: $n temperature pauses, ${p}s paused of ${w}s; tonight so far $GPU_PAUSES pauses, ${GPU_PAUSED_S}s of ${GPU_WALL_S}s (D143)"
  return $rc
}

partial_batches() {  # batches of the eval's partial file (D142); 0 without one
  if [ -f "$1/completions.partial.jsonl" ]; then awk '/"kind": "batch"/ {n++} END {print n + 0}' "$1/completions.partial.jsonl"
  else echo 0; fi
}

b2_projection() {  # b2_projection T0: while xm_train_b2 runs, project its end from its own rate; stop it if too late
  local t0=$1 pid line n el s rem end
  for _ in $(seq 120); do [ -f "$LOG/xm_train_b2.pid" ] && [ "$(stat -c %Y "$LOG/xm_train_b2.pid")" -ge "$t0" ] && break; sleep 5; done
  pid=$(cat "$LOG/xm_train_b2.pid" 2>/dev/null) || return 0
  sleep "$PROJ_FIRST_S"
  while kill -0 "$pid" 2>/dev/null; do
    line=$(tr '\r' '\n' < "$LOG/xm_train_b2.log" | grep -oE "[0-9]+/$B2_STEPS \[[0-9:]+" | tail -1)
    n=${line%%/*}; el=${line##*[}
    if [ -n "$line" ] && [ "${n:-0}" -ge "$PROJ_MIN_STEPS" ]; then
      s=$(echo "$el" | awk -F: '{t = 0; for (i = 1; i <= NF; i++) t = t * 60 + $i; print t}')
      rem=$(( s * (B2_STEPS - n) / n + B2_TAIL_S )); end=$(( $(date +%s) + rem ))
      if [ "$end" -gt $((DEADLINE - SLACK_S)) ]; then
        log "B-v2 cannot finish by $(ist $((DEADLINE - SLACK_S))) IST: $n / $B2_STEPS steps in ${el} (pauses included) projects $(ist "$end"); stopping it (D144)"
        kill -TERM -- "-$pid" 2>/dev/null; sleep "$PROJ_GRACE_S"; kill -KILL -- "-$pid" 2>/dev/null
        return 0
      fi
      log "B-v2 projection: $n / $B2_STEPS steps in ${el} (pauses included) projects $(ist "$end") IST, before $(ist $((DEADLINE - SLACK_S))) (D144)"
    fi
    sleep "$PROJ_EVERY_S"
  done
}

[ "${XM2_LIB_ONLY:-0}" = 1 ] && return 0   # tests source the functions above without running the chain

exec 8>>"$LOG/queue.lock"
if ! flock -n 8; then
  log "already running: another queue holds $LOG/queue.lock; this copy (PID $$) exits 0"; exit 0
fi
echo $$ > "$LOG/queue.pid"
DEADLINE=$(TZ=Asia/Kolkata date -d "${GPU_DEADLINE:-13:00}" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log "XM night-2 chain start (D144); GPU deadline $(ist "$DEADLINE") IST ($(( $(left) / 60 )) min left); chain PID $$"
export HF_HUB_OFFLINE=1   # every model is local (D78)
for f in data/eval_sets/2.0.0+bdbb2279/confirmatory/a6c_pairs.jsonl data/eval_sets/2.0.0+bdbb2279/confirmatory/a6b_pairs.jsonl \
         results/phase12_sft_data_r4/explicit.jsonl results/phase12_sft_data_r4/supervised.jsonl configs/base_llama32_3b_4bit.yaml; do
  [ -e "$f" ] || log "BLOCKED: missing $f"
done
CAP=$(launch_cap)
log "launch cap ${CAP} GiB (max ${CAP_MAX})"

# ---- 1. eval D-pos-v2 (Llama, seed 0), resumable (D142) ---------------------------------------------------------------
dir=$XM_OUT/dpos2/seed0
ev=$dir/confirmatory
attempt=0
if [ -f "$ev/metrics.json" ]; then log "SKIP eval xm dpos2: already evaluated ($ev/metrics.json)"; fi
while [ ! -f "$ev/metrics.json" ] && [ "$attempt" -lt "$EVAL_ATTEMPTS" ]; do
  attempt=$((attempt + 1))
  if [ "$DRY_RUN" != 1 ] && ! complete_adapter "$dir"; then log "BLOCKED eval xm dpos2: no complete adapter in $dir"; break; fi
  if [ "$(left)" -lt "$EVAL_MIN_S" ] && [ "$DRY_RUN" != 1 ]; then
    log "SKIP eval xm dpos2 attempt $attempt: $(( $(left) / 60 )) min left, needs $(( EVAL_MIN_S / 60 ))"; break
  fi
  log "eval xm dpos2 attempt $attempt of $EVAL_ATTEMPTS: $(partial_batches "$ev") of $EVAL_BATCHES batches already on disk"
  cool
  gpu_stage "xm_eval_dpos2_a$attempt" "$(left)" env PHYSCAL_CONFIRMATORY=1 $PY -m physcal.train.stage_xm eval \
    --arm dpos2 --max-reserved-gib "$CAP"; rc=$?
  [ "$DRY_RUN" = 1 ] && break
  if [ -f "$ev/metrics.json" ]; then
    gzip -kfc "$ev/rollouts.jsonl" > "$ev/rollouts.jsonl.gz"
    commit_stage "XM pilot (D137, D144): dpos2 (Llama, seed 0) greedy eval on the frozen A6c + A6b, attempt $attempt (exit $rc)" \
      "$ev/rollouts.jsonl.gz" "$ev/rollouts.meta.json" "$ev/metrics.json" "$ev/summary.md" "$ev"/*.png
  else
    log "BLOCKED-resumable xm_eval_dpos2 attempt $attempt (exit $rc): $(partial_batches "$ev") of $EVAL_BATCHES batches on disk; the next attempt or run continues it (D142)"
    commit_stage "XM pilot (D144): dpos2 (Llama, seed 0) eval attempt $attempt cut off (exit $rc), $(partial_batches "$ev") of $EVAL_BATCHES batches on disk, resumable"
  fi
done

# ---- 2. the go rule on D-pos-v2 (D137, run early per D139), once, when its eval is complete -----------------------------
GO=$XM_OUT/go_dpos2/go.json
if [ -f "$GO" ]; then
  log "SKIP xm_go: the one-time go rule exists ($GO)"
elif [ -f "$ev/metrics.json" ] || [ "$DRY_RUN" = 1 ]; then
  run_stage xm_go 3600 $CPU_ENV env PHYSCAL_CONFIRMATORY=1 $PY -m physcal.eval.xm_go analyze \
    || log "go rule not evaluated (see $LOG/xm_go.log)"
  commit_stage "XM pilot (D139, D144): D137's go rule on D-pos-v2 (Llama, seed 0)" "$XM_OUT/go_dpos2"
  [ "$DRY_RUN" != 1 ] && log "go rule: $(json_get "$GO" "['go']") (K1 $(json_get "$GO" "['K1']"), C2 $(json_get "$GO" "['C2']"), C3 $(json_get "$GO" "['C3']")); no seed-1 stage tonight either way (D144)"
else
  log "go rule not evaluated: the D-pos-v2 eval is not complete (D139); nothing is added"
fi

# ---- 3. MC-B A6b (D132): GPU generation, then CPU execution and analysis -----------------------------------------------
for set in $XM_MCB_SETS; do
  if mcb_generated "$MCB_OUT/$set"; then
    log "SKIP mc_b generate $set: already generated"
  elif [ "$(left)" -lt $((MCB_EST + SLACK_S)) ] && [ "$DRY_RUN" != 1 ]; then
    log "SKIP mc_b generate $set: $(( $(left) / 60 )) min left, needs $(( (MCB_EST + SLACK_S) / 60 )) min"
  else
    cool
    lim=$((2 * MCB_EST)); [ "$(left)" -lt "$lim" ] && lim=$(left)
    gpu_stage "mc_b_gen_$set" "$lim" $PY -m physcal.eval.mc_b generate --set "$set" --max-reserved-gib "$CAP"; rc=$?
    [ "$rc" != 0 ] && log "BLOCKED? mc_b_gen_$set exit $rc (see $LOG/mc_b_gen_$set.log)"
    commit_stage "MC-B (D119, D132, D144): $set, one greedy program per problem (exit $rc)" \
      "$MCB_OUT/$set/generations.jsonl" "$MCB_OUT/$set/meta.json"
  fi
  if [ -f "$MCB_OUT/$set/results.json" ]; then log "SKIP mc_b analysis $set: already analyzed"; continue; fi
  if [ "$DRY_RUN" != 1 ]; then
    [ -f "$MCB_OUT/$set/generations.jsonl" ] || { log "BLOCKED mc_b analysis $set: no generations"; continue; }
    mcb_generated "$MCB_OUT/$set" || { log "BLOCKED mc_b analysis $set: generation incomplete (it resumes on the next run; D127)"; continue; }
  fi
  run_stage "mc_b_exec_$set" 3600 $CPU_ENV $PY -m physcal.eval.mc_b execute --set "$set" \
    && run_stage "mc_b_analyze_$set" 3600 $CPU_ENV $PY -m physcal.eval.mc_b analyze --set "$set" \
    || log "BLOCKED mc_b execute/analyze $set (see $LOG/mc_b_exec_$set.log, $LOG/mc_b_analyze_$set.log)"
  commit_stage "MC-B (D119, D132, D144): $set executed in the sandbox and analyzed" "$MCB_OUT/$set/executions.jsonl" \
    "$MCB_OUT/$set/results.json" "$MCB_OUT/$set/summary.md"
done

# ---- 4. SFT B-v2 (Llama, seed 0), from scratch, only if it can finish by the deadline -------------------------------
dir=$XM_OUT/b2/seed0
share=$(awk -v p="$GPU_PAUSED_S" -v w="$GPU_WALL_S" 'BEGIN {printf "%.3f", (w > 0 ? p / w : 0)}')
est=$(awk -v a="$B2_ACTIVE_S" -v f="$share" 'BEGIN {printf "%d", a / (1 - (f < 0.9 ? f : 0.9))}')
if [ -e "$dir/final_adapter" ]; then
  log "SKIP train xm b2: $dir/final_adapter exists"
elif [ "$(left)" -lt $((est + SLACK_S)) ] && [ "$DRY_RUN" != 1 ]; then
  log "SKIP train xm b2: $(( $(left) / 60 )) min left, needs $(( (est + SLACK_S) / 60 )) min ($B2_ACTIVE_S s active / (1 - $share paused tonight) + slack; D144)"
else
  cool
  log "train xm b2: estimate $(( est / 60 )) min ($B2_ACTIVE_S s active / (1 - $share paused tonight)); the projection check decides from its own rate (D144)"
  t0=$(date +%s)
  proj=
  if [ "$DRY_RUN" != 1 ]; then b2_projection "$t0" & proj=$!; fi
  gpu_stage xm_train_b2 $(( $(left) - 300 )) $PY -m physcal.train.stage_xm train --arm b2 \
    --stop-at "$(iso $((DEADLINE - SLACK_S)))" --max-reserved-gib "$CAP"; rc=$?
  [ -n "$proj" ] && { kill "$proj" 2>/dev/null; wait "$proj" 2>/dev/null; }
  if [ "$DRY_RUN" != 1 ]; then
    if complete_adapter "$dir"; then
      log "train xm b2: $(json_get "$dir/summary.json" "['steps']") steps of $(json_get "$dir/summary.json" "['steps_planned']"), $(( ($(date +%s) - t0) / 60 )) min"
      commit_stage "XM pilot (D137, D144): SFT b2 on Llama-3.2-3B, seed 0, R4-1 data, $(json_get "$dir/summary.json" "['steps']") steps (exit $rc)" \
        "$dir/summary.json" "$dir/events.jsonl" "$dir/final_adapter/physcal_fingerprint.json" "$dir/final_adapter/adapter_config.json"
    else
      aside=results/discarded/xm-b2-seed0-incomplete-$(TZ=Asia/Kolkata date +%Y%m%d-%H%M)
      [ -e "$dir" ] && mkdir -p "$aside" && mv "$dir" "$aside/seed0"
      log "BLOCKED? xm_train_b2 exit $rc: no complete adapter; its directory is set aside in $aside (D144)"
      commit_stage "XM pilot (D144): SFT b2 on Llama-3.2-3B, seed 0, incomplete (exit $rc), set aside"
    fi
  fi
fi
log "XM night-2 chain done: tonight $GPU_PAUSES temperature pauses, ${GPU_PAUSED_S}s paused of ${GPU_WALL_S}s GPU-stage time"

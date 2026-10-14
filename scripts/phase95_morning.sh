#!/usr/bin/env bash
# Phase 9.5 morning chain (split-credit run; same as phase37_morning.sh with the run dir as an argument): wait for the overnight run to exit, then eval the step-287 adapter, then the final
# adapter (frozen eval sets, verifier 1.1.0), then the paired comparisons. GPU temp > 87 C -> pause 10 min.
# Every GPU stage is killed at 08:30 IST at the latest.
set -u
cd "$(dirname "$0")/.."
R=${2:?run dir, e.g. results/train/<run_id>}
LABEL=qwen2.5-math-1.5b-instruct
DEADLINE=$(TZ=Asia/Kolkata date -d "08:30" +%s); [ "$DEADLINE" -lt "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
log() { echo "[$(TZ=Asia/Kolkata date +%H:%M:%S)] $*"; }
cool() { while t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "${t:-0}" -gt 87 ]; do
           log "GPU at ${t}C > 87C: pausing 600s"; sleep 600; done; log "GPU at ${t}C"; }

# usage: phase95_morning.sh TRAIN_PID RUN_DIR  (waits on the PID: a pgrep pattern would also match watcher shells)
TRAIN_PID=${1:?training python PID}
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 20; done
log "training process exited"; sleep 10
for spec in "permanent_checkpoints/checkpoint-287 rl287" "final_adapter rlfinal"; do
  set -- $spec
  if [ ! -d "$R/$1" ]; then log "MISSING adapter $R/$1, skipping"; continue; fi
  cool
  left=$((DEADLINE - $(date +%s)))
  if [ "$left" -lt 300 ]; then log "less than 5 min before 08:30 IST, not starting $2"; continue; fi
  log "eval $2 ($1), timeout ${left}s"
  timeout "$left" uv run python -m physcal.train.compare --part rl --run "$R" --label "$LABEL" --budget 512 \
    --adapter "$1" --out-name "$2" > "$R/compare_$2.log" 2>&1
  log "eval $2 exit $?"
done
uv run python scripts/phase95_compare.py "$R" > "$R/phase95_compare.log" 2>&1
log "comparisons exit $?"

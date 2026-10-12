#!/usr/bin/env bash
# Phase 4e: evaluate the final GRPO adapter on the selected model's probe eval sets, under the same
# transformers version as the "before" probe (5.19.0), killed no later than the GPU deadline.
set -u
RUN=${1:-results/train/phase4-grpo-qwen3.5-2b-crps_log}
DEADLINE=${2:-"15:44"}
now=$(date +%s); stop=$(TZ=Asia/Kolkata date -d "today $DEADLINE" +%s)
budget=$(( stop - now ))
echo "eval budget: ${budget}s (until $DEADLINE IST); GPU temp $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)C"
if [ "$budget" -le 60 ]; then echo "no time left before the GPU deadline; skipping eval"; exit 3; fi
timeout --signal=TERM "$budget" uv run --with "transformers==5.19.0" python -m physcal.train.evaluate \
  --run "$RUN" --label qwen3.5-2b --model-class causal_lm

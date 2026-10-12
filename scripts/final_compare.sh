#!/usr/bin/env bash
# Phase 3.6 tail: after the overnight GRPO exits, evaluate raw base / SFT-only / SFT+RL with identical settings
# (frozen eval sets, prompt v4, greedy, the training budget, text-only loading) and run the paired tests.
set -u
RUN=${1:?run dir}; LABEL=${2:?probe label of the base model}; SFT_LABEL=${3:?sft label}; BUDGET=${4:?budget}
cd "$(dirname "$0")/.."
for part in rl sft base; do
  ./scripts/gpu_cool.sh
  echo "=== compare $part $(date -Is)"
  uv run python -m physcal.train.compare --part "$part" --run "$RUN" --label "$LABEL" \
     --sft-dir "results/sft/$SFT_LABEL/merged" --budget "$BUDGET" 2>&1 | grep -vE "Loading weights|Warning|warn|generated"
done
echo "=== paired tests $(date -Is)"
uv run python -m physcal.train.compare --part tests --run "$RUN" > /dev/null
uv run python -m physcal.train.plots "$RUN"
echo "FINAL_DONE $(date -Is)"

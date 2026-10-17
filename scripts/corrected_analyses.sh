#!/usr/bin/env bash
# D67: the pre-registered families S, N and E on the corrected evals (adapters unmerged on the 4-bit training base),
# then the corrected-vs-superseded table. CPU only; called by scripts/queue_corrected_evals.sh after tier 1 and at the
# end. A part whose inputs are not there yet is skipped (and says so); outputs go to results/corrected/.
set -u
cd "$(dirname "$0")/.."
T=results/train
SPLIT=$T/split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed
UNI=$T/overnight2-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512
rc=0
need() { for f in "$@"; do [ -f "$f" ] || { echo "missing $f"; return 1; }; done; }
controls_ok() { need $UNI/compare/sft_4bit/rollouts.jsonl $UNI/compare/rl287_4bit/rollouts.jsonl; }

for s in 0 1 2; do  # families S and N, per seed
  if controls_ok && need ${SPLIT}${s}/compare/rl287_4bit/rollouts.jsonl; then
    uv run python scripts/phase95_compare.py ${SPLIT}${s} --corrected > /dev/null || { echo "phase95_compare seed $s FAILED"; rc=1; }
    echo "family S/N seed $s: done"
  else
    echo "family S/N seed $s: skipped"
  fi
done
if controls_ok && need ${SPLIT}0/compare/rl287_4bit/rollouts.jsonl ${SPLIT}1/compare/rl287_4bit/rollouts.jsonl \
                        ${SPLIT}2/compare/rl287_4bit/rollouts.jsonl; then  # family S on the seed mean (D62)
  uv run python scripts/phase95_seed_mean.py ${SPLIT}0 ${SPLIT}1 ${SPLIT}2 --corrected > /dev/null \
    || { echo "phase95_seed_mean FAILED"; rc=1; }
  echo "family S seed mean: done"
else
  echo "family S seed mean: skipped"
fi
if need results/e1/sft_4bit/rows.jsonl $T/e1a-sigma-only-free-seed0/e1_eval/final_4bit/rows.jsonl; then  # family E
  uv run python -c "from physcal.train.e1_compare import compare; compare(corrected=True)" \
    || { echo "family E FAILED"; rc=1; }
  echo "family E: done (an E1 run without a corrected eval counts as p = 1, as pre-registered)"
else
  echo "family E: skipped"
fi
uv run python scripts/corrected_vs_superseded.py > /dev/null || { echo "side by side FAILED"; rc=1; }
echo "side by side: results/corrected/side_by_side.md"
exit $rc

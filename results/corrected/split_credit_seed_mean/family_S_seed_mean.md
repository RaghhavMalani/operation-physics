# Family S on the seed mean (D62), corrected evals: 4-bit training base, adapters unmerged (D67)

Runs: results/train/split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed0, results/train/split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed1, results/train/split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed2; sanity: [True, True, True]

| test | split credit, seed mean | control | difference | 95% CI | p (one-sided) | Holm threshold | supported | reported as | per-seed p (runs in the order listed) |
|---|---|---|---|---|---|---|---|---|---|
| S1a | 0.4183 | 0.1200 | +0.2983 | [+0.2550, +0.3417] | 0 | 0.0125 | True | supported | 7.17e-34 / 2.92e-32 / 9.41e-35 |
| S1b | 0.4183 | 0.1050 | +0.3133 | [+0.2700, +0.3575] | 0 | 0.0167 | True | supported | 7.52e-37 / 4.78e-35 / 9.62e-38 |
| S2 | 0.0607 | 0.0146 | +0.0461 | [-0.0945, +0.1837] | 0.262 | 0.05 | False | not supported | 0.207 / 0.396 / 0.225 |
| S3 | 0.3767 | 0.3725 | +0.0042 | [-0.0100, +0.0183] | 0 | 0.025 | True | supported | 0 / 0 / 0 |

Per seed (as phase95_compare.py; one-sided p, no Holm):

| run | S1a covered vs SFT | p | S1b covered vs uniform | p | S2 ρ split vs uniform | p | S3 acc split vs uniform | p |
|---|---|---|---|---|---|---|---|---|
| seed0 | 167 vs 48 | 7.17e-34 | 167 vs 42 | 7.52e-37 | 0.077 vs 0.015 | 0.207 | 37.8% vs 37.2% | 0 |
| seed1 | 165 vs 48 | 2.92e-32 | 165 vs 42 | 4.78e-35 | 0.034 vs 0.015 | 0.396 | 36.8% vs 37.2% | 0 |
| seed2 | 170 vs 48 | 9.41e-35 | 170 vs 42 | 9.62e-38 | 0.071 vs 0.015 | 0.225 | 38.5% vs 37.2% | 0 |

Range across seeds: 1σ coverage 41.2%–42.5%, accuracy 36.8%–38.5%, ρ 0.034–0.077.

# Family S on the seed mean (D62)

Runs: results/train/split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed0, results/train/split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed1, results/train/split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed2; sanity: [True, True, True]

| test | split credit, seed mean | control | difference | 95% CI | p (one-sided) | Holm threshold | supported | reported as | per-seed p (runs in the order listed) |
|---|---|---|---|---|---|---|---|---|---|
| S1a | 0.2167 | 0.1325 | +0.0842 | [+0.0575, +0.1125] | 0 | 0.0125 | True | supported | 4.67e-09 / 6.17e-08 / 1.8e-08 |
| S1b | 0.2167 | 0.1175 | +0.0992 | [+0.0717, +0.1283] | 0 | 0.0167 | True | supported | 8.02e-12 / 7.28e-11 / 1.08e-10 |
| S2 | 0.0810 | -0.0170 | +0.0980 | [-0.0123, +0.2077] | 0.04 | 0.05 | True | narrowly supported; sensitive to the seed-mean definition | 0.0608 / 0.0202 / 0.0919 |
| S3 | 0.3567 | 0.3425 | +0.0142 | [+0.0008, +0.0292] | 0 | 0.025 | True | supported | 0 / 0 / 0 |

Per seed (as phase95_compare.py; one-sided p, no Holm):

| run | S1a covered vs SFT | p | S1b covered vs uniform | p | S2 ρ split vs uniform | p | S3 acc split vs uniform | p |
|---|---|---|---|---|---|---|---|---|
| seed0 | 90 vs 53 | 4.67e-09 | 90 vs 47 | 8.02e-12 | 0.078 vs -0.017 | 0.0608 | 34.5% vs 34.2% | 0 |
| seed1 | 84 vs 53 | 6.17e-08 | 84 vs 47 | 7.28e-11 | 0.099 vs -0.017 | 0.0202 | 36.2% vs 34.2% | 0 |
| seed2 | 86 vs 53 | 1.8e-08 | 86 vs 47 | 1.08e-10 | 0.066 vs -0.017 | 0.0919 | 36.2% vs 34.2% | 0 |

Range across seeds: 1σ coverage 21.0%–22.5%, accuracy 34.5%–36.2%, ρ 0.066–0.099.

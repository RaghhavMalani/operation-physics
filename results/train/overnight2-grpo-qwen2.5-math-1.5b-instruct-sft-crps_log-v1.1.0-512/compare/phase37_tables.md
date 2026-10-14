| split | model | acc ≤1% | 1σ cov | 2σ cov | median rel. σ | σ = 0 | ρ(σ, err) | mean crps_log | mean reward |
|---|---|---|---|---|---|---|---|---|---|
| id | sft | 38.0% | 15.0% | 20.5% | 0.050% | 25.5% | -0.04 | -0.3199 | -0.2242 |
| id | rl_lastnight | 40.0% | 15.5% | 20.0% | 0.042% | 27.0% | -0.10 | -0.2750 | -0.1778 |
| id | posthoc_crps_fit | 38.0% | 35.0% | 40.5% | 0.908% | 25.5% | -0.04 | -0.3187 | -0.2230 |
| id | posthoc_coverage_matched | 38.0% | 65.0% | 69.5% | 29.268% | 25.5% | -0.04 | -0.5193 | -0.4237 |
| id | rl287 | 39.5% | 14.0% | 18.5% | 0.041% | 27.0% | -0.06 | -0.2822 | -0.1848 |
| id | rlfinal | 41.0% | 11.5% | 16.0% | 0.000% | 50.5% | 0.01 | -0.2390 | -0.1410 |
| depth_extrapolation | sft | 31.0% | 11.5% | 16.0% | 0.081% | 12.0% | 0.06 | -0.3573 | -0.2153 |
| depth_extrapolation | rl_lastnight | 31.0% | 10.0% | 15.0% | 0.064% | 13.5% | 0.07 | -0.3605 | -0.2205 |
| depth_extrapolation | posthoc_crps_fit | 31.0% | 33.5% | 38.0% | 1.463% | 12.0% | 0.06 | -0.3535 | -0.2115 |
| depth_extrapolation | posthoc_coverage_matched | 31.0% | 72.5% | 76.0% | 47.192% | 12.0% | 0.06 | -0.5056 | -0.3636 |
| depth_extrapolation | rl287 | 29.0% | 9.5% | 14.0% | 0.064% | 15.0% | 0.01 | -0.4367 | -0.2972 |
| depth_extrapolation | rlfinal | 31.0% | 8.1% | 13.1% | 0.026% | 35.9% | -0.12 | -0.3436 | -0.2028 |

| group | comparison | split | 1σ covered: before → after (only after / only before) | McNemar p | 2σ covered | McNemar p | strict ≤1% | McNemar p | crps_log Δ | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|---|
| primary | rl287_vs_sft | depth_extrapolation | 23 → 19 (0 / 4) | 0.125 | 32 → 28 | 0.125 | 62 → 58 | 0.289 | -0.0795 | 0.55 |
| primary | rl287_vs_sft | id | 30 → 28 (0 / 2) | 0.5 | 41 → 37 | 0.219 | 76 → 79 | 0.375 | +0.0377 | 0.115 |
| primary | rl287_vs_rl_lastnight | depth_extrapolation | 20 → 19 (2 / 3) | 1 | 30 → 28 | 0.5 | 62 → 58 | 0.219 | -0.0762 | 0.609 |
| primary | rl287_vs_rl_lastnight | id | 31 → 28 (0 / 3) | 0.25 | 40 → 37 | 0.508 | 80 → 79 | 1 | -0.0072 | 0.646 |
| primary | rl287_vs_posthoc_crps_fit | depth_extrapolation | 67 → 19 (0 / 48) | 7.11e-15 | 76 → 28 | 7.11e-15 | 62 → 58 | 0.289 | -0.0832 | 0.014 |
| primary | rl287_vs_posthoc_crps_fit | id | 70 → 28 (0 / 42) | 4.55e-13 | 81 → 37 | 1.14e-13 | 76 → 79 | 0.375 | +0.0365 | 0.873 |
| primary | rl287_vs_posthoc_coverage_matched | depth_extrapolation | 145 → 19 (0 / 126) | 2.35e-38 | 152 → 28 | 9.4e-38 | 62 → 58 | 0.289 | +0.0689 | 0.000277 |
| primary | rl287_vs_posthoc_coverage_matched | id | 130 → 28 (0 / 102) | 3.94e-31 | 139 → 37 | 3.94e-31 | 76 → 79 | 0.375 | +0.2372 | 2.99e-10 |
| secondary | rlfinal_vs_rl287 | depth_extrapolation | 19 → 16 (4 / 7) | 0.549 | 28 → 26 | 0.754 | 58 → 62 | 0.388 | +0.0932 | 0.448 |
| secondary | rlfinal_vs_rl287 | id | 28 → 23 (2 / 7) | 0.18 | 37 → 32 | 0.267 | 79 → 82 | 0.453 | +0.0432 | 0.986 |
| secondary | rlfinal_vs_sft | depth_extrapolation | 23 → 16 (3 / 10) | 0.0923 | 32 → 26 | 0.146 | 62 → 62 | 1 | +0.0137 | 0.936 |
| secondary | rlfinal_vs_sft | id | 30 → 23 (2 / 9) | 0.0654 | 41 → 32 | 0.0352 | 76 → 82 | 0.0703 | +0.0809 | 0.229 |
| secondary | rlfinal_vs_posthoc_crps_fit | depth_extrapolation | 67 → 16 (1 / 52) | 1.2e-14 | 76 → 26 | 1.65e-13 | 62 → 62 | 1 | +0.0100 | 0.763 |
| secondary | rlfinal_vs_posthoc_crps_fit | id | 70 → 23 (1 / 48) | 1.78e-13 | 81 → 32 | 4.62e-14 | 76 → 82 | 0.0703 | +0.0797 | 0.243 |

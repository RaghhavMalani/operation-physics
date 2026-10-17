| model | split | acc ≤1% | 1σ cov | 2σ cov | σ = 0 | median rel. σ | ρ(σ, err) | mean crps_log |
|---|---|---|---|---|---|---|---|---|
| sft | id | 39.5% | 14.0% | 18.5% | 17.5% | 0.0568% | 0.09 | -0.2371 |
| sft | depth_extrapolation | 34.0% | 10.0% | 15.5% | 16.1% | 0.0804% | 0.00 | -0.4294 |
| sft | pooled | 36.8% | 12.0% | 17.0% | 16.8% | 0.0631% | 0.05 | -0.3332 |
| posthoc | id | 39.5% | 34.0% | 40.5% | 17.5% | 1.0302% | 0.09 | -0.2337 |
| posthoc | depth_extrapolation | 34.0% | 34.5% | 44.0% | 16.1% | 1.4582% | 0.00 | -0.4236 |
| posthoc | pooled | 36.8% | 34.2% | 42.2% | 16.8% | 1.1445% | 0.05 | -0.3286 |
| sft_a5_strongest | id | 39.5% | 69.5% | 81.5% | 0.0% | 6.5201% | 0.77 | -0.2095 |
| sft_a5_strongest | depth_extrapolation | 34.0% | 66.0% | 75.0% | 0.0% | 5.5747% | 0.57 | -0.3944 |
| sft_a5_strongest | pooled | 36.8% | 67.8% | 78.2% | 0.0% | 6.1811% | 0.68 | -0.3020 |
| uniform_rl287 | id | 41.5% | 11.5% | 16.5% | 23.5% | 0.0426% | 0.05 | -0.2352 |
| uniform_rl287 | depth_extrapolation | 33.0% | 9.5% | 13.5% | 15.5% | 0.0619% | -0.04 | -0.4136 |
| uniform_rl287 | pooled | 37.2% | 10.5% | 15.0% | 19.5% | 0.0521% | 0.01 | -0.3244 |
| split_rl287 | id | 40.5% | 43.0% | 52.0% | 0.0% | 1.5457% | 0.05 | -0.2036 |
| split_rl287 | depth_extrapolation | 33.0% | 39.5% | 50.5% | 0.0% | 1.8237% | 0.01 | -0.4307 |
| split_rl287 | pooled | 36.8% | 41.2% | 51.2% | 0.0% | 1.6417% | 0.03 | -0.3171 |
| split_rl287_a5_strongest | id | 40.5% | 72.5% | 83.5% | 0.0% | 6.7374% | 0.78 | -0.1772 |
| split_rl287_a5_strongest | depth_extrapolation | 33.0% | 64.5% | 75.5% | 0.0% | 5.3736% | 0.55 | -0.4018 |
| split_rl287_a5_strongest | pooled | 36.8% | 68.5% | 79.5% | 0.0% | 6.2318% | 0.67 | -0.2895 |

| hypothesis | p (one-sided) | Holm threshold | supported |
|---|---|---|---|
| S3 | 0 | 0.0125 | True |
| S1b | 4.78e-35 | 0.0167 | True |
| S1a | 2.92e-32 | 0.025 | True |
| S2 | 0.396 | 0.05 | False |

N_vs_A5_strongest_same_completions (split-credit RL@287 stated σ vs the control; RL σ beats it: False)

| test | RL | control | 95% CI of RL − control | p (one-sided) | Holm threshold | supported |
|---|---|---|---|---|---|---|
| N1_cov1_error | 0.2702 | 0.0023 | [0.1879, 0.3025] | 1 | 0.0167 | False |
| N2_rho | 0.0340 | 0.6736 | [-0.7402, -0.5385] | 1 | 0.025 | False |
| N3_crps_log | -0.3171 | -0.2895 | [-0.0362, -0.0198] | 1 | 0.05 | False |

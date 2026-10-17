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
| split_rl287 | id | 42.5% | 45.0% | 55.0% | 0.0% | 1.5353% | 0.06 | -0.1897 |
| split_rl287 | depth_extrapolation | 34.5% | 40.0% | 48.0% | 0.5% | 1.6472% | 0.08 | -0.4233 |
| split_rl287 | pooled | 38.5% | 42.5% | 51.5% | 0.3% | 1.5866% | 0.07 | -0.3065 |
| split_rl287_a5_strongest | id | 42.5% | 73.0% | 82.5% | 0.0% | 6.6929% | 0.75 | -0.1643 |
| split_rl287_a5_strongest | depth_extrapolation | 34.5% | 63.0% | 73.0% | 0.5% | 6.3161% | 0.53 | -0.4013 |
| split_rl287_a5_strongest | pooled | 38.5% | 68.0% | 77.8% | 0.3% | 6.4155% | 0.64 | -0.2828 |

| hypothesis | p (one-sided) | Holm threshold | supported |
|---|---|---|---|
| S3 | 0 | 0.0125 | True |
| S1b | 9.62e-38 | 0.0167 | True |
| S1a | 9.41e-35 | 0.025 | True |
| S2 | 0.225 | 0.05 | False |

N_vs_A5_strongest_same_completions (split-credit RL@287 stated σ vs the control; RL σ beats it: False)

| test | RL | control | 95% CI of RL − control | p (one-sided) | Holm threshold | supported |
|---|---|---|---|---|---|---|
| N1_cov1_error | 0.2577 | 0.0027 | [0.1804, 0.2879] | 1 | 0.0167 | False |
| N2_rho | 0.0712 | 0.6444 | [-0.6791, -0.4673] | 1 | 0.025 | False |
| N3_crps_log | -0.3065 | -0.2828 | [-0.0310, -0.0168] | 1 | 0.05 | False |

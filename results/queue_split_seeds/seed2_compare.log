| model | split | acc ≤1% | 1σ cov | 2σ cov | σ = 0 | median rel. σ | ρ(σ, err) | mean crps_log |
|---|---|---|---|---|---|---|---|---|
| sft | id | 38.0% | 15.0% | 20.5% | 25.5% | 0.0500% | -0.04 | -0.3199 |
| sft | depth_extrapolation | 31.0% | 11.5% | 16.0% | 12.0% | 0.0807% | 0.06 | -0.3573 |
| sft | pooled | 34.5% | 13.2% | 18.2% | 18.8% | 0.0629% | 0.02 | -0.3386 |
| posthoc | id | 38.0% | 35.0% | 40.5% | 25.5% | 0.9076% | -0.04 | -0.3187 |
| posthoc | depth_extrapolation | 31.0% | 33.5% | 38.0% | 12.0% | 1.4634% | 0.06 | -0.3535 |
| posthoc | pooled | 34.5% | 34.2% | 39.2% | 18.8% | 1.1403% | 0.02 | -0.3361 |
| sft_text_scale | id | 38.0% | 42.5% | 53.5% | 0.0% | 2.4325% | 0.14 | -0.3143 |
| sft_text_scale | depth_extrapolation | 31.0% | 45.5% | 52.5% | 0.0% | 4.7606% | 0.22 | -0.3408 |
| sft_text_scale | pooled | 34.5% | 44.0% | 53.0% | 0.0% | 3.3156% | 0.18 | -0.3276 |
| sft_text_direct | id | 38.0% | 79.5% | 88.5% | 0.0% | 13.8003% | 0.66 | -0.2925 |
| sft_text_direct | depth_extrapolation | 31.0% | 77.0% | 86.0% | 0.0% | 15.7428% | 0.41 | -0.3320 |
| sft_text_direct | pooled | 34.5% | 78.2% | 87.2% | 0.0% | 14.9997% | 0.55 | -0.3123 |
| sft_oracle_cell | id | 38.0% | 31.0% | 39.5% | 0.0% | 0.1804% | 0.33 | -0.3065 |
| sft_oracle_cell | depth_extrapolation | 31.0% | 34.5% | 42.0% | 0.0% | 0.7677% | 0.29 | -0.3402 |
| sft_oracle_cell | pooled | 34.5% | 32.8% | 40.8% | 0.0% | 0.5620% | 0.33 | -0.3234 |
| sft_a5_strongest | id | 38.0% | 67.0% | 78.0% | 0.0% | 7.2536% | 0.78 | -0.2895 |
| sft_a5_strongest | depth_extrapolation | 31.0% | 66.5% | 75.5% | 0.0% | 6.5874% | 0.59 | -0.3207 |
| sft_a5_strongest | pooled | 34.5% | 66.8% | 76.8% | 0.0% | 6.7650% | 0.69 | -0.3051 |
| uniform_rl287 | id | 39.5% | 14.0% | 18.5% | 27.0% | 0.0412% | -0.06 | -0.2822 |
| uniform_rl287 | depth_extrapolation | 29.0% | 9.5% | 14.0% | 15.0% | 0.0642% | 0.01 | -0.4367 |
| uniform_rl287 | pooled | 34.2% | 11.8% | 16.2% | 21.0% | 0.0496% | -0.02 | -0.3594 |
| split_rl287 | id | 40.0% | 23.5% | 33.0% | 7.0% | 0.2122% | 0.09 | -0.2846 |
| split_rl287 | depth_extrapolation | 32.5% | 19.5% | 25.5% | 0.0% | 0.2429% | 0.02 | -0.3877 |
| split_rl287 | pooled | 36.2% | 21.5% | 29.2% | 3.5% | 0.2327% | 0.07 | -0.3361 |
| split_rl287_a5_strongest | id | 40.0% | 68.5% | 79.0% | 0.0% | 6.2338% | 0.76 | -0.2521 |
| split_rl287_a5_strongest | depth_extrapolation | 32.5% | 65.5% | 75.5% | 0.0% | 5.6424% | 0.54 | -0.3488 |
| split_rl287_a5_strongest | pooled | 36.2% | 67.0% | 77.2% | 0.0% | 6.0831% | 0.66 | -0.3005 |

| hypothesis | p (one-sided) | Holm threshold | supported |
|---|---|---|---|
| S3 | 0 | 0.0125 | True |
| S1b | 1.08e-10 | 0.0167 | True |
| S1a | 1.8e-08 | 0.025 | True |
| S2 | 0.0919 | 0.05 | False |

N_vs_A5_strongest_same_completions (split-credit RL@287 stated σ vs the control; RL σ beats it: False)

| test | RL | control | 95% CI of RL − control | p (one-sided) | Holm threshold | supported |
|---|---|---|---|---|---|---|
| N1_cov1_error | 0.4677 | 0.0127 | [0.4000, 0.4900] | 1 | 0.0167 | False |
| N2_rho | 0.0663 | 0.6614 | [-0.7041, -0.4920] | 1 | 0.025 | False |
| N3_crps_log | -0.3361 | -0.3005 | [-0.0459, -0.0264] | 1 | 0.05 | False |

superseded_D49_A_vs_text_scale (split-credit RL@287 stated σ vs the control; RL σ beats it: False)

| test | RL | control | 95% CI of RL − control | p (one-sided) | Holm threshold | supported |
|---|---|---|---|---|---|---|
| A1_cov1_error | 0.4677 | 0.2427 | [0.1775, 0.2725] | 1 | 0.05 | False |
| A2_rho | 0.0663 | 0.1823 | [-0.2311, -0.0003] | 0.976 | 0.025 | False |
| A3_crps_log | -0.3361 | -0.3276 | [-0.0594, 0.0432] | 0.629 | 0.0167 | False |

superseded_D49_sensitivity_A_vs_text_direct (split-credit RL@287 stated σ vs the control; RL σ beats it: False)

| test | RL | control | 95% CI of RL − control | p (one-sided) | Holm threshold | supported |
|---|---|---|---|---|---|---|
| A1_cov1_error | 0.4677 | 0.0998 | [0.3029, 0.4329] | 1 | 0.025 | False |
| A2_rho | 0.0663 | 0.5470 | [-0.5891, -0.3757] | 1 | 0.05 | False |
| A3_crps_log | -0.3361 | -0.3123 | [-0.0746, 0.0267] | 0.822 | 0.0167 | False |

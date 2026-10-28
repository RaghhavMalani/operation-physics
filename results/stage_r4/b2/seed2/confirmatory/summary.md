# Eval summary: stage-r4-b2-seed2-confirmatory

- model: `.cache/huggingface/hub/models--unsloth--Qwen2.5-Math-1.5B-Instruct-bnb-4bit/snapshots/e24e9530845f4e3a6648343796883cb509935039` @ `e24e9530845f4e3a6648343796883cb509935039`
- generator: `2.0.0+bdbb2279`, reward rule: `crps_log`, prompt: `v5_two_channel`
- generation time: 3582 s
- torch_max_allocated_gib: 3.35
- torch_max_reserved_gib: 4.13
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.09

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a6_pairs | 1020 | 12.2% | 0.0669 | 0.1% | 0.0% | 0.0% | 41.1% | 60.6% | 0.0475 | -0.125 | 0.0% |
| a6b_pairs | 1020 | 13.4% | 0.0564 | 0.0% | 0.0% | 0.0% | 44.3% | 62.9% | 0.0375 | -0.292 | 0.0% |
| a6c_pairs | 1020 | 13.1% | 0.0559 | 0.0% | 0.0% | 0.0% | 42.7% | 64.9% | 0.0475 | -0.103 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split (**not comparable here**: different generator or verifier version, so omitted).

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| a6_pairs | this model | 0.0% | 41.1% | 60.6% | -0.244 |
| a6_pairs | this model, σ × 18.1 |  | 92.9% | 95.6% | -0.387 |
| a6b_pairs | this model | 0.0% | 44.3% | 62.9% | -0.394 |
| a6b_pairs | this model, σ × 18.1 |  | 85.2% | 87.9% | -0.507 |
| a6c_pairs | this model | 0.0% | 42.7% | 64.9% | -0.21 |
| a6c_pairs | this model, σ × 18.1 |  | 93.1% | 95.7% | -0.321 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × (a) text scale | 96.3% | 97.9% | -0.873 |
| a6_pairs | this model × (a') text direct | 65.9% | 78.6% | -0.24 |
| a6_pairs | this model × (b) oracle per cell (upper bound) | 62.3% | 78.6% | -0.238 |
| a6b_pairs | this model × (a) text scale | 90.6% | 93.5% | -1.09 |
| a6b_pairs | this model × (a') text direct | 65.7% | 76.1% | -0.397 |
| a6b_pairs | this model × (b) oracle per cell (upper bound) | 62.3% | 76.0% | -0.385 |
| a6c_pairs | this model × (a) text scale | 96.5% | 97.6% | -0.735 |
| a6c_pairs | this model × (a') text direct | 69.8% | 84.1% | -0.209 |
| a6c_pairs | this model × (b) oracle per cell (upper bound) | 67.5% | 83.5% | -0.203 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × A5 σ | 39.6% | 51.4% | -0.24 |
| a6b_pairs | this model × A5 σ | 50.0% | 63.5% | -0.383 |
| a6c_pairs | this model × A5 σ | 50.3% | 64.2% | -0.209 |

## a6_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 340 | 20.6% | 35.0% | 68.6% | 0.00281 |
| 0.02 | 340 | 340 | 21.8% | 42.4% | 59.5% | -0.0718 |
| 0.05 | 340 | 339 | 21.5% | 46.0% | 71.2% | -0.0532 |

## a6_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 16.7% | 0.0509 | 0.0% | 0.0% | 49.5% |
| 2 | 204 | 11.8% | 0.061 | 0.5% | 0.0% | 40.9% |
| 3 | 204 | 8.8% | 0.0863 | 0.0% | 0.0% | 32.8% |
| 4 | 204 | 10.8% | 0.0714 | 0.0% | 0.0% | 42.6% |
| 5 | 204 | 12.7% | 0.0637 | 0.0% | 0.0% | 39.7% |

## a6_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 308 | 14.9% | 0.0494 | 0.3% | 0.0% | 46.6% |
| energy | 332 | 8.4% | 0.0928 | 0.0% | 0.0% | 33.7% |
| kinematics | 380 | 13.2% | 0.0698 | 0.0% | 0.0% | 43.2% |

Mean score by rule (failures at the rule's floor): crps -0.281, binary 0.122, log_score -1.65, abs_rel_error -0.302, crps_log -0.244
Mean step score: 0.742; copied-given rate: 0.2%; steps block present: 99.9%; mean completion tokens: 244

## a6b_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 340 | 9.7% | 45.3% | 81.8% | 0.0218 |
| 0.02 | 340 | 340 | 10.3% | 39.7% | 68.6% | 0.00916 |
| 0.05 | 340 | 340 | 8.5% | 47.9% | 72.4% | 0.011 |

## a6b_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 26.5% | 0.022 | 0.0% | 0.0% | 62.3% |
| 2 | 204 | 14.2% | 0.0486 | 0.0% | 0.0% | 49.5% |
| 3 | 204 | 11.8% | 0.0685 | 0.0% | 0.0% | 36.8% |
| 4 | 204 | 8.8% | 0.0827 | 0.0% | 0.0% | 41.2% |
| 5 | 204 | 5.9% | 0.135 | 0.0% | 0.0% | 31.9% |

## a6b_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 438 | 16.2% | 0.0426 | 0.0% | 0.0% | 50.5% |
| energy | 438 | 9.1% | 0.0932 | 0.0% | 0.0% | 32.9% |
| kinematics | 144 | 18.1% | 0.032 | 0.0% | 0.0% | 60.4% |

Mean score by rule (failures at the rule's floor): crps -0.269, binary 0.134, log_score -1.68, abs_rel_error -0.283, crps_log -0.394
Mean step score: 0.637; copied-given rate: 0.9%; steps block present: 100.0%; mean completion tokens: 228

## a6c_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.02 | 340 | 340 | 17.9% | 39.1% | 73.8% | -0.0224 |
| 0.05 | 680 | 680 | 14.0% | 44.6% | 60.0% | -0.000155 |

## a6c_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 19.1% | 0.0458 | 0.0% | 0.0% | 48.5% |
| 2 | 204 | 14.2% | 0.0557 | 0.0% | 0.0% | 39.7% |
| 3 | 204 | 12.7% | 0.0519 | 0.0% | 0.0% | 40.7% |
| 4 | 204 | 9.3% | 0.0699 | 0.0% | 0.0% | 42.6% |
| 5 | 204 | 10.3% | 0.0568 | 0.0% | 0.0% | 42.2% |

## a6c_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 320 | 17.5% | 0.0403 | 0.0% | 0.0% | 55.3% |
| energy | 338 | 6.5% | 0.0877 | 0.0% | 0.0% | 30.5% |
| kinematics | 362 | 15.5% | 0.05 | 0.0% | 0.0% | 43.1% |

Mean score by rule (failures at the rule's floor): crps -0.19, binary 0.131, log_score -1.2, abs_rel_error -0.207, crps_log -0.21
Mean step score: 0.667; copied-given rate: 0.2%; steps block present: 100.0%; mean completion tokens: 247

# Eval summary: stage-b-seed0-a6-sets

- model: `.cache/huggingface/hub/models--unsloth--Qwen2.5-Math-1.5B-Instruct-bnb-4bit/snapshots/e24e9530845f4e3a6648343796883cb509935039` @ `e24e9530845f4e3a6648343796883cb509935039`
- generator: `2.0.0+bdbb2279`, reward rule: `crps_log`, prompt: `v5_two_channel`
- generation time: 3181 s
- torch_max_allocated_gib: 5.52
- torch_max_reserved_gib: 6.72
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 7.30

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a6_pairs | 1020 | 11.5% | 0.0783 | 0.1% | 0.0% | 0.1% | 30.0% | 49.1% | 0.0329 | -0.168 | 0.0% |
| a6b_pairs | 1020 | 11.1% | 0.0583 | 0.0% | 0.0% | 0.0% | 35.5% | 54.3% | 0.027 | -0.306 | 0.0% |
| a6c_pairs | 1020 | 14.7% | 0.0485 | 0.2% | 0.0% | 0.0% | 31.7% | 49.5% | 0.0246 | -0.0888 | 0.1% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split (**not comparable here**: different generator or verifier version, so omitted).

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| a6_pairs | this model | 0.0% | 30.0% | 49.1% | -0.277 |
| a6_pairs | this model, σ × 18.1 |  | 89.7% | 94.1% | -0.356 |
| a6b_pairs | this model | 0.0% | 35.5% | 54.3% | -0.407 |
| a6b_pairs | this model, σ × 18.1 |  | 84.9% | 88.8% | -0.486 |
| a6c_pairs | this model | 0.0% | 31.7% | 49.5% | -0.194 |
| a6c_pairs | this model, σ × 18.1 |  | 91.7% | 96.2% | -0.235 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × (a) text scale | 95.4% | 97.6% | -0.706 |
| a6_pairs | this model × (a') text direct | 60.3% | 74.5% | -0.271 |
| a6_pairs | this model × (b) oracle per cell (upper bound) | 63.9% | 79.4% | -0.261 |
| a6b_pairs | this model × (a) text scale | 90.7% | 92.9% | -0.954 |
| a6b_pairs | this model × (a') text direct | 70.3% | 80.2% | -0.41 |
| a6b_pairs | this model × (b) oracle per cell (upper bound) | 58.3% | 73.4% | -0.397 |
| a6c_pairs | this model × (a) text scale | 97.2% | 98.0% | -0.496 |
| a6c_pairs | this model × (a') text direct | 73.0% | 83.4% | -0.202 |
| a6c_pairs | this model × (b) oracle per cell (upper bound) | 67.4% | 84.1% | -0.182 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × A5 σ | 37.1% | 47.5% | -0.273 |
| a6b_pairs | this model × A5 σ | 55.5% | 66.7% | -0.401 |
| a6c_pairs | this model × A5 σ | 47.9% | 61.6% | -0.193 |

## a6_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 339 | 20.3% | 24.5% | 47.8% | -0.256 |
| 0.02 | 340 | 340 | 21.2% | 29.4% | 48.6% | -0.204 |
| 0.05 | 340 | 340 | 21.5% | 36.2% | 56.2% | -0.114 |

## a6_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 16.2% | 0.0455 | 0.0% | 0.0% | 37.7% |
| 2 | 204 | 12.3% | 0.0887 | 0.0% | 0.0% | 28.9% |
| 3 | 204 | 8.3% | 0.0897 | 0.0% | 0.0% | 28.4% |
| 4 | 204 | 12.3% | 0.0775 | 0.0% | 0.0% | 30.9% |
| 5 | 204 | 8.3% | 0.102 | 0.5% | 0.0% | 24.1% |

## a6_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 318 | 15.1% | 0.0477 | 0.0% | 0.0% | 33.3% |
| energy | 330 | 7.9% | 0.104 | 0.3% | 0.0% | 21.3% |
| kinematics | 372 | 11.6% | 0.0872 | 0.0% | 0.0% | 34.9% |

Mean score by rule (failures at the rule's floor): crps -0.331, binary 0.115, log_score -2.76, abs_rel_error -0.351, crps_log -0.277
Mean step score: 0.684; copied-given rate: 0.6%; steps block present: 99.9%; mean completion tokens: 245

## a6b_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 340 | 9.1% | 29.7% | 41.9% | -0.104 |
| 0.02 | 340 | 340 | 8.8% | 32.1% | 50.0% | -0.124 |
| 0.05 | 340 | 340 | 8.5% | 44.7% | 62.1% | -0.0195 |

## a6b_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 16.2% | 0.0288 | 0.0% | 0.0% | 44.1% |
| 2 | 204 | 17.2% | 0.0468 | 0.0% | 0.0% | 40.7% |
| 3 | 204 | 11.3% | 0.0696 | 0.0% | 0.0% | 36.3% |
| 4 | 204 | 6.4% | 0.0993 | 0.0% | 0.0% | 30.4% |
| 5 | 204 | 4.4% | 0.116 | 0.0% | 0.0% | 26.0% |

## a6b_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 438 | 10.7% | 0.0489 | 0.0% | 0.0% | 37.0% |
| energy | 438 | 7.3% | 0.0827 | 0.0% | 0.0% | 29.9% |
| kinematics | 144 | 23.6% | 0.0326 | 0.0% | 0.0% | 47.9% |

Mean score by rule (failures at the rule's floor): crps -0.32, binary 0.111, log_score -2.28, abs_rel_error -0.334, crps_log -0.407
Mean step score: 0.629; copied-given rate: 0.9%; steps block present: 100.0%; mean completion tokens: 232

## a6c_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.02 | 340 | 338 | 24.7% | 31.7% | 58.3% | -0.194 |
| 0.05 | 680 | 680 | 13.7% | 31.8% | 57.0% | -0.237 |

## a6c_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 17.6% | 0.0443 | 1.0% | 0.0% | 34.2% |
| 2 | 204 | 17.6% | 0.0388 | 0.0% | 0.0% | 30.9% |
| 3 | 204 | 12.3% | 0.0651 | 0.0% | 0.0% | 30.9% |
| 4 | 204 | 16.2% | 0.0411 | 0.0% | 0.0% | 31.9% |
| 5 | 204 | 9.8% | 0.0602 | 0.0% | 0.0% | 30.9% |

## a6c_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 318 | 19.2% | 0.0381 | 0.0% | 0.0% | 36.8% |
| energy | 338 | 10.1% | 0.0766 | 0.0% | 0.0% | 23.4% |
| kinematics | 364 | 15.1% | 0.0424 | 0.5% | 0.0% | 35.1% |

Mean score by rule (failures at the rule's floor): crps -0.215, binary 0.147, log_score -2.2, abs_rel_error -0.228, crps_log -0.194
Mean step score: 0.659; copied-given rate: 0.4%; steps block present: 99.8%; mean completion tokens: 248

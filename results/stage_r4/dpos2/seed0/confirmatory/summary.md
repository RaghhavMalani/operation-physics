# Eval summary: stage-r4-dpos2-seed0-confirmatory

- model: `.cache/huggingface/hub/models--unsloth--Qwen2.5-Math-1.5B-Instruct-bnb-4bit/snapshots/e24e9530845f4e3a6648343796883cb509935039` @ `e24e9530845f4e3a6648343796883cb509935039`
- generator: `2.0.0+bdbb2279`, reward rule: `crps_log`, prompt: `v5_two_channel`
- generation time: 6873 s
- torch_max_allocated_gib: 3.56
- torch_max_reserved_gib: 4.49
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.83

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a6_pairs | 1020 | 12.3% | 0.0773 | 0.3% | 0.3% | 0.3% | 47.1% | 66.3% | 0.0582 | -0.173 | 0.0% |
| a6b_pairs | 1020 | 13.1% | 0.0615 | 0.0% | 0.1% | 0.0% | 42.1% | 61.3% | 0.0351 | -0.4 | 0.0% |
| a6c_pairs | 1020 | 12.7% | 0.0622 | 0.2% | 0.3% | 0.2% | 44.1% | 63.3% | 0.0498 | -0.158 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split (**not comparable here**: different generator or verifier version, so omitted).

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| a6_pairs | this model | 0.0% | 47.1% | 66.3% | -0.289 |
| a6_pairs | this model, σ × 18.1 |  | 92.3% | 95.7% | -0.529 |
| a6b_pairs | this model | 0.0% | 42.1% | 61.3% | -0.505 |
| a6b_pairs | this model, σ × 18.1 |  | 84.0% | 87.6% | -0.614 |
| a6c_pairs | this model | 0.0% | 44.1% | 63.3% | -0.26 |
| a6c_pairs | this model, σ × 18.1 |  | 92.7% | 96.5% | -0.424 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × (a) text scale | 96.2% | 97.7% | -1.17 |
| a6_pairs | this model × (a') text direct | 61.1% | 75.0% | -0.287 |
| a6_pairs | this model × (b) oracle per cell (upper bound) | 60.7% | 77.6% | -0.283 |
| a6b_pairs | this model × (a) text scale | 89.3% | 91.4% | -1.16 |
| a6b_pairs | this model × (a') text direct | 59.9% | 73.4% | -0.499 |
| a6b_pairs | this model × (b) oracle per cell (upper bound) | 63.2% | 74.8% | -0.497 |
| a6c_pairs | this model × (a) text scale | 97.4% | 98.3% | -0.941 |
| a6c_pairs | this model × (a') text direct | 65.3% | 81.7% | -0.253 |
| a6c_pairs | this model × (b) oracle per cell (upper bound) | 62.3% | 78.8% | -0.249 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × A5 σ | 34.2% | 48.1% | -0.289 |
| a6b_pairs | this model × A5 σ | 49.9% | 61.8% | -0.498 |
| a6c_pairs | this model × A5 σ | 44.5% | 59.3% | -0.257 |

## a6_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 338 | 18.5% | 37.9% | 73.0% | 0.00427 |
| 0.02 | 340 | 338 | 20.3% | 49.4% | 78.3% | -0.00345 |
| 0.05 | 340 | 337 | 21.2% | 54.3% | 72.2% | -0.00812 |

## a6_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 14.7% | 0.0593 | 0.0% | 0.0% | 53.4% |
| 2 | 204 | 11.8% | 0.0626 | 0.0% | 1.0% | 48.5% |
| 3 | 204 | 6.9% | 0.0813 | 0.5% | 0.0% | 43.3% |
| 4 | 204 | 14.2% | 0.0795 | 1.0% | 0.5% | 51.7% |
| 5 | 204 | 13.7% | 0.105 | 0.0% | 0.0% | 38.7% |

## a6_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 308 | 15.3% | 0.0493 | 0.0% | 0.6% | 60.1% |
| energy | 332 | 9.9% | 0.119 | 0.0% | 0.0% | 37.7% |
| kinematics | 380 | 11.8% | 0.0846 | 0.8% | 0.3% | 44.9% |

Mean score by rule (failures at the rule's floor): crps -0.308, binary 0.123, log_score -1.63, abs_rel_error -0.333, crps_log -0.289
Mean step score: 0.724; copied-given rate: 0.4%; steps block present: 99.7%; mean completion tokens: 431

## a6b_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 339 | 11.2% | 37.2% | 76.3% | -0.00247 |
| 0.02 | 340 | 340 | 10.9% | 41.2% | 62.2% | 0.000173 |
| 0.05 | 340 | 340 | 6.5% | 47.9% | 72.7% | 0.00462 |

## a6b_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 25.0% | 0.0263 | 0.0% | 0.0% | 55.4% |
| 2 | 204 | 12.3% | 0.0509 | 0.0% | 0.5% | 44.8% |
| 3 | 204 | 10.3% | 0.0679 | 0.0% | 0.0% | 40.2% |
| 4 | 204 | 10.3% | 0.0769 | 0.0% | 0.0% | 35.8% |
| 5 | 204 | 7.8% | 0.136 | 0.0% | 0.0% | 34.3% |

## a6b_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 438 | 14.4% | 0.0497 | 0.0% | 0.0% | 47.3% |
| energy | 438 | 10.5% | 0.0861 | 0.0% | 0.0% | 32.9% |
| kinematics | 144 | 17.4% | 0.0353 | 0.0% | 0.7% | 54.5% |

Mean score by rule (failures at the rule's floor): crps -0.273, binary 0.131, log_score -1.8, abs_rel_error -0.289, crps_log -0.505
Mean step score: 0.656; copied-given rate: 1.1%; steps block present: 100.0%; mean completion tokens: 301

## a6c_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.02 | 340 | 337 | 18.5% | 40.7% | 74.6% | -0.00581 |
| 0.05 | 680 | 676 | 11.5% | 46.0% | 62.8% | -0.00251 |

## a6c_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 18.1% | 0.0471 | 0.0% | 0.0% | 51.0% |
| 2 | 204 | 12.7% | 0.0647 | 1.0% | 0.5% | 40.3% |
| 3 | 204 | 15.7% | 0.0597 | 0.0% | 0.0% | 46.1% |
| 4 | 204 | 9.8% | 0.0712 | 0.0% | 1.0% | 44.1% |
| 5 | 204 | 7.4% | 0.0848 | 0.0% | 0.0% | 39.2% |

## a6c_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 320 | 16.2% | 0.0424 | 0.0% | 0.3% | 58.3% |
| energy | 338 | 8.0% | 0.0955 | 0.0% | 0.3% | 34.7% |
| kinematics | 362 | 14.1% | 0.0654 | 0.6% | 0.3% | 40.4% |

Mean score by rule (failures at the rule's floor): crps -0.251, binary 0.127, log_score -1.47, abs_rel_error -0.273, crps_log -0.26
Mean step score: 0.636; copied-given rate: 0.6%; steps block present: 99.8%; mean completion tokens: 426

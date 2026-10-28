# Eval summary: stage-r4-dpos2-seed1-confirmatory

- model: `.cache/huggingface/hub/models--unsloth--Qwen2.5-Math-1.5B-Instruct-bnb-4bit/snapshots/e24e9530845f4e3a6648343796883cb509935039` @ `e24e9530845f4e3a6648343796883cb509935039`
- generator: `2.0.0+bdbb2279`, reward rule: `crps_log`, prompt: `v5_two_channel`
- generation time: 6162 s
- torch_max_allocated_gib: 3.56
- torch_max_reserved_gib: 4.49
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.28

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a6_pairs | 1020 | 11.6% | 0.0763 | 0.2% | 0.0% | 0.2% | 43.2% | 64.1% | 0.0558 | -0.139 | 0.0% |
| a6b_pairs | 1020 | 13.0% | 0.0575 | 0.0% | 0.3% | 0.0% | 39.8% | 60.1% | 0.0312 | -0.276 | 0.0% |
| a6c_pairs | 1020 | 13.6% | 0.0633 | 0.2% | 0.1% | 0.2% | 39.4% | 59.4% | 0.0463 | -0.16 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split (**not comparable here**: different generator or verifier version, so omitted).

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| a6_pairs | this model | 0.1% | 43.2% | 64.1% | -0.256 |
| a6_pairs | this model, σ × 18.1 |  | 91.6% | 95.6% | -0.456 |
| a6b_pairs | this model | 0.1% | 39.8% | 60.1% | -0.383 |
| a6b_pairs | this model, σ × 18.1 |  | 85.6% | 89.9% | -0.463 |
| a6c_pairs | this model | 0.0% | 39.4% | 59.4% | -0.263 |
| a6c_pairs | this model, σ × 18.1 |  | 91.2% | 94.9% | -0.372 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × (a) text scale | 96.3% | 98.6% | -1.04 |
| a6_pairs | this model × (a') text direct | 61.3% | 76.3% | -0.252 |
| a6_pairs | this model × (b) oracle per cell (upper bound) | 61.8% | 76.4% | -0.25 |
| a6b_pairs | this model × (a) text scale | 90.9% | 93.1% | -0.948 |
| a6b_pairs | this model × (a') text direct | 61.0% | 74.2% | -0.385 |
| a6b_pairs | this model × (b) oracle per cell (upper bound) | 63.4% | 76.1% | -0.375 |
| a6c_pairs | this model × (a) text scale | 96.3% | 97.5% | -0.827 |
| a6c_pairs | this model × (a') text direct | 65.3% | 79.2% | -0.255 |
| a6c_pairs | this model × (b) oracle per cell (upper bound) | 65.4% | 79.5% | -0.246 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × A5 σ | 35.3% | 47.6% | -0.257 |
| a6b_pairs | this model × A5 σ | 50.1% | 64.4% | -0.373 |
| a6c_pairs | this model × A5 σ | 44.1% | 57.5% | -0.259 |

## a6_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 338 | 17.9% | 34.9% | 70.5% | -0.00591 |
| 0.02 | 340 | 339 | 20.6% | 46.6% | 75.7% | -0.00457 |
| 0.05 | 340 | 340 | 18.2% | 48.2% | 71.0% | -0.00694 |

## a6_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 16.7% | 0.0531 | 0.0% | 0.0% | 53.9% |
| 2 | 204 | 10.8% | 0.0607 | 0.0% | 0.0% | 45.1% |
| 3 | 204 | 7.4% | 0.091 | 0.0% | 0.0% | 40.7% |
| 4 | 204 | 8.8% | 0.0919 | 1.0% | 0.0% | 39.1% |
| 5 | 204 | 14.2% | 0.0733 | 0.0% | 0.0% | 37.3% |

## a6_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 308 | 15.6% | 0.0479 | 0.0% | 0.0% | 57.1% |
| energy | 332 | 7.2% | 0.112 | 0.0% | 0.0% | 35.2% |
| kinematics | 380 | 12.1% | 0.0881 | 0.5% | 0.0% | 38.9% |

Mean score by rule (failures at the rule's floor): crps -0.33, binary 0.116, log_score -1.66, abs_rel_error -0.357, crps_log -0.256
Mean step score: 0.732; copied-given rate: 0.5%; steps block present: 99.8%; mean completion tokens: 430

## a6b_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 339 | 9.7% | 36.3% | 66.7% | -0.00959 |
| 0.02 | 340 | 339 | 10.0% | 38.1% | 55.9% | -0.00435 |
| 0.05 | 340 | 339 | 8.8% | 45.1% | 63.3% | -0.00239 |

## a6b_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 26.0% | 0.0211 | 0.0% | 0.0% | 61.3% |
| 2 | 204 | 13.7% | 0.0476 | 0.0% | 0.5% | 44.3% |
| 3 | 204 | 11.3% | 0.063 | 0.0% | 0.5% | 36.5% |
| 4 | 204 | 7.8% | 0.0757 | 0.0% | 0.5% | 29.6% |
| 5 | 204 | 6.4% | 0.136 | 0.0% | 0.0% | 27.5% |

## a6b_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 438 | 14.8% | 0.0463 | 0.0% | 0.7% | 43.4% |
| energy | 438 | 9.1% | 0.0768 | 0.0% | 0.0% | 31.1% |
| kinematics | 144 | 19.4% | 0.0378 | 0.0% | 0.0% | 55.6% |

Mean score by rule (failures at the rule's floor): crps -0.273, binary 0.13, log_score -1.87, abs_rel_error -0.287, crps_log -0.383
Mean step score: 0.668; copied-given rate: 0.8%; steps block present: 100.0%; mean completion tokens: 300

## a6c_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.02 | 340 | 338 | 17.6% | 37.3% | 73.3% | -0.00923 |
| 0.05 | 680 | 679 | 14.3% | 40.5% | 57.7% | -0.0115 |

## a6c_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 21.1% | 0.0449 | 0.0% | 0.0% | 50.0% |
| 2 | 204 | 13.2% | 0.0648 | 1.0% | 0.5% | 37.3% |
| 3 | 204 | 14.2% | 0.0569 | 0.0% | 0.0% | 41.7% |
| 4 | 204 | 11.8% | 0.0722 | 0.0% | 0.0% | 42.2% |
| 5 | 204 | 7.8% | 0.121 | 0.0% | 0.0% | 26.0% |

## a6c_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 320 | 16.9% | 0.0464 | 0.0% | 0.3% | 50.8% |
| energy | 338 | 9.5% | 0.104 | 0.0% | 0.0% | 29.6% |
| kinematics | 362 | 14.6% | 0.0617 | 0.6% | 0.0% | 38.6% |

Mean score by rule (failures at the rule's floor): crps -0.287, binary 0.136, log_score -1.85, abs_rel_error -0.308, crps_log -0.263
Mean step score: 0.642; copied-given rate: 0.5%; steps block present: 99.8%; mean completion tokens: 421

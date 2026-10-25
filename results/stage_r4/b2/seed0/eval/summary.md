# Eval summary: stage-r4-b2-seed0-a6-sets

- model: `.cache/huggingface/hub/models--unsloth--Qwen2.5-Math-1.5B-Instruct-bnb-4bit/snapshots/e24e9530845f4e3a6648343796883cb509935039` @ `e24e9530845f4e3a6648343796883cb509935039`
- generator: `2.0.0+bdbb2279`, reward rule: `crps_log`, prompt: `v5_two_channel`
- generation time: 3807 s
- torch_max_allocated_gib: 3.37
- torch_max_reserved_gib: 4.13
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.79

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a6_pairs | 1020 | 10.7% | 0.071 | 0.0% | 0.0% | 0.0% | 32.7% | 51.4% | 0.0356 | -0.134 | 0.0% |
| a6b_pairs | 1020 | 12.5% | 0.0542 | 0.0% | 0.0% | 0.0% | 37.1% | 57.4% | 0.0299 | -0.24 | 0.0% |
| a6c_pairs | 1020 | 14.0% | 0.0466 | 0.0% | 0.0% | 0.0% | 42.6% | 61.8% | 0.0375 | -0.0413 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split (**not comparable here**: different generator or verifier version, so omitted).

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| a6_pairs | this model | 0.0% | 32.7% | 51.4% | -0.248 |
| a6_pairs | this model, σ × 18.1 |  | 88.4% | 93.5% | -0.346 |
| a6b_pairs | this model | 0.0% | 37.1% | 57.4% | -0.342 |
| a6b_pairs | this model, σ × 18.1 |  | 86.6% | 90.8% | -0.466 |
| a6c_pairs | this model | 0.0% | 42.6% | 61.8% | -0.15 |
| a6c_pairs | this model, σ × 18.1 |  | 92.9% | 96.0% | -0.253 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × (a) text scale | 95.3% | 98.1% | -0.698 |
| a6_pairs | this model × (a') text direct | 60.4% | 75.4% | -0.24 |
| a6_pairs | this model × (b) oracle per cell (upper bound) | 60.6% | 76.0% | -0.236 |
| a6b_pairs | this model × (a) text scale | 93.0% | 94.4% | -0.953 |
| a6b_pairs | this model × (a') text direct | 72.3% | 82.1% | -0.348 |
| a6b_pairs | this model × (b) oracle per cell (upper bound) | 53.6% | 70.7% | -0.336 |
| a6c_pairs | this model × (a) text scale | 97.9% | 99.6% | -0.653 |
| a6c_pairs | this model × (a') text direct | 72.3% | 84.2% | -0.163 |
| a6c_pairs | this model × (b) oracle per cell (upper bound) | 63.2% | 79.1% | -0.145 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × A5 σ | 36.3% | 47.1% | -0.239 |
| a6b_pairs | this model × A5 σ | 56.0% | 67.9% | -0.332 |
| a6c_pairs | this model × A5 σ | 47.4% | 61.2% | -0.151 |

## a6_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 340 | 18.5% | 31.8% | 61.9% | -0.0819 |
| 0.02 | 340 | 340 | 19.7% | 27.9% | 46.3% | -0.16 |
| 0.05 | 340 | 340 | 22.1% | 38.5% | 53.3% | -0.0843 |

## a6_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 14.2% | 0.045 | 0.0% | 0.0% | 39.7% |
| 2 | 204 | 10.8% | 0.0832 | 0.0% | 0.0% | 28.4% |
| 3 | 204 | 7.4% | 0.0653 | 0.0% | 0.0% | 30.9% |
| 4 | 204 | 12.3% | 0.0752 | 0.0% | 0.0% | 33.3% |
| 5 | 204 | 8.8% | 0.0967 | 0.0% | 0.0% | 31.4% |

## a6_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 318 | 13.2% | 0.049 | 0.0% | 0.0% | 38.7% |
| energy | 330 | 7.3% | 0.117 | 0.0% | 0.0% | 22.1% |
| kinematics | 372 | 11.6% | 0.0711 | 0.0% | 0.0% | 37.1% |

Mean score by rule (failures at the rule's floor): crps -0.301, binary 0.107, log_score -2.46, abs_rel_error -0.321, crps_log -0.248
Mean step score: 0.713; copied-given rate: 0.5%; steps block present: 100.0%; mean completion tokens: 244

## a6b_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 340 | 6.8% | 33.8% | 43.5% | -0.0046 |
| 0.02 | 340 | 340 | 7.6% | 35.0% | 46.2% | -0.137 |
| 0.05 | 340 | 340 | 7.9% | 42.4% | 66.7% | -0.114 |

## a6b_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 15.7% | 0.0295 | 0.0% | 0.0% | 44.6% |
| 2 | 204 | 21.1% | 0.0454 | 0.0% | 0.0% | 41.7% |
| 3 | 204 | 11.3% | 0.063 | 0.0% | 0.0% | 39.2% |
| 4 | 204 | 6.4% | 0.0839 | 0.0% | 0.0% | 32.8% |
| 5 | 204 | 8.3% | 0.106 | 0.0% | 0.0% | 27.0% |

## a6b_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 438 | 14.2% | 0.0439 | 0.0% | 0.0% | 41.3% |
| energy | 438 | 8.2% | 0.0958 | 0.0% | 0.0% | 29.0% |
| kinematics | 144 | 20.8% | 0.0341 | 0.0% | 0.0% | 48.6% |

Mean score by rule (failures at the rule's floor): crps -0.293, binary 0.125, log_score -1.86, abs_rel_error -0.306, crps_log -0.342
Mean step score: 0.64; copied-given rate: 0.7%; steps block present: 100.0%; mean completion tokens: 232

## a6c_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.02 | 340 | 340 | 23.2% | 42.4% | 51.9% | -0.094 |
| 0.05 | 680 | 680 | 15.1% | 42.8% | 68.0% | -0.0658 |

## a6c_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 14.7% | 0.043 | 0.0% | 0.0% | 43.1% |
| 2 | 204 | 20.1% | 0.0415 | 0.0% | 0.0% | 43.1% |
| 3 | 204 | 13.7% | 0.0627 | 0.0% | 0.0% | 44.1% |
| 4 | 204 | 10.8% | 0.0472 | 0.0% | 0.0% | 42.2% |
| 5 | 204 | 10.8% | 0.0525 | 0.0% | 0.0% | 40.7% |

## a6c_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 318 | 18.9% | 0.0363 | 0.0% | 0.0% | 53.1% |
| energy | 338 | 7.1% | 0.0812 | 0.0% | 0.0% | 31.4% |
| kinematics | 364 | 16.2% | 0.0381 | 0.0% | 0.0% | 44.0% |

Mean score by rule (failures at the rule's floor): crps -0.149, binary 0.14, log_score -1.45, abs_rel_error -0.163, crps_log -0.15
Mean step score: 0.679; copied-given rate: 0.3%; steps block present: 100.0%; mean completion tokens: 248

# Eval summary: stage-r4-b2-seed0-confirmatory

- model: `.cache/huggingface/hub/models--unsloth--Qwen2.5-Math-1.5B-Instruct-bnb-4bit/snapshots/e24e9530845f4e3a6648343796883cb509935039` @ `e24e9530845f4e3a6648343796883cb509935039`
- generator: `2.0.0+bdbb2279`, reward rule: `crps_log`, prompt: `v5_two_channel`
- generation time: 3501 s
- torch_max_allocated_gib: 3.35
- torch_max_reserved_gib: 4.13
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.47

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a6_pairs | 1020 | 11.6% | 0.0678 | 0.1% | 0.0% | 0.1% | 35.7% | 53.1% | 0.0357 | -0.114 | 0.0% |
| a6b_pairs | 1020 | 13.5% | 0.053 | 0.0% | 0.0% | 0.0% | 39.7% | 58.3% | 0.0288 | -0.222 | 0.0% |
| a6c_pairs | 1020 | 14.4% | 0.0556 | 0.2% | 0.0% | 0.2% | 38.4% | 59.7% | 0.0348 | -0.047 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split (**not comparable here**: different generator or verifier version, so omitted).

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| a6_pairs | this model | 0.0% | 35.7% | 53.1% | -0.235 |
| a6_pairs | this model, σ × 18.1 |  | 90.0% | 94.1% | -0.339 |
| a6b_pairs | this model | 0.0% | 39.7% | 58.3% | -0.328 |
| a6b_pairs | this model, σ × 18.1 |  | 87.7% | 90.7% | -0.431 |
| a6c_pairs | this model | 0.0% | 38.4% | 59.7% | -0.155 |
| a6c_pairs | this model, σ × 18.1 |  | 93.8% | 97.2% | -0.257 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × (a) text scale | 95.4% | 97.4% | -0.728 |
| a6_pairs | this model × (a') text direct | 64.2% | 77.6% | -0.226 |
| a6_pairs | this model × (b) oracle per cell (upper bound) | 60.5% | 77.1% | -0.225 |
| a6b_pairs | this model × (a) text scale | 92.8% | 94.8% | -0.874 |
| a6b_pairs | this model × (a') text direct | 67.2% | 78.9% | -0.34 |
| a6b_pairs | this model × (b) oracle per cell (upper bound) | 58.8% | 74.3% | -0.321 |
| a6c_pairs | this model × (a) text scale | 98.7% | 99.2% | -0.608 |
| a6c_pairs | this model × (a') text direct | 70.4% | 85.5% | -0.155 |
| a6c_pairs | this model × (b) oracle per cell (upper bound) | 65.1% | 81.1% | -0.145 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × A5 σ | 38.0% | 50.6% | -0.226 |
| a6b_pairs | this model × A5 σ | 52.5% | 65.9% | -0.318 |
| a6c_pairs | this model × A5 σ | 49.5% | 65.7% | -0.151 |

## a6_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 339 | 20.3% | 32.7% | 62.3% | -0.0807 |
| 0.02 | 340 | 340 | 22.1% | 35.3% | 53.3% | -0.147 |
| 0.05 | 340 | 340 | 20.6% | 39.1% | 54.3% | -0.128 |

## a6_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 16.2% | 0.0502 | 0.0% | 0.0% | 43.1% |
| 2 | 204 | 12.7% | 0.06 | 0.0% | 0.0% | 38.7% |
| 3 | 204 | 9.3% | 0.0778 | 0.0% | 0.0% | 32.8% |
| 4 | 204 | 9.3% | 0.0785 | 0.0% | 0.0% | 32.8% |
| 5 | 204 | 10.3% | 0.0689 | 0.5% | 0.0% | 31.0% |

## a6_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 308 | 16.6% | 0.0461 | 0.0% | 0.0% | 41.6% |
| energy | 332 | 6.6% | 0.0961 | 0.3% | 0.0% | 25.7% |
| kinematics | 380 | 11.8% | 0.0734 | 0.0% | 0.0% | 39.7% |

Mean score by rule (failures at the rule's floor): crps -0.272, binary 0.116, log_score -2.18, abs_rel_error -0.291, crps_log -0.235
Mean step score: 0.753; copied-given rate: 0.6%; steps block present: 99.9%; mean completion tokens: 245

## a6b_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 340 | 9.7% | 39.1% | 66.7% | 0.0123 |
| 0.02 | 340 | 340 | 10.6% | 37.9% | 50.0% | -0.131 |
| 0.05 | 340 | 340 | 8.5% | 42.1% | 62.1% | -0.121 |

## a6b_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 25.5% | 0.0208 | 0.0% | 0.0% | 52.9% |
| 2 | 204 | 14.7% | 0.0441 | 0.0% | 0.0% | 43.6% |
| 3 | 204 | 8.8% | 0.0622 | 0.0% | 0.0% | 37.3% |
| 4 | 204 | 10.3% | 0.0839 | 0.0% | 0.0% | 33.3% |
| 5 | 204 | 8.3% | 0.0908 | 0.0% | 0.0% | 31.4% |

## a6b_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 438 | 16.9% | 0.0404 | 0.0% | 0.0% | 45.0% |
| energy | 438 | 9.1% | 0.0787 | 0.0% | 0.0% | 31.5% |
| kinematics | 144 | 16.7% | 0.0353 | 0.0% | 0.0% | 48.6% |

Mean score by rule (failures at the rule's floor): crps -0.26, binary 0.135, log_score -1.79, abs_rel_error -0.271, crps_log -0.328
Mean step score: 0.662; copied-given rate: 0.9%; steps block present: 100.0%; mean completion tokens: 229

## a6c_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.02 | 340 | 338 | 18.8% | 39.1% | 68.8% | -0.078 |
| 0.05 | 680 | 680 | 11.8% | 38.1% | 55.0% | -0.0594 |

## a6c_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 20.1% | 0.042 | 1.0% | 0.0% | 46.5% |
| 2 | 204 | 14.7% | 0.0563 | 0.0% | 0.0% | 34.3% |
| 3 | 204 | 16.7% | 0.0595 | 0.0% | 0.0% | 39.7% |
| 4 | 204 | 10.3% | 0.063 | 0.0% | 0.0% | 37.3% |
| 5 | 204 | 10.3% | 0.0658 | 0.0% | 0.0% | 34.3% |

## a6c_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 320 | 17.5% | 0.0447 | 0.0% | 0.0% | 47.2% |
| energy | 338 | 7.4% | 0.082 | 0.6% | 0.0% | 26.8% |
| kinematics | 362 | 18.2% | 0.0485 | 0.0% | 0.0% | 41.4% |

Mean score by rule (failures at the rule's floor): crps -0.178, binary 0.144, log_score -1.46, abs_rel_error -0.196, crps_log -0.155
Mean step score: 0.673; copied-given rate: 0.4%; steps block present: 99.8%; mean completion tokens: 248

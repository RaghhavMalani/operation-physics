# Eval summary: stage-b-seed1-a6-sets

- model: `.cache/huggingface/hub/models--unsloth--Qwen2.5-Math-1.5B-Instruct-bnb-4bit/snapshots/e24e9530845f4e3a6648343796883cb509935039` @ `e24e9530845f4e3a6648343796883cb509935039`
- generator: `2.0.0+bdbb2279`, reward rule: `crps_log`, prompt: `v5_two_channel`
- generation time: 3202 s
- torch_max_allocated_gib: 5.52
- torch_max_reserved_gib: 6.72
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 7.30

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a6_pairs | 1020 | 11.0% | 0.0796 | 0.0% | 0.0% | 0.0% | 27.1% | 44.7% | 0.0279 | -0.171 | 0.0% |
| a6b_pairs | 1020 | 11.6% | 0.0653 | 0.0% | 0.0% | 0.0% | 31.3% | 50.7% | 0.0248 | -0.385 | 0.0% |
| a6c_pairs | 1020 | 14.7% | 0.0507 | 0.0% | 0.0% | 0.0% | 30.9% | 49.4% | 0.0234 | -0.0974 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split (**not comparable here**: different generator or verifier version, so omitted).

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| a6_pairs | this model | 0.0% | 27.1% | 44.7% | -0.282 |
| a6_pairs | this model, σ × 18.1 |  | 86.8% | 92.3% | -0.34 |
| a6b_pairs | this model | 0.0% | 31.3% | 50.7% | -0.481 |
| a6b_pairs | this model, σ × 18.1 |  | 82.3% | 86.2% | -0.537 |
| a6c_pairs | this model | 0.0% | 30.9% | 49.4% | -0.203 |
| a6c_pairs | this model, σ × 18.1 |  | 89.9% | 94.5% | -0.236 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × (a) text scale | 94.7% | 96.6% | -0.646 |
| a6_pairs | this model × (a') text direct | 60.0% | 74.4% | -0.273 |
| a6_pairs | this model × (b) oracle per cell (upper bound) | 61.0% | 77.5% | -0.268 |
| a6b_pairs | this model × (a) text scale | 88.0% | 90.9% | -0.92 |
| a6b_pairs | this model × (a') text direct | 68.4% | 77.6% | -0.478 |
| a6b_pairs | this model × (b) oracle per cell (upper bound) | 59.4% | 73.1% | -0.468 |
| a6c_pairs | this model × (a) text scale | 96.7% | 97.7% | -0.469 |
| a6c_pairs | this model × (a') text direct | 70.3% | 83.1% | -0.209 |
| a6c_pairs | this model × (b) oracle per cell (upper bound) | 68.3% | 82.8% | -0.189 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × A5 σ | 36.6% | 46.5% | -0.274 |
| a6b_pairs | this model × A5 σ | 55.3% | 66.3% | -0.47 |
| a6c_pairs | this model × A5 σ | 45.1% | 59.7% | -0.197 |

## a6_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 340 | 21.2% | 24.1% | 38.9% | -0.297 |
| 0.02 | 340 | 340 | 20.6% | 25.9% | 44.3% | -0.189 |
| 0.05 | 340 | 340 | 18.5% | 31.2% | 49.2% | -0.183 |

## a6_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 14.2% | 0.0462 | 0.0% | 0.0% | 29.4% |
| 2 | 204 | 10.3% | 0.0844 | 0.0% | 0.0% | 27.9% |
| 3 | 204 | 8.3% | 0.0719 | 0.0% | 0.0% | 27.5% |
| 4 | 204 | 12.3% | 0.0834 | 0.0% | 0.0% | 27.0% |
| 5 | 204 | 9.8% | 0.113 | 0.0% | 0.0% | 23.5% |

## a6_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 318 | 14.8% | 0.0545 | 0.0% | 0.0% | 30.2% |
| energy | 330 | 7.6% | 0.107 | 0.0% | 0.0% | 18.2% |
| kinematics | 372 | 10.8% | 0.0825 | 0.0% | 0.0% | 32.3% |

Mean score by rule (failures at the rule's floor): crps -0.302, binary 0.11, log_score -2.98, abs_rel_error -0.32, crps_log -0.282
Mean step score: 0.695; copied-given rate: 0.2%; steps block present: 100.0%; mean completion tokens: 245

## a6b_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 340 | 7.4% | 25.6% | 44.0% | -0.124 |
| 0.02 | 340 | 340 | 8.8% | 29.7% | 46.7% | -0.135 |
| 0.05 | 340 | 340 | 6.5% | 38.5% | 63.6% | -0.0791 |

## a6b_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 17.6% | 0.028 | 0.0% | 0.0% | 41.2% |
| 2 | 204 | 13.7% | 0.0508 | 0.0% | 0.0% | 34.8% |
| 3 | 204 | 9.8% | 0.0856 | 0.0% | 0.0% | 29.4% |
| 4 | 204 | 8.8% | 0.0928 | 0.0% | 0.0% | 28.4% |
| 5 | 204 | 7.8% | 0.116 | 0.0% | 0.0% | 22.5% |

## a6b_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 438 | 12.3% | 0.0528 | 0.0% | 0.0% | 34.5% |
| energy | 438 | 8.0% | 0.101 | 0.0% | 0.0% | 23.3% |
| kinematics | 144 | 20.1% | 0.0296 | 0.0% | 0.0% | 45.8% |

Mean score by rule (failures at the rule's floor): crps -0.281, binary 0.116, log_score -2.51, abs_rel_error -0.294, crps_log -0.481
Mean step score: 0.597; copied-given rate: 0.8%; steps block present: 100.0%; mean completion tokens: 232

## a6c_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.02 | 340 | 340 | 25.3% | 33.2% | 50.0% | -0.21 |
| 0.05 | 680 | 680 | 15.4% | 29.7% | 50.5% | -0.269 |

## a6c_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 19.6% | 0.0394 | 0.0% | 0.0% | 36.3% |
| 2 | 204 | 21.6% | 0.0398 | 0.0% | 0.0% | 32.4% |
| 3 | 204 | 11.8% | 0.0552 | 0.0% | 0.0% | 27.9% |
| 4 | 204 | 12.3% | 0.0616 | 0.0% | 0.0% | 28.4% |
| 5 | 204 | 8.3% | 0.0617 | 0.0% | 0.0% | 29.4% |

## a6c_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 318 | 18.9% | 0.0371 | 0.0% | 0.0% | 37.7% |
| energy | 338 | 9.2% | 0.0862 | 0.0% | 0.0% | 22.5% |
| kinematics | 364 | 16.2% | 0.04 | 0.0% | 0.0% | 32.7% |

Mean score by rule (failures at the rule's floor): crps -0.186, binary 0.147, log_score -2.33, abs_rel_error -0.198, crps_log -0.203
Mean step score: 0.659; copied-given rate: 0.3%; steps block present: 100.0%; mean completion tokens: 249

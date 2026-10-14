# Eval summary: overnight2-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-compare-rl287

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 659 s
- torch_max_allocated_gib: 4.48
- torch_max_reserved_gib: 4.76
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.06

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 39.5% | 0.0282 | 0.0% | 0.0% | 0.0% | 14.0% | 18.5% | 0.000412 | -0.185 | 0.0% |
| depth_extrapolation | 200 | 29.0% | 0.0407 | 0.0% | 0.0% | 0.0% | 9.5% | 14.0% | 0.000642 | -0.297 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split.

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| id | this model | – | 14.0% | 18.5% | -0.282 |
| id | this model, σ × 18.1 |  | 32.0% | 39.5% | -0.281 |
| id | SFT-only |  | 15.0% | 20.5% | -0.32 |
| id | SFT-only, σ × 18.1 (baseline) |  | 35.0% | 40.5% | -0.319 |
| depth_extrapolation | this model | – | 9.5% | 14.0% | -0.437 |
| depth_extrapolation | this model, σ × 18.1 |  | 31.0% | 37.5% | -0.432 |
| depth_extrapolation | SFT-only |  | 11.5% | 16.0% | -0.357 |
| depth_extrapolation | SFT-only, σ × 18.1 (baseline) |  | 33.5% | 38.0% | -0.354 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| depth_extrapolation | this model × (a) text scale | 45.0% | 52.0% | -0.422 |
| depth_extrapolation | this model × (a') text direct | 77.0% | 83.5% | -0.413 |
| depth_extrapolation | this model × (b) oracle per cell (upper bound) | 35.0% | 42.5% | -0.42 |
| depth_extrapolation | SFT-only × (a) text scale (**control**) | 45.5% | 52.5% | -0.341 |
| depth_extrapolation | SFT-only × (a') text direct | 77.0% | 86.0% | -0.332 |
| depth_extrapolation | SFT-only × (b) oracle per cell (upper bound) | 34.5% | 42.0% | -0.34 |
| id | this model × (a) text scale | 42.0% | 52.0% | -0.276 |
| id | this model × (a') text direct | 79.5% | 89.5% | -0.257 |
| id | this model × (b) oracle per cell (upper bound) | 30.5% | 39.0% | -0.269 |
| id | SFT-only × (a) text scale (**control**) | 42.5% | 53.5% | -0.314 |
| id | SFT-only × (a') text direct | 79.5% | 88.5% | -0.293 |
| id | SFT-only × (b) oracle per cell (upper bound) | 31.0% | 39.5% | -0.306 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| id | this model × A5 σ | 68.8% | 80.9% | -0.254 |
| id | SFT-only × A5 σ | 67.3% | 77.9% | -0.29 |
| depth_extrapolation | this model × A5 σ | 64.3% | 74.9% | -0.399 |
| depth_extrapolation | SFT-only × A5 σ | 66.5% | 75.5% | -0.321 |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 68.0% | 0.00195 | 0.0% | 0.0% | 30.0% |
| 2 | 50 | 32.0% | 0.0305 | 0.0% | 0.0% | 6.0% |
| 3 | 50 | 34.0% | 0.0585 | 0.0% | 0.0% | 10.0% |
| 4 | 50 | 24.0% | 0.0643 | 0.0% | 0.0% | 10.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 63.2% | 0.00322 | 0.0% | 0.0% | 22.1% |
| energy | 64 | 14.1% | 0.0713 | 0.0% | 0.0% | 4.7% |
| kinematics | 68 | 39.7% | 0.0256 | 0.0% | 0.0% | 14.7% |

Mean score by rule (failures at the rule's floor): crps -0.343, binary 0.395, log_score -4.92, abs_rel_error -0.344, crps_log -0.282
Mean step score: 0.649; copied-given rate: 0.5%; steps block present: 100.0%; mean completion tokens: 195

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 29.0% | 0.0407 | 0.0% | 0.0% | 9.5% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 41.8% | 0.0125 | 0.0% | 0.0% | 13.4% |
| energy | 66 | 1.5% | 0.266 | 0.0% | 0.0% | 0.0% |
| kinematics | 67 | 43.3% | 0.0153 | 0.0% | 0.0% | 14.9% |

Mean score by rule (failures at the rule's floor): crps -0.459, binary 0.29, log_score -6.15, abs_rel_error -0.46, crps_log -0.437
Mean step score: 0.698; copied-given rate: 0.5%; steps block present: 100.0%; mean completion tokens: 317

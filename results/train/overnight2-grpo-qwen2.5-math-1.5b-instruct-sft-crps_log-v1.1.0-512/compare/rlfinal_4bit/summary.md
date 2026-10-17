# Eval summary: overnight2-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-compare-rlfinal_4bit

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 505 s
- torch_max_allocated_gib: 3.11
- torch_max_reserved_gib: 3.56
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 3.70

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 42.5% | 0.0214 | 0.0% | 0.0% | 0.0% | 5.5% | 6.0% | 0 | -0.0213 | 0.0% |
| depth_extrapolation | 200 | 34.0% | 0.0463 | 0.5% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | -0.213 | 0.5% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split.

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| id | this model | 97.5% | 5.5% | 6.0% | -0.121 |
| id | this model, σ × 18.1 |  | 6.0% | 6.0% | -0.121 |
| id | SFT-only |  | 15.0% | 20.5% | -0.32 |
| id | SFT-only, σ × 18.1 (baseline) |  | 35.0% | 40.5% | -0.319 |
| depth_extrapolation | this model | 97.5% | 0.0% | 0.0% | -0.348 |
| depth_extrapolation | this model, σ × 18.1 |  | 0.5% | 0.5% | -0.347 |
| depth_extrapolation | SFT-only |  | 11.5% | 16.0% | -0.357 |
| depth_extrapolation | SFT-only, σ × 18.1 (baseline) |  | 33.5% | 38.0% | -0.354 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| depth_extrapolation | this model × (a) text scale | 0.5% | 1.0% | -0.347 |
| depth_extrapolation | this model × (a') text direct | 80.9% | 88.9% | -0.339 |
| depth_extrapolation | this model × (b) oracle per cell (upper bound) | 25.6% | 26.6% | -0.344 |
| depth_extrapolation | SFT-only × (a) text scale (**control**) | 45.5% | 52.5% | -0.341 |
| depth_extrapolation | SFT-only × (a') text direct | 77.0% | 86.0% | -0.332 |
| depth_extrapolation | SFT-only × (b) oracle per cell (upper bound) | 34.5% | 42.0% | -0.34 |
| id | this model × (a) text scale | 8.0% | 8.0% | -0.121 |
| id | this model × (a') text direct | 87.5% | 96.0% | -0.12 |
| id | this model × (b) oracle per cell (upper bound) | 49.0% | 60.5% | -0.112 |
| id | SFT-only × (a) text scale (**control**) | 42.5% | 53.5% | -0.314 |
| id | SFT-only × (a') text direct | 79.5% | 88.5% | -0.293 |
| id | SFT-only × (b) oracle per cell (upper bound) | 31.0% | 39.5% | -0.306 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| id | this model × A5 σ | 79.5% | 88.5% | -0.0998 |
| id | SFT-only × A5 σ | 67.3% | 77.9% | -0.29 |
| depth_extrapolation | this model × A5 σ | 67.3% | 77.9% | -0.32 |
| depth_extrapolation | SFT-only × A5 σ | 66.5% | 75.5% | -0.321 |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 78.0% | 0.00229 | 0.0% | 0.0% | 18.0% |
| 2 | 50 | 36.0% | 0.0364 | 0.0% | 0.0% | 2.0% |
| 3 | 50 | 30.0% | 0.0327 | 0.0% | 0.0% | 2.0% |
| 4 | 50 | 26.0% | 0.0508 | 0.0% | 0.0% | 0.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 69.1% | 0.00331 | 0.0% | 0.0% | 8.8% |
| energy | 64 | 20.3% | 0.0639 | 0.0% | 0.0% | 4.7% |
| kinematics | 68 | 36.8% | 0.0193 | 0.0% | 0.0% | 2.9% |

Mean score by rule (failures at the rule's floor): crps -0.129, binary 0.425, log_score -5.3, abs_rel_error -0.129, crps_log -0.121
Mean step score: 0.664; copied-given rate: 0.0%; steps block present: 100.0%; mean completion tokens: 191

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 34.0% | 0.0463 | 0.5% | 0.0% | 0.0% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 50.7% | 0.00999 | 0.0% | 0.0% | 0.0% |
| energy | 66 | 1.5% | 0.13 | 0.0% | 0.0% | 0.0% |
| kinematics | 67 | 49.3% | 0.00953 | 1.5% | 0.0% | 0.0% |

Mean score by rule (failures at the rule's floor): crps -0.498, binary 0.34, log_score -6.76, abs_rel_error -0.498, crps_log -0.348
Mean step score: 0.675; copied-given rate: 0.0%; steps block present: 99.5%; mean completion tokens: 315

# Eval summary: split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed0-compare-rl287

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 618 s
- torch_max_allocated_gib: 4.48
- torch_max_reserved_gib: 4.76
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.77

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 39.0% | 0.0282 | 0.0% | 0.0% | 0.0% | 25.5% | 33.5% | 0.0024 | -0.201 | 0.0% |
| depth_extrapolation | 200 | 30.0% | 0.0501 | 0.0% | 0.0% | 0.0% | 19.5% | 24.5% | 0.00248 | -0.211 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split.

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| id | this model | 4.0% | 25.5% | 33.5% | -0.3 |
| id | this model, σ × 18.1 |  | 50.0% | 58.0% | -0.297 |
| id | SFT-only |  | 15.0% | 20.5% | -0.32 |
| id | SFT-only, σ × 18.1 (baseline) |  | 35.0% | 40.5% | -0.319 |
| depth_extrapolation | this model | 0.5% | 19.5% | 24.5% | -0.351 |
| depth_extrapolation | this model, σ × 18.1 |  | 51.5% | 58.5% | -0.341 |
| depth_extrapolation | SFT-only |  | 11.5% | 16.0% | -0.357 |
| depth_extrapolation | SFT-only, σ × 18.1 (baseline) |  | 33.5% | 38.0% | -0.354 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| depth_extrapolation | this model × (a) text scale | 69.5% | 77.0% | -0.321 |
| depth_extrapolation | this model × (a') text direct | 77.0% | 85.0% | -0.324 |
| depth_extrapolation | this model × (b) oracle per cell (upper bound) | 59.0% | 66.5% | -0.313 |
| depth_extrapolation | SFT-only × (a) text scale (**control**) | 45.5% | 52.5% | -0.341 |
| depth_extrapolation | SFT-only × (a') text direct | 77.0% | 86.0% | -0.332 |
| depth_extrapolation | SFT-only × (b) oracle per cell (upper bound) | 34.5% | 42.0% | -0.34 |
| id | this model × (a) text scale | 63.0% | 74.5% | -0.292 |
| id | this model × (a') text direct | 75.5% | 85.0% | -0.275 |
| id | this model × (b) oracle per cell (upper bound) | 47.0% | 58.0% | -0.276 |
| id | SFT-only × (a) text scale (**control**) | 42.5% | 53.5% | -0.314 |
| id | SFT-only × (a') text direct | 79.5% | 88.5% | -0.293 |
| id | SFT-only × (b) oracle per cell (upper bound) | 31.0% | 39.5% | -0.306 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| id | this model × A5 σ | 66.3% | 78.4% | -0.271 |
| id | SFT-only × A5 σ | 67.3% | 77.9% | -0.29 |
| depth_extrapolation | this model × A5 σ | 65.5% | 75.0% | -0.313 |
| depth_extrapolation | SFT-only × A5 σ | 66.5% | 75.5% | -0.321 |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 68.0% | 0.00177 | 0.0% | 0.0% | 50.0% |
| 2 | 50 | 30.0% | 0.0305 | 0.0% | 0.0% | 14.0% |
| 3 | 50 | 30.0% | 0.0601 | 0.0% | 0.0% | 20.0% |
| 4 | 50 | 28.0% | 0.0492 | 0.0% | 0.0% | 18.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 61.8% | 0.00322 | 0.0% | 0.0% | 41.2% |
| energy | 64 | 14.1% | 0.0746 | 0.0% | 0.0% | 9.4% |
| kinematics | 68 | 39.7% | 0.024 | 0.0% | 0.0% | 25.0% |

Mean score by rule (failures at the rule's floor): crps -0.375, binary 0.39, log_score -4.23, abs_rel_error -0.376, crps_log -0.3
Mean step score: 0.658; copied-given rate: 0.0%; steps block present: 100.0%; mean completion tokens: 195

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 30.0% | 0.0501 | 0.0% | 0.0% | 19.5% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 43.3% | 0.0125 | 0.0% | 0.0% | 31.3% |
| energy | 66 | 0.0% | 0.238 | 0.0% | 0.0% | 0.0% |
| kinematics | 67 | 46.3% | 0.014 | 0.0% | 0.0% | 26.9% |

Mean score by rule (failures at the rule's floor): crps -0.418, binary 0.3, log_score -5.4, abs_rel_error -0.42, crps_log -0.351
Mean step score: 0.696; copied-given rate: 0.0%; steps block present: 100.0%; mean completion tokens: 316

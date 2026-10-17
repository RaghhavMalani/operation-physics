# Eval summary: split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed1-compare-rl287_4bit

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 506 s
- torch_max_allocated_gib: 3.11
- torch_max_reserved_gib: 3.56
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 3.70

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 40.5% | 0.0266 | 0.0% | 0.0% | 0.0% | 43.0% | 52.0% | 0.0155 | -0.104 | 0.0% |
| depth_extrapolation | 200 | 33.0% | 0.0476 | 1.0% | 0.0% | 0.0% | 39.9% | 51.0% | 0.0182 | -0.301 | 1.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split.

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| id | this model | 0.0% | 43.0% | 52.0% | -0.204 |
| id | this model, σ × 18.1 |  | 82.5% | 90.0% | -0.228 |
| id | SFT-only |  | 15.0% | 20.5% | -0.32 |
| id | SFT-only, σ × 18.1 (baseline) |  | 35.0% | 40.5% | -0.319 |
| depth_extrapolation | this model | 0.0% | 39.9% | 51.0% | -0.431 |
| depth_extrapolation | this model, σ × 18.1 |  | 77.8% | 84.3% | -0.446 |
| depth_extrapolation | SFT-only |  | 11.5% | 16.0% | -0.357 |
| depth_extrapolation | SFT-only, σ × 18.1 (baseline) |  | 33.5% | 38.0% | -0.354 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| depth_extrapolation | this model × (a) text scale | 89.4% | 93.4% | -0.619 |
| depth_extrapolation | this model × (a') text direct | 78.8% | 84.8% | -0.406 |
| depth_extrapolation | this model × (b) oracle per cell (upper bound) | 65.2% | 73.2% | -0.415 |
| depth_extrapolation | SFT-only × (a) text scale (**control**) | 45.5% | 52.5% | -0.341 |
| depth_extrapolation | SFT-only × (a') text direct | 77.0% | 86.0% | -0.332 |
| depth_extrapolation | SFT-only × (b) oracle per cell (upper bound) | 34.5% | 42.0% | -0.34 |
| id | this model × (a) text scale | 94.0% | 96.5% | -0.374 |
| id | this model × (a') text direct | 80.0% | 89.5% | -0.184 |
| id | this model × (b) oracle per cell (upper bound) | 59.0% | 72.5% | -0.191 |
| id | SFT-only × (a) text scale (**control**) | 42.5% | 53.5% | -0.314 |
| id | SFT-only × (a') text direct | 79.5% | 88.5% | -0.293 |
| id | SFT-only × (b) oracle per cell (upper bound) | 31.0% | 39.5% | -0.306 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| id | this model × A5 σ | 72.5% | 83.5% | -0.177 |
| id | SFT-only × A5 σ | 67.3% | 77.9% | -0.29 |
| depth_extrapolation | this model × A5 σ | 65.5% | 76.1% | -0.402 |
| depth_extrapolation | SFT-only × A5 σ | 66.5% | 75.5% | -0.321 |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 76.0% | 0.00166 | 0.0% | 0.0% | 80.0% |
| 2 | 50 | 34.0% | 0.0417 | 0.0% | 0.0% | 32.0% |
| 3 | 50 | 24.0% | 0.0428 | 0.0% | 0.0% | 32.0% |
| 4 | 50 | 28.0% | 0.0629 | 0.0% | 0.0% | 28.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 63.2% | 0.00481 | 0.0% | 0.0% | 66.2% |
| energy | 64 | 20.3% | 0.0673 | 0.0% | 0.0% | 23.4% |
| kinematics | 68 | 36.8% | 0.0383 | 0.0% | 0.0% | 38.2% |

Mean score by rule (failures at the rule's floor): crps -0.266, binary 0.405, log_score -2.42, abs_rel_error -0.272, crps_log -0.204
Mean step score: 0.667; copied-given rate: 0.5%; steps block present: 100.0%; mean completion tokens: 193

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 33.0% | 0.0476 | 1.0% | 0.0% | 39.9% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 49.3% | 0.0108 | 0.0% | 0.0% | 55.2% |
| energy | 66 | 3.0% | 0.171 | 0.0% | 0.0% | 10.6% |
| kinematics | 67 | 46.3% | 0.012 | 3.0% | 0.0% | 53.8% |

Mean score by rule (failures at the rule's floor): crps -0.44, binary 0.33, log_score -2.89, abs_rel_error -0.446, crps_log -0.431
Mean step score: 0.647; copied-given rate: 0.5%; steps block present: 99.0%; mean completion tokens: 317

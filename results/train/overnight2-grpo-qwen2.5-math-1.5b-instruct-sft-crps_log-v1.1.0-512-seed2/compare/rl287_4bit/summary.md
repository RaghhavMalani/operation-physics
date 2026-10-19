# Eval summary: overnight2-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed2-compare-rl287_4bit

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 642 s
- torch_max_allocated_gib: 3.11
- torch_max_reserved_gib: 4.10
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 4.52

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 42.0% | 0.0262 | 0.0% | 0.0% | 0.0% | 13.0% | 17.0% | 0.000334 | -0.126 | 0.0% |
| depth_extrapolation | 200 | 31.5% | 0.051 | 1.0% | 0.0% | 0.0% | 8.1% | 12.1% | 0.000552 | -0.396 | 1.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split.

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| id | this model | 28.5% | 13.0% | 17.0% | -0.223 |
| id | this model, σ × 18.1 |  | 29.0% | 36.0% | -0.222 |
| id | SFT-only |  | 15.0% | 20.5% | -0.32 |
| id | SFT-only, σ × 18.1 (baseline) |  | 35.0% | 40.5% | -0.319 |
| depth_extrapolation | this model | 17.7% | 8.1% | 12.1% | -0.524 |
| depth_extrapolation | this model, σ × 18.1 |  | 30.3% | 36.9% | -0.519 |
| depth_extrapolation | SFT-only |  | 11.5% | 16.0% | -0.357 |
| depth_extrapolation | SFT-only, σ × 18.1 (baseline) |  | 33.5% | 38.0% | -0.354 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| depth_extrapolation | this model × (a) text scale | 40.9% | 53.0% | -0.506 |
| depth_extrapolation | this model × (a') text direct | 75.8% | 85.4% | -0.485 |
| depth_extrapolation | this model × (b) oracle per cell (upper bound) | 41.4% | 54.0% | -0.505 |
| depth_extrapolation | SFT-only × (a) text scale (**control**) | 45.5% | 52.5% | -0.341 |
| depth_extrapolation | SFT-only × (a') text direct | 77.0% | 86.0% | -0.332 |
| depth_extrapolation | SFT-only × (b) oracle per cell (upper bound) | 34.5% | 42.0% | -0.34 |
| id | this model × (a) text scale | 41.0% | 49.5% | -0.217 |
| id | this model × (a') text direct | 80.5% | 89.0% | -0.203 |
| id | this model × (b) oracle per cell (upper bound) | 33.5% | 41.5% | -0.21 |
| id | SFT-only × (a) text scale (**control**) | 42.5% | 53.5% | -0.314 |
| id | SFT-only × (a') text direct | 79.5% | 88.5% | -0.293 |
| id | SFT-only × (b) oracle per cell (upper bound) | 31.0% | 39.5% | -0.306 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| id | this model × A5 σ | 71.0% | 82.0% | -0.193 |
| id | SFT-only × A5 σ | 67.3% | 77.9% | -0.29 |
| depth_extrapolation | this model × A5 σ | 63.8% | 73.5% | -0.49 |
| depth_extrapolation | SFT-only × A5 σ | 66.5% | 75.5% | -0.321 |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 78.0% | 0.00133 | 0.0% | 0.0% | 30.0% |
| 2 | 50 | 34.0% | 0.0359 | 0.0% | 0.0% | 6.0% |
| 3 | 50 | 24.0% | 0.0414 | 0.0% | 0.0% | 4.0% |
| 4 | 50 | 32.0% | 0.0574 | 0.0% | 0.0% | 12.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 63.2% | 0.00399 | 0.0% | 0.0% | 22.1% |
| energy | 64 | 25.0% | 0.0561 | 0.0% | 0.0% | 6.2% |
| kinematics | 68 | 36.8% | 0.0281 | 0.0% | 0.0% | 10.3% |

Mean score by rule (failures at the rule's floor): crps -0.3, binary 0.42, log_score -5.03, abs_rel_error -0.301, crps_log -0.223
Mean step score: 0.649; copied-given rate: 0.5%; steps block present: 100.0%; mean completion tokens: 194

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 31.5% | 0.051 | 1.0% | 0.0% | 8.1% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 49.3% | 0.0108 | 0.0% | 0.0% | 13.4% |
| energy | 66 | 3.0% | 0.162 | 1.5% | 0.0% | 0.0% |
| kinematics | 67 | 41.8% | 0.0129 | 1.5% | 0.0% | 10.6% |

Mean score by rule (failures at the rule's floor): crps -0.528, binary 0.315, log_score -6.51, abs_rel_error -0.529, crps_log -0.524
Mean step score: 0.636; copied-given rate: 0.0%; steps block present: 99.0%; mean completion tokens: 318

# Eval summary: split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed0-compare-rl287_4bit_rerun

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 515 s
- torch_max_allocated_gib: 3.11
- torch_max_reserved_gib: 3.56
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 3.70

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 40.0% | 0.0258 | 0.0% | 0.0% | 0.0% | 43.0% | 52.5% | 0.0157 | -0.106 | 0.0% |
| depth_extrapolation | 200 | 35.5% | 0.0452 | 0.5% | 0.0% | 0.0% | 40.7% | 47.2% | 0.0167 | -0.299 | 0.5% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split.

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| id | this model | 0.0% | 43.0% | 52.5% | -0.204 |
| id | this model, σ × 18.1 |  | 82.5% | 88.5% | -0.228 |
| id | SFT-only |  | 15.0% | 20.5% | -0.32 |
| id | SFT-only, σ × 18.1 (baseline) |  | 35.0% | 40.5% | -0.319 |
| depth_extrapolation | this model | 0.0% | 40.7% | 47.2% | -0.428 |
| depth_extrapolation | this model, σ × 18.1 |  | 78.4% | 84.9% | -0.432 |
| depth_extrapolation | SFT-only |  | 11.5% | 16.0% | -0.357 |
| depth_extrapolation | SFT-only, σ × 18.1 (baseline) |  | 33.5% | 38.0% | -0.354 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| depth_extrapolation | this model × (a) text scale | 90.5% | 95.0% | -0.599 |
| depth_extrapolation | this model × (a') text direct | 77.4% | 87.4% | -0.392 |
| depth_extrapolation | this model × (b) oracle per cell (upper bound) | 67.3% | 76.4% | -0.408 |
| depth_extrapolation | SFT-only × (a) text scale (**control**) | 45.5% | 52.5% | -0.341 |
| depth_extrapolation | SFT-only × (a') text direct | 77.0% | 86.0% | -0.332 |
| depth_extrapolation | SFT-only × (b) oracle per cell (upper bound) | 34.5% | 42.0% | -0.34 |
| id | this model × (a) text scale | 93.5% | 97.0% | -0.383 |
| id | this model × (a') text direct | 78.0% | 88.5% | -0.185 |
| id | this model × (b) oracle per cell (upper bound) | 58.5% | 70.0% | -0.191 |
| id | SFT-only × (a) text scale (**control**) | 42.5% | 53.5% | -0.314 |
| id | SFT-only × (a') text direct | 79.5% | 88.5% | -0.293 |
| id | SFT-only × (b) oracle per cell (upper bound) | 31.0% | 39.5% | -0.306 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| id | this model × A5 σ | 71.5% | 80.5% | -0.18 |
| id | SFT-only × A5 σ | 67.3% | 77.9% | -0.29 |
| depth_extrapolation | this model × A5 σ | 66.7% | 76.8% | -0.397 |
| depth_extrapolation | SFT-only × A5 σ | 66.5% | 75.5% | -0.321 |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 76.0% | 0.00191 | 0.0% | 0.0% | 80.0% |
| 2 | 50 | 30.0% | 0.039 | 0.0% | 0.0% | 32.0% |
| 3 | 50 | 24.0% | 0.0423 | 0.0% | 0.0% | 32.0% |
| 4 | 50 | 30.0% | 0.0518 | 0.0% | 0.0% | 28.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 63.2% | 0.00457 | 0.0% | 0.0% | 66.2% |
| energy | 64 | 20.3% | 0.0795 | 0.0% | 0.0% | 21.9% |
| kinematics | 68 | 35.3% | 0.0316 | 0.0% | 0.0% | 39.7% |

Mean score by rule (failures at the rule's floor): crps -0.258, binary 0.4, log_score -2.22, abs_rel_error -0.264, crps_log -0.204
Mean step score: 0.654; copied-given rate: 0.0%; steps block present: 100.0%; mean completion tokens: 193

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 35.5% | 0.0452 | 0.5% | 0.0% | 40.7% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 52.2% | 0.00753 | 0.0% | 0.0% | 61.2% |
| energy | 66 | 3.0% | 0.171 | 0.0% | 0.0% | 6.1% |
| kinematics | 67 | 50.7% | 0.00851 | 1.5% | 0.0% | 54.5% |

Mean score by rule (failures at the rule's floor): crps -0.466, binary 0.355, log_score -2.99, abs_rel_error -0.473, crps_log -0.428
Mean step score: 0.646; copied-given rate: 1.0%; steps block present: 99.5%; mean completion tokens: 317

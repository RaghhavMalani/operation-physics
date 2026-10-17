# Eval summary: split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed0-compare-rl287_sampled

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 2438 s
- torch_max_allocated_gib: 4.48
- torch_max_reserved_gib: 4.76
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.05

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 800 | 35.9% | 0.0317 | 0.2% | 0.2% | 0.1% | 24.7% | 30.4% | 0.00322 | -0.289 | 0.1% |
| depth_extrapolation | 800 | 23.4% | 0.112 | 1.2% | 0.0% | 0.1% | 15.3% | 21.4% | 0.00419 | -0.539 | 0.2% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split.

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| id | this model | 7.7% | 24.7% | 30.4% | -0.379 |
| id | this model, σ × 18.1 |  | 52.8% | 60.2% | -0.377 |
| id | SFT-only |  | 15.0% | 20.5% | -0.32 |
| id | SFT-only, σ × 18.1 (baseline) |  | 35.0% | 40.5% | -0.319 |
| depth_extrapolation | this model | 3.4% | 15.3% | 21.4% | -0.655 |
| depth_extrapolation | this model, σ × 18.1 |  | 43.8% | 54.3% | -0.641 |
| depth_extrapolation | SFT-only |  | 11.5% | 16.0% | -0.357 |
| depth_extrapolation | SFT-only, σ × 18.1 (baseline) |  | 33.5% | 38.0% | -0.354 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| depth_extrapolation | this model × (a) text scale | 61.4% | 71.0% | -0.653 |
| depth_extrapolation | this model × (a') text direct | 64.6% | 74.3% | -0.61 |
| depth_extrapolation | this model × (b) oracle per cell (upper bound) | 47.2% | 54.1% | -0.633 |
| depth_extrapolation | SFT-only × (a) text scale (**control**) | 45.5% | 52.5% | -0.341 |
| depth_extrapolation | SFT-only × (a') text direct | 77.0% | 86.0% | -0.332 |
| depth_extrapolation | SFT-only × (b) oracle per cell (upper bound) | 34.5% | 42.0% | -0.34 |
| id | this model × (a) text scale | 65.3% | 74.0% | -0.4 |
| id | this model × (a') text direct | 71.7% | 82.5% | -0.353 |
| id | this model × (b) oracle per cell (upper bound) | 42.8% | 52.8% | -0.366 |
| id | SFT-only × (a) text scale (**control**) | 42.5% | 53.5% | -0.314 |
| id | SFT-only × (a') text direct | 79.5% | 88.5% | -0.293 |
| id | SFT-only × (b) oracle per cell (upper bound) | 31.0% | 39.5% | -0.306 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| id | this model × A5 σ | 63.6% | 77.4% | -0.347 |
| id | SFT-only × A5 σ | 67.3% | 77.9% | -0.29 |
| depth_extrapolation | this model × A5 σ | 50.0% | 60.8% | -0.61 |
| depth_extrapolation | SFT-only × A5 σ | 66.5% | 75.5% | -0.321 |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 200 | 66.5% | 0.00217 | 0.5% | 0.0% | 49.2% |
| 2 | 200 | 26.5% | 0.0515 | 0.5% | 0.0% | 18.1% |
| 3 | 200 | 31.5% | 0.0578 | 0.0% | 0.5% | 18.6% |
| 4 | 200 | 19.0% | 0.1 | 0.0% | 0.5% | 13.1% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 272 | 52.2% | 0.00842 | 0.0% | 0.4% | 35.8% |
| energy | 256 | 16.4% | 0.0915 | 0.4% | 0.0% | 11.4% |
| kinematics | 272 | 37.9% | 0.0318 | 0.4% | 0.4% | 26.3% |

Mean score by rule (failures at the rule's floor): crps -0.435, binary 0.359, log_score -4.48, abs_rel_error -0.437, crps_log -0.379
Mean step score: 0.598; copied-given rate: 0.0%; steps block present: 99.8%; mean completion tokens: 195

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 800 | 23.4% | 0.112 | 1.2% | 0.0% | 15.3% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 268 | 32.1% | 0.0238 | 1.5% | 0.0% | 20.5% |
| energy | 264 | 5.3% | 0.386 | 1.1% | 0.0% | 3.4% |
| kinematics | 268 | 32.5% | 0.0641 | 1.1% | 0.0% | 21.9% |

Mean score by rule (failures at the rule's floor): crps -0.696, binary 0.234, log_score -6.15, abs_rel_error -0.7, crps_log -0.655
Mean step score: 0.576; copied-given rate: 0.3%; steps block present: 98.8%; mean completion tokens: 317

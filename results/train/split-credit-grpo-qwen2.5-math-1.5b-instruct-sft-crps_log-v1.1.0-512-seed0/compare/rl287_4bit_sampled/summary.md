# Eval summary: split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed0-compare-rl287_4bit_sampled

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 1964 s
- torch_max_allocated_gib: 3.11
- torch_max_reserved_gib: 3.56
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 4.84

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 800 | 35.8% | 0.0333 | 0.2% | 0.1% | 0.1% | 40.3% | 50.1% | 0.0178 | -0.242 | 0.0% |
| depth_extrapolation | 800 | 22.0% | 0.0892 | 0.9% | 0.1% | 0.0% | 29.2% | 36.6% | 0.0169 | -0.388 | 0.5% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split.

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| id | this model | 0.4% | 40.3% | 50.1% | -0.327 |
| id | this model, σ × 18.1 |  | 78.3% | 85.6% | -0.346 |
| id | SFT-only |  | 15.0% | 20.5% | -0.32 |
| id | SFT-only, σ × 18.1 (baseline) |  | 35.0% | 40.5% | -0.319 |
| depth_extrapolation | this model | 0.4% | 29.2% | 36.6% | -0.496 |
| depth_extrapolation | this model, σ × 18.1 |  | 68.6% | 78.4% | -0.491 |
| depth_extrapolation | SFT-only |  | 11.5% | 16.0% | -0.357 |
| depth_extrapolation | SFT-only, σ × 18.1 (baseline) |  | 33.5% | 38.0% | -0.354 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| depth_extrapolation | this model × (a) text scale | 85.7% | 91.5% | -0.67 |
| depth_extrapolation | this model × (a') text direct | 67.4% | 81.2% | -0.462 |
| depth_extrapolation | this model × (b) oracle per cell (upper bound) | 58.6% | 70.1% | -0.469 |
| depth_extrapolation | SFT-only × (a) text scale (**control**) | 45.5% | 52.5% | -0.341 |
| depth_extrapolation | SFT-only × (a') text direct | 77.0% | 86.0% | -0.332 |
| depth_extrapolation | SFT-only × (b) oracle per cell (upper bound) | 34.5% | 42.0% | -0.34 |
| id | this model × (a) text scale | 90.7% | 94.5% | -0.512 |
| id | this model × (a') text direct | 74.3% | 83.9% | -0.304 |
| id | this model × (b) oracle per cell (upper bound) | 57.6% | 69.9% | -0.31 |
| id | SFT-only × (a) text scale (**control**) | 42.5% | 53.5% | -0.314 |
| id | SFT-only × (a') text direct | 79.5% | 88.5% | -0.293 |
| id | SFT-only × (b) oracle per cell (upper bound) | 31.0% | 39.5% | -0.306 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| id | this model × A5 σ | 63.0% | 76.0% | -0.299 |
| id | SFT-only × A5 σ | 67.3% | 77.9% | -0.29 |
| depth_extrapolation | this model × A5 σ | 53.8% | 68.8% | -0.457 |
| depth_extrapolation | SFT-only × A5 σ | 66.5% | 75.5% | -0.321 |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 200 | 71.0% | 0.00287 | 0.0% | 0.0% | 73.5% |
| 2 | 200 | 26.0% | 0.0636 | 0.0% | 0.0% | 31.0% |
| 3 | 200 | 25.5% | 0.0547 | 0.5% | 0.5% | 32.8% |
| 4 | 200 | 20.5% | 0.0775 | 0.5% | 0.0% | 23.6% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 272 | 55.5% | 0.00664 | 0.0% | 0.0% | 57.4% |
| energy | 256 | 16.8% | 0.0856 | 0.8% | 0.4% | 23.7% |
| kinematics | 272 | 33.8% | 0.0399 | 0.0% | 0.0% | 38.6% |

Mean score by rule (failures at the rule's floor): crps -0.366, binary 0.357, log_score -2.89, abs_rel_error -0.372, crps_log -0.327
Mean step score: 0.567; copied-given rate: 0.0%; steps block present: 99.5%; mean completion tokens: 194

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 800 | 22.0% | 0.0892 | 0.9% | 0.1% | 29.2% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 268 | 32.1% | 0.0267 | 0.4% | 0.4% | 41.4% |
| energy | 264 | 1.9% | 0.297 | 0.8% | 0.0% | 6.1% |
| kinematics | 268 | 31.7% | 0.0366 | 1.5% | 0.0% | 39.8% |

Mean score by rule (failures at the rule's floor): crps -0.574, binary 0.22, log_score -4.49, abs_rel_error -0.585, crps_log -0.496
Mean step score: 0.541; copied-given rate: 0.0%; steps block present: 99.1%; mean completion tokens: 317

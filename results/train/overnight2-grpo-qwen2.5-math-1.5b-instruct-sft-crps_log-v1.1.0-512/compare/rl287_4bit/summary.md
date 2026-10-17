# Eval summary: overnight2-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-compare-rl287_4bit

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 502 s
- torch_max_allocated_gib: 3.11
- torch_max_reserved_gib: 3.56
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 3.70

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 41.5% | 0.0253 | 0.0% | 0.0% | 0.0% | 11.5% | 16.5% | 0.000426 | -0.137 | 0.0% |
| depth_extrapolation | 200 | 33.0% | 0.05 | 0.0% | 0.0% | 0.0% | 9.5% | 13.5% | 0.000619 | -0.284 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split.

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| id | this model | 23.5% | 11.5% | 16.5% | -0.235 |
| id | this model, σ × 18.1 |  | 32.5% | 39.5% | -0.232 |
| id | SFT-only |  | 15.0% | 20.5% | -0.32 |
| id | SFT-only, σ × 18.1 (baseline) |  | 35.0% | 40.5% | -0.319 |
| depth_extrapolation | this model | 15.5% | 9.5% | 13.5% | -0.414 |
| depth_extrapolation | this model, σ × 18.1 |  | 33.0% | 40.0% | -0.408 |
| depth_extrapolation | SFT-only |  | 11.5% | 16.0% | -0.357 |
| depth_extrapolation | SFT-only, σ × 18.1 (baseline) |  | 33.5% | 38.0% | -0.354 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| depth_extrapolation | this model × (a) text scale | 46.0% | 55.0% | -0.394 |
| depth_extrapolation | this model × (a') text direct | 78.5% | 85.5% | -0.38 |
| depth_extrapolation | this model × (b) oracle per cell (upper bound) | 44.5% | 54.0% | -0.396 |
| depth_extrapolation | SFT-only × (a) text scale (**control**) | 45.5% | 52.5% | -0.341 |
| depth_extrapolation | SFT-only × (a') text direct | 77.0% | 86.0% | -0.332 |
| depth_extrapolation | SFT-only × (b) oracle per cell (upper bound) | 34.5% | 42.0% | -0.34 |
| id | this model × (a) text scale | 43.5% | 52.0% | -0.224 |
| id | this model × (a') text direct | 78.0% | 88.0% | -0.213 |
| id | this model × (b) oracle per cell (upper bound) | 35.0% | 43.5% | -0.216 |
| id | SFT-only × (a) text scale (**control**) | 42.5% | 53.5% | -0.314 |
| id | SFT-only × (a') text direct | 79.5% | 88.5% | -0.293 |
| id | SFT-only × (b) oracle per cell (upper bound) | 31.0% | 39.5% | -0.306 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| id | this model × A5 σ | 69.5% | 79.5% | -0.204 |
| id | SFT-only × A5 σ | 67.3% | 77.9% | -0.29 |
| depth_extrapolation | this model × A5 σ | 64.3% | 72.9% | -0.382 |
| depth_extrapolation | SFT-only × A5 σ | 66.5% | 75.5% | -0.321 |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 76.0% | 0.00172 | 0.0% | 0.0% | 28.0% |
| 2 | 50 | 32.0% | 0.0489 | 0.0% | 0.0% | 6.0% |
| 3 | 50 | 26.0% | 0.0484 | 0.0% | 0.0% | 4.0% |
| 4 | 50 | 32.0% | 0.0488 | 0.0% | 0.0% | 8.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 64.7% | 0.00483 | 0.0% | 0.0% | 17.6% |
| energy | 64 | 21.9% | 0.0795 | 0.0% | 0.0% | 6.2% |
| kinematics | 68 | 36.8% | 0.0316 | 0.0% | 0.0% | 10.3% |

Mean score by rule (failures at the rule's floor): crps -0.323, binary 0.415, log_score -5.08, abs_rel_error -0.324, crps_log -0.235
Mean step score: 0.657; copied-given rate: 0.0%; steps block present: 100.0%; mean completion tokens: 194

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 33.0% | 0.05 | 0.0% | 0.0% | 9.5% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 47.8% | 0.0108 | 0.0% | 0.0% | 11.9% |
| energy | 66 | 3.0% | 0.163 | 0.0% | 0.0% | 0.0% |
| kinematics | 67 | 47.8% | 0.012 | 0.0% | 0.0% | 16.4% |

Mean score by rule (failures at the rule's floor): crps -0.426, binary 0.33, log_score -6.37, abs_rel_error -0.427, crps_log -0.414
Mean step score: 0.647; copied-given rate: 0.0%; steps block present: 100.0%; mean completion tokens: 317

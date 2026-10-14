# Eval summary: overnight-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-512-compare-sft

- model: `results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 622 s
- torch_max_allocated_gib: 4.48
- torch_max_reserved_gib: 4.76
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.45

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 38.0% | 0.0282 | 0.0% | 0.0% | 0.0% | 15.0% | 20.5% | 0.0005 | -0.224 | 0.0% |
| depth_extrapolation | 200 | 31.0% | 0.0407 | 0.0% | 0.0% | 0.0% | 11.5% | 16.0% | 0.000807 | -0.215 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split.

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| id | this model | – | 15.0% | 20.5% | -0.32 |
| id | this model, σ × 18.1 |  | 35.0% | 40.5% | -0.319 |
| id | SFT-only |  | 15.0% | 20.5% | -0.32 |
| id | SFT-only, σ × 18.1 (baseline) |  | 35.0% | 40.5% | -0.319 |
| depth_extrapolation | this model | – | 11.5% | 16.0% | -0.357 |
| depth_extrapolation | this model, σ × 18.1 |  | 33.5% | 38.0% | -0.354 |
| depth_extrapolation | SFT-only |  | 11.5% | 16.0% | -0.357 |
| depth_extrapolation | SFT-only, σ × 18.1 (baseline) |  | 33.5% | 38.0% | -0.354 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| depth_extrapolation | this model × (a) text scale | 45.5% | 52.5% | -0.341 |
| depth_extrapolation | this model × (a') text direct | 77.0% | 86.0% | -0.332 |
| depth_extrapolation | this model × (b) oracle per cell (upper bound) | 34.5% | 42.0% | -0.34 |
| depth_extrapolation | SFT-only × (a) text scale (**control**) | 45.5% | 52.5% | -0.341 |
| depth_extrapolation | SFT-only × (a') text direct | 77.0% | 86.0% | -0.332 |
| depth_extrapolation | SFT-only × (b) oracle per cell (upper bound) | 34.5% | 42.0% | -0.34 |
| id | this model × (a) text scale | 42.5% | 53.5% | -0.314 |
| id | this model × (a') text direct | 79.5% | 88.5% | -0.293 |
| id | this model × (b) oracle per cell (upper bound) | 31.0% | 39.5% | -0.306 |
| id | SFT-only × (a) text scale (**control**) | 42.5% | 53.5% | -0.314 |
| id | SFT-only × (a') text direct | 79.5% | 88.5% | -0.293 |
| id | SFT-only × (b) oracle per cell (upper bound) | 31.0% | 39.5% | -0.306 |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 68.0% | 0.00195 | 0.0% | 0.0% | 30.0% |
| 2 | 50 | 32.0% | 0.0325 | 0.0% | 0.0% | 8.0% |
| 3 | 50 | 30.0% | 0.0585 | 0.0% | 0.0% | 10.0% |
| 4 | 50 | 22.0% | 0.0644 | 0.0% | 0.0% | 12.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 60.3% | 0.00322 | 0.0% | 0.0% | 23.5% |
| energy | 64 | 14.1% | 0.0844 | 0.0% | 0.0% | 6.2% |
| kinematics | 68 | 38.2% | 0.0255 | 0.0% | 0.0% | 14.7% |

Mean score by rule (failures at the rule's floor): crps -0.4, binary 0.38, log_score -4.92, abs_rel_error -0.401, crps_log -0.32
Mean step score: 0.638; copied-given rate: 0.5%; steps block present: 100.0%; mean completion tokens: 196

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 31.0% | 0.0407 | 0.0% | 0.0% | 11.5% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 43.3% | 0.0125 | 0.0% | 0.0% | 14.9% |
| energy | 66 | 3.0% | 0.221 | 0.0% | 0.0% | 0.0% |
| kinematics | 67 | 46.3% | 0.014 | 0.0% | 0.0% | 19.4% |

Mean score by rule (failures at the rule's floor): crps -0.424, binary 0.31, log_score -5.97, abs_rel_error -0.425, crps_log -0.357
Mean step score: 0.71; copied-given rate: 0.5%; steps block present: 100.0%; mean completion tokens: 317

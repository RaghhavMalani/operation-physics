# Eval summary: overnight2-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-compare-rlfinal

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 670 s
- torch_max_allocated_gib: 4.48
- torch_max_reserved_gib: 4.76
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.18

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 41.0% | 0.0284 | 0.0% | 0.0% | 0.0% | 11.5% | 16.0% | 0 | -0.141 | 0.0% |
| depth_extrapolation | 200 | 31.0% | 0.0342 | 1.0% | 0.0% | 0.5% | 8.1% | 13.1% | 0.000261 | -0.203 | 0.5% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split.

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| id | this model | – | 11.5% | 16.0% | -0.239 |
| id | this model, σ × 18.1 |  | 24.5% | 29.5% | -0.238 |
| id | SFT-only |  | 15.0% | 20.5% | -0.32 |
| id | SFT-only, σ × 18.1 (baseline) |  | 35.0% | 40.5% | -0.319 |
| depth_extrapolation | this model | – | 8.1% | 13.1% | -0.344 |
| depth_extrapolation | this model, σ × 18.1 |  | 25.8% | 32.8% | -0.342 |
| depth_extrapolation | SFT-only |  | 11.5% | 16.0% | -0.357 |
| depth_extrapolation | SFT-only, σ × 18.1 (baseline) |  | 33.5% | 38.0% | -0.354 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| depth_extrapolation | this model × (a) text scale | 36.4% | 41.9% | -0.337 |
| depth_extrapolation | this model × (a') text direct | 79.3% | 88.4% | -0.34 |
| depth_extrapolation | this model × (b) oracle per cell (upper bound) | 25.8% | 30.8% | -0.334 |
| depth_extrapolation | SFT-only × (a) text scale (**control**) | 45.5% | 52.5% | -0.341 |
| depth_extrapolation | SFT-only × (a') text direct | 77.0% | 86.0% | -0.332 |
| depth_extrapolation | SFT-only × (b) oracle per cell (upper bound) | 34.5% | 42.0% | -0.34 |
| id | this model × (a) text scale | 30.5% | 37.0% | -0.235 |
| id | this model × (a') text direct | 82.5% | 91.5% | -0.215 |
| id | this model × (b) oracle per cell (upper bound) | 26.0% | 32.0% | -0.226 |
| id | SFT-only × (a) text scale (**control**) | 42.5% | 53.5% | -0.314 |
| id | SFT-only × (a') text direct | 79.5% | 88.5% | -0.293 |
| id | SFT-only × (b) oracle per cell (upper bound) | 31.0% | 39.5% | -0.306 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| id | this model × A5 σ | 71.9% | 83.9% | -0.208 |
| id | SFT-only × A5 σ | 67.3% | 77.9% | -0.29 |
| depth_extrapolation | this model × A5 σ | 70.2% | 80.3% | -0.312 |
| depth_extrapolation | SFT-only × A5 σ | 66.5% | 75.5% | -0.321 |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 74.0% | 0.00177 | 0.0% | 0.0% | 28.0% |
| 2 | 50 | 32.0% | 0.0398 | 0.0% | 0.0% | 6.0% |
| 3 | 50 | 32.0% | 0.0423 | 0.0% | 0.0% | 6.0% |
| 4 | 50 | 26.0% | 0.0854 | 0.0% | 0.0% | 6.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 63.2% | 0.00384 | 0.0% | 0.0% | 17.6% |
| energy | 64 | 14.1% | 0.0981 | 0.0% | 0.0% | 4.7% |
| kinematics | 68 | 44.1% | 0.0173 | 0.0% | 0.0% | 11.8% |

Mean score by rule (failures at the rule's floor): crps -0.293, binary 0.41, log_score -4.76, abs_rel_error -0.294, crps_log -0.239
Mean step score: 0.653; copied-given rate: 0.5%; steps block present: 100.0%; mean completion tokens: 193

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 31.0% | 0.0342 | 1.0% | 0.0% | 8.1% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 41.8% | 0.014 | 0.0% | 0.0% | 9.0% |
| energy | 66 | 3.0% | 0.184 | 1.5% | 0.0% | 1.5% |
| kinematics | 67 | 47.8% | 0.0113 | 1.5% | 0.0% | 13.6% |

Mean score by rule (failures at the rule's floor): crps -0.386, binary 0.31, log_score -6.29, abs_rel_error -0.387, crps_log -0.344
Mean step score: 0.704; copied-given rate: 0.0%; steps block present: 99.0%; mean completion tokens: 317

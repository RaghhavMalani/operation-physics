# Eval summary: overnight-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-512-compare-rl_4bit

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 498 s
- torch_max_allocated_gib: 3.11
- torch_max_reserved_gib: 3.56
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 3.70

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 42.0% | 0.0252 | 0.0% | 0.0% | 0.0% | 14.0% | 19.0% | 0.000508 | -0.116 | 0.0% |
| depth_extrapolation | 200 | 36.0% | 0.0452 | 1.0% | 0.0% | 0.0% | 11.6% | 16.2% | 0.000706 | -0.339 | 1.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split.

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| id | this model | 23.0% | 14.0% | 19.0% | -0.214 |
| id | this model, σ × 18.1 |  | 35.5% | 44.0% | -0.211 |
| id | SFT-only |  | 15.0% | 20.5% | -0.32 |
| id | SFT-only, σ × 18.1 (baseline) |  | 35.0% | 40.5% | -0.319 |
| depth_extrapolation | this model | 17.2% | 11.6% | 16.2% | -0.47 |
| depth_extrapolation | this model, σ × 18.1 |  | 31.8% | 39.9% | -0.466 |
| depth_extrapolation | SFT-only |  | 11.5% | 16.0% | -0.357 |
| depth_extrapolation | SFT-only, σ × 18.1 (baseline) |  | 33.5% | 38.0% | -0.354 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| depth_extrapolation | this model × (a) text scale | 46.0% | 56.1% | -0.46 |
| depth_extrapolation | this model × (a') text direct | 78.8% | 87.4% | -0.438 |
| depth_extrapolation | this model × (b) oracle per cell (upper bound) | 34.3% | 43.4% | -0.457 |
| depth_extrapolation | SFT-only × (a) text scale (**control**) | 45.5% | 52.5% | -0.341 |
| depth_extrapolation | SFT-only × (a') text direct | 77.0% | 86.0% | -0.332 |
| depth_extrapolation | SFT-only × (b) oracle per cell (upper bound) | 34.5% | 42.0% | -0.34 |
| id | this model × (a) text scale | 49.0% | 57.0% | -0.205 |
| id | this model × (a') text direct | 80.0% | 90.5% | -0.194 |
| id | this model × (b) oracle per cell (upper bound) | 38.5% | 45.0% | -0.199 |
| id | SFT-only × (a) text scale (**control**) | 42.5% | 53.5% | -0.314 |
| id | SFT-only × (a') text direct | 79.5% | 88.5% | -0.293 |
| id | SFT-only × (b) oracle per cell (upper bound) | 31.0% | 39.5% | -0.306 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| id | this model × A5 σ | 72.5% | 82.5% | -0.184 |
| id | SFT-only × A5 σ | 67.3% | 77.9% | -0.29 |
| depth_extrapolation | this model × A5 σ | 66.2% | 75.9% | -0.44 |
| depth_extrapolation | SFT-only × A5 σ | 66.5% | 75.5% | -0.321 |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 76.0% | 0.00172 | 0.0% | 0.0% | 28.0% |
| 2 | 50 | 32.0% | 0.039 | 0.0% | 0.0% | 6.0% |
| 3 | 50 | 28.0% | 0.0411 | 0.0% | 0.0% | 8.0% |
| 4 | 50 | 32.0% | 0.0525 | 0.0% | 0.0% | 14.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 66.2% | 0.00323 | 0.0% | 0.0% | 22.1% |
| energy | 64 | 21.9% | 0.069 | 0.0% | 0.0% | 6.2% |
| kinematics | 68 | 36.8% | 0.0316 | 0.0% | 0.0% | 13.2% |

Mean score by rule (failures at the rule's floor): crps -0.292, binary 0.42, log_score -4.9, abs_rel_error -0.292, crps_log -0.214
Mean step score: 0.653; copied-given rate: 0.0%; steps block present: 100.0%; mean completion tokens: 193

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 36.0% | 0.0452 | 1.0% | 0.0% | 11.6% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 52.2% | 0.00915 | 0.0% | 0.0% | 16.4% |
| energy | 66 | 4.5% | 0.187 | 0.0% | 0.0% | 0.0% |
| kinematics | 67 | 50.7% | 0.00921 | 3.0% | 0.0% | 18.5% |

Mean score by rule (failures at the rule's floor): crps -0.471, binary 0.36, log_score -6.23, abs_rel_error -0.472, crps_log -0.47
Mean step score: 0.656; copied-given rate: 1.0%; steps block present: 99.0%; mean completion tokens: 317

# Eval summary: split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed1-compare-rl287

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 653 s
- torch_max_allocated_gib: 4.48
- torch_max_reserved_gib: 4.76
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.03

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 41.5% | 0.0243 | 0.0% | 0.0% | 0.0% | 22.5% | 31.5% | 0.00181 | -0.16 | 0.0% |
| depth_extrapolation | 200 | 31.0% | 0.0402 | 0.0% | 0.0% | 0.0% | 19.5% | 23.5% | 0.00225 | -0.208 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split.

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| id | this model | 11.0% | 22.5% | 31.5% | -0.256 |
| id | this model, σ × 18.1 |  | 48.0% | 54.0% | -0.253 |
| id | SFT-only |  | 15.0% | 20.5% | -0.32 |
| id | SFT-only, σ × 18.1 (baseline) |  | 35.0% | 40.5% | -0.319 |
| depth_extrapolation | this model | 2.0% | 19.5% | 23.5% | -0.348 |
| depth_extrapolation | this model, σ × 18.1 |  | 52.5% | 58.0% | -0.341 |
| depth_extrapolation | SFT-only |  | 11.5% | 16.0% | -0.357 |
| depth_extrapolation | SFT-only, σ × 18.1 (baseline) |  | 33.5% | 38.0% | -0.354 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| depth_extrapolation | this model × (a) text scale | 67.0% | 74.5% | -0.325 |
| depth_extrapolation | this model × (a') text direct | 77.0% | 84.5% | -0.323 |
| depth_extrapolation | this model × (b) oracle per cell (upper bound) | 51.0% | 62.0% | -0.324 |
| depth_extrapolation | SFT-only × (a) text scale (**control**) | 45.5% | 52.5% | -0.341 |
| depth_extrapolation | SFT-only × (a') text direct | 77.0% | 86.0% | -0.332 |
| depth_extrapolation | SFT-only × (b) oracle per cell (upper bound) | 34.5% | 42.0% | -0.34 |
| id | this model × (a) text scale | 59.0% | 72.0% | -0.248 |
| id | this model × (a') text direct | 79.0% | 87.0% | -0.226 |
| id | this model × (b) oracle per cell (upper bound) | 45.5% | 54.0% | -0.233 |
| id | SFT-only × (a) text scale (**control**) | 42.5% | 53.5% | -0.314 |
| id | SFT-only × (a') text direct | 79.5% | 88.5% | -0.293 |
| id | SFT-only × (b) oracle per cell (upper bound) | 31.0% | 39.5% | -0.306 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| id | this model × A5 σ | 69.0% | 79.0% | -0.226 |
| id | SFT-only × A5 σ | 67.3% | 77.9% | -0.29 |
| depth_extrapolation | this model × A5 σ | 67.0% | 77.0% | -0.312 |
| depth_extrapolation | SFT-only × A5 σ | 66.5% | 75.5% | -0.321 |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 72.0% | 0.0014 | 0.0% | 0.0% | 44.0% |
| 2 | 50 | 34.0% | 0.0293 | 0.0% | 0.0% | 14.0% |
| 3 | 50 | 36.0% | 0.0585 | 0.0% | 0.0% | 18.0% |
| 4 | 50 | 24.0% | 0.0766 | 0.0% | 0.0% | 14.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 64.7% | 0.00284 | 0.0% | 0.0% | 33.8% |
| energy | 64 | 15.6% | 0.0684 | 0.0% | 0.0% | 6.2% |
| kinematics | 68 | 42.6% | 0.017 | 0.0% | 0.0% | 26.5% |

Mean score by rule (failures at the rule's floor): crps -0.333, binary 0.415, log_score -3.96, abs_rel_error -0.335, crps_log -0.256
Mean step score: 0.64; copied-given rate: 0.0%; steps block present: 100.0%; mean completion tokens: 194

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 31.0% | 0.0402 | 0.0% | 0.0% | 19.5% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 43.3% | 0.0125 | 0.0% | 0.0% | 28.4% |
| energy | 66 | 3.0% | 0.222 | 0.0% | 0.0% | 1.5% |
| kinematics | 67 | 46.3% | 0.014 | 0.0% | 0.0% | 28.4% |

Mean score by rule (failures at the rule's floor): crps -0.419, binary 0.31, log_score -5.34, abs_rel_error -0.421, crps_log -0.348
Mean step score: 0.7; copied-given rate: 0.5%; steps block present: 100.0%; mean completion tokens: 317

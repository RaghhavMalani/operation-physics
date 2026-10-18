# Eval summary: overnight2-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed1-compare-rl287_4bit

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 647 s
- torch_max_allocated_gib: 3.11
- torch_max_reserved_gib: 4.10
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 4.40

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 41.0% | 0.0222 | 0.0% | 0.0% | 0.0% | 12.5% | 18.5% | 0.000523 | -0.101 | 0.0% |
| depth_extrapolation | 200 | 34.0% | 0.043 | 1.0% | 0.0% | 0.0% | 10.6% | 15.7% | 0.000634 | -0.277 | 1.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split.

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| id | this model | 19.5% | 12.5% | 18.5% | -0.2 |
| id | this model, σ × 18.1 |  | 34.5% | 42.0% | -0.196 |
| id | SFT-only |  | 15.0% | 20.5% | -0.32 |
| id | SFT-only, σ × 18.1 (baseline) |  | 35.0% | 40.5% | -0.319 |
| depth_extrapolation | this model | 14.6% | 10.6% | 15.7% | -0.409 |
| depth_extrapolation | this model, σ × 18.1 |  | 34.8% | 41.9% | -0.405 |
| depth_extrapolation | SFT-only |  | 11.5% | 16.0% | -0.357 |
| depth_extrapolation | SFT-only, σ × 18.1 (baseline) |  | 33.5% | 38.0% | -0.354 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| depth_extrapolation | this model × (a) text scale | 46.5% | 58.1% | -0.397 |
| depth_extrapolation | this model × (a') text direct | 78.3% | 87.9% | -0.382 |
| depth_extrapolation | this model × (b) oracle per cell (upper bound) | 38.4% | 49.5% | -0.397 |
| depth_extrapolation | SFT-only × (a) text scale (**control**) | 45.5% | 52.5% | -0.341 |
| depth_extrapolation | SFT-only × (a') text direct | 77.0% | 86.0% | -0.332 |
| depth_extrapolation | SFT-only × (b) oracle per cell (upper bound) | 34.5% | 42.0% | -0.34 |
| id | this model × (a) text scale | 47.5% | 57.5% | -0.188 |
| id | this model × (a') text direct | 80.5% | 90.0% | -0.183 |
| id | this model × (b) oracle per cell (upper bound) | 37.5% | 48.5% | -0.183 |
| id | SFT-only × (a) text scale (**control**) | 42.5% | 53.5% | -0.314 |
| id | SFT-only × (a') text direct | 79.5% | 88.5% | -0.293 |
| id | SFT-only × (b) oracle per cell (upper bound) | 31.0% | 39.5% | -0.306 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| id | this model × A5 σ | 73.0% | 82.5% | -0.173 |
| id | SFT-only × A5 σ | 67.3% | 77.9% | -0.29 |
| depth_extrapolation | this model × A5 σ | 66.0% | 75.6% | -0.381 |
| depth_extrapolation | SFT-only × A5 σ | 66.5% | 75.5% | -0.321 |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 76.0% | 0.00166 | 0.0% | 0.0% | 30.0% |
| 2 | 50 | 32.0% | 0.046 | 0.0% | 0.0% | 6.0% |
| 3 | 50 | 24.0% | 0.0428 | 0.0% | 0.0% | 4.0% |
| 4 | 50 | 32.0% | 0.0518 | 0.0% | 0.0% | 10.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 63.2% | 0.00462 | 0.0% | 0.0% | 20.6% |
| energy | 64 | 23.4% | 0.0698 | 0.0% | 0.0% | 6.2% |
| kinematics | 68 | 35.3% | 0.0327 | 0.0% | 0.0% | 10.3% |

Mean score by rule (failures at the rule's floor): crps -0.255, binary 0.41, log_score -5.08, abs_rel_error -0.255, crps_log -0.2
Mean step score: 0.658; copied-given rate: 0.0%; steps block present: 100.0%; mean completion tokens: 194

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 34.0% | 0.043 | 1.0% | 0.0% | 10.6% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 50.7% | 0.00999 | 0.0% | 0.0% | 14.9% |
| energy | 66 | 4.5% | 0.162 | 1.5% | 0.0% | 3.1% |
| kinematics | 67 | 46.3% | 0.0121 | 1.5% | 0.0% | 13.6% |

Mean score by rule (failures at the rule's floor): crps -0.44, binary 0.34, log_score -6.24, abs_rel_error -0.441, crps_log -0.409
Mean step score: 0.656; copied-given rate: 0.0%; steps block present: 99.0%; mean completion tokens: 318

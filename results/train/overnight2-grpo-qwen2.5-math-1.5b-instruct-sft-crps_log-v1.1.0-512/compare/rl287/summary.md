# Eval summary: overnight2-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-compare-rl287

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 659 s
- torch_max_allocated_gib: 4.48
- torch_max_reserved_gib: 4.76
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.06

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 39.5% | 0.0282 | 0.0% | 0.0% | 0.0% | 14.0% | 18.5% | 0.000412 | -0.185 | 0.0% |
| depth_extrapolation | 200 | 29.0% | 0.0407 | 0.0% | 0.0% | 0.0% | 9.5% | 14.0% | 0.000642 | -0.297 | 0.0% |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 68.0% | 0.00195 | 0.0% | 0.0% | 30.0% |
| 2 | 50 | 32.0% | 0.0305 | 0.0% | 0.0% | 6.0% |
| 3 | 50 | 34.0% | 0.0585 | 0.0% | 0.0% | 10.0% |
| 4 | 50 | 24.0% | 0.0643 | 0.0% | 0.0% | 10.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 63.2% | 0.00322 | 0.0% | 0.0% | 22.1% |
| energy | 64 | 14.1% | 0.0713 | 0.0% | 0.0% | 4.7% |
| kinematics | 68 | 39.7% | 0.0256 | 0.0% | 0.0% | 14.7% |

Mean score by rule (failures at the rule's floor): crps -0.343, binary 0.395, log_score -4.92, abs_rel_error -0.344, crps_log -0.282
Mean step score: 0.649; copied-given rate: 0.5%; steps block present: 100.0%; mean completion tokens: 195

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 29.0% | 0.0407 | 0.0% | 0.0% | 9.5% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 41.8% | 0.0125 | 0.0% | 0.0% | 13.4% |
| energy | 66 | 1.5% | 0.266 | 0.0% | 0.0% | 0.0% |
| kinematics | 67 | 43.3% | 0.0153 | 0.0% | 0.0% | 14.9% |

Mean score by rule (failures at the rule's floor): crps -0.459, binary 0.29, log_score -6.15, abs_rel_error -0.46, crps_log -0.437
Mean step score: 0.698; copied-given rate: 0.5%; steps block present: 100.0%; mean completion tokens: 317

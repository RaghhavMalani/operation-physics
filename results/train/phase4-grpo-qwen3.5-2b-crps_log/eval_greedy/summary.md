# Eval summary: phase4-grpo-qwen3.5-2b-crps_log-eval-greedy

- model: `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 1616 s
- torch_max_allocated_gib: 6.30
- torch_max_reserved_gib: 7.04
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 7.67

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 9.0% | 0.00649 | 78.0% | 4.5% | 10.5% | 37.1% | 37.1% | 0.000673 | -4.28 | 66.5% |
| depth_extrapolation | 200 | 2.5% | 0.465 | 85.5% | 1.5% | 0.5% | 19.2% | 23.1% | 0.00296 | -4.48 | 84.5% |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 26.0% | 0.000609 | 50.0% | 4.0% | 39.1% |
| 2 | 50 | 4.0% | 0.958 | 88.0% | 2.0% | 40.0% |
| 3 | 50 | 4.0% | 0.345 | 86.0% | 4.0% | 40.0% |
| 4 | 50 | 2.0% | 0.34 | 88.0% | 8.0% | 0.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 8.8% | 0.0433 | 76.5% | 5.9% | 50.0% |
| energy | 64 | 12.5% | 0.00649 | 71.9% | 4.7% | 26.7% |
| kinematics | 68 | 5.9% | 0.0864 | 85.3% | 2.9% | 37.5% |

Mean score by rule (failures at the rule's floor): crps -4.21, binary 0.09, log_score -8.74, abs_rel_error -4.21, crps_log -4.29
Mean step score: 0.0444; copied-given rate: 2.9%; steps block present: 22.0%; mean completion tokens: 833

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 2.5% | 0.465 | 85.5% | 1.5% | 19.2% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 7.5% | 0.13 | 79.1% | 1.5% | 23.1% |
| energy | 66 | 0.0% | 0.594 | 83.3% | 3.0% | 22.2% |
| kinematics | 67 | 0.0% | 0.588 | 94.0% | 0.0% | 0.0% |

Mean score by rule (failures at the rule's floor): crps -4.44, binary 0.025, log_score -9.46, abs_rel_error -4.44, crps_log -4.49
Mean step score: 0.0437; copied-given rate: 3.8%; steps block present: 14.0%; mean completion tokens: 969

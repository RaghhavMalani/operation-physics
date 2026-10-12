# Eval summary: model-probe-greedy-qwen3.5-2b

- model: `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 2132 s
- torch_max_allocated_gib: 6.92
- torch_max_reserved_gib: 7.62
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 7.67

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 7.0% | 0.376 | 82.0% | 2.5% | 11.5% | 29.0% | 35.5% | 0.000139 | -4.41 | 69.5% |
| depth_extrapolation | 200 | 2.5% | 0.513 | 90.0% | 2.0% | 2.0% | 6.2% | 12.5% | 0.00198 | -4.69 | 87.5% |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 16.0% | 0.00212 | 68.0% | 2.0% | 33.3% |
| 2 | 50 | 4.0% | 0.96 | 84.0% | 2.0% | 14.3% |
| 3 | 50 | 8.0% | 0.00176 | 86.0% | 2.0% | 50.0% |
| 4 | 50 | 0.0% | 6.55 | 90.0% | 4.0% | 0.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 8.8% | 0.376 | 76.5% | 4.4% | 38.5% |
| energy | 64 | 7.8% | 0.337 | 81.2% | 3.1% | 10.0% |
| kinematics | 68 | 4.4% | 0.516 | 88.2% | 0.0% | 37.5% |

Mean score by rule (failures at the rule's floor): crps -4.38, binary 0.07, log_score -8.95, abs_rel_error -4.38, crps_log -4.41
Mean step score: 0.0544; copied-given rate: 3.2%; steps block present: 18.0%; mean completion tokens: 853

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 2.5% | 0.513 | 90.0% | 2.0% | 6.2% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 7.5% | 0.00275 | 85.1% | 1.5% | 11.1% |
| energy | 66 | 0.0% | 1 | 89.4% | 3.0% | 0.0% |
| kinematics | 67 | 0.0% | 2.37 | 95.5% | 1.5% | 0.0% |

Mean score by rule (failures at the rule's floor): crps -4.69, binary 0.025, log_score -9.63, abs_rel_error -4.69, crps_log -4.7
Mean step score: 0.0362; copied-given rate: 0.0%; steps block present: 9.5%; mean completion tokens: 983

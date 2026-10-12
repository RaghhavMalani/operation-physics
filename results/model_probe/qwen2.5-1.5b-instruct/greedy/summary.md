# Eval summary: model-probe-greedy-qwen2.5-1.5b-instruct

- model: `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 1944 s
- torch_max_allocated_gib: 4.48
- torch_max_reserved_gib: 4.76
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 6.84

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 2.5% | 0.983 | 51.5% | 2.5% | 0.5% | 3.3% | 5.4% | 0.00138 | -3.64 | 42.5% |
| depth_extrapolation | 200 | 1.0% | 0.891 | 54.0% | 0.5% | 0.0% | 1.1% | 2.2% | 0.00183 | -3.61 | 51.0% |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 8.0% | 0.635 | 56.0% | 2.0% | 14.3% |
| 2 | 50 | 0.0% | 0.983 | 44.0% | 2.0% | 0.0% |
| 3 | 50 | 0.0% | 0.986 | 60.0% | 2.0% | 0.0% |
| 4 | 50 | 2.0% | 0.999 | 46.0% | 4.0% | 0.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 2.9% | 0.988 | 30.9% | 1.5% | 2.2% |
| energy | 64 | 0.0% | 0.997 | 64.1% | 4.7% | 0.0% |
| kinematics | 68 | 4.4% | 0.897 | 60.3% | 1.5% | 7.7% |

Mean score by rule (failures at the rule's floor): crps -3.57, binary 0.025, log_score -9.65, abs_rel_error -3.57, crps_log -3.64
Mean step score: 0.0556; copied-given rate: 7.6%; steps block present: 38.0%; mean completion tokens: 665

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 1.0% | 0.891 | 54.0% | 0.5% | 1.1% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 3.0% | 0.921 | 38.8% | 1.5% | 2.5% |
| energy | 66 | 0.0% | 0.881 | 69.7% | 0.0% | 0.0% |
| kinematics | 67 | 0.0% | 0.805 | 53.7% | 0.0% | 0.0% |

Mean score by rule (failures at the rule's floor): crps -3.34, binary 0.01, log_score -9.81, abs_rel_error -3.34, crps_log -3.62
Mean step score: 0.0338; copied-given rate: 5.5%; steps block present: 45.5%; mean completion tokens: 735

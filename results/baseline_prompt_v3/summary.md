# Eval summary: baseline-prompt-v3-qwen2.5-1.5b-instruct

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- generator: `1.0.0+afcc2019`, reward rule: `crps`, prompt: `v3_worked_example`
- generation time: 885 s
- torch_max_allocated_gib: 4.56
- torch_max_reserved_gib: 4.69
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.89
- split `heldout_family` skipped: no implemented families for 'heldout'
- split `simulators` skipped: no implemented families for 'simulators'

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 1.5% | 0.931 | 49.0% | 1.0% | 3.5% | 3.0% | 3.0% | 0.000528 | -3.27 | 45.0% |
| depth_extrapolation | 200 | 0.0% | 0.982 | 75.0% | 0.0% | 4.5% | 2.0% | 2.0% | 0.00124 | -4.23 | 68.0% |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 0.0% | 0.961 | 14.0% | 2.0% | 2.4% |
| 2 | 50 | 2.0% | 0.757 | 70.0% | 0.0% | 6.7% |
| 3 | 50 | 2.0% | 0.737 | 52.0% | 2.0% | 0.0% |
| 4 | 50 | 2.0% | 0.998 | 60.0% | 0.0% | 5.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 2.9% | 0.968 | 42.6% | 0.0% | 5.1% |
| energy | 64 | 1.6% | 0.967 | 50.0% | 1.6% | 3.2% |
| kinematics | 68 | 0.0% | 0.824 | 54.4% | 1.5% | 0.0% |

Mean score by rule (failures at the rule's floor): crps -3.27, binary 0.015, log_score -9.8, abs_rel_error -3.28, crps_log -3.56
Mean step score: 0.0367; copied-given rate: 5.0%; steps block present: 29.5%; mean completion tokens: 341

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 0.0% | 0.982 | 75.0% | 0.0% | 2.0% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 0.0% | 0.999 | 64.2% | 0.0% | 0.0% |
| energy | 66 | 0.0% | 0.945 | 83.3% | 0.0% | 0.0% |
| kinematics | 67 | 0.0% | 0.976 | 77.6% | 0.0% | 6.7% |

Mean score by rule (failures at the rule's floor): crps -4.23, binary 0, log_score -9.94, abs_rel_error -4.23, crps_log -4.32
Mean step score: 0.0112; copied-given rate: 10.0%; steps block present: 25.0%; mean completion tokens: 436

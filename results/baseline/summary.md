# Eval summary: baseline-qwen2.5-1.5b-instruct

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- generator: `1.0.0+afcc2019`, reward rule: `crps`
- generation time: 451 s
- torch_max_allocated_gib: 3.76
- torch_max_reserved_gib: 3.84
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 4.95
- split `heldout_family` skipped: no implemented families for 'heldout'
- split `simulators` skipped: no implemented families for 'simulators'

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 0.0% | 0.921 | 26.5% | 0.0% | 20.0% | 0.7% | 1.4% | 0.00465 | -2.41 | 6.0% |
| depth_extrapolation | 200 | 0.0% | 0.934 | 21.5% | 0.5% | 20.0% | 0.0% | 3.2% | 0.0104 | -2.31 | 1.5% |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 0.0% | 0.934 | 30.0% | 0.0% | 2.9% |
| 2 | 50 | 0.0% | 0.958 | 30.0% | 0.0% | 0.0% |
| 3 | 50 | 0.0% | 0.915 | 30.0% | 0.0% | 0.0% |
| 4 | 50 | 0.0% | 0.859 | 16.0% | 0.0% | 0.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 0.0% | 0.83 | 17.6% | 0.0% | 1.8% |
| energy | 64 | 0.0% | 0.92 | 23.4% | 0.0% | 0.0% |
| kinematics | 68 | 0.0% | 0.982 | 38.2% | 0.0% | 0.0% |

Mean score by rule (failures at the rule's floor): crps -2.41, binary 0, log_score -9.71, abs_rel_error -2.42, crps_log -2.83
Mean step score: 0.01; copied-given rate: 8.8%; steps block present: 56.5%; mean completion tokens: 115

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 0.0% | 0.934 | 21.5% | 0.5% | 0.0% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 0.0% | 0.992 | 11.9% | 1.5% | 0.0% |
| energy | 66 | 0.0% | 0.895 | 30.3% | 0.0% | 0.0% |
| kinematics | 67 | 0.0% | 0.856 | 22.4% | 0.0% | 0.0% |

Mean score by rule (failures at the rule's floor): crps -2.31, binary 0, log_score -9.58, abs_rel_error -2.31, crps_log -2.8
Mean step score: 0.0025; copied-given rate: 3.8%; steps block present: 78.0%; mean completion tokens: 126

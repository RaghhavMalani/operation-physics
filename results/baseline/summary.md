# Eval summary: baseline-qwen2.5-1.5b-instruct

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- generator: `1.0.0+afcc2019`, reward rule: `crps`
- generation time: 621 s
- torch_max_allocated_gib: 3.76
- torch_max_reserved_gib: 3.84
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 4.93
- split `heldout_family` skipped: no implemented families for 'heldout'
- split `simulators` skipped: no implemented families for 'simulators'

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 0.5% | 0.947 | 35.0% | 1.5% | 20.5% | 0.8% | 0.8% | 0.00503 | -2.82 | 11.0% |
| depth_extrapolation | 200 | 0.0% | 0.866 | 32.5% | 0.5% | 18.5% | 0.0% | 0.7% | 0.00625 | -2.64 | 13.5% |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 2.0% | 0.93 | 38.0% | 0.0% | 3.2% |
| 2 | 50 | 0.0% | 0.947 | 42.0% | 2.0% | 0.0% |
| 3 | 50 | 0.0% | 0.883 | 28.0% | 2.0% | 0.0% |
| 4 | 50 | 0.0% | 0.98 | 32.0% | 2.0% | 0.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 0.0% | 0.94 | 30.9% | 1.5% | 0.0% |
| energy | 64 | 1.6% | 0.901 | 25.0% | 0.0% | 2.1% |
| kinematics | 68 | 0.0% | 0.957 | 48.5% | 2.9% | 0.0% |

Mean score by rule (failures at the rule's floor): crps -2.83, binary 0.005, log_score -9.77, abs_rel_error -2.83, crps_log -3.13
Mean step score: 0.00667; copied-given rate: 8.7%; steps block present: 49.5%; mean completion tokens: 214

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 0.0% | 0.866 | 32.5% | 0.5% | 0.0% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 0.0% | 0.966 | 41.8% | 0.0% | 0.0% |
| energy | 66 | 0.0% | 0.818 | 24.2% | 0.0% | 0.0% |
| kinematics | 67 | 0.0% | 0.824 | 31.3% | 1.5% | 0.0% |

Mean score by rule (failures at the rule's floor): crps -2.64, binary 0, log_score -9.89, abs_rel_error -2.65, crps_log -3.02
Mean step score: 0.00125; copied-given rate: 10.4%; steps block present: 67.0%; mean completion tokens: 255

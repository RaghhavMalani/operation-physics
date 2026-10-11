# Eval summary: baseline-prompt-v2-qwen2.5-1.5b-instruct

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- generator: `1.0.0+afcc2019`, reward rule: `crps`, prompt: `v2_reason_first`
- generation time: 551 s
- torch_max_allocated_gib: 3.83
- torch_max_reserved_gib: 3.93
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.02
- split `heldout_family` skipped: no implemented families for 'heldout'
- split `simulators` skipped: no implemented families for 'simulators'

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 0.0% | 0.934 | 31.5% | 2.0% | 26.0% | 2.3% | 2.3% | 0.00456 | -2.55 | 5.5% |
| depth_extrapolation | 200 | 0.0% | 0.905 | 36.0% | 0.0% | 28.5% | 0.8% | 3.9% | 0.00805 | -2.65 | 7.5% |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 0.0% | 0.948 | 30.0% | 0.0% | 2.9% |
| 2 | 50 | 0.0% | 0.959 | 32.0% | 4.0% | 3.1% |
| 3 | 50 | 0.0% | 0.842 | 38.0% | 2.0% | 0.0% |
| 4 | 50 | 0.0% | 0.974 | 26.0% | 2.0% | 2.8% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 0.0% | 0.907 | 22.1% | 1.5% | 1.9% |
| energy | 64 | 0.0% | 0.947 | 39.1% | 1.6% | 2.6% |
| kinematics | 68 | 0.0% | 0.958 | 33.8% | 2.9% | 2.3% |

Mean score by rule (failures at the rule's floor): crps -2.55, binary 0, log_score -9.74, abs_rel_error -2.56, crps_log -3.12
Mean step score: 0; copied-given rate: 10.5%; steps block present: 51.0%; mean completion tokens: 107

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 0.0% | 0.905 | 36.0% | 0.0% | 0.8% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 0.0% | 0.897 | 25.4% | 0.0% | 2.0% |
| energy | 66 | 0.0% | 0.934 | 50.0% | 0.0% | 0.0% |
| kinematics | 67 | 0.0% | 0.883 | 32.8% | 0.0% | 0.0% |

Mean score by rule (failures at the rule's floor): crps -2.65, binary 0, log_score -9.55, abs_rel_error -2.65, crps_log -3.1
Mean step score: 0.00375; copied-given rate: 11.7%; steps block present: 63.0%; mean completion tokens: 141

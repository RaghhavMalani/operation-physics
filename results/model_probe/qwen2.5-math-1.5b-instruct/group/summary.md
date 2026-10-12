# Eval summary: model-probe-group-qwen2.5-math-1.5b-instruct

- model: `Qwen/Qwen2.5-Math-1.5B-Instruct` @ `aafeb0fc6f22cbf0eaeed126eff8be45b0360a35`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 1750 s
- torch_max_allocated_gib: 4.47
- torch_max_reserved_gib: 4.76
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.80

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 384 | 0.3% | 64.8 | 99.5% | 0.0% | 0.0% | 0.0% | 0.0% | 1.31 | -4.99 | 17.2% |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 96 | 0.0% | – | 100.0% | 0.0% | – |
| 2 | 96 | 1.0% | 0.00243 | 99.0% | 0.0% | 0.0% |
| 3 | 96 | 0.0% | 129 | 99.0% | 0.0% | 0.0% |
| 4 | 96 | 0.0% | – | 100.0% | 0.0% | – |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 128 | 0.8% | 0.00243 | 99.2% | 0.0% | 0.0% |
| energy | 128 | 0.0% | 129 | 99.2% | 0.0% | 0.0% |
| kinematics | 128 | 0.0% | – | 100.0% | 0.0% | – |

Mean score by rule (failures at the rule's floor): crps -4.99, binary 0.0026, log_score -9.97, abs_rel_error -4.99, crps_log -4.99
Mean step score: 0.00174; copied-given rate: 0.0%; steps block present: 0.5%; mean completion tokens: 708

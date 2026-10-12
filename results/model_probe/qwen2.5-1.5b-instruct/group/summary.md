# Eval summary: model-probe-group-qwen2.5-1.5b-instruct

- model: `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 1699 s
- torch_max_allocated_gib: 4.47
- torch_max_reserved_gib: 4.76
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 6.97

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 384 | 0.8% | 0.966 | 44.8% | 6.0% | 2.1% | 2.6% | 2.6% | 0.00195 | -3.71 | 5.5% |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 96 | 1.0% | 0.828 | 68.8% | 5.2% | 4.0% |
| 2 | 96 | 0.0% | 0.988 | 38.5% | 4.2% | 0.0% |
| 3 | 96 | 0.0% | 0.985 | 39.6% | 7.3% | 0.0% |
| 4 | 96 | 2.1% | 0.902 | 32.3% | 7.3% | 6.9% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 128 | 1.6% | 0.995 | 44.5% | 6.2% | 4.8% |
| energy | 128 | 0.0% | 0.957 | 44.5% | 6.2% | 1.6% |
| kinematics | 128 | 0.8% | 0.953 | 45.3% | 5.5% | 1.6% |

Mean score by rule (failures at the rule's floor): crps -3.4, binary 0.00781, log_score -9.82, abs_rel_error -3.4, crps_log -3.71
Mean step score: 0.0104; copied-given rate: 2.1%; steps block present: 41.7%; mean completion tokens: 489

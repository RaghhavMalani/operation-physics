# Eval summary: model-probe-group-qwen3.5-2b

- model: `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 1884 s
- torch_max_allocated_gib: 5.43
- torch_max_reserved_gib: 5.71
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 6.19

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 384 | 3.4% | 0.832 | 87.8% | 1.0% | 8.9% | 27.9% | 27.9% | 0.00162 | -4.67 | 76.3% |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 96 | 11.5% | 0.108 | 74.0% | 1.0% | 41.7% |
| 2 | 96 | 2.1% | 0.989 | 88.5% | 1.0% | 20.0% |
| 3 | 96 | 0.0% | 1.56 | 92.7% | 1.0% | 0.0% |
| 4 | 96 | 0.0% | 0.998 | 95.8% | 1.0% | 0.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 128 | 3.1% | 0.994 | 84.4% | 3.1% | 25.0% |
| energy | 128 | 2.3% | 0.755 | 92.2% | 0.0% | 20.0% |
| kinematics | 128 | 4.7% | 0.5 | 86.7% | 0.0% | 35.3% |

Mean score by rule (failures at the rule's floor): crps -4.61, binary 0.0339, log_score -9.5, abs_rel_error -4.61, crps_log -4.68
Mean step score: 0.0185; copied-given rate: 0.0%; steps block present: 12.0%; mean completion tokens: 885

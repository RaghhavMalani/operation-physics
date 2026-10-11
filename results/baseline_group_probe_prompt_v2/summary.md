# Eval summary: baseline-group-probe-prompt-v2

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- generator: `1.0.0+afcc2019`, reward rule: `crps`, prompt: `v2_reason_first`
- generation time: 607 s
- torch_max_allocated_gib: 3.80
- torch_max_reserved_gib: 3.90
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.09

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 384 | 0.3% | 0.994 | 43.8% | 4.7% | 18.5% | 2.0% | 4.0% | 0.0222 | -3.5 | 15.9% |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 96 | 0.0% | 0.969 | 43.8% | 4.2% | 0.0% |
| 2 | 96 | 0.0% | 0.98 | 37.5% | 6.2% | 3.7% |
| 3 | 96 | 0.0% | 0.994 | 44.8% | 4.2% | 0.0% |
| 4 | 96 | 1.0% | 2.1 | 49.0% | 4.2% | 4.4% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 128 | 0.0% | 1.86 | 47.7% | 5.5% | 1.7% |
| energy | 128 | 0.8% | 0.938 | 42.2% | 2.3% | 4.2% |
| kinematics | 128 | 0.0% | 0.996 | 41.4% | 6.2% | 0.0% |

Mean score by rule (failures at the rule's floor): crps -3.5, binary 0.0026, log_score -9.62, abs_rel_error -3.51, crps_log -3.68
Mean step score: 0.00231; copied-given rate: 0.0%; steps block present: 40.1%; mean completion tokens: 199

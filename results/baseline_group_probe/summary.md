# Eval summary: baseline-group-probe-qwen2.5-1.5b-instruct

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- generator: `1.0.0+afcc2019`, reward rule: `crps`, prompt: `v1`
- generation time: 507 s
- torch_max_allocated_gib: 3.74
- torch_max_reserved_gib: 3.92
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.01

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 384 | 0.0% | 0.982 | 32.6% | 7.0% | 22.4% | 1.7% | 2.6% | 0.0164 | -3.18 | 6.5% |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 96 | 0.0% | 0.962 | 43.8% | 4.2% | 4.0% |
| 2 | 96 | 0.0% | 0.94 | 25.0% | 8.3% | 3.1% |
| 3 | 96 | 0.0% | 2.38 | 38.5% | 5.2% | 0.0% |
| 4 | 96 | 0.0% | 1.14 | 22.9% | 10.4% | 0.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 128 | 0.0% | 1.04 | 30.5% | 7.0% | 0.0% |
| energy | 128 | 0.0% | 0.899 | 32.8% | 7.8% | 1.3% |
| kinematics | 128 | 0.0% | 0.992 | 34.4% | 6.2% | 3.9% |

Mean score by rule (failures at the rule's floor): crps -3.19, binary 0, log_score -9.76, abs_rel_error -3.2, crps_log -3.24
Mean step score: 0.00347; copied-given rate: 1.7%; steps block present: 51.8%; mean completion tokens: 172

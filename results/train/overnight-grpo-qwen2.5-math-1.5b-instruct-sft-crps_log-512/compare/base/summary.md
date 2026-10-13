# Eval summary: overnight-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-512-compare-base

- model: `Qwen/Qwen2.5-Math-1.5B-Instruct` @ `aafeb0fc6f22cbf0eaeed126eff8be45b0360a35`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 885 s
- torch_max_allocated_gib: 4.48
- torch_max_reserved_gib: 4.76
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.67

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 0.0% | – | 100.0% | 0.0% | 0.0% | – | – | – | -5 | 76.5% |
| depth_extrapolation | 200 | 0.0% | – | 100.0% | 0.0% | 0.0% | – | – | – | -5 | 98.0% |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 0.0% | – | 100.0% | 0.0% | – |
| 2 | 50 | 0.0% | – | 100.0% | 0.0% | – |
| 3 | 50 | 0.0% | – | 100.0% | 0.0% | – |
| 4 | 50 | 0.0% | – | 100.0% | 0.0% | – |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 0.0% | – | 100.0% | 0.0% | – |
| energy | 64 | 0.0% | – | 100.0% | 0.0% | – |
| kinematics | 68 | 0.0% | – | 100.0% | 0.0% | – |

Mean score by rule (failures at the rule's floor): crps -5, binary 0, log_score -10, abs_rel_error -5, crps_log -5
Mean step score: 0; copied-given rate: –; steps block present: 0.0%; mean completion tokens: 484

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 0.0% | – | 100.0% | 0.0% | – |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 0.0% | – | 100.0% | 0.0% | – |
| energy | 66 | 0.0% | – | 100.0% | 0.0% | – |
| kinematics | 67 | 0.0% | – | 100.0% | 0.0% | – |

Mean score by rule (failures at the rule's floor): crps -5, binary 0, log_score -10, abs_rel_error -5, crps_log -5
Mean step score: 0; copied-given rate: –; steps block present: 0.0%; mean completion tokens: 511

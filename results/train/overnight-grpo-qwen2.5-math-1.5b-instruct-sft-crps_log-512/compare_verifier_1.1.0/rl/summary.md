# Eval summary: overnight-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-512-compare-rl

- model: `results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 636 s
- torch_max_allocated_gib: 4.48
- torch_max_reserved_gib: 4.76
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.53

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 200 | 40.0% | 0.0254 | 0.0% | 0.0% | 0.0% | 15.5% | 20.0% | 0.000416 | -0.178 | 0.0% |
| depth_extrapolation | 200 | 31.0% | 0.042 | 0.0% | 0.0% | 0.0% | 10.0% | 15.0% | 0.000642 | -0.221 | 0.0% |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 50 | 70.0% | 0.00166 | 0.0% | 0.0% | 34.0% |
| 2 | 50 | 32.0% | 0.0293 | 0.0% | 0.0% | 6.0% |
| 3 | 50 | 30.0% | 0.0569 | 0.0% | 0.0% | 10.0% |
| 4 | 50 | 28.0% | 0.0613 | 0.0% | 0.0% | 12.0% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 68 | 61.8% | 0.00272 | 0.0% | 0.0% | 26.5% |
| energy | 64 | 15.6% | 0.0743 | 0.0% | 0.0% | 4.7% |
| kinematics | 68 | 41.2% | 0.024 | 0.0% | 0.0% | 14.7% |

Mean score by rule (failures at the rule's floor): crps -0.331, binary 0.4, log_score -4.92, abs_rel_error -0.331, crps_log -0.275
Mean step score: 0.648; copied-given rate: 0.5%; steps block present: 100.0%; mean completion tokens: 195

## depth_extrapolation: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 5 | 200 | 31.0% | 0.042 | 0.0% | 0.0% | 10.0% |

## depth_extrapolation: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 67 | 44.8% | 0.0125 | 0.0% | 0.0% | 14.9% |
| energy | 66 | 1.5% | 0.238 | 0.0% | 0.0% | 0.0% |
| kinematics | 67 | 46.3% | 0.0148 | 0.0% | 0.0% | 14.9% |

Mean score by rule (failures at the rule's floor): crps -0.409, binary 0.31, log_score -6.11, abs_rel_error -0.41, crps_log -0.361
Mean step score: 0.7; copied-given rate: 0.5%; steps block present: 100.0%; mean completion tokens: 316

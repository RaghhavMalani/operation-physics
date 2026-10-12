# Eval summary: model-probe-group-qwen2.5-math-1.5b-instruct-sft

- model: `results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 455 s
- torch_max_allocated_gib: 4.47
- torch_max_reserved_gib: 5.13
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 6.01

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 384 | 36.2% | 0.0424 | 0.3% | 0.5% | 0.0% | 20.2% | 23.6% | 0.000804 | -0.409 | 0.0% |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 96 | 78.1% | 0.000302 | 0.0% | 0.0% | 56.2% |
| 2 | 96 | 21.9% | 0.305 | 0.0% | 0.0% | 3.1% |
| 3 | 96 | 33.3% | 0.0432 | 0.0% | 2.1% | 19.1% |
| 4 | 96 | 11.5% | 0.176 | 1.0% | 0.0% | 2.1% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 128 | 51.6% | 0.00909 | 0.0% | 0.8% | 18.1% |
| energy | 128 | 19.5% | 0.245 | 0.0% | 0.0% | 13.3% |
| kinematics | 128 | 37.5% | 0.0271 | 0.8% | 0.8% | 29.4% |

Mean score by rule (failures at the rule's floor): crps -0.516, binary 0.362, log_score -5.01, abs_rel_error -0.517, crps_log -0.49
Mean step score: 0.541; copied-given rate: 0.0%; steps block present: 99.7%; mean completion tokens: 197

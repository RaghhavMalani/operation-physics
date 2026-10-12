# Eval summary: model-probe-group-qwen3.5-2b-sft

- model: `results/sft/qwen3.5-2b/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 571 s
- torch_max_allocated_gib: 4.82
- torch_max_reserved_gib: 5.09
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.92

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| id | 384 | 35.7% | 0.036 | 0.3% | 0.0% | 0.0% | 30.5% | 37.3% | 0.00519 | -0.323 | 0.3% |

## id: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 96 | 77.1% | 0.000302 | 0.0% | 0.0% | 69.8% |
| 2 | 96 | 25.0% | 0.0967 | 0.0% | 0.0% | 20.8% |
| 3 | 96 | 27.1% | 0.0691 | 0.0% | 0.0% | 21.9% |
| 4 | 96 | 13.5% | 0.245 | 1.0% | 0.0% | 9.5% |

## id: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 128 | 54.7% | 0.00567 | 0.0% | 0.0% | 46.9% |
| energy | 128 | 18.8% | 0.0666 | 0.0% | 0.0% | 17.2% |
| kinematics | 128 | 33.6% | 0.0748 | 0.8% | 0.0% | 27.6% |

Mean score by rule (failures at the rule's floor): crps -0.525, binary 0.357, log_score -4.01, abs_rel_error -0.527, crps_log -0.409
Mean step score: 0.579; copied-given rate: 0.3%; steps block present: 99.7%; mean completion tokens: 205

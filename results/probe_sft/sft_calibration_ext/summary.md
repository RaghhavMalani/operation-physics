# Eval summary: probe-sft-calibration-ext

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 2318 s
- torch_max_allocated_gib: 4.48
- torch_max_reserved_gib: 4.75
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.22

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| calibration | 1600 | 39.9% | 0.0221 | 0.1% | 0.0% | 0.0% | 12.4% | 16.5% | 0.000502 | -0.158 | 0.1% |

## calibration: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 400 | 64.2% | 0.00293 | 0.2% | 0.0% | 26.8% |
| 2 | 400 | 37.5% | 0.0213 | 0.2% | 0.0% | 9.8% |
| 3 | 400 | 30.0% | 0.0401 | 0.0% | 0.0% | 6.0% |
| 4 | 400 | 27.8% | 0.0501 | 0.0% | 0.0% | 7.0% |

## calibration: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 532 | 57.5% | 0.00585 | 0.0% | 0.0% | 16.0% |
| energy | 532 | 19.9% | 0.0561 | 0.0% | 0.0% | 6.6% |
| kinematics | 536 | 42.2% | 0.023 | 0.4% | 0.0% | 14.6% |

Mean score by rule (failures at the rule's floor): crps -0.27, binary 0.399, log_score -5.21, abs_rel_error -0.271, crps_log -0.251
Mean step score: 0.619; copied-given rate: 0.9%; steps block present: 99.9%; mean completion tokens: 196

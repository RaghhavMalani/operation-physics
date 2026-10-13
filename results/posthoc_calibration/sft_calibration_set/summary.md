# Eval summary: posthoc-sft-calibration-set

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 562 s
- torch_max_allocated_gib: 4.47
- torch_max_reserved_gib: 4.63
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.37

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| calibration | 400 | 40.2% | 0.0246 | 0.2% | 0.2% | 0.2% | 13.6% | 17.8% | 0.000621 | -0.158 | 0.0% |

## calibration: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 100 | 62.0% | 0.00447 | 0.0% | 0.0% | 18.0% |
| 2 | 100 | 36.0% | 0.0249 | 0.0% | 0.0% | 15.0% |
| 3 | 100 | 37.0% | 0.0314 | 1.0% | 1.0% | 12.2% |
| 4 | 100 | 26.0% | 0.0617 | 0.0% | 0.0% | 9.0% |

## calibration: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 132 | 63.6% | 0.00478 | 0.0% | 0.0% | 24.2% |
| energy | 132 | 17.4% | 0.0693 | 0.8% | 0.8% | 5.4% |
| kinematics | 136 | 39.7% | 0.0237 | 0.0% | 0.0% | 11.0% |

Mean score by rule (failures at the rule's floor): crps -0.266, binary 0.403, log_score -5.02, abs_rel_error -0.267, crps_log -0.251
Mean step score: 0.625; copied-given rate: 0.5%; steps block present: 99.8%; mean completion tokens: 197

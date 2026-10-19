# Eval summary: x2-sft-4bit-k8-chunk2

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 1217 s
- torch_max_allocated_gib: 4.77
- torch_max_reserved_gib: 5.75
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 6.98

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| x2 | 1600 | 31.9% | 0.0423 | 0.2% | 0.1% | 0.1% | 13.1% | 16.9% | 0.000809 | -0.209 | 0.1% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split.

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| x2 | this model | 19.5% | 13.1% | 16.9% | -0.289 |
| x2 | this model, σ × 18.1 |  | 36.0% | 42.4% | -0.285 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| x2 | this model × (a) text scale | 48.9% | 57.1% | -0.298 |
| x2 | this model × (a') text direct | 75.5% | 85.3% | -0.268 |
| x2 | this model × (b) oracle per cell (upper bound) | 30.8% | 38.7% | -0.283 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| x2 | this model × A5 σ | 60.2% | 73.1% | -0.265 |

## x2: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 400 | 54.5% | 0.00581 | 0.0% | 0.0% | 20.8% |
| 2 | 400 | 34.5% | 0.0356 | 0.0% | 0.0% | 17.2% |
| 3 | 400 | 21.2% | 0.0677 | 0.0% | 0.2% | 7.3% |
| 4 | 400 | 17.2% | 0.0736 | 0.8% | 0.0% | 7.1% |

## x2: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 544 | 53.9% | 0.00824 | 0.0% | 0.2% | 20.3% |
| energy | 544 | 12.3% | 0.0716 | 0.0% | 0.0% | 5.0% |
| kinematics | 512 | 29.3% | 0.0744 | 0.6% | 0.0% | 14.1% |

Mean score by rule (failures at the rule's floor): crps -0.335, binary 0.319, log_score -5.87, abs_rel_error -0.337, crps_log -0.289
Mean step score: 0.538; copied-given rate: 0.2%; steps block present: 99.6%; mean completion tokens: 196

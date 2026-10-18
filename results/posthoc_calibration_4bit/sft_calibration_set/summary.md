# Eval summary: posthoc-sft-4bit-calibration-set

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 438 s
- torch_max_allocated_gib: 2.93
- torch_max_reserved_gib: 3.44
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 4.92

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| calibration | 400 | 41.5% | 0.019 | 0.0% | 0.0% | 0.0% | 14.2% | 17.5% | 0.000617 | -0.098 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split.

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| calibration | this model | 19.8% | 14.2% | 17.5% | -0.194 |
| calibration | this model, σ × 18.1 |  | 38.2% | 47.2% | -0.19 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| calibration | this model × (a) text scale | 51.5% | 59.5% | -0.187 |
| calibration | this model × (a') text direct | 82.0% | 90.0% | -0.183 |
| calibration | this model × (b) oracle per cell (upper bound) | 38.8% | 47.2% | -0.187 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| calibration | this model × A5 σ | 75.2% | 83.7% | -0.174 |

## calibration: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 100 | 65.0% | 0.00461 | 0.0% | 0.0% | 25.0% |
| 2 | 100 | 41.0% | 0.0195 | 0.0% | 0.0% | 19.0% |
| 3 | 100 | 34.0% | 0.0233 | 0.0% | 0.0% | 7.0% |
| 4 | 100 | 26.0% | 0.0566 | 0.0% | 0.0% | 6.0% |

## calibration: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 132 | 60.6% | 0.00608 | 0.0% | 0.0% | 21.2% |
| energy | 132 | 22.7% | 0.0506 | 0.0% | 0.0% | 7.6% |
| kinematics | 136 | 41.2% | 0.0211 | 0.0% | 0.0% | 14.0% |

Mean score by rule (failures at the rule's floor): crps -0.208, binary 0.415, log_score -5.32, abs_rel_error -0.209, crps_log -0.194
Mean step score: 0.639; copied-given rate: 0.2%; steps block present: 100.0%; mean completion tokens: 196

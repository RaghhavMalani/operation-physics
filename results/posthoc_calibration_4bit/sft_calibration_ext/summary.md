# Eval summary: posthoc-sft-4bit-calibration-ext

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 1644 s
- torch_max_allocated_gib: 2.95
- torch_max_reserved_gib: 3.46
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 4.57

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| calibration | 1600 | 39.4% | 0.0224 | 0.1% | 0.0% | 0.0% | 13.0% | 17.1% | 0.000586 | -0.135 | 0.1% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split.

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| calibration | this model | 18.4% | 13.0% | 17.1% | -0.228 |
| calibration | this model, σ × 18.1 |  | 37.3% | 44.3% | -0.225 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| calibration | this model × (a) text scale | 50.5% | 58.8% | -0.219 |
| calibration | this model × (a') text direct | 81.6% | 90.6% | -0.199 |
| calibration | this model × (b) oracle per cell (upper bound) | 41.3% | 50.2% | -0.222 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| calibration | this model × A5 σ | 72.2% | 84.6% | -0.203 |

## calibration: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 400 | 64.0% | 0.00372 | 0.0% | 0.0% | 27.5% |
| 2 | 400 | 38.0% | 0.0277 | 0.0% | 0.0% | 11.2% |
| 3 | 400 | 31.0% | 0.0385 | 0.2% | 0.0% | 7.0% |
| 4 | 400 | 24.8% | 0.0468 | 0.0% | 0.0% | 6.2% |

## calibration: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 532 | 57.0% | 0.00642 | 0.0% | 0.0% | 17.5% |
| energy | 532 | 19.5% | 0.0564 | 0.0% | 0.0% | 7.5% |
| kinematics | 536 | 41.8% | 0.0225 | 0.2% | 0.0% | 14.0% |

Mean score by rule (failures at the rule's floor): crps -0.264, binary 0.394, log_score -5.22, abs_rel_error -0.264, crps_log -0.228
Mean step score: 0.62; copied-given rate: 0.9%; steps block present: 99.9%; mean completion tokens: 194

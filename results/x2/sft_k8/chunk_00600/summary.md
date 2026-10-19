# Eval summary: x2-sft-4bit-k8-chunk3

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 1228 s
- torch_max_allocated_gib: 4.75
- torch_max_reserved_gib: 5.72
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 6.95

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| x2 | 1600 | 35.4% | 0.0414 | 0.2% | 0.1% | 0.0% | 16.4% | 19.9% | 0.000698 | -0.286 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split.

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| x2 | this model | 20.3% | 16.4% | 19.9% | -0.37 |
| x2 | this model, σ × 18.1 |  | 36.5% | 42.6% | -0.367 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| x2 | this model × (a) text scale | 47.2% | 53.7% | -0.366 |
| x2 | this model × (a') text direct | 71.5% | 81.6% | -0.333 |
| x2 | this model × (b) oracle per cell (upper bound) | 34.2% | 42.5% | -0.36 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| x2 | this model × A5 σ | 61.3% | 74.4% | -0.336 |

## x2: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 400 | 55.8% | 0.00709 | 0.2% | 0.0% | 31.3% |
| 2 | 400 | 35.2% | 0.0386 | 0.0% | 0.2% | 13.5% |
| 3 | 400 | 30.2% | 0.0649 | 0.2% | 0.0% | 11.8% |
| 4 | 400 | 20.5% | 0.091 | 0.2% | 0.0% | 8.8% |

## x2: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 544 | 61.8% | 0.00531 | 0.0% | 0.0% | 27.9% |
| energy | 512 | 15.6% | 0.132 | 0.2% | 0.2% | 5.9% |
| kinematics | 544 | 27.8% | 0.0666 | 0.4% | 0.0% | 14.6% |

Mean score by rule (failures at the rule's floor): crps -0.397, binary 0.354, log_score -5.48, abs_rel_error -0.398, crps_log -0.37
Mean step score: 0.565; copied-given rate: 0.7%; steps block present: 99.6%; mean completion tokens: 197

# Eval summary: x2-sft-4bit-k8-chunk5

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 1208 s
- torch_max_allocated_gib: 4.77
- torch_max_reserved_gib: 5.72
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 6.95

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| x2 | 1600 | 31.4% | 0.0451 | 0.5% | 0.2% | 0.1% | 11.3% | 14.2% | 0.000701 | -0.283 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split.

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| x2 | this model | 21.0% | 11.3% | 14.2% | -0.357 |
| x2 | this model, σ × 18.1 |  | 31.4% | 38.4% | -0.351 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| x2 | this model × (a) text scale | 42.8% | 51.6% | -0.352 |
| x2 | this model × (a') text direct | 70.2% | 79.6% | -0.328 |
| x2 | this model × (b) oracle per cell (upper bound) | 32.6% | 40.8% | -0.344 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| x2 | this model × A5 σ | 58.2% | 71.2% | -0.327 |

## x2: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 400 | 53.2% | 0.00879 | 0.5% | 0.0% | 16.6% |
| 2 | 400 | 27.8% | 0.0518 | 0.5% | 0.5% | 12.4% |
| 3 | 400 | 30.0% | 0.0489 | 0.0% | 0.2% | 11.8% |
| 4 | 400 | 14.8% | 0.157 | 1.0% | 0.2% | 4.3% |

## x2: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 544 | 46.5% | 0.0131 | 0.7% | 0.4% | 14.9% |
| energy | 544 | 16.0% | 0.0843 | 0.0% | 0.2% | 7.4% |
| kinematics | 512 | 31.8% | 0.0548 | 0.8% | 0.2% | 11.6% |

Mean score by rule (failures at the rule's floor): crps -0.372, binary 0.314, log_score -6.19, abs_rel_error -0.374, crps_log -0.357
Mean step score: 0.495; copied-given rate: 1.0%; steps block present: 99.4%; mean completion tokens: 197

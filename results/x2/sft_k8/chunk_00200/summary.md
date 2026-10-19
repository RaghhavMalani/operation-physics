# Eval summary: x2-sft-4bit-k8-chunk1

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 1233 s
- torch_max_allocated_gib: 4.76
- torch_max_reserved_gib: 5.75
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 7.00

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| x2 | 1600 | 34.1% | 0.0341 | 0.2% | 0.2% | 0.1% | 15.6% | 19.3% | 0.000851 | -0.262 | 0.1% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split.

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| x2 | this model | 22.5% | 15.6% | 19.3% | -0.342 |
| x2 | this model, σ × 18.1 |  | 36.5% | 43.7% | -0.341 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| x2 | this model × (a) text scale | 47.8% | 56.1% | -0.348 |
| x2 | this model × (a') text direct | 73.6% | 84.1% | -0.317 |
| x2 | this model × (b) oracle per cell (upper bound) | 29.8% | 37.2% | -0.335 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| x2 | this model × A5 σ | 61.4% | 75.0% | -0.311 |

## x2: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 400 | 63.2% | 0.00361 | 0.0% | 0.0% | 34.5% |
| 2 | 400 | 36.0% | 0.0339 | 0.8% | 0.2% | 15.4% |
| 3 | 400 | 20.8% | 0.0938 | 0.2% | 0.0% | 6.5% |
| 4 | 400 | 16.5% | 0.0795 | 0.0% | 0.5% | 6.0% |

## x2: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 512 | 55.3% | 0.00736 | 0.0% | 0.2% | 21.9% |
| energy | 544 | 19.3% | 0.0881 | 0.2% | 0.2% | 14.8% |
| kinematics | 544 | 29.0% | 0.0377 | 0.6% | 0.2% | 10.6% |

Mean score by rule (failures at the rule's floor): crps -0.349, binary 0.341, log_score -5.62, abs_rel_error -0.35, crps_log -0.342
Mean step score: 0.533; copied-given rate: 1.6%; steps block present: 99.6%; mean completion tokens: 196

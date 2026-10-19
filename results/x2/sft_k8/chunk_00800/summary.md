# Eval summary: x2-sft-4bit-k8-chunk4

- model: `/home/flash2404/operation-physics/results/sft/qwen2.5-math-1.5b-instruct/merged` @ `None`
- generator: `1.1.0+c735a0d4`, reward rule: `crps_log`, prompt: `v4_worked_example`
- generation time: 1200 s
- torch_max_allocated_gib: 4.76
- torch_max_reserved_gib: 5.67
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 6.91

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| x2 | 1600 | 33.8% | 0.0331 | 0.1% | 0.1% | 0.1% | 11.7% | 16.2% | 0.000673 | -0.242 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split.

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| x2 | this model | 22.9% | 11.7% | 16.2% | -0.321 |
| x2 | this model, σ × 18.1 |  | 34.6% | 42.2% | -0.317 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| x2 | this model × (a) text scale | 46.3% | 53.4% | -0.321 |
| x2 | this model × (a') text direct | 71.4% | 82.3% | -0.295 |
| x2 | this model × (b) oracle per cell (upper bound) | 31.7% | 39.8% | -0.312 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged.

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| x2 | this model × A5 σ | 61.3% | 74.5% | -0.293 |

## x2: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 400 | 63.7% | 0.00317 | 0.2% | 0.0% | 26.8% |
| 2 | 400 | 33.8% | 0.0273 | 0.0% | 0.2% | 11.5% |
| 3 | 400 | 20.8% | 0.0847 | 0.0% | 0.2% | 5.8% |
| 4 | 400 | 17.0% | 0.111 | 0.2% | 0.0% | 2.8% |

## x2: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 512 | 52.1% | 0.00839 | 0.2% | 0.4% | 17.1% |
| energy | 544 | 15.6% | 0.0649 | 0.2% | 0.0% | 4.6% |
| kinematics | 544 | 34.7% | 0.0527 | 0.0% | 0.0% | 13.8% |

Mean score by rule (failures at the rule's floor): crps -0.333, binary 0.338, log_score -5.7, abs_rel_error -0.335, crps_log -0.321
Mean step score: 0.527; copied-given rate: 0.6%; steps block present: 99.7%; mean completion tokens: 194

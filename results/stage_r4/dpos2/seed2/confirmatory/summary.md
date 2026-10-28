# Eval summary: stage-r4-dpos2-seed2-confirmatory

- model: `.cache/huggingface/hub/models--unsloth--Qwen2.5-Math-1.5B-Instruct-bnb-4bit/snapshots/e24e9530845f4e3a6648343796883cb509935039` @ `e24e9530845f4e3a6648343796883cb509935039`
- generator: `2.0.0+bdbb2279`, reward rule: `crps_log`, prompt: `v5_two_channel`
- generation time: 10245 s
- torch_max_allocated_gib: 3.68
- torch_max_reserved_gib: 4.49
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.22

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a6_pairs | 1020 | 10.9% | 0.0763 | 0.1% | 0.6% | 0.1% | 41.3% | 62.0% | 0.0511 | -0.186 | 0.0% |
| a6b_pairs | 1020 | 12.5% | 0.0605 | 0.1% | 0.4% | 0.0% | 39.3% | 58.3% | 0.0293 | -0.391 | 0.0% |
| a6c_pairs | 1020 | 14.6% | 0.0593 | 0.4% | 0.4% | 0.2% | 42.7% | 62.8% | 0.0435 | -0.107 | 0.2% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split (**not comparable here**: different generator or verifier version, so omitted).

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| a6_pairs | this model | 0.1% | 41.3% | 62.0% | -0.302 |
| a6_pairs | this model, σ × 18.1 |  | 89.6% | 94.8% | -0.5 |
| a6b_pairs | this model | 0.0% | 39.3% | 58.3% | -0.498 |
| a6b_pairs | this model, σ × 18.1 |  | 82.5% | 86.8% | -0.587 |
| a6c_pairs | this model | 0.0% | 42.7% | 62.8% | -0.213 |
| a6c_pairs | this model, σ × 18.1 |  | 92.4% | 95.9% | -0.349 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × (a) text scale | 95.5% | 97.7% | -1.02 |
| a6_pairs | this model × (a') text direct | 60.9% | 74.8% | -0.297 |
| a6_pairs | this model × (b) oracle per cell (upper bound) | 59.4% | 74.8% | -0.296 |
| a6b_pairs | this model × (a) text scale | 88.4% | 91.6% | -1.06 |
| a6b_pairs | this model × (a') text direct | 59.7% | 72.5% | -0.493 |
| a6b_pairs | this model × (b) oracle per cell (upper bound) | 60.8% | 73.5% | -0.491 |
| a6c_pairs | this model × (a) text scale | 97.0% | 98.3% | -0.816 |
| a6c_pairs | this model × (a') text direct | 67.8% | 82.0% | -0.214 |
| a6c_pairs | this model × (b) oracle per cell (upper bound) | 61.7% | 79.2% | -0.205 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × A5 σ | 34.3% | 45.8% | -0.3 |
| a6b_pairs | this model × A5 σ | 48.8% | 62.3% | -0.49 |
| a6c_pairs | this model × A5 σ | 46.0% | 60.2% | -0.215 |

## a6_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 333 | 19.1% | 32.7% | 72.3% | -0.0189 |
| 0.02 | 340 | 339 | 20.3% | 46.0% | 79.7% | -0.00966 |
| 0.05 | 340 | 338 | 24.1% | 45.3% | 69.5% | -0.021 |

## a6_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 14.7% | 0.0587 | 0.5% | 0.0% | 53.7% |
| 2 | 204 | 10.3% | 0.0726 | 0.0% | 1.5% | 43.3% |
| 3 | 204 | 8.3% | 0.0774 | 0.0% | 0.5% | 37.9% |
| 4 | 204 | 9.8% | 0.0787 | 0.0% | 0.5% | 38.9% |
| 5 | 204 | 11.3% | 0.0871 | 0.0% | 0.5% | 32.5% |

## a6_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 308 | 15.3% | 0.0489 | 0.0% | 0.6% | 56.2% |
| energy | 332 | 7.8% | 0.101 | 0.0% | 0.9% | 34.3% |
| kinematics | 380 | 10.0% | 0.0883 | 0.3% | 0.3% | 35.2% |

Mean score by rule (failures at the rule's floor): crps -0.334, binary 0.109, log_score -1.84, abs_rel_error -0.357, crps_log -0.302
Mean step score: 0.722; copied-given rate: 0.1%; steps block present: 99.7%; mean completion tokens: 430

## a6b_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 337 | 9.7% | 34.4% | 72.7% | -0.00554 |
| 0.02 | 340 | 339 | 9.7% | 40.1% | 57.6% | -0.00793 |
| 0.05 | 340 | 339 | 10.9% | 43.4% | 75.7% | 0.00124 |

## a6b_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 25.5% | 0.0225 | 0.0% | 0.0% | 57.8% |
| 2 | 204 | 12.3% | 0.05 | 0.0% | 0.0% | 43.1% |
| 3 | 204 | 10.8% | 0.0683 | 0.0% | 0.5% | 36.5% |
| 4 | 204 | 10.8% | 0.0777 | 0.0% | 1.0% | 31.2% |
| 5 | 204 | 3.4% | 0.121 | 0.5% | 0.5% | 27.7% |

## a6b_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 438 | 13.2% | 0.0532 | 0.2% | 0.7% | 40.6% |
| energy | 438 | 10.7% | 0.0828 | 0.0% | 0.2% | 33.2% |
| kinematics | 144 | 16.0% | 0.0385 | 0.0% | 0.0% | 54.2% |

Mean score by rule (failures at the rule's floor): crps -0.314, binary 0.125, log_score -2.03, abs_rel_error -0.329, crps_log -0.498
Mean step score: 0.669; copied-given rate: 0.9%; steps block present: 99.9%; mean completion tokens: 301

## a6c_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.02 | 340 | 337 | 19.7% | 37.1% | 77.6% | -0.0082 |
| 0.05 | 680 | 675 | 11.3% | 45.5% | 58.4% | -0.00901 |

## a6c_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 18.1% | 0.0449 | 0.0% | 0.0% | 51.0% |
| 2 | 204 | 12.3% | 0.0647 | 2.0% | 1.5% | 37.6% |
| 3 | 204 | 18.1% | 0.0577 | 0.0% | 0.0% | 42.6% |
| 4 | 204 | 13.2% | 0.0712 | 0.0% | 0.5% | 43.8% |
| 5 | 204 | 11.3% | 0.0646 | 0.0% | 0.0% | 38.2% |

## a6c_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 320 | 15.9% | 0.0445 | 0.0% | 0.9% | 52.1% |
| energy | 338 | 10.9% | 0.0811 | 0.0% | 0.3% | 37.1% |
| kinematics | 362 | 16.9% | 0.0583 | 1.1% | 0.0% | 39.7% |

Mean score by rule (failures at the rule's floor): crps -0.226, binary 0.146, log_score -1.54, abs_rel_error -0.244, crps_log -0.213
Mean step score: 0.66; copied-given rate: 0.1%; steps block present: 99.6%; mean completion tokens: 423

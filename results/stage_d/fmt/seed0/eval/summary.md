# Eval summary: stage-d-fmt-seed0-a6-sets

- model: `.cache/huggingface/hub/models--unsloth--Qwen2.5-Math-1.5B-Instruct-bnb-4bit/snapshots/e24e9530845f4e3a6648343796883cb509935039` @ `e24e9530845f4e3a6648343796883cb509935039`
- generator: `2.0.0+bdbb2279`, reward rule: `crps_log`, prompt: `v5_two_channel`
- generation time: 3951 s
- torch_max_allocated_gib: 5.52
- torch_max_reserved_gib: 6.72
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 7.06

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a6_pairs | 1020 | 10.8% | 0.0764 | 2.1% | 0.0% | 0.1% | 17.2% | 19.4% | 0 | -0.444 | 1.4% |
| a6b_pairs | 1020 | 12.6% | 0.0549 | 1.5% | 0.0% | 0.0% | 11.5% | 13.0% | 0 | -0.469 | 1.5% |
| a6c_pairs | 1020 | 14.7% | 0.0473 | 2.7% | 0.0% | 0.6% | 19.8% | 21.4% | 0 | -0.361 | 2.3% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split (**not comparable here**: different generator or verifier version, so omitted).

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| a6_pairs | this model | 69.2% | 17.2% | 19.4% | -0.553 |
| a6_pairs | this model, σ × 18.1 |  | 25.1% | 26.7% | -0.872 |
| a6b_pairs | this model | 74.3% | 11.5% | 13.0% | -0.567 |
| a6b_pairs | this model, σ × 18.1 |  | 17.9% | 18.5% | -0.823 |
| a6c_pairs | this model | 63.6% | 19.8% | 21.4% | -0.465 |
| a6c_pairs | this model, σ × 18.1 |  | 28.1% | 29.7% | -0.793 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × (a) text scale | 26.7% | 27.8% | -1.04 |
| a6_pairs | this model × (a') text direct | 57.9% | 72.8% | -0.388 |
| a6_pairs | this model × (b) oracle per cell (upper bound) | 3.8% | 4.7% | -0.411 |
| a6b_pairs | this model × (a) text scale | 19.2% | 20.0% | -0.931 |
| a6b_pairs | this model × (a') text direct | 69.7% | 80.7% | -0.447 |
| a6b_pairs | this model × (b) oracle per cell (upper bound) | 5.0% | 6.9% | -0.457 |
| a6c_pairs | this model × (a) text scale | 30.9% | 33.0% | -0.955 |
| a6c_pairs | this model × (a') text direct | 69.7% | 82.0% | -0.326 |
| a6c_pairs | this model × (b) oracle per cell (upper bound) | 6.0% | 8.5% | -0.329 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × A5 σ | 34.3% | 45.6% | -0.39 |
| a6b_pairs | this model × A5 σ | 56.1% | 68.4% | -0.44 |
| a6c_pairs | this model × A5 σ | 47.7% | 61.7% | -0.313 |

## a6_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 333 | 20.9% | 23.4% | 25.4% | -299 |
| 0.02 | 340 | 328 | 19.1% | 12.8% | 10.8% | -300 |
| 0.05 | 340 | 335 | 19.4% | 15.5% | 16.7% | -300 |

## a6_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 15.2% | 0.0463 | 0.5% | 0.0% | 12.8% |
| 2 | 204 | 11.3% | 0.0901 | 1.0% | 0.0% | 21.3% |
| 3 | 204 | 8.3% | 0.0794 | 2.0% | 0.0% | 22.0% |
| 4 | 204 | 10.8% | 0.0736 | 2.5% | 0.0% | 18.6% |
| 5 | 204 | 8.3% | 0.0965 | 4.4% | 0.0% | 11.3% |

## a6_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 318 | 12.9% | 0.0539 | 0.9% | 0.0% | 19.0% |
| energy | 330 | 7.9% | 0.103 | 2.4% | 0.0% | 18.9% |
| kinematics | 372 | 11.6% | 0.0814 | 2.7% | 0.0% | 14.1% |

Mean score by rule (failures at the rule's floor): crps -0.618, binary 0.108, log_score -7.62, abs_rel_error -0.482, crps_log -0.553
Mean step score: 0.682; copied-given rate: 0.4%; steps block present: 97.9%; mean completion tokens: 295

## a6b_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 340 | 9.1% | 16.8% | 19.4% | -298 |
| 0.02 | 340 | 334 | 8.8% | 9.6% | 13.3% | -299 |
| 0.05 | 340 | 330 | 7.9% | 8.2% | 3.7% | -300 |

## a6b_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 18.1% | 0.0276 | 0.5% | 0.0% | 5.4% |
| 2 | 204 | 18.1% | 0.0469 | 0.0% | 0.0% | 7.8% |
| 3 | 204 | 11.8% | 0.0619 | 3.9% | 0.0% | 15.8% |
| 4 | 204 | 10.3% | 0.0769 | 1.0% | 0.0% | 14.9% |
| 5 | 204 | 4.9% | 0.136 | 2.0% | 0.0% | 14.0% |

## a6b_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 438 | 11.6% | 0.0481 | 2.3% | 0.0% | 13.3% |
| energy | 438 | 10.5% | 0.083 | 0.9% | 0.0% | 11.1% |
| kinematics | 144 | 22.2% | 0.0323 | 0.7% | 0.0% | 7.7% |

Mean score by rule (failures at the rule's floor): crps -0.508, binary 0.126, log_score -7.9, abs_rel_error -0.409, crps_log -0.567
Mean step score: 0.613; copied-given rate: 0.9%; steps block present: 98.5%; mean completion tokens: 277

## a6c_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.02 | 340 | 329 | 21.5% | 21.9% | 28.8% | -299 |
| 0.05 | 680 | 660 | 14.6% | 18.8% | 11.1% | -300 |

## a6c_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 16.2% | 0.0445 | 0.5% | 0.0% | 15.3% |
| 2 | 204 | 19.6% | 0.0368 | 1.0% | 0.0% | 15.8% |
| 3 | 204 | 14.7% | 0.0475 | 2.5% | 0.0% | 22.6% |
| 4 | 204 | 15.2% | 0.0465 | 5.9% | 0.0% | 19.8% |
| 5 | 204 | 7.8% | 0.0719 | 3.9% | 0.0% | 25.5% |

## a6c_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 318 | 18.9% | 0.0361 | 1.3% | 0.0% | 23.9% |
| energy | 338 | 10.7% | 0.0793 | 3.8% | 0.0% | 20.0% |
| kinematics | 364 | 14.8% | 0.0419 | 3.0% | 0.0% | 15.9% |

Mean score by rule (failures at the rule's floor): crps -0.49, binary 0.147, log_score -7.19, abs_rel_error -0.355, crps_log -0.465
Mean step score: 0.655; copied-given rate: 0.3%; steps block present: 97.3%; mean completion tokens: 299

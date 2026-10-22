# Eval summary: stage-d-pos-seed0-a6-sets

- model: `.cache/huggingface/hub/models--unsloth--Qwen2.5-Math-1.5B-Instruct-bnb-4bit/snapshots/e24e9530845f4e3a6648343796883cb509935039` @ `e24e9530845f4e3a6648343796883cb509935039`
- generator: `2.0.0+bdbb2279`, reward rule: `crps_log`, prompt: `v5_two_channel`
- generation time: 3241 s
- torch_max_allocated_gib: 5.52
- torch_max_reserved_gib: 6.72
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 7.07

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a6_pairs | 1020 | 10.4% | 0.0794 | 0.8% | 0.1% | 0.3% | 33.1% | 50.1% | 0.037 | -0.198 | 0.1% |
| a6b_pairs | 1020 | 12.9% | 0.0556 | 0.0% | 0.0% | 0.0% | 42.2% | 62.8% | 0.0352 | -0.293 | 0.0% |
| a6c_pairs | 1020 | 14.8% | 0.0484 | 0.3% | 0.1% | 0.0% | 33.8% | 52.7% | 0.0254 | -0.0845 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split (**not comparable here**: different generator or verifier version, so omitted).

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| a6_pairs | this model | 0.0% | 33.1% | 50.1% | -0.309 |
| a6_pairs | this model, σ × 18.1 |  | 89.4% | 94.4% | -0.405 |
| a6b_pairs | this model | 0.0% | 42.2% | 62.8% | -0.394 |
| a6b_pairs | this model, σ × 18.1 |  | 87.1% | 90.7% | -0.497 |
| a6c_pairs | this model | 0.4% | 33.8% | 52.7% | -0.189 |
| a6c_pairs | this model, σ × 18.1 |  | 90.6% | 96.0% | -0.253 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × (a) text scale | 95.6% | 97.2% | -0.78 |
| a6_pairs | this model × (a') text direct | 59.0% | 73.4% | -0.304 |
| a6_pairs | this model × (b) oracle per cell (upper bound) | 62.9% | 79.5% | -0.296 |
| a6b_pairs | this model × (a) text scale | 92.2% | 93.5% | -1.02 |
| a6b_pairs | this model × (a') text direct | 69.9% | 79.4% | -0.399 |
| a6b_pairs | this model × (b) oracle per cell (upper bound) | 62.9% | 75.6% | -0.386 |
| a6c_pairs | this model × (a) text scale | 96.8% | 98.2% | -0.522 |
| a6c_pairs | this model × (a') text direct | 71.9% | 84.4% | -0.202 |
| a6c_pairs | this model × (b) oracle per cell (upper bound) | 62.4% | 77.9% | -0.181 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × A5 σ | 33.9% | 45.5% | -0.305 |
| a6b_pairs | this model × A5 σ | 54.9% | 67.6% | -0.39 |
| a6c_pairs | this model × A5 σ | 46.9% | 60.8% | -0.186 |

## a6_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 338 | 20.0% | 29.0% | 58.8% | -0.0105 |
| 0.02 | 340 | 333 | 18.5% | 33.0% | 55.6% | -0.0628 |
| 0.05 | 340 | 340 | 20.3% | 37.4% | 58.0% | -0.0046 |

## a6_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 13.7% | 0.0461 | 1.0% | 0.0% | 40.6% |
| 2 | 204 | 12.7% | 0.0854 | 1.0% | 0.0% | 34.7% |
| 3 | 204 | 4.9% | 0.0873 | 0.5% | 0.0% | 33.5% |
| 4 | 204 | 11.3% | 0.0827 | 0.5% | 0.5% | 32.7% |
| 5 | 204 | 9.3% | 0.105 | 1.0% | 0.0% | 24.3% |

## a6_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 318 | 12.9% | 0.0592 | 0.6% | 0.0% | 41.5% |
| energy | 330 | 7.6% | 0.105 | 0.3% | 0.0% | 24.9% |
| kinematics | 372 | 10.8% | 0.0766 | 1.3% | 0.3% | 33.3% |

Mean score by rule (failures at the rule's floor): crps -0.379, binary 0.104, log_score -2.52, abs_rel_error -0.398, crps_log -0.309
Mean step score: 0.689; copied-given rate: 0.4%; steps block present: 99.2%; mean completion tokens: 303

## a6b_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 340 | 7.4% | 33.8% | 56.0% | -0.00563 |
| 0.02 | 340 | 340 | 9.1% | 41.8% | 74.2% | -0.00123 |
| 0.05 | 340 | 340 | 7.4% | 50.9% | 84.0% | -0.00454 |

## a6b_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 16.2% | 0.0306 | 0.0% | 0.0% | 57.8% |
| 2 | 204 | 21.6% | 0.039 | 0.0% | 0.0% | 52.5% |
| 3 | 204 | 10.8% | 0.0772 | 0.0% | 0.0% | 36.8% |
| 4 | 204 | 10.8% | 0.0889 | 0.0% | 0.0% | 35.3% |
| 5 | 204 | 5.4% | 0.0955 | 0.0% | 0.0% | 28.4% |

## a6b_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 438 | 12.8% | 0.0482 | 0.0% | 0.0% | 42.9% |
| energy | 438 | 10.0% | 0.0801 | 0.0% | 0.0% | 35.6% |
| kinematics | 144 | 22.2% | 0.0322 | 0.0% | 0.0% | 59.7% |

Mean score by rule (failures at the rule's floor): crps -0.329, binary 0.129, log_score -1.64, abs_rel_error -0.343, crps_log -0.394
Mean step score: 0.63; copied-given rate: 0.7%; steps block present: 100.0%; mean completion tokens: 278

## a6c_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.02 | 340 | 337 | 25.6% | 38.0% | 62.1% | -0.0162 |
| 0.05 | 680 | 675 | 14.6% | 31.9% | 48.5% | -0.331 |

## a6c_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 17.2% | 0.0368 | 1.0% | 0.0% | 45.5% |
| 2 | 204 | 21.1% | 0.0381 | 0.0% | 0.5% | 36.9% |
| 3 | 204 | 13.7% | 0.0624 | 0.5% | 0.0% | 28.6% |
| 4 | 204 | 14.2% | 0.0446 | 0.0% | 0.0% | 35.8% |
| 5 | 204 | 7.8% | 0.0646 | 0.0% | 0.0% | 22.1% |

## a6c_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 318 | 16.4% | 0.0369 | 0.0% | 0.3% | 41.6% |
| energy | 338 | 10.7% | 0.0819 | 0.0% | 0.0% | 25.7% |
| kinematics | 364 | 17.3% | 0.0357 | 0.8% | 0.0% | 34.3% |

Mean score by rule (failures at the rule's floor): crps -0.203, binary 0.148, log_score -2.13, abs_rel_error -0.215, crps_log -0.189
Mean step score: 0.653; copied-given rate: 0.4%; steps block present: 99.7%; mean completion tokens: 306

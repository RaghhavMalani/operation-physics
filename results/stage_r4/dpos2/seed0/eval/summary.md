# Eval summary: stage-r4-dpos2-seed0-a6-sets

- model: `.cache/huggingface/hub/models--unsloth--Qwen2.5-Math-1.5B-Instruct-bnb-4bit/snapshots/e24e9530845f4e3a6648343796883cb509935039` @ `e24e9530845f4e3a6648343796883cb509935039`
- generator: `2.0.0+bdbb2279`, reward rule: `crps_log`, prompt: `v5_two_channel`
- generation time: 6608 s
- torch_max_allocated_gib: 3.60
- torch_max_reserved_gib: 4.50
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 6.64

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a6_pairs | 1020 | 10.2% | 0.0836 | 0.0% | 0.1% | 0.0% | 42.2% | 64.2% | 0.0556 | -0.212 | 0.0% |
| a6b_pairs | 1020 | 11.7% | 0.0644 | 0.0% | 0.1% | 0.0% | 41.1% | 61.4% | 0.0349 | -0.352 | 0.0% |
| a6c_pairs | 1020 | 14.2% | 0.0541 | 0.3% | 0.0% | 0.2% | 45.5% | 65.2% | 0.0502 | -0.116 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split (**not comparable here**: different generator or verifier version, so omitted).

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| a6_pairs | this model | 0.0% | 42.2% | 64.2% | -0.323 |
| a6_pairs | this model, σ × 18.1 |  | 89.5% | 93.7% | -0.526 |
| a6b_pairs | this model | 0.0% | 41.1% | 61.4% | -0.451 |
| a6b_pairs | this model, σ × 18.1 |  | 85.2% | 87.9% | -0.593 |
| a6c_pairs | this model | 0.0% | 45.5% | 65.2% | -0.22 |
| a6c_pairs | this model, σ × 18.1 |  | 94.7% | 97.6% | -0.372 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × (a) text scale | 94.5% | 95.7% | -1.1 |
| a6_pairs | this model × (a') text direct | 58.0% | 71.3% | -0.321 |
| a6_pairs | this model × (b) oracle per cell (upper bound) | 62.1% | 77.0% | -0.315 |
| a6b_pairs | this model × (a) text scale | 89.4% | 91.4% | -1.14 |
| a6b_pairs | this model × (a') text direct | 64.6% | 76.3% | -0.447 |
| a6b_pairs | this model × (b) oracle per cell (upper bound) | 54.3% | 72.2% | -0.448 |
| a6c_pairs | this model × (a) text scale | 97.7% | 98.5% | -0.948 |
| a6c_pairs | this model × (a') text direct | 68.4% | 82.5% | -0.228 |
| a6c_pairs | this model × (b) oracle per cell (upper bound) | 66.9% | 82.2% | -0.211 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × A5 σ | 33.4% | 45.4% | -0.324 |
| a6b_pairs | this model × A5 σ | 54.0% | 67.2% | -0.443 |
| a6c_pairs | this model × A5 σ | 43.1% | 58.0% | -0.217 |

## a6_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 340 | 20.6% | 40.9% | 80.0% | -0.00486 |
| 0.02 | 340 | 339 | 21.8% | 43.1% | 66.2% | -0.00591 |
| 0.05 | 340 | 339 | 20.6% | 42.8% | 67.1% | -0.00347 |

## a6_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 12.3% | 0.0499 | 0.0% | 0.0% | 51.5% |
| 2 | 204 | 10.8% | 0.0867 | 0.0% | 0.0% | 41.2% |
| 3 | 204 | 5.4% | 0.0859 | 0.0% | 0.0% | 39.2% |
| 4 | 204 | 12.3% | 0.0859 | 0.0% | 0.0% | 37.7% |
| 5 | 204 | 10.3% | 0.115 | 0.0% | 0.5% | 41.4% |

## a6_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 318 | 13.8% | 0.0566 | 0.0% | 0.0% | 56.9% |
| energy | 330 | 7.0% | 0.106 | 0.0% | 0.0% | 29.4% |
| kinematics | 372 | 9.9% | 0.0865 | 0.0% | 0.3% | 41.0% |

Mean score by rule (failures at the rule's floor): crps -0.287, binary 0.102, log_score -1.76, abs_rel_error -0.313, crps_log -0.323
Mean step score: 0.697; copied-given rate: 0.3%; steps block present: 100.0%; mean completion tokens: 432

## a6b_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 340 | 7.1% | 32.9% | 58.3% | -0.00463 |
| 0.02 | 340 | 340 | 9.1% | 40.9% | 74.2% | 0.000894 |
| 0.05 | 340 | 339 | 6.8% | 49.6% | 78.3% | -0.00202 |

## a6b_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 19.1% | 0.0322 | 0.0% | 0.0% | 54.4% |
| 2 | 204 | 15.7% | 0.0519 | 0.0% | 0.0% | 46.1% |
| 3 | 204 | 10.3% | 0.0528 | 0.0% | 0.0% | 44.1% |
| 4 | 204 | 5.9% | 0.112 | 0.0% | 0.0% | 30.4% |
| 5 | 204 | 7.4% | 0.111 | 0.0% | 0.5% | 30.5% |

## a6b_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 438 | 12.1% | 0.0499 | 0.0% | 0.2% | 43.7% |
| energy | 438 | 7.3% | 0.0933 | 0.0% | 0.0% | 33.3% |
| kinematics | 144 | 23.6% | 0.0338 | 0.0% | 0.0% | 56.9% |

Mean score by rule (failures at the rule's floor): crps -0.248, binary 0.117, log_score -1.8, abs_rel_error -0.263, crps_log -0.451
Mean step score: 0.621; copied-given rate: 0.8%; steps block present: 100.0%; mean completion tokens: 307

## a6c_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.02 | 340 | 340 | 21.5% | 40.9% | 67.1% | -0.005 |
| 0.05 | 680 | 677 | 14.1% | 47.9% | 81.2% | -0.00115 |

## a6c_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 15.2% | 0.0478 | 0.0% | 0.0% | 52.5% |
| 2 | 204 | 20.1% | 0.0386 | 0.0% | 0.0% | 53.4% |
| 3 | 204 | 12.7% | 0.0624 | 0.5% | 0.0% | 45.3% |
| 4 | 204 | 15.7% | 0.0466 | 1.0% | 0.0% | 47.0% |
| 5 | 204 | 7.4% | 0.102 | 0.0% | 0.0% | 29.4% |

## a6c_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 318 | 19.5% | 0.0386 | 0.3% | 0.0% | 60.3% |
| energy | 338 | 7.4% | 0.0927 | 0.0% | 0.0% | 32.0% |
| kinematics | 364 | 15.9% | 0.0441 | 0.5% | 0.0% | 45.3% |

Mean score by rule (failures at the rule's floor): crps -0.23, binary 0.142, log_score -1.22, abs_rel_error -0.251, crps_log -0.22
Mean step score: 0.65; copied-given rate: 0.0%; steps block present: 99.7%; mean completion tokens: 427

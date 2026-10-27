# Eval summary: stage-r4-b2-seed1-confirmatory

- model: `.cache/huggingface/hub/models--unsloth--Qwen2.5-Math-1.5B-Instruct-bnb-4bit/snapshots/e24e9530845f4e3a6648343796883cb509935039` @ `e24e9530845f4e3a6648343796883cb509935039`
- generator: `2.0.0+bdbb2279`, reward rule: `crps_log`, prompt: `v5_two_channel`
- generation time: 3771 s
- torch_max_allocated_gib: 3.35
- torch_max_reserved_gib: 4.13
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 6.33

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a6_pairs | 1020 | 11.8% | 0.0745 | 0.0% | 0.0% | 0.0% | 38.9% | 58.0% | 0.048 | -0.129 | 0.0% |
| a6b_pairs | 1020 | 11.8% | 0.0568 | 0.0% | 0.0% | 0.0% | 42.1% | 61.1% | 0.0393 | -0.26 | 0.0% |
| a6c_pairs | 1020 | 12.0% | 0.0611 | 0.0% | 0.0% | 0.0% | 40.3% | 59.8% | 0.0481 | -0.115 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split (**not comparable here**: different generator or verifier version, so omitted).

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| a6_pairs | this model | 0.0% | 38.9% | 58.0% | -0.246 |
| a6_pairs | this model, σ × 18.1 |  | 91.2% | 95.3% | -0.386 |
| a6b_pairs | this model | 0.0% | 42.1% | 61.1% | -0.361 |
| a6b_pairs | this model, σ × 18.1 |  | 87.8% | 90.8% | -0.46 |
| a6c_pairs | this model | 0.0% | 40.3% | 59.8% | -0.215 |
| a6c_pairs | this model, σ × 18.1 |  | 94.0% | 96.7% | -0.319 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × (a) text scale | 96.4% | 97.7% | -0.876 |
| a6_pairs | this model × (a') text direct | 62.2% | 77.0% | -0.24 |
| a6_pairs | this model × (b) oracle per cell (upper bound) | 63.0% | 78.3% | -0.236 |
| a6b_pairs | this model × (a) text scale | 92.3% | 94.4% | -0.988 |
| a6b_pairs | this model × (a') text direct | 66.4% | 78.0% | -0.369 |
| a6b_pairs | this model × (b) oracle per cell (upper bound) | 64.7% | 77.2% | -0.352 |
| a6c_pairs | this model × (a) text scale | 97.5% | 98.3% | -0.735 |
| a6c_pairs | this model × (a') text direct | 67.2% | 80.7% | -0.216 |
| a6c_pairs | this model × (b) oracle per cell (upper bound) | 68.3% | 83.2% | -0.206 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × A5 σ | 34.7% | 48.4% | -0.241 |
| a6b_pairs | this model × A5 σ | 52.1% | 63.8% | -0.358 |
| a6c_pairs | this model × A5 σ | 50.1% | 62.6% | -0.214 |

## a6_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 340 | 20.9% | 33.5% | 69.0% | -0.026 |
| 0.02 | 340 | 340 | 20.9% | 37.9% | 62.0% | -0.0561 |
| 0.05 | 340 | 340 | 20.3% | 45.3% | 63.8% | -0.064 |

## a6_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 16.7% | 0.0503 | 0.0% | 0.0% | 46.6% |
| 2 | 204 | 13.2% | 0.059 | 0.0% | 0.0% | 43.6% |
| 3 | 204 | 8.3% | 0.0837 | 0.0% | 0.0% | 36.3% |
| 4 | 204 | 9.8% | 0.0818 | 0.0% | 0.0% | 39.7% |
| 5 | 204 | 10.8% | 0.0892 | 0.0% | 0.0% | 28.4% |

## a6_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 308 | 14.3% | 0.048 | 0.0% | 0.0% | 45.8% |
| energy | 332 | 8.1% | 0.112 | 0.0% | 0.0% | 30.7% |
| kinematics | 380 | 12.9% | 0.0774 | 0.0% | 0.0% | 40.5% |

Mean score by rule (failures at the rule's floor): crps -0.257, binary 0.118, log_score -1.88, abs_rel_error -0.28, crps_log -0.246
Mean step score: 0.729; copied-given rate: 0.4%; steps block present: 100.0%; mean completion tokens: 246

## a6b_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 340 | 7.9% | 39.4% | 63.0% | 0.017 |
| 0.02 | 340 | 340 | 10.6% | 43.8% | 66.7% | -0.0549 |
| 0.05 | 340 | 340 | 9.1% | 42.9% | 67.7% | -0.0352 |

## a6b_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 24.0% | 0.0228 | 0.0% | 0.0% | 54.9% |
| 2 | 204 | 12.7% | 0.0448 | 0.0% | 0.0% | 42.6% |
| 3 | 204 | 8.8% | 0.0598 | 0.0% | 0.0% | 40.2% |
| 4 | 204 | 7.8% | 0.0836 | 0.0% | 0.0% | 36.8% |
| 5 | 204 | 5.4% | 0.111 | 0.0% | 0.0% | 35.8% |

## a6b_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 438 | 14.4% | 0.0434 | 0.0% | 0.0% | 48.9% |
| energy | 438 | 6.8% | 0.0841 | 0.0% | 0.0% | 32.6% |
| kinematics | 144 | 18.8% | 0.0373 | 0.0% | 0.0% | 50.0% |

Mean score by rule (failures at the rule's floor): crps -0.313, binary 0.118, log_score -1.67, abs_rel_error -0.329, crps_log -0.361
Mean step score: 0.634; copied-given rate: 0.6%; steps block present: 100.0%; mean completion tokens: 230

## a6c_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.02 | 340 | 340 | 17.4% | 36.8% | 69.5% | -0.00866 |
| 0.05 | 680 | 680 | 9.7% | 42.1% | 60.6% | -0.0838 |

## a6c_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 19.1% | 0.0394 | 0.0% | 0.0% | 49.0% |
| 2 | 204 | 10.8% | 0.0587 | 0.0% | 0.0% | 38.7% |
| 3 | 204 | 12.3% | 0.0616 | 0.0% | 0.0% | 39.7% |
| 4 | 204 | 9.3% | 0.0787 | 0.0% | 0.0% | 37.7% |
| 5 | 204 | 8.3% | 0.0733 | 0.0% | 0.0% | 36.3% |

## a6c_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 320 | 15.9% | 0.0436 | 0.0% | 0.0% | 47.5% |
| energy | 338 | 6.8% | 0.0818 | 0.0% | 0.0% | 31.4% |
| kinematics | 362 | 13.3% | 0.0697 | 0.0% | 0.0% | 42.3% |

Mean score by rule (failures at the rule's floor): crps -0.255, binary 0.12, log_score -1.49, abs_rel_error -0.276, crps_log -0.215
Mean step score: 0.629; copied-given rate: 0.3%; steps block present: 100.0%; mean completion tokens: 249

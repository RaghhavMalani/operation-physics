# Eval summary: stage-xm-dpos2-seed0-confirmatory

- model: `.cache/huggingface/hub/models--unsloth--Llama-3.2-3B-Instruct-bnb-4bit/snapshots/bb1d317a108579fb40e646af8924a5e7ec5604b1` @ `bb1d317a108579fb40e646af8924a5e7ec5604b1`
- generator: `2.0.0+bdbb2279`, reward rule: `crps_log`, prompt: `v5_two_channel`
- generation time: 12283 s
- resumed (D142): session 1; 0 of 340 batches read from `completions.partial.jsonl`, 340 generated in this session
- torch_max_allocated_gib: 4.85
- torch_max_reserved_gib: 5.86
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 6.76

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a6b_pairs | 1020 | 8.3% | 0.215 | 0.0% | 0.0% | 0.0% | 24.0% | 34.7% | 0.0263 | -0.702 | 0.0% |
| a6c_pairs | 1020 | 10.2% | 0.124 | 0.4% | 0.3% | 0.0% | 28.5% | 44.8% | 0.0366 | -0.305 | 0.4% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split (**not comparable here**: different generator or verifier version, so omitted).

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| a6b_pairs | this model | 0.0% | 24.0% | 34.7% | -0.781 |
| a6b_pairs | this model, σ × 18.1 |  | 65.5% | 73.8% | -0.747 |
| a6c_pairs | this model | 0.0% | 28.5% | 44.8% | -0.389 |
| a6c_pairs | this model, σ × 18.1 |  | 85.2% | 91.8% | -0.449 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6b_pairs | this model × (a) text scale | 79.1% | 83.2% | -1.07 |
| a6b_pairs | this model × (a') text direct | 38.4% | 54.4% | -0.725 |
| a6b_pairs | this model × (b) oracle per cell (upper bound) | 51.5% | 62.3% | -0.71 |
| a6c_pairs | this model × (a) text scale | 93.9% | 96.2% | -0.84 |
| a6c_pairs | this model × (a') text direct | 50.5% | 68.1% | -0.358 |
| a6c_pairs | this model × (b) oracle per cell (upper bound) | 57.5% | 72.3% | -0.357 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6b_pairs | this model × A5 σ | 25.8% | 40.8% | -0.735 |
| a6c_pairs | this model × A5 σ | 29.3% | 45.5% | -0.361 |

## a6b_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 340 | 4.4% | 19.7% | 66.7% | -0.00264 |
| 0.02 | 340 | 340 | 4.4% | 23.2% | 53.3% | -0.00217 |
| 0.05 | 340 | 340 | 5.9% | 29.1% | 90.0% | 0.000808 |

## a6b_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 20.1% | 0.0405 | 0.0% | 0.0% | 44.6% |
| 2 | 204 | 7.8% | 0.17 | 0.0% | 0.0% | 23.5% |
| 3 | 204 | 7.4% | 0.242 | 0.0% | 0.0% | 22.5% |
| 4 | 204 | 3.4% | 0.417 | 0.0% | 0.0% | 17.2% |
| 5 | 204 | 2.9% | 0.48 | 0.0% | 0.0% | 12.3% |

## a6b_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 438 | 12.3% | 0.102 | 0.0% | 0.0% | 33.8% |
| energy | 438 | 3.0% | 0.471 | 0.0% | 0.0% | 10.3% |
| kinematics | 144 | 12.5% | 0.0724 | 0.0% | 0.0% | 36.1% |

Mean score by rule (failures at the rule's floor): crps -0.548, binary 0.0833, log_score -5.24, abs_rel_error -0.557, crps_log -0.781
Mean step score: 0.495; copied-given rate: 0.9%; steps block present: 100.0%; mean completion tokens: 255

## a6c_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.02 | 340 | 339 | 14.7% | 34.2% | 72.0% | -0.00445 |
| 0.05 | 680 | 674 | 9.4% | 25.7% | 59.4% | -0.00898 |

## a6c_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 16.7% | 0.0749 | 0.0% | 1.0% | 36.1% |
| 2 | 204 | 6.9% | 0.119 | 2.0% | 0.5% | 24.6% |
| 3 | 204 | 9.8% | 0.155 | 0.0% | 0.0% | 24.0% |
| 4 | 204 | 11.3% | 0.185 | 0.0% | 0.0% | 31.4% |
| 5 | 204 | 6.4% | 0.123 | 0.0% | 0.0% | 26.5% |

## a6c_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 320 | 13.1% | 0.0645 | 0.0% | 0.3% | 41.4% |
| energy | 338 | 4.4% | 0.342 | 0.0% | 0.0% | 13.0% |
| kinematics | 362 | 13.0% | 0.0823 | 1.1% | 0.6% | 31.7% |

Mean score by rule (failures at the rule's floor): crps -0.378, binary 0.102, log_score -3.69, abs_rel_error -0.391, crps_log -0.389
Mean step score: 0.53; copied-given rate: 0.1%; steps block present: 99.6%; mean completion tokens: 356

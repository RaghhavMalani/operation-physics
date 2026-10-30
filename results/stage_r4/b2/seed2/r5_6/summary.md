# Eval summary: stage-r4-b2-seed2-r5-6

- model: `.cache/huggingface/hub/models--unsloth--Qwen2.5-Math-1.5B-Instruct-bnb-4bit/snapshots/e24e9530845f4e3a6648343796883cb509935039` @ `e24e9530845f4e3a6648343796883cb509935039`
- generator: `2.0.0+bdbb2279`, reward rule: `crps_log`, prompt: `v5_two_channel`
- generation time: 1892 s
- resumed (D142): session 1; 0 of 102 batches read from `completions.partial.jsonl`, 102 generated in this session
- torch_max_allocated_gib: 3.34
- torch_max_reserved_gib: 4.13
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.27

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a6b_pairs | 816 | 15.8% | 0.0528 | 0.0% | 0.0% | 0.0% | 45.2% | 64.8% | 0.0475 | -0.044 | 0.0% |
| a6c_pairs | 816 | 12.3% | 0.057 | 0.0% | 0.0% | 0.0% | 42.0% | 60.0% | 0.0459 | -0.0696 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split (**not comparable here**: different generator or verifier version, so omitted).

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| a6b_pairs | this model | 0.0% | 45.2% | 64.8% | -0.152 |
| a6b_pairs | this model, σ × 18.1 |  | 94.9% | 98.2% | -0.318 |
| a6c_pairs | this model | 0.0% | 42.0% | 60.0% | -0.169 |
| a6c_pairs | this model, σ × 18.1 |  | 93.0% | 96.4% | -0.268 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6b_pairs | this model × (a) text scale | 98.3% | 99.4% | -0.797 |
| a6b_pairs | this model × (a') text direct | 62.3% | 77.6% | -0.151 |
| a6b_pairs | this model × (b) oracle per cell (upper bound) | 62.0% | 78.4% | -0.148 |
| a6c_pairs | this model × (a) text scale | 96.8% | 98.4% | -0.669 |
| a6c_pairs | this model × (a') text direct | 65.0% | 79.5% | -0.164 |
| a6c_pairs | this model × (b) oracle per cell (upper bound) | 66.2% | 80.0% | -0.161 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6b_pairs | this model × A5 σ | 36.5% | 48.7% | -0.152 |
| a6c_pairs | this model × A5 σ | 39.0% | 52.0% | -0.164 |

## a6b_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 272 | 272 | 28.3% | 41.5% | 62.3% | -0.0596 |
| 0.02 | 272 | 272 | 28.3% | 44.1% | 66.2% | -0.0582 |
| 0.05 | 272 | 272 | 24.6% | 50.0% | 73.1% | 0.0375 |

## a6b_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 19.1% | 0.0483 | 0.0% | 0.0% | 50.0% |
| 2 | 204 | 18.6% | 0.0501 | 0.0% | 0.0% | 48.0% |
| 3 | 204 | 13.7% | 0.0471 | 0.0% | 0.0% | 39.7% |
| 4 | 204 | 11.8% | 0.0841 | 0.0% | 0.0% | 43.1% |

## a6b_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 238 | 16.4% | 0.0444 | 0.0% | 0.0% | 49.2% |
| energy | 278 | 14.4% | 0.0724 | 0.0% | 0.0% | 41.7% |
| kinematics | 300 | 16.7% | 0.0553 | 0.0% | 0.0% | 45.3% |

Mean score by rule (failures at the rule's floor): crps -0.171, binary 0.158, log_score -1.11, abs_rel_error -0.191, crps_log -0.152
Mean step score: 0.723; copied-given rate: 0.2%; steps block present: 100.0%; mean completion tokens: 218

## a6c_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.02 | 272 | 272 | 16.9% | 34.9% | 67.4% | -0.0405 |
| 0.05 | 544 | 544 | 23.2% | 45.6% | 68.3% | -0.0262 |

## a6c_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 15.2% | 0.0627 | 0.0% | 0.0% | 41.7% |
| 2 | 204 | 13.7% | 0.0504 | 0.0% | 0.0% | 40.7% |
| 3 | 204 | 12.3% | 0.0481 | 0.0% | 0.0% | 47.5% |
| 4 | 204 | 7.8% | 0.0667 | 0.0% | 0.0% | 38.2% |

## a6c_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 182 | 15.9% | 0.0386 | 0.0% | 0.0% | 53.3% |
| energy | 290 | 10.3% | 0.0788 | 0.0% | 0.0% | 38.6% |
| kinematics | 344 | 11.9% | 0.0613 | 0.0% | 0.0% | 39.0% |

Mean score by rule (failures at the rule's floor): crps -0.167, binary 0.123, log_score -1.59, abs_rel_error -0.184, crps_log -0.169
Mean step score: 0.66; copied-given rate: 0.0%; steps block present: 100.0%; mean completion tokens: 223

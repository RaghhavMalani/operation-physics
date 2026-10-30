# Eval summary: stage-r4-b2-seed1-r5-6

- model: `.cache/huggingface/hub/models--unsloth--Qwen2.5-Math-1.5B-Instruct-bnb-4bit/snapshots/e24e9530845f4e3a6648343796883cb509935039` @ `e24e9530845f4e3a6648343796883cb509935039`
- generator: `2.0.0+bdbb2279`, reward rule: `crps_log`, prompt: `v5_two_channel`
- generation time: 1948 s
- resumed (D142): session 1; 0 of 102 batches read from `completions.partial.jsonl`, 102 generated in this session
- torch_max_allocated_gib: 3.34
- torch_max_reserved_gib: 4.13
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.24

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a6b_pairs | 816 | 15.3% | 0.0538 | 0.0% | 0.0% | 0.0% | 42.3% | 59.8% | 0.0428 | -0.0493 | 0.0% |
| a6c_pairs | 816 | 13.0% | 0.0592 | 0.0% | 0.0% | 0.0% | 40.0% | 57.7% | 0.0436 | -0.107 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split (**not comparable here**: different generator or verifier version, so omitted).

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| a6b_pairs | this model | 0.0% | 42.3% | 59.8% | -0.157 |
| a6b_pairs | this model, σ × 18.1 |  | 95.2% | 97.8% | -0.287 |
| a6c_pairs | this model | 0.0% | 40.0% | 57.7% | -0.205 |
| a6c_pairs | this model, σ × 18.1 |  | 91.8% | 96.3% | -0.291 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6b_pairs | this model × (a) text scale | 98.2% | 99.1% | -0.698 |
| a6b_pairs | this model × (a') text direct | 62.1% | 78.2% | -0.155 |
| a6b_pairs | this model × (b) oracle per cell (upper bound) | 63.7% | 79.5% | -0.148 |
| a6c_pairs | this model × (a) text scale | 97.2% | 98.3% | -0.657 |
| a6c_pairs | this model × (a') text direct | 63.6% | 78.6% | -0.196 |
| a6c_pairs | this model × (b) oracle per cell (upper bound) | 64.8% | 81.1% | -0.19 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6b_pairs | this model × A5 σ | 36.0% | 49.4% | -0.155 |
| a6c_pairs | this model × A5 σ | 36.8% | 50.5% | -0.199 |

## a6b_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 272 | 272 | 29.0% | 36.4% | 55.7% | -0.131 |
| 0.02 | 272 | 272 | 29.0% | 42.6% | 58.2% | -0.103 |
| 0.05 | 272 | 272 | 23.9% | 47.8% | 63.1% | -0.065 |

## a6b_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 19.6% | 0.0482 | 0.0% | 0.0% | 46.6% |
| 2 | 204 | 19.1% | 0.0405 | 0.0% | 0.0% | 49.5% |
| 3 | 204 | 11.3% | 0.0568 | 0.0% | 0.0% | 35.3% |
| 4 | 204 | 11.3% | 0.087 | 0.0% | 0.0% | 37.7% |

## a6b_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 238 | 15.1% | 0.0425 | 0.0% | 0.0% | 46.2% |
| energy | 278 | 13.7% | 0.0654 | 0.0% | 0.0% | 39.2% |
| kinematics | 300 | 17.0% | 0.064 | 0.0% | 0.0% | 42.0% |

Mean score by rule (failures at the rule's floor): crps -0.189, binary 0.153, log_score -1.52, abs_rel_error -0.211, crps_log -0.157
Mean step score: 0.72; copied-given rate: 0.0%; steps block present: 100.0%; mean completion tokens: 219

## a6c_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.02 | 272 | 272 | 18.4% | 31.6% | 64.0% | -0.0249 |
| 0.05 | 544 | 544 | 23.7% | 44.1% | 66.7% | -0.0452 |

## a6c_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 17.6% | 0.0451 | 0.0% | 0.0% | 42.2% |
| 2 | 204 | 18.1% | 0.0404 | 0.0% | 0.0% | 50.0% |
| 3 | 204 | 9.3% | 0.0671 | 0.0% | 0.0% | 37.3% |
| 4 | 204 | 6.9% | 0.0865 | 0.0% | 0.0% | 30.4% |

## a6c_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 182 | 16.5% | 0.0377 | 0.0% | 0.0% | 52.2% |
| energy | 290 | 12.8% | 0.0932 | 0.0% | 0.0% | 34.5% |
| kinematics | 344 | 11.3% | 0.065 | 0.0% | 0.0% | 38.1% |

Mean score by rule (failures at the rule's floor): crps -0.231, binary 0.13, log_score -2.02, abs_rel_error -0.248, crps_log -0.205
Mean step score: 0.65; copied-given rate: 0.5%; steps block present: 100.0%; mean completion tokens: 224

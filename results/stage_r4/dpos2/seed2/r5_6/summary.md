# Eval summary: stage-r4-dpos2-seed2-r5-6

- model: `.cache/huggingface/hub/models--unsloth--Qwen2.5-Math-1.5B-Instruct-bnb-4bit/snapshots/e24e9530845f4e3a6648343796883cb509935039` @ `e24e9530845f4e3a6648343796883cb509935039`
- generator: `2.0.0+bdbb2279`, reward rule: `crps_log`, prompt: `v5_two_channel`
- generation time: 3453 s
- resumed (D142): session 1; 0 of 102 batches read from `completions.partial.jsonl`, 102 generated in this session
- torch_max_allocated_gib: 3.62
- torch_max_reserved_gib: 4.49
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.60

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a6b_pairs | 816 | 15.4% | 0.0565 | 0.4% | 0.0% | 0.4% | 43.1% | 67.5% | 0.0424 | -0.0615 | 0.0% |
| a6c_pairs | 816 | 13.0% | 0.0567 | 0.5% | 0.4% | 0.4% | 42.3% | 64.3% | 0.0403 | -0.093 | 0.1% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split (**not comparable here**: different generator or verifier version, so omitted).

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| a6b_pairs | this model | 0.0% | 43.1% | 67.5% | -0.171 |
| a6b_pairs | this model, σ × 18.1 |  | 93.6% | 97.3% | -0.343 |
| a6c_pairs | this model | 0.0% | 42.3% | 64.3% | -0.192 |
| a6c_pairs | this model, σ × 18.1 |  | 95.1% | 97.4% | -0.331 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6b_pairs | this model × (a) text scale | 97.8% | 98.6% | -0.793 |
| a6b_pairs | this model × (a') text direct | 61.1% | 78.0% | -0.167 |
| a6b_pairs | this model × (b) oracle per cell (upper bound) | 60.3% | 78.7% | -0.166 |
| a6c_pairs | this model × (a) text scale | 97.9% | 99.3% | -0.795 |
| a6c_pairs | this model × (a') text direct | 65.8% | 80.6% | -0.189 |
| a6c_pairs | this model × (b) oracle per cell (upper bound) | 61.8% | 78.9% | -0.185 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6b_pairs | this model × A5 σ | 35.7% | 47.2% | -0.172 |
| a6c_pairs | this model × A5 σ | 37.7% | 50.2% | -0.193 |

## a6b_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 272 | 271 | 28.3% | 33.9% | 61.0% | -0.0265 |
| 0.02 | 272 | 272 | 24.6% | 45.6% | 67.2% | -0.0158 |
| 0.05 | 272 | 270 | 25.0% | 49.6% | 73.5% | -0.0174 |

## a6b_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 19.6% | 0.0499 | 0.0% | 0.0% | 54.9% |
| 2 | 204 | 18.6% | 0.0487 | 0.0% | 0.0% | 39.7% |
| 3 | 204 | 9.8% | 0.0545 | 1.5% | 0.0% | 35.8% |
| 4 | 204 | 13.7% | 0.0784 | 0.0% | 0.0% | 41.7% |

## a6b_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 238 | 16.8% | 0.0393 | 0.0% | 0.0% | 57.6% |
| energy | 278 | 13.7% | 0.0816 | 0.0% | 0.0% | 36.7% |
| kinematics | 300 | 16.0% | 0.0702 | 1.0% | 0.0% | 37.4% |

Mean score by rule (failures at the rule's floor): crps -0.164, binary 0.154, log_score -1.14, abs_rel_error -0.182, crps_log -0.171
Mean step score: 0.729; copied-given rate: 0.5%; steps block present: 99.6%; mean completion tokens: 389

## a6c_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.02 | 272 | 268 | 15.8% | 31.7% | 55.8% | -0.018 |
| 0.05 | 544 | 541 | 23.7% | 47.5% | 70.5% | -0.013 |

## a6c_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 15.7% | 0.0594 | 1.0% | 0.5% | 43.8% |
| 2 | 204 | 15.2% | 0.0459 | 1.0% | 0.0% | 41.6% |
| 3 | 204 | 9.8% | 0.0555 | 0.0% | 0.5% | 41.4% |
| 4 | 204 | 11.3% | 0.0651 | 0.0% | 0.5% | 42.4% |

## a6c_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 182 | 18.7% | 0.0395 | 0.0% | 0.0% | 58.2% |
| energy | 290 | 11.4% | 0.0937 | 0.0% | 0.7% | 36.5% |
| kinematics | 344 | 11.3% | 0.0585 | 1.2% | 0.3% | 38.6% |

Mean score by rule (failures at the rule's floor): crps -0.221, binary 0.13, log_score -1.59, abs_rel_error -0.24, crps_log -0.192
Mean step score: 0.658; copied-given rate: 0.0%; steps block present: 99.5%; mean completion tokens: 405

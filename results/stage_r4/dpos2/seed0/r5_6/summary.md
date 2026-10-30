# Eval summary: stage-r4-dpos2-seed0-r5-6

- model: `.cache/huggingface/hub/models--unsloth--Qwen2.5-Math-1.5B-Instruct-bnb-4bit/snapshots/e24e9530845f4e3a6648343796883cb509935039` @ `e24e9530845f4e3a6648343796883cb509935039`
- generator: `2.0.0+bdbb2279`, reward rule: `crps_log`, prompt: `v5_two_channel`
- generation time: 3534 s
- resumed (D142): session 1; 0 of 102 batches read from `completions.partial.jsonl`, 102 generated in this session
- torch_max_allocated_gib: 3.71
- torch_max_reserved_gib: 4.49
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 6.58

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a6b_pairs | 816 | 15.2% | 0.052 | 0.2% | 0.0% | 0.2% | 47.7% | 71.3% | 0.0497 | -0.0768 | 0.0% |
| a6c_pairs | 816 | 11.6% | 0.059 | 0.0% | 0.0% | 0.0% | 45.1% | 65.4% | 0.0418 | -0.128 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split (**not comparable here**: different generator or verifier version, so omitted).

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| a6b_pairs | this model | 0.0% | 47.7% | 71.3% | -0.187 |
| a6b_pairs | this model, σ × 18.1 |  | 94.2% | 96.9% | -0.387 |
| a6c_pairs | this model | 0.0% | 45.1% | 65.4% | -0.226 |
| a6c_pairs | this model, σ × 18.1 |  | 92.0% | 95.5% | -0.371 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6b_pairs | this model × (a) text scale | 97.5% | 98.5% | -0.915 |
| a6b_pairs | this model × (a') text direct | 60.6% | 76.5% | -0.183 |
| a6b_pairs | this model × (b) oracle per cell (upper bound) | 66.2% | 80.7% | -0.181 |
| a6c_pairs | this model × (a) text scale | 96.7% | 98.3% | -0.889 |
| a6c_pairs | this model × (a') text direct | 65.2% | 78.8% | -0.22 |
| a6c_pairs | this model × (b) oracle per cell (upper bound) | 62.4% | 79.0% | -0.217 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6b_pairs | this model × A5 σ | 34.7% | 46.0% | -0.189 |
| a6c_pairs | this model × A5 σ | 37.3% | 48.6% | -0.223 |

## a6b_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 272 | 271 | 26.5% | 41.0% | 75.0% | -0.00678 |
| 0.02 | 272 | 272 | 30.5% | 49.3% | 67.5% | 0.00105 |
| 0.05 | 272 | 270 | 28.3% | 53.0% | 76.6% | -0.00197 |

## a6b_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 19.6% | 0.0465 | 0.0% | 0.0% | 61.3% |
| 2 | 204 | 17.6% | 0.0507 | 0.0% | 0.0% | 40.2% |
| 3 | 204 | 11.3% | 0.0485 | 1.0% | 0.0% | 44.6% |
| 4 | 204 | 12.3% | 0.0768 | 0.0% | 0.0% | 44.6% |

## a6b_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 238 | 18.1% | 0.0387 | 0.0% | 0.0% | 63.0% |
| energy | 278 | 12.9% | 0.0648 | 0.0% | 0.0% | 42.1% |
| kinematics | 300 | 15.0% | 0.0759 | 0.7% | 0.0% | 40.6% |

Mean score by rule (failures at the rule's floor): crps -0.194, binary 0.152, log_score -0.847, abs_rel_error -0.218, crps_log -0.187
Mean step score: 0.736; copied-given rate: 0.2%; steps block present: 99.8%; mean completion tokens: 393

## a6c_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.02 | 272 | 270 | 16.2% | 34.1% | 59.1% | -0.00914 |
| 0.05 | 544 | 542 | 25.4% | 50.9% | 74.6% | -0.00453 |

## a6c_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 14.7% | 0.0495 | 0.0% | 0.0% | 43.6% |
| 2 | 204 | 13.7% | 0.0444 | 0.0% | 0.0% | 45.1% |
| 3 | 204 | 7.8% | 0.0641 | 0.0% | 0.0% | 46.1% |
| 4 | 204 | 10.3% | 0.0711 | 0.0% | 0.0% | 45.6% |

## a6c_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 182 | 17.0% | 0.0356 | 0.0% | 0.0% | 65.4% |
| energy | 290 | 9.0% | 0.0889 | 0.0% | 0.0% | 41.4% |
| kinematics | 344 | 11.0% | 0.0635 | 0.0% | 0.0% | 37.5% |

Mean score by rule (failures at the rule's floor): crps -0.212, binary 0.116, log_score -1.44, abs_rel_error -0.231, crps_log -0.226
Mean step score: 0.652; copied-given rate: 0.4%; steps block present: 100.0%; mean completion tokens: 404

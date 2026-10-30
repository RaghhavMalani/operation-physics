# Eval summary: stage-r4-dpos2-seed1-r5-6

- model: `.cache/huggingface/hub/models--unsloth--Qwen2.5-Math-1.5B-Instruct-bnb-4bit/snapshots/e24e9530845f4e3a6648343796883cb509935039` @ `e24e9530845f4e3a6648343796883cb509935039`
- generator: `2.0.0+bdbb2279`, reward rule: `crps_log`, prompt: `v5_two_channel`
- generation time: 3465 s
- resumed (D142): session 1; 0 of 102 batches read from `completions.partial.jsonl`, 102 generated in this session
- torch_max_allocated_gib: 3.55
- torch_max_reserved_gib: 4.49
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 5.61

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a6b_pairs | 816 | 15.3% | 0.0541 | 0.1% | 0.0% | 0.1% | 44.5% | 67.1% | 0.0457 | -0.0823 | 0.0% |
| a6c_pairs | 816 | 13.4% | 0.0594 | 0.0% | 0.1% | 0.0% | 42.2% | 61.6% | 0.04 | -0.111 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split (**not comparable here**: different generator or verifier version, so omitted).

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| a6b_pairs | this model | 0.0% | 44.5% | 67.1% | -0.191 |
| a6b_pairs | this model, σ × 18.1 |  | 93.7% | 96.2% | -0.35 |
| a6c_pairs | this model | 0.0% | 42.2% | 61.6% | -0.209 |
| a6c_pairs | this model, σ × 18.1 |  | 90.9% | 94.8% | -0.341 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6b_pairs | this model × (a) text scale | 97.3% | 98.3% | -0.817 |
| a6b_pairs | this model × (a') text direct | 59.6% | 76.3% | -0.184 |
| a6b_pairs | this model × (b) oracle per cell (upper bound) | 65.6% | 80.1% | -0.182 |
| a6c_pairs | this model × (a) text scale | 96.4% | 98.5% | -0.791 |
| a6c_pairs | this model × (a') text direct | 63.4% | 78.5% | -0.199 |
| a6c_pairs | this model × (b) oracle per cell (upper bound) | 59.0% | 73.7% | -0.203 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6b_pairs | this model × A5 σ | 33.5% | 44.0% | -0.192 |
| a6c_pairs | this model × A5 σ | 35.6% | 46.0% | -0.206 |

## a6b_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 272 | 272 | 29.0% | 39.0% | 64.6% | -0.00489 |
| 0.02 | 272 | 272 | 29.8% | 46.7% | 61.7% | 0.00203 |
| 0.05 | 272 | 271 | 21.0% | 48.0% | 80.7% | -0.00563 |

## a6b_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 20.1% | 0.0485 | 0.0% | 0.0% | 61.3% |
| 2 | 204 | 20.6% | 0.0472 | 0.0% | 0.0% | 42.2% |
| 3 | 204 | 10.3% | 0.0485 | 0.5% | 0.0% | 35.0% |
| 4 | 204 | 10.3% | 0.0861 | 0.0% | 0.0% | 39.7% |

## a6b_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 238 | 16.4% | 0.04 | 0.0% | 0.0% | 59.7% |
| energy | 278 | 13.3% | 0.0822 | 0.0% | 0.0% | 38.1% |
| kinematics | 300 | 16.3% | 0.063 | 0.3% | 0.0% | 38.5% |

Mean score by rule (failures at the rule's floor): crps -0.203, binary 0.153, log_score -1.24, abs_rel_error -0.223, crps_log -0.191
Mean step score: 0.725; copied-given rate: 0.5%; steps block present: 99.9%; mean completion tokens: 392

## a6c_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.02 | 272 | 272 | 15.8% | 32.0% | 60.5% | -0.0124 |
| 0.05 | 544 | 543 | 22.8% | 47.3% | 78.2% | -0.0138 |

## a6c_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 16.2% | 0.0545 | 0.0% | 0.0% | 43.1% |
| 2 | 204 | 15.2% | 0.046 | 0.0% | 0.5% | 40.9% |
| 3 | 204 | 10.3% | 0.0601 | 0.0% | 0.0% | 43.6% |
| 4 | 204 | 11.8% | 0.0727 | 0.0% | 0.0% | 41.2% |

## a6c_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 182 | 19.2% | 0.0396 | 0.0% | 0.5% | 55.2% |
| energy | 290 | 10.7% | 0.0955 | 0.0% | 0.0% | 39.7% |
| kinematics | 344 | 12.5% | 0.0578 | 0.0% | 0.0% | 37.5% |

Mean score by rule (failures at the rule's floor): crps -0.242, binary 0.134, log_score -1.86, abs_rel_error -0.259, crps_log -0.209
Mean step score: 0.652; copied-given rate: 0.6%; steps block present: 100.0%; mean completion tokens: 404

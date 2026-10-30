# Eval summary: stage-r4-b2-seed0-r5-6

- model: `.cache/huggingface/hub/models--unsloth--Qwen2.5-Math-1.5B-Instruct-bnb-4bit/snapshots/e24e9530845f4e3a6648343796883cb509935039` @ `e24e9530845f4e3a6648343796883cb509935039`
- generator: `2.0.0+bdbb2279`, reward rule: `crps_log`, prompt: `v5_two_channel`
- generation time: 1909 s
- resumed (D142): session 1; 0 of 102 batches read from `completions.partial.jsonl`, 102 generated in this session
- torch_max_allocated_gib: 3.34
- torch_max_reserved_gib: 4.13
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 6.02

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a6b_pairs | 816 | 14.6% | 0.0516 | 0.0% | 0.0% | 0.0% | 40.3% | 60.3% | 0.0391 | -0.0305 | 0.0% |
| a6c_pairs | 816 | 13.1% | 0.0556 | 0.1% | 0.0% | 0.1% | 39.5% | 57.1% | 0.0358 | -0.0788 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split (**not comparable here**: different generator or verifier version, so omitted).

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| a6b_pairs | this model | 0.0% | 40.3% | 60.3% | -0.141 |
| a6b_pairs | this model, σ × 18.1 |  | 94.2% | 98.3% | -0.293 |
| a6c_pairs | this model | 0.0% | 39.5% | 57.1% | -0.178 |
| a6c_pairs | this model, σ × 18.1 |  | 91.3% | 95.1% | -0.262 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6b_pairs | this model × (a) text scale | 98.5% | 99.3% | -0.716 |
| a6b_pairs | this model × (a') text direct | 61.5% | 79.8% | -0.133 |
| a6b_pairs | this model × (b) oracle per cell (upper bound) | 57.6% | 76.3% | -0.137 |
| a6c_pairs | this model × (a) text scale | 96.6% | 98.0% | -0.594 |
| a6c_pairs | this model × (a') text direct | 66.7% | 80.7% | -0.17 |
| a6c_pairs | this model × (b) oracle per cell (upper bound) | 61.3% | 76.2% | -0.171 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6b_pairs | this model × A5 σ | 37.7% | 49.8% | -0.136 |
| a6c_pairs | this model × A5 σ | 40.4% | 51.5% | -0.171 |

## a6b_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 272 | 272 | 30.9% | 37.5% | 52.4% | -0.144 |
| 0.02 | 272 | 272 | 29.0% | 39.3% | 53.2% | -0.176 |
| 0.05 | 272 | 272 | 24.6% | 44.1% | 58.2% | -0.0891 |

## a6b_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 18.6% | 0.0447 | 0.0% | 0.0% | 44.1% |
| 2 | 204 | 17.6% | 0.0488 | 0.0% | 0.0% | 40.7% |
| 3 | 204 | 10.3% | 0.0533 | 0.0% | 0.0% | 36.8% |
| 4 | 204 | 11.8% | 0.0806 | 0.0% | 0.0% | 39.7% |

## a6b_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 238 | 16.4% | 0.0401 | 0.0% | 0.0% | 45.0% |
| energy | 278 | 12.6% | 0.0787 | 0.0% | 0.0% | 35.3% |
| kinematics | 300 | 15.0% | 0.0563 | 0.0% | 0.0% | 41.3% |

Mean score by rule (failures at the rule's floor): crps -0.147, binary 0.146, log_score -1.4, abs_rel_error -0.166, crps_log -0.141
Mean step score: 0.737; copied-given rate: 0.2%; steps block present: 100.0%; mean completion tokens: 219

## a6c_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.02 | 272 | 272 | 15.8% | 33.5% | 55.8% | -0.0685 |
| 0.05 | 544 | 543 | 23.7% | 42.5% | 65.1% | -0.121 |

## a6c_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 16.2% | 0.0526 | 0.0% | 0.0% | 40.7% |
| 2 | 204 | 13.2% | 0.0457 | 0.0% | 0.0% | 41.2% |
| 3 | 204 | 11.3% | 0.0586 | 0.5% | 0.0% | 38.9% |
| 4 | 204 | 11.8% | 0.069 | 0.0% | 0.0% | 37.3% |

## a6c_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 182 | 18.7% | 0.0364 | 0.5% | 0.0% | 50.3% |
| energy | 290 | 11.0% | 0.0939 | 0.0% | 0.0% | 32.8% |
| kinematics | 344 | 11.9% | 0.0527 | 0.0% | 0.0% | 39.5% |

Mean score by rule (failures at the rule's floor): crps -0.188, binary 0.131, log_score -1.95, abs_rel_error -0.204, crps_log -0.178
Mean step score: 0.662; copied-given rate: 0.2%; steps block present: 99.9%; mean completion tokens: 224

# Eval summary: stage-xm-b2-seed0-confirmatory

- model: `.cache/huggingface/hub/models--unsloth--Llama-3.2-3B-Instruct-bnb-4bit/snapshots/bb1d317a108579fb40e646af8924a5e7ec5604b1` @ `bb1d317a108579fb40e646af8924a5e7ec5604b1`
- generator: `2.0.0+bdbb2279`, reward rule: `crps_log`, prompt: `v5_two_channel`
- generation time: 7028 s
- resumed (D142): session 1; 0 of 340 batches read from `completions.partial.jsonl`, 340 generated in this session
- torch_max_allocated_gib: 4.27
- torch_max_reserved_gib: 5.20
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 7.46

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a6b_pairs | 1020 | 9.3% | 0.203 | 0.0% | 0.0% | 0.0% | 20.1% | 32.8% | 0.0229 | -0.825 | 0.0% |
| a6c_pairs | 1020 | 10.3% | 0.113 | 0.0% | 0.0% | 0.0% | 25.3% | 41.1% | 0.035 | -0.292 | 0.0% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split (**not comparable here**: different generator or verifier version, so omitted).

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| a6b_pairs | this model | 0.1% | 20.1% | 32.8% | -0.91 |
| a6b_pairs | this model, σ × 18.1 |  | 66.4% | 75.4% | -0.871 |
| a6c_pairs | this model | 0.0% | 25.3% | 41.1% | -0.383 |
| a6c_pairs | this model, σ × 18.1 |  | 85.7% | 93.0% | -0.394 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6b_pairs | this model × (a) text scale | 80.6% | 83.2% | -1.11 |
| a6b_pairs | this model × (a') text direct | 48.4% | 61.8% | -0.856 |
| a6b_pairs | this model × (b) oracle per cell (upper bound) | 60.4% | 70.3% | -0.84 |
| a6c_pairs | this model × (a) text scale | 94.8% | 97.5% | -0.659 |
| a6c_pairs | this model × (a') text direct | 55.0% | 73.8% | -0.35 |
| a6c_pairs | this model × (b) oracle per cell (upper bound) | 68.1% | 83.1% | -0.337 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6b_pairs | this model × A5 σ | 30.8% | 46.1% | -0.87 |
| a6c_pairs | this model × A5 σ | 33.7% | 51.1% | -0.357 |

## a6b_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 340 | 6.8% | 15.0% | 60.9% | -0.13 |
| 0.02 | 340 | 340 | 3.2% | 22.1% | 81.8% | -0.124 |
| 0.05 | 340 | 340 | 5.9% | 23.2% | 55.0% | -0.134 |

## a6b_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 20.1% | 0.0405 | 0.0% | 0.0% | 31.9% |
| 2 | 204 | 9.8% | 0.139 | 0.0% | 0.0% | 23.5% |
| 3 | 204 | 8.3% | 0.235 | 0.0% | 0.0% | 18.1% |
| 4 | 204 | 3.9% | 0.307 | 0.0% | 0.0% | 14.7% |
| 5 | 204 | 4.4% | 0.593 | 0.0% | 0.0% | 12.3% |

## a6b_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 438 | 13.0% | 0.0834 | 0.0% | 0.0% | 29.0% |
| energy | 438 | 3.0% | 0.443 | 0.0% | 0.0% | 7.5% |
| kinematics | 144 | 17.4% | 0.0615 | 0.0% | 0.0% | 31.2% |

Mean score by rule (failures at the rule's floor): crps -0.641, binary 0.0931, log_score -5.15, abs_rel_error -0.656, crps_log -0.91
Mean step score: 0.53; copied-given rate: 0.7%; steps block present: 100.0%; mean completion tokens: 191

## a6c_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.02 | 340 | 340 | 15.0% | 25.3% | 66.7% | 0.0282 |
| 0.05 | 680 | 676 | 8.5% | 25.4% | 58.6% | -0.108 |

## a6c_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 16.2% | 0.0671 | 0.0% | 0.0% | 31.4% |
| 2 | 204 | 10.3% | 0.099 | 0.0% | 0.0% | 25.0% |
| 3 | 204 | 8.8% | 0.13 | 0.0% | 0.0% | 20.6% |
| 4 | 204 | 9.3% | 0.159 | 0.0% | 0.0% | 24.0% |
| 5 | 204 | 6.9% | 0.153 | 0.0% | 0.0% | 25.5% |

## a6c_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 320 | 13.4% | 0.0604 | 0.0% | 0.0% | 36.6% |
| energy | 338 | 4.4% | 0.341 | 0.0% | 0.0% | 9.2% |
| kinematics | 362 | 13.0% | 0.0713 | 0.0% | 0.0% | 30.4% |

Mean score by rule (failures at the rule's floor): crps -0.42, binary 0.103, log_score -3.82, abs_rel_error -0.44, crps_log -0.383
Mean step score: 0.568; copied-given rate: 0.1%; steps block present: 100.0%; mean completion tokens: 205

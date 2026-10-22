# Eval summary: stage-b-seed2-a6-sets

- model: `.cache/huggingface/hub/models--unsloth--Qwen2.5-Math-1.5B-Instruct-bnb-4bit/snapshots/e24e9530845f4e3a6648343796883cb509935039` @ `e24e9530845f4e3a6648343796883cb509935039`
- generator: `2.0.0+bdbb2279`, reward rule: `crps_log`, prompt: `v5_two_channel`
- generation time: 3196 s
- torch_max_allocated_gib: 5.52
- torch_max_reserved_gib: 6.72
- device_total_gib: 8.00
- device_name: NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia_smi_peak_used_gib_device_wide: 7.30

| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov | median rel σ | mean reward | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a6_pairs | 1020 | 11.4% | 0.0801 | 0.3% | 0.0% | 0.3% | 31.0% | 49.1% | 0.0313 | -0.211 | 0.0% |
| a6b_pairs | 1020 | 13.0% | 0.0544 | 0.0% | 0.2% | 0.0% | 37.3% | 57.8% | 0.0283 | -0.41 | 0.0% |
| a6c_pairs | 1020 | 14.5% | 0.0519 | 0.3% | 0.0% | 0.2% | 31.5% | 49.6% | 0.025 | -0.111 | 0.1% |

## Standing calibration baseline: σ × 18.1 (crps_fit, `results/posthoc_calibration/posthoc_fit.json`)

SFT-only rows are on the same frozen split (**not comparable here**: different generator or verifier version, so omitted).

| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|---|
| a6_pairs | this model | 0.0% | 31.0% | 49.1% | -0.325 |
| a6_pairs | this model, σ × 18.1 |  | 86.4% | 91.2% | -0.398 |
| a6b_pairs | this model | 0.0% | 37.3% | 57.8% | -0.511 |
| a6b_pairs | this model, σ × 18.1 |  | 84.5% | 87.4% | -0.616 |
| a6c_pairs | this model | 0.0% | 31.5% | 49.6% | -0.22 |
| a6c_pairs | this model, σ × 18.1 |  | 89.4% | 93.9% | -0.254 |

## Conditional post-hoc baselines (D49)

(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| (ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample on the evaluated completions: an upper bound only. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × (a) text scale | 93.9% | 97.1% | -0.782 |
| a6_pairs | this model × (a') text direct | 57.4% | 72.4% | -0.313 |
| a6_pairs | this model × (b) oracle per cell (upper bound) | 64.0% | 77.4% | -0.307 |
| a6b_pairs | this model × (a) text scale | 88.8% | 90.5% | -1.11 |
| a6b_pairs | this model × (a') text direct | 71.1% | 80.5% | -0.524 |
| a6b_pairs | this model × (b) oracle per cell (upper bound) | 58.9% | 72.2% | -0.506 |
| a6c_pairs | this model × (a) text scale | 96.5% | 98.2% | -0.525 |
| a6c_pairs | this model × (a') text direct | 69.6% | 82.2% | -0.215 |
| a6c_pairs | this model × (b) oracle per cell (upper bound) | 67.6% | 81.9% | -0.203 |

## Strongest A5 nested baseline (D50, the judgement since D50): `problem_answer_reasoning/gbt`

Fitted on the SFT-only calibration split; applied to the same completions, so μ is unchanged. SFT-only rows omitted (not comparable here).

| split | rows | 1σ cov | 2σ cov | mean crps_log |
|---|---|---|---|---|
| a6_pairs | this model × A5 σ | 34.3% | 44.7% | -0.315 |
| a6b_pairs | this model × A5 σ | 56.9% | 67.9% | -0.509 |
| a6c_pairs | this model × A5 σ | 46.2% | 59.9% | -0.206 |

## a6_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 338 | 17.6% | 26.0% | 48.3% | -0.182 |
| 0.02 | 340 | 340 | 20.9% | 27.9% | 40.8% | -0.141 |
| 0.05 | 340 | 339 | 23.2% | 38.9% | 54.4% | -0.0817 |

## a6_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 14.2% | 0.0463 | 0.5% | 0.0% | 32.5% |
| 2 | 204 | 11.3% | 0.0819 | 0.5% | 0.0% | 33.5% |
| 3 | 204 | 8.3% | 0.0754 | 0.0% | 0.0% | 33.3% |
| 4 | 204 | 14.7% | 0.0926 | 0.0% | 0.0% | 33.3% |
| 5 | 204 | 8.3% | 0.117 | 0.5% | 0.0% | 22.2% |

## a6_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 318 | 14.2% | 0.048 | 0.0% | 0.0% | 35.8% |
| energy | 330 | 8.5% | 0.115 | 0.0% | 0.0% | 22.4% |
| kinematics | 372 | 11.6% | 0.0883 | 0.8% | 0.0% | 34.4% |

Mean score by rule (failures at the rule's floor): crps -0.349, binary 0.114, log_score -2.61, abs_rel_error -0.369, crps_log -0.325
Mean step score: 0.712; copied-given rate: 0.4%; steps block present: 99.7%; mean completion tokens: 245

## a6b_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.01 | 340 | 339 | 7.4% | 28.9% | 48.0% | -0.0806 |
| 0.02 | 340 | 340 | 7.4% | 34.7% | 52.0% | -0.139 |
| 0.05 | 340 | 339 | 7.6% | 48.4% | 61.5% | -0.0267 |

## a6b_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 19.1% | 0.0287 | 0.0% | 0.0% | 44.6% |
| 2 | 204 | 19.1% | 0.0495 | 0.0% | 0.0% | 41.7% |
| 3 | 204 | 11.8% | 0.0548 | 0.0% | 0.0% | 40.2% |
| 4 | 204 | 6.9% | 0.0874 | 0.0% | 0.0% | 27.9% |
| 5 | 204 | 8.3% | 0.091 | 0.0% | 1.0% | 32.2% |

## a6b_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 438 | 13.5% | 0.0493 | 0.0% | 0.5% | 40.4% |
| energy | 438 | 8.9% | 0.083 | 0.0% | 0.0% | 30.8% |
| kinematics | 144 | 24.3% | 0.0322 | 0.0% | 0.0% | 47.9% |

Mean score by rule (failures at the rule's floor): crps -0.26, binary 0.13, log_score -1.93, abs_rel_error -0.274, crps_log -0.511
Mean step score: 0.636; copied-given rate: 1.0%; steps block present: 100.0%; mean completion tokens: 232

## a6c_pairs: by noise level (D50/A1; σ/σ* on computation-correct answers)

| noise | n | answered | computation-correct | 1σ cov (answered) | 1σ cov (comp.-correct) | median log10(σ/σ*) |
|---|---|---|---|---|---|---|
| 0.02 | 340 | 337 | 22.9% | 31.8% | 59.0% | -0.15 |
| 0.05 | 680 | 680 | 15.0% | 31.3% | 52.9% | -0.254 |

## a6c_pairs: by depth

| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| 1 | 204 | 16.2% | 0.0435 | 1.0% | 0.0% | 35.6% |
| 2 | 204 | 18.6% | 0.04 | 0.0% | 0.0% | 31.9% |
| 3 | 204 | 14.2% | 0.0628 | 0.0% | 0.0% | 31.9% |
| 4 | 204 | 14.2% | 0.0505 | 0.0% | 0.0% | 31.9% |
| 5 | 204 | 9.3% | 0.0878 | 0.5% | 0.0% | 26.1% |

## a6c_pairs: by family

| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |
|---|---|---|---|---|---|---|
| dynamics | 318 | 17.6% | 0.0366 | 0.0% | 0.0% | 39.3% |
| energy | 338 | 8.0% | 0.0986 | 0.3% | 0.0% | 22.3% |
| kinematics | 364 | 17.9% | 0.0433 | 0.5% | 0.0% | 33.1% |

Mean score by rule (failures at the rule's floor): crps -0.216, binary 0.145, log_score -2.27, abs_rel_error -0.229, crps_log -0.22
Mean step score: 0.677; copied-given rate: 0.3%; steps block present: 99.7%; mean completion tokens: 248

# Corrected (4-bit training base, adapters unmerged; D67) vs superseded (bf16-merged) results

## Family S on the seed mean (D62)

| test | superseded: split / control / diff / p / verdict | corrected: split / control / diff / p / verdict | corrected per-seed p |
|---|---|---|---|
| S1a | 0.2167 / 0.1325 / +0.0842 / 0 / supported | 0.4183 / 0.1200 / +0.2983 / 0 / supported | 7.17e-34 / 2.92e-32 / 9.41e-35 |
| S1b | 0.2167 / 0.1175 / +0.0992 / 0 / supported | 0.4183 / 0.1050 / +0.3133 / 0 / supported | 7.52e-37 / 4.78e-35 / 9.62e-38 |
| S2 | 0.0810 / -0.0170 / +0.0980 / 0.04 / narrowly supported; sensitive to the seed-mean definition | 0.0607 / 0.0146 / +0.0461 / 0.262 / not supported | 0.207 / 0.396 / 0.225 |
| S3 | 0.3567 / 0.3425 / +0.0142 / 0 / supported | 0.3767 / 0.3725 / +0.0042 / 0 / supported | 0 / 0 / 0 |

## Families S and N per seed (Holm within each family)

| seed | eval | S1a | S1b | S2 | S3 | N1 (RL vs A5) | N2 | N3 |
|---|---|---|---|---|---|---|---|---|
| 0 | superseded | p 4.67e-09 ✓ | p 8.02e-12 ✓ | p 0.0608 ✗ | p 0 ✓ | 0.458 vs 0.025, p 1 ✗ | 0.078 vs 0.696, p 1 ✗ | -0.325 vs -0.292, p 1 ✗ |
| 0 | corrected | missing | | | | | | |
| 1 | superseded | p 6.17e-08 ✓ | p 7.28e-11 ✓ | p 0.0202 ✓ | p 0 ✓ | 0.473 vs 0.003, p 1 ✗ | 0.099 vs 0.686, p 1 ✗ | -0.302 vs -0.269, p 1 ✗ |
| 1 | corrected | missing | | | | | | |
| 2 | superseded | p 1.8e-08 ✓ | p 1.08e-10 ✓ | p 0.0919 ✗ | p 0 ✓ | 0.468 vs 0.013, p 1 ✗ | 0.066 vs 0.661, p 1 ✗ | -0.336 vs -0.300, p 1 ✗ |
| 2 | corrected | missing | | | | | | |

N1 = |1σ coverage − 68.3%| (lower is better), N2 = ρ(σ, err), N3 = mean crps_log; ✓ = supported (RL σ better than the A5 σ on the same answers).

## Headline metrics, pooled 400 problems (greedy)

| arm | eval | acc ≤ 1% | 1σ cov | 2σ cov | σ = 0 | median rel. σ | ρ(σ, err) | mean crps_log |
|---|---|---|---|---|---|---|---|---|
| sft | superseded | 34.5% | 13.2% | 18.2% | 18.8% | 0.063% | 0.02 | -0.3386 |
| posthoc | superseded | 34.5% | 34.2% | 39.2% | 18.8% | 1.140% | 0.02 | -0.3361 |
| sft_a5_strongest | superseded | 34.5% | 66.8% | 76.8% | 0.0% | 6.765% | 0.69 | -0.3051 |
| uniform_rl287 | superseded | 34.2% | 11.8% | 16.2% | 21.0% | 0.050% | -0.02 | -0.3594 |
| split_rl287 (seed 0) | superseded | 34.5% | 22.5% | 29.0% | 2.2% | 0.244% | 0.08 | -0.3253 |
| split_rl287 (seed 1) | superseded | 36.2% | 21.0% | 27.5% | 6.5% | 0.209% | 0.10 | -0.3022 |
| split_rl287 (seed 2) | superseded | 36.2% | 21.5% | 29.2% | 3.5% | 0.233% | 0.07 | -0.3361 |
| split_rl287_a5_strongest (seed 0) | superseded | 34.5% | 65.8% | 76.5% | 0.0% | 6.516% | 0.70 | -0.2918 |
| split_rl287_a5_strongest (seed 1) | superseded | 36.2% | 68.0% | 78.0% | 0.0% | 6.858% | 0.69 | -0.2691 |
| split_rl287_a5_strongest (seed 2) | superseded | 36.2% | 67.0% | 77.2% | 0.0% | 6.083% | 0.66 | -0.3005 |
| #5 overnight RL, final (step 287) | superseded | 35.5% | 12.8% | 17.5% | 20.2% | 0.050% | -0.01 | -0.3178 |
| #6 overnight2 (uniform), final (step 759) | superseded | 36.0% | 9.8% | 14.5% | 43.2% | 0.017% | -0.04 | -0.2913 |

## Family E (D52; μ fixed, the same 400 prefixes)

| test | superseded: Δcrps_log [95% CI], p, verdict | corrected: Δcrps_log [95% CI], p, verdict |
|---|---|---|
| E1a-P | +0.0024 [+0.0018, +0.0030], p 0, supported | +0.0077 [+0.0062, +0.0092], p 0, supported |
| E1b-P | +0.0020 [+0.0015, +0.0025], p 0, supported | +0.0075 [+0.0061, +0.0090], p 0, supported |
| E1a-N | -0.0310 [-0.0400, -0.0228], p 1, not supported | -0.0258 [-0.0342, -0.0181], p 1, not supported |
| E1b-N | -0.0314 [-0.0404, -0.0233], p 1, not supported | -0.0259 [-0.0345, -0.0181], p 1, not supported |

| σ source | eval | mean crps_log | 1σ cov | ρ(σ, err) | median rel. σ | σ = 0 |
|---|---|---|---|---|---|---|
| SFT stated σ | superseded | -0.3386 | 13.2% | -0.02 | 0.061% | 18.8% |
| A5 strongest σ | superseded | -0.3051 | 66.8% | 0.70 | 7.985% | 0.0% |
| E1a | superseded | -0.3362 | 29.2% | 0.06 | 0.567% | 1.0% |
| E1b | superseded | -0.3366 | 28.2% | 0.08 | 0.500% | 0.0% |
| SFT stated σ | corrected | -0.3386 | 12.2% | 0.00 | 0.057% | 16.3% |
| A5 strongest σ | corrected | -0.3051 | 66.8% | 0.70 | 7.985% | 0.0% |
| E1a | corrected | -0.3309 | 48.5% | 0.01 | 3.224% | 0.0% |
| E1b | corrected | -0.3310 | 45.5% | 0.01 | 2.000% | 0.0% |
| E1a step 150 (superseded) | (descriptive) | -0.3369 | 26.0% | 0.10 | 0.418% | 1.3% |
| E1a step 150 (corrected) | (descriptive) | -0.3315 | 46.5% | -0.00 | 2.660% | 0.0% |
| E1b step 150 (superseded) | (descriptive) | -0.3372 | 25.2% | 0.08 | 0.500% | 0.0% |
| E1b step 150 (corrected) | (descriptive) | -0.3323 | 41.2% | 0.02 | 2.000% | 0.0% |

## Determinism: D65 tapes (primary, D67) vs tonight's reruns

| eval | rows | identical completions | identical rewards |
|---|---|---|---|
| split seed 0 @287 greedy | 400 | 100.0% | 100.0% |
| E1a final greedy | 400 | 100.0% | 100.0% |
| split seed 0 @287 sampled k=4 | missing | | |
| E1a final sampled k=4 | missing | | |

## σ instability (D62): within-problem spread of sampled σ (k = 4)

| model, sampled run | median SD of ln σ | greedy σ mean mid-rank | greedy σ below all samples |
|---|---|---|---|
| split seed 0 @287, D65 | 0.85 | 0.50 | 4.5% |
| split seed 0 @287, superseded (bf16 merged) | 1.01 | 0.43 | 7.5% |
| E1a final, D65 | 0.35 | 0.45 | 2.2% |
| E1a final, superseded (bf16 merged) | 0.94 | 0.45 | 3.5% |

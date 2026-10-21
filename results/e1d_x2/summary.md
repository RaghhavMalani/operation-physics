# E1d on the X2 pairs (D102, pre-registered): 516 kept problems

**Reading (D102):** T1 not supported: E1d's ρ is between-problem ranking only (the X3 reading).

Every comparator reproduces its published pairwise accuracy (max gap 0.0e+00). All CIs: cluster bootstrap over problems (B = 10,000).

| test | estimate [95% CI] | p (one-sided) | Holm |
|---|---|---|---|
| T1: E1d > 0.5 | 0.512 [0.483, 0.541] | 0.2083 | not supported |
| T2: E1d − SFT stated σ > 0 | +0.013 [-0.021, +0.048] | 0.225 | not supported |
| T3: E1d − X3 student (a) > 0 | -0.001 [-0.032, +0.031] | 0.5211 | not supported |
| e1d - self_consistency (reported, no test) | -0.131 [-0.170, -0.091] | – | – |
| e1d - sft_suffix (reported, no test) | +0.024 [-0.009, +0.056] | – | – |

| method | pairwise acc. [95% CI] | pooled pair-weighted | separated pairs | ρ(σ, error) across answers | AUROC (correct) | answers without confidence |
|---|---|---|---|---|---|---|
| SFT stated σ (X2 method 1, sampled) | 0.499 [0.474, 0.524] | 0.496 [0.472, 0.520] | 0.501 [0.442, 0.558] | -0.001 [-0.042, +0.039] | 0.490 [0.467, 0.512] | 1 |
| X3 student (a) σ suffix | 0.513 [0.484, 0.543] | 0.515 [0.485, 0.544] | 0.525 [0.459, 0.590] | +0.346 [+0.298, +0.389] | 0.678 [0.653, 0.700] | 5 |
| self-consistency (D92c, k = 8; post-hoc) | 0.643 [0.616, 0.671] | 0.681 [0.654, 0.708] | 0.861 [0.820, 0.901] | +0.672 [+0.643, +0.698] | 0.823 [0.803, 0.841] | 1 |
| E1d σ suffix (lr 5e-5, step 300) | 0.512 [0.483, 0.541] | 0.516 [0.487, 0.544] | 0.502 [0.435, 0.568] | +0.350 [+0.307, +0.389] | 0.679 [0.657, 0.700] | 21 |
| SFT greedy σ suffix (no-RL control) | 0.489 [0.461, 0.515] | 0.490 [0.462, 0.517] | 0.461 [0.396, 0.524] | +0.024 [-0.026, +0.074] | 0.499 [0.473, 0.525] | 4 |

| method | depth 1 | depth 2 | depth 3 | depth 4 |
|---|---|---|---|---|
| SFT stated σ (X2 method 1, sampled) | 0.497 (n 109) | 0.460 (n 137) | 0.509 (n 137) | 0.530 (n 133) |
| X3 student (a) σ suffix | 0.482 (n 109) | 0.501 (n 137) | 0.520 (n 137) | 0.543 (n 133) |
| self-consistency (D92c, k = 8; post-hoc) | 0.666 (n 109) | 0.590 (n 137) | 0.669 (n 137) | 0.652 (n 133) |
| E1d σ suffix (lr 5e-5, step 300) | 0.509 (n 109) | 0.526 (n 137) | 0.506 (n 137) | 0.506 (n 133) |
| SFT greedy σ suffix (no-RL control) | 0.463 (n 109) | 0.502 (n 137) | 0.470 (n 137) | 0.515 (n 133) |

| suffix eval | rows | failed | μ changed | σ = 0 share | top suffix share | collapsed | pipeline flag |
|---|---|---|---|---|---|---|---|
| E1d σ suffix (lr 5e-5, step 300) | 4116 | 21 | 13 | 0.0% | 2.1% (`0.01 s</answer>`) | False | False |
| SFT greedy σ suffix (no-RL control) | 4116 | 4 | 3 | 18.3% | 4.0% (`0.01 s</answer>`) | False | False |

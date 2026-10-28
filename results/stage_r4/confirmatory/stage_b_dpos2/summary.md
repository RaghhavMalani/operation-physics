# Phase 12 stage B (D104): the supervised σ* ceiling, seeds [0, 1, 2]

**Reading (D104):** the ceiling propagates (B1, B2 supported; A6b equivalent to 0; A6 label full propagation).

3060 problems, 1530 pairs (A6 + A6b + A6c). Seed means; CIs from one bootstrap over pairs shared by every seed (B = 10,000). Pooled problem-level evidence, not seed-level significance.

| test | seed mean [95% CI] | per seed | p (one-sided) | Holm |
|---|---|---|---|---|
| B1: A6 slope > 0 (computation-correct pairs) | +1.195 [+0.950, +1.472] | +1.056, +1.312, +1.217 | 0 | supported |
| B2: A6c slope > 0 (all answered pairs) | +0.816 [+0.686, +0.944] | +0.778, +0.770, +0.900 | 0 | supported |
| B3: expected crps_log − RSS > 0 | +0.000 [-0.000, +0.001] | -0.000, +0.001, +0.000 | 0.2014 | not supported |
| B4: expected crps_log − X1 > 0 | -0.001 [-0.001, -0.000] | -0.001, +0.000, -0.001 | 0.9908 | not supported |
| B5: A6 slope − X1's > 0 | +1.180 [+0.938, +1.444] | +1.046, +1.276, +1.219 | 0 | supported |

- A6 label (seed mean, 90% CI +1.195 [+0.992, +1.417]): **full propagation**
- A6b slope on Δ ln truth (seed mean): +0.036 [-0.086, +0.290], 90% CI +0.036 [-0.057, +0.212]; equivalent to 0 (TOST): **True**
- A6c on computation-correct pairs (seed mean): +0.955 [+0.785, +1.111]
- R3-6 σ secondary (τ_σ = max(0.25%, 0.25·σ*)): A6 +1.106 [+0.925, +1.282]; A6c +0.816 [+0.608, +1.010]

| source | A6 slope | A6b slope | A6c slope | expected crps_log | realized | median ln(σ/σ*) | median abs | vol beta | 1σ / 2σ cov. |
|---|---|---|---|---|---|---|---|---|---|
| stage B seed 0 | +1.056 (n 60, full propagation) | -0.025 | +0.778 | -0.0404 | -0.0413 | +0.007 | 0.076 | 0.852 | 69.7% / 91.0% |
| stage B seed 1 | +1.312 (n 51, full propagation) | +0.133 | +0.770 | -0.0389 | -0.0398 | -0.040 | 0.092 | 0.893 | 63.9% / 87.5% |
| stage B seed 2 | +1.217 (n 64, full propagation) | -0.000 | +0.900 | -0.0399 | -0.0408 | -0.058 | 0.102 | 0.892 | 62.9% / 87.1% |
| sigma_star (all pairs) | +1.000 | +0.000 | +1.000 | -0.0370 | -0.0379 | +0.000 | 0.000 | 1.000 | 70.9% / 94.4% |
| rss (all pairs) | +0.000 | +0.000 | +0.000 | -0.0399 | -0.0405 | -0.047 | 0.401 | 0.484 | 63.1% / 84.5% |
| x1 (all pairs) | +0.032 | +0.001 | +0.002 | -0.0392 | -0.0398 | -0.020 | 0.331 | 0.584 | 67.0% / 87.8% |

| seed | steps (planned) | train loss | answered | parse fail | unit mismatch | A1 comp.-correct | σ = 0 | confidence block | D59 verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 750 (750) | 0.116 | 99.5% | 0.2% | 0.2% | 14.5% | 0.0% | 100.0% | full propagation |
| 1 | 750 (750) | 0.117 | 99.7% | 0.1% | 0.1% | 14.7% | 0.0% | 100.0% | magnitude-driven |
| 2 | 750 (750) | 0.117 | 99.2% | 0.2% | 0.5% | 15.2% | 0.0% | 99.9% | full propagation |

# Phase 12 stage B (D104): the supervised σ* ceiling, seeds [0]

**Reading (D104):** the ceiling propagates (B1, B2 supported; A6b equivalent to 0; A6 label full propagation).

3060 problems, 1530 pairs (A6 + A6b + A6c). Seed means; CIs from one bootstrap over pairs shared by every seed (B = 10,000). Pooled problem-level evidence, not seed-level significance.

| test | seed mean [95% CI] | per seed | p (one-sided) | Holm |
|---|---|---|---|---|
| B1: A6 slope > 0 (computation-correct pairs) | +0.883 [+0.431, +1.219] | +0.883 | 0.0002 | supported |
| B2: A6c slope > 0 (all answered pairs) | +0.526 [+0.290, +0.744] | +0.526 | 0 | supported |
| B3: expected crps_log − RSS > 0 | -0.000 [-0.001, +0.001] | -0.000 | 0.5992 | not supported |
| B4: expected crps_log − X1 > 0 | -0.001 [-0.002, -0.000] | -0.001 | 0.9977 | not supported |
| B5: A6 slope − X1's > 0 | +0.864 [+0.413, +1.269] | +0.864 | 0.0002 | supported |

- A6 label (seed mean, 90% CI +0.883 [+0.517, +1.180]): **full propagation**
- A6b slope on Δ ln truth (seed mean): -0.004 [-0.008, +0.003], 90% CI -0.004 [-0.007, +0.002]; equivalent to 0 (TOST): **True**
- A6c on computation-correct pairs (seed mean): +0.954 [+0.834, +1.065]
- R3-6 σ secondary (τ_σ = max(0.25%, 0.25·σ*)): A6 +1.032 [+0.824, +1.219]; A6c +0.735 [+0.354, +1.030]

| source | A6 slope | A6b slope | A6c slope | expected crps_log | realized | median ln(σ/σ*) | median abs | vol beta | 1σ / 2σ cov. |
|---|---|---|---|---|---|---|---|---|---|
| stage B seed 0 | +0.883 (n 54, full propagation) | -0.004 | +0.526 | -0.0395 | -0.0413 | +0.006 | 0.074 | 0.847 | 67.3% / 91.4% |
| sigma_star (all pairs) | +1.000 | +0.000 | +1.000 | -0.0365 | -0.0384 | +0.000 | 0.000 | 1.000 | 67.8% / 94.9% |
| rss (all pairs) | +0.000 | +0.000 | +0.000 | -0.0394 | -0.0412 | -0.081 | 0.398 | 0.489 | 61.6% / 83.8% |
| x1 (all pairs) | +0.063 | -0.000 | -0.002 | -0.0386 | -0.0405 | -0.026 | 0.329 | 0.589 | 64.0% / 87.3% |

| seed | steps (planned) | train loss | answered | parse fail | unit mismatch | A1 comp.-correct | σ = 0 | confidence block | D59 verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 750 (750) | 0.116 | 99.8% | 0.1% | 0.1% | 15.1% | 0.0% | 100.0% | full propagation |

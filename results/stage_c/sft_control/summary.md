# Phase 12 stage B (D104): the supervised σ* ceiling, seeds [0]

**Reading (D104):** B2 not supported: supervised σ* does not teach noise allocation with this base and budget; stage C's bar is X1 / RSS, not stage B.

3060 problems, 1530 pairs (A6 + A6b + A6c). Seed means; CIs from one bootstrap over pairs shared by every seed (B = 10,000). Pooled problem-level evidence, not seed-level significance.

| test | seed mean [95% CI] | per seed | p (one-sided) | Holm |
|---|---|---|---|---|
| B1: A6 slope > 0 (computation-correct pairs) | -0.621 [-1.964, +0.695] | -0.621 | 0.8211 | not supported |
| B2: A6c slope > 0 (all answered pairs) | -1.028 [-1.996, -0.075] | -1.028 | 0.9832 | not supported |
| B3: expected crps_log − RSS > 0 | -0.048 [-0.066, -0.032] | -0.048 | 1 | not supported |
| B4: expected crps_log − X1 > 0 | -0.049 [-0.066, -0.033] | -0.049 | 1 | not supported |
| B5: A6 slope − X1's > 0 | -0.684 [-2.034, +0.642] | -0.684 | 0.8454 | not supported |

- A6 label (seed mean, 90% CI -0.621 [-1.740, +0.495]): **no propagation**
- A6b slope on Δ ln truth (seed mean): +0.245 [+0.041, +0.450], 90% CI +0.245 [+0.072, +0.418]; equivalent to 0 (TOST): **False**
- A6c on computation-correct pairs (seed mean): -1.028 [-1.996, -0.075]
- R3-6 σ secondary (τ_σ = max(0.25%, 0.25·σ*)): A6 -0.621 [-1.964, +0.695]; A6c -1.028 [-1.996, -0.075]

| source | A6 slope | A6b slope | A6c slope | expected crps_log | realized | median ln(σ/σ*) | median abs | vol beta | 1σ / 2σ cov. |
|---|---|---|---|---|---|---|---|---|---|
| stage B seed 0 | -0.621 (n 510, no propagation) | +0.245 | -1.028 | -0.0874 | -0.0896 | -10.527 | 10.527 | -0.763 | 8.5% / 10.5% |
| sigma_star (all pairs) | +1.000 | +0.000 | +1.000 | -0.0365 | -0.0384 | +0.000 | 0.000 | 1.000 | 67.8% / 94.9% |
| rss (all pairs) | +0.000 | +0.000 | +0.000 | -0.0394 | -0.0412 | -0.081 | 0.398 | 0.489 | 61.6% / 83.8% |
| x1 (all pairs) | +0.063 | -0.000 | -0.002 | -0.0386 | -0.0405 | -0.026 | 0.329 | 0.589 | 64.0% / 87.3% |

| seed | steps (planned) | train loss | answered | parse fail | unit mismatch | A1 comp.-correct | σ = 0 | confidence block | D59 verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0 | None (None) | – | 100.0% | 0.0% | 0.0% | 100.0% | 78.7% | 0.0% | no propagation |

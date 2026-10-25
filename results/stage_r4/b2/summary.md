# Phase 12 stage B (D104): the supervised σ* ceiling, seeds [0]

**Reading (D104):** another combination: reported as observed.

3060 problems, 1530 pairs (A6 + A6b + A6c). Seed means; CIs from one bootstrap over pairs shared by every seed (B = 10,000). Pooled problem-level evidence, not seed-level significance.

| test | seed mean [95% CI] | per seed | p (one-sided) | Holm |
|---|---|---|---|---|
| B1: A6 slope > 0 (computation-correct pairs) | -0.291 [-0.954, +0.387] | -0.291 | 0.7999 | not supported |
| B2: A6c slope > 0 (all answered pairs) | +0.468 [+0.298, +0.636] | +0.468 | 0 | supported |
| B3: expected crps_log − RSS > 0 | -0.002 [-0.002, -0.001] | -0.002 | 1 | not supported |
| B4: expected crps_log − X1 > 0 | -0.002 [-0.003, -0.002] | -0.002 | 1 | not supported |
| B5: A6 slope − X1's > 0 | -0.346 [-0.985, +0.338] | -0.346 | 0.8407 | not supported |

- A6 label (seed mean, 90% CI -0.291 [-0.834, +0.273]): **no propagation**
- A6b slope on Δ ln truth (seed mean): -0.020 [-0.129, +0.138], 90% CI -0.020 [-0.108, +0.094]; equivalent to 0 (TOST): **True**
- A6c on computation-correct pairs (seed mean): +0.326 [-0.018, +0.725]
- R3-6 σ secondary (τ_σ = max(0.25%, 0.25·σ*)): A6 +0.078 [-0.261, +0.427]; A6c +0.593 [+0.344, +0.846]

| source | A6 slope | A6b slope | A6c slope | expected crps_log | realized | median ln(σ/σ*) | median abs | vol beta | 1σ / 2σ cov. |
|---|---|---|---|---|---|---|---|---|---|
| stage B seed 0 | -0.291 (n 57, no propagation) | -0.020 | +0.468 | -0.0409 | -0.0430 | -0.275 | 0.480 | 0.616 | 54.7% / 78.9% |
| sigma_star (all pairs) | +1.000 | +0.000 | +1.000 | -0.0365 | -0.0384 | +0.000 | 0.000 | 1.000 | 67.8% / 94.9% |
| rss (all pairs) | +0.000 | +0.000 | +0.000 | -0.0394 | -0.0412 | -0.081 | 0.398 | 0.489 | 61.6% / 83.8% |
| x1 (all pairs) | +0.063 | -0.000 | -0.002 | -0.0386 | -0.0405 | -0.026 | 0.329 | 0.589 | 64.0% / 87.3% |

| seed | steps (planned) | train loss | answered | parse fail | unit mismatch | A1 comp.-correct | σ = 0 | confidence block | D59 verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 750 (750) | 0.119 | 100.0% | 0.0% | 0.0% | 15.1% | 0.0% | 100.0% | no propagation |

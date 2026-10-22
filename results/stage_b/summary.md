# Phase 12 stage B (D104): the supervised σ* ceiling, seeds [0, 1, 2]

**Reading (D104):** B2 not supported: supervised σ* does not teach noise allocation with this base and budget; stage C's bar is X1 / RSS, not stage B.

3060 problems, 1530 pairs (A6 + A6b + A6c). Seed means; CIs from one bootstrap over pairs shared by every seed (B = 10,000). Pooled problem-level evidence, not seed-level significance.

| test | seed mean [95% CI] | per seed | p (one-sided) | Holm |
|---|---|---|---|---|
| B1: A6 slope > 0 (computation-correct pairs) | +0.449 [-0.049, +0.892] | +0.181, +0.170, +0.996 | 0.0368 | not supported |
| B2: A6c slope > 0 (all answered pairs) | +0.044 [-0.066, +0.157] | +0.128, +0.010, -0.006 | 0.2146 | not supported |
| B3: expected crps_log − RSS > 0 | -0.002 [-0.003, -0.001] | -0.003, -0.002, -0.002 | 1 | not supported |
| B4: expected crps_log − X1 > 0 | -0.003 [-0.004, -0.002] | -0.004, -0.002, -0.002 | 1 | not supported |
| B5: A6 slope − X1's > 0 | +0.371 [-0.141, +0.855] | +0.132, +0.043, +0.939 | 0.0717 | not supported |

- A6 label (seed mean, 90% CI +0.449 [+0.040, +0.813]): **partial propagation**
- A6b slope on Δ ln truth (seed mean): -0.058 [-0.119, +0.005], 90% CI -0.058 [-0.107, -0.007]; equivalent to 0 (TOST): **True**
- A6c on computation-correct pairs (seed mean): +0.091 [-0.081, +0.256]
- R3-6 σ secondary (τ_σ = max(0.25%, 0.25·σ*)): A6 +0.320 [+0.109, +0.537]; A6c +0.090 [-0.065, +0.246]

| source | A6 slope | A6b slope | A6c slope | expected crps_log | realized | median ln(σ/σ*) | median abs | vol beta | 1σ / 2σ cov. |
|---|---|---|---|---|---|---|---|---|---|
| stage B seed 0 | +0.181 (n 58, no propagation) | -0.024 | +0.128 | -0.0423 | -0.0443 | -0.402 | 0.528 | 0.644 | 49.8% / 74.2% |
| stage B seed 1 | +0.170 (n 58, no propagation) | -0.050 | +0.010 | -0.0409 | -0.0429 | -0.453 | 0.531 | 0.627 | 47.2% / 72.0% |
| stage B seed 2 | +0.996 (n 57, full propagation) | -0.101 | -0.006 | -0.0411 | -0.0428 | -0.309 | 0.476 | 0.647 | 52.5% / 76.3% |
| sigma_star (all pairs) | +1.000 | +0.000 | +1.000 | -0.0365 | -0.0384 | +0.000 | 0.000 | 1.000 | 67.8% / 94.9% |
| rss (all pairs) | +0.000 | +0.000 | +0.000 | -0.0394 | -0.0412 | -0.081 | 0.398 | 0.489 | 61.6% / 83.8% |
| x1 (all pairs) | +0.063 | -0.000 | -0.002 | -0.0386 | -0.0405 | -0.026 | 0.329 | 0.589 | 64.0% / 87.3% |

| seed | steps (planned) | train loss | answered | parse fail | unit mismatch | A1 comp.-correct | σ = 0 | confidence block | D59 verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 750 (750) | 0.114 | 99.9% | 0.1% | 0.0% | 15.7% | 0.0% | 100.0% | no propagation |
| 1 | 750 (750) | 0.114 | 100.0% | 0.0% | 0.0% | 15.5% | 0.0% | 100.0% | no propagation |
| 2 | 750 (750) | 0.114 | 99.7% | 0.2% | 0.1% | 15.3% | 0.0% | 100.0% | full propagation |

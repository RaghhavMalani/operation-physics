# Phase 12 stage B (D104): the supervised σ* ceiling, seeds [0]

**Reading (D104):** B2 not supported: supervised σ* does not teach noise allocation with this base and budget; stage C's bar is X1 / RSS, not stage B.

3060 problems, 1530 pairs (A6 + A6b + A6c). Seed means; CIs from one bootstrap over pairs shared by every seed (B = 10,000). Pooled problem-level evidence, not seed-level significance.

| test | seed mean [95% CI] | per seed | p (one-sided) | Holm |
|---|---|---|---|---|
| B1: A6 slope > 0 (computation-correct pairs) | -0.000 [-0.000, +0.000] | -0.000 | 0.5456 | not supported |
| B2: A6c slope > 0 (all answered pairs) | +0.086 [-0.158, +0.324] | +0.086 | 0.2426 | not supported |
| B3: expected crps_log − RSS > 0 | -0.003 [-0.004, -0.003] | -0.003 | 1 | not supported |
| B4: expected crps_log − X1 > 0 | -0.004 [-0.004, -0.004] | -0.004 | 1 | not supported |
| B5: A6 slope − X1's > 0 | -0.063 [-0.103, -0.025] | -0.063 | 0.9993 | not supported |

- A6 label (seed mean, 90% CI -0.000 [-0.000, +0.000]): **no propagation**
- A6b slope on Δ ln truth (seed mean): -0.000 [-0.000, +0.000], 90% CI -0.000 [-0.000, +0.000]; equivalent to 0 (TOST): **True**
- A6c on computation-correct pairs (seed mean): +0.086 [-0.158, +0.324]
- R3-6 σ secondary (τ_σ = max(0.25%, 0.25·σ*)): A6 -0.000 [-0.000, +0.000]; A6c +0.086 [-0.158, +0.324]

| source | A6 slope | A6b slope | A6c slope | expected crps_log | realized | median ln(σ/σ*) | median abs | vol beta | 1σ / 2σ cov. |
|---|---|---|---|---|---|---|---|---|---|
| stage B seed 0 | -0.000 (n 510, no propagation) | -0.000 | +0.086 | -0.0427 | -0.0448 | -0.812 | 0.821 | 0.440 | 36.0% / 59.1% |
| sigma_star (all pairs) | +1.000 | +0.000 | +1.000 | -0.0365 | -0.0384 | +0.000 | 0.000 | 1.000 | 67.8% / 94.9% |
| rss (all pairs) | +0.000 | +0.000 | +0.000 | -0.0394 | -0.0412 | -0.081 | 0.398 | 0.489 | 61.6% / 83.8% |
| x1 (all pairs) | +0.063 | -0.000 | -0.002 | -0.0386 | -0.0405 | -0.026 | 0.329 | 0.589 | 64.0% / 87.3% |

| seed | steps (planned) | train loss | answered | parse fail | unit mismatch | A1 comp.-correct | σ = 0 | confidence block | D59 verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0 | None (None) | 0.001 | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | no propagation |

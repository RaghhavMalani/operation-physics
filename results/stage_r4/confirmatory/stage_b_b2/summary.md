# Phase 12 stage B (D104): the supervised σ* ceiling, seeds [0, 1, 2]

**Reading (D104):** another combination: reported as observed.

3060 problems, 1530 pairs (A6 + A6b + A6c). Seed means; CIs from one bootstrap over pairs shared by every seed (B = 10,000). Pooled problem-level evidence, not seed-level significance.

| test | seed mean [95% CI] | per seed | p (one-sided) | Holm |
|---|---|---|---|---|
| B1: A6 slope > 0 (computation-correct pairs) | +0.503 [-0.062, +1.130] | +0.037, +0.439, +1.033 | 0.04 | not supported |
| B2: A6c slope > 0 (all answered pairs) | +0.416 [+0.293, +0.535] | +0.455, +0.373, +0.419 | 0 | supported |
| B3: expected crps_log − RSS > 0 | -0.001 [-0.001, -0.001] | -0.002, -0.001, -0.001 | 1 | not supported |
| B4: expected crps_log − X1 > 0 | -0.002 [-0.002, -0.002] | -0.002, -0.002, -0.001 | 1 | not supported |
| B5: A6 slope − X1's > 0 | +0.424 [-0.163, +1.073] | -0.016, +0.415, +0.872 | 0.077 | not supported |

- A6 label (seed mean, 90% CI +0.503 [+0.032, +1.038]): **full propagation**
- A6b slope on Δ ln truth (seed mean): +0.003 [-0.103, +0.161], 90% CI +0.003 [-0.086, +0.129]; equivalent to 0 (TOST): **True**
- A6c on computation-correct pairs (seed mean): +0.639 [+0.272, +0.908]
- R3-6 σ secondary (τ_σ = max(0.25%, 0.25·σ*)): A6 +0.508 [+0.295, +0.722]; A6c +0.622 [+0.416, +0.826]

| source | A6 slope | A6b slope | A6c slope | expected crps_log | realized | median ln(σ/σ*) | median abs | vol beta | 1σ / 2σ cov. |
|---|---|---|---|---|---|---|---|---|---|
| stage B seed 0 | +0.037 (n 61, no propagation) | +0.095 | +0.455 | -0.0416 | -0.0422 | -0.286 | 0.487 | 0.581 | 56.6% / 79.1% |
| stage B seed 1 | +0.439 (n 58, no propagation) | -0.120 | +0.373 | -0.0408 | -0.0414 | -0.109 | 0.435 | 0.613 | 63.2% / 84.7% |
| stage B seed 2 | +1.033 (n 63, full propagation) | +0.033 | +0.419 | -0.0406 | -0.0413 | -0.053 | 0.412 | 0.578 | 64.6% / 85.7% |
| sigma_star (all pairs) | +1.000 | +0.000 | +1.000 | -0.0370 | -0.0379 | +0.000 | 0.000 | 1.000 | 70.9% / 94.4% |
| rss (all pairs) | +0.000 | +0.000 | +0.000 | -0.0399 | -0.0405 | -0.047 | 0.401 | 0.484 | 63.1% / 84.5% |
| x1 (all pairs) | +0.032 | +0.001 | +0.002 | -0.0392 | -0.0398 | -0.020 | 0.331 | 0.584 | 67.0% / 87.8% |

| seed | steps (planned) | train loss | answered | parse fail | unit mismatch | A1 comp.-correct | σ = 0 | confidence block | D59 verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 750 (750) | 0.119 | 99.9% | 0.1% | 0.0% | 14.9% | 0.0% | 100.0% | no propagation |
| 1 | 750 (750) | 0.119 | 100.0% | 0.0% | 0.0% | 14.1% | 0.0% | 100.0% | no propagation |
| 2 | 750 (750) | 0.119 | 100.0% | 0.0% | 0.0% | 15.4% | 0.0% | 100.0% | full propagation |

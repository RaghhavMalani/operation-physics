# R4-1 to R4-3 (D113): D-pos-v2 and B-v2 on mixed-level data

**Reading:** B-v2: propagates below the practical threshold; no go (R4-3): D-pos-v2 meets the propagation criterion on none of (a), (b), (c).

One seed per arm, development sets (A6 / A6b / A6c). A6c is in-distribution for both arms (mixed per-given levels in training); depth 5 is held out from training. Slopes: 90% CIs (R4-3) from a pairs bootstrap (B = 10,000); paired score differences: 95% CIs, cluster bootstrap over pairs.

**R4-3 per σ source** (C1: A6c 90% lower bound > 0; C2: A6c slope ≥ 0.5; C3: A6b 90% CI in ±0.3):

| source | A6c slope [90% CI] (n) | A6b slope [90% CI] | C1 | C2 | C3 | criterion | A6 slope | expected crps_log | median ln(σ/σ*) | 1σ cov. |
|---|---|---|---|---|---|---|---|---|---|---|
| b2 (a) | +0.468 [+0.33, +0.61] (510) | -0.020 [-0.11, +0.09] | yes | no | yes | propagates below the practical threshold | -0.291 [-0.84, +0.26] | -0.0409 | -0.275 | 54.7% |
| sigma_star | +1.000 [+1.00, +1.00] (510) | +0.000 [+0.00, +0.00] | yes | yes | yes | meets the propagation criterion | +1.000 [+1.00, +1.00] | -0.0365 | +0.000 | 67.8% |
| rss | +0.000 [+0.00, +0.00] (510) | +0.000 [+0.00, +0.00] | no | no | yes | C1 fails: no propagation shown on A6c | +0.000 [+0.00, +0.00] | -0.0394 | -0.081 | 61.6% |
| x1 | -0.002 [-0.00, -0.00] (510) | -0.000 [-0.00, +0.00] | no | no | yes | C1 fails: no propagation shown on A6c | +0.063 [+0.03, +0.10] | -0.0386 | -0.026 | 64.0% |

**Tests for the record (Holm over R1–R8):**

| test | estimate | CI (slopes 90%, differences 95%) | p (one-sided) | Holm |
|---|---|---|---|---|
| R1: D-pos-v2 (a) answer σ: A6c slope > 0 | missing | | 1 | not supported |
| R2: D-pos-v2 (b) line X: A6c slope > 0 | missing | | 1 | not supported |
| R3: D-pos-v2 (c) exact from terms: A6c slope > 0 | missing | | 1 | not supported |
| R4: B-v2: A6c slope > 0 | +0.4680 | [+0.3282, +0.6125] | 0 | supported |
| R5: expected crps_log, D-pos-v2 − B-v2 > 0 | missing | | 1 | not supported |
| R6: D-pos-v2 − RSS > 0 | missing | | 1 | not supported |
| R7: D-pos-v2 − X1 > 0 | missing | | 1 | not supported |
| R8: B-v2 − RSS > 0 | -0.0016 | [-0.0020, -0.0013] | 1 | not supported |

**A6c slope by depth (depth 5 held out from training) and by chain type:**

| source | d1 | d2 | d3 | d4 | d5 | power-law chains | value-dependent chains |
|---|---|---|---|---|---|---|---|
| b2 (a) | +0.535 [+0.17, +0.92] | +0.409 [+0.10, +0.73] | +0.723 [+0.35, +1.06] | +0.400 [+0.09, +0.71] | +0.421 [+0.16, +0.69] | -0.039 [-0.61, +0.83] | +0.474 [+0.33, +0.62] |
| sigma_star | +1.000 [+1.00, +1.00] | +1.000 [+1.00, +1.00] | +1.000 [+1.00, +1.00] | +1.000 [+1.00, +1.00] | +1.000 [+1.00, +1.00] | +1.000 [+1.00, +1.00] | +1.000 [+1.00, +1.00] |
| rss | +0.000 [+0.00, +0.00] | +0.000 [+0.00, +0.00] | +0.000 [+0.00, +0.00] | +0.000 [+0.00, +0.00] | +0.000 [+0.00, +0.00] | +0.000 [+0.00, +0.00] | +0.000 [+0.00, +0.00] |
| x1 | -0.003 [-0.01, +0.00] | -0.001 [-0.00, +0.00] | -0.002 [-0.01, +0.00] | -0.005 [-0.01, -0.00] | +0.001 [-0.00, +0.00] | -0.002 [-0.01, +0.01] | -0.002 [-0.00, -0.00] |

**Data effect (descriptive, expected crps_log):** dpos2_minus_dpos: missing; b2_minus_stage_b_mean: +0.0005 [-0.0002, +0.0016].

b2: answered 100.0%, parse fail 0.0%, A1 computation-correct 15.1%, σ = 0 0.0%, D59 verdict no propagation.

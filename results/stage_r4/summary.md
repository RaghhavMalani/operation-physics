# R4-1 to R4-3 (D113): D-pos-v2 and B-v2 on mixed-level data

**Reading:** D-pos-v2 (a): meets the propagation criterion; D-pos-v2 (b): meets the propagation criterion; D-pos-v2 (c): meets the propagation criterion; B-v2: propagates below the practical threshold; go (R4-3): 3 seeds of both arms, met on a, b, c.

One seed per arm, development sets (A6 / A6b / A6c). A6c is in-distribution for both arms (mixed per-given levels in training); depth 5 is held out from training. Slopes: 90% CIs (R4-3) from a pairs bootstrap (B = 10,000); paired score differences: 95% CIs, cluster bootstrap over pairs.

**R4-3 per σ source** (C1: A6c 90% lower bound > 0; C2: A6c slope ≥ 0.5; C3: A6b 90% CI in ±0.3):

| source | A6c slope [90% CI] (n) | A6b slope [90% CI] | C1 | C2 | C3 | criterion | A6 slope | expected crps_log | median ln(σ/σ*) | 1σ cov. |
|---|---|---|---|---|---|---|---|---|---|---|
| dpos2 (a) | +0.526 [+0.33, +0.71] (508) | -0.004 [-0.01, +0.00] | yes | yes | yes | meets the propagation criterion | +0.883 [+0.53, +1.18] | -0.0395 | +0.006 | 67.3% |
| dpos2 (b) | +0.597 [+0.41, +0.77] (510) | +0.004 [-0.00, +0.01] | yes | yes | yes | meets the propagation criterion | +0.799 [+0.43, +1.12] | -0.0393 | +0.006 | 67.6% |
| dpos2 (c) | +0.670 [+0.51, +0.82] (510) | +0.009 [-0.00, +0.02] | yes | yes | yes | meets the propagation criterion | +0.824 [+0.45, +1.15] | -0.0408 | -0.002 | 66.3% |
| dpos2 (c_S) | +0.667 [+0.51, +0.82] (510) | +0.010 [-0.00, +0.02] | yes | yes | yes | meets the propagation criterion | +1.056 [+0.88, +1.24] | -0.0408 | -0.002 | 66.3% |
| b2 (a) | +0.468 [+0.33, +0.61] (510) | -0.020 [-0.11, +0.09] | yes | no | yes | propagates below the practical threshold | -0.291 [-0.84, +0.26] | -0.0409 | -0.275 | 54.7% |
| sigma_star | +1.000 [+1.00, +1.00] (510) | +0.000 [+0.00, +0.00] | yes | yes | yes | meets the propagation criterion | +1.000 [+1.00, +1.00] | -0.0365 | +0.000 | 67.8% |
| rss | +0.000 [+0.00, +0.00] (510) | +0.000 [+0.00, +0.00] | no | no | yes | C1 fails: no propagation shown on A6c | +0.000 [+0.00, +0.00] | -0.0394 | -0.081 | 61.6% |
| x1 | -0.002 [-0.00, -0.00] (510) | -0.000 [-0.00, +0.00] | no | no | yes | C1 fails: no propagation shown on A6c | +0.063 [+0.03, +0.10] | -0.0386 | -0.026 | 64.0% |

**Tests for the record (Holm over R1–R8):**

| test | estimate | CI (slopes 90%, differences 95%) | p (one-sided) | Holm |
|---|---|---|---|---|
| R1: D-pos-v2 (a) answer σ: A6c slope > 0 | +0.5264 | [+0.3316, +0.7120] | 0 | supported |
| R2: D-pos-v2 (b) line X: A6c slope > 0 | +0.5968 | [+0.4143, +0.7686] | 0 | supported |
| R3: D-pos-v2 (c) exact from terms: A6c slope > 0 | +0.6698 | [+0.5073, +0.8216] | 0 | supported |
| R4: B-v2: A6c slope > 0 | +0.4680 | [+0.3282, +0.6125] | 0 | supported |
| R5: expected crps_log, D-pos-v2 − B-v2 > 0 | +0.0015 | [+0.0008, +0.0022] | 0.0001 | supported |
| R6: D-pos-v2 − RSS > 0 | -0.0001 | [-0.0008, +0.0005] | 0.5992 | not supported |
| R7: D-pos-v2 − X1 > 0 | -0.0008 | [-0.0015, -0.0002] | 0.9977 | not supported |
| R8: B-v2 − RSS > 0 | -0.0016 | [-0.0020, -0.0013] | 1 | not supported |

**A6c slope by depth (depth 5 held out from training) and by chain type:**

| source | d1 | d2 | d3 | d4 | d5 | power-law chains | value-dependent chains |
|---|---|---|---|---|---|---|---|
| dpos2 (a) | +1.000 [+0.86, +1.14] | +0.969 [+0.89, +1.04] | +0.757 [+0.50, +0.95] | +0.383 [-0.05, +0.85] | +0.060 [-0.36, +0.49] | +0.977 [+0.55, +1.24] | +0.492 [+0.29, +0.68] |
| dpos2 (b) | +0.999 [+0.87, +1.14] | +0.965 [+0.89, +1.04] | +0.745 [+0.50, +0.94] | +0.685 [+0.29, +1.07] | +0.109 [-0.29, +0.53] | +1.019 [+0.57, +1.27] | +0.567 [+0.38, +0.75] |
| dpos2 (c) | +1.031 [+0.98, +1.09] | +0.974 [+0.92, +1.02] | +0.820 [+0.59, +0.99] | +0.717 [+0.31, +1.11] | +0.275 [-0.06, +0.62] | +0.922 [+0.56, +1.15] | +0.660 [+0.49, +0.82] |
| dpos2 (c_S) | +1.031 [+0.98, +1.09] | +0.974 [+0.92, +1.02] | +0.815 [+0.58, +0.99] | +0.699 [+0.29, +1.09] | +0.280 [-0.06, +0.62] | +0.922 [+0.56, +1.15] | +0.657 [+0.49, +0.82] |
| b2 (a) | +0.535 [+0.17, +0.92] | +0.409 [+0.10, +0.73] | +0.723 [+0.35, +1.06] | +0.400 [+0.09, +0.71] | +0.421 [+0.16, +0.69] | -0.039 [-0.61, +0.83] | +0.474 [+0.33, +0.62] |
| sigma_star | +1.000 [+1.00, +1.00] | +1.000 [+1.00, +1.00] | +1.000 [+1.00, +1.00] | +1.000 [+1.00, +1.00] | +1.000 [+1.00, +1.00] | +1.000 [+1.00, +1.00] | +1.000 [+1.00, +1.00] |
| rss | +0.000 [+0.00, +0.00] | +0.000 [+0.00, +0.00] | +0.000 [+0.00, +0.00] | +0.000 [+0.00, +0.00] | +0.000 [+0.00, +0.00] | +0.000 [+0.00, +0.00] | +0.000 [+0.00, +0.00] |
| x1 | -0.003 [-0.01, +0.00] | -0.001 [-0.00, +0.00] | -0.002 [-0.01, +0.00] | -0.005 [-0.01, -0.00] | +0.001 [-0.00, +0.00] | -0.002 [-0.01, +0.01] | -0.002 [-0.00, -0.00] |

**D-pos-v2's S lines against the builder's (per term):**

| scope | terms | present | exact | within 10% | median abs ln ratio |
|---|---|---|---|---|---|
| all | 11210 | 99.5% | 47.8% | 66.0% | 0.000 |
| power_law | 5196 | 99.7% | 86.4% | 86.6% | 0.000 |
| value_dependent | 6014 | 99.4% | 14.4% | 48.1% | 0.059 |
| depth 1 | 1418 | 100.0% | 58.2% | 82.3% | 0.000 |
| depth 2 | 2024 | 99.7% | 57.0% | 81.4% | 0.000 |
| depth 3 | 2442 | 99.6% | 50.3% | 68.7% | 0.000 |
| depth 4 | 2624 | 99.3% | 45.7% | 64.4% | 0.000 |
| depth 5 | 2702 | 99.3% | 35.0% | 44.9% | 0.025 |
| depth 1, value-dependent | 852 | 100.0% | 30.6% | 70.7% | 0.005 |
| depth 2, value-dependent | 1114 | 99.6% | 24.8% | 69.1% | 0.009 |
| depth 3, value-dependent | 1252 | 99.7% | 10.8% | 46.6% | 0.056 |
| depth 4, value-dependent | 1338 | 99.0% | 10.5% | 46.9% | 0.067 |
| depth 5, value-dependent | 1458 | 99.2% | 3.6% | 21.5% | 0.506 |

Line: present 100.0%, parsed 99.9%, term count right 97.9%, levels right in order 99.5%, arithmetic right 34.8%, copy right 84.5% (3052 checkable), σ = 0 while X > 0 0.0%; extra S lines 1.

**Data effect (descriptive, expected crps_log):** dpos2_minus_dpos: +0.0010 [+0.0003, +0.0017]; b2_minus_stage_b_mean: +0.0005 [-0.0002, +0.0016].

dpos2: answered 99.8%, parse fail 0.1%, A1 computation-correct 15.1%, σ = 0 0.0%, D59 verdict full propagation.

b2: answered 100.0%, parse fail 0.0%, A1 computation-correct 15.1%, σ = 0 0.0%, D59 verdict no propagation.

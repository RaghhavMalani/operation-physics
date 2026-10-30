# R5-6 part J: Jacobians outside the training range (D147), D-pos-v2 and B-v2, 3 seeds

Seeds: D-pos-v2 [0, 1, 2], B-v2 [0, 1, 2] (complete). Problems 1632. Bootstrap: stage B's joint pairs bootstrap, B = 10000. Comparators: clamped, rss, sigma_star, x1 ({'x1': "used: D99's reproduction passed"}).

**P_J: CONFIRMED** (J1 Holm-rejected: True; C2, seed-mean J-sensitivity slope ≥ 0.5: True; C3, J-invariance seed-mean 90% CI inside ±0.3: True).

Reading (fixed in D147 §12): P_J confirmed: D-pos-v2's explicit sensitivity computation propagates measurement noise through Jacobians outside its training range (same chains, same value ranges), across 3 seeds, and is invariant to a change of scale there.

| test | seed-mean value [90% CI] | 95% CI | p (one-sided) | Holm | per seed |
|---|---|---|---|---|---|
| J1: D-pos-v2 (a) answer σ: J-sensitivity seed-mean slope > 0 (primary) | +0.9851 [+0.8681, +1.1034] | [+0.8442, +1.1281] | 0 | rejected | 0: +0.9409, 1: +0.9672, 2: +1.0471 |
| J2: explicit − implicit: D-pos-v2 (a) slope − B-v2 slope > 0 | +0.5296 [+0.3770, +0.6758] | [+0.3492, +0.7044] | 0 | rejected | explicit +0.9851, implicit +0.4555 |
| J3: D-pos-v2 − RSS > 0 (expected crps_log) | +0.0011 [-0.0002, +0.0020] | [-0.0004, +0.0020] | 0.072 | – | 0: +0.0003, 1: +0.0012, 2: +0.0019 |
| J4: D-pos-v2 − X1 > 0 (expected crps_log) | +0.0000 [-0.0013, +0.0008] | [-0.0015, +0.0009] | 0.4419 | – | 0: -0.0009, 1: +0.0001, 2: +0.0008 |
| J5: B-v2 − RSS > 0 (expected crps_log) | -0.0005 [-0.0008, -0.0002] | [-0.0009, -0.0001] | 0.9974 | – | 0: -0.0010, 1: -0.0003, 2: -0.0002 |
| J6: B-v2 − X1 > 0 (expected crps_log) | -0.0016 [-0.0019, -0.0013] | [-0.0020, -0.0012] | 1 | – | 0: -0.0021, 1: -0.0014, 2: -0.0013 |

## Criteria per σ source (seed means; J sensitivity in A6c's place, J invariance in A6b's)

| source | J-sensitivity slope [90% CI] | C1 | C2 | C3 (J-invariance in ±0.3) | J-invariance slope |
|---|---|---|---|---|---|
| dpos2_a | +0.985 [+0.868, +1.103] | True | True | True | +0.026 [-0.055, +0.119] |
| dpos2_b | +1.007 [+0.861, +1.172] | True | True | True | +0.026 [-0.028, +0.078] |
| dpos2_c | +0.980 [+0.835, +1.143] | True | True | True | +0.004 [-0.081, +0.082] |
| dpos2_c_S | +0.930 [+0.815, +1.046] | True | True | True | +0.045 [-0.044, +0.128] |
| b2_a | +0.455 [+0.348, +0.564] | True | False | False | +0.184 [-0.032, +0.400] |
| comparator sigma_star | +1.000 [+1.000, +1.000] | | | | +0.000 [-0.000, +0.000] |
| comparator rss | +0.000 [+0.000, +0.000] | | | | +0.000 [+0.000, +0.000] |
| comparator x1 | -0.001 [-0.003, +0.002] | | | | +0.006 [-0.002, +0.014] |
| comparator clamped | +0.831 [+0.737, +0.919] | | | | +0.000 [+0.000, +0.000] |

Mean oracle-centred expected crps_log: sigma_star -0.0367, rss -0.0403, x1 -0.0392, clamped -0.0378.

## Descriptive (D147 §10; outside the Holm family)

| split | pairs | D-pos-v2 (a) J-sens | B-v2 (a) J-sens | D-pos-v2 (a) J-inv | σ* J-sens | clamped J-sens |
|---|---|---|---|---|---|---|
| depth_1 | 204 | +0.955 [+0.683, +1.243] | +0.252 [+0.027, +0.489] | +0.060 [+0.005, +0.144] | +1.000 [+1.000, +1.000] | +0.637 [+0.418, +0.823] |
| depth_2 | 204 | +0.820 [+0.610, +1.036] | +0.665 [+0.462, +0.852] | -0.029 [-0.284, +0.197] | +1.000 [+1.000, +1.000] | +0.884 [+0.689, +1.049] |
| depth_3 | 204 | +1.134 [+0.932, +1.350] | +0.295 [+0.089, +0.486] | +nan [+nan, +nan] | +1.000 [+1.000, +1.000] | +0.813 [+0.574, +0.996] |
| depth_4 | 204 | +0.999 [+0.789, +1.210] | +0.591 [+0.367, +0.819] | +nan [+nan, +nan] | +1.000 [+1.000, +1.000] | +0.958 [+0.839, +1.070] |
| exceedance_below_median | 408 | +1.086 [+0.963, +1.217] | +0.526 [+0.365, +0.689] | +0.004 [-0.123, +0.152] | +1.000 [+1.000, +1.000] | +0.987 [+0.910, +1.059] |
| exceedance_above_median | 408 | +0.835 [+0.630, +1.043] | +0.360 [+0.220, +0.493] | +0.067 [+0.011, +0.157] | +1.000 [+1.000, +1.000] | +0.623 [+0.446, +0.784] |
| training_count_1_to_2 | 385 | +0.941 [+0.754, +1.140] | +0.279 [+0.137, +0.410] | +0.167 [+nan, +nan] | +1.000 [+1.000, +1.000] | +0.772 [+0.629, +0.896] |
| training_count_3_plus | 431 | +1.026 [+0.891, +1.160] | +0.615 [+0.463, +0.772] | +0.005 [-0.075, +0.093] | +1.000 [+1.000, +1.000] | +0.885 [+0.754, +0.998] |
| strict | 255 | +0.895 [+0.650, +1.116] | +0.390 [+0.086, +0.724] | +0.055 [-0.008, +0.148] | +1.000 [+1.000, +1.000] | +0.710 [+0.571, +0.848] |
| not_strict | 561 | +1.007 [+0.874, +1.144] | +0.475 [+0.366, +0.582] | -0.015 [-0.128, +0.101] | +1.000 [+1.000, +1.000] | +0.871 [+0.757, +0.976] |
| rank_flip_sensitivity | 43 | +1.123 [+0.878, +1.380] | +0.178 [-0.134, +0.412] | – | +1.000 [+1.000, +1.000] | +0.357 [+0.004, +0.801] |
| no_rank_flip_sensitivity | 365 | +0.969 [+0.841, +1.099] | +0.489 [+0.372, +0.607] | – | +1.000 [+1.000, +1.000] | +0.891 [+0.834, +0.947] |

**D-pos-v2's S lines, out-of-range vs in-range terms (per seed):**

- seed 0: out of range 3952 terms, present 99.8%, within 10% 62.9%; in range 2056 terms, present 99.5%, within 10% 75.0%
- seed 1: out of range 3952 terms, present 99.3%, within 10% 63.7%; in range 2056 terms, present 99.5%, within 10% 70.7%
- seed 2: out of range 3952 terms, present 99.3%, within 10% 57.8%; in range 2056 terms, present 99.2%, within 10% 69.1%

**How far outside (quartiles; per pair in pair_distances.jsonl):**

- a6b_pairs: exceedance 0.177, 0.338, 0.736; nearest_training_linf 0.185, 0.376, 0.782; n_training 1, 3, 5; clamped_log_ratio_hi -0.097, 0.0599, 0.218
- a6c_pairs: exceedance 0.175, 0.383, 0.773; nearest_training_linf 0.19, 0.406, 0.84; n_training 1, 3, 5; clamped_log_ratio_hi -0.111, 0.0183, 0.231

**Context, not a test: R4-3's confirmatory depths 1–4 (#32), in-range Jacobians, other chain mix:**

- dpos2_a: A6c +0.855 [+0.734, +0.966], A6b +0.035 [-0.055, +0.213]
- dpos2_b: A6c +0.942 [+0.782, +1.129], A6b -0.017 [-0.040, +0.001]
- dpos2_c: A6c +0.912 [+0.769, +1.082], A6b -0.016 [-0.039, +0.001]
- dpos2_c_S: A6c +0.845 [+0.749, +0.933], A6b +0.001 [-0.004, +0.007]
- b2_a: A6c +0.379 [+0.259, +0.497], A6b +0.003 [-0.088, +0.133]

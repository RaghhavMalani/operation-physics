# The depth-5 discrepancy: D-pos-v2 seed 0, A6c, development vs frozen (descriptive; D133 item 2)

Estimator: r4_analysis.slope (FINDINGS #30); σ (a), the answer's σ. Same adapter, same code; only the problem set differs.

| depth | set | pairs used / all | slope [90% CI] | Theil-Sen [90% CI] | SD Δln σ* | corr | gross (>10×) | slope w/o gross | w/o top 1 / 3 / 5 influential | power-law n, slope | value-dep. n, slope |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | development | 102 / 102 | +1.000 [+0.863, +1.140] | +1.021 [+0.926, +1.133] | 0.253 | +0.84 | 0 | +1.000 | +1.050 / +1.070 / +1.085 | 26, +1.020 | 76, +0.974 |
| 1 | frozen | 102 / 102 | +0.855 [+0.561, +1.084] | +0.869 [+0.784, +0.942] | 0.268 | +0.56 | 0 | +0.855 | +0.979 / +0.964 / +0.984 | 28, +0.995 | 74, +0.864 |
| 2 | development | 102 / 102 | +0.969 [+0.890, +1.042] | +0.963 [+0.876, +1.026] | 0.295 | +0.81 | 0 | +0.969 | +0.998 / +1.001 / +1.003 | 20, +0.612 | 82, +0.913 |
| 2 | frozen | 100 / 102 | +0.956 [+0.810, +1.082] | +0.911 [+0.773, +1.018] | 0.276 | +0.61 | 1 | +0.993 | +0.993 / +0.984 / +0.988 | 24, +1.379 | 76, +0.957 |
| 3 | development | 101 / 102 | +0.757 [+0.501, +0.950] | +0.874 [+0.739, +0.996] | 0.252 | +0.55 | 0 | +0.757 | +0.817 / +0.715 / +0.822 | 21, +0.632 | 80, +0.736 |
| 3 | frozen | 102 / 102 | +0.726 [+0.330, +1.075] | +0.876 [+0.722, +0.981] | 0.291 | +0.36 | 1 | +0.836 | +0.603 / +0.849 / +0.869 | 21, +0.048 | 81, +0.678 |
| 4 | development | 101 / 102 | +0.383 [-0.055, +0.847] | +0.875 [+0.651, +1.053] | 0.294 | +0.20 | 0 | +0.383 | +0.508 / +0.793 / +0.886 | 14, +1.310 | 87, +0.300 |
| 4 | frozen | 99 / 102 | +0.764 [+0.468, +1.058] | +0.870 [+0.706, +1.021] | 0.300 | +0.36 | 2 | +0.777 | +0.854 / +1.002 / +1.017 | 18, +0.637 | 81, +0.792 |
| 5 | development | 102 / 102 | +0.060 [-0.356, +0.488] | +0.477 [+0.121, +0.734] | 0.350 | +0.03 | 3 | +0.425 | +0.202 / +0.425 / +0.603 | 4, +0.323 | 98, +0.048 |
| 5 | frozen | 102 / 102 | +0.803 [+0.502, +1.089] | +0.745 [+0.399, +1.008] | 0.336 | +0.39 | 2 | +0.758 | +0.874 / +0.865 / +0.952 | 2, – | 100, +0.798 |

## By level cell (the pair's larger level; x = Δ ln σ* > 0 by construction, so y < 0 is a reversed direction)

| depth | set | level | n | mean Δ ln σ* | mean Δ ln σ_stated | reversed (y < 0) | within-cell slope |
|---|---|---|---|---|---|---|---|
| 1 | development | 0.02 | 34 | +0.555 | +0.562 | 0 | +1.221 |
| 1 | development | 0.05 | 68 | +0.758 | +0.778 | 0 | +0.980 |
| 1 | frozen | 0.02 | 34 | +0.551 | +0.516 | 1 | +0.745 |
| 1 | frozen | 0.05 | 68 | +0.755 | +0.694 | 2 | +0.856 |
| 2 | development | 0.02 | 34 | +0.502 | +0.494 | 0 | +0.535 |
| 2 | development | 0.05 | 68 | +0.748 | +0.730 | 1 | +0.982 |
| 2 | frozen | 0.02 | 33 | +0.527 | +0.523 | 3 | +0.980 |
| 2 | frozen | 0.05 | 67 | +0.748 | +0.645 | 4 | +1.024 |
| 3 | development | 0.02 | 34 | +0.505 | +0.430 | 0 | +0.198 |
| 3 | development | 0.05 | 67 | +0.744 | +0.641 | 4 | +0.746 |
| 3 | frozen | 0.02 | 34 | +0.518 | +0.508 | 1 | +0.536 |
| 3 | frozen | 0.05 | 68 | +0.772 | +0.647 | 4 | +0.768 |
| 4 | development | 0.02 | 34 | +0.499 | +0.274 | 7 | +0.063 |
| 4 | development | 0.05 | 67 | +0.795 | +0.496 | 12 | +0.281 |
| 4 | frozen | 0.02 | 33 | +0.506 | +0.233 | 7 | -2.031 |
| 4 | frozen | 0.05 | 66 | +0.794 | +0.678 | 2 | +0.612 |
| 5 | development | 0.02 | 34 | +0.488 | +0.255 | 10 | -0.431 |
| 5 | development | 0.05 | 68 | +0.866 | +0.452 | 10 | -0.100 |
| 5 | frozen | 0.02 | 34 | +0.490 | +0.123 | 10 | -3.145 |
| 5 | frozen | 0.05 | 68 | +0.881 | +0.463 | 9 | +0.831 |

## The most influential depth-5 pairs (leave-one-out change of the slope)

**development:**
- ['kinematics', 5, 0.05, 2001200000] (value_dependent): Δ ln σ* +1.606, Δ ln σ_stated -1.443, LOO change -0.142
- ['kinematics', 5, 0.05, 2001200015] (value_dependent): Δ ln σ* +1.478, Δ ln σ_stated -1.127, LOO change -0.098
- ['kinematics', 5, 0.05, 2001200012] (value_dependent): Δ ln σ* +1.381, Δ ln σ_stated -1.095, LOO change -0.082
- ['kinematics', 5, 0.05, 2001200024] (value_dependent): Δ ln σ* +1.434, Δ ln σ_stated -0.852, LOO change -0.076
- ['dynamics', 5, 0.02, 2001400022] (value_dependent): Δ ln σ* +0.444, Δ ln σ_stated +2.670, LOO change -0.056

**frozen:**
- ['kinematics', 5, 0.02, 2301400024] (value_dependent): Δ ln σ* +0.440, Δ ln σ_stated +2.663, LOO change -0.071
- ['kinematics', 5, 0.02, 2301400012] (value_dependent): Δ ln σ* +0.414, Δ ln σ_stated -1.974, LOO change +0.062
- ['energy', 5, 0.02, 2301400029] (value_dependent): Δ ln σ* +0.433, Δ ln σ_stated +1.831, LOO change -0.049
- ['kinematics', 5, 0.05, 2301200012] (value_dependent): Δ ln σ* +1.244, Δ ln σ_stated -0.232, LOO change -0.044
- ['kinematics', 5, 0.05, 2301200003] (value_dependent): Δ ln σ* +1.247, Δ ln σ_stated -0.136, LOO change -0.040


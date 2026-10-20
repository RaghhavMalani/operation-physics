# X2 robustness checks (D92: post-hoc, exploratory)

X2's 516 kept problems. Every recomputed X2 method reproduces results/x2/results.json (max gap 0.0e+00). All CIs: bootstrap over problems.

Separated pairs: the correct answer |error| <= 0.25%, the incorrect >= 10%.

| method | pairwise accuracy (X2) | pooled pair-weighted (cluster bootstrap) | separated pairs only | separated: problems / pairs |
|---|---|---|---|---|
| 1. stated σ, SFT | 0.499 [0.474, 0.524] | 0.496 [0.472, 0.520] | 0.501 [0.442, 0.558] | 158 / 719 |
| 2. stated σ, split credit seed 0 @287 | 0.491 [0.461, 0.521] | 0.489 [0.459, 0.520] | 0.458 [0.391, 0.522] | 158 / 719 |
| 3. A5 p channel (4-bit) | 0.497 [0.481, 0.514] | 0.499 [0.482, 0.515] | 0.516 [0.470, 0.563] | 158 / 719 |
| 4. problem-text-only (check: must be 0.5) | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 158 / 719 |
| A5 σ (secondary) | 0.517 [0.494, 0.538] | 0.521 [0.500, 0.542] | 0.532 [0.477, 0.585] | 158 / 719 |
| 5. hidden-state probe (SFT, 4-bit) | 0.547 [0.520, 0.574] | 0.553 [0.526, 0.580] | 0.635 [0.570, 0.697] | 158 / 719 |
| 6. hidden-state probe, random init | 0.530 [0.503, 0.557] | 0.535 [0.509, 0.561] | 0.584 [0.523, 0.645] | 158 / 719 |
| probe cross-fitted on X2 answers (secondary) | 0.540 [0.514, 0.565] | 0.539 [0.514, 0.564] | 0.568 [0.506, 0.629] | 158 / 719 |
| baseline: shorter completion (D92) | 0.535 [0.511, 0.559] | 0.547 [0.524, 0.569] | 0.607 [0.549, 0.663] | 158 / 719 |
| baseline: agreement with the other samples (D92) | 0.643 [0.616, 0.671] | 0.681 [0.654, 0.708] | 0.861 [0.820, 0.901] | 158 / 719 |
| x4_sft | 0.524 [0.498, 0.550] | 0.513 [0.489, 0.539] | 0.544 [0.485, 0.603] | 158 / 719 |
| x4_direct | 0.526 [0.500, 0.553] | 0.515 [0.490, 0.541] | 0.542 [0.483, 0.603] | 158 / 719 |
| x4_perturbed | 0.524 [0.498, 0.551] | 0.515 [0.489, 0.541] | 0.557 [0.494, 0.620] | 158 / 719 |
| x4_shuffled | 0.513 [0.488, 0.539] | 0.509 [0.484, 0.534] | 0.490 [0.427, 0.550] | 158 / 719 |
| x4_random_init | 0.489 [0.465, 0.514] | 0.492 [0.468, 0.515] | 0.506 [0.452, 0.561] | 158 / 719 |

| completion length (tokens) | value [95% CI] |
|---|---|
| correct answers, mean | 186.363 [179.679, 193.006] |
| incorrect answers, mean | 203.220 [197.547, 208.786] |
| within problem, correct − incorrect | -1.792 [-2.734, -0.991] |
| problems where the correct answers are shorter on average | 0.545 |

| paired difference (over problems) | mean [95% CI] | published |
|---|---|---|
| probe - sft_sigma | +0.048 [+0.011, +0.086] | +0.048 [+0.011, +0.086] |
| probe - random_init_probe | +0.017 [-0.015, +0.050] | +0.017 [-0.015, +0.050] |
| a5_p - probe | -0.050 [-0.080, -0.020] | -0.050 [-0.080, -0.020] |
| self_consistency - probe | +0.096 [+0.058, +0.133] | – |
| length - probe | -0.012 [-0.045, +0.021] | – |

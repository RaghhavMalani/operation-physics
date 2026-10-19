# X2: within-problem correctness pairs (D74)

516 kept problems of 1200 sampled candidates; 5390 (correct, incorrect) pairs. Correct answers per kept problem: {'1': 183, '2': 87, '3': 58, '4': 41, '5': 43, '6': 40, '7': 64}.

**Reading (D74):** another combination: reported as observed.

| method | pairwise accuracy [95% CI] | p (> 0.5) | Holm | ties / missing |
|---|---|---|---|---|
| 1. stated σ, SFT | 0.499 [0.474, 0.524] | 0.54 | not supported | 8.9% |
| 2. stated σ, split credit seed 0 @287 | missing | – | not supported (p = 1) | – |
| 3. A5 p channel (4-bit) | 0.497 [0.481, 0.514] | 0.63 | not supported | 64.2% |
| 4. problem-text-only (check: must be 0.5) | 0.500 [0.500, 0.500] | 1 | – | 100.0% |
| 5. hidden-state probe (SFT, 4-bit) | 0.547 [0.520, 0.574] | 0.0001 | supported | 0.5% |
| 6. hidden-state probe, random init | 0.530 [0.503, 0.557] | 0.0132 | not supported | 0.5% |
| A5 σ (secondary) | 0.517 [0.494, 0.538] | 0.0706 | – | 37.5% |
| probe cross-fitted on X2 answers (secondary) | 0.540 [0.514, 0.565] | 0.0014 | – | 0.5% |

| paired difference (over problems) | mean | 95% CI |
|---|---|---|
| probe - sft_sigma | +0.048 | [+0.011, +0.086] |
| probe - random_init_probe | +0.017 | [-0.015, +0.050] |
| a5_p - probe | -0.050 | [-0.080, -0.020] |

| method | depth 1 | depth 2 | depth 3 | depth 4 |
|---|---|---|---|---|
| 1. stated σ, SFT | 0.497 (n 109) | 0.460 (n 137) | 0.509 (n 137) | 0.530 (n 133) |
| 3. A5 p channel (4-bit) | 0.507 (n 109) | 0.492 (n 137) | 0.498 (n 137) | 0.493 (n 133) |
| 4. problem-text-only (check: must be 0.5) | 0.500 (n 109) | 0.500 (n 137) | 0.500 (n 137) | 0.500 (n 133) |
| 5. hidden-state probe (SFT, 4-bit) | 0.544 (n 109) | 0.539 (n 137) | 0.529 (n 137) | 0.577 (n 133) |
| 6. hidden-state probe, random init | 0.502 (n 109) | 0.506 (n 137) | 0.549 (n 137) | 0.559 (n 133) |
| A5 σ (secondary) | 0.537 (n 109) | 0.503 (n 137) | 0.507 (n 137) | 0.524 (n 133) |
| probe cross-fitted on X2 answers (secondary) | 0.565 (n 109) | 0.520 (n 137) | 0.541 (n 137) | 0.538 (n 133) |

Probe (fit set, 1998 calibration completions): layer 12, C 0.001, CV AUROC 0.875. Random init: layer 1, CV AUROC 0.863.
Problem-only check (must be exactly 0.5): {'pairwise_accuracy': 0.5, 'pass': True}.

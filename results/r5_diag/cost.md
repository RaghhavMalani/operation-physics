# R5-8: MC-B's failure split, and sample agreement against its inference cost (descriptive; D133 item 5)

## MC-B programs (every problem counted; D119)

| set | interface (no program / signature / rejected) | runtime (exception / timeout / bad output) | semantic (runs, wrong chain) | correct (run 0) | exact program |
|---|---|---|---|---|---|
| a6c_pairs | 79.7% | 5.1% | 14.7% | 0.5% | 0.5% |
| a6_pairs | 83.5% | 7.5% | 8.7% | 0.3% | 0.3% |

| set, depth | interface | runtime | semantic | correct |
|---|---|---|---|---|
| a6c_pairs, d1 | 77.0% | 5.9% | 15.7% | 1.5% |
| a6c_pairs, d2 | 72.1% | 2.9% | 24.0% | 1.0% |
| a6c_pairs, d3 | 80.9% | 4.4% | 14.7% | 0.0% |
| a6c_pairs, d4 | 84.3% | 6.4% | 9.3% | 0.0% |
| a6c_pairs, d5 | 84.3% | 5.9% | 9.8% | 0.0% |
| a6_pairs, d1 | 79.4% | 10.3% | 9.3% | 1.0% |
| a6_pairs, d2 | 88.2% | 6.9% | 4.4% | 0.5% |
| a6_pairs, d3 | 83.3% | 6.9% | 9.8% | 0.0% |
| a6_pairs, d4 | 86.3% | 4.9% | 8.8% | 0.0% |
| a6_pairs, d5 | 80.4% | 8.3% | 11.3% | 0.0% |

## Self-consistency (X2, #22): within-problem pairwise accuracy vs cost (1200 problems, k = 8)

Measured: 1,877,002 tokens and 7,286 GPU-s for all k = 8 samples, i.e. 1,564 tokens and 6.07 GPU-s per problem; one sample 196 tokens and 0.76 GPU-s.

| readout | pairwise accuracy [95% CI] | generations per problem |
|---|---|---|
| self_consistency | 0.643 [0.616, 0.671] | 8 |
| probe | 0.547 [0.520, 0.574] | 1 (plus a forward pass or a short check) |
| random_init_probe | 0.530 [0.503, 0.557] | 1 (plus a forward pass or a short check) |
| length | 0.535 [0.511, 0.559] | 1 |
| x4_sft | 0.524 [0.498, 0.550] | 1 (plus a forward pass or a short check) |
| sft_sigma | 0.499 [0.474, 0.524] | 1 |

## MC-A's raw spread (#31): AUROC for run 0 wrong vs cost (B-v2; run 0 is its eval)

| set | signal | AUROC | extra generations | extra tokens / problem | extra GPU-s / problem |
|---|---|---|---|---|---|
| a6c_pairs | sigma_star (no model call) | 0.665 | 0 | 0 | 0 |
| a6c_pairs | rss (no model call) | 0.654 | 0 | 0 | 0 |
| a6c_pairs | x1 (no model call) | 0.666 | 0 | 0 | 0 |
| a6c_pairs | B-v2 stated σ (run 0 itself) | 0.655 | 0 | 0 | 0 |
| a6c_pairs | raw spread, k = 2 | 0.638 | 2 | 496 | 1.72 |
| a6c_pairs | raw spread, k = 4 | 0.710 | 4 | 992 | 3.44 |
| a6c_pairs | raw spread, k = 8 | 0.745 | 8 | 1,984 | 6.89 |
| a6c_pairs | (run 0's own cost) | – | 1 | 248 | – |
| a6_pairs | sigma_star (no model call) | 0.560 | 0 | 0 | 0 |
| a6_pairs | rss (no model call) | 0.576 | 0 | 0 | 0 |
| a6_pairs | x1 (no model call) | 0.499 | 0 | 0 | 0 |
| a6_pairs | B-v2 stated σ (run 0 itself) | 0.513 | 0 | 0 | 0 |
| a6_pairs | raw spread, k = 2 | 0.561 | 2 | 488 | 2.12 |
| a6_pairs | raw spread, k = 4 | 0.635 | 4 | 977 | 4.23 |
| a6_pairs | raw spread, k = 8 | 0.670 | 8 | 1,954 | 8.47 |
| a6_pairs | (run 0's own cost) | – | 1 | 244 | – |
| a6b_pairs | sigma_star (no model call) | 0.569 | 0 | 0 | 0 |
| a6b_pairs | rss (no model call) | 0.586 | 0 | 0 | 0 |
| a6b_pairs | x1 (no model call) | 0.583 | 0 | 0 | 0 |
| a6b_pairs | B-v2 stated σ (run 0 itself) | 0.598 | 0 | 0 | 0 |
| a6b_pairs | raw spread, k = 2 | 0.622 | 2 | 464 | 1.94 |
| a6b_pairs | raw spread, k = 4 | 0.665 | 4 | 927 | 3.88 |
| a6b_pairs | raw spread, k = 8 | 0.696 | 8 | 1,855 | 7.75 |
| a6b_pairs | (run 0's own cost) | – | 1 | 232 | – |

# Item 3: within-problem readouts, 7B vs 1.5B (D151 / D152; exploratory, not the paper)

| readout | 7b (844 kept) | SFT 1.5B X2, same prefix | SFT 1.5B X2 published (516 kept) |
|---|---|---|---|
| self-consistency (k − 1 others within 1%) | 0.827 [0.809, 0.845] | 0.645 [0.617, 0.673] | 0.643 |
| direct P(yes) (X4 direct) | 0.683 [0.663, 0.701] | 0.526 [0.500, 0.551] | 0.526 |
| hidden-state probe, cross-fitted | 0.753 [0.734, 0.771] | 0.539 [0.513, 0.564] | 0.540 |
| the same probe on random weights (D160) | 0.664 [0.646, 0.683] | – | – |
| length only (shorter = more confident) | 0.634 [0.614, 0.654] | 0.534 [0.510, 0.558] | 0.535 |

**7b:** 1176 candidates, 9408 samples; answered 91.1%, correct per sample 51.4%, truncated 8.7%, mean 1127 tokens; kept 844 (71.8%), 8476 pairs. Reasons: {'no_box': 11, 'no_number': 6, 'ok_converted': 85, 'ok_no_unit': 6023, 'ok_unit_after_box': 2448, 'ok_unit_unparsed': 11, 'truncated': 820, 'unit_mismatch': 4}.
probe folds (layer, C): [(19, 0.001), (19, 0.001), (19, 0.001), (19, 0.001), (19, 0.001)] (29 layers).
probe_random folds (layer, C): [(1, 0.001), (1, 0.001), (1, 0.001), (1, 0.001), (1, 0.001)] (29 layers).

| readout / model | comparison | difference [95% CI] |
|---|---|---|
| self_consistency | 7b - SFT 1.5b X2 (same prefix, unpaired) | +0.182 [+0.148, +0.215] |
| yes | 7b - SFT 1.5b X2 (same prefix, unpaired) | +0.157 [+0.125, +0.189] |
| probe | 7b - SFT 1.5b X2 (same prefix, unpaired) | +0.214 [+0.182, +0.245] |
| length | 7b - SFT 1.5b X2 (same prefix, unpaired) | +0.100 [+0.067, +0.131] |
| 7b | probe - probe_random (paired) | +0.088 [+0.068, +0.109] |
| 7b | yes - length (paired) | +0.048 [+0.027, +0.071] |
| 7b | probe - length (paired) | +0.118 [+0.098, +0.139] |

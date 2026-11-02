# Item 3: within-problem readouts, 7B vs 1.5B (D151 / D152; exploratory, not the paper)

| readout | 7b (844 kept) | 1.5b (766 kept) | SFT 1.5B X2, same prefix | SFT 1.5B X2 published (516 kept) |
|---|---|---|---|---|
| self-consistency (k − 1 others within 1%) | 0.827 [0.809, 0.845] | 0.774 [0.753, 0.796] | 0.645 [0.617, 0.673] | 0.643 |
| direct P(yes) (X4 direct) | 0.683 [0.663, 0.701] | 0.572 [0.553, 0.592] | 0.526 [0.500, 0.551] | 0.526 |
| hidden-state probe, cross-fitted | 0.753 [0.734, 0.771] | 0.758 [0.740, 0.775] | 0.539 [0.513, 0.564] | 0.540 |
| the same probe on random weights (D160) | 0.664 [0.646, 0.683] | 0.673 [0.653, 0.692] | – | – |
| length only (shorter = more confident) | 0.634 [0.614, 0.654] | 0.598 [0.578, 0.619] | 0.534 [0.510, 0.558] | 0.535 |

**7b:** 1176 candidates, 9408 samples; answered 91.1%, correct per sample 51.4%, truncated 8.7%, mean 1127 tokens; kept 844 (71.8%), 8476 pairs. Reasons: {'no_box': 11, 'no_number': 6, 'ok_converted': 85, 'ok_no_unit': 6023, 'ok_unit_after_box': 2448, 'ok_unit_unparsed': 11, 'truncated': 820, 'unit_mismatch': 4}.
probe folds (layer, C): [(19, 0.001), (19, 0.001), (19, 0.001), (19, 0.001), (19, 0.001)] (29 layers).
probe_random folds (layer, C): [(1, 0.001), (1, 0.001), (1, 0.001), (1, 0.001), (1, 0.001)] (29 layers).

**1.5b:** 1176 candidates, 9408 samples; answered 97.6%, correct per sample 41.4%, truncated 1.0%, mean 913 tokens; kept 766 (65.1%), 8849 pairs. Reasons: {'no_box': 44, 'no_number': 41, 'ok_converted': 110, 'ok_no_unit': 5643, 'ok_unit_after_box': 3405, 'ok_unit_unparsed': 28, 'truncated': 96, 'unit_mismatch': 41}.
probe folds (layer, C): [(18, 0.01), (20, 0.01), (21, 0.01), (18, 0.01), (19, 0.01)] (29 layers).
probe_random folds (layer, C): [(3, 0.001), (2, 0.001), (2, 0.001), (3, 0.001), (3, 0.001)] (29 layers).

| readout / model | comparison | difference [95% CI] |
|---|---|---|
| self_consistency | 7b - native 1.5b (unpaired) | +0.053 [+0.025, +0.081] |
| self_consistency | 7b - native 1.5b (paired, common kept problems) | +0.076 [+0.045, +0.106] |
| self_consistency | 7b - SFT 1.5b X2 (same prefix, unpaired) | +0.182 [+0.148, +0.215] |
| self_consistency | native 1.5b - SFT 1.5b X2 (same prefix, unpaired) | +0.129 [+0.094, +0.164] |
| yes | 7b - native 1.5b (unpaired) | +0.110 [+0.082, +0.138] |
| yes | 7b - native 1.5b (paired, common kept problems) | +0.120 [+0.090, +0.150] |
| yes | 7b - SFT 1.5b X2 (same prefix, unpaired) | +0.157 [+0.125, +0.189] |
| yes | native 1.5b - SFT 1.5b X2 (same prefix, unpaired) | +0.047 [+0.014, +0.079] |
| probe | 7b - native 1.5b (unpaired) | -0.005 [-0.031, +0.020] |
| probe | 7b - native 1.5b (paired, common kept problems) | -0.003 [-0.029, +0.024] |
| probe | 7b - SFT 1.5b X2 (same prefix, unpaired) | +0.214 [+0.182, +0.245] |
| probe | native 1.5b - SFT 1.5b X2 (same prefix, unpaired) | +0.219 [+0.188, +0.249] |
| probe_random | 7b - native 1.5b (unpaired) | -0.008 [-0.035, +0.019] |
| probe_random | 7b - native 1.5b (paired, common kept problems) | +0.001 [-0.029, +0.029] |
| length | 7b - native 1.5b (unpaired) | +0.036 [+0.008, +0.065] |
| length | 7b - native 1.5b (paired, common kept problems) | +0.040 [+0.011, +0.070] |
| length | 7b - SFT 1.5b X2 (same prefix, unpaired) | +0.100 [+0.067, +0.131] |
| length | native 1.5b - SFT 1.5b X2 (same prefix, unpaired) | +0.064 [+0.033, +0.096] |
| 7b | probe - probe_random (paired) | +0.088 [+0.068, +0.109] |
| 7b | yes - length (paired) | +0.048 [+0.027, +0.071] |
| 7b | probe - length (paired) | +0.118 [+0.098, +0.139] |
| 1.5b | probe - probe_random (paired) | +0.085 [+0.064, +0.105] |
| 1.5b | yes - length (paired) | -0.026 [-0.050, -0.002] |
| 1.5b | probe - length (paired) | +0.159 [+0.138, +0.182] |
| 7b vs native 1.5b | (probe - probe_random): 7b - native 1.5b (unpaired) | +0.003 [-0.026, +0.032] |

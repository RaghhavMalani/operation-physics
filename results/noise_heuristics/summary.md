# Noise heuristics on A6 / A6b / A6c (D95, R3-4 A)

Stated relative σ = max / mean / RSS of the visible levels; `givens` = the chain's givens, `shown` = every measurement shown. Pair slope: A6 and A6c on Δ ln σ* (ideal 1), A6b on Δ ln truth (ideal 0).

## a6_pairs (1020 problems)

| heuristic | pair slope | median abs ln(σ_h / σ*) | median ln(σ_h / σ*) | Spearman ρ(σ_h, σ*) | slope of ln σ_h on ln σ* |
|---|---|---|---|---|---|
| max/givens | +0.000 (not supported; 510 pairs) | 0.972 | -0.972 | 0.648 | 0.491 |
| max/shown | +0.000 (not supported; 510 pairs) | 0.972 | -0.972 | 0.648 | 0.491 |
| mean/givens | +0.000 (not supported; 510 pairs) | 0.972 | -0.972 | 0.648 | 0.491 |
| mean/shown | +0.000 (not supported; 510 pairs) | 0.972 | -0.972 | 0.640 | 0.491 |
| rss/givens | +0.000 (not supported; 510 pairs) | 0.553 | -0.275 | 0.618 | 0.491 |
| rss/shown | +0.000 (not supported; 510 pairs) | 0.559 | -0.165 | 0.620 | 0.490 |

## a6b_pairs (1020 problems)

| heuristic | pair slope | median abs ln(σ_h / σ*) | median ln(σ_h / σ*) | Spearman ρ(σ_h, σ*) | slope of ln σ_h on ln σ* |
|---|---|---|---|---|---|
| max/givens | +0.000 (equivalent to 0; 510 pairs) | 0.573 | -0.573 | 0.865 | 0.776 |
| max/shown | +0.000 (equivalent to 0; 510 pairs) | 0.573 | -0.573 | 0.865 | 0.776 |
| mean/givens | +0.000 (equivalent to 0; 510 pairs) | 0.573 | -0.573 | 0.865 | 0.776 |
| mean/shown | +0.000 (equivalent to 0; 510 pairs) | 0.573 | -0.573 | 0.847 | 0.776 |
| rss/givens | +0.000 (equivalent to 0; 510 pairs) | 0.278 | -0.005 | 0.883 | 0.817 |
| rss/shown | +0.000 (equivalent to 0; 510 pairs) | 0.260 | +0.119 | 0.866 | 0.805 |

## a6c_pairs (1020 problems)

| heuristic | pair slope | median abs ln(σ_h / σ*) | median ln(σ_h / σ*) | Spearman ρ(σ_h, σ*) | slope of ln σ_h on ln σ* |
|---|---|---|---|---|---|
| max/givens | +0.000 (not supported; 510 pairs) | 0.527 | -0.257 | 0.469 | 0.258 |
| max/shown | +0.000 (not supported; 510 pairs) | 0.527 | -0.257 | 0.469 | 0.258 |
| mean/givens | +0.000 (not supported; 510 pairs) | 0.892 | -0.889 | 0.412 | 0.198 |
| mean/shown | +0.000 (not supported; 510 pairs) | 0.960 | -0.960 | 0.421 | 0.190 |
| rss/givens | +0.000 (not supported; 510 pairs) | 0.495 | -0.099 | 0.468 | 0.231 |
| rss/shown | +0.000 (not supported; 510 pairs) | 0.491 | -0.049 | 0.460 | 0.222 |

# X1: a shallow text model of σ* on the A6 / A6b pairs (D73)

Primary feature set: **noise_text_magnitudes**. Parse coverage (fit set): 100.0% of problems with every shown measurement parsed; units converted 100.0%.

| σ source | test R² (ln σ*) | test median abs ln error | A6 slope [90% CI] | A6 label | A6b slope [90% CI] | A6b ≡ 0 | D59 verdict | labelled verdict |
|---|---|---|---|---|---|---|---|---|
| noise | 0.623 | 0.353 | +0.000 [+0.00, +0.00] | no propagation | +0.000 [+0.00, +0.00] | True | no propagation (lookup-like) | no propagation |
| noise_text | 0.789 | 0.225 | +0.002 [-0.00, +0.00] | no propagation | +0.000 [-0.00, +0.00] | True | no propagation (lookup-like) | no propagation |
| noise_text_magnitudes (primary) | 0.831 | 0.200 | +0.063 [+0.03, +0.10] | partial propagation | -0.000 [-0.00, +0.00] | True | propagates | partial propagation |
| reference: σ̂ = σ* (exact propagation) | – | – | +1.000 [+1.00, +1.00] | full propagation | +0.000 [+0.00, +0.00] | True | propagates | full propagation |

## Primary, by depth

| depth | A6 pairs | A6 slope [90% CI] | A6 label | A6b pairs | A6b slope [90% CI] | labelled verdict |
|---|---|---|---|---|---|---|
| 1 | 102 | +0.063 [-0.02, +0.15] | no propagation | 102 | +0.003 [-0.00, +0.01] | no propagation |
| 2 | 102 | +0.110 [+0.04, +0.18] | partial propagation | 102 | +0.001 [-0.00, +0.00] | partial propagation |
| 3 | 102 | +0.074 [-0.00, +0.17] | no propagation | 102 | -0.001 [-0.01, +0.00] | no propagation |
| 4 | 102 | +0.051 [-0.03, +0.14] | no propagation | 102 | +0.002 [-0.00, +0.00] | no propagation |
| 5 | 102 | -0.011 [-0.05, +0.03] | no propagation | 102 | -0.005 [-0.01, -0.00] | no propagation |

## Within-pair agreement (descriptive)

| σ source | A6 corr(Δ ln σ̂, Δ ln σ*) | A6 same sign | A6b mean Δ ln σ̂ | A6 member R² | A6b member R² |
|---|---|---|---|---|---|
| noise | – | 0.0% | +0.000 | 0.273 | 0.761 |
| noise_text | 0.074 | 1.8% | +0.000 | 0.371 | 0.866 |
| noise_text_magnitudes | 0.161 | 48.6% | +0.005 | 0.423 | 0.877 |

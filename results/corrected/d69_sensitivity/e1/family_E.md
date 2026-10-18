# E1 family E (D52), 400 eval prefixes — corrected evals: 4-bit training base, adapters unmerged (D67) — A5 re-fitted on 4-bit SFT completions (D69 sensitivity)

| test | mean Δcrps_log | 95% CI | p (one-sided) | Holm threshold | supported |
|---|---|---|---|---|---|
| E1a-P | +0.0077 | [+0.0062, +0.0092] | 0 | 0.0125 | True |
| E1a-N | -0.0255 | [-0.0349, -0.0170] | 1 | 0.025 | False |
| E1b-P | +0.0075 | [+0.0061, +0.0090] | 0 | 0.0167 | True |
| E1b-N | -0.0257 | [-0.0351, -0.0170] | 1 | 0.05 | False |

| σ source | mean crps_log | 1σ cov | 2σ cov | ρ(σ, err) | median rel σ | σ = 0 | bin entropy | failed | μ changed |
|---|---|---|---|---|---|---|---|---|---|
| SFT stated σ | -0.3386 | 12.2% | 17.2% | 0.001285 | 0.0005666 | 16.3% | 1.393 | 0 | 0 |
| A5 strongest σ | -0.3053 | 67.2% | 76.2% | 0.681 | 0.06951 | 0.0% | 2.829 | 0 | 0 |
| E1a | -0.3309 | 48.5% | 58.2% | 0.01267 | 0.03224 | 0.0% | 2.015 | 0 | 0 |
| E1b | -0.3310 | 45.5% | 56.8% | 0.01435 | 0.02 | 0.0% | 2.004 | 0 | 0 |

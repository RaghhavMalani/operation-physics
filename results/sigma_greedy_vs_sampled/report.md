# Greedy vs sampled stated σ at eval (exploratory, D62; no tests)

| model | decoding | rows | fail | median rel. σ | σ = 0 | 1σ cov | 2σ cov | mean crps_log | ρ(σ, err) |
|---|---|---|---|---|---|---|---|---|---|
| split-credit seed 0 @287 | greedy | 400 | 0.0% | 0.244% | 2.2% | 22.5% | 29.0% | -0.3253 | 0.08 |
| split-credit seed 0 @287 | sampled | 1600 | 0.9% | 0.367% | 5.5% | 19.9% | 25.7% | -0.5168 | 0.08 |
| E1a | greedy | 400 | 0.0% | 0.567% | 1.0% | 29.2% | 35.8% | -0.3362 | 0.06 |
| E1a | sampled | 1600 | 0.2% | 0.773% | 2.7% | 32.6% | 39.1% | -0.3426 | 0.03 |
| E1b | greedy | 400 | 0.0% | 0.500% | 0.0% | 28.2% | 33.0% | -0.3366 | 0.08 |
| E1b | sampled | 1600 | 0.2% | 0.500% | 0.0% | 30.6% | 38.0% | -0.3429 | 0.03 |

| model | median within-problem SD of ln σ | greedy σ mean mid-rank among samples | greedy σ below all samples |
|---|---|---|---|
| split-credit seed 0 @287 | 1.008 | 0.43 | 7.5% |
| E1a | 0.941 | 0.45 | 3.5% |
| E1b | 0.823 | 0.43 | 2.8% |

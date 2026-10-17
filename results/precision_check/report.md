# RL adapters on the 4-bit training base vs the bf16 merge (exploratory, D65; no tests)

| model | base | decoding | rows | fail | acc ≤ 1% | median rel. σ | σ = 0 | 1σ cov | 2σ cov | mean crps_log | ρ(σ, err) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| split-credit seed 0 @287 | bf16 | greedy | 400 | 0.0% | 34.5% | 0.244% | 2.2% | 22.5% | 29.0% | -0.3253 | 0.08 |
| split-credit seed 0 @287 | 4bit | greedy | 400 | 0.2% | 37.8% | 1.614% | 0.0% | 41.8% | 49.8% | -0.3163 | 0.08 |
| split-credit seed 0 @287 | bf16 | sampled | 1600 | 0.9% | 29.6% | 0.367% | 5.5% | 19.9% | 25.7% | -0.5168 | 0.08 |
| split-credit seed 0 @287 | 4bit | sampled | 1600 | 0.7% | 28.9% | 1.732% | 0.4% | 34.5% | 43.1% | -0.4117 | 0.12 |
| split-credit seed 0 @287 | 4bit (training) | sampled, steps 263–287 | 800 | 0.6% | | 1.341% | | | | | |
| E1a final | bf16 | greedy | 400 | 0.0% | – | 0.567% | 1.0% | 29.2% | 35.8% | -0.3362 | 0.06 |
| E1a final | 4bit | greedy | 400 | 0.0% | – | 3.228% | 0.0% | 48.5% | 58.2% | -0.3309 | 0.01 |
| E1a final | bf16 | sampled | 1600 | 0.2% | – | 0.773% | 2.7% | 32.6% | 39.1% | -0.3426 | 0.03 |
| E1a final | 4bit | sampled | 1600 | 0.2% | – | 3.546% | 0.1% | 49.9% | 59.5% | -0.3426 | 0.01 |
| E1a final | 4bit (training) | sampled, steps 276–300 | 800 | 0.1% | | 3.556% | | | | | |

**Share of the train/eval σ gap closed** (D65 rule, sampled medians):

| model | σ̃ bf16 sampled | σ̃ 4-bit sampled | σ̃ training | g | label | greedy g (descriptive) |
|---|---|---|---|---|---|---|
| split-credit seed 0 @287 | 0.367% | 1.732% | 1.341% | 1.20 | precision explains most of the gap | 1.11 |
| E1a final | 0.773% | 3.546% | 3.556% | 1.00 | precision explains most of the gap | 0.95 |

**Paired, greedy, 4-bit vs bf16** (same problems / prefixes):

| model | pairs | identical completion | same μ | both answered | median ln(σ_4bit / σ_bf16) | 4-bit σ wider | 4-bit σ narrower |
|---|---|---|---|---|---|---|---|
| split-credit seed 0 @287 | 400 | 1.8% | 12.8% | 399 | +2.303 | 75.4% | 3.0% |
| E1a final | 400 | 21.8% | – | 400 | +2.303 | 77.8% | 0.5% |

**Median relative σ by depth** (controls for the problem mix):

split-credit seed 0 @287:

| depth | training | bf16 greedy | 4bit greedy | bf16 sampled | 4bit sampled |
|---|---|---|---|---|---|
| 1 | 1.273% | 0.124% | 1.321% | 0.274% | 1.609% |
| 2 | 1.397% | 0.226% | 1.405% | 0.370% | 1.988% |
| 3 | 1.163% | 0.187% | 1.602% | 0.285% | 1.624% |
| 4 | 1.710% | 0.339% | 1.800% | 0.394% | 1.834% |
| 5 | – | 0.248% | 1.670% | 0.419% | 1.691% |

E1a final:

| depth | training | bf16 greedy | 4bit greedy | bf16 sampled | 4bit sampled |
|---|---|---|---|---|---|
| 1 | 2.916% | 0.404% | 3.063% | 0.531% | 3.169% |
| 2 | 3.111% | 0.627% | 2.711% | 0.949% | 3.382% |
| 3 | 3.556% | 0.515% | 3.862% | 0.715% | 3.989% |
| 4 | 4.165% | 0.491% | 3.344% | 0.582% | 3.629% |
| 5 | – | 0.659% | 3.299% | 0.860% | 3.622% |

**Within-problem spread of sampled σ** (k = 4):

| model | base | median SD of ln σ | greedy σ mean mid-rank | greedy σ below all samples |
|---|---|---|---|---|
| split-credit seed 0 @287 | bf16 | 1.01 | 0.43 | 7.5% |
| split-credit seed 0 @287 | 4bit | 0.85 | 0.50 | 4.5% |
| E1a final | bf16 | 0.94 | 0.45 | 3.5% |
| E1a final | 4bit | 0.35 | 0.45 | 2.2% |

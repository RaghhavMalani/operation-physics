# X3 pilot (D79): distillation capacity, 399 frozen-set answers (μ fixed)

**Student (a), A5 teacher: capacity likely** (bar ρ = 0.5).

| σ source | ρ(σ, err) [95% CI] | mean crps_log | 1σ / 2σ cov | median rel σ | σ = 0 | failed | μ changed | fidelity vs teacher | X2 pairwise acc. |
|---|---|---|---|---|---|---|---|---|---|
| student (a) | +0.575 [+0.499, +0.643] | -0.306 | 66.4% / 79.4% | 10.8% | 0.0% | 0 | 0 | 0.716 | 0.513 [0.484, 0.543] |
| student (b) | +0.583 [+0.509, +0.650] | -0.348 | 30.8% / 37.6% | 0.3% | 0.0% | 0 | 0 | 0.601 | 0.488 [0.460, 0.517] |
| A5 teacher σ (4-bit) | 0.664 | -0.291 | 66.4% / 76.9% | 6.6% | 0.0% | 0 | – | – | – |
| SFT stated σ (4-bit) | 0.038 | -0.322 | 12.0% / 17.0% | 0.1% | 16.6% | 0 | – | – | – |

- student (a): ρ − teacher -0.088 [-0.155, -0.024]; ρ − SFT +0.537 [+0.402, +0.669]
  - training: 500 steps (2.00 epochs), loss 0.508, 45 min
- student (b): ρ − teacher -0.080 [-0.157, -0.003]; ρ − SFT +0.545 [+0.417, +0.671]
  - training: 500 steps (2.00 epochs), loss 0.558, 43 min
- ρ(b) − ρ(a): +0.008 [-0.054, +0.072]

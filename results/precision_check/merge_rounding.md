# How much of each LoRA update survives a bf16 merge (post-hoc diagnostic for D65; weights only)

| adapter | modules | retained ‖eff‖/‖Δ‖ | rel. error ‖eff − Δ‖/‖Δ‖ | entries rounded to 0 | median |Δ| / ulp(W) | |Δ| < ½ ulp |
|---|---|---|---|---|---|---|
| split credit seed 0 @287 | 196 | 0.553 | 0.918 | 93.7% | 0.048 | 92.9% |
| split credit seed 1 @287 | 196 | 0.544 | 0.921 | 93.9% | 0.046 | 93.1% |
| split credit seed 2 @287 | 196 | 0.559 | 0.916 | 93.6% | 0.048 | 92.8% |
| uniform credit @287 | 196 | 0.439 | 0.950 | 95.9% | 0.033 | 95.2% |
| overnight RL, final (FINDINGS #5) | 196 | 0.447 | 0.948 | 95.8% | 0.033 | 95.1% |
| E1a final | 196 | 0.629 | 0.891 | 90.9% | 0.071 | 89.6% |
| E1b final | 196 | 0.623 | 0.893 | 91.3% | 0.068 | 90.0% |
| SFT adapter (on its original base) | 196 | 1.022 | 0.280 | 41.2% | 0.701 | 40.7% |

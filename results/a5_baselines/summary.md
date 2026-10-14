# A5 nested baselines (D50)

Calibration split: 2000 completions (1995 answered), 865 templates; 5-fold CV grouped by template. Strongest σ: **problem_answer_reasoning/gbt**; strongest p: **problem_answer_reasoning/gbt** (chosen on CV before any eval use).

| baseline | CV crps_log (σ) | CV 1σ cov | CV ρ(σ, err) | CV Brier (p) | CV AUROC (p) |
|---|---|---|---|---|---|
| metadata/linear | -0.2256 | 68.6% | 0.42 | 0.1976 | 0.741 |
| metadata/gbt | -0.2261 | 65.9% | 0.42 | 0.1982 | 0.735 |
| problem/linear | -0.2224 | 72.1% | 0.54 | 0.1596 | 0.835 |
| problem/gbt | -0.2211 | 70.2% | 0.59 | 0.1483 | 0.853 |
| problem_answer/linear | -0.2242 | 69.0% | 0.53 | 0.1643 | 0.825 |
| problem_answer/gbt | -0.2211 | 70.0% | 0.59 | 0.1476 | 0.853 |
| problem_answer_reasoning/linear | -0.2219 | 67.1% | 0.62 | 0.1473 | 0.858 |
| problem_answer_reasoning/gbt | -0.2188 | 67.8% | 0.67 | 0.1307 | 0.879 |

## SFT-only frozen eval, depth_extrapolation (n = 200, answered 200; answered rows only)

| σ source | mean crps_log | 1σ cov | 2σ cov | ρ(σ, err) | Brier (p) | AUROC (p) |
|---|---|---|---|---|---|---|
| stated σ | -0.3573 | 11.5% | 16.0% | 0.00 | – | – |
| metadata/linear | -0.3367 | 55.5% | 67.5% | 0.42 | 0.1952 | 0.708 |
| metadata/gbt | -0.3302 | 65.5% | 75.0% | 0.42 | 0.1832 | 0.708 |
| problem/linear | -0.3195 | 69.5% | 81.5% | 0.56 | 0.1552 | 0.829 |
| problem/gbt | -0.3243 | 69.0% | 80.0% | 0.60 | 0.1248 | 0.89 |
| problem_answer/linear | -0.3208 | 69.5% | 79.0% | 0.57 | 0.1499 | 0.842 |
| problem_answer/gbt | -0.3241 | 69.5% | 81.0% | 0.61 | 0.1295 | 0.883 |
| problem_answer_reasoning/linear | -0.3132 | 67.5% | 77.0% | 0.64 | 0.1333 | 0.889 |
| problem_answer_reasoning/gbt (strongest σ) | -0.3207 | 66.5% | 75.5% | 0.65 | 0.1267 | 0.878 |

## SFT-only frozen eval, id (n = 200, answered 199; answered rows only)

| σ source | mean crps_log | 1σ cov | 2σ cov | ρ(σ, err) | Brier (p) | AUROC (p) |
|---|---|---|---|---|---|---|
| stated σ | -0.2963 | 15.1% | 20.6% | -0.06 | – | – |
| metadata/linear | -0.2746 | 63.3% | 76.4% | 0.55 | 0.1725 | 0.805 |
| metadata/gbt | -0.2752 | 60.8% | 73.9% | 0.55 | 0.1668 | 0.824 |
| problem/linear | -0.2716 | 68.3% | 78.9% | 0.62 | 0.1268 | 0.898 |
| problem/gbt | -0.2677 | 67.8% | 79.9% | 0.68 | 0.1192 | 0.909 |
| problem_answer/linear | -0.2726 | 66.8% | 74.9% | 0.63 | 0.1278 | 0.895 |
| problem_answer/gbt | -0.2677 | 69.3% | 79.4% | 0.68 | 0.1203 | 0.909 |
| problem_answer_reasoning/linear | -0.2711 | 65.8% | 76.4% | 0.69 | 0.1158 | 0.92 |
| problem_answer_reasoning/gbt (strongest σ) | -0.2659 | 67.3% | 77.9% | 0.75 | 0.0962 | 0.939 |

## SFT-only frozen eval, pooled (n = 400, answered 399; answered rows only)

| σ source | mean crps_log | 1σ cov | 2σ cov | ρ(σ, err) | Brier (p) | AUROC (p) |
|---|---|---|---|---|---|---|
| stated σ | -0.3269 | 13.3% | 18.3% | -0.02 | – | – |
| metadata/linear | -0.3057 | 59.4% | 71.9% | 0.49 | 0.1838 | 0.763 |
| metadata/gbt | -0.3028 | 63.2% | 74.4% | 0.49 | 0.1750 | 0.779 |
| problem/linear | -0.2956 | 68.9% | 80.2% | 0.59 | 0.1410 | 0.867 |
| problem/gbt | -0.2961 | 68.4% | 79.9% | 0.65 | 0.1220 | 0.898 |
| problem_answer/linear | -0.2967 | 68.2% | 76.9% | 0.61 | 0.1389 | 0.871 |
| problem_answer/gbt | -0.2960 | 69.4% | 80.2% | 0.65 | 0.1249 | 0.894 |
| problem_answer_reasoning/linear | -0.2922 | 66.7% | 76.7% | 0.66 | 0.1245 | 0.906 |
| problem_answer_reasoning/gbt (strongest σ) | -0.2934 | 66.9% | 76.7% | 0.70 | 0.1114 | 0.913 |

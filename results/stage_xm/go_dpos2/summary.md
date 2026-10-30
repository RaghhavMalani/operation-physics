# The pilot's go rule on D-pos-v2 (D137's rule, run early per D139): Llama-3.2-3B-Instruct, seed 0

**NO-GO** (K1 True, C2 True, C3 False; C1 True). 2040 frozen A6c + A6b problems.

- A6c slope (answer σ, OLS within pairs): +0.785 [+0.595, +0.974] (90% CI), one-sided p 0.0, 505 pairs (K1: p < 0.05; C2: slope ≥ 0.5)
- A6b slope on computation-correct pairs: +0.567 [-3.552, +843.415] (90% CI), 4 pairs (C3: CI inside ±0.3; fewer than 3 pairs = not met)

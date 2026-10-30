# Frozen J set: R5-6 part J (DECISIONS.md D147)

Built 2026-10-30 (~14:45 IST) with `physcal/env/pairs_j.py` and `configs/pairs.yaml`'s `j_allocation_*` and
`j_invariance_*` sections, after D147 was committed (8405f0f), and before any model saw it.

- **Jacobians outside the training range.** Every pair is on a chain that occurs in the R4-1 training problems, and
  at least one given's log-sensitivity d ln f / d ln x lies outside that chain's training range by ≥ 0.1.
  - The support is `data/jacobian_support/r4_1.json` (SHA-256 `eda3c3b2…`, stored in every problem's construction).
  - Every displayed given and every computed value lies inside that quantity's training range.
- **`a6c_pairs.jsonl`: sensitivity pairs** (A6c's allocation construction on J worlds). Pair seeds from 2,400,000,000;
  408 pairs; SHA-256 `42f2b3d1…`.
- **`a6b_pairs.jsonl`: invariance pairs** (a J world and its dimensionally scaled copy; identical Jacobian, target
  ×2–8). Pair seeds from 2,500,000,000; 408 pairs; SHA-256 `395f04dc…`.
- **Cells:** depths 1–4 × 3 level cells × 34 pairs, all 24 full.
  - Dynamics is thin at depth 1 in the sensitivity set (0–3 pairs per cell): most of its depth-1 chains are power
    laws, whose Jacobian is constant.
  - Per-cell counts, distances and checksums are in each `*_manifest.json`.
- **Replay:** all 1,632 problems regenerate byte-identically (checked before the freeze).

**Do not evaluate any model on these files, and do not compute any statistic from model outputs on them,** except
D147's six evals (D-pos-v2 and B-v2, seeds 0–2) and D147's one-time analysis. `physcal.eval.run_eval.load_frozen_split`
refuses this directory unless `PHYSCAL_CONFIRMATORY=1` is set.

**Generator-only distances (no model; D147 §10), sensitivity / invariance:**

| | sensitivity | invariance |
|---|---|---|
| exceedance outside the chain range (median [IQR]) | 0.38 [0.17, 0.77] | 0.34 [0.18, 0.74] |
| L∞ to the nearest training Jacobian of the chain | 0.41 [0.19, 0.84] | 0.38 [0.18, 0.78] |
| training problems of the chain | 3 [1, 5] | 3 [1, 5] |
| ln(σ_δ / σ_δ clamped into the range), hi member | 0.018 [−0.111, 0.231] | 0.060 [−0.097, 0.218] |
| strict (also outside the (family, given, target) range) | 27% | 35% |
| rank flip (hi given never the largest in training) | 11% | – |
| distinct chains | 238 | 222 |

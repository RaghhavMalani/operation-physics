# Frozen confirmatory set (R4-5; DECISIONS.md D112 / D113)

Built 2026-10-25 from `configs/pairs.yaml`'s three constructions, unchanged, with fresh pair-seed offsets
(A6 2,100,000,000; A6b 2,200,000,000; A6c 2,300,000,000), the same 15 cells × 34 pairs as the development sets
(`../a6{,b,c}_pairs.jsonl`). Checksums are in each `*_manifest.json`.

**Do not evaluate any model on these files, and do not compute any statistic from them,** until a 3-seed confirmatory
run has its own pre-registered entry in DECISIONS.md. `physcal.eval.run_eval.load_frozen_split` refuses this directory
unless `PHYSCAL_CONFIRMATORY=1` is set.

From 2026-10-25 on, A6 / A6b / A6c (the parent directory) are development sets.

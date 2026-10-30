# Report: 2026-10-30 afternoon/evening: R5-6 part J (D147–D149)

Written on 2026-10-30 at ~19:35 IST, as soon as the chain ended.
- **Sources:** `results/queue_r5_6/chain.log`, the stage logs and the result files. The working log is in
  `docs/lab_notebook.md`.
- **The previous report** (29/30 night) is in git history: `git show fef5b15:MORNING.md`.

**Short version.**
1. **Pre-registered, then built.** D147 was pushed (`8405f0f`) before any J set, completion or statistic existed.
   The freeze (D148) and the Llama caveat in the outline are done too.
2. **Built, frozen and CPU-tested (D149).**
   - **The frozen J set:** 816 sensitivity + 816 invariance problems; all 24 cells full; every problem replays.
   - **X1** reproduces exactly.
   - **The analysis** reproduces R4-3's numbers on the development files.
   - **Tests:** the full suite, 728 passed, 0 failed.
3. **The chain ran 14:48 → 19:26 IST, 14.5 h before the 10:00 deadline.**
   - Six evals, each on its first attempt; 0 OOM backoffs; every R2-1 fingerprint check passed.
   - 5 temperature pauses of 30 s, 150 s in all (0.9%).
   - **Every commit was pushed:** `main` = `origin/main` when the chain ended.
4. **The result: P_J is CONFIRMED (FINDINGS #39).** D-pos-v2's answer σ propagates through Jacobians outside its
   training range:
   - J1: slope **+0.985 [+0.868, +1.103]**, seeds 0.94 / 0.97 / 1.05;
   - C2: ≥ 0.5;
   - C3: J-invariance **+0.026 [−0.055, +0.119]**, inside ±0.3.
5. **Explicit beats implicit on the slope:** J2 +0.530 [+0.377, +0.676], Holm-rejected. B-v2 (+0.455) fails C2 and C3.
6. **No score gain:** D-pos-v2 vs RSS +0.0011 (p = 0.072) and vs X1 +0.0000, neither rejected. This is #32's pattern.
7. **The caveat to report with it:** a propagator clamped to the training range would already reach +0.831.
   - So the set shows "propagates, not a lookup". It only weakly shows "uses the out-of-range part".
   - Descriptively, D-pos-v2 is above the clamped one where clamping matters: rank-flip pairs +1.12 vs +0.36 (43 pairs).
8. **No decision is needed from you for the experiment.** Under D148, R5-6 was the last one. What's next is the
   write-up.

## 1. What was built (D147 → D149)

- **The training support** (`data/jacobian_support/r4_1.json`, SHA-256 `eda3c3b2…`): from the 3,000 R4-1 problems
  both arms trained on, per chain, the log-sensitivity of every given.
- **A J pair:**
  - its chain occurs in training;
  - one given's sensitivity is ≥ 0.1 outside that chain's training range;
  - every shown and computed value lies inside its training range.
- **Sensitivity pairs:** A6c's allocation on J worlds.
- **Invariance pairs:** new, by dimensional scaling. The world and a copy scaled along a direction that fixes g have
  identical Jacobians and a target ×2–8. A6b's power-law chains could not be used: their Jacobians never change.
- **How far outside** (median [IQR], sensitivity): exceedance 0.38 [0.17, 0.77]; nearest trained Jacobian (L∞)
  0.41; the chain's training problems 3 [1, 5].
  - **Thin cell:** dynamics at depth 1 is nearly absent from the sensitivity set; its chains are mostly power laws.

## 2. The evals (operational)

| eval | time (IST) | generation | answered (inv / sens) | parse fail | pauses |
|---|---|---|---|---|---|
| D-pos-v2 s0 | 14:49 → 15:49 | 3,534 s | 814 / 816 | 0.2% / 0.0% | 3 × 30 s |
| B-v2 s0 | 15:49 → 16:22 | 1,909 s | 816 / 815 | 0.0% / 0.1% | 1 |
| D-pos-v2 s1 | 16:22 → 17:20 | 3,465 s | 815 / 815 | 0.1% / 0.0% | 0 |
| B-v2 s1 | 17:21 → 17:54 | 1,948 s | 816 / 816 | 0 | 1 |
| D-pos-v2 s2 | 17:54 → 18:53 | 3,453 s | 813 / 809 | 0.4% / 0.5% | 0 |
| B-v2 s2 | 18:53 → 19:25 | 1,892 s | 816 / 816 | 0 | 0 |

- **Every eval:** truncation ≤ 0.1%, 0 OOM backoffs, 102 / 102 batches in one session.
- **Memory:** the device-wide peak was 5.2–6.6 GiB.
- **The analysis:** 46 s, CPU, 1 thread.

## 3. The result (`results/r5_6/summary.md`; FINDINGS #39)

| test (Holm over J1–J6) | seed mean [90% CI] | p | Holm |
|---|---|---|---|
| **J1** D-pos-v2 (a) J-sensitivity slope > 0 (primary) | +0.985 [+0.868, +1.103] | < 1e-4 | rejected |
| **J2** explicit − implicit slope | +0.530 [+0.377, +0.676] | < 1e-4 | rejected |
| J3 D-pos-v2 − RSS (expected crps_log) | +0.0011 [−0.0002, +0.0020] | 0.072 | – |
| J4 D-pos-v2 − X1 | +0.0000 [−0.0013, +0.0008] | 0.44 | – |
| J5 B-v2 − RSS | −0.0005 [−0.0008, −0.0002] | 0.997 | – |
| J6 B-v2 − X1 | −0.0016 [−0.0019, −0.0013] | 1.0 | – |

- **The line sources (b) and (c):** both slopes are ≈ 1.0, and their invariance is inside ±0.3 on all 408 pairs.
- **The limits** (FINDINGS #39 states them at the result's prominence):
  1. **The clamped comparator** (sensitivities clipped into the training range) reaches +0.831 [+0.737, +0.919]. No
     test compares D-pos-v2 with it; that comparison was not pre-registered.
     - D-pos-v2 is above it descriptively in the subgroups where clamping matters: exceedance above the median +0.835 vs
       +0.623; strict +0.895 vs +0.710; rank flip +1.123 vs +0.357.
  2. **C3 rests on 72–86 computation-correct invariance pairs per seed,** almost all at depths 1–2.
  3. **D-pos-v2's S lines** are less accurate on the out-of-range terms: 58–64% within 10%, against 69–75% in range.
     Its answer σ still tracks σ\* at slope ≈ 1.
- **Context (#32, depths 1–4, in-range):** 0.855. The J slope is not lower.

## 4. State, and what's next

- **The chain:** it exited; the lock is free; no GPU process.
- **The checks:** the integrity check on the 122 files written since 14:30 found 0 problems; free space 47.17 GiB.
- **Updated:** `docs/decision_timeline.md` (D147–D149 → #39), `docs/paper_outline.md` (status, two claims to avoid,
  the ledger), PROGRESS.md, `docs/lab_notebook.md`.
- **D148 is now in force:** no new experiments for this paper. The next step is the write-up (Phase 15).
  - **Report with it:** R5-6 alongside the clamped comparator, and the depth-1–2 basis of C3.

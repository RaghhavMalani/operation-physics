# Report: 2026-10-20/21 night: X4 with direct + perturbed (GPU), X2 robustness, R3 amendments, E1c/E1d and A6c built

The previous report (2026-10-20 day) is in git history: `git show 53585b2:MORNING.md`.

**Short version.**
1. **The GPU finished at 03:57 IST** (deadline 11:30).
   - **All five X4 arms** ran on all 4,116 X2 answers, and the analysis is committed.
   - **No OOM, no BLOCKED.** There was one 10-minute temperature pause.
2. **X4 result (FINDINGS #21): self-verification fails within problems.** That is D89's outcome 4, consistent with
   X2.
   - **`sft`:** 0.524 [0.498, 0.550], p = 0.033. It is not Holm-supported under D91's five-arm family, nor under
     D89's original three-arm one.
   - **The model says "yes" to 99.9% of its own answers.** It recomputes only ~13 tokens (median), so direct =
     recompute (−0.002).
   - **μ × 1.25 lowers P(yes)** on 98.6% of the correct answers, but uniformly, so the ranking does not move. It
     checks "does the answer match the steps", not "are the steps right".
3. **X2 robustness (FINDINGS #22, post-hoc):** agreement with the problem's other 7 samples ranks right above wrong at
   **0.643** (0.861 on clearly separated pairs).
   - That is the strongest within-problem signal so far: +0.096 over the probe.
   - **Length alone:** 0.535, about the probe's level.
   - **X2's CIs** were already problem-level (recomputed = published).
4. **CPU builds:**
   - R3-1 to R3-10 recorded (D93);
   - E1c / E1d pre-registered and built, **not launched** (D94);
   - A6c frozen (510 pairs) plus the noise heuristics (D95).
   - **Tests:** 508 passed, 0 failed, 0 skipped.
5. **To review (§5):**
   - D94: E1c's reading of "normalized across the batch"; E1d's learning rate 5e-5.
   - D95: min ratio 1.5; identical displayed values.
   - R3-3 said RL's learning rate is ~40× below SFT's; it is 20×.

---

## 1. What ran (IST)

| time | step | result | commit |
|---|---|---|---|
| 23:22 | D91 (X4 amendment) committed **before any X4 data**; main stage launched | frozen sft-only queue copy (D96) | `4f016c6`, `3599ebb` |
| 23:22–00:09 | GPU: X4 `sft` (recompute) | 4,116 / 4,116; 46 min; peak 6.3 GiB of a 6.34 cap; paused 23:43–23:53 (88 °C) | `3473157` |
| 23:32 | direct + perturbed built and tested; the chain armed (waits on the PID) | 28 X4 tests; 4,115 / 4,116 answers pass the perturbation check | `c19f890`, `29c8030` |
| 00:09–00:22 | GPU: X4 `direct` | 4,116; 0 generated tokens | `01fbd97` |
| 00:22–00:52 | GPU: X4 `perturbed` | 4,115 (1 μ = 0 excluded, as D91 says) | `757f514` |
| 00:52–01:26 | GPU: X4 `shuffled` | 4,116 | `6823bfe` |
| 01:26–03:53 | GPU: X4 `random_init` | 4,116 (always 512 tokens) | `2007bb9` |
| 03:53–03:57 | CPU: the X4 analysis (D89 + D91) | `results/x4/summary.md` | `8210778` |
| night | D92 X2 robustness: defined, built, run; refits under X2's thread settings | every X2 method reproduces exactly | `5b868f3` … `ef87525` |
| night | D93 (R3-1…R3-10), D94 (E1c/E1d), D95 (A6c + heuristics), D96 (tonight's decisions) | built and tested, none launched | `1faf388`, `43cc2af`, `5edf1eb`, `4fd3c77` |
| 04:00 | full test suite (GPU visible) | **508 passed**, 0 failed, 0 skipped | – |

## 2. X4: self-verification on the X2 pairs (D89 + D91; FINDINGS #21)

| arm (516 problems, 5,390 pairs; CIs: bootstrap over problems) | pairwise accuracy | p (> 0.5) | Holm (5 arms) | AUROC across problems | extra tokens / answer |
|---|---|---|---|---|---|
| `sft`: recompute (primary) | 0.524 [0.498, 0.550] | 0.033 | not supported | 0.469 [0.442, 0.496] | 93.4 (18.4 generated) |
| `direct`: no recomputation | 0.526 [0.500, 0.553] | 0.025 | not supported | 0.463 [0.436, 0.490] | 75.0 (0 generated) |
| `perturbed`: final μ × 1.25 | 0.524 [0.498, 0.551] | 0.038 | not supported | 0.450 [0.423, 0.477] | 94.5 |
| `shuffled`: another problem's text | 0.513 [0.488, 0.539] | 0.153 | not supported | 0.463 [0.435, 0.491] | 99.9 |
| `random_init` | 0.489 [0.465, 0.514] | 0.81 | not supported | 0.517 [0.495, 0.539] | 587 |
| X2 probe (reference) | 0.547 [0.520, 0.574] | – | – | 0.727 [0.708, 0.745] | – |

- **Paired differences:**
  - `sft − X2 probe` −0.023 [−0.060, +0.013];
  - `sft − random_init` +0.035 [−0.001, +0.071];
  - `sft − shuffled` +0.011 [−0.021, +0.042];
  - `sft − perturbed` −0.000 [−0.028, +0.028];
  - `sft − direct` −0.002 [−0.008, +0.004];
  - `direct − random_init` +0.037 [+0.001, +0.073]. Its CI is above 0, but `direct` is not Holm-supported, so
    "direct succeeds" is False.
- **The D91 flags are all False:** final-value-sensitive, direct succeeds, recomputation helps.
- **What the model does:**
  - **"Verdict: yes"** on 99.9% of correct and 99.8% of incorrect answers.
  - **Mean P(yes):** 0.80 on correct answers and 0.82 on incorrect ones.
  - **Perturbing μ** lowers P(yes) by ~0.08 on both: detection is 0.986 [0.977, 0.994] on the correct answers.
  - **Another problem's text** lowers it to ~0.6.
- **Limits:**
  - one model and one prompt, greedy;
  - direct mode keeps the prompt's "You may redo the calculation";
  - the random-init model puts no mass on yes/no, so it is a weak control;
  - R3-1's yes/no order variant and the deterministic checker ceiling are still to do.

## 3. X2 robustness (D92, post-hoc; FINDINGS #22)

| method | X2 pairwise acc. | separated pairs (right ≤ 0.25%, wrong ≥ 10%; 158 problems / 719 pairs) |
|---|---|---|
| **self-consistency** (share of the other 7 samples within 1%) | **0.643 [0.616, 0.671]** | **0.861 [0.820, 0.901]** |
| hidden-state probe (SFT) | 0.547 [0.520, 0.574] | 0.635 [0.570, 0.697] |
| hidden-state probe (random init) | 0.530 [0.503, 0.557] | 0.584 [0.523, 0.645] |
| length only (shorter = more confident) | 0.535 [0.511, 0.559] | 0.607 [0.549, 0.663] |
| X4 `sft` | 0.524 [0.498, 0.550] | 0.544 [0.485, 0.603] |
| SFT stated σ | 0.499 [0.474, 0.524] | 0.501 [0.442, 0.558] |

- **Length:**
  - **Pooled,** correct answers are 17 tokens shorter (186 vs 203).
  - **Within problems,** only 1.8 [1.0, 2.7] tokens.
  - **Against the probe,** length alone is −0.012 [−0.045, +0.021], i.e. indistinguishable.
- **The pooled pair-weighted cluster-bootstrap version** gives the same picture (`results/x2/robustness/summary.md`).
- **X2's published CIs** already resampled problems; the recomputed ones match.

## 4. Built tonight, not run

- **E1c / E1d (D94, the R3-3 pilots):** each is E1a with one change.
  - **E1c:** `scale_rewards: batch`. The baseline stays the prompt's own mean; the std is taken over the step's 32
    completions.
  - **E1d:** learning rate 5e-5, against 5e-6.
  - **The test:** ρ(σ, |error|) vs E1a's 0.013 on the corrected eval. Paired bootstrap over problems, Holm over the two.
    The winner sets Phase 12's RL configuration; if neither wins, E1's settings stay.
  - **Launch (about 2.7 h each):**
    - `uv run python -m physcal.train.e1 train --config grpo_e1c` (then `grpo_e1d`);
    - `uv run python -m physcal.train.e1 eval --run results/train/e1c-sigma-only-free-seed0 --out-name final_4bit`
      (and for step 150);
    - then `uv run python -c "from physcal.train.e1_compare import compare_pilots as c; c()"`.
  - **No queue script is prepared yet.**
- **A6c (D95):** `data/eval_sets/2.0.0+bdbb2279/a6c_pairs.jsonl` (sha256 `b90455b0…`).
  - **The pairs:** 510 noise-allocation pairs, 34 in each of the 15 cells (depth 1–5 × (1, 5)%, (2, 5)%, (1, 2)%).
    Both members show identical values and the same multiset of "± p%". Only the given that carries the high level
    moves.
  - **σ\* ratios:** MC σ\*_hi / σ\*_lo has a median of 1.6–3.6 per cell, minimum 1.35.
  - **Replay** is exact, and A6/A6b regenerate unchanged.
- **The noise heuristics on the frozen sets** (`results/noise_heuristics/summary.md`):
  - every heuristic gives Δ = 0 within every pair, so it fails A6 and A6c and passes A6b;
  - per problem, RSS over the givens has a median |ln(σ_h / σ\*)| of 0.55 on A6, 0.28 on A6b and 0.50 on A6c.
  - **The comparison with X1 on A6c** is not done yet (D95).

## 5. Decisions to review

1. **D94 [REVIEW], E1c:** "normalized across the whole batch". I kept the per-prompt mean as the baseline and
   scaled by the batch std (TRL `batch`). Centering at the batch mean as well would mix μ error back into σ's
   advantage. It can change before launch.
2. **D94 [REVIEW], E1d:** the learning rate is 5e-5 (10×). The alternatives are 2.5e-5 and 1e-4 (SFT's).
3. **D95:**
   - **min ratio 1.5,** not A6's 2. With (1%, 2%) no allocation reaches 2.
   - **Identical displayed values:** member hi's true givens are backed out of them, so the swap is the only visible
     change.
   - **Distractors** are shown at the low level.
4. **D93, R3-6:** D83's τ = 0.1% becomes 1% (sensitivity analyses at 0.1% and 0.5%). It needs a `reward_v12.yaml`
   change. Not done: verifier, and p is deferred.
5. **D96: the in-stage temperature guard** is new; earlier queues checked only before a stage. It fired once.

## 6. BLOCKED / problems

- **Nothing is BLOCKED.**
- **The X2 refits were refused under 1 thread** (cross-fitted probe 0.5390 vs 0.5396). The reproduction check did
  its job, and the rerun under X2's thread settings reproduced exactly (D96).
- **A heuristic float-order bug** (RSS "supported" with slope +0.000) was caught and fixed before any result was
  committed.

## 7. Recommendation

- **Self-consistency is the confidence source to take seriously.**
  - **What it does:** 0.643 within problems, without any training.
  - **For R3-4 E:** X4 failed, so verification-derived p is not a candidate teacher. Agreement across k samples is.
    Its cost is k samples at inference, and a k-sample consensus σ is close to DAR's rollout spread (Phase 13 source
    2).
  - **Next step:** I suggest a short D-entry making it the p baseline and teacher candidate before Phase 12 stage E.
- **Tomorrow night (R3 schedule):** E1c + E1d after you answer §5.1–5.2. I can write the queue script by day.
- **Daytime CPU:** X1 on A6c; Phase 12 stage A (scoring audit, the oracle-centred diagnostic).
- **Optional, cheap X4 follow-ups (R3-1 "later"):** the yes/no order variant, and `perturbed` in direct mode
  (about 13 GPU minutes).

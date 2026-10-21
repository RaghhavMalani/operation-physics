# Report: 2026-10-21 day: the E1c / E1d pilots (GPU), X1 on A6c and Phase 12 stage A (CPU)

The previous report (2026-10-20/21 night) is in git history: `git show bf16bc4:MORNING.md`.

**Short version.**
1. **The GPU finished at 10:43 IST** (deadline 14:00). Both pilots ran 300 steps and all four evals completed. No
   OOM, no temperature pause, nothing BLOCKED.
2. **E1d works (FINDINGS #25).** At learning rate 5e-5, ρ(σ, |error|) is **0.586** against E1a's 0.013:
   Δρ +0.573 [+0.465, +0.680], Holm-supported.
   - **E1c** (batch-std scaling) is null: ρ 0.002.
   - **D94's rule:** 5e-5 becomes Phase 12's RL setting (D101).
   - **But:** E1d's σ ranks errors only between problems, at X3's supervised level (0.58). It is overconfident (1σ
     coverage 0.30) and still below the A5 text baseline (ρ 0.70; crps_log −0.023).
3. **X1 on A6c (FINDINGS #23):** the exact refit reproduces D73 exactly (max difference 0.0). On A6c X1's slope is
   −0.002, as the construction predicts. No surface reader propagates noise allocation. X1's A6 "partial
   propagation" came from the displayed magnitudes.
4. **Phase 12 stage A (FINDINGS #24):** crps_log is maximized at σ\* (c = 1.000 [0.966, 1.017]). But on one realized
   truth, σ\* beats a 2×-off σ on only 56–77% of problems. Against the known predictive distribution it does on
   ~100%, with 5–23× less variance.
5. **Decisions:** your four items are recorded (D97, D98). D99 / D100 were defined before computing; D101 applies
   D94's rule.
6. **To review (§5):** D100's reading of "scoring audit"; D101's caveats on using 5e-5 for whole completions.
7. **Tests:** 519 passed, 0 failed, 0 skipped (GPU visible, after the queue). That is 508 + 11 new.

---

## 1. What ran (IST)

| time | step | result | commit |
|---|---|---|---|
| 06:02 | queue written, dry-run, launched (`scripts/queue_e1_pilots.sh`, deadline 14:00) | – | `5c20ca3` |
| 06:03 | D97 (your approvals + the queue) and D98 (self-consistency = p baseline / teacher candidate) | – | `0a0f718` |
| 06:02–08:21 | GPU: E1c train | 300 steps, 2.29 h; `scale_rewards: batch` confirmed in `summary.json` | `221e231` |
| 08:21–08:27 | GPU: E1c evals (`final_4bit`, `step150_4bit`) | 400 / 400 each | `f67b4fc`, `3ba8432` |
| 08:27–10:35 | GPU: E1d train (started with 5 h 33 min left; your rule needs ≥ 3 h) | 300 steps, 2.10 h | `d505321` |
| 10:35–10:43 | GPU: E1d evals | 400 / 400 each | `7430a61`, `f359561` |
| 10:43 | CPU: compare_pilots (D94) | E1d supported, E1c not | `c1fa6f7` |
| 06:07 | D99 (X1 on A6c) and D100 (stage A), before computing | – | `7792502` |
| 06:07–06:13 | CPU, 1 thread, nice 19: X1 refit + A6c | reproduction exact; A6c slope −0.002 | `144eab7` |
| 06:14–06:15 | CPU: stage A (16 s) | FINDINGS #24 | `5cda743`, `19e7a33` |
| 10:44–10:45 | D101, FINDINGS #25 | – | `b24a782`, `6ed5fe6` |

## 2. E1c / E1d (D94; FINDINGS #25)

| run (300 steps, 1 seed) | ρ final | Δρ vs E1a [95% CI] | Holm | ρ step 150 | 1σ cov. | median rel. σ | crps_log | vs A5 σ (crps_log) |
|---|---|---|---|---|---|---|---|---|
| E1a (reference) | 0.013 | – | – | −0.001 | 0.485 | 3.2% | −0.3309 | – |
| E1c: batch std | 0.002 | −0.011 [−0.092, +0.069] | no (p = 0.62) | 0.052 | 0.520 | 4.3% | −0.3300 | −0.025 [−0.033, −0.017] |
| **E1d: lr 5e-5** | **0.586** | **+0.573 [+0.465, +0.680]** | **yes (p < 1e-4)** | 0.269 | 0.298 | 0.95% | −0.3286 | −0.023 [−0.032, −0.016] |

- **Sanity:**
  - 400 / 400 rows in each eval; no parse failure, unit mismatch or μ change.
  - Neither run is collapsed (top suffix 1.8% / 4.8%).
  - The trainer used the intended settings.
- **What it means:** E1's flat σ (R3-3's question) was largely the learning rate. Batch-std scaling did nothing.
- **What it does not mean:**
  - **Between problems only:** one answer per problem, so this is the X3 kind of ranking.
  - **Below the baseline:** E1d does not beat the A5 text baseline.
  - **Not calibrated:** coverage 0.30.
  - **Not converged:** ρ is still rising at step 300.
- **Memory:** both runs repeat E1a's pattern. After the step-250 save, reserved memory reaches 7.47 GiB and steps go
  from ~23 s to ~50 s. I kept E1a's 7.5 GiB cap for parity (D97); it cost ~20 min per run.

## 3. X1 on A6c (D99; FINDINGS #23)

| σ source | A6 slope | A6b ≡ 0 | A6c slope (supported?) |
|---|---|---|---|
| X1 primary (text + magnitudes) | +0.063 [+0.031, +0.097] | yes | −0.002 [−0.003, −0.000] (no) |
| X1 noise / noise_text | +0.000 / +0.002 | yes | +0.000 / −0.002 (no) |
| heuristics max / mean / RSS | +0.000 | yes | +0.000 (no) |
| σ̂ = σ\* | +1.000 | yes | +1.000 (yes) |

- **The refit:** 40 published numbers per feature set reproduce with max difference 0.0.
- **Within A6c pairs:** X1's σ̂ is bit-identical in 57.5% of pairs. The rest differ only through punctuation-carrying
  tokens.
- **The bar for Phase 12:** any positive A6c slope beats every surface baseline built so far.

## 4. Phase 12 stage A (D100; FINDINGS #24)

- **The optimum:** on 10,005 gate problems, the mean realized crps_log over σ = c·σ\* peaks at
  **c = 1.000 [0.966, 1.017]**. All 15 cells' CIs contain 1.
- **The curve is flat and asymmetric:** c = 0.5 costs 8% of the score and c = 2 costs 17%. That is the Gaussian CRPS
  formula.
- **The resolution per problem** (A6 + A6b + A6c, 3,060 problems):

  | σ\* vs | σ\* better, realized | σ\* better, expected | variance ratio realized / expected | problems for 2 SE (realized / expected) |
  |---|---|---|---|---|
  | 0.5·σ\* | 56.5% | 100% | 22.7 | 53 / 3 |
  | 2·σ\* | 76.5% | 100% | 10.0 | 28 / 3 |
  | RSS heuristic | 58.0% | 91.3% | 6.4 | 79 / 12 |
  | X1 primary | 60.7% | 97.0% | 7.6 | 101 / 13 |

- **μ error swamps σ:** a 5% μ error costs 6× the σ\*-vs-RSS gap. That is why R3-6 centres at f(x_meas).
- **The verifier's details:**
  - no clip and no floor binds;
  - the diagnostic reproduces `reward()`'s `score_final` exactly (300 / 300).
- **The oracle-centred baselines:** X1 primary has median |ln(σ / σ\*)| 0.33 and RSS 0.40. Both have an
  across-problem vol beta of ~0.5–0.6, with within-pair slopes of 0.
- **New, reusable code:**
  - `physcal/eval/oracle_centred.py` (the diagnostic);
  - `physcal/eval/stage_a.py`, `configs/stage_a.yaml`;
  - 11 new tests across `tests/test_oracle_centred.py`, `tests/test_stage_a.py` and `tests/test_x1_a6c.py`.

## 5. Decisions to review

1. **D100 [REVIEW]: my reading of "scoring audit."**
   - **R3-4 names it without a definition.** I read it as: does the scoring Phase 12 will use reward σ propagation
     on noisy problems?
   - **The four questions:** the optimum, the per-problem resolution, the ranking, and the clip / floor details.
   - **If you meant something else** (e.g. an audit of the verifier's code paths), it still needs doing.
2. **D101 [REVIEW]: E1d's 5e-5 for Phase 12.**
   - **Validated only on 24-token σ suffixes with μ fixed.** On whole completions it may destabilize μ.
   - **The options:** run stage C's pilot at 5e-5 with a μ-accuracy guard (the rule as written), or add a 1e-5 arm.
   - **Also:** the step budget (still improving at 300), and a calibration fix for coverage 0.30.
3. **FINDINGS #24 → Phase 12's pre-registration (my suggestion):**
   - **The metric:** report σ propagation primarily with the oracle-centred expected score, plus A6 / A6b / A6c.
   - **The reward:** consider the expected score against the generator's predictive distribution as stage C's σ
     reward. That would be a verifier change; I made none.
4. **D97: the 7.5 GiB cap was kept for E1a parity,** at a cost of ~20 min per run. For Phase 12, a 7.1 cap
   (total − host − 0.5) probably avoids the post-checkpoint spill. It would need a check that it does not OOM.

## 6. BLOCKED / problems

- **Nothing is BLOCKED.** No stage failed or was skipped.

## 7. Recommendation

- **Phase 12 RL now has a learning rate that moves σ.** The next question is whether E1d's σ ranks answers within a
  problem.
  - **The experiment:** the X2 pairs with an E1d-style σ-only adapter (X2's 4,116 answers as prefixes).
  - **Cost:** about 30 GPU-min, inference only (400 prefixes took ~3 min).
  - **Why first:** it is the cheapest test of whether RL learned something X3 did not.
- **Then, per the R3 schedule (Thu 22):** the Phase 12 SFT data check and supervised σ\* (3 seeds), and the Phase 12
  pre-registration with D101's caveats decided.
- **Before any σ-RL night:** settle §5.3. FINDINGS #24 says a realized-score σ reward is a noisy teacher.

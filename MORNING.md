# Report: 2026-10-25/26 night: Round 4 (R4-1 to R4-5), D-pos-v2 / B-v2 and MC-A

The previous report (2026-10-22/23 night and the disk-full morning) is in git history: `git show 34678b5:MORNING.md`.

**Short version.**
1. **R4-3's verdict is GO.** D-pos-v2 meets the propagation criterion on all three graded σ values, (a), (b) and (c)
   (FINDINGS #30).
   - **The numbers:** an A6c slope of +0.53 [90% CI +0.33, +0.71] for the answer's σ; A6b equivalent to 0; an A6
     slope of +0.88.
   - **A first for Phase 12:** it is the first model labelled "full propagation" (D59's verdict).
   - **The pre-registered next step:** 3 seeds of both arms on the frozen R4-5 set. It needs its own entry, so it was
     not started (§6.1).
2. **But it only ties RSS on the expected score** (R6: −0.0001 [−0.0008, +0.0005]) and stays below X1 (R7).
   - **Why:** its σ is nearly calibrated in the median (median abs ln(σ / σ\*) 0.07; 1σ coverage 67%). But 80% of its
     regret sits in its worst 5% of problems: gross misses on deep, value-dependent chains.
   - **By depth:** it beats RSS at depths 1–2 and loses at 4–5. At the held-out depth 5 the A6c slope falls to +0.06.
3. **B-v2 (implicit, σ\* supervised on the same mixed-level data) learns where the noise is, not how the values
   amplify it.** A6c +0.47 (stage B: +0.04), A6 −0.29. It scores below RSS.
4. **MC-A does not beat RSS on A6c: the stopping rule fires, so k is not increased** (FINDINGS #31).
   - **The scores:** robust −0.0446 vs RSS −0.0383. The exact solver on the same 8 perturbations reaches −0.0358
     (σ\* −0.0351).
   - **Why:** the model's arithmetic errors swamp the input noise. The step-name filter removes 0.1% of runs.
   - **Side result:** the raw spread is the best correctness signal (AUROC 0.745) and the best end-to-end σ.
5. **One bug found and fixed (D116).** D-pos-v2's first eval OOM'd because `run_eval`'s backoff retried inside its
   `except` block, keeping every failed attempt's memory alive. It is fixed and tested, and the eval was rerun after
   B-v2's (with batch 16).
6. **The GPU ran 17:32–05:54 IST** (deadline 09:00), with three 10-min temperature pauses. Every planned stage ran.
   **Tests: 602 passed, 0 failed, 1 skipped** (GPU hidden), plus 1 new eval test after D116.
7. **Decisions:** D112–D117. The frozen confirmatory set (R4-5) is built and guarded. A6 / A6b / A6c are now
   development sets.

---

## 1. What ran (IST)

| time | step | result | commit |
|---|---|---|---|
| 17:05–17:10 | D78 GPU check rerun; trained-adapter load check | PASS; 32 / 32 exact | `87a5933` |
| 17:10–17:20 | D112 (R4-1 to R4-9 recorded), D113 (R4-1 to R4-5 pre-registered) | before anything was built | `87a5933`, `e504498` |
| 17:20–17:31 | build (D114): mixed-level generator option, R4-1 data, stage R4, confirmatory set, tests | 2 deviations | `a4b0a2a` |
| 17:32 | R4 chain launched (`queue_r4.sh`, PID 22552); cap 5.9 GiB (D115) | – | `6283c8a` |
| 17:32–18:55 | GPU: D-pos-v2 SFT, 750 / 750 (pause 17:57) | loss 0.116 | (queue) |
| 17:35–18:00 | CPU: D113's analysis and MC-A built and smoke-tested; full test suite | 602 passed | `435081a`, `f1fd50b` |
| 18:55–19:01 | GPU: D-pos-v2 eval: **BLOCKED** (OOM in the backoff) | fixed (D116) | `d1bbe33`, `dc54969` |
| 19:02–20:22 | GPU: B-v2 SFT, 750 / 750 (pause 19:24) | loss 0.119 | `5efa8a8` |
| 20:22–21:27 | GPU: B-v2 eval, 3,060 / 3,060, batch 16, no OOM | – | `5483e37` |
| 21:27–23:18 | GPU: D-pos-v2 eval (rerun), 3,060 / 3,060, no OOM | – | `f151c66` |
| 23:19 | CPU: D113's analysis | GO | `5d2e465` |
| 23:18–01:16 | GPU: MC-A A6c, 8,160 runs | stopping rule fires | `a2a71b3` |
| 01:16–03:41 | GPU: MC-A A6, 8,160 runs (pause 02:19) | – | `481a0f9` |
| 03:41–05:54 | GPU: MC-A A6b, 8,160 runs | – | `9ad4956` |
| ~05:55 | FINDINGS #30 / #31, D117, this report | – | `f48cb1b` + this commit |

## 2. D-pos-v2 and B-v2 (D113; FINDINGS #30)

- **The data (R4-1):**
  - **The problems:** D80's 3,000 seeds, each given's level drawn independently from 0 / 1 / 2 / 5%. 2,067
    problems mix noisy levels.
  - **D-pos-v2's added lines:** a derivation per noisy given, e.g. `S_t1 = -t1/t2 = -11.34/24.51 = -0.463`, with
    power-law terms as exponents (`S_v1 = -2`).
  - **Then** the line with each given's own level, and σ = X·|μ|.

| σ source (oracle-centred, 3,060) | A6c slope [90% CI] | A6b | A6 slope | expected crps_log | median abs ln(σ/σ\*) | 1σ cov. |
|---|---|---|---|---|---|---|
| σ\* (ideal) | +1.00 | 0 | +1.00 | −0.0365 | 0 | 68% |
| X1 | −0.00 | −0.00 | +0.06 | **−0.0386** | 0.33 | 64% |
| D-pos-v2 (b), the line's X | +0.60 [+0.41, +0.77] | +0.00 | +0.80 | −0.0393 | 0.07 | 68% |
| RSS | 0 | 0 | 0 | −0.0394 | 0.40 | 62% |
| **D-pos-v2 (a), the answer's σ** | **+0.53 [+0.33, +0.71]** | −0.00 | **+0.88** | **−0.0395** | **0.07** | **67%** |
| D-pos-v2 (c), exact from its terms | +0.67 [+0.51, +0.82] | +0.01 | +0.82 | −0.0408 | 0.02 | 66% |
| B-v2 | +0.47 [+0.33, +0.61] | −0.02 | −0.29 | −0.0409 | 0.48 | 55% |
| D-pos (v1, #29) | −0.03 | – | −0.18 | −0.0405 | 0.31 | 54% |

- **R1–R8 (Holm):**
  - **Supported:** R1–R3 (D-pos-v2's A6c slope > 0 on (a), (b), (c)), R4 (B-v2's > 0), and R5 (D-pos-v2 − B-v2:
    +0.0015 [+0.0008, +0.0022]).
  - **Not supported:** R6 (− RSS: −0.0001), R7 (− X1: −0.0008) and R8 (B-v2 − RSS: −0.0016).
- **The line:**
  - **Levels read right:** 99.5% (D-pos v1: 1% on A6c).
  - **Sensitivities exact (within 10%):** power-law terms 86% (87%); value-dependent terms 14% (48%), falling from
    31% at depth 1 to 4% at the held-out depth 5.
  - **The copy:** 85%.
- **The A6c slope by depth (answer's σ):** 1.00 / 0.97 / 0.76 / 0.38 / 0.06. The explicit computation does not
  extrapolate to the unseen depth.
- **Exploratory (not pre-registered):**
  - **Concentration:** 80% of (a)'s regret is in its worst 5% of problems; 1.5% of its σ are off by more than 10×.
  - **By depth:** its mean regret is 0.0007 / 0.0014 / 0.0024 / 0.0032 / 0.0071 at depths 1–5, against RSS's flat
    ~0.003.
- **The data effect (descriptive):** D-pos-v2 − D-pos +0.0010 [+0.0003, +0.0017]; B-v2 − stage B +0.0005 (CI spans 0).

## 3. MC-A (D113 §7; FINDINGS #31)

- **Setup:** B-v2, k = 8 perturbations per problem, greedy decoding. Each noisy given is drawn as
  x_meas / (1 + s·z).

| A6c, k = 8 | expected crps_log | slope [90% CI] | epistemic AUROC | end-to-end |
|---|---|---|---|---|
| oracle (exact chain) | **−0.0358** | +1.01 | 0.656 | −0.146 |
| X1 / RSS | −0.0373 / −0.0383 | 0 / 0 | 0.666 / 0.654 | −0.146 / −0.149 |
| robust (filter + MAD) | −0.0446 | +0.48 [+0.04, +0.90] | 0.656 | −0.140 |
| raw (SD) | −0.0580 | +0.53 | **0.745** | **−0.136** |

- **The stopping rule (R4-4):** robust − RSS is −0.0062 [−0.0088, −0.0039]. **Robust MC-A does not beat RSS; k is not
  increased.**
- **The k curve:**
  - robust and oracle improve from k = 2 to 8;
  - **raw gets worse** (−0.0556 → −0.0580), since each extra run is another chance of an arithmetic outlier.
- **On A6 (descriptive):** MC-A recovers the value-dependent sensitivity B-v2's stated σ lacks (robust slope +1.12),
  but it still scores below RSS.
- **Raw is the best end-to-end σ on all three sets.** Run 0's μ is within 0.1% of f(x_meas) on only 8–20%, so a σ
  that also covers the model's own error is the better delivered answer.

## 4. Built tonight (all tested)

- **The generator:** `mixed_levels`, an option of `generate_problem`. The default is unchanged: D80's problems and
  the frozen sets regenerate byte-identically.
- **The data:** `physcal/train/sft_r4.py` builds the R4-1 data. Every S value is checked against finite-difference
  sensitivities.
- **The arms and their analysis:**
  - `physcal/train/stage_r4.py` and `physcal/eval/r4_analysis.py`: the three σ values, the criteria, R1–R8, and the
    depth / type splits;
  - `physcal/eval/mc_a.py`: resumable chunks; the raw, robust and oracle variants; the k curve; R4-6's estimands.
- **The confirmatory set (R4-5):** `data/eval_sets/2.0.0+bdbb2279/confirmatory/` (510 pairs per set, checksums,
  `FROZEN.md`). `run_eval` refuses it without `PHYSCAL_CONFIRMATORY=1`.
- **The fix:** `run_eval`'s OOM backoff (D116), with a regression test that fails on the old code.
- **Checks:** `scripts/adapter_load_check.py`.
- **The queues:** `queue_r4.sh`, `queue_r4_tail.sh` and `queue_mc_a.sh`, each with the free-space guard.

## 5. BLOCKED / problems

- **D-pos-v2's first eval (18:55–19:01): BLOCKED by a CUDA OOM** that the backoff could not recover from, even at 1
  prompt.
  - **The cause:** the retry recursion ran inside the `except` block, so every failed attempt's tensors stayed alive.
  - **Fixed (D116)** and rerun at 21:27, with no OOM.
  - **Two consequences:**
    - the order became B-v2 eval, then D-pos-v2 eval;
    - both R4 evals and MC-A use batch 16 (D-pos v1 used 32).
- **Temperature:** three 10-min pauses at 88 °C (17:57, 19:24, 02:19). The first came while the test suite ran on one
  CPU thread.
- **Host VRAM was 1.6 GiB tonight** (0.6 on earlier nights), so the cap was 5.9 GiB, not 7.1 (D115).
- **No spill, no disk problem:** 53.9 GiB free on MLStorage.

## 6. Decisions to review

1. **The R4-3 go: a 3-seed confirmatory run of D-pos-v2 and B-v2 on the frozen R4-5 set.** It needs its own
   pre-registration; nothing has been run.
   - **Budget:** 6 × (~80 min SFT + ~110 min eval). Two nights, or one night if the evals keep batch 16 and the
     3-seed SFTs run first.
   - **Also to fix in that entry:** whether "propagates" there needs C2 (≥ 0.5) at the 3-seed mean, and whether R6
     (beats RSS) becomes primary. Tonight it ties.
2. **D113's [REVIEW] reading of R4-3:** C2 is part of the criterion. It did not matter tonight, since (a), (b) and (c)
   all exceed 0.5.
3. **D114's deviations:** the 1,280-token eval budget (the S lines make completions long), and 6 training rows
   dropped as too long.
4. **D116:** batch 16 for the R4 evals and MC-A, and the changed order.
5. **The mixed-level option is not a version bump** (D113 [REVIEW]), so the generator version string is unchanged.
6. **MC-A's model is B-v2** (D113 [REVIEW]). Its arithmetic is what limits MC-A, so another implicit model would
   likely do the same.

## 7. Recommendation

- **The pre-registered next step is the 3-seed confirmation on R4-5's frozen set** (D-pos-v2 and B-v2). D-pos-v2's
  propagation is large (slopes of 0.5–0.9 against ~0 for every earlier arm) and is unlikely to vanish with seeds.
- **The open question is the score, not the propagation.** D-pos-v2 is calibrated in the median but ties RSS because
  of gross errors in long symbolic derivatives at depth ≥ 4.
  - **The R4-8 "computing vs predicting" story now has both halves in one night:**
    - an explicit learned computation propagates, and fails where its arithmetic fails;
    - an exact executor (MC-A's oracle) reaches σ\*.
  - **That points to MC-B (R4-7: the model writes the program, an interpreter runs it)** as the cleanest next arm,
    under its own pre-registration.
- **Write-up items from tonight:**
  - the depth curve (training depths vs the held-out depth 5);
  - the regret-concentration analysis (it explains why calibration and score diverge);
  - MC-A's epistemic AUROC next to self-consistency's, at matched k. That needs X2's k-cost curve (R4-7's
    cross-family item).

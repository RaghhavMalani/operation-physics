# Report: 2026-10-27/28 night: the R4-3 confirmatory run (D118), night 2 of 2, and MC-B (D119)

The previous report (night 1) is in git history: `git show 682b522:MORNING.md`.

**Short version.**
1. **All six confirmatory evals exist** (3,060 / 3,060 completions each). **D118's one-time analysis ran** (09:29–09:31,
   CPU, exit 0). It is labelled complete (3 seeds per arm) and is final; it is not rerun.
2. **P is CONFIRMED** (D125: K1 ∧ C2 ∧ C3). D-pos-v2's answer σ (a) propagates on A6c at the 3-seed mean:
   - **K1:** the slope is **+0.816**, 90% CI [+0.708, +0.923], Holm-rejected (bootstrap p < 0.0001);
   - **C2:** +0.816 ≥ 0.5;
   - **C3:** the A6b slope is +0.036, 90% CI [−0.057, +0.212], inside ±0.3.
3. **The secondaries fail, with the same prominence (D118 §12):**
   - **K6, beats RSS:** not supported (+0.0002 [−0.0002, +0.0006], p = 0.20).
   - **K7, beats X1:** not supported. The difference is negative and its CI excludes 0: −0.0005 [−0.0009, −0.0001].
   - So the stated σ propagates noise, but its expected score is no better than either heuristic.
4. **MC-B fails all five tests on A6c (M1–M5, Holm), and none is close.**
   - **Program failures, unconditional:** 84.8% on A6c and 91.0% on A6. Most are `signature` failures (64–68%).
   - **Correct programs:** 0.3–0.5% compute the chain correctly.
   - **A6b's generation did not fit before 10:00.** Its analysis is BLOCKED and resumes on the next run (D127).
5. **One stage hung.** D-pos-v2 seed 1's first eval froze after a Windows GPU driver reset at 23:33:31 (nvlddmkm 153).
   - It timed out at 02:48 (exit 124, BLOCKED).
   - The retry (attempt 2 of 3) ran 05:40–07:24 with exit 0 (D131).
   - The night's System log has no other driver reset.
6. **Temperature:** six 10-min pauses, all during evals (at 87.1–88.0 °C). Night 1's evals had none.
7. **Free space:** 53 GiB on MLStorage all night.
8. **Every push landed.** `main` is in sync with `origin/main`.

---

## 1. What ran (IST)

| time | stage | result | duration | commit |
|---|---|---|---|---|
| 22:56 | Windows task `physcal-night-queue` deleted (it existed, Ready); waiter 67051 already dead (VM restart 22:51) | – | – | D130 |
| 22:57–22:59 | chain 1046 (09:00 deadline) stopped before any stage: you moved the deadline to 10:00 | – | – | – |
| 22:59–23:00 | chain 1762 stopped before any stage: `cool()` read 55.70 °C as 5570 °C | – | – | – |
| 23:00–23:04 | temperature parser fixed (both scripts), 2 tests added (14 passed, 1 skipped) | – | – | `4e5aa77` (D129) |
| 23:04 | chain 8120 launched: 10:00 deadline, cap 5.9 GiB, 53 GiB free | – | – | – |
| 23:04–02:48 | eval D-pos-v2 s1, attempt 1 (pause 23:12–23:22); hung after the 23:33:31 driver reset | exit 124, BLOCKED; nothing written | 223 min | `7bf215f` |
| 02:52 | D131: retry s1 after s2 and before MC-B; stall detection in the session's watcher | – | – | `c9af378` |
| 02:48–05:40 | eval D-pos-v2 s2 (five pauses: 03:24, 03:40, 04:24, 04:46, 05:22) | 3,060 / 3,060, exit 0 | 172 min | `a29c6e6` |
| 05:40 | chain 8120 stopped at its commit wait; the session made the identical commit and relaunched (D131) | – | – | – |
| 05:40 | chain 55404 launched (same deadline and cap) | – | – | – |
| 05:40–07:24 | eval D-pos-v2 s1, attempt 2 | 3,060 / 3,060, exit 0 | 104 min | `f1c111e` |
| 07:24–08:25 | eval B-v2 s2 | 3,060 / 3,060, exit 0 | 61 min | `01d0a50` |
| 08:25–08:56 | MC-B generation A6c (dev set) | 1,020 / 1,020, exit 0 | 31 min | `1daf260` |
| 08:56–09:27 | MC-B generation A6 | 1,020 / 1,020, exit 0 | 31 min | `81e2c18` |
| 09:27 | MC-B generation A6b: SKIP (32 min left, needs 55) | – | – | – |
| 09:27–09:28 | MC-B A6c: execute (sandbox, 1 thread) + analyze | exit 0 / exit 0 | 9 s + 33 s | `c823c35` |
| 09:28–09:29 | MC-B A6: execute + analyze | exit 0 / exit 0 | 9 s + 30 s | `4791795` |
| 09:29 | MC-B A6b analysis: BLOCKED, no generations (resumes next run, D127) | – | – | – |
| 09:29–09:31 | `r4_confirm` (CPU, 1 thread, nice 19): D118's one-time analysis | exit 0 | 2 min | `d8bc805` |
| 09:31 | chain done | – | – | – |

The GPU has been idle since 09:27.

## 2. Operational counts

### The six confirmatory evals (frozen R4-5 set, 1,020 problems × A6c / A6 / A6b)

| eval | completions | truncated | parse-fail | unit mismatch | OOM backoffs | mean tokens | eval time | reload check (max logit Δ) |
|---|---|---|---|---|---|---|---|---|
| D-pos-v2 s0 (night 1) | 3,060 / 3,060 | 0 | 5 (0.16%) | 7 | 0 | 386 | 6,873 s | pass (4.11) |
| D-pos-v2 s1 (retry) | 3,060 / 3,060 | 0 | 4 (0.13%) | 4 | 0 | 384 | 6,162 s | pass (3.89) |
| D-pos-v2 s2 | 3,060 / 3,060 | 2 (0.07%) | 6 (0.20%) | 14 | 0 | 385 | 10,245 s (incl. 50 min of pauses) | pass (4.26) |
| B-v2 s0 (night 1) | 3,060 / 3,060 | 0 | 3 (0.10%) | 0 | 0 | 241 | 3,501 s | pass (4.79) |
| B-v2 s1 (night 1) | 3,060 / 3,060 | 0 | 0 | 0 | 0 | 241 | 3,771 s | pass (4.70) |
| B-v2 s2 | 3,060 / 3,060 | 0 | 1 (0.03%) | 0 | 0 | 240 | 3,582 s | pass (4.61) |

- **Identical code for all six:** no file under `physcal/` or `configs/`, nor `pyproject.toml` or `uv.lock`, changed
  between night 1's first eval and tonight's last.
- **The fingerprint check** (R2-1) passed on every adapter.
  - **The s1 retry:** its logit Δ equals what the hung attempt's check printed.
- **Not committed by the queue:** the 12 calibration / reliability PNGs of tonight's three evals. They are committed
  with this report.

### MC-B generations (base model, no adapter; 640 new tokens, greedy, batch 16)

| set | programs | truncated | OOM | loader check | time |
|---|---|---|---|---|---|
| A6c | 1,020 / 1,020 | 11.2% | 0 | pass | 1,833 s |
| A6 | 1,020 / 1,020 | 11.2% | 0 | pass | 1,872 s |
| A6b | not generated (deadline gate) | – | – | – | – |

## 3. D118's verdict (one-time confirmatory analysis, complete: 3 seeds per arm)

Source: `results/stage_r4/confirmatory/summary.md` (`results.json` beside it).
- **The design:** 3,060 problems (1,530 pairs); stage B's joint bootstrap over pairs, B = 10,000.
- **X1:** used, because D99's reproduction passed.
- **"p < 0.0001"** below means no bootstrap resample fell at or below 0.

### The primary P (D125): CONFIRMED

| criterion | rule | value | holds |
|---|---|---|---|
| K1 | A6c seed-mean slope of (a) > 0, Holm-adjusted one-sided p < 0.05 | +0.816, p < 0.0001 (Holm threshold 0.00625) | yes |
| C2 | A6c seed-mean point estimate ≥ 0.5 | +0.816 | yes |
| C3 | A6b seed-mean 90% CI inside (−0.3, +0.3), computation-correct pairs | +0.036 [−0.057, +0.212] | yes |

- **D113's full criterion with C1's unadjusted 90% bound:** also holds.
- **The reading fixed in D118 §12:** "D-pos-v2's explicit sensitivity computation propagates measurement noise on fresh
  problems, across 3 seeds."

### K1–K8, Holm at α = 0.05 (seed means)

| test | seed mean [90% CI] | p (one-sided) | Holm | per seed (0 / 1 / 2) |
|---|---|---|---|---|
| **K1** (primary): D-pos-v2 (a) answer σ, A6c slope > 0 | **+0.816 [+0.708, +0.923]** | < 0.0001 | **rejected** | +0.778 / +0.770 / +0.900 |
| K2: D-pos-v2 (b) line X, A6c slope > 0 | +0.884 [+0.745, +1.041] | < 0.0001 | rejected | +0.801 / +0.723 / +1.128 |
| K3: D-pos-v2 (c) exact from terms, A6c slope > 0 | +0.843 [+0.713, +0.987] | < 0.0001 | rejected | +0.838 / +0.709 / +0.982 |
| K4: B-v2, A6c slope > 0 | +0.416 [+0.315, +0.517] | < 0.0001 | rejected | +0.455 / +0.373 / +0.419 |
| K5: expected crps_log, D-pos-v2 (a) − B-v2 > 0 | +0.0013 [+0.0008, +0.0017] | < 0.0001 | rejected | +0.0012 / +0.0019 / +0.0008 |
| **K6** (secondary): D-pos-v2 (a) − RSS > 0, "beats RSS" | **+0.0002 [−0.0002, +0.0006]** | 0.20 | **not rejected** | −0.0005 / +0.0010 / +0.0001 |
| **K7** (secondary): D-pos-v2 (a) − X1 > 0, "beats X1" | **−0.0005 [−0.0009, −0.0001]** | 0.99 | **not rejected** | −0.0012 / +0.0003 / −0.0007 |
| K8: B-v2 − RSS > 0 | −0.0011 [−0.0013, −0.0009] | 1 | not rejected | −0.0016 / −0.0009 / −0.0007 |

**Reading:**
- **K1–K5 hold.** Both arms propagate on A6c, and D-pos-v2 propagates more and scores better than B-v2.
- **D-pos-v2 does not beat RSS (K6) or X1 (K7) on expected score.**
  - **On X1:** the point estimate is below zero, with a CI that excludes zero. "X1 is better" is not a tested claim:
    K7 is one-sided in the other direction.
- **B-v2 scores below RSS** (K8, negative).

### The three σ sources of D-pos-v2 (seed means)

| source | A6c slope [90% CI] | A6b slope [90% CI] (C3) | A6 slope, comp.-correct | C1 / C2 / C3 | full D113 |
|---|---|---|---|---|---|
| (a) answer σ | +0.816 [+0.708, +0.923] | +0.036 [−0.057, +0.212] | +1.195 [+0.992, +1.417] | ✓ / ✓ / ✓ | yes |
| (b) the line's X | +0.884 [+0.745, +1.041] | −0.026 [−0.049, −0.007] | +0.814 [+0.639, +0.997] | ✓ / ✓ / ✓ | yes |
| (c) exact from the line's terms | +0.843 [+0.713, +0.987] | −0.023 [−0.047, −0.005] | +0.829 [+0.647, +1.017] | ✓ / ✓ / ✓ | yes |
| (c_S), descriptive | +0.804 [+0.710, +0.894] | +0.002 [−0.003, +0.008] | +0.815 [+0.660, +0.966] | ✓ / ✓ / ✓ | yes |
| B-v2 (a) | +0.416 [+0.315, +0.517] | +0.003 [−0.086, +0.129] | +0.503 [+0.032, +1.038] | ✓ / ✗ / ✓ | no |

### The pre-specified depth split (training depths 1–4 vs held-out depth 5; outside Holm, no confirmatory claim)

| test | all | depths 1–4 | depth 5 |
|---|---|---|---|
| K1 | +0.816 [+0.708, +0.923] | +0.855 [+0.734, +0.966] | +0.820 [+0.569, +1.054] |
| K2 | +0.884 [+0.745, +1.041] | +0.942 [+0.782, +1.129] | +0.821 [+0.542, +1.079] |
| K3 | +0.843 [+0.713, +0.987] | +0.912 [+0.769, +1.082] | +0.755 [+0.491, +1.008] |
| K4 | +0.416 [+0.315, +0.517] | +0.379 [+0.259, +0.497] | +0.547 [+0.350, +0.743] |
| K5 | +0.0013 [+0.0008, +0.0017] | +0.0016 [+0.0011, +0.0020] | +0.0001 [−0.0011, +0.0012] |
| K6 | +0.0002 [−0.0002, +0.0006] | +0.0007 [+0.0003, +0.0011] | −0.0018 [−0.0029, −0.0008] |
| K7 | −0.0005 [−0.0009, −0.0001] | −0.0001 [−0.0005, +0.0003] | −0.0022 [−0.0033, −0.0012] |
| K8 | −0.0011 [−0.0013, −0.0009] | −0.0009 [−0.0011, −0.0007] | −0.0019 [−0.0025, −0.0013] |

What the split shows, descriptively:
- **Propagation holds at the held-out depth 5:** (a)'s slope there is +0.820.
  - FINDINGS #30's development set (one seed) had (a)'s A6c slope fall to 0.06 at depth 5.
  - The 3-seed confirmatory estimate does not reproduce that fall.
- **The score is where it falls short.**
  - **Depths 1–4:** D-pos-v2 is ahead of RSS (+0.0007, CI above 0).
  - **Depth 5:** it is behind both RSS and X1 (CIs below 0).
  - So pooled K6 / K7 fail on the deep chains.
- **Regret concentration (D118 §9):** the worst 5% of problems hold 83% / 69% / 76% of (a)'s expected-score regret
  against σ\* (seeds 0 / 1 / 2). In every seed the mean regret is highest at depth 5. This matches FINDINGS #30's
  "gross misses on deep chains".

### Stage B's analysis per arm (D118 §9, untested here; `stage_b_dpos2/`, `stage_b_b2/`)

- **D-pos-v2:** "the ceiling propagates".
  - B1, B2 and B5 supported; B3 / B4 (vs RSS / X1) not supported.
  - A6b is equivalent to 0; the A6 label is full propagation.
  - **Coverage per seed:** 1σ 62.9–69.7% and 2σ 87.1–91.0%, against σ\*'s 70.9% / 94.4%.
  - **Median ln(σ/σ\*):** −0.06 to +0.01.
  - **Per-seed D59 labels:** full / magnitude-driven / full.
- **B-v2:** "another combination".
  - B2 supported; B1, B3, B4 and B5 not.
  - **Per-seed A6 labels:** none / none / full.

## 4. MC-B (D119 / D122): a prototype on the development sets

Sources: `results/mc_b/{a6c,a6}_pairs/summary.md`.
- **The problems:** 1,020 per set, none excluded by MC-A.
- **Failure counting:** every problem is in the denominator; a failed program gives σ = 0 (the floor) and μ unanswered.

### M1–M5 (A6c only, Holm over the five at α = 0.05): none rejected

| test | value [95% CI] | p (one-sided) | Holm |
|---|---|---|---|
| M1: MC-B − RSS > 0 | −0.0099 [−0.0105, −0.0093] | 1 | not rejected |
| M2: MC-B − X1 > 0 | −0.0109 [−0.0116, −0.0102] | 1 | not rejected |
| M3: MC-B − D-pos-v2 (a) > 0 | −0.0110 [−0.0119, −0.0102] | 1 | not rejected |
| M4: MC-B − MC-A robust > 0 | −0.0037 [−0.0060, −0.0011] | 0.996 | not rejected |
| M5: MC-B's A6c slope > 0 | −0.444 [−1.135, +0.151] | 0.92 | not rejected |

- **The attainable-bound gap, untested:** MC-B − MC-A oracle = −0.0125 [−0.0133, −0.0117].
- **A6, descriptive:** the same differences, all negative except M5 (+0.229 [−0.583, +1.016]).

### Programs: the unconditional failure rate (D119 §5, every problem counted)

| set | failure rate | no_program | rejected | signature | exception | bad_output | run-0 comp.-correct | exact program |
|---|---|---|---|---|---|---|---|---|
| A6c | **84.8%** | 15.1% | 0.7% | 63.9% | 3.9% | 1.2% | 0.5% | 0.5% |
| A6 | **91.0%** | 15.5% | 0.4% | 67.6% | 4.1% | 3.3% | 0.3% | 0.3% |

- **By depth:**
  - **Failure rate:** A6c 82.8 / 75.0 / 85.3 / 90.7 / 90.2%; A6 89.7 / 95.1 / 90.2 / 91.2 / 88.7%.
  - **Run-0 computation-correct:** 1.5% at depth 1, 0% from depth 3 on (A6c).
- **Where it loses (D119 §9's reading for a failed M1):**
  - **The signature.** A `solve` parameter that is not a displayed given, without a default, fails (D122 deviation 2).
    The top A6c reasons: g (152), m (96), h (85), k (71), t1 (63).
  - **15% have no extractable program.** Example: the first A6c generation is a top-level script with hard-coded
    givens and an imitated "```output" block, in the base model's tool-use style, with no `solve`.
  - **Programs that run are almost never right.** 0.3–0.5% compute the chain.
- **Truncation:** 11.2% at 640 tokens on both sets.
- **No k increase and no prompt change follow** without a new entry (D119 §9).

### Comparison on A6c (expected crps_log, oracle-centred; higher is better)

| σ source | expected crps_log | median ln(σ/σ\*) | 1σ / 2σ cov. | slope [90% CI] |
|---|---|---|---|---|
| σ\* | −0.0351 | 0.000 | 70.2% / 95.7% | +1.000 |
| MC-A oracle | −0.0358 | −0.041 | 67.5% / 91.0% | +1.006 [+0.89, +1.12] |
| D-pos-v2 (a), seed 0 dev eval | −0.0372 | +0.008 | 68.8% / 91.8% | +0.554 [+0.36, +0.74] |
| X1 | −0.0373 | +0.018 | 70.2% / 90.6% | −0.002 |
| RSS | −0.0383 | −0.099 | 66.7% / 87.0% | 0.000 |
| MC-A robust | −0.0446 | −0.121 | 61.2% / 81.9% | +0.475 [+0.04, +0.90] |
| **MC-B, k = 8** | **−0.0482** | **−10.663** | **7.7% / 11.8%** | **−0.444 [−1.01, +0.06]** |

- **MC-B's median ln(σ/σ\*) of −10.7:** the failed programs (the majority) sit at the σ = 0 floor.
- **The k curve (2 / 4 / 8) is flat:** −0.0486 / −0.0483 / −0.0482.

## 5. BLOCKED items and problems

1. **The D-pos-v2 s1 hang (D131).**
   - **What happened:** after the Windows driver reset at 23:33:31, the eval's CUDA process stopped making progress
     without exiting. It sat for ~3 h until its 13,400 s timeout (exit 124).
   - **The retry:** attempt 2 of 3 succeeded.
   - **The cost:** ~3.2 h of GPU time. That cost is why A6b's MC-B generation missed the 10:00 gate.
   - **My watcher missed the stall** (it woke only on chain.log lines). It now also wakes when the running GPU stage's
     log is silent for 15 min while the stage is not temperature-paused. **The queue itself has no stall watchdog**
     (§6.3).
2. **MC-B A6b: BLOCKED**, with no generations: the deadline gate skipped it (32 min left, needs 55). Under D127 it
   resumes on the next run: ~31 min of GPU, then ~40 s of CPU.
3. **Two launches stopped before any stage started** (22:59, 23:00):
   - the deadline change;
   - the temperature parser. The driver now reports 55.70 rather than 55. It is fixed and tested (D129).
4. **Temperature:** six 10-min pauses, at 23:12 (88.00 °C), 03:24 (87.30), 03:40 (87.40), 04:24 (87.10), 04:46 (87.90)
   and 05:22 (87.10).
   - **Where:** all in D-pos-v2 evals, five of them in seed 2's.
   - **No pauses** in the s1 retry, B-v2 s2 or MC-B.
   - **Possibly relevant:** while the stage was paused, `dwm.exe` used ~16–29% of the GPU's 3D engine (12–16 W). The
     display was on and rendering. I did not act on this (§6.4).
5. **Memory and disk:**
   - **Host VRAM:** 572–765 MiB at the three launches that ran, so the cap stayed 5.9 GiB, at parity with seed 0.
     The first read at 22:57 gave 2,302 MiB; it had settled by 22:59.
   - **Free space:** 53 GiB on MLStorage throughout.
6. **Not touched:** the untracked `PHASES_10-15.md`.

## 6. Decisions for you

**Decided at ~10:00 (D132):**
1. FINDINGS #32 and #33 are drafted.
2. A6b goes into the next queue.
3. The queue-level stall watchdog is added and tested.
4. You set the display to turn off after 1 min on AC; no TDR changes.
No GPU work until your next plan.

1. **FINDINGS entries.** I have not written them.
   - **#32 (the confirmatory result):** P confirmed; K6 / K7 not supported, with K7's estimate negative; depth 5 keeps
     the slope but loses the score.
   - **#33 (MC-B):** all M tests fail; 85–91% of programs fail, mostly at the signature.
   - Shall I draft both?
2. **MC-B A6b next run (D127).** It is descriptive only (M1–M5 are on A6c), and it needs ~31 min of GPU. Launch it from
   a session tonight, or fold it into the next queue?
3. **A stall watchdog in `queue_r4_seeds.sh`:**
   - **The rule:** kill a GPU stage whose log has been silent for N min while it is not SIGSTOPped by the temperature
     guard; it is then BLOCKED and the queue continues.
   - **Status:** needs a D-entry and tests. Today only the session's watcher detects a stall.
4. **Heat and the display:**
   - **The current setting:** the display turns off after 15 min on AC.
   - **The options:** turn the display off at launch, or shorten the timeout. Either might cut dwm's GPU load during
     runs.
   - **Speculative:** whether this lowers the temperature, or bears on the driver reset. A Windows TDR setting change is
     possible but outside the repo, and I do not recommend it without your call.
5. **D130 is the D-entry you asked for:** scheduled starts retired; launches are manual from a Claude Code session.
   It was pushed at 23:04, with D129 (`4e5aa77`). Tonight's launches followed it: no Windows task, no waiter.

## 7. Recommendation

- **Record the confirmatory result as it stands.** D-pos-v2's stated σ propagates measurement noise on fresh problems
  across 3 seeds (P confirmed, all three σ sources). It does not buy a better expected score than RSS or X1, and the
  shortfall sits at the held-out depth 5. Both halves deserve equal prominence in FINDINGS and the write-up.
- **Treat MC-B as a negative result for this base model and prompt.** The program arm loses at program writing, not
  at propagation: 85–91% of programs fail, and almost none of the rest compute the chain. Finish A6b for completeness
  on the next GPU slot (~31 min); expect nothing to change.
- **Before the next long unattended run,** add the queue-level stall watchdog (§6.3). One driver reset cost 3.2 GPU
  hours tonight.

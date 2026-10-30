# Report: 2026-10-29/30 night: recovery, resumable evals, temperature hysteresis, the XM queue (D140–D145)

Written on 2026-10-30 at ~08:55 IST, as soon as the chain ended. The source is `results/queue_xm2/chain.log`, the stage
logs and the result files; the working log is in `docs/lab_notebook.md`. The 28/29 report is in git history:
`git show d911d45:MORNING.md`.

**Short version.**
1. **Every stage finished, each first time.**
   - The chain ran 02:18:47 → 08:31:28 (6 h 13 min), 4.5 h before the 13:00 deadline.
   - **Every commit was pushed:** `main` = `origin/main` = `f31dcc7` when the chain ended.
2. **The D-pos-v2 Llama eval is complete** (2,040 / 2,040, exit 0, `2f0435f`). This is the eval that timed out with
   nothing saved on 28/29. It ran in one session, with every batch on disk as it went (D142).
3. **The go rule (D137 / D139) is NO-GO.**
   - K1 and C2 hold: the A6c slope is +0.785 [+0.595, +0.974], 505 pairs.
   - **C3 fails:** only 4 computation-correct A6b pairs, so the CI is [−3.55, +843].
   - D137 named this risk in advance. Under the pre-registered rule, there is no seed expansion, and it is reported as a
     failed replication at one seed. **Your decision 1 below.**
4. **MC-B A6b** (descriptive, D132) repeats A6c.
   - 87.7 % of the programs fail, and only 1.3 % compute correctly. MC-B's σ is collapsed: 1σ coverage 6.5 %.
   - It is worse than RSS, X1 and D-pos-v2 (a). It beats MC-A robust on A6b, unlike on A6c (`876f4ef`).
5. **SFT B-v2 on Llama, seed 0, finished:** 750 / 750, exit 0, loss 0.155, fingerprint recorded (`f31dcc7`). The
   projection check put its end at 08:36–08:38 from the first check on.
6. **Temperature:**
   - 16 pauses, every one 30–31 s, 482 s in all: 2.2 % of 22,276 s of GPU-stage time.
   - On 28/29 the night was paused 46 % of the time (40 pauses of 10 min). D143's ≤ 75 °C resume did this. The GPU was
     at 60–64 °C each time it continued.
7. **Checks after the chain:**
   - No process or GPU memory left. Free space 47.17 GiB.
   - The integrity check passed on all 64 files written since the launch: 0 problems.
   - The full test suite: 700 passed, 0 failed.
8. **One bug in tonight's queue, fixed after the night (D145):** the stopped projection check's `sleep 900` held the
   queue lock for 2 min 17 s after the chain ended. No effect tonight. Also a cosmetic fix to the guard's log line.

---

## 1. What ran (IST)

| time | stage | result | wall | pauses | commit |
|---|---|---|---|---|---|
| 02:18:47 | chain 35695: `GPU_DEADLINE=13:00`, launch cap 5.9 GiB (host 741–751 MiB), 47 GiB free | – | – | – | – |
| 02:19:07–05:44:35 | eval D-pos-v2 (Llama, s0) on frozen A6c + A6b, attempt 1 of 3; reload check passed | **2,040 / 2,040, exit 0** | 12,330 s | 3, 91 s | `2f0435f` |
| 05:44:40–05:44:58 | go rule `xm_go` (CPU, 1 thread, nice 19) | **NO-GO** (K1 ✓, C2 ✓, C3 ✗) | 18 s | – | `37d8723` |
| 05:45:02–06:18:02 | MC-B A6b generation (Qwen2.5-Math-1.5B) | 1,020 / 1,020, exit 0 | 1,982 s | 2, 60 s | `023bcfa` |
| 06:18:06–06:18:38 | MC-B A6b execute + analyze (CPU) | exit 0, exit 0 | 31 s | – | `876f4ef` |
| 06:18:42–08:31:25 | SFT B-v2 (Llama, s0); start gate 148 min est.; projection every 15 min | **750 / 750, exit 0** | 7,964 s | 11, 331 s | `f31dcc7` |
| 08:31:28 | chain done: "16 temperature pauses, 482s paused of 22276s GPU-stage time" | – | – | – | – |

Lab-notebook commits during the night: `bb792d5`, `ff4a0f7`, `58fe961`, `d3ffeec`, `f116a30`.

## 2. Numbers

### Eval D-pos-v2 (Llama-3.2-3B, seed 0), frozen A6c + A6b

- **Generation:** 12,283 s for 2,040 problems, ~6.0 s / problem. Batch 6, greedy, 340 batches.
- **Memory:** peak reserved 5.86 GiB under the 5.9 GiB cap; nvidia-smi peak 6.76 GiB device-wide.
- **Resume record (D142):** session 1; 0 batches read, 340 generated; nothing moved aside. The partial file was
  deleted after metrics.json.

| split | n | acc ≤ 1 % | median rel. err | 1σ cov | 2σ cov | median rel σ | parse fail | unit mismatch |
|---|---|---|---|---|---|---|---|---|
| a6b_pairs | 1,020 | 8.3 % | 0.215 | 24.0 % | 34.7 % | 0.026 | 0.0 % | 0.0 % |
| a6c_pairs | 1,020 | 10.2 % | 0.124 | 28.5 % | 44.8 % | 0.037 | 0.4 % | 0.3 % |

- **Accuracy by depth on A6b:** 20.1 / 7.8 / 7.4 / 3.4 / 2.9 % (depth 1 → 5).
- **By family:** energy is the weak family (3.0 % on A6b, 4.4 % on A6c); dynamics and kinematics are at 12–13 %.
- **Coverage on computation-correct answers is near nominal** (A6b, 1 % noise: 66.7 % at 1σ). Pooled under-coverage
  comes from wrong answers.
- **The standing post-hoc scale (σ × 18.1)** gives 1σ coverage of 65.5 % on A6b and 85.2 % on A6c.

### The go rule (D137, run early per D139): NO-GO

| criterion | estimate | 90 % CI | pairs | rule | met |
|---|---|---|---|---|---|
| K1 (= C1 at one seed) | A6c slope +0.785 | [+0.595, +0.974] | 505 | one-sided bootstrap p < 0.05 (p = 0.0) | yes |
| C2 | A6c slope +0.785 | | 505 | ≥ 0.5 | yes |
| C3 | A6b slope +0.567 | [−3.55, +843] | 4 | CI inside (−0.3, +0.3); ≥ 3 pairs | **no** |

- **What C3's failure is:**
  - It is a power failure: with 8.3 % accuracy on A6b, only 4 pairs have both members computation-correct.
  - It is **not** evidence that the answer σ responds to Δ ln truth on A6b; the 4-pair CI covers everything.
- **D137's own wording, written before any result:** "A model with weaker arithmetic may have too few for C3 to hold,
  whatever its propagation. Then the pilot is NO-GO, reported with that reason."
- **What replicated on Llama:** the answer σ propagates on A6c, with the slope's CI clear of 0 and of 0.5.
- **What could not be shown:** the A6b control.
- **Files:** `results/stage_xm/go_dpos2/{go.json, summary.md}`.

### MC-B on A6b (D119 / D132, descriptive; A6c holds the tests)

- **The programs:**
  - failure 87.7 %: signature 69.1 %, no_program 16.5 %, exception 1.7 %;
  - truncated 11.1 %;
  - run 0 computation-correct 1.3 %;
  - failure by depth: 83.8 % at depth 1 to 98.5 % at depth 5;
  - on A6c: failure 84.8 %, correct 0.5 %.
- **MC-B's σ:** median ln(σ/σ\*) = −10.4; 1σ / 2σ coverage 6.5 % / 8.9 % (k = 2).

| difference (expected crps_log) | value | 95 % CI | on A6c |
|---|---|---|---|
| M1: MC-B − RSS | −0.0102 | [−0.0109, −0.0094] | −0.0099 |
| M2: MC-B − X1 | −0.0105 | [−0.0113, −0.0097] | −0.0109 |
| M3: MC-B − D-pos-v2 (a) | −0.0075 | [−0.0091, −0.0056] | −0.0110 |
| M4: MC-B − MC-A robust | **+0.0110** | [+0.0058, +0.0168] | −0.0037 |
| M5: MC-B's slope | −0.018 | [−0.15, +0.11] (A6b, on Δ ln truth) | −0.444 (A6c) |

- **M4 changes sign** because MC-A robust has a non-zero A6b slope (+0.103, 90 % CI [+0.01, +0.19]) on a set where the
  right slope is 0. MC-B's collapsed σ does not move with truth there.
- **A6b agrees with FINDINGS #33:** MC-B fails as a σ source because its programs fail.
- **Two display problems in `results/mc_b/a6b_pairs/summary.md`** (no number is wrong):
  - The M5 row is labelled "MC-B's A6c slope", but the value is A6b's. It is a fixed label in the summary text.
  - The per-depth slope table is degenerate on A6b, with values in the millions. Within a depth, A6b pairs barely
    differ in ln truth. Do not read it.

### SFT B-v2 (Llama-3.2-3B, seed 0, R4-1 data)

- **The run:** 750 / 750 steps (2 epochs of 3,000 rows); train loss 0.155 (D-pos-v2: 0.138).
  - The train loop took 7,706 s, so 10.27 s / step with pauses and ~9.8 s / step without.
  - Peak reserved 4.48 GiB; nvidia-smi peak 5.43 GiB device-wide.
- **The fingerprint** is recorded in `final_adapter/physcal_fingerprint.json`. The adapter weights (110 MB) are on
  disk, untracked as usual.
- **The projections** (pauses included) all gave an end at 08:36–08:38, from 171 / 750 at 06:48 to 698 / 750 at 08:18.
  The run ended at 08:31.

### Temperature (D143 in its first full night)

| stage | pauses | paused | stage wall | share |
|---|---|---|---|---|
| eval D-pos-v2 | 3 | 91 s | 12,330 s | 0.7 % |
| MC-B A6b generation | 2 | 60 s | 1,982 s | 3.0 % |
| SFT B-v2 | 11 | 331 s | 7,964 s | 4.2 % |
| **night** | **16** | **482 s** | **22,276 s** | **2.2 %** |

- **Every pause** tripped at 87.1–87.9 °C and continued after 30–31 s at 60.3–64.4 °C. None needed the 10-min
  maximum.
- **Under SFT load** the GPU trips about every 10–15 min. Under eval load it tripped 3 times in 3.4 h.
- **For comparison, 28/29:** 46 % of the night was paused, 63 % of each SFT stage.

## 3. Problems and fixes

1. **The projection check's sleep held the queue lock after the chain ended. Fixed after the night (D145).**
   - When B-v2 ended, the queue killed the projection subshell, but its `sleep 900` (started 08:18:45) lived on. It
     inherited the lock's descriptor (fd 8), so `results/queue_xm2/queue.lock` stayed held until 08:33:45.
   - A queue started in that window would have logged "already running" and exited.
   - **The fix** (`scripts/queue_xm2.sh`):
     - the projection starts with fd 8 closed;
     - `stop_projection` freezes it, kills its sleep, then ends it.
   - **The test:** `test_stopping_the_projection_leaves_nothing_holding_the_queue_lock` passes. Against the old stop
     pattern the orphan survived, as it did in the queue.
2. **The guard's continue line repeated the reading** ("GPU at 64.40C <= 75C; GPU at 64.40C").
   - It now names one reading when the pause ended by cooling. It still adds the end reading when the pause hit the
     maximum or the stage ended.
   - `tests/test_temp_guard.py` checks both forms.
3. **The MC-B summary's M5 label on A6b** (§2). It is a display problem in analysis code, and it is not changed without
   you. Your decision 3.
4. **Full test suite after the night,** with the D145 fixes (08:40–08:51, GPU idle): **700 passed**, 0 failed, 0
   skipped (699 before the launch + 1 new).
5. **No stall, no BLOCKED, no retry, no free-space or memory issue.** The 28/29 freeze did not recur. Its cause is still
   unknown, and it stays an open risk (D140).

## 4. Decisions for you

1. **The go rule's NO-GO.**
   - D137 fixed the consequence: no seed expansion, and it is reported as a failed replication at one seed, with the
     same prominence as a success.
   - The honest reading is "propagation replicated on A6c; the A6b control is unmeasurable on Llama at this
     accuracy".
   - **If you want the control measured on Llama,** that needs a new pre-registered design under its own D-entry, for
     example a larger A6b or problems Llama computes correctly. It cannot turn this verdict into GO.
2. **B-v2's Llama eval and the pilot analysis (`xm_pilot`, D137's descriptive items)** were out of tonight's scope.
   - The eval is resumable (D142). At tonight's rate it takes ~3.4 h of GPU time.
   - The GPU is idle with ~4 h left before 13:00 today. Nothing has been launched.
   - Tell me if and when to queue it.
3. **The MC-B label fix:** correct the M5 row label to name the set it was computed on, then re-run the A6b analysis
   (CPU, ~20 s) so its summary.md is regenerated. The numbers do not change. Approve or skip.
4. **For your information, D145:** the two script fixes above were made after the chain ended, with tests. No generator,
   verifier or analysis code was touched.

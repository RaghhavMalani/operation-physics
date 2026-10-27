# Report: 2026-10-26/27 night: the R4-3 confirmatory run (D118), night 1 of 2

The previous report (the 2026-10-25/26 night: Round 4, D-pos-v2 / B-v2 and MC-A) is in git history:
`git show 4c40132:MORNING.md`.

**Short version.**
1. **Every stage that started finished, with exit 0.**
   - **SFT:** all four new adapters (D-pos-v2 and B-v2, seeds 1 and 2) trained 750 / 750 steps.
   - **Evals:** three of the six confirmatory evals exist: D-pos-v2 seed 0, and B-v2 seeds 0 and 1.
2. **The deadline gate skipped the other three evals, as planned.** D-pos-v2 seeds 1 and 2 (126 min each with slack)
   and B-v2 seed 2 (80 min) did not fit before 10:00. They run tonight.
3. **No results yet, by design (D118 §10).** The analysis runs once, after all six evals exist. Operational counts
   only:
   - 3,060 / 3,060 completions per eval, 0 truncated;
   - parse-fail 0–0.16%, no OOM backoffs;
   - every adapter passed the reload check, including B-v2 seed 1's (the first new adapter evaluated).
4. **The 23:00 waiter never fired.** The WSL VM booted at 23:36 IST, so it had stopped earlier and taken the waiter
   with it. I launched the queue by hand at 23:36 with your 10:00 deadline (D126; D127 corrects its reading).
5. **Two queue fixes for tonight (D127):**
   - if MC-B's generation is cut off by its timeout, its execution and analysis now wait instead of running on a
     partial set;
   - the dry-run tests no longer read the real results directory.
   **Full suite: 641 passed, 0 failed, 1 skipped** (the GPU test; the GPU was hidden).
6. **Tonight's chain is armed** (§5): the three remaining evals, MC-B, then D118's one-time analysis.
7. **Temperature:** three 10-min pauses at 88 °C, all during SFT, none during the evals.

---

## 1. What ran (IST)

| time | stage | result | commit |
|---|---|---|---|
| 23:36 | queue launched by hand (D126); chain PID 632; cap 5.9 GiB; 53 GiB free | – | `ed42926` |
| 23:37–00:58 | SFT D-pos-v2 seed 1 | 750 / 750, 81 min | `d2cfbf8` |
| 00:58–02:30 | SFT D-pos-v2 seed 2 (pause 01:46–01:56) | 750 / 750, 91 min | `71821b1` |
| 02:30–03:54 | SFT B-v2 seed 1 (pause 03:13–03:23) | 750 / 750, 84 min | `f8cb3bd` |
| 03:54–05:19 | SFT B-v2 seed 2 (pause 04:52–05:02) | 750 / 750, 84 min | `a3b0483` |
| 05:19–07:15 | eval D-pos-v2 seed 0, frozen R4-5 set | 3,060 / 3,060, 116 min | `03e9b63` |
| 07:15–08:14 | eval B-v2 seed 0 | 3,060 / 3,060, 59 min | `3f17d9f` |
| 08:14 | eval D-pos-v2 seed 1: SKIP (105 min left, needs 126) | – | – |
| 08:14–09:18 | eval B-v2 seed 1 | 3,060 / 3,060, 64 min | `452aedb` |
| 09:18 | evals D-pos-v2 seed 2 and B-v2 seed 2: SKIP (41 min left); analysis waits (3 of 6) | – | – |
| 09:20–09:35 | D127: two queue fixes, tests, this report | – | this commit |

The GPU has been idle since 09:18.

## 2. SFT, seeds 1 and 2 (D118 §2)

The data (R4-1), config and step count are those of seed 0; only the seed changes. Seed 0's rows are from the
2026-10-25 night, for comparison.

| arm | seed | steps | final train loss | peak reserved | s / step |
|---|---|---|---|---|---|
| D-pos-v2 | 0 | 750 / 750 | 0.1162 | – | 6.54 |
| D-pos-v2 | 1 | 750 / 750 | 0.1167 | 3.63 GiB | 6.32 |
| D-pos-v2 | 2 | 750 / 750 | 0.1165 | 3.54 GiB | 7.17 |
| B-v2 | 0 | 750 / 750 | 0.1189 | – | 6.26 |
| B-v2 | 1 | 750 / 750 | 0.1187 | 3.76 GiB | 6.59 |
| B-v2 | 2 | 750 / 750 | 0.1189 | 3.76 GiB | 6.58 |

- **Losses agree across seeds:** within 0.0005 of seed 0 in both arms.
- **Memory:** peaks of 3.5–3.8 GiB, well under the 5.9 GiB cap.
- **D-pos-v2 seed 2's** slower step time includes its 10-min temperature pause.

## 3. Confirmatory evals: operational counts only (D118 §10)

| eval | completions | truncated | parse-fail | OOM backoffs | mean completion tokens | eval time | reload check |
|---|---|---|---|---|---|---|---|
| D-pos-v2 seed 0 | 3,060 / 3,060 | 0 | 5 (0.16%) | 0 | 386 | 6,873 s | pass |
| B-v2 seed 0 | 3,060 / 3,060 | 0 | 3 (0.10%) | 0 | 241 | 3,501 s | pass |
| B-v2 seed 1 | 3,060 / 3,060 | 0 | 0 | 0 | 242 | 3,771 s | pass |

- **Settings:** 1,020 problems per set (A6c, A6, A6b on the frozen R4-5 set), greedy, 1,280 tokens, batch 16, prompt v5.
- **Not given until all six evals exist:** slopes, scores and coverage. I did not open the metrics, summaries or plots.
- **The plots:** the queue does not commit them (12 PNGs). They are committed with this report, unopened (D127).

## 4. BLOCKED / problems

- **Nothing was BLOCKED.**
- **The waiter (D126, corrected in D127):**
  - `uptime -s` (23:36 IST) is the VM's boot time, not a restart in the middle of the night. The VM was down before
    then, and the waiter died when it stopped.
  - **When and why are unknown:** some time after 07:42, from the app closing, sleep, or Windows stopping the VM.
  - **What it cost:** 36 min of the night. That is why D-pos-v2 seed 1's eval missed the gate.
- **Temperature:** three 10-min pauses at 88 °C (01:46, 03:13, 04:52), all during SFT, with the CPU idle. GPU-only SFT
  reaches 88 °C on its own. The evals stayed below the limit.
- **Memory and disk:** host VRAM was 1.16 GiB (cap 5.9 GiB); 53 GiB free on MLStorage throughout.
- **Tests after the night:** two dry-run tests failed because they read the real `results/stage_r4`, where the new
  adapters and evals now exist. Fixed by isolating them (D127). Full suite after the fixes: **641 passed, 0
  failed, 1 skipped** (the GPU test, with the GPU hidden).

## 5. Tonight: night 2 (armed)

- **The waiter: PID 67051**, armed 09:34 IST for 23:00 IST, with `GPU_DEADLINE=09:00 MC_B=1`. It execs
  `queue_r4_seeds.sh` under the same PID, so the chain's PID will also be 67051.
  - **To cancel:** `kill 67051` (or `kill $(cat results/queue_r4_seeds/waiter.pid)`).
- **The expected night, from 23:00:**
  - eval D-pos-v2 seed 1 to ~00:56, seed 2 to ~02:52, then B-v2 seed 2 to ~03:56;
  - MC-B generation on A6c, A6 and A6b, ~40 min each (estimated, not measured), to ~06:00;
  - MC-B execution and analysis on CPU;
  - then D118's confirmatory analysis (`r4_confirm`, CPU), which runs once all six evals exist.
- **If an MC-B generation overruns its 80-min timeout:** that set's analysis is BLOCKED and its generation resumes the
  next night (D127).
- **Tomorrow's report will have:**
  - the confirmatory verdict: P = K1 ∧ C2 ∧ C3 (D125), with R6 / R7 and the depth split;
  - MC-B's M1–M5 and its program failure rate.
- **The risk:** a VM stop kills the waiter, as it did yesterday.
  - **Keep running overnight:** the laptop (awake, on power) and this session.
  - **A more robust option, if you want it:** a Windows Task Scheduler task that runs `wsl.exe` at 23:00. It would
    boot the VM if it were down. I have not set this up; it is outside the repo.

## 6. Decisions to review

1. **D127's MC-B guard:** a partial generation is not analyzed; it resumes the following night. The alternative is to
   analyze the subset labelled "incomplete".
2. **D127's correction of D126:** the VM was down before 23:36; it did not restart mid-night.
3. **The Task Scheduler option** (§5): only if you want the waiter to survive a VM stop.

## 7. Recommendation

- **Let tonight's chain run as armed.** It completes D118 (six evals and the one-time analysis) and runs MC-B.
- **Nothing in night 1's operational counts argues for a change.** No truncation, near-zero parse-fail, no OOM, and
  every reload check passed.

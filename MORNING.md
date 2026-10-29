# Report: 2026-10-28/29 night: the cross-model pilot on Llama-3.2-3B (D137–D139)

Written on 2026-10-30 at ~01:40 IST, after the fact, from the committed logs (`results/queue_xm/chain.log`,
`results/queue_xm/followup/chain.log`, the stage logs) and the Windows System log. The session that watched the night
ended with the laptop, so no report was written then. The previous report (night 2 of D118) is in git history:
`git show be0d338:MORNING.md`.

**Short version.**
1. **SFT D-pos-v2 on Llama, seed 0, finished:** 750 / 750 steps, exit 0, R2-1 fingerprint recorded (`fd8d1fa`).
2. **Its eval on the frozen A6c + A6b timed out and saved nothing.**
   - It ran 00:51 to 05:51, hit its 5-hour timeout (exit 124) at 1,926 / 2,040 problems, and left no file
     (`889c72d`).
   - The cause is structural: `run_eval` writes completions only after the last batch. D139 flagged this as a risk.
3. **So the go rule (D139) could not be evaluated.** `xm_go` exited 1 ("no metrics.json"), so no seed-1 stage was added
   (`ac78c5f`). There is no GO / NO-GO verdict.
4. **SFT B-v2 on Llama, seed 0, is incomplete.**
   - It started at 05:51 and reached step 427 / 750 by ~09:00.
   - At 09:00:50 the temperature guard paused it. After that, nothing in WSL wrote another line.
   - Windows kept logging normally until a `shutdown.exe` shutdown at 09:54:29 IST, with no GPU driver reset.
   - It left no adapter or checkpoint. Its empty directory is set aside in `results/discarded/` (D140).
5. **Temperature:**
   - **The whole night:** 40 pauses of 10 min each, 390 of the 854 min from the first stage to the freeze (46%).
   - **The two SFT stages:** each was paused 63% of its time.
   - **The eval:** paused 17% of its time.
6. **Free space:** 47 GiB on MLStorage all night. **Every push landed;** `main` = `origin/main` = `ac78c5f` this morning.
7. **Nothing was damaged** (D140): `git fsck` is clean, and the integrity check passed on all 64 files written since
   28 Oct 18:00.

---

## 1. What ran (IST)

| time | stage | result | duration | commit |
|---|---|---|---|---|
| 18:46 | chain 10129 launched: `GPU_DEADLINE=10:00`, launch cap 5.9 GiB (host 1,299 MiB), 47 GiB free | – | – | – |
| 18:46–00:51 | SFT D-pos-v2 (Llama, seed 0) | 750 / 750, exit 0; loss 0.138; fingerprint recorded | 365 min (23 pauses, 230 min) | `fd8d1fa` |
| 00:13 | follow-up queue 46728 started beside the chain (D139) | – | – | `9296d7b` |
| 00:51–05:51 | eval D-pos-v2 on frozen A6c + A6b (batch 6, reload check passed) | **exit 124 (timeout) at 1,926 / 2,040; nothing saved** | 300 min (5 pauses, 50 min) | `889c72d` |
| 05:51 | go rule `xm_go` (CPU) | exit 1: not evaluable (no metrics.json); nothing added | 3 s | `ac78c5f` |
| 05:51–~09:00 | SFT B-v2 (Llama, seed 0) | step 427 / 750, then frozen | ~189 min (12 pauses, ≥ 110 min) | – |
| 09:00:50 | guard pause #12 of B-v2 | no further line from any WSL process | – | – |
| 09:54:29 | Windows `shutdown.exe` on your account (Kernel-Boot: "last shutdown's success status was true") | – | – | – |

## 2. Numbers

- **SFT D-pos-v2 (Llama):**
  - 750 steps in 21,499 s of wall time. That is 28.7 s / step with pauses, and ~9.5 s / step without them.
  - Train loss 0.138; peak reserved 4.64 GiB under the 5.9 GiB cap.
  - The fingerprint recorded at sampling batch 4 (D138).
- **Eval D-pos-v2 (Llama):**
  - 1,926 problems in 17,597 s, five pauses included. That is ~7.6 s / problem without the pauses.
  - 114 problems were left at the timeout, ~15 min of work.
  - The R2-1 reload check passed.
- **SFT B-v2 (Llama):** about 9.6 s / step without pauses (427 steps in 3 h 08 min, 12 pauses). It is incomplete.
- **The go rule:** not evaluated, so there is no verdict.
- **MC-B A6b:** not reached. The chain never got past B-v2's SFT.
- **The pilot analysis (`xm_pilot`):** not reached; it needs both Llama evals.

## 3. Problems

1. **An eval that is cut off loses everything.** This cost 5 GPU hours and is fixed before tonight. Evals now write
   every batch to disk and resume (D142, written next, with GPU byte-identity checks).
2. **The temperature guard's fixed 10-min pause.**
   - After each pause the GPU was back at 53–64 °C, far below the 87 °C trip point. So most of each pause was idle
     time.
   - SFT stages ran at about 1/3 duty.
   - Tonight the guard resumes at ≤ 75 °C, checked every 30 s, with 10 min as the maximum pause (D143, written next).
3. **A freeze with no driver reset.**
   - Every WSL process stopped writing at 09:00:50, while the stage was SIGSTOPped by the guard.
   - Nothing inside WSL can detect a freeze of WSL itself. The stall guard also treats a stopped stage as paused.
   - Recorded as an open risk; there is no fix. A watcher outside WSL (Windows side) would be needed to catch it.

## 4. Decisions for you

- **None from this night alone.** Your recovery instructions for the night of 2026-10-29/30 start at D140 (this
  recovery) and D141 (family H not run). Tonight's changes follow as D142 onward.

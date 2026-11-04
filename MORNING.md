# Report: night of 2026-11-03/04, the R6 queue (D167–D168; paper #2's exploratory track, not paper #1)

Written on 2026-11-04 at ~07:55 IST. The GPU work had ended at 06:10, and the CPU analysis was running.
- **Sources:**
  - the plan: `docs/exploratory/DECISIONS_FOLLOWUP.md` D167, pre-registered before any code;
  - the launch: D168;
  - the queue logs: `results/queue_followup_r6/`;
  - the outputs: `results/followup/r6/`.
- **The previous report** (the night of 2026-10-30/31) is in git history: `git show 4f217c2:MORNING.md`.
- **D148's freeze held.** Nothing tonight touched `paper/`, FINDINGS.md or the frozen sets.

**Short version.**
1. **All 18 GPU stages finished, by 06:10 IST**, 2 h 50 min before the 09:00 deadline.
   - **Time:** 6.39 h of GPU time, against the 6.92 h estimate.
   - **Pauses:** 65 temperature pauses, 33 min in all.
   - **Every stage** exited 0 and was committed and pushed.
2. **Every plumbing check on real data passed, exactly:**
   - **R6-3's spot checks:** the re-extracted own-answer states are bitwise D160's (max gap 0.0) for the 1.5B and the
     7B. D162's stored states are reused as planned.
   - **R6-4's anchor:** all 2,722 native-trace states from the native 1.5B are bitwise D160's.
   - **R6-4's same-text gate:** every reader (native, SFT, random weights) read byte-identical texts in both
     presentations (2,722 native, 2,766 terse).
   - **P(yes):** complete (4,369 + 4,683 bank answers, 466 truncated × 2 readers, no missing values). Every logit check
     passes (relative gap 0.002–0.004).
   - **p1 and p2** fall on one token for 1.2% of native items (counted, as D167 asks).
3. **The CPU analysis started at 06:10:** 4 jobs × 1 BLAS thread, nice 19, the GPU hidden. **No result exists yet, and
   none has been read.**
   - **By 07:49:**
     - R6-3's bank text classifier is done (1,300 s, cached);
     - the 7B reader's bank probe is running (~2.4 h).
   - **The estimate** for the whole analysis is ~10 h, so ~16:00 IST.
   - **What is kept:** every fit is cached and `results.json` is rewritten after each experiment (D168), so a failure
     keeps the finished parts.
   - **When it finishes,** the queue commits `results.json`.

## 1. The GPU stages

| stage (set, reader) | wall | pauses | estimate |
|---|---|---|---|
| spot, 1.5B / 7B | 26 s / 44 s | 0 / 0 | 0.03 / 0.04 h |
| R6-3: 7B's bank read by the 1.5B (states, P(yes)) | 1,069 s, 1,094 s | 8, 2 | 0.31 h, 0.34 h |
| R6-3: 1.5B's bank read by the 7B (states, P(yes)) | 3,528 s, 4,264 s | 4, 11 | 1.03 h, 1.19 h |
| R6-3: truncated, P(yes), 1.5B / 7B | 285 s / 1,122 s | 0 / 4 | 0.12 / 0.35 h |
| R6-4: 6 stages (3 readers × 2 presentations) | 473–569 s each, 3,225 s in all | 12 | 0.20 h each |
| R6-5: 1.5B answer only / swap | 443 s / 1,207 s | 2 / 7 | 0.12 / 0.39 h |
| R6-5: 7B answer only / swap | 1,685 s / 5,043 s | 4 / 11 | 0.35 / 1.45 h |

- **The estimate held** except for the 7B's answer-only stage (+34%). Short contexts cost more per token than the
  linear rate assumed.
- **The states take 7.5 GB** (estimated 7.9), inside the disk image. MLStorage is unchanged at 21.6 GiB free.

## 2. Decisions made tonight (logged in D168; please review)

1. **The 7B gate waits until the deadline** (`CAP_WAIT_S=21600`) instead of 60 min, as D159 did.
   - **Why:** Windows held ~1.05 GiB of VRAM at launch (cap 6.4 GiB, under the 7B's 6.5).
   - **Not needed in the end:** by the 7B stages Windows held ~0.75 GiB, and the gate opened at once (6.7–7.0 GiB).
2. **A restart between stages, at 23:40** (no stage running; the chain went from PID 503 to PID 2366).
   - **What changed:** `HOST_TRIES=1`, so each stage reads the cap once.
   - **Why:** with 5 tries, every stage spent up to 4 min re-reading while the cap was under 7.5 GiB, ~70 idle minutes
     over the night.
   - **Unchanged:** the cap formula and the VRAM gate.
3. **The analysis got a cache and per-experiment saves** before the GPU stages it does not use had finished.
   - The GPU-stage functions are byte-identical (checked by diff), and the module was swapped in by an atomic rename.
   - The tests: 18 of 18 pass.

## 3. Problems, and the state

- **No stage failed, stalled or was retried.**
- **Temperature:** the 1.5B's first stage paused every 1–4 min for ~20 min after launch (23:42–23:51). After that,
  2–11 pauses per stage. The guard's 30 s pauses handled it.
- **My session watcher** raised one false "silent" alarm on the CPU analysis, a long fit that prints only when done.
  The watcher now ignores the analysis.
- **The state now:**
  - the GPU is free;
  - the queue chain (PID 2366) is alive, running the analysis, and exits when it finishes;
  - nothing else runs.

## 4. Recommendation

- **Let the analysis finish (~16:00 IST).** Until then, no GPU job and no multi-core CPU job: the probes' numbers depend
  on the thread settings (D97).
- **Then:** the write-up (`docs/exploratory/followup_feasibility.md`) and a D-entry with D167's readings. R6-6 is
  decided after those (D165).

# Report: night of 2026-11-04/05, R6-3 to R6-5's analysis and readings (D167–D172; paper #2's exploratory track, not paper #1)

Written on 2026-11-05 at ~02:15 IST.
- **Sources:**
  - the readings: `docs/exploratory/DECISIONS_FOLLOWUP.md` D172, under D167's pre-registration;
  - the write-up: `docs/exploratory/followup_feasibility.md`, "Paper #2's next experiments";
  - the outputs: `results/followup/r6/`;
  - the logs: `results/queue_followup_r6/`.
- **The previous report** (the night of 2026-11-03/04) is in git history: `git show bb14f8f:MORNING.md`.
- **D148's freeze held.** Nothing touched `paper/`, FINDINGS.md or the frozen sets.
- **Nothing of R6-6 was built or run,** on Kaggle or locally.

**Short version.**
1. **The analysis finished at 01:49 IST (exit 0).** All three experiments are saved, committed and pushed, and each has
   its D167 reading below. All of it is exploratory and uncorrected for multiplicity.
2. **R6-3:** on identical answers, the 7B's verdict beats the 1.5B's by +0.094, and its probe by a small +0.022.
3. **R6-4:** the primary test reads **"neither factor is shown"**. It could not resolve presentation: every readout on
   its 93 problems sits at 0.54–0.59, random weights included.
4. **R6-5:** the probe beats a rich text reader at both sizes, but by ~0.06, not D164's ~0.09. The reasoning adds signal
   beyond the answer. The swap splits: "neither" at 7B, "it tracks the reasoning" at 1.5B.
5. **R6-6's premise stands, and the go decision is yours** (§5).

## 1. The readings

| experiment | quantity | result | D167's reading |
|---|---|---|---|
| R6-3 | P(yes), reader 7B − 1.5B | +0.094 [+0.075, +0.112] | "the verdict's size effect holds on identical inputs" |
| R6-3 | probe, reader 7B − 1.5B | +0.022 [+0.011, +0.034] | "the 7B's states carry more" |
| R6-4 | weights W | +0.001 [−0.048, +0.052] | "neither factor is shown" |
| R6-4 | presentation P | −0.029 [−0.148, +0.094] | (the same reading) |
| R6-4 | interaction | −0.047 [−0.138, +0.043] | no interaction shown |
| R6-5 (a) | probe − rich text, 7B; 1.5B | +0.054 [+0.035, +0.072]; +0.059 [+0.040, +0.079] | "the probe reads more than a rich text reader", both |
| R6-5 (c) | full − answer only, 7B; 1.5B | +0.098 [+0.074, +0.122]; +0.111 [+0.088, +0.135] | "the reasoning carries the signal", both |
| R6-5 (d) | swapped accuracy, 7B | 0.480 [0.453, 0.506] | neither |
| R6-5 (d) | swapped accuracy, 1.5B | 0.391 [0.364, 0.420] | "it tracks the reasoning" |

R6-5 (a) and (c) read the same on length-matched pairs. R6-5 (b), the random-weights probe beside them: 0.664 / 0.673.

## 2. What to be careful with

- **R6-3's probe difference is small and less robust than the verdict's.** Its CI contains 0 on length-matched pairs
  (+0.011 [−0.011, +0.032]) and on the 7B's answers alone (+0.015 [−0.002, +0.032]).
- **R6-4's big numbers are a secondary.**
  - **The large presentation effect** (P −0.238 [−0.276, −0.199], full banks) mixes presentation with which answers each
    bank holds. The code labels it "the presentation factor dominates", but that is not R6-4's reading (D172).
  - **The clean part of the full banks is W:** −0.001 [−0.016, +0.013], on identical text. The fine-tuned weights read
    native traces as well as the native weights do (0.770 and 0.765). Terse completions give every reader 0.52–0.53.
- **R6-4's primary set is selected.** An incorrect candidate enters only if both models reached the same wrong value.
- **The swap may partly measure the mismatch itself.** In both models the probe falls for correct and for incorrect
  boxes (mean change −0.52 / −0.26 at 7B, −0.43 / −0.14 at 1.5B). A pure reasoning-follower would raise the incorrect
  ones. This is my interpretation, not a registered reading.
- **The answer-only probe's 0.65 cannot be attributed yet.** It has no random-weights reference.
- **The rich text reader is a lower bound on the surface.** It already took the margin from ~0.09 to ~0.06.
- **New and not asked for:** the 7B's verdict does not recognize unfinished answers. It rates a truncated sample like a
  correct one (log-odds −1.49 and −1.48), and above an incorrect one in 79% of pairs. The 1.5B's verdict does separate
  them (0.93 and 0.88).

## 3. What ran

- **The resume:** D169's script, unchanged, from 23:08:29 to 01:49:34 IST: 2 h 41 min, against D169's 4–5 h. Same thread
  settings; R6-3 and R6-4 rebuilt from the 15 cached fits in 34 s.
- **The new fits:**

| fit | 7B | 1.5B |
|---|---|---|
| rich text classifier | 1,515 s | 1,209 s |
| text on problem + answer | 426 s | 360 s |
| answer-only probe | 4,537 s | 1,448 s |

- **The checks, all passed:**
  - **Before:** 15 of 15 fits hash-match the states on disk (D170).
  - **After:** R6-3 and R6-4 are identical to the saved copy, and no value is non-finite.
  - **The swap's refit gate:** max |Δp| = 0.0 for both models.
- **Commits:** `4d1ae87` (R6-5's 7B half, when it was saved), `bb14f8f` (the whole analysis), then the write-up. Each was
  checked against the remote.
- **The machine:** at each of my checks the GPU sat idle at 54–59 °C, with 9 GB of RAM or more available. The watcher
  raised no stall, clock jump or temperature alarm.

## 4. Decisions made tonight (logged in D172; please review)

1. **I added a pre-registered secondary after the fact.**
   - **What happened:** D167 lists pair weighting for R6-4, the analysis never computed it, and I only noticed while
     reading the results.
   - **What I did:** computed it from the cached fits, into its own file (`r6_4_pair_weighted.json`), so `results.json`
     stays as the registered run left it. It must first give back the saved cells exactly, and it does.
   - **Result:** no reading changes (primary set: W −0.004, P −0.048, "neither factor is shown").
2. **I did not compute length-matched pairs on R6-4's 93-problem set.** D167 names no set, and the analysis used the full
   banks. On the matched set a pair would have to match in both presentations, leaving few of the 248 pairs.
3. **I treated the code's full-bank label as a description, not a reading,** because D167 fixed the table for the primary
   set. If you read D167 differently, the full banks would read "the presentation factor dominates".
4. **I committed R6-5's 7B half by hand,** since the resume script commits only at its end.

## 5. Problems, and the state

- **BLOCKED:** nothing.
- **No analysis failure.** D170's fix held: R6-5 ran through.
- **My own slips, with no effect on results:**
  - Sourcing the queue library from a one-off shell changed directory and the first commit attempt failed, with nothing
    staged or written. The commit then went through under the same lock.
  - I briefly told you the 1.5B reader was poor on truncated samples. It is the 7B. The write-up has it right.
- **Tests:** the full suite passes: 814 passed, 1 skipped, in 623 s. It covers D170's fix and this report's two new tests.
  A first run was cut off by my own 580 s limit, with no result.
- **The state:** nothing is running, and the GPU and CPU are free. Your untracked `PHASES_10-15.md` is untouched.

## 6. Recommendation

- **R6-6a: go on to its decisions, not yet to a run.** The premise stands: the probe (0.75–0.78) reads more than the
  model's own verdict (0.60–0.69) and more than the rich text reader. The next step is your six decisions in
  `docs/exploratory/r6_6_kaggle_plan.md` §1, then its pre-registration.
- **Keep the text-distilled arm (D).** With a margin of ~0.06 over surface, a student's gain cannot be read without it.
- **The readings against the plan's §5:**
  - **C − D is the headline contrast** (R6-5 (a)).
  - **Add the swap to S5:** the student is the native 1.5B, which reads "it tracks the reasoning".
  - **A 7B student becomes an option,** costed separately at likely 25 h or more (R6-3).
  - **R6-4's "neither" has no row in §5.** W ≈ 0 gives no reason for an SFT second student, so I would keep the plan as
    it is.
- **Two cheap controls would close the open interpretations, if you want them first:**
  - a same-label swap: an incorrect box after a different incorrect reasoning keeps the mismatch without changing the
    label, which isolates the mismatch's cost. ~20 min of GPU at 1.5B;
  - a random-weights answer-only probe at 1.5B: ~7 min of GPU and ~25 min of CPU.
  - Both times are from R6-5's measured 1.5B stages. Neither control is registered, so each needs a D-entry before its
    data. Both reuse local 4-bit states, so they would be an exception to D171's Kaggle-first rule.

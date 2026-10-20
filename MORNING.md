# Report: 2026-10-20 day: tests, X4 pre-registered and built, tonight's queue prepared (CPU only)

The previous report (the 2026-10-19/20 night: X2 and the X3 pilot) is in git history: `git show c333885:MORNING.md`.

**Short version.**
1. **Full test suite: 457 passed, nothing to fix.** That is the by-day rerun the night report asked for.
2. **X4 is pre-registered (D89)** before any X4 data.
   - **The task:** the 4-bit SFT model sees its own completed X2 answer, is asked a fixed yes/no question, and may
     recompute first.
   - **Confidence:** the yes-vs-no log-odds at a canonical `Verdict:` position.
   - **Scoring:** on exactly X2's 5,390 pairs.
3. **X4 is built and tested on CPU.**
   - **Tests:** 14 new, including a tiny model end to end. X4 + X2 tests: 24 pass; the full suite: 470.
   - **The X2 probe reference** is recomputed from X2's own outputs and reproduces X2's 0.5471 exactly.
4. **Tonight's queue is prepared, NOT launched (D90):** `sft` → `shuffled` → `random_init` → the analysis.
   - **Estimate:** 6.1–8.1 GPU-hours, so launch by about 23:00 for all three arms to finish before 09:00.
5. **Four [REVIEW] items in D89 can still change before launch**, because no X4 data exists (§4).

---

## 1. What ran (IST, CPU only; the GPU was not used today)

| step | result | commit |
|---|---|---|
| full test suite | 457 passed, 0 failed, 0 skipped (5 min 13 s) | `4e49eea` |
| X4 pre-registration (D89) + the fixed prompt `x4_v1` | committed before any X4 code ran on data | `eddda39` |
| X4 build: runs, analysis, config, tests, the X2 reference | 14 tests pass; the reference gap is 0.0 | `3b8057e` |
| tonight's queue (D90) | dry-run checked; not launched | `89262ce` |

## 2. X4 as pre-registered (D89; details in DECISIONS.md)

- **The conversation (one per answer, 4,116 answers):**
  1. the v4 system prompt;
  2. the problem;
  3. the answer's own sampled completion, as the assistant turn;
  4. then, as a second user turn:
     > Now check your solution above. Is the final value in your <answer> correct to within 1%? You may redo the
     > calculation before you decide. Then end your reply with one line, either "Verdict: yes" (the value is correct)
     > or "Verdict: no" (it is not).
- **Recomputation:** greedy, up to 512 tokens.
- **The read-out:**
  - **The position:** the text is cut before its last "verdict" and `Verdict:` is appended. So every answer is read
    at the same position, whether the model wrote a verdict or not.
  - **The confidence:** the log-odds of the {yes, Yes, …} tokens vs the {no, No, …} tokens. It is computed from
    float32 logits: lm_head is applied in float32 to the final hidden state, because bf16 logits would create spurious
    ties within a problem.
  - **The check:** the float32 logits are checked against the model's own logits on the first row of every run.
- **Arms:**
  - **`sft`:** primary.
  - **`shuffled`:** the same model, but each answer is shown under the text of the next kept problem in the same
    (family, depth) cell.
  - **`random_init`:** X2's random-init model.
- **Primary metric:** X2's within-problem pairwise accuracy on X2's pairs.
  - **Holm** (α 0.05) over the three arms' "> 0.5" tests.
  - **Paired CIs:** `sft − X2 probe`, `sft − random_init`, `sft − shuffled`.
- **Secondary:** AUROC across problems; per depth; the raw P(yes); diagnostics; recomputation agreement (exploratory).
- **Reading rules:**
  - **"Succeeds":** `sft` is Holm-supported and beats random init.
  - **"Problem-specific":** `sft` beats `shuffled`.
  - **"Above the probe":** `sft` beats the X2 probe.
  - **The labels:** five outcome labels, fixed now.

## 3. Tonight's queue (D90)

- **Launch:** `nohup scripts/queue_x4.sh > results/queue_x4/nohup.log 2>&1 &`. The 09:00 deadline is taken as the
  next morning's.
- **The VRAM cap is automatic:** before each GPU stage the queue reads `nvidia-smi` and sets total − used − 0.5 GiB,
  clipped to [4, 7.5]. A host process holding VRAM (D86) is therefore handled without an edit. This morning's
  reading: 7.2–7.35 GiB.
- **Time:** soft limits keep 4 h for the two controls after `sft`, and 2 h for `random_init` after `shuffled`.
  - **What a stopped arm leaves:** a stratified prefix, which D89 analyzes as such. Rerunning the same command
    resumes it.
- **Estimates (from last night's measurements; nothing of X4 has run on the GPU):**
  - `sft` / `shuffled`: 1.7–2.7 h each;
  - `random_init`: about 2.7 h.
- **Not checked:** any GPU stage. If the float32-head check fails, an arm uses the model's own logits (slower, and
  recorded); it does not stop.

## 4. Decisions to review before launch

1. **[REVIEW] The prompt `x4_v1`'s wording.** Changing it means a new hash in the config and in D89.
2. **[REVIEW] The confidence:** the yes-vs-no log-odds (= P(yes) renormalized over yes + no), with raw P(yes) as
   secondary.
   - **Why not raw P(yes) as primary:** mass on other tokens (formatting) would then count as "no".
3. **[REVIEW] The donor rule** for `shuffled`: the next kept problem in the same cell, so the mismatch is in the
   numbers, not the format.
4. **[REVIEW] The control order:** `shuffled` before `random_init`. The shuffled control is the more informative one,
   and `random_init` is the most expensive, since it never stops before 512 tokens.
5. **D90's runtime detail:** after an OOM backoff, X4 keeps the halved batch for the rest of the run.

## 5. Housekeeping

- **Not committed:** `PHASES_10-15.md` is still untracked, and X2's hidden-state arrays are still local. X4's
  analysis does not need them; it uses the committed reference.
- **The full suite now has 471 tests:** 470 ran in the full pass, plus the OOM-halving test added after it, which
  passes with the X4/X2 tests.

## 6. Recommendation

- **Review the four [REVIEW] items, then launch by about 23:00.**
- **What X4 adds to X2:** X2 showed that no passive readout ranks answers within a problem. X4 asks whether active
  checking, with room to recompute, does.
  - **If `sft` beats `shuffled`:** the model can check an answer against its problem when asked. That bears on how
    the Phase 12 c channel is designed.
  - **If it does not:** the negative result of FINDINGS #19 extends to self-verification, which strengthens the R2-3
    framing.

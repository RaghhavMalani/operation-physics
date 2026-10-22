# Report: 2026-10-21/22 night: E1d on the X2 pairs, Phase 12 stage B (3 seeds), verifier 1.3.0, the stage C pilot built

The previous report (2026-10-21 day) is in git history: `git show 97fba4d:MORNING.md`.

**Short version.**
1. **The GPU was busy from 23:14 to 06:19 IST** (deadline 09:00). Every stage exited 0; nothing is BLOCKED.
   - **Stage (a)** ran first (launched 4 min after your brief).
   - **Stage B** then ran all 3 seeds back to back: 750 / 750 steps each, plus all evals.
   - **One temperature pause:** 10 min at 00:37, from GPU load alone.
   - **The stage C pilot was skipped by your 3.5 h rule** (160 min left at 06:19). It is built, CPU-tested and
     marked ready.
2. **(a) E1d does not rank answers within problems (FINDINGS #26).** On the X2 pairs E1d scores 0.512 [0.483, 0.541];
   T1–T3 are not supported.
   - **It is X3's student, to the third decimal:** 0.513 within problems; ρ 0.350 vs 0.346 and AUROC 0.679 vs 0.678
     across answers.
   - **Self-consistency (0.643) stays the only within-problem signal.**
3. **(b) Stage B: SFT on σ\* itself does not teach propagation (FINDINGS #27).** No B test survives Holm.
   - **A6c:** +0.04.
   - **A6:** +0.45 on ~58 pairs per seed, p 0.037, labelled "partial" from its 90% CI.
   - **What the model does instead:** it reads the stated percentages (ρ 0.72 with RSS), states σ ≈ 0.65·σ\*, and its
     expected score is below both RSS and X1.
   - **So** the "ceiling" is a surface reader, and stage C's bar is X1 / RSS (D104's rule).
4. **Verifier 1.3.0 is built (D105):** σ reward = expected crps_log against the predictive distribution, centred at
   f(x_meas).
   - **Nothing existing was edited:** it is a new module and a new config.
   - **Old tapes** replay exactly through it. 17 tests.
5. **Decisions:** D102–D107. To review (§6): stage B's base and budget, stage C's base and prompt, `weights.sigma`, and
   the pending one-line dispatch edits in `reward.py` / `tape.py`.
6. **Tests:** 553 passed, 0 failed, 0 skipped (GPU visible, after the queue); 34 are new.

---

## 1. What ran (IST)

| time | step | result | commit |
|---|---|---|---|
| 23:14 | D102 (stage a) pre-registered, queue launched | – | `2970d07` |
| 23:14–23:32 | GPU: E1d σ suffixes on X2's 4,116 answers | 4,116 / 4,116, no OOM | `8e3fb3d` |
| 23:32–23:46 | GPU: the SFT model's own σ suffixes (D102's no-RL control) | 4,116 / 4,116 | (queue) |
| 23:25 | D103 (your decisions), D104 (stage B) pre-registered; stage B built (8 CPU tests); chain launched on (a)'s PID | – | `dcd9b12`, `2407cb4`, `bfd651a` |
| 23:47–23:49 | CPU: (a)'s analysis | FINDINGS #26 | (queue), `48da91a` |
| 23:47–01:08 | GPU: stage B seed 0 SFT | 750 steps, 81 min (incl. a 10-min pause at 00:37, 88 °C) | (queue) |
| 01:08–02:02 | GPU: seed 0 eval (A6 / A6b / A6c) | 3,060 / 3,060; fingerprint check passes | (queue) |
| 02:02–04:10 | GPU: seed 1 SFT (73 min) + eval (54 min) | as seed 0 | (queue) |
| 04:11–06:18 | GPU: seed 2 SFT (73 min) + eval (54 min) | as seed 0 | (queue) |
| 06:19 | CPU: stage B analysis (17 s) | FINDINGS #27 | (queue), `77ebfe7` |
| 06:19 | stage C pilot | SKIP: 160 min left, your rule needs 210 | – |
| ~23:35–00:00 | CPU: verifier 1.3.0 (D105), 17 tests | – | `1c84e9a` |
| ~23:40–00:10 | CPU: stage C pre-registered (D106), built, 6 tests, marked ready | – | `34738a1`, `49f70bc` |
| night | D107 (implementation decisions), PROGRESS | – | `647f73a`, `9dd9038` |

## 2. (a) E1d on the X2 pairs (D102; FINDINGS #26)

| source (same 516 problems, 4,116 answers) | within problems [95% CI] | separated pairs | ρ(σ, error) across answers | AUROC (correct) |
|---|---|---|---|---|
| SFT stated σ (X2 method 1) | 0.499 [0.474, 0.524] | 0.501 | −0.001 | 0.490 |
| SFT greedy σ suffix (no-RL control) | 0.489 [0.461, 0.515] | 0.461 | +0.024 | 0.499 |
| X3 student (a) | 0.513 [0.484, 0.543] | 0.525 | +0.346 | 0.678 |
| **E1d (lr 5e-5)** | **0.512 [0.483, 0.541]** | 0.502 | **+0.350** | **0.679** |
| self-consistency (k = 8) | 0.643 [0.616, 0.671] | 0.861 | +0.672 | 0.823 |

- **T1** (E1d > 0.5): p 0.21. **T2** (− SFT σ): +0.013 [−0.021, +0.048]. **T3** (− X3): −0.001 [−0.032, +0.031].
  Holm: none supported.
- **Reading (D102):** E1d's ρ is between-problem ranking only. RL at 5e-5 moved σ from no ranking to exactly the X3
  kind, not beyond.
- **Sanity:** 21 / 4,116 failed suffixes; not collapsed (top suffix 2.1%). Every comparator reproduced its published
  number exactly.

## 3. (b) Phase 12 stage B, the supervised σ\* "ceiling" (D104; FINDINGS #27)

| test (seed mean of 3) | estimate [95% CI] | p | Holm |
|---|---|---|---|
| B1: A6 slope > 0 | +0.449 [−0.049, +0.892] (seeds +0.18 / +0.17 / +1.00) | 0.037 | not supported |
| B2: A6c slope > 0 | +0.044 [−0.066, +0.157] | 0.21 | not supported |
| B3: expected crps_log − RSS | −0.002 [−0.003, −0.001] | 1 | not supported |
| B4: expected crps_log − X1 | −0.003 [−0.004, −0.002] | 1 | not supported |
| B5: A6 slope − X1's | +0.371 [−0.141, +0.855] | 0.072 | not supported |

- **The labels (reported):**
  - **A6:** "partial propagation" on the seed mean (90% CI [+0.04, +0.81]), but from ~58 computation-correct pairs per
    seed (A1's τ = 0.1%; only ~15% of answers qualify). Seed 2 reads "full", seeds 0 and 1 "no propagation".
  - **A6b:** equivalent to 0.
  - **A6c:** ≈ 0.
- **Oracle-centred:**
  - **Expected crps_log:** −0.041 to −0.042 per seed, against RSS −0.039, X1 −0.039 and σ\* −0.037.
  - **Calibration:** median ln(σ / σ\*) is −0.31 to −0.45; 1σ coverage ~50%.
- **Exploratory, computed after the analysis:**
  - **It tracks the stated percentages:** σ correlates with RSS (ρ 0.72) more than with σ\* (0.64); its level sits
    between "max" and RSS.
  - **The ~35% underestimate is the same at every depth and noise level.**
- **What it means:**
  - **SFT imitation of a `.2g` σ\* with token cross-entropy, 2 epochs, does not yield propagation** on this base. It
    yields an overconfident surface reader.
  - **Under D104's rule,** stage C is read against RSS / X1 and its own SFT control, not against stage B.

## 4. Verifier 1.3.0 (D105)

- **score_sigma** = the mean over 4,000 fixed predictive draws of crps_log(N(ln f(x_meas), σ_rel²), draw).
  - It depends only on the stated σ_rel and the problem, never on μ's error or one realized truth.
  - It peaks at the predictive's CRPS-optimal scale (tested).
- **Invalid draws are dropped** (24 of stage A's 3,060 problems had some); below 99% valid, a Gaussian fallback is used.
  Exact givens → σ = 0 is optimal.
- **`weights.sigma = 0` [REVIEW]:** the 1.3.0 total equals 1.2.0's; score_sigma sits in the breakdown for σ-only
  training (stage C), as c did in D83.
- **Not edited:** `reward.py`, `tape.py`, `noise.py` (tonight's stages imported them).
  - **Until then,** 1.3.0 must be scored and replayed through `physcal.verify.sigma_reward`.
  - **Pending (daytime):** the version dispatch in `reward.reward` / `tape.replay`.
- **Replay:** both committed 1.1.0 tapes replay through the new entry point with identical rewards and breakdowns, and
  so does a 1.3.0 tape.

## 5. (c) The stage C pilot (D106): built, tested, not run

- **What it is:** E1d's exact settings (merged SFT base, fresh LoRA, lr 5e-5, 300 steps, σ-only suffixes), with the
  bundle D106 lists:
  - prompt v5;
  - D80's implicit-arm prefixes (answer = f(x_meas)), 2,250 noisy training prefixes;
  - training reward = score_sigma (the floor for a failed or μ-moving suffix);
  - D103's 7.1 GiB cap and OOM check.
- **Evals:** σ suffixes on 3,060 A6 / A6b / A6c prefixes for the final adapter, step 150 and the SFT-only control.
- **Tests:** C1–C5 (stage B's code) + C6 (RL − control), Holm over six.
- **Amendment before any rollout (D107.6):** `max_prompt_length` 1,536 → 2,016. v5 prompts reach 1,669 tokens; nothing
  is truncated.
- **Run it:** `GPU_DEADLINE=HH:MM bash scripts/queue_stage_c_pilot.sh` (~2.7 h: training ~2.1 h at E1d's speed, three
  evals of ~12 min). Its GPU path has not been exercised.

## 6. Decisions to review

1. **D104 [REVIEW], stage B's base and budget.** I used D70 / D78's pinned base and 2 epochs.
   - **After FINDINGS #27,** the open question is the loss rather than the base: token cross-entropy on a `.2g` number
     (D80's own [REVIEW]).
   - **A real ceiling test is probably R3-4 D's positive control,** the explicit propagation line with its numbers
     trained, or a log-scale σ loss.
2. **D106 [REVIEW], stage C's base (merged SFT, E1d-style) and prompt v5 on a v4-trained model.**
   - **The alternative:** the pinned base with an implicit-arm SFT adapter that RL continues (D70). It costs ~1.2 h
     more, and its GRPO path is untested.
3. **D105 [REVIEW]: `weights.sigma = 0`, and approval to add the version dispatch to `reward.reward` and `tape.replay`**
   (a few lines; with it, `tape.replay` covers 1.3.0 tapes).
4. **D103 as applied:**
   - **Host use tonight was 0.59 GiB,** so 7.1 left a 0.3 GiB margin.
   - **Training** peaked at 4.45 GiB device-wide. **The evals** (batch 32) reserved up to 6.72 GiB, 7.30 GiB
     device-wide, under the 8 GiB physical: no OOM backoff, and no sign of a spill (53–54 min each, steady).
   - **The OOM check passed:** 2.0 GiB reserved on the longest batch.
5. **D107.9:** stage B's eval loader flags `tokenizer_matches.backend: False`. It is benign: only the post-processor
   differs, and the encodings are identical on 300 texts.

## 7. BLOCKED / problems / tests

- **Nothing is BLOCKED.** One temperature pause (seed 0, 00:37, 88 °C with no CPU job running; memory note updated).
  Stage C was skipped by the 3.5 h rule, as expected when the night was planned.
- **Tests:** 553 passed, 0 failed, 0 skipped (GPU visible, after the queue). That is 519 + 34 new: E1d-X2 3, stage B
  5 + 3, verifier 1.3.0 17, stage C 6.

## 8. Recommendation

- **The σ-only RL line has now answered its own question twice.**
  - **Within problems (FINDINGS #26):** E1d's RL σ ranks problems, not answers.
  - **Propagation (FINDINGS #27):** supervised σ\* imitation alone gives a surface reader.
- **Tonight (if you agree with D106's base):** the stage C pilot, ~2.7 h. It is the first test of the predictive σ
  reward. C6 (RL vs its SFT control) and C2 (A6c) decide whether 3 seeds are warranted.
- **In parallel (CPU, daytime):**
  - the 1.3.0 dispatch edit (if approved);
  - building R3-4 D's positive control (explicit line, numbers trained) as the propagation ceiling that stage B
    turned out not to be.
- **For p (stage E):** self-consistency remains the baseline to beat (0.643 within problems). Nothing tonight came
  close.

# Report: 2026-10-16 day + night (DRAFT, written 23:45 IST; the overnight sections are filled in the morning)

The previous report (2026-10-14) is in git history: `git show d5abb76:MORNING.md`.

**Short version so far.**
1. **D61 recorded** (A6 slope labels) before any A6/A6b result. One open point is flagged for you (§6).
2. **Split credit, seed 0:** it removed the σ = 0 collapse and raised 1σ coverage, but σ still does not track the
   error. On all three N tests the A5 surface-feature σ beats it on the same answers. **No S1–S3 verdict:** that
   needs the 3-seed mean.
3. **E1 (family E, pre-registered, single seed):** E1a-P and E1b-P are supported and E1a-N and E1b-N are not.
   This is D52's "clean negative": with μ fixed, a σ is learnable, but it does not beat surface features.
4. **The laptop shut down at about 23:30** during split-credit seed 1, at step 162 of 287. Seed 1 was
   **restarted from step 0** (D63: resume would not restore the curriculum sampler's state). The queue was
   relaunched at 23:38.

---

## 1. What ran (IST)

| time | stage | result | commit |
|---|---|---|---|
| 12:33–15:27 | split-credit seed 0, training to step 287 | 36 s/step, no OOM | `af61aea` |
| 14:34 | **D61** recorded (during the seed-0 run, before any A6/A6b result) | – | `1203fb4` |
| 15:27–15:38 | its step-287 eval + sanity | sanity PASS, 0% parse failures | `46f0ca9` |
| 15:38 | family S / N comparison (CPU) | §2 | `13da449` |
| 15:38–18:21 | E1a training (300 steps) | 32.5 s/step, no OOM | `e745616` |
| 18:22–18:35 | E1a evals (final, step 150) | 0 failures | `dac3046`, `eacdcec` |
| 18:35–20:57 | E1b training (300 steps) | 28.4 s/step, no OOM | `9dac35a` |
| 20:57–21:06 | E1b evals + family E (CPU) | §3 | `74254d7`, `d371ca2` |
| 19:12–19:29 | **D62** (seeds 1/2, seed-mean test, sampled σ) + code, before any seed-1/2 data | 348 tests pass | `7a8f9f1`, `873dee5` |
| 21:06–23:30 | split-credit seed 1, first attempt | **lost to the shutdown at step 162** | – |
| 23:38 | **D63**; queue relaunched, seed 1 from step 0 | running | `6499145` |

GPU work for the day queue ended at 21:06, inside the 21:30 cutoff.

## 2. Split credit, seed 0, at step 287 (family S and family N)

**Run.** It was identical to the 2026-10-13 uniform-credit run except for split credit (σ_ref = 1% of μ).
- **Training:** 287 steps in 2 h 53 min (36 s/step). No OOM; peak reserved 7.05 GiB.
- **Sanity check:** PASS.
- **Matched control:** the step-1 training metrics are identical to the uniform run's (same seeds). From step 2
  on, the runs differ only in credit assignment.

**Eval at step 287** (frozen 1.1.0 sets, 200 ID + 200 depth 5, greedy, pooled n = 400):

| model | acc ≤ 1% | 1σ cov | 2σ cov | σ = 0 | median rel. σ | ρ(σ, err) | mean crps_log |
|---|---|---|---|---|---|---|---|
| SFT-only (stated σ) | 34.5% | 13.2% | 18.2% | 18.8% | 0.063% | 0.02 | −0.339 |
| SFT-only, σ × 18.1 (standing post-hoc) | 34.5% | 34.2% | 39.2% | 18.8% | 1.14% | 0.02 | −0.336 |
| uniform-credit RL@287 | 34.2% | 11.8% | 16.2% | 21.0% | 0.050% | −0.02 | −0.359 |
| **split-credit RL@287, seed 0** | **34.5%** | **22.5%** | **29.0%** | **2.2%** | **0.244%** | **0.08** | **−0.325** |
| A5 strongest σ on the split-credit answers | 34.5% | 65.8% | 76.5% | 0.0% | 6.52% | 0.70 | −0.292 |
| A5 strongest σ on the SFT answers | 34.5% | 66.8% | 76.8% | 0.0% | 6.77% | 0.69 | −0.305 |

- **crps_log per split:** ID −0.300 for split credit (uniform −0.282, SFT −0.320); depth 5 −0.351 (uniform
  −0.437, SFT −0.357).

**Family S: seed-0 numbers only, NO VERDICT** (PREREGISTRATION.md: supported only on the seed mean; your brief:
no S1–S3 verdict until 3 seeds exist). `phase95_compare.py` prints a "supported" column computed on seed 0
alone. Read it as a per-seed result, not a verdict.

| test | seed 0 | one-sided p | Holm threshold | per-seed outcome |
|---|---|---|---|---|
| S1a: 1σ cov > SFT-only | 90 vs 53 covered of 400 (41 vs 4 discordant) | 4.7e-9 | 0.025 | holds on seed 0 |
| S1b: 1σ cov > uniform RL@287 | 90 vs 47 (45 vs 2 discordant) | 8.0e-12 | 0.0167 | holds on seed 0 |
| S2: ρ(σ, err) > uniform RL@287 | 0.078 vs −0.017, diff CI [−0.026, 0.214] | 0.061 | 0.05 | does not hold on seed 0 |
| S3: accuracy not > 5 pts below uniform | 34.5% vs 34.25%, diff CI [−1.3, +2.0] pts | < 1e-4 | 0.0125 | holds on seed 0 |

**Family N (D50/A5):** the strongest A5 σ applied to the *same* split-credit answers (identical μ). D50 gives
no seed rule for N, so this is a seed-0 result until seeds 1–2 exist.

| test | split-credit stated σ | A5 σ, same answers | 95% CI of difference | supported |
|---|---|---|---|---|
| N1: \|1σ cov − 68.3%\| smaller | 0.458 | 0.025 | [0.382, 0.475] | no |
| N2: ρ(σ, err) higher | 0.078 | 0.696 | [−0.729, −0.511] | no |
| N3: crps_log higher | −0.325 | −0.292 | [−0.042, −0.026] | no |

- **Reading:** split credit removed σ = 0 answers (2.2% vs 18.8% / 21.0%) and raised 1σ coverage from 13.2%
  (SFT) and 11.8% (uniform) to 22.5%, at no cost in accuracy. It did **not** make σ track the error: ρ ≈ 0.08,
  against 0.70 for a text-feature baseline on the same answers. The stated σ is still far too narrow, and the
  surface-feature σ beats it on all three N tests, with CIs well clear of zero.
- **Also reported:** crps_log vs the standing post-hoc baseline (SFT σ × 18.1) is +0.011 (Wilcoxon two-sided
  p = 0.64). The superseded D49 family A also fails.

**σ during training** (reported, not tested; T = 1.0 training samples, per 25-step window):

| steps | split: σ = 0 | split: median rel. σ | split: 1σ cov | uniform: σ = 0 | uniform: median rel. σ | uniform: 1σ cov |
|---|---|---|---|---|---|---|
| 1–25 | 30% | 0.05% | 20% | 30% | 0.06% | 20% |
| 101–125 | 14% | 0.17% | 27% | 25% | 0.05% | 17% |
| 176–200 | 2% | 1.37% | 44% | 29% | 0.06% | 19% |
| 251–275 | 0% | 1.39% | 49% | 22% | 0.07% | 17% |
| 276–287 | 0% | 1.27% | 51% | 25%\* | 0.05% | 13% |

\* The uniform run's σ = 0 window here is 276–300 (`sigma_windows.txt`).

- **Training vs eval gap:** sampled training σ settled near 1.3% (close to σ_ref = 1%), but the greedy eval σ is
  0.24%, five times narrower. Hypothesis, not tested: greedy picks the modal σ string, and that mode is still
  narrow. This is why the greedy-vs-sampled exploratory eval (D62) was added.
- **The σ = 0 alert (> 50%) never fired.**

## 3. E1, the fixed-μ σ-only pilot (family E, D52)

**Runs.** E1a (free σ) and E1b (σ mapped to 9 bins) are both seed 0, starting from the SFT-only model with a
fresh LoRA. Both completed the full 300 steps (permanent checkpoint at 150). These were the trainer's first GPU
runs, and both were clean:

| run | s/step | peak reserved | OOM | parse ok (train) | μ changed (train) |
|---|---|---|---|---|---|
| E1a | 32.5 | 7.47 GiB | 0 | ≥ 99.8% | ≤ 0.1% |
| E1b | 28.4 | 7.47 GiB | 0 | ≥ 99.8% | 0% |

- **Memory:** peak reserved is close to the 7.5 GiB allocator cap, but under it.
- **σ in training:** for both runs the median relative σ rose from 0.08% to about 3%, and σ = 0 was gone by step
  ~100–150. The two runs followed almost the same path.

**Family E (pre-registered, D52):** 400 eval prefixes, one-sided paired bootstrap, Holm over 4. The SFT answers
are the same in every arm, so μ is identical and only σ differs.

| test | mean Δcrps_log | 95% CI | p | Holm threshold | supported |
|---|---|---|---|---|---|
| E1a-P: E1a σ > SFT stated σ | +0.0024 | [+0.0018, +0.0030] | < 1e-4 | 0.0125 | **yes** |
| E1b-P: E1b σ > SFT stated σ | +0.0020 | [+0.0015, +0.0025] | < 1e-4 | 0.0167 | **yes** |
| E1a-N: E1a σ > strongest A5 σ | −0.0310 | [−0.0400, −0.0228] | 1 | 0.025 | **no** |
| E1b-N: E1b σ > strongest A5 σ | −0.0314 | [−0.0404, −0.0233] | 1 | 0.05 | **no** |

| σ source (same 400 answers) | mean crps_log | 1σ cov | 2σ cov | ρ(σ, err) | median rel. σ | σ = 0 |
|---|---|---|---|---|---|---|
| SFT stated σ | −0.3386 | 13.2% | 18.2% | −0.02 | 0.061% | 18.8% |
| E1a final (step 300) | −0.3362 | 29.2% | 35.8% | 0.06 | 0.57% | 1.0% |
| E1b final (step 300) | −0.3366 | 28.2% | 33.0% | 0.08 | 0.50% | 0.0% |
| A5 strongest σ | −0.3051 | 66.8% | 76.8% | 0.70 | 8.0%\* | 0.0% |

\* `e1 compare`'s median for A5; the family S table gives 6.8% for the same tape under a different σ
convention. The difference is not material.

- **At step 150** (descriptive): E1a crps_log −0.3369, 1σ cov 26.0%, ρ 0.10; E1b −0.3372, 25.2%, ρ 0.08. Most of
  the small gain was already there at step 150.
- **E1a vs E1b at the matched step (300):** indistinguishable (Δcrps_log 0.0004).
- **Failures at eval:** none (no parse failures, no unit mismatches, μ never changed).

**Verdict under D52's interpretation rule: E1x-P is supported without E1x-N, for both runs.** With μ held fixed
(clean σ credit), a σ is learnable that beats SFT's own σ. It is not better than surface features: the gain is
tiny (+0.002 crps_log, about 7% of the gap to A5), and σ does not track the error (ρ ≈ 0.06–0.08 vs 0.70). This
is D52's "clean negative result".

## 4. Overnight: split-credit seeds 1 and 2 (PENDING, filled in the morning)

## 5. Overnight: seed-mean S1–S3 and greedy vs sampled σ (PENDING)

## 6. Decisions to review

1. **D61, open point:** read literally, a 90% CI lying entirely below 0 is labelled "partial propagation" (it
   "excludes 0 but lies entirely below 1"). D59 treats a negative slope as a distinct habit (the absolute-σ
   pattern). Which label should it get? The labels are not in `a6_pairs.py` until this is settled.
2. **D62 [REVIEW]: how the seed-mean test works** (the pre-registration did not specify it).
   - Per-problem 1σ coverage averaged over the 3 seeds for S1a/S1b; the mean of the per-seed ρ for S2; seed-mean
     accuracy for S3.
   - One-sided paired bootstrap over problems, Holm over 4. Only one uniform-credit seed exists for the controls.
   - Recorded before any seed-1/2 data; the rejected alternatives are listed in D62.
3. **D62: exploratory sampled eval:** k = 4 at the training settings. For split credit μ also varies between
   samples, while E1's μ is fixed.
4. **D63: seed 1 restarted rather than resumed.** The checkpoint lacks the curriculum sampler's state and RNG,
   and `grpo.py` has no resume path. If you want crash-resume in future, the sampler state has to be
   checkpointed.

## 7. BLOCKED / housekeeping

- **Laptop shutdown at about 23:30:** the damage check was clean. No stale lock, `git fsck` clean, local =
  origin, and the models and adapters load. The partial seed-1 run is kept, git-ignored, in
  `results/discarded/`.
- **The restart is not bit-identical past step 2:** the restarted seed 1 matches the crashed run exactly at steps
  1–2 and drifts from step 3 (GPU non-determinism). Same code, config and seed.
- **The scratchpad in /tmp was wiped by the reboot.** Watchers were recreated; nothing in the repo was affected.
- `PHASES_10-15.md` is still untracked (its content is in CLAUDE.md).

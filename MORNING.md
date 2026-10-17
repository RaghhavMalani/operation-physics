# Report: 2026-10-16 day + 2026-10-17 night (final, 07:30 IST)

The previous report (2026-10-14) is in git history: `git show d5abb76:MORNING.md`.

**Short version.**
1. **Split credit, 3 seeds: S1a, S1b, S2 and S3 are SUPPORTED on the seed mean** (FINDINGS #13).
   - **S1a/S1b:** 1σ coverage is 21.7%, against 13.25% for SFT and 11.75% for uniform credit.
   - **S3:** accuracy is not worse (+1.4 pts).
   - **S2 passes narrowly:** ρ 0.081 vs −0.017, p = 0.040 at a Holm threshold of 0.05. It rests on D62's
     seed-mean definition [REVIEW].
   - **But family N fails on every seed.** A surface-feature σ (A5) on the same answers still wins clearly:
     coverage 66–68%, ρ ≈ 0.7. Split credit cures the σ = 0 collapse; it does not make σ informative.
2. **E1 (family E):** E1a-P and E1b-P are supported and E1a-N and E1b-N are not (FINDINGS #14). This is D52's
   "clean negative": with μ fixed, a σ is learnable, but it does not beat surface features.
3. **The laptop shut down at about 23:30** during seed 1. Damage check clean; seed 1 was restarted from step 0
   (D63). Every stage after that ran, and GPU work ended at 07:11, inside the 08:30 cutoff. The sampled-σ evals
   also fit.
4. **New observation, needs your call (§6):** RL adapters are trained on a 4-bit base but evaluated on the bf16
   base. This may explain why eval σ is about 4× narrower than training σ.
5. **D61** (A6 slope labels) is recorded, with one open point.

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
| 23:38 | **D63**; queue relaunched, seed 1 from step 0 | – | `6499145` |
| 23:38–02:26 | split-credit seed 1 (restart), training to step 287 | 34.9 s/step, no OOM | `c6e1662` |
| 02:26–02:38 | its eval, sanity, per-seed family S/N | sanity PASS, 0% parse failures | `7ec6ffe` |
| 02:38–05:52 | split-credit seed 2, training to step 287 | 40.3 s/step (longer completions), no OOM | `63fbfce` |
| 05:52–06:04 | its eval, sanity, per-seed family S/N | sanity PASS, 0% parse failures | `94515eb` |
| 06:04–07:11 | sampled-σ evals (k = 4): split-credit seed 0, E1a, E1b | all ran | `26f1b96` and earlier |
| 07:11 | seed-mean family S + greedy-vs-sampled report (CPU) | §4, §5 | `f7ab1e6`, `8b8cfca` |

- **GPU cutoffs met:** the day queue ended at 21:06 (cutoff 21:30), the night queue at 07:11 (cutoff 08:30).
- **Heat:** no thermal pause was triggered (no reading above 87 °C; the highest I saw was 83 °C, during E1b).
- **Commits:** every stage was committed and pushed (the night queue did it itself).

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

**Family S: seed-0 numbers only, no verdict here (the 3-seed verdict is in §4)** (PREREGISTRATION.md: supported only on the seed mean; your brief:
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

## 4. Split credit, 3 seeds: the pre-registered family S verdict

**Per seed** (step-287 greedy eval, pooled n = 400; all three seeds passed sanity with 0% parse failures):

| seed | s/step | acc ≤ 1% | 1σ cov | 2σ cov | σ = 0 | median rel. σ | ρ(σ, err) | crps_log | A5 σ on same answers: cov / ρ / crps_log |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 36.2 | 34.5% | 22.5% | 29.0% | 2.2% | 0.244% | 0.078 | −0.325 | 65.8% / 0.70 / −0.292 |
| 1 | 34.9 | 36.2% | 21.0% | 27.5% | 6.5% | 0.209% | 0.099 | −0.302 | 68.0% / 0.69 / −0.269 |
| 2 | 40.3 | 36.2% | 21.5% | 29.2% | 3.5% | 0.233% | 0.066 | −0.336 | 67.0% / 0.66 / −0.301 |
| SFT-only | – | 34.5% | 13.2% | 18.2% | 18.8% | 0.063% | 0.02 | −0.339 | 66.8% / 0.69 / −0.305 |
| uniform RL@287 (1 seed) | – | 34.2% | 11.8% | 16.2% | 21.0% | 0.050% | −0.02 | −0.359 | – |

**Family S on the seed mean (D62; this is the pre-registered verdict):**

| test | split credit, mean of 3 seeds | control | difference [95% CI] | one-sided p | Holm threshold | **verdict** |
|---|---|---|---|---|---|---|
| S1a: 1σ cov > SFT-only | 21.7% | 13.25% | +8.4 pts [+5.8, +11.3] | < 1e-4 | 0.0125 | **supported** |
| S1b: 1σ cov > uniform RL@287 | 21.7% | 11.75% | +9.9 pts [+7.2, +12.8] | < 1e-4 | 0.0167 | **supported** |
| S2: ρ(σ, err) > uniform RL@287 | 0.081 | −0.017 | +0.098 [−0.012, +0.208] | 0.040 | 0.05 | **supported (narrow)** |
| S3: accuracy not > 5 pts below uniform | 35.7% | 34.25% | +1.4 pts [+0.1, +2.9] | < 1e-4 | 0.025 | **supported** |

- **Per-seed tests, as pre-registered:**
  - S1a and S1b hold on every seed (McNemar p < 1e-7), and so does S3.
  - S2's per-seed p is 0.061 / 0.020 / 0.092, so it holds alone only on seed 1.
  - The seed mean passes S2 because the three per-seed effects all point the same way.
- **How much S2 depends on D62:** S2 is the hypothesis most sensitive to how the seed mean was defined (D62
  [REVIEW], the mean of per-seed ρ). D62 was fixed before seeds 1/2 existed, but no pre-registration chose it.
- **Family N (A5 strongest σ, same answers) fails on all three seeds:** all of N1–N3 go the wrong way with CIs
  well clear of zero. For "RL σ beats the baseline" the answer is **no**, consistently.
- **What it means:** split credit is a real fix for the collapse (σ = 0 falls from about 20% to 2–6%, and
  coverage rises 9 points at no accuracy cost). The σ it produces is still far too narrow (21.7% vs 68.3% at 1σ)
  and hardly informative (ρ ≈ 0.08 vs 0.70). S2's narrow pass is a statistically detectable but tiny ranking gain.
- **Caveats:**
  - Only one uniform-credit seed exists (control for S1b/S2/S3).
  - Inference is over problems, conditional on these 3 seeds.
  - Seed 1 is the restarted run (D63).
- **Files:** `results/split_credit_seed_mean/family_S_seed_mean.{md,json}`, plus per-seed
  `…/compare/phase95_family_S.md`.

## 5. Exploratory: greedy vs sampled σ at eval (D62; no tests)

k = 4 samples per problem at the training settings (T = 1.0), on the same frozen eval problems/prefixes:

| model | decoding | fail | median rel. σ | σ = 0 | 1σ cov | 2σ cov | mean crps_log | ρ(σ, err) |
|---|---|---|---|---|---|---|---|---|
| split-credit seed 0 @287 | greedy | 0.0% | 0.244% | 2.2% | 22.5% | 29.0% | −0.325 | 0.08 |
| split-credit seed 0 @287 | sampled | 0.9% | 0.367% | 5.5% | 19.9% | 25.7% | −0.517 | 0.08 |
| E1a | greedy | 0.0% | 0.567% | 1.0% | 29.2% | 35.8% | −0.336 | 0.06 |
| E1a | sampled | 0.2% | 0.773% | 2.7% | 32.6% | 39.1% | −0.343 | 0.03 |
| E1b | greedy | 0.0% | 0.500% | 0.0% | 28.2% | 33.0% | −0.337 | 0.08 |
| E1b | sampled | 0.2% | 0.500% | 0.0% | 30.6% | 38.0% | −0.343 | 0.03 |

| model | median within-problem SD of ln σ (k = 4) | greedy σ mean mid-rank among the samples | greedy σ below all 4 samples |
|---|---|---|---|
| split-credit seed 0 | 1.01 | 0.43 | 7.5% |
| E1a | 0.94 | 0.45 | 3.5% |
| E1b | 0.82 | 0.43 | 2.8% |

- **Greedy decoding does not explain the training-vs-eval σ gap.**
  - The greedy σ sits near the middle of its own samples (mid-rank 0.43–0.45), not at the narrow end.
  - Sampled eval σ is only about 1.5× the greedy σ (0.37% vs 0.24%), still far below the ~1.3% sampled in late
    training.
- **Problem mix does not explain it either** (descriptive, seed 0): at every depth, training samples in steps
  263–287 state a median σ of 1.2–1.7%, while sampled eval at the same temperature states 0.27–0.42%, about 4×
  narrower.
- **The stated σ is very unstable within a problem:** the SD of ln σ across 4 samples is about 1. A resample
  typically moves σ by a factor of about e.
- **The sampled crps_log for split credit is much worse** (−0.517). Sampling also changes μ (more wrong answers,
  0.9% failures at the −5 floor); it is not a σ effect. E1 (μ fixed) changes little.
- **One remaining candidate for the gap (not tested):** the train/eval precision mismatch in §6.

## 6. Decisions to review

1. **NEW, train/eval precision mismatch (found this morning, nothing changed).**
   - **What it is:** RL LoRAs are trained on a 4-bit base (`physcal/train/grpo.py:329`, `load_in_4bit=True`),
     but every RL eval applies them to the bf16 merged SFT model (`configs/eval_probe_greedy.yaml`, plain
     transformers).
   - **What it affects:** all RL arms are evaluated the same way, so between-arm comparisons (S, N, E) stay fair.
     But the eval measures a slightly different model than the trained policy. This is the only candidate left
     for the ~4× σ gap (§5).
   - **Proposed check, about 45 min of GPU:** evaluate split-credit seed 0 @287 on the 4-bit base (the same
     loader as training), greedy and k = 4 sampled, and compare σ. If the gap closes, every RL eval so far has
     understated the trained policy's σ.
2. **D62 [REVIEW], how the seed-mean test works.**
   - Per-problem coverage averaged over seeds; the mean of per-seed ρ; seed-mean accuracy; paired bootstrap
     over problems; Holm.
   - S2's support depends on it (§4).
   - The alternatives are listed in D62. If you prefer one, it should be recorded as a deviation, since the
     results are now known.
3. **D61, open point:** read literally, a 90% CI lying entirely below 0 is labelled "partial propagation". Which
   label should it get? The labels are not in `a6_pairs.py` until this is settled.
4. **D63: seed 1 restarted, not resumed.** If you want crash-resume in future, the curriculum sampler's state
   (including its RNG) has to be checkpointed with the adapter.
5. **Only one uniform-credit seed exists.** A second and third uniform seed would make S1b, S2 and S3 symmetric
   (about 6 GPU-hours).

## 7. Recommendation

- **Stop iterating on RL credit assignment for σ at this scale.** Split credit (3 seeds) and fixed-μ E1 both fix
  the collapse but neither comes near the A5 surface-feature σ. The bottleneck is not credit assignment.
- **First, run the 4-bit eval check** (§6.1). It is cheap and could change how every RL σ number reads.
- **Then move to the noise-mode work (Phase 11/12)**, where σ\* is known per problem. A6/A6b then show directly
  whether any σ source propagates input noise, which is the question A8 makes the headline.

## 8. BLOCKED / housekeeping

- **Laptop shutdown at about 23:30:** the damage check was clean. No stale lock, `git fsck` clean, local =
  origin, and the models and adapters load. The partial seed-1 run is kept, git-ignored, in
  `results/discarded/`.
- **The restart is not bit-identical past step 2:** the restarted seed 1 matches the crashed run exactly at steps
  1–2 and drifts from step 3 (GPU non-determinism). Same code, config and seed.
- **The scratchpad in /tmp was wiped by the reboot.** Watchers were recreated; nothing in the repo was affected.
- **Not run:** nothing. All planned stages ran, including the optional sampled evals.
- **BLOCKED:** none.
- `PHASES_10-15.md` is still untracked (its content is in CLAUDE.md).

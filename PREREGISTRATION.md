# Pre-registration: phases 9.5 and 10–14

Written 2026-10-14, **before any Phase 9.5 or Phase 10+ result exists**. Committed to git before the first
result-producing run of each experiment below; the commit hash is the timestamp. This file is never edited
afterwards. Deviations are recorded in DECISIONS.md with reasons, and the write-up reports every hypothesis below
as supported / not supported / not run.

Conventions for every test below:
- α = 0.05, one-sided where a direction is stated. Holm correction **within each family** (P, S, H).
- Paired bootstrap: B = 10,000 resamples of *problems* (seed 0), percentile intervals. "A beats B" means the
  one-sided bootstrap p-value of (A − B) ≤ 0 is below the Holm-adjusted α.
- Coverage counts: a failed answer (parse failure, unit mismatch, missing ±) counts as not covered (D41).
- Fitting of anything (probes, post-hoc scales, conformal quantiles, layer choice, hyperparameters) uses only
  training-seed data (the calibration split). The frozen eval sets are used only for the final numbers.
- σ = 0 answers: where a log of σ is needed, σ is replaced by the verifier's crps_log floor (1e-6 relative) and
  a sensitivity analysis excluding σ = 0 answers is reported next to it.

---

## Family P — Phase 9.5 step 2: does the SFT-only model "know" its error? (FINDINGS #7)

**Model.** SFT-only, `results/sft/qwen2.5-math-1.5b-instruct/merged` (the start point of every RL run).

**Data.**
- *Fit set:* SFT-only greedy completions (prompt v4, 512 tokens, same settings as the eval) on the calibration
  split: generator train split, seeds 20,000,000–20,001,999 (2,000 problems, round-robin over the 12 training
  cells; the first 400 are the existing post-hoc calibration set). Disjoint from GRPO, SFT and eval seeds.
- *Test set:* the existing SFT-only greedy completions on the frozen generator-1.1.0 eval sets (200 ID +
  200 depth-5), as re-scored by verifier 1.1.0. Never used for fitting or selection.
- Completions that fail to parse are excluded from both (expected < 1%); the count is reported.

**Features.** Hidden states of every layer (embedding output + 28 blocks = 29) at the last completion token
before the `<answer>` tag, from a teacher-forced forward pass over the exact chat-template prompt + completion.
Stored float16.

**Labels.** (i) correct = |relative error| ≤ 1%. (ii) y = log10(max(|relative error|, 1e-6)).

**Probes.** Standardized features. Linear (primary): L2 logistic regression for (i), ridge regression for (ii);
the regularization strength is chosen by 5-fold CV on the fit set from {1e-3, 1e-2, 1e-1, 1} (logistic C) and
{1, 10, 100, 1000, 10000} (ridge α). MLP (secondary): one hidden layer of 64 units, early stopping, same
selection rule. **Layer selection:** the layer with the best 5-fold CV AUROC (i) / Spearman ρ (ii) on the fit set.

**Baselines and controls** (same pipeline, same layer-selection rule):
- *Stated σ:* AUROC of −log(relative σ) for (i); Spearman ρ of relative σ with |error| for (ii).
- *Problem metadata:* logistic / ridge regression on one-hot (family, depth), fit on the fit set.
- *Shuffled labels:* probes fit on permuted fit-set labels.
- *Random-init model:* the same architecture with randomly initialized weights (seed 0), same texts.

**Hypotheses** (pooled test set, n ≈ 400; per-split numbers reported as secondary):
- **P1 (primary):** the selected linear probe's AUROC for correctness exceeds the stated-σ AUROC.
- **P2:** the selected linear probe's AUROC for correctness exceeds the metadata baseline's AUROC.
- **P3:** the selected linear probe's Spearman ρ with log|error| exceeds stated σ's Spearman ρ with |error|.

Sanity conditions (not hypotheses; if violated, P1–P3 are reported as uninterpretable): the shuffled-label probe's
AUROC 95% CI contains 0.5, and the real-model probe beats the random-init probe on P1's metric (paired bootstrap).

**Interpretation rule:** "the model knows its error better than its stated σ says" = P1 and P2 supported.

---

## Family S — Phase 9.5 step 3: split credit for σ tokens

**Run.** Identical to the 2026-10-13 run (`configs/grpo_overnight2.yaml`: verifier 1.1.0, same SFT start, seeds,
lr, curriculum, 512 tokens, 4 × 8 per step, micro-batch 2 × 16) except split credit:
r_answer = crps_log(μ, σ_ref) + 0.2·steps with σ_ref = 1% of μ, and r_sigma = crps_log(μ, σ) − crps_log(μ, σ_ref).
Answer and reasoning tokens get the group-normalized advantage of r_answer; σ tokens get the group-normalized
advantage of r_sigma. Seed 0 first; seeds 1 and 2 once seed 0 is sane (no OOM, no parse collapse).

**Comparison point.** The permanent step-287 adapter, evaluated greedy on the frozen eval sets (200 ID +
200 depth 5, pooled n = 400). Controls: SFT-only and the uniform-credit run's step-287 adapter (2026-10-13 run),
both re-scored by verifier 1.1.0. Post-hoc control: SFT-only × 18.1 (FINDINGS #6).

**Hypotheses:**
- **S1a (primary):** split-credit 1σ coverage > SFT-only 1σ coverage (McNemar, one-sided).
- **S1b (primary):** split-credit 1σ coverage > uniform-credit RL@287 1σ coverage (McNemar, one-sided).
- **S2:** Spearman ρ(stated relative σ, |relative error|) of split-credit RL@287 > that of uniform-credit RL@287
  (paired bootstrap).
- **S3 (non-inferiority):** split-credit strict accuracy (≤ 1%) is not lower than uniform-credit RL@287 by more
  than 5 points (one-sided 95% bootstrap lower bound of the difference > −5 points).

Reported, not tested: σ = 0 share in training rollouts per 25-step window (the collapse criterion is a rolling
share > 50%); 2σ coverage; mean crps_log against the post-hoc control (Wilcoxon, two-sided); the final checkpoint
as "longer training". With 3 seeds, S1–S3 are evaluated per seed and on the seed mean; a hypothesis is supported
only if it holds on the seed-mean comparison.

---

## Family H — Phases 10–14 (noise mode, σ\*, five-way comparison)

Definitions. σ\* is the generator's per-problem true uncertainty (Phase 10; in log space for crps_log).
"Noise > 0 problems" = eval problems with noise level ≥ 1%. Cells = depth × noise level. "RL" = the Phase 12
stated-σ model (seed mean over 3 seeds); "SFT" = the SFT-only model; "post-hoc" = a single global σ scale fitted
on the calibration split; "conformal" = split-conformal intervals fitted on the calibration split.

Precondition (Phase 10.4): the generator's own σ\* intervals pass the Kupiec test (p > 0.05) at 68% and 95% in
every depth × noise cell on 10,000 problems. If it fails, Family H is not run.

- **H1, propagation (vol beta).** On noise > 0 problems, β = OLS slope of log(stated σ) on log(σ\*).
  H1a: β_RL > β_SFT. H1b: |1 − β_RL| < |1 − β_SFT|. Paired bootstrap over problems.
- **H2, conditional calibration.** For post-hoc and conformal: pooled Kupiec test at 68% and 95% passes
  (p > 0.05) *and* the chi-square test of coverage homogeneity across cells (cells with n ≥ 20) fails (p < 0.05).
  H2a: this pattern holds for both post-hoc and conformal (descriptive, two pre-stated outcomes).
  H2b: RL's worst-cell |coverage − 0.68| < post-hoc's and < conformal's (paired bootstrap, problems resampled
  within cells; both comparisons must hold).
- **H3, uncertainty sources (exploratory).** Ranking of the five sources by median |log(σ / σ\*)|. No test.
- **H4, decision value.** H4a: RL-stated σ has lower AURC than post-hoc (risk = crps_log loss; abstention by σ).
  H4b: RL's Kelly log-growth against the fixed-width quote (width = median relative error on the calibration
  split) is higher than post-hoc's. Paired bootstrap.
- **H5, stress.** Over stress-perturbed problem pairs (unit swaps, extreme magnitudes, extra distractors,
  paraphrases): OLS slope of Δlog σ on Δlog|error|. H5a: slope > 0 (95% CI excludes 0). H5b: the 95% CI of the
  slope contains 1 ("in proportion").

Holm family H: {H1a, H1b, H2b, H4a, H4b, H5a}. H2a and H5b are reported as pre-stated descriptive outcomes.

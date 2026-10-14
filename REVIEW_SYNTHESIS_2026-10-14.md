# Review synthesis — external reviews (Gemini, ChatGPT), 2026-10-14

Commit this file, then record items A1–A8 as a dated amendment in DECISIONS.md
BEFORE any Phase 10+ or new-experiment result exists. PREREGISTRATION.md is not edited.
If any affected result was already produced today, report it under both the old and the
amended criterion and say plainly that the amendment came after.

## Headline reframing (adopted)

Research question:
> Can a small language model learn to distinguish uncertainty inherited from the world
> (noisy inputs) from uncertainty introduced by its own reasoning, and communicate both
> in a way that improves decisions?

Contribution framing: a controlled testbed that separates the two, plus an analysis of
when and why RL fails or succeeds at teaching stated numeric uncertainty. A clear
mechanistic negative result counts as a contribution.

## Consensus findings from both reviewers

1. **Target misspecification (both, top priority).** sigma\* (propagated measurement noise)
   is not the ideal sigma for a fallible model. On problems the model gets wrong, an honest
   sigma should exceed sigma\*. Grading total sigma against sigma\* on all problems is invalid.
2. **Split credit is a reasonable experiment, not a principled fix (both).** r_sigma still
   depends on how wrong each mu was, so comparing sigma across completions with different mu
   stays confounded. Cleaner: hold mu fixed and compare alternative sigmas (forked / two-turn).
3. **Sigma exploration is a bottleneck (both).** Use discrete log-spaced sigma bins and track
   bin entropy. ChatGPT adds: masking sigma in SFT may preserve a near-zero confidence prior
   (cf. Wang et al. 2026, arXiv 2609.32470).
4. **Two channels (both):** confidence p trained by Brier on correctness within tolerance;
   sigma for aleatoric spread. Both reviewers prefer decoupled losses to the mixture CRPS.
5. **Christoffersen is the wrong name (both):** it tests serial independence over time.
   Our cross-cell test is a Pearson chi-square / Cochran–Mantel–Haenszel homogeneity test.
6. **Kelly growth is redundant (both):** for continuous forecasts it reduces to a log-score
   difference against a benchmark.
7. **Calculator: diagnostic only (both).**

## Amendments (record in DECISIONS.md before results)

**A1. Two estimands.**
- Epistemic: P(answer correct within tolerance), stated as confidence p, scored by Brier.
- Aleatoric: sigma\* under the measurement model.
- Proper scores (crps_log, log score) against the realized truth remain valid on all problems.
- Every comparison of stated sigma with sigma\* (H1, H3, calibration surface) is computed only
  on computation-correct answers, defined as |log mu − log f(x_meas)| ≤ tau with
  tau = 0.1% relative (configurable). f(x_meas) is the chain evaluated on the displayed noisy
  givens, NOT the clean truth: a perfect solver lands on f(x_meas), which differs from the
  clean truth by about sigma\*. All-problem versions are reported as secondary.
- Noise-free and noisy problems are reported as separate regimes.

**A2. Definition of sigma\*.** sigma\* comes from the Monte Carlo predictive distribution of the
clean truth given the displayed givens, in log space. Primary target: the CRPS-optimal
Gaussian log-scale for that distribution, centred at log f(x_meas). Also report its standard
deviation. Distribution check: the PIT of the truth under the MC distribution is
approximately uniform (KS test).

**A3. Generator self-consistency gate (replaces "Kupiec p > 0.05 in every cell").**
With many cells, a valid generator fails some per-cell tests by chance. New gate:
- pooled Kupiec at 68% and 95%: p > 0.05;
- per-cell Kupiec with Holm correction across cells: no rejection;
- per-cell coverage within its 99% binomial interval around nominal for at least 95% of cells;
- PIT KS check from A2 passes.

**A4. Statistics.**
- Cross-cell coverage homogeneity: Pearson chi-square (CMH when stratified). Never call it
  Christoffersen.
- Paired bootstrap / permutation tests are primary; Diebold–Mariano optional as robustness.
- Add the Mincer–Zarnowitz regression log|error| = alpha + beta·log sigma (ideal alpha = 0,
  beta = 1, report R²). On computation-correct answers versus sigma\* it is the formal vol beta
  for H1; on all answers it measures total-uncertainty calibration.
- Add interval-score / WIS decomposition, PIT histograms, and coverage stratified by the
  chain's numerical condition number.

**A5. Nested baselines (supersede this morning's "text-feature" amendment).**
All fitted on the calibration split, matched model-selection budget, templates held out where
possible:
1. metadata only (family, depth);
2. problem text;
3. problem text + generated answer;
4. problem text + answer + reasoning steps;
features via TF-IDF / bag-of-words, models logistic or ridge regression plus a small
gradient-boosted tree. The hidden-state probe must add information beyond baseline 4 to be
reported as a separate source. No verifier-derived features. RL uncertainty claims are judged
against the strongest of these.

**A6. Counterfactual sensitivity pairs (propagation vs lookup).** Matched problem pairs with the
same template, depth and stated noise levels, where the numbers differ so that propagated
sigma\* differs (vary the condition number: e.g. near-cancelling differences vs well-separated
values, linear vs quadratic dependence). Test whether stated sigma tracks the within-pair
change in sigma\*. This is the primary test for whether the model propagates uncertainty or
learned a lookup from surface features.

**A7. Cuts.**
- Kelly growth: replaced by the log-score difference against a fixed-width benchmark (H4b dropped).
- Market-scoring-rule idea: dropped.
- Basel traffic light: at most one summary table.
- Stress tests: reduced to the A6 pairs plus unit swaps (H5 reduced accordingly).
- LoRA ensemble: kept only as a free appendix from the 3 seeds (no extra runs).
- Phase 13 core comparison: stated sigma, strongest nested baseline, k = 8 sampling spread;
  hidden-state probe only if it beats baseline 4.

**A8. Reframed headline question** as above.

## New experiments (pre-register each before results)

**E1. Fixed-mu sigma-only pilot (highest information per GPU night; both reviewers).**
- Data: existing SFT-only greedy completions on training seeds (the 2,000-problem fit and
  calibration sets), truncated right after "±". No new generation needed.
- Training: GRPO on the sigma suffix only (max_new_tokens about 12), 8 samples per prefix,
  reward crps_log(mu fixed, sigma). All 8 share mu, so the advantage compares sigmas only.
- E1a: free-form sigma. E1b: sigma restricted to nine relative bins
  {0.1%, 0.2%, 0.5%, 1%, 2%, 5%, 10%, 30%, 100%} (constrained decoding or verifier mapping).
- Evaluation: frozen eval-set SFT-only completions truncated at "±", greedy sigma.
  Primary: paired crps_log improvement over the SFT-only stated sigma.
  Secondary: improvement over the strongest A5 baseline; Spearman(sigma, |error|);
  sigma-bin entropy; 68% coverage; sharpness.
- Interpretation: if E1 cannot beat the A5 baselines, the LM's own sigma adds nothing beyond
  surface features. That is a clean negative result.

**E2. Broad-support sigma warm-up.** Short SFT with sigma labels drawn uniformly over the nine
bins (deliberately uninformative), against the current masked-sigma SFT. Tests the
confidence-prior hypothesis without teaching calibration through SFT. Run after E1.

**E3. Two-channel p + sigma with forked sampling (replaces Fix B's mixture CRPS).** Output
`value ± sigma unit, confidence p`. p trained by Brier on correctness within 1% on every
rollout; sigma trained by crps_log only on computation-correct rollouts, using fixed-mu forks.
The p channel follows RLCR: cite it, and claim only the sigma channel and the decomposition
as new.

**E4. Exact-arithmetic diagnostic (later, 1 night).** Deterministic arithmetic execution with
formula selection left to the model. Identical conditions across compared methods.

## Optional design choice (owner decides)

Held-out family: DC circuit networks instead of optics (ChatGPT). Circuits give exact
solutions, natural sensor noise, controllable topology and wide condition-number variety.
Future second domain: finance (Black–Scholes, bond duration and convexity; Gemini), where
"quoting volatility" becomes literal.

## References suggested by reviewers (verify each before citing)

- Gneiting & Raftery (2007), Strictly Proper Scoring Rules, Prediction, and Estimation, JASA 102(477).
- Mincer & Zarnowitz (1969), The Evaluation of Economic Forecasts, NBER.
- Lin, Hilton & Evans (2022), Teaching Models to Express Their Uncertainty in Words, TMLR, arXiv 2205.14334.
- Kadavath et al. (2022), Language Models (Mostly) Know What They Know, arXiv 2207.05221.
- Xiong et al. (2024), Can LLMs Express Their Uncertainty?, ICLR 2024, arXiv 2306.13063.
- Kapoor et al. (2024), Large Language Models Must Be Taught to Know What They Don't Know, NeurIPS 2024, arXiv 2406.08391.
- Kendall & Gal (2017), What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?, arXiv 1703.04977.
- Romano, Patterson & Candès (2019), Conformalized Quantile Regression, arXiv 1905.03222.
- Kazemnejad et al., VinePPO, arXiv 2410.01679.
- Wang et al. (2026), On the Pitfalls of Verbalized Confidence Priors for Calibrating Large Reasoning Models, arXiv 2609.32470 (author list unverified).
- Unverified, do not cite without a source: "Xu et al., 2025".

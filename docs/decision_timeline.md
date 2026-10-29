# Decision timeline: every pre-registration and amendment, in date order

R5-8 asks for a chronological timeline that separates confirmatory from exploratory analyses. This is it, for the
paper's methods section and the reproducibility appendix.

**How to read it:**
- **Dates** are each entry's first commit, converted to IST. The commit hash is the timestamp: every entry was pushed
  before the data it governs existed, unless the row says otherwise.
- **Sources:** PREREGISTRATION.md (written once, never edited) and the D-entries in DECISIONS.md.
- **What is listed:** every pre-registration and amendment. Pure operations (queues, launches, memory caps) are left
  out, except deviations that touched a registered analysis.

**The analysis labels:**
- **Confirmatory:** a test with a decision rule, committed before any data it analyses existed, that ran as
  registered. Deviations are recorded before results.
- **Confirmatory (pilot):** the same, at one seed. It decided what ran next. The paper reports its verdict as
  pre-registered, never as a general claim.
- **Exploratory:** everything else, labelled by kind:
  - *post-hoc*: defined after related results were seen;
  - *descriptive*: defined before computing, but with no test;
  - *diagnostic*: a counterfactual that locates a loss, never a method.
- **Amendment / deviation:** these change a registered analysis. Each row says whether it came before or after the
  results it affects.

## Before pre-registration (2026-10-12 to 2026-10-14 morning): exploratory

**FINDINGS:** [#1][f1], [#2][f2], [#3][f3], [#4][f4], [#5][f5], [#6][f6].
- **Status:** these predate PREREGISTRATION.md, so the paper treats them as exploratory.
- **Decision rules were still fixed before the runs:**
  - **D36** (2026-10-13 00:31, `ab8e79e`): the SFT A/B rule ([#4][f4]) and the overnight GRPO design ([#5][f5]).
  - **D41** (2026-10-13 23:54, `9ad3768`): the matched-step-287 comparison protocol ([#6][f6]).

## The timeline

| date (IST) | entry (commit) | what it fixed | kind | analysis | FINDINGS |
|---|---|---|---|---|---|
| 10-14 09:21 | PREREGISTRATION.md (`f601bb7`) | Families P (does the SFT model know its error), S (split credit), H (H1–H5, phases 10–14); Holm within each family | pre-registration | P, S: **confirmatory**. H: **not run** (see the end) | P: [#7][f7]; S: [#13][f13] (superseded), [#16][f16] |
| 10-14 10:41 | D45, D46 (`b662ab6`) | Family P's probe implementation; family S's split-credit trainer | implementation, before results | confirmatory (P, S) | [#7][f7], [#16][f16] |
| 10-14 10:41 | D47 (`b662ab6`) | P's pre-registered sanity check failed; exploratory baselines added afterwards | amendment, **after** P's result | exploratory (post-hoc) | [#7][f7] |
| 10-14 10:42 | D48 (`f58eb20`) | Scope of the rounding audit (Phase 10 step 2) | analysis plan | exploratory (descriptive) | [#8][f8] |
| 10-14 11:54 | D49 (`6eeef4b`) | Two post-hoc calibration baselines; RL calibration judged against the text-feature one | amendment, before RL calibration results | comparator rule (superseded by D50/A5) | [#10][f10] |
| 10-14 11:59 | D50 (`5f2a846`) | Review round 1, A1–A8: two estimands; the σ\* definition; the generator self-consistency gate (A3); statistics; nested A5 baselines and family N (A5); A6 counterfactual pairs; cuts to family H (A7); the headline question | amendment, before any result it affects | gate and family N: confirmatory | gate: [#9][f9]; A5: [#10][f10]; N: [#13][f13], [#16][f16] |
| 10-14 12:08 | D52 (`fbb29a7`) | E1, the fixed-μ σ-only GRPO pilot | pre-registration | confirmatory (pilot) | [#14][f14] (merged evals), corrected in [#16][f16] |
| 10-14 12:43 | D54 (`8acdba3`) | Noise capped at 5%; 10% becomes a heavy-tail regime | amendment, **after** the gate failed ([#9][f9]) | gate rerun | [#11][f11] |
| 10-14 12:56 | D57 (`5202b1f`) | A3's cell-share rule replaced by a miss-count test; the next rerun is FINAL | amendment, **after** [#11][f11], before the final rerun | gate (confirmatory, with two disclosed amendments) | [#12][f12] (pass) |
| 10-14 13:00 | D58 (`113eb72`) | Construction of the A6 sensitivity pairs | analysis plan, before any model on A6 | the A6 slope tests | [#17][f17], [#27][f27], [#29][f29], [#30][f30], [#32][f32] |
| 10-14 13:11 | D59 (`0225b0a`) | A6b invariance pairs (the slope must be ≡ 0) | pre-registration | confirmatory (C3 of R4-3 uses it) | [#17][f17], [#27][f27], [#30][f30], [#32][f32] |
| 10-16 14:34 | D61 (`1203fb4`) | A6 slope labels ("partial / full propagation") | amendment, before any A6 model result | labels | [#17][f17], [#27][f27] |
| 10-16 19:12 | D62 (`7a8f9f1`) | Split-credit seeds 1 and 2; the seed-mean S1–S3 test; greedy vs sampled σ | amendment to family S, before seed-1/2 data | S: confirmatory; greedy vs sampled: exploratory | [#13][f13] → [#16][f16] |
| 10-17 21:57 | D64 (`9578ee6`) | The label "anti-propagation" for a CI below 0 | amendment, before any A6 model result | labels | – |
| 10-17 22:07 | D65 (`deff739`) | RL adapters evaluated on the 4-bit training base | analysis plan, before any 4-bit eval | exploratory | [#15][f15] |
| 10-18 00:06 | D67 (`bb7ff3c`) | The bf16 merge was a bug: corrected (unmerged, 4-bit) evals become PRIMARY for families S, N, E; merged results kept, labelled superseded | amendment, before any corrected result | confirmatory (S, N, E re-run) | [#16][f16] (supersedes [#13][f13], [#14][f14]) |
| 10-18 09:48 | D69 (`92b868d`) | Post-hoc baselines re-fitted on 4-bit SFT calibration completions | amendment, before any such completion | comparator | – |
| 10-18 09:50 | D71 (`55c8947`) | Review round 2, R2-1–R2-7: eval hygiene (reload fingerprint), SFT revalidation, seed-level statistics (R2-4), uniform-credit seeds (R2-6), … | amendment, before any result it affects | R2-6: exploratory (robustness) | [#18][f18] |
| 10-18 10:06 | D72 (`3f70b9a`) | R2-1's fingerprint and reload-check tolerances | analysis plan, before any fingerprint | reproducibility check | appendix |
| 10-18 10:07 | D73 (`3eaab53`) | X1, a shallow text model of σ\* on A6 / A6b | pre-registration | confirmatory | [#17][f17] |
| 10-18 22:12 | D74 (`d9a36c9`) | X2: within-problem correctness pairs | pre-registration | confirmatory | [#19][f19] |
| 10-19 10:31 | D77 (`42bd61e`) | R2-1: below the log-prob floor, the σ-token criterion decides; the check is enforced | amendment (future fingerprints only) | reproducibility check | appendix |
| 10-19 10:40 | D79 (`de83afd`) | X3, the distillation capacity test | pre-registration | confirmatory (pilot) | [#20][f20] |
| 10-20 08:35 | D89 (`eddda39`) | X4, self-verification on the X2 pairs | pre-registration | confirmatory | [#21][f21] |
| 10-20 23:22 | D91 (`4f016c6`) | X4: a direct mode, a perturbed control, the cluster bootstrap, token cost | amendment, before any X4 data | confirmatory | [#21][f21] |
| 10-20 23:38 | D92 (`5b868f3`) | X2 robustness checks, defined before computing | analysis plan, **after** X2's result | exploratory (post-hoc) | [#22][f22] |
| 10-20 23:45 | D93 (`1faf388`) | Review round 3, R3-1–R3-10: X4 additions; X2 robustness; optimization pilots; Phase 12 scope = aleatoric propagation (R3-4); A6c (R3-5); tolerances; statistics; claim strength | amendment, before any result it affects | – | [#21][f21]–[#26][f26] |
| 10-20 23:49 | D94 (`43cc2af`) | E1c and E1d, the R3-3 optimization pilots | pre-registration | confirmatory (pilot). The paper states E1d as exploratory (R5-7) | [#25][f25] |
| 10-20 23:53 | D95 (`4ff29f2`) | A6c noise-allocation pairs and the noise heuristics: construction and analysis | analysis plan, before the set is built | exploratory (descriptive bars) | [#23][f23] |
| 10-21 06:07 | D99 (`7792502`) | X1 on A6c | analysis plan, before any prediction | exploratory (descriptive) | [#23][f23] |
| 10-21 06:07 | D100 (`7792502`) | Stage A: the scoring audit and the oracle-centred diagnostic | analysis plan, before computing | exploratory (descriptive) | [#24][f24] |
| 10-21 10:44 | D101 (`b24a782`) | D94's decision rule applied: E1d's learning rate is adopted | decision under a registered rule | – | [#25][f25] |
| 10-21 23:14 | D102 (`2970d07`) | E1d's within-problem test on the X2 pairs (T1–T3) | pre-registration | confirmatory | [#26][f26] |
| 10-21 23:20 | D104 (`dcd9b12`) | Stage B, the supervised σ\* ceiling, 3 seeds | pre-registration | confirmatory | [#27][f27] |
| 10-21 23:36 | D106 (`34738a1`) | Stage C pilot: σ-only RL with verifier 1.3.0's σ reward | pre-registration | confirmatory (pilot) | [#28][f28] |
| 10-22 23:44 | D109 (`ff59b4a`) | Stage D: explicit propagation, a positive control and a format-only ablation, 1 seed per arm | pre-registration | confirmatory (pilot) | [#29][f29] |
| 10-25 17:13 | D112 (`87a5933`) | Review round 4, R4-1–R4-9: D-pos-v2 vs B-v2 on mixed-level data, MC-A, the frozen set, framing (R4-8) | amendment, before any result it affects | – | [#30][f30]–[#33][f33] |
| 10-25 17:20 | D113 (`e504498`) | R4-1–R4-5: D-pos-v2 and B-v2, the three-σ grading, the stop/go rule (K1 ∧ C2 ∧ C3), MC-A, the frozen confirmatory set | pre-registration | confirmatory (pilot: the go rule); MC-A a registered diagnostic | [#30][f30], [#31][f31] |
| 10-25 17:31 | D114 (`a4b0a2a`) | Two deviations found while building: eval budget 1,280 tokens; 6 rows dropped | deviation, before any R4 output | – | [#30][f30] |
| 10-25 19:07 | D116 (`dc54969`) | D-pos-v2's eval OOM'd in its backoff; fixed and rerun at batch 16 | deviation, before any completion was scored | – | [#30][f30] |
| 10-26 07:26 | D118 (`f64443e`) | **The R4-3 confirmatory run:** D-pos-v2 and B-v2, 3 seeds each, on the frozen R4-5 set; K1–K8, Holm | pre-registration | **confirmatory (primary)** | [#32][f32] |
| 10-26 07:28 | D119 (`d3de520`) | MC-B: the model writes the problem's program; a sandbox runs it | pre-registration | confirmatory | [#33][f33] |
| 10-26 07:52 | D122 (`f43ccb8`) | MC-B: two deviations found while building | deviation, before any output | – | [#33][f33] |
| 10-26 08:00 | D123 (`e93ab45`) | D118's analysis code built and checked; X1's confirmatory predictions computed | analysis frozen before data | confirmatory | [#32][f32] |
| 10-26 08:34 | D125 (`5ecdeea`) | D118 §6 amended: the primary P is D113's full criterion (K1 ∧ C2 ∧ C3) | amendment, before any confirmatory result | **confirmatory (primary)** | [#32][f32] |
| 10-28 02:52 | D131 (`c9af378`) | D-pos-v2 seed 1's eval hung after a GPU driver reset; rerun on identical code | deviation (operational), before the analysis | – | [#32][f32] |
| 10-28 10:04 | D133 (`be0d338`) | Review round 5, R5-1–R5-10: framing; tail autopsy (R5-2); hybrid oracle (R5-3); RSS fallback (R5-4); cross-model replication (R5-5); cuts (R5-7); reporting rules (R5-8) | amendment, **after** [#32][f32] / [#33][f33], before anything it asks for | R5-2, R5-8: exploratory (descriptive). R5-3: exploratory (diagnostic) | [#34][f34], [#35][f35], [#37][f37] |
| 10-28 10:24 | D135 (`bad601f`) | R5-4's RSS-fallback bounds, σ/RSS ∈ [0.333, 5], chosen on the development sets and frozen before one application to the confirmatory logs | analysis plan; grid designed **after** seeing [#35][f35] | exploratory (post-hoc) | [#36][f36] |
| 10-28 18:07 | D137 (`7892e89`) | The cross-model pilot (R5-5): D-pos-v2 and B-v2 on Llama-3.2-3B-Instruct, seed 0, frozen A6c + A6b; go rule = R4-3's criterion on D-pos-v2; 3 seeds only on GO | pre-registration | confirmatory (pilot) | pending |
| 10-28 18:46 | D138 (`b50cae1`) | R2-1's fingerprint for the Llama base records at sampling batch 4 | deviation (implementation), before any pilot output | – | pending |
| 10-29 00:13 | D139 (`9296d7b`) | D137's go rule runs as soon as the D-pos-v2 eval exists (same code, same logs); SFT seed 1 only on GO | amendment (timing), before the go rule runs | confirmatory (pilot) | pending |

## Summary for the paper

- **Confirmatory, with verdicts the paper reports as pre-registered:**
  - families P ([#7][f7]), S and N ([#16][f16]; [#13][f13] superseded);
  - X1 ([#17][f17]), X2 ([#19][f19]), X4 ([#21][f21]), E1d within problems ([#26][f26]), stage B ([#27][f27]);
  - MC-B ([#33][f33]);
  - **the primary: D118 / D125 ([#32][f32]).**
- **Confirmatory pilots** (one seed; they steered the next step):
  - E1 ([#14][f14] → [#16][f16]);
  - X3 ([#20][f20]);
  - E1c / E1d ([#25][f25], stated as exploratory in the paper, R5-7);
  - stage C ([#28][f28]); stage D ([#29][f29]);
  - R4-1's go rule ([#30][f30]) and MC-A ([#31][f31]);
  - the cross-model pilot (D137, D139; pending).
- **Exploratory:**
  - pre-registration-era results ([#1][f1]–[#6][f6]);
  - the rounding audit ([#8][f8]); the A5 baselines ([#10][f10]); the 4-bit exploratory eval ([#15][f15]);
  - robustness: 3 vs 3 seeds ([#18][f18]); X2 robustness ([#22][f22]);
  - A6c descriptive bars ([#23][f23]); stage A ([#24][f24]);
  - the depth-5 explanation ([#34][f34]);
  - the tail autopsy and hybrid oracle ([#35][f35]);
  - the RSS fallback ([#36][f36]);
  - the cost and MC-B split ([#37][f37]).
- **Amended after a result:** the paper must disclose each.
  - **D47:** P's exploratory baselines.
  - **D54 and D57:** the generator gate, changed twice after failing ([#9][f9], [#11][f11]), then passed under the
    FINAL rule ([#12][f12]).
  - **D92:** X2 robustness.
  - **D135:** the fallback grid, designed after [#35][f35].
- **Not run: family H (H1–H5).**
  - The gate it required passed ([#12][f12]), and D50/A7 trimmed it.
  - But the stated-σ model it was written for was replaced by the Phase 12 stage designs (R3-4: aleatoric propagation
    only), and R5-7 cut the Kelly and finance metrics.
  - **D141 (2026-10-30) declares H not run,** with these reasons. PREREGISTRATION.md requires every hypothesis to be
    reported as supported, not supported or not run.

[f1]: ../FINDINGS.md#1-on-the-untrained-15b-model-a-binary-reward-gives-no-grpo-signal-crps-does-2026-10-12
[f2]: ../FINDINGS.md#2-which-small-model-gives-grpo-a-signal-format-following-and-length-beat-raw-math-skill-2026-10-12
[f3]: ../FINDINGS.md#3-first-grpo-run-49-steps-of-crps_log-on-qwen35-2b-change-nothing-measurable-2026-10-12
[f4]: ../FINDINGS.md#4-30-minutes-of-σ-masked-sft-on-generator-solutions-fixes-format-and-length-the-ab-is-close-and-math-sft-wins-by-rule-2026-10-13
[f5]: ../FINDINGS.md#5-overnight-grpo-on-the-sftd-model-sft-did-the-heavy-lifting-rl-added-a-small-non-significant-id-gain-and-no-calibration-2026-10-13
[f6]: ../FINDINGS.md#6-removing-the-σ-floor-did-not-let-rl-move-calibration-at-matched-steps-a-free-post-hoc-σ-scale-beats-it-and-longer-training-collapses-σ-to-zero-2026-10-14
[f7]: ../FINDINGS.md#7-a-linear-readout-predicts-correctness-at-auroc-092-where-stated-σ-is-at-chance-but-a-random-init-model-does-as-well-the-signal-is-in-the-text-not-demonstrably-in-the-model-2026-10-14
[f8]: ../FINDINGS.md#8-rounding-audit-ground-truth-already-uses-the-displayed-givens-rounding-sets-an-error-floor-well-inside-the-1-band-2026-10-14
[f9]: ../FINDINGS.md#9-phase-10-gate-d50a3-the-generators-σ-fails-on-one-cell-the-2σ-tail-at-depth-3--10-noise-2026-10-14
[f10]: ../FINDINGS.md#10-a5-nested-baselines-the-sft-models-own-text-predicts-its-error-far-better-than-its-stated-σ-2026-10-14
[f11]: ../FINDINGS.md#11-phase-10-gate-rerun-with-noise-capped-at-5-d54-fails-again-on-the-cell-share-rule-only-2026-10-14
[f12]: ../FINDINGS.md#12-phase-10-gate-final-run-d57-passed-generator-200s-σ-is-self-consistent-at-noise--5-2026-10-14
[f13]: ../FINDINGS.md#13-evaluated-on-merged-models-superseded-by-16-s2-withdrawn-split-credit-phase-95-step-3-3-seeds-s1a-s1b-and-s3-supported-on-the-seed-mean-s2-narrowly-supported-sensitive-to-the-seed-mean-definition-σ-still-does-not-track-the-error-2026-10-17
[f14]: ../FINDINGS.md#14-e1-fixed-μ-σ-only-grpo-σ-is-learnable-but-not-beyond-surface-features-family-e-2026-10-16
[f15]: ../FINDINGS.md#15-every-rl-eval-so-far-measured-a-bf16-merge-that-erases--90-of-the-rl-update-evaluated-as-trained-rl-σ-is-5-wider-but-still-does-not-track-the-error-exploratory-d65-2026-10-17
[f16]: ../FINDINGS.md#16-corrected-evals-d67-adapters-unmerged-on-the-4-bit-training-base-are-the-primary-results-s1a-s1b-s3-supported-s2-not-supported-family-n-fails-on-every-seed-family-e-unchanged-2026-10-18
[f17]: ../FINDINGS.md#17-x1-a-shallow-text-model-of-σ-reaches-partial-propagation-on-a6-but-its-slope-is-006-against-an-ideal-1-d73-2026-10-18
[f18]: ../FINDINGS.md#18-uniform-credit-seeds-1-and-2-r2-6-the-corrected-family-s-verdicts-hold-3-vs-3-split-credits-coverage-gain-is-seed-robust-its-ρ-gain-is-consistent-but-tiny-robustness-d71-2026-10-19
[f19]: ../FINDINGS.md#19-x2-no-stated-σ-and-no-text-model-ranks-a-correct-answer-above-an-incorrect-one-to-the-same-problem-a-hidden-state-probe-does-slightly-but-no-better-than-the-same-probe-on-a-random-init-model-d74-2026-10-20
[f20]: ../FINDINGS.md#20-x3-pilot-the-rl-runs-own-base--lora-can-learn-a-σ-that-ranks-errors-ρ-058-capacity-likely-but-only-between-problems-not-within-them-d79-2026-10-20
[f21]: ../FINDINGS.md#21-x4-asked-to-check-its-own-answer-the-sft-model-says-yes-to-almost-everything-its-pyes-does-not-rank-right-answers-above-wrong-ones-to-the-same-problem-with-or-without-recomputation-d89--d91-2026-10-21-night
[f22]: ../FINDINGS.md#22-x2-robustness-post-hoc-exploratory-d92-asking-the-other-7-samples-beats-every-single-sample-readout-within-problems-clearly-separated-pairs-sharpen-the-ranking-but-change-no-verdict-x2s-cis-were-already-problem-level-2026-10-21-night
[f23]: ../FINDINGS.md#23-x1-and-the-noise-heuristics-on-a6c-no-surface-reader-propagates-noise-allocation-x1s-a6-partial-propagation-does-not-transfer-d95--d99-2026-10-21
[f24]: ../FINDINGS.md#24-phase-12-stage-a-crps_log-targets-σ-but-one-realized-truth-barely-tells-a-2-off-σ-from-σ-the-expected-score-against-the-known-predictive-distribution-is-523-less-noisy-d100-2026-10-21
[f25]: ../FINDINGS.md#25-e1c--e1d-pilots-r3-3-at-a-10-learning-rate-grpo-with-clean-credit-learns-a-σ-that-ranks-errors-ρ-059-vs-e1as-001-at-x3s-supervised-level-but-overconfident-and-still-below-the-text-baseline-batch-std-scaling-does-nothing-d94--d101-2026-10-21
[f26]: ../FINDINGS.md#26-e1ds-σ-does-not-rank-a-right-answer-above-a-wrong-one-to-the-same-problem-within-problems-it-sits-exactly-at-x3s-supervised-level-0512-vs-0513-far-below-self-consistency-0643-d102-2026-10-2122-night
[f27]: ../FINDINGS.md#27-phase-12-stage-b-sft-on-σ-itself-does-not-teach-propagation-the-supervised-ceiling-reads-the-stated-percentages-states-σ--065σ-and-scores-below-the-rss-heuristic-and-x1-d104-2026-10-22-night
[f28]: ../FINDINGS.md#28-phase-12-stage-c-pilot-σ-only-rl-with-the-predictive-σ-reward-beats-its-sft-start-but-learns-to-copy-the-smallest-stated-percentage-it-does-not-propagate-scores-below-rss-and-x1-and-its-policy-turns-deterministic-by-step-40-d106-2026-10-2223-night
[f29]: ../FINDINGS.md#29-phase-12-stage-d-training-the-explicit-propagation-line-does-not-give-propagation-either-the-model-writes-the-line-every-time-but-has-learned-exponent-templates-not-derivatives-never-reads-mixed-levels-and-its-σ-stays-below-rss-and-x1-format-only-sft-yields-degenerate-lines-and-σ--0-d109-2026-10-23-night
[f30]: ../FINDINGS.md#30-r4-1-with-mixed-level-training-data-and-visible-sensitivity-derivations-explicit-propagation-works-d-pos-v2s-stated-σ-propagates-on-all-three-graded-σ-values-and-passes-r4-3s-go-rule-but-gross-misses-on-deep-value-dependent-chains-keep-its-expected-score-at-the-rss-heuristic-not-above-it-the-implicit-control-b-v2-learns-noise-allocation-only-d113-2026-10-25-night
[f31]: ../FINDINGS.md#31-r4-4-mc-a-perturbing-the-inputs-and-re-asking-the-model-does-not-beat-the-rss-heuristic-the-models-own-arithmetic-errors-swamp-the-input-noise-the-stopping-rule-fires-k-is-not-increased-the-exact-solver-on-the-same-perturbations-reaches-σs-level-and-the-raw-spread-is-the-best-end-to-end-σ-and-correctness-signal-d113-7-2026-10-26-night
[f32]: ../FINDINGS.md#32-r4-3-confirmatory-3-seeds-per-arm-frozen-r4-5-set-p-is-confirmed-d-pos-v2s-stated-σ-propagates-measurement-noise-on-fresh-problems-a6c-slope-082-but-its-expected-score-does-not-beat-rss-k6-and-sits-below-x1-k7-estimate-negative-and-at-the-held-out-depth-5-it-loses-to-both-although-it-propagates-there-d118--d125-2026-10-2728
[f33]: ../FINDINGS.md#33-r4-7-mc-b-the-base-model-writing-the-problems-program-fails-8591-of-programs-fail-mostly-at-the-signature-about-05-compute-the-chain-and-mc-b-loses-to-rss-x1-d-pos-v2-and-mc-a-robust-on-every-pre-registered-test-d119--d122-2026-10-28
[f34]: ../FINDINGS.md#34-the-depth-5-discrepancy-006-on-the-development-set-080-on-the-frozen-set-for-the-same-seed-0-adapter-is-a-fragile-estimator-not-a-different-model-or-a-different-construction-a-few-high-leverage-pairs-with-reversed-direction-drive-the-100-pair-ols-slope-robust-slopes-are-048-and-075-and-the-two-sets-share-the-same-reversal-count-and-cell-mean-under-propagation-descriptive-d133-item-2-2026-10-28
[f35]: ../FINDINGS.md#35-r5-2--r5-3-on-the-frozen-d-pos-v2-logs-the-tail-is-wrong-value-dependent-sensitivities-formula-43-substituting-the-models-own-wrong-intermediates-19-with-chain-selection-σ-decimal-shifts-and-missing-answers-behind-most-of-the-rest-making-only-the-sensitivities-exact-closes-51-of-the-rss-to-σ-gap-and-beats-both-rss-and-x1-levels-are-already-right-and-exact-aggregation-alone-makes-it-worse-descriptive-and-diagnostic-d133-2026-10-28
[f36]: ../FINDINGS.md#36-r5-4s-rss-fallback-post-hoc-exploratory-bounds-σrss--0333-5-chosen-on-the-development-sets-and-frozen-first-d135-it-lifts-the-expected-score-above-rss-and-x1-on-the-frozen-set-but-erases-most-of-the-measured-propagation-a6c-slope-084--014-because-a-fallback-on-one-member-of-a-pair-breaks-the-within-pair-contrast-2026-10-28
[f37]: ../FINDINGS.md#37-r5-8-reporting-from-existing-logs-mc-b-fails-mostly-at-the-interface-8084-of-the-programs-that-run-97-compute-the-wrong-chain-sample-agreement-is-the-strongest-correctness-signal-but-it-costs-about-8-the-tokens-of-one-answer-010-within-problem-accuracy-self-consistency-and-008-auroc-on-a6c-mc-as-raw-spread-over-the-best-single-sample-or-free-signal-descriptive-d133-item-5-2026-10-28

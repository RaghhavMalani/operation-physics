# Paper outline (R5-1 framing)

**Working title:** *When Language Models Quote Uncertainty: Separating Reasoning Errors from Measurement Noise*.
"Quoting Volatility" stays the project name (R5-1, D133).

**The central message** (R5, both reviewers):
- **The tension:** learning how uncertainty should propagate (structural correctness) is not the same as forecasting
  it well (forecast quality).
- **Equal prominence:** D-pos-v2's confirmed propagation (A6c +0.816) and its unsupported score secondaries (K6 vs
  RSS, K7 vs X1) are reported with equal weight. That tension is the contribution.

**Scope:** every claim is scoped to this model (Qwen2.5-Math-1.5B-Instruct, 4-bit, LoRA), this synthetic benchmark
and these training procedures, until the cross-model replication (R5-5) says otherwise.

**Sources for every number:**
- [FINDINGS.md](../FINDINGS.md);
- the confirmatory / exploratory status of each analysis: [decision_timeline.md](decision_timeline.md);
- the headline figure: `results/figures/headline.{png,pdf}`, built by `scripts/figures/headline.py` from committed
  results only. Its sidecar `headline_data.json` traces every point.

**Venue (R5-9):** an arXiv preprint around 2026-11-20, then a 2027 workshop. Main track only after the cross-model
replication and genuine family or formula transfer.

---

## 1. Introduction

**Content:**
- **The question:** where should an LLM's numeric uncertainty come from? Can one stated σ track the true uncertainty
  problem by problem?
- **Why a synthetic benchmark:** the generator knows each problem's σ\* exactly, and can dial measurement noise
  independently of chain depth. Real datasets cannot.
- **The three findings, one per panel of the headline figure:**
  1. **Optimization.** With sequence-level credit, a proper scoring rule did not stop σ collapsing. Split credit fixes
     coverage, but not ranking.
  2. **Epistemic.** No single-answer readout tells a right answer from a wrong one to the same problem. Cross-sample
     agreement does, at ~8× the tokens.
  3. **Aleatoric.** Visible sensitivity derivations give a σ that propagates measurement noise on fresh problems, yet
     its expected score is not shown to beat a surface heuristic. The loss is in value-dependent derivatives.
- **Contributions, stated narrowly:**
  - the benchmark with known σ\* and its counterfactual pair sets;
  - the separation of reasoning error from measurement noise;
  - the propagation-vs-forecast gap with its tail autopsy;
  - a pre-registered, chronologically documented protocol.

**FINDINGS:** #32 (central), #35, #19 / #22, #16 / #18.

**Figure:** the headline (three panels), as Figure 1.

## 2. Related work

- **Verbalized confidence and RL calibration:**
  - RLCR (Damani et al., 2507.16806);
  - Rewarding Doubt (2503.02623);
  - Pitfalls of verbalized confidence priors (2609.32470);
  - Overconfidence is Key (2405.02917).
- **Distributional answers and CRPS as an RL reward:**
  - DAR (Park et al., 2605.20740). CRPS as an RL reward is not new; DAR's uncertainty is rollout spread, not a stated
    σ.
  - CARE-PPO.
- **Interval evaluation:** FermiEval (2510.26995) and QuantSightBench (2604.15859).
- **Sampling-based confidence:** Self-Consistency (2203.11171).
- **Executable computation:** PAL (2211.10435) and Program of Thoughts (2211.12588). These limit MC-B's novelty.
- **Propagation of distributions:** JCGM 101:2008.
- **Proper scoring:** Gneiting & Raftery (2007), *JASA* (D134).
- **Silent GRPO groups:** Dr.GRPO / DAPO (2607.00152, 2503.14476). FINDINGS #1 is motivation, not a contribution.
- **Before citing (R5-10, R4-9):** every reference is verified. That includes Guo et al. (2604.05306), Grünefeld
  et al. (2605.07776), Kapoor et al. (2406.08391) and Kamath et al. (ACL 2020).

## 3. Benchmark, verifier and statistics

**The generator** (formula graph, depth 1–5, held-out depth 5):
- generator 2.0.0's measurement-noise mode (noise ≤ 5%, D54);
- σ\* by Monte Carlo through the chain;
- the self-consistency gate:
  - **FINDINGS #9 and #11:** it failed twice;
  - **FINDINGS #12:** it passed under the FINAL rule of D57;
  - **Disclose both amendments** (decision_timeline).

**The verifier:**
- the crps_log scoring rule (D15 / D29) and the σ floor (D37 / D38);
- the units gate and parse rules.

**The oracle-centred expected score** (#24):
- **Why:** one realized truth barely tells a 2×-off σ from σ\*, and the expected score is 5–23× less noisy.
- **Normalization:** score differences are reported against the RSS-to-σ\* gap (0.00296 on the frozen set).

**The counterfactual pair sets:**
- **A6:** value changes (D58);
- **A6b:** invariance pairs, where the slope must be ≡ 0 (D59);
- **A6c:** noise-allocation pairs (D95). The surface is identical within a pair, so no surface reader can move (#23).
- **The frozen confirmatory R4-5 set:** 1,020 problems each of A6c / A6 / A6b (D113).

**Statistics (R5-8, R3-7):**
- clustered bootstraps over problems or pairs; seed means over 3 seeds;
- never 5,390 pairs or 3 seeds treated as independent;
- Holm within families.

**FINDINGS:** #8, #9, #11, #12, #23, #24.

**Table:** the pair sets and what each one isolates.

## 4. Act 1: optimization. σ rides on the answer's advantage

**The claim, scoped to the frozen-answer setting (R5-7):**
- **The failure:** with sequence-level credit, the proper scoring rule did not stop σ collapsing to 0 (#6, the
  "± 0" collapse; post-hoc needs ×585).
- **The fix:** split credit gives σ tokens their own advantage. 1σ coverage goes from 10.8% to 41.8%, +31.0 pts
  [+26.8, +35.3], in 9 / 9 cross-seed pairs (#16, #18).
  - **But ρ(σ, error) stays tiny:** S2 is not supported.
  - **And coverage stays far below the nominal 68%.**
- **Corrected evals only:** the bf16 merge pitfall (#15) is a box, with the superseded merged results labelled (#13,
  #14).
- **σ is learnable:**
  - **E1 (#14 → #16):** σ is learnable, but not beyond surface features.
  - **E1c / E1d (#25):** a 10× learning rate gives ranking at X3's supervised level. **Exploratory** (R5-7).
- **The judge:** RL calibration claims are judged against the A5 text baselines on the same answers (#10, family N
  fails, #16), not against stated σ alone.

**FINDINGS:** #1 (motivation), #5, #6, #10, #13 (superseded), #14, #15, #16, #18, #25.

**Figures:**
- headline panel A;
- `results/corrected/robustness_3v3/trajectories.png` (training σ and the σ = 0 share per seed).

## 5. Act 2: does the model know which of its answers is right?

**The claim:**
- **The test design:** problem difficulty is held fixed (X2: 516 problems with right and wrong samples, 5,390 pairs;
  CIs clustered over problems).
- **Single-answer readouts sit near chance:** the stated σ (SFT, split credit), the text model (A5), self-verification
  (X4), the RL σ suffix (E1d) and the supervised σ suffix (X3) all score 0.49–0.53.
- **The hidden-state probe** reaches 0.547, but a random-init model reads 0.530 from the same tokens, so "the model
  knows" is not shown (#7, #19).
- **Self-consistency reaches 0.643 [0.616, 0.671],** at 8.0× the tokens of one answer (#22, #37).
- **Across problems is different:** the ranking there is difficulty prediction (X3, E1d: ρ(σ, error) 0.58–0.59 on the
  frozen eval answers, 0.35 on the X2 answers), not answer-specific (#20, #25, #26).

**FINDINGS:** #7, #10, #19, #20, #21, #22, #26, #37.

**Figures:**
- headline panel B;
- `results/probe_sft/probe_by_layer.png` (with its random-init control);
- to build from `results/r5_diag/cost.json`: the k-cost curve (MC-A k = 2 / 4 / 8 against tokens and GPU-seconds;
  R5-8).

## 6. Act 3: propagation of measurement noise. Structure is not forecast quality

**What failed first** (each pre-registered, one seed unless noted):
- **X1:** a text model of σ\* gets A6 "partial propagation" (0.06) only from magnitudes. It is 0 on A6c (#17, #23).
- **Stage B:** SFT on σ\* reads the stated percentages (σ ≈ 0.65·σ\*) and scores below RSS and X1 (#27; 3 seeds).
- **Stage C:** σ-only RL copies the smallest percentage (#28).
- **Stage D:** an explicit line with no value-dependent derivations learns exponent templates (#29).
- **Why:** the training data never contained the behaviour (R4-1).

**What worked:**
- **R4-1's mixed-level data with visible sensitivity derivations:** the pilot passes the go rule (#30).
- **R4-3, the confirmatory run (#32; D118 / D125; 3 seeds, frozen set):**
  - **P is confirmed:** A6c slope +0.816 [+0.708, +0.923].
  - **C3:** A6b inside ±0.3.
  - **Close to σ\*:** coverage 63–70% at 1σ.
- **With equal prominence, the score secondaries:**
  - **K6 (beats RSS) is not rejected:** +0.0002 [−0.0002, +0.0006]. It closes 7% of the RSS-to-σ\* gap (90% CI −7% to
    +20%).
  - **K7's estimate is negative:** X1 scores better, descriptively.
  - **B-v2** (no visible derivations) gives a 0.42 slope and scores below RSS.

**Where the loss is:**
- **The tail autopsy (#35, R5-2, descriptive):** the worst 5% of problems hold 69–83% of the regret.
  - wrong value-dependent sensitivities: 43% formula, 19% substituted own wrong intermediates;
  - then decimal shifts in the copy, chain selection and missing answers.
  - **The 95%-trimmed scores** keep the ordering.
- **The hybrid oracle (#35, R5-3, a diagnostic and never a method):** with the sensitivities exact, σ beats RSS and
  X1, closes 51% of the gap, and the slope rises to 0.97.
- **The RSS fallback (#36, post-hoc exploratory):** it lifts the score above RSS and X1, but cuts the A6c slope to
  0.14. A guardrail must report both numbers.

**Executing instead of predicting:**
- **MC-A (#31):** the model's own arithmetic errors swamp the input noise. A diagnostic.
- **MC-B (#33, #37):** 80–84% of programs fail at the interface, and 97% of the programs that run compute the wrong
  chain. Nested MC is future work (R5-7).

**Depth 5:**
- **It is extrapolation in chain length,** not to new physical laws (R5-8).
- **Per-depth slopes are fragile:** report OLS and Theil–Sen (#34; D137 item 4).
- **The score is below RSS and X1 there,** with CIs below 0 (#32).

**Cross-model replication (R5-5; D137 / D139):** Llama-3.2-3B-Instruct, D-pos-v2 and B-v2, seed 0, on the frozen
A6c + A6b.
- **The go rule:** R4-3's criterion on D-pos-v2. Three seeds only on GO.
- **Reported separately:** format-following and arithmetic accuracy after SFT, and the R5-2 tail categories.
- **Status: pending.** The section is written either way (go or no-go).

**FINDINGS:** #17, #23, #27, #28, #29, #30, #31, #32, #33, #34, #35, #36, #37, and the pending pilot.

**Figures:**
- headline panel C;
- `results/stage_r4/dpos2/seed{0,1,2}/confirmatory/{calibration_a6c_pairs,reliability_all_splits}.png`.
- **To build** from committed JSON:
  - the R5-3 component-swap chart (`results/r5_diag/tail.json`);
  - the per-depth OLS / Theil–Sen slopes (`results/r5_diag/depth_ts.json`);
  - the MC-B failure split (`results/r5_diag/cost.json`).

## 7. Discussion

- **Two error streams:**
  - **Measurement noise** is handled once the structure is right: levels, root-sum-square and copying are correct.
  - **Reasoning errors in derivatives of value-dependent relations** cost the forecast. The same wrong intermediate
    corrupts both μ and its sensitivity (#35).
- **Why sampling works and single readouts do not:** right answers cluster and wrong ones scatter (#22). This
  information is not in a single answer, its σ, its hidden state or its self-check, for this model.
- **What would close the gap:**
  - correct sensitivities (#35's counterfactual), not fallbacks (#36);
  - an executor that computes the chain (MC-B did not get there; future work in ChatGPT's sequence: semantic
    validity, then execution tests, then input sampling).

## 8. Limitations

- **Scope:**
  - **One model family** until R5-5 passes: Qwen2.5-Math-1.5B-Instruct, 4-bit, LoRA, an 8 GB GPU.
  - **A synthetic domain:** three formula families. The optics family is cut (R5-7). Depth 5 is chain-length
    extrapolation only.
- **The data:**
  - A6c is in-distribution for the R4 arms (D112).
  - Development-set and frozen-set estimates differ at single depths (#34).
- **Seeds:**
  - **One seed** for the pilots: E1, X3, E1c / E1d, C, D, R4-1, the cross-model pilot.
  - **Three seeds:** S, uniform credit, stage B, R4-3.
- **Statistics:**
  - Within-problem tests use sampled answers at T = 1.
  - X1's score difference against RSS has no committed CI (headline panel C).
- **Not run:**
  - **Family H** (H1–H5): not run (D141; see decision_timeline).
  - **Kelly / finance metrics:** cut (R5-7).

## Appendix A. Pre-registration ledger

Every pre-registered hypothesis is marked supported, not supported, or not run, in PREREGISTRATION.md's and
DECISIONS.md's order:
- families P, S, N, E1, X1–X4, E1c / E1d, D102, stages B / C / D, R4-1–R4-5, D118 (K1–K8, P), MC-B, D137;
- **family H: not run.**

Source: [decision_timeline.md](decision_timeline.md), which also lists every amendment made after a result (D47,
D54, D57, D92, D135).

## Appendix B. Reproducibility (R5-8)

- **Reload and provenance checks:** R2-1's training-time fingerprint and reload check (D72, D77), and provenance in
  every tape.
- **Pinned bases:** with content hashes (D78; D137 for Llama).
- **The merge results:** the bf16 merge pitfall with its superseded merged results kept and labelled (#13–#15, D67,
  D68).
- **Frozen eval sets:** with checksums.
- **Compute:**
  - the hardware: one RTX 4060 laptop GPU with the temperature guard;
  - one eval hang after a driver reset, rerun on identical code (D131).
- **The full decision timeline.**

---

## Claims to avoid

| do not write | write instead / why | source |
|---|---|---|
| "small models cannot propagate uncertainty" | scope to this model, benchmark and training | R5-1 |
| "explicit propagation gives better calibration / a better forecast" | it propagates (K1 ∧ C2 ∧ C3); its expected score is not shown to beat RSS (K6), and X1's estimate is higher (K7) | R5-1, #32 |
| "D-pos-v2 ties / matches / is level with RSS" | "not rejected" or "not distinguishable": failure to reject is not equivalence | D133, #32 |
| "X1 beats D-pos-v2" as a test result | K7 tested the other direction; say "X1 scores better, descriptively" | #32 |
| "all bf16 merging is unsafe" | "merging this LoRA into the bf16 SFT model erased > 90% of the RL update in our pipeline" | R5-1, #15 |
| "uncertainty is not felt inside one forward pass" | the narrower R4-8 framing: single-response estimators do not reveal answer-specific correctness here | R4 synthesis, R4-8 |
| "the model knows its error" (from probes) | a random-init model of the same architecture reads nearly as much from the same tokens | #7, #19 |
| "self-consistency is a free confidence signal" | 0.643 within problems at 8.0× the tokens of one answer | #22, #37 |
| "split credit calibrates σ" | it raises 1σ coverage to 41.8% (nominal 68%); ρ(σ, error) stays ≤ 0.08 and S2 is not supported | #16, #18 |
| "RL improves calibration" against stated σ alone | judge against the A5 text baselines on the same answers (family N fails) | D49, D50/A5, #16 |
| E1d's learning-rate result as confirmatory evidence for joint RL | exploratory; Act 1 is scoped to the frozen-answer setting | R5-7, #25 |
| "propagates / does not propagate at depth 5" | report OLS and Theil–Sen per depth and call them fragile | D137 item 4, #34 |
| depth 5 as generalization to new physics | extrapolation in chain length within the same three families | R5-8, #32 |
| the RSS fallback as an improvement, alone | report its score gain and its A6c slope (0.14) together | #36 |
| the exact-sensitivity variant as a method | a counterfactual that locates the loss | R5-3, #35 |
| MC-A as a primary contribution; "computing beats predicting" | MC-A is a diagnostic (#31); MC-B failed at the interface and semantics (#33, #37) | R4 call, #31, #33 |
| "CRPS as an RL reward is our contribution" | DAR uses CRPS over rollouts; ours is a stated σ against a known σ\* | CLAUDE.md prior work |
| silent GRPO groups (#1) as a finding | known (Dr.GRPO / DAPO); motivation only | CLAUDE.md prior work |
| any statistic treating 5,390 pairs or 3 seeds as independent | clustered over problems / pairs; seed means | R5-8, R3-7 |
| cross-model claims before R5-5 passes | "on Qwen2.5-Math-1.5B"; the Llama pilot is reported as a pilot, go or no-go | R5-5, D137 |
| "the generator's σ\* was validated" without the history | the gate failed twice and passed under the amended FINAL rule (D54, D57) | #9, #11, #12 |

# Calibrated physics reasoning via RLVR — build spec

Working name: `physcal` (rename freely).

## What this project is

Train a small open LLM with reinforcement learning to solve procedurally generated physics problems and
report an **honest uncertainty** with every answer (`v = 12.4 ± 0.8 m/s`). Rewards come from a
deterministic verifier, never from humans or another LLM.

Research questions the code must be able to answer:
1. Does a dense, proper-scoring-rule reward beat a binary correct/incorrect reward?
2. Is the model's stated σ calibrated, and does it stay calibrated out of distribution (held-out formula families, deeper chains)?
3. Do step credit and an adaptive curriculum measurably help?
4. Does an HMM knowledge-tracing curriculum reach the same accuracy in fewer steps than EMA or uniform sampling?
5. Where in the network does the model "know" the answer and its own error, and does RL move that?

## Hard constraints

- **Hardware:** single RTX 4060, 8 GB VRAM. Everything must run on it. Cloud GPUs are only for final benchmark runs.
- **Model:** `Qwen2.5-1.5B-Instruct` (fallback `Qwen2.5-0.5B-Instruct`), 4-bit, LoRA.
- **Trainer:** Unsloth GRPO (TRL `GRPOTrainer` under the hood). Group size 4–8, max completion 512 tokens.
- **Determinism:** every problem is reproducible from `(generator_version, seed)`. Every rollout is logged.
- Python 3.11, `uv` for deps. Libraries: `sympy`, `pint`, `scipy`, `numpy`, `unsloth`, `trl`, `pytest`, `hypothesis`.

## Repo layout

```
physcal/
  env/
    formulas/        # one module per family: kinematics, dynamics, energy, circuits, optics, thermo
    graph.py         # formula graph + chaining
    simulators.py    # numerical ground truth (no closed form)
    render.py        # problem text: phrasing templates, unit randomization, distractors
    sampler.py       # curriculum samplers: uniform, EMA (baseline), HMM (phase 8)
    kt.py            # HMM knowledge tracing (phase 8)
  monitor/
    run_hmm.py       # run-level HMM: improving / plateau / reward hacking (phase 8, optional)
  interp/
    extract.py       # cache hidden states per layer (phase 9)
    probes.py        # linear probes per layer (phase 9)
    plots.py         # probe accuracy vs layer, base vs RL
  verify/
    parse.py         # extract steps + final answer from completion
    units.py         # pint gate
    scoring.py       # CRPS / log score
    reward.py        # combines everything, returns float + breakdown dict
  train/
    grpo.py          # Unsloth GRPO entrypoint, config via YAML
    configs/
  tape/
    tape.py          # JSONL rollout log + replay
  eval/
    run_eval.py      # ID, held-out family, depth extrapolation
    calibration.py   # coverage, PIT histogram, reliability plot
tests/
```

## Component specs

### 1. Formula graph generator (`env/graph.py`, `env/formulas/`)
- Each formula is a sympy relation over named quantities with dimensions, e.g. `v = u + a*t`.
- A problem = a path through the graph: sample given quantities, solve for an intermediate, feed it forward, ask for the final target. **Chain depth 1–5** is the difficulty knob.
- Randomize: values (log-uniform within physical ranges), units (mixed SI prefixes, km/h, g, etc.), phrasing, and 0–2 **distractor quantities** irrelevant to the answer.
- Ground truth is computed symbolically in SI with `pint`, then converted to the requested unit.
- Each problem returns: `text`, `target_symbol`, `target_value_si`, `target_unit`, `intermediates: {symbol: value_si}`, `family`, `depth`, `seed`.
- **Held-out split:** reserve at least one whole family (suggest `optics`) and depth 5 for eval only. Training must never sample them.

### 2. Simulators (`env/simulators.py`)
Problems with no closed form, ground truth via `scipy.integrate.solve_ivp` (rtol=1e-10, atol=1e-12):
- projectile with quadratic drag (range, max height, time of flight)
- damped driven oscillator (amplitude after time t)
- two coupled pendulums (energy in pendulum B at time t)
Assert numerical convergence in tests by halving tolerances and checking agreement.

### 3. Output format (enforced in the system prompt)
```
<steps>
a = 2.45 m/s^2
t = 4.1 s
</steps>
<answer>12.4 ± 0.8 m/s</answer>
```
The problem text tells the model which intermediate symbols to report.

### 4. Verifier (`verify/`)
- **Parse:** a malformed completion gets reward 0 with reason `parse_fail`. Accept `±`, `+/-`, and scientific notation.
- **Units gate:** convert the answer with pint. If dimensions mismatch → final score 0 with reason `unit_mismatch`. Correct dimensions in a different unit (km vs m) are fine.
- **Scoring rule (main):** Gaussian CRPS in relative units.
  - `y` = truth, `μ, σ` = prediction, all in SI. Normalize: `e = (μ − y)/|y|`, `s = max(σ/|y|, 1e-3)`.
  - `CRPS(z) = s·[z(2Φ(z) − 1) + 2φ(z) − 1/√π]`, with `z = e/s`.
  - `score_final = −CRPS`, clipped below at −5.
  - **Only affine transforms of the score are allowed.** A non-linear map like `exp(−CRPS)` breaks propriety, and the model would no longer be rewarded for honest σ. GRPO's group normalization is affine, so it is fine.
  - Ablation variants behind a config flag: `binary` (correct within 1% relative), `log_score` (Gaussian NLL, clipped), `abs_rel_error` (no σ).
- **Step credit:** for each reported intermediate, award credit if within 2% relative of truth. `score_steps = mean over requested intermediates`.
- **Total:** `reward = 1.0·score_final + 0.2·score_steps`. The step weight is configurable and must stay small enough that it can never outweigh the final answer.
- `reward()` returns `(float, breakdown_dict)`. The breakdown is logged to the tape.

### 5. Curriculum sampler (`env/sampler.py`)
- Track an EMA pass rate per `(family, depth)` cell, where pass = binary correctness.
- Sampling weight `w = p·(1 − p) + 0.02`. This peaks at p = 0.5, where GRPO groups have reward variance. All-correct and all-wrong groups give zero advantage and waste compute.
- Log the **fraction of zero-variance groups** per step. This is a first-class training metric.
- The sampler is an interface (`Sampler.next_batch()`, `Sampler.observe(results)`), so the HMM sampler from phase 8 drops in without touching training code. EMA stays as the baseline.

### 8. HMM knowledge tracing curriculum (`env/kt.py`, phase 8)
Bayesian Knowledge Tracing: one 2-state HMM per `(family, depth)` skill cell.
- Hidden state: `not_mastered` / `mastered`. Transition `P(learn)`, no forgetting by default (configurable `P(forget)`).
- Observation per GRPO group: number of binary-correct answers out of G, modeled as Binomial with
  `p = 1 − P(slip)` if mastered, `p = P(guess)` if not.
- Posterior `P(mastered)` updated each step with the forward algorithm (exact, two states, cheap).
- Parameters fit by EM (Baum-Welch) on the rollout tape every N steps, with priors that keep `P(slip)`, `P(guess)` < 0.3 so the states stay identifiable.
- Decisions, all thresholds in config:
  - uncertain (0.2 < P < 0.8) → highest sampling weight
  - mastered (P > 0.95) → low maintenance weight, unlock depth + 1 for that family
  - stuck (P < 0.2 for K consecutive updates) → back off one depth
- **The HMM never touches the reward.** It only decides which problems are sampled. The reward stays pure verifier output.
- Tests: synthetic learner with known slip/guess/learn parameters; EM must recover them within tolerance; posterior must cross 0.95 within a bounded number of steps after the synthetic switch.

Optional run-level monitor (`monitor/run_hmm.py`): 3-state Gaussian HMM (`improving`, `plateau`, `reward_hacking`) over per-step features: mean reward, mean predicted relative σ, 1σ coverage. Alert, don't act: log the state and warn when `reward_hacking` probability > 0.5 (reward rising while σ inflates and coverage drifts from 68%).

### 9. Layer-wise linear probes (`interp/`, phase 9)
- Extract hidden states for a fixed probe set of ~5,000 problems (stratified by family and depth) at **every layer**, at the token just before `<answer>`. Cache to disk as float16 `.npy`, one file per layer. Batch small; this must fit on 8 GB.
- Two probes per layer, both linear, trained with scikit-learn on CPU:
  - **Probe A:** ridge regression predicting `log10(|true value in SI|)`. Report R².
  - **Probe B:** logistic regression predicting "the model's final answer will be wrong (>1% relative)". Report AUROC.
- Split probe data by problem seed (no leakage), and include held-out families in a separate probe test set.
- Run on: base model, and the RL-trained LoRA checkpoints (merge LoRA before extraction).
- Controls: probes on a randomly initialized model of the same architecture, and probes on shuffled labels. Without these, the plots don't mean anything.
- Output: one figure, two panels (R² vs layer, AUROC vs layer), base vs RL as two lines. This is the headline visual.

### 6. Rollout tape (`tape/tape.py`)
- JSONL, one line per completion: `run_id, step, seed, generator_version, model_id, lora_hash, problem, completion, reward, breakdown`.
- `replay(run_id)` regenerates every problem from its seed and re-scores the completions. It must reproduce the logged rewards exactly; add a test for this.

### 7. Eval (`eval/`)
Splits: in-distribution, held-out family, depth extrapolation (train ≤4 → test 5), simulator problems.
Metrics per split:
- accuracy (within 1%), median relative error
- **1σ and 2σ interval coverage** (target 68% / 95%)
- PIT histogram and reliability plot (PNG)
- unit-mismatch rate, parse-fail rate
- reward-hacking checks: mean predicted σ over training (watch for σ inflation), answers that copy given values

## Phases (finish and test each before moving on)

1. **Environment.** Generator for kinematics + dynamics + energy, renderer, pint units. Tests: 1,000 random seeds per family, ground truth matches sympy numeric evaluation; property tests with hypothesis.
2. **Verifier.** Parser, units gate, CRPS, step credit. Tests: hand-written edge cases (wrong units, missing ±, negative values, scientific notation). Test that the expected CRPS is minimized at the true σ, using Monte Carlo.
3. **Baseline eval.** Run the untrained model through `run_eval.py`. Save numbers — every later claim is relative to this.
4. **First GRPO run** on the 4060 with the binary reward. Goal: reward curve goes up, zero-variance fraction is logged.
5. **Main run** with the CRPS reward + step credit + curriculum.
6. **Remaining families + simulators,** held-out eval.
7. **Ablations:** binary vs abs_rel_error vs CRPS; ± step credit; ± curriculum. Same seeds, same step budget.
8. **HMM curriculum.** Knowledge tracing sampler + tests, then the ablation: uniform vs EMA vs HMM at an equal step budget. Metric: steps to reach a target accuracy, plus final calibration. Optional run-level monitor.
9. **Layer probes.** Hidden-state extraction, probes A and B, controls, base vs RL figure.
9.5. **σ credit assignment and representation** (runs FIRST, ahead of the open phase 5–9 items; approved
   2026-10-14). See "Research extension" below.
10. **σ\* audit and measurement-noise mode** (generator 2.0.0).
11. **Rigor upgrades:** 3 seeds, held-out optics family, 1,000-problem frozen eval splits, calibration split.
12. **Train the stated-σ model** (curriculum over depth and noise, 3 seeds).
13. **Five-way uncertainty comparison** (stated σ, rollout spread, hidden-state probe, LoRA ensemble, post-hoc).
14. **Decision value and stress tests.**
15. **Write-up** (former phase 10: README with results tables, calibration plots, the probe figure, and an honest
    limits section; now expanded, see below).

Do not start phase 8 or 9 until phase 7 has produced numbers. (Phase 9.5 is the approved exception: its probe
step borrows from phases 9/13 early.)

## Research extension: Quoting Volatility (phases 9.5–15)

Working title: *Quoting Volatility: Training Language Models to Price Their Own Uncertainty*. Merged from
PHASES_10-15.md on 2026-10-14. Do not start Phase 10 until FINDINGS #6 (the σ-floor run) exists (it does).

### Phase 9.5: σ credit assignment and representation (runs FIRST)

Context, FINDINGS #6: removing the σ floor did not move calibration. Stated σ has ~zero rank correlation with
error; post-hoc needs ×585 to reach 68% coverage (bimodal errors: exactly right or far off); with longer training
the policy collapsed to "± 0" even though σ = 1% scored better on those answers. Diagnosis: (a) σ rides on the
answer's advantage (sequence-level credit); (b) one Gaussian σ is the wrong shape for spike-and-slab errors.

1. **Monitoring (approved):** alert and log when the σ = 0 share exceeds 50%; report the post-hoc scale as the
   standing calibration baseline in every eval.
2. **Hidden-state probe, early (from Phase 13):** on the SFT-only model, extract the pre-answer hidden state per
   layer on the calibration split and train a linear/MLP probe to predict (i) correct within 1% and
   (ii) log|error|. Controls: shuffled labels, random-init model. Question: does the model "know" its error even
   though its stated σ does not? Record as FINDINGS #7.
3. **Split credit (verifier unchanged):** decompose the reward r = r_answer + r_sigma, where
   r_answer = crps_log(μ, σ_ref) with a fixed reference σ_ref (configurable, e.g. 1% relative) and
   r_sigma = crps_log(μ, σ) − crps_log(μ, σ_ref). Reasoning/answer tokens get the group-normalized advantage of
   r_answer; σ tokens (located by parsing) get the group-normalized advantage of r_sigma. Implement as a
   GRPOTrainer subclass with a custom loss. Tests: on synthetic batches, σ-token gradients depend only on
   r_sigma; answer-token gradients only on r_answer; with σ_ref = σ the method reduces to the original loss on
   answer tokens. Run: identical to the Oct-13 run except split credit; matched-step comparison at step 287;
   3 seeds once the single-seed run looks sane.
4. **Two-channel format (needs a short re-SFT; next):** the model states `value ± sigma unit, confidence p`.
   Score with the CRPS (log space) of the mixture p·N(log μ, s²) + (1 − p)·N(log μ, S²), with S fixed
   (configurable, e.g. 1.0 in log space) and s from σ. Proper for the stated distribution. p captures epistemic
   uncertainty (am I right?), σ captures aleatoric spread, which the Phase 10 noise mode tests directly. Verify
   propriety by Monte Carlo before training. Split credit applies to p and σ tokens.

Pre-register 9.5's hypotheses in PREREGISTRATION.md together with Phase 10's. The "± 0" collapse (FINDINGS #6)
is reported as a finding in the write-up: with sequence-level credit, a proper scoring rule did not prevent
uncertainty collapse.

### Research framing

Question: **where should an LLM's numeric uncertainty come from, and can RL teach a single stated σ that tracks
the true uncertainty problem by problem?**

What makes this project different from prior work: our generator can produce problems whose **correct
uncertainty σ\* is known exactly, per problem**, and can dial that uncertainty independently of difficulty
(chain depth). Real datasets cannot do this.

Prior work to cite (re-verify every reference before publishing):
- RLCR, Damani et al., ICLR 2026 (arXiv 2507.16806): binary correctness + Brier reward for verbalized
  confidence. Closest in spirit; confidence is a probability of being correct, not a distribution over a
  continuous answer.
- DAR, Park et al. 2026 (arXiv 2605.20740): CRPS over K rollouts with leave-one-out credit for LLM regression.
  CRPS as an RL reward is NOT new; their uncertainty is rollout spread, not a stated σ.
- CARE-PPO, Dhaliwal et al. 2026: numeric prediction with the PPO critic as confidence.
- Rewarding Doubt (arXiv 2503.02623): log-score RL for verbalized confidence.
- FermiEval (arXiv 2510.26995): LLM intervals are overconfident; conformal adjustment.
- QuantSightBench (arXiv 2604.15859): evaluates LLM prediction intervals (eval only).
- Overconfidence is Key (arXiv 2405.02917): verbalized mean/std poorly calibrated.
- Pitfalls of verbalized confidence priors (arXiv 2609.32470): post-RL confidence barely tracks difficulty.
  This is the failure our conditional-calibration claim targets.
- Grouping loss / epistemic UQ (ICLR 2026): calibration metrics miss within-bin error heterogeneity.
- BeamPERL (arXiv 2603.04124): RLVR on beam mechanics induces templates, not equations. Be careful with any
  extrapolation claim.
- GRPO/Dr.GRPO/DAPO group-std identity (arXiv 2607.00152) and DAPO (arXiv 2503.14476): silent groups are known.
  FINDINGS #1 is motivation, not a contribution.
- Turtel et al. (arXiv 2505.17989): RL forecasting with Brier reward; betting payoff metric.

Claims, pre-registered in PREREGISTRATION.md before any results:
- **H1, propagation:** on problems with noise > 0, the slope of log(stated σ) on log(σ\*), the "vol beta", is
  higher for the RL model than for the SFT-only model, and closer to 1.
- **H2, conditional calibration:** a single post-hoc scale factor and split-conformal correction pass the pooled
  (unconditional) coverage test but fail coverage homogeneity across depth × noise cells. RL-stated σ has lower
  worst-cell coverage error than both.
- **H3, uncertainty sources:** exploratory ranking of the five sources by median |log(σ / σ\*)|. No directional
  claim.
- **H4, decision value:** RL-stated σ beats post-hoc on abstention (AURC) and on Kelly log-growth versus a
  fixed-width quote.
- **H5, stress:** under stress perturbations, stated σ rises in proportion to the rise in actual error.

### Phase 10: σ\* audit and measurement-noise mode (generator 2.0.0)

1. Write PREREGISTRATION.md: H1–H5 as above, primary metrics, thresholds, tests and multiple-comparison
   correction (Holm). Commit it BEFORE any Phase 10+ results exist.
2. **Rounding audit:** check whether ground truth uses unrounded world values while the problem text shows
   rounded givens. If so, compute the implied σ\* from significant figures (rounding error uniform within half a
   unit in the last place, propagated through the chain) for the existing eval sets. Report its size relative to
   the 1% accuracy band. Record as FINDINGS #8 (renumbered from #7 on 2026-10-14: the Phase 9.5 probe is #7).
3. **Noise mode:** noise level in {0, 1%, 2%, 5%, 10%} relative, per given.
   - True world values generate the target (truth = noise-free chain output).
   - Displayed given = true value × (1 + e), e ~ N(0, s²), shown with its stated uncertainty ("2.40 kg ± 2%"),
     with enough significant figures that rounding is negligible against s.
   - σ\*: Monte Carlo (N ≥ 2000) over the true-input posterior given the measurement
     (x_true ~ x_meas / (1 + e'), e' ~ N(0, s²)), pushed through the chain. Store σ\* in SI and in log space
     (crps_log uses log space).
   - Cross-check against first-order (delta-method) propagation for s ≤ 2%.
   - Noise level is an independent dial from depth. Mixed per-given noise levels are allowed.
4. **Generator self-consistency test (critical):** on 10,000 problems, intervals centred on the point estimate
   computed from the displayed givens, f(x_meas), with half-width σ\* (and 2σ\*) must cover the noise-free truth
   at nominal rates (68% / 95%; Kupiec test p > 0.05 in every depth × noise cell). If the generator's own σ\*
   fails this, nothing downstream is valid.
5. With noise > 0, "within 1%" is no longer a valid accuracy metric. Primary metrics for those problems:
   crps_log, σ / σ\*, coverage. Keep 1% accuracy for noise = 0.
6. Bump the generator to 2.0.0. Old tapes remain replayable under 1.1.0.

### Phase 11: rigor upgrades
- **Seeds:** every RL condition from here on runs with 3 seeds. Report mean and spread.
- **Held-out family:** build optics (thin lens, mirror, refraction, magnification) in the formula graph. It is
  NEVER sampled for SFT or RL training.
- **Eval sets:** 1,000 problems per split, stratified by depth × noise, frozen with checksums. Splits: ID,
  depth-5, held-out family, noise-dial grid.
- **Calibration split:** a separate set from training seeds only, used to fit every post-hoc method. Never use
  eval sets for fitting.

### Phase 12: train the stated-σ model
- Start from the SFT-only checkpoint (re-run SFT if needed so SFT data includes noisy problems in the output
  format; σ tokens stay masked from the SFT loss).
- GRPO with crps_log (floor fixed per D37), curriculum over depth AND noise level.
- 3 seeds. Control: SFT-only (same eval).
- One variable changes per comparison. Log every run's config hash in the tape.

### Phase 13: five-way uncertainty comparison
Sources, all evaluated on identical frozen eval sets:
1. **Stated σ** from the RL model (one sample).
2. **Rollout spread:** std of k = 8 samples' point answers (DAR-style, evaluation only), from both the SFT-only
   and RL models.
3. **Hidden-state probe:** a small MLP on frozen hidden states at the pre-answer token, trained on the
   calibration split to predict log|error| (and, separately, log σ\*). CPU training. Controls: random-init
   model, shuffled labels.
4. **LoRA ensemble:** disagreement across the 3 RL seeds.
5. **Post-hoc:** (a) one global scale factor on stated σ; (b) split-conformal intervals. Both fitted on the
   calibration split only.

Metrics (finance toolkit):
- **Vol beta:** slope of log(σ) on log(σ\*), noise > 0 problems.
- **Implied vs realized:** median σ / |error| over training and per cell (the model's "volatility risk premium",
  i.e. systematic overconfidence).
- **Calibration surface:** heatmap of median σ / σ\* over depth × noise. This is the headline figure.
- **Kupiec test** (pooled coverage at 68% and 95%).
- **Conditional coverage:** chi-square test of coverage homogeneity across depth × noise cells (the
  cross-sectional analogue of Christoffersen; state this adaptation explicitly in the write-up), plus worst-cell
  coverage error.
- **Basel-style traffic light** per source: green / yellow / red by exception count at 95%, as a one-glance
  summary.
- **Diebold–Mariano tests** on paired crps_log between sources, Holm-corrected.
- **Sharpness:** median relative interval width, reported next to calibration always.
- Spearman correlation between σ and |error|.

### Phase 14: decision value and stress tests
- **Abstention:** answer only when σ is below a threshold; risk–coverage curve and AURC per source.
- **Kelly log-growth:** log-wealth growth from betting the model's distribution against a naive fixed-width
  quote (median relative error from the calibration split). Report per source with bootstrap CIs.
- **Stress suite** (new generator flags, eval only): unit swaps, extreme magnitudes, extra distractors,
  paraphrased wording. Report change in σ vs change in error.
- **Optional, label as exploratory:** market-scoring-rule consensus. Treat the k = 8 samples as traders; build
  the leave-one-out mixture forecast; compare the consensus forecast's calibration against individual samples.
  Evaluation only; do not change the training reward.

### Phase 15: write-up (former Phase 10)
- README with: framing, prior work and exactly what is new, the calibration surface, the five-way table,
  backtests, decision-value results, stress results, honest limits (scale, synthetic domain, single model
  family), and every pre-registered hypothesis marked supported / not supported.
- FINDINGS.md summary in the same order as PREREGISTRATION.md.

### Working rules for these phases
- PREREGISTRATION.md is written before results and never edited afterwards; deviations go in DECISIONS.md with
  reasons.
- Every comparison has a control and changes one variable.
- Negative results are reported with the same prominence as positive ones.

## Working rules for Claude Code
- Write tests alongside each module; don't move phases with failing tests.
- No magic numbers in code. Thresholds and weights live in YAML configs.
- Prefer small, readable modules over frameworks.
- When a design choice is ambiguous, stop and ask rather than guess.

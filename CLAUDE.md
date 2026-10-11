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
10. **README** with results tables, calibration plots, the probe figure, and an honest limits section.

Do not start phase 8 or 9 until phase 7 has produced numbers.

## Working rules for Claude Code
- Write tests alongside each module; don't move phases with failing tests.
- No magic numbers in code. Thresholds and weights live in YAML configs.
- Prefer small, readable modules over frameworks.
- When a design choice is ambiguous, stop and ask rather than guess.

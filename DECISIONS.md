# Decisions log

Every ambiguous choice made during the unattended overnight session (2026-10-11).
Format: **decision** — alternatives considered — why. Entries marked **[REVIEW]** are the ones most
worth a second look.

## Setup

### D1. Python 3.11 and the uv cache live inside the project
- **Decision:** Python 3.11.17 was installed with `uv python install --install-dir .cache/uv-python`,
  and `[tool.uv] cache-dir = ".cache/uv"` is set in `pyproject.toml`. Hugging Face downloads go to
  `.cache/huggingface` (`HF_HOME`). `.cache/` is gitignored.
- **Alternatives:** use the system Python 3.12 (CLAUDE.md asks for 3.11), or the default uv/HF caches in
  `~/.cache` (outside the project folder).
- **Why:** the overnight rules say to work only inside `~/operation-physics`.
- **Side effect:** `uv python install` also created the symlink `~/.local/bin/python3.11 ->
  .cache/uv-python/...`. My attempt to delete it was blocked by the permission system, so it is still there.
  It is harmless, and you can remove it with `rm ~/.local/bin/python3.11`.

### D2. torch / transformers / accelerate installed now; unsloth and trl deferred to phase 4
- **Alternatives:** install unsloth now so its pinned torch is used from the start.
- **Why:** phases 1–3 need no training. Phase 3 asks for plain transformers in bf16. Unsloth pins torch
  versions and is the most likely install to break, so it is better done in phase 4 when it's actually needed.

## Phase 1 — Environment

### D3. A family is one coherent physical scenario, and chains stay inside it **[REVIEW]**
- **Decision:** each family is a single "world" with its own named quantities: a cart that accelerates,
  cruises and brakes (kinematics); a crate pulled across a rough floor, then released (dynamics); a block on a
  ramp, then a rough patch, then a spring (energy). The problem text always gives the full scenario, defining
  every symbol, then lists the givens. Chains of depth 1–5 are paths inside that one scenario.
- **Alternatives:** (a) one global graph across families, with "family" = the family of the last formula;
  (b) generic, scenario-free formula lists per family.
- **Why:** with (b), chained problems cannot be stated unambiguously. If `v` from one formula and `v` from
  another are different objects, the solver can't tell from the text that they are linked. A shared scenario
  makes the relations inferable. (a) would blur the per-(family, depth) curriculum cells. Depth 5 inside one
  family needs multi-stage scenarios, which is why each scenario has stages.

### D4. Ground truth is recomputed from the displayed (rounded) givens
- **Decision:** sample a consistent world from a few independent basis quantities (log-uniform within
  physical ranges). Display each given in a random unit at 3 significant figures. Then recompute the whole chain
  with pint from exactly the displayed values. The rounded value's SI magnitude is stored and tested.
- **Alternative:** use the unrounded world values as truth.
- **Why:** with the alternative, text and truth would disagree by up to ~1% at depth 5, the same size as the
  1% accuracy threshold.

### D5. Depth is the minimal derivation depth, enforced. Chains are enumerated exhaustively
- **Decision:** a chain of length d is valid only if a breadth-first closure over *all* formulas of the family
  gives the target minimal depth d and gives intermediate i minimal depth i. Every valid chain per
  (family, depth) is enumerated (takes milliseconds). Sampling is uniform over targets that have a valid chain,
  then uniform over that target's chains.
- **Alternatives:** a random backward walk (my first version); labelling depth by path length without
  checking for shortcuts.
- **Why:** without the shortcut check, "depth 4" problems can be solvable in 1 step. The random walk was
  heavily biased: kinematics depth 5 produced `v_avg` as the target almost every time. Standard shortcut
  formulas a student would use (v1² = 2gh, v2² = v1² − 2μgL, s = vt − ½at², f_k = m·a2) are included as formulas,
  so depth reflects a competent solver.
- **Limitation:** kinematics depth 5 has only 24 chains over 5 targets (`v_avg, u, t1, s1, a1`). Energy
  depth 5 has 36 chains over 7 targets. The depth-extrapolation split is therefore narrower for these families.

### D6. Givens must be algebraically independent (Jacobian rank), not just "not derivable"
- **Decision:** reject a chain unless the givens' log-Jacobian with respect to the world's basis has full row rank
  (evaluated at one fixed random reference world, with a relative singular-value tolerance of 1e-8).
- **Alternative:** only check that no given follows from the others through the one-formula-at-a-time closure.
  That was my first version, and it let an over-determined 6-given kinematics problem through.
- **Why:** over-determined problems become slightly inconsistent after rounding, so different solution paths give
  different "correct" answers.

### D7. Ill-conditioned chains are rejected (relative condition number ≤ 5)
- **Decision:** κ = max |d ln(output) / d ln(given)| over all chain outputs and givens, by finite differences.
  Reject if κ > `max_condition_number` (5.0). Then resample the world, and after `worlds_per_chain` failures
  resample the chain.
- **Alternative:** no filter.
- **Why:** subtraction steps (e.g. u = 2·s1/t1 − v with u ≪ v) amplify rounding, so a model that rounds an
  intermediate to 3 s.f. would miss the 1% band through no fault of its physics.
- **Known gap:** κ only covers chain outputs. Hidden quantities that the chain never computes (e.g. u in a
  chain that doesn't use it) are not checked, so in rare edge cases rounding could make an implied hidden
  quantity slightly unphysical.

### D8. Problem identity is (generator_version, family, depth, seed) **[REVIEW]**
- **Decision:** `generate_problem(family, depth, seed)`. The RNG is seeded from SHA-256 of all four values.
  `generator_version` = `<code version>+<8-char hash of the generator config>`, so editing `configs/env.yaml`
  changes the version string.
- **Alternative:** derive family and depth from the seed alone, matching the literal "(generator_version, seed)".
- **Why:** the curriculum chooses the (family, depth) cell, then a seed. The tape stores all four fields, so
  replay is still exact.

### D9. Held-out enforcement lives in the generator. Eval seeds are disjoint from training seeds
- **Decision:** `generate_problem(..., split="train")` (the default) raises `HeldOutError` for held-out families
  (`optics`) and depths (5). Eval seeds start at 1,000,000,000 (`splits.eval_seed_offset`), so no ID eval problem
  can coincide with a training problem.
- **Alternative:** enforce only in the sampler.
- **Why:** this makes it impossible for any training code path to sample a held-out cell by accident.

### D10. Distractors are dedicated irrelevant quantities
- **Decision:** each family defines 3 quantities that appear in no formula (e.g. the cart's mass in
  kinematics, the rope length in dynamics, the air pressure). 0–2 are added per problem.
- **Alternative:** use unneeded world quantities as distractors.
- **Why:** world quantities can create shortcuts or over-determine the problem (see D6). Dedicated ones are
  irrelevant by construction.

### D11. Plausibility bounds on derived quantities
- **Decision:** worlds are resampled if a derived quantity leaves a plausibility band (cart v ≤ 40 m/s,
  crate v ≤ 25 m/s, slide distances ≤ 300 m, spring compression ≤ 1 m, rough patch ≤ 50 m).
- **Why:** the first draft produced crates moving at 108 m/s and 18 m spring compressions.

### D12. Constants and tolerances
- **Decision:** g = 9.81 m/s² is a code constant (`physcal/env/constants.py`) and is stated in every problem
  of a family that uses it. Numerical-analysis tolerances (rank tolerance, reference-world seed) are named
  constants in `graph.py`. Every experiment-level threshold is in `configs/*.yaml`.
- **Why:** g defines the physics and is not tunable. Putting it in YAML would invite silent changes to ground
  truth. (Changing it in code would require bumping the generator version.)

## Phase 2 — Verifier

### D13. Failures score the rule's floor, not a literal 0 **[FINAL 2026-10-12: accepted]**
- **Decision:** a parse failure scores `score_final = floor` and `score_steps = 0`, so the total is
  w_final·floor. A unit mismatch scores `score_final = floor`, and its steps are still credited. The floor is
  −5 for CRPS (the clip), −10 for log_score, −5 for abs_rel_error and 0 for binary. For the binary reward this
  is exactly the spec's "reward 0".
- **Alternative:** the literal spec, "reward 0" / "final score 0".
- **Why:** under CRPS every valid answer scores below 0 (−CRPS < 0). A literal 0 for malformed output would make
  garbage the *best* possible output: an immediate reward hack. The spec's 0 was clearly written with the
  binary reward in mind. A second alternative was to remap CRPS affinely to [0, 1] (1 − CRPS/5), so failure = 0
  holds literally. I rejected it because it multiplies the relative weight of step credit by 5 compared with
  the spec's −CRPS + 0.2·steps.

### D14. "Steps can never outweigh the final answer" is checked at load time
- **Decision:** `validate_reward_config` requires w_steps < w_final·(ceiling − floor) of the active rule. In
  words: a perfect final answer with no steps must beat a failed final answer with every step correct. The
  defaults pass with a large margin (0.2 vs 5).

### D15. The spec's relative CRPS is only approximately proper; `crps_log` was added as an option **[FINAL 2026-10-12: crps_log is the default]**
- **Finding:** dividing by the *truth* |y| weights each outcome by 1/|y|. So the score is proper for the
  re-weighted belief p(y)/|y|, not for p(y). By Monte Carlo, the optimal μ is biased by −0.24% at 5% relative
  spread, −1.0% at 10%, −4.4% at 20% and −11% at 30%. The optimal σ is inflated by up to 7%. The same issue
  applies to `log_score`. For a small model, whose relative errors are often 10–100%, the incentive to shade
  μ downwards is real, and it confounds research question 2 (is σ calibrated?).
- **Decision:** keep the spec's `crps` as the default, as written. Add `crps_log` as an extra config option:
  score ln(y) against N(ln μ, (σ/μ)²). It is strictly proper for log-normal beliefs and still scale-free. Its
  failure floor and clip are the same as `crps`. Tests pin both behaviours.
- **Recommendation:** consider making `crps_log` the main rule before phase 5, or at least run it as one of
  the phase-7 ablations.

### D16. Parser leniency
- **Decision:** lenient about notation, strict about structure. Accepted: ±, +/-, +-, \pm; unicode minus and
  superscripts; ×10^k with x, ×, *, \times, and unicode exponents; LaTeX wrappers; `\boxed{}`; markdown bold;
  `[unit]`; a `symbol =` prefix; `(μ ± σ) × 10^k unit`; `μ unit ± σ unit` with different units; σ as a percent;
  "dimensionless"; thousands separators. Rejected (parse_fail): no `<answer>` block, more than one `<answer>`
  (prevents hedging), an unclosed tag (truncation), no leading number, two ± signs, non-finite numbers, and a
  missing ± under rules that need σ. A negative σ is read as its absolute value.
- **Why:** the untrained baseline will use many notations. Rejecting notation variants would measure formatting
  rather than physics, and would make the zero-variance problem worse early in RL.

### D17. `<steps>` is optional; free reasoning is allowed before it
- **Decision:** the system prompt allows thinking before `<steps>`. A missing `<steps>` block is not a parse
  failure; it just earns no step credit (`steps_block_found=False` is logged). Step symbols are matched after
  canonicalization (`t_1` = `t1`, `v_{avg}` = `v_avg`), case-sensitively (`F` ≠ `f`). The last line wins. Each
  step needs the right dimensions.
- **Alternative:** require the exact two-block output with no reasoning.
- **Why:** a 1.5B model benefits from chain of thought, and the 512-token budget allows a short one. The spec's
  format block shows the *ending* of the completion.

### D18. Depth-1 problems: step score = `steps.empty_score` = 0
- **Alternatives:** 1.0 (vacuously all correct), or renormalizing the reward without the steps term.
- **Why:** GRPO normalizes within a group of the same problem, so a per-problem constant has no effect on
  training. 0 keeps the mean reward of a depth-1 problem from looking inflated. It is configurable.

### D19. Copied-given check
- **Decision:** `copied_given` is true if the answer's number equals a given's (or distractor's) *displayed*
  number within 0.1%, or its SI value equals a same-dimension given's SI value within 0.1%. It is logged only;
  it never touches the reward.

## Phase 3 — Baseline eval

### D20. Baseline uses greedy decoding, one completion per problem **[REVIEW]**
- **Decision:** `do_sample: false`, `max_new_tokens: 512`, batch 16, bf16, plain transformers.
- **Alternatives:** sample at the future training temperature, possibly k samples per problem (which would
  also give pass@k and sample-based calibration).
- **Why:** deterministic and reproducible, and the standard way to report a single-number baseline. Later
  checkpoints must be evaluated the same way (greedy) for the comparison to be fair. Note that GRPO rollouts
  will be sampled, so training-time reward will not equal this eval's reward.

### D21. Held-out-family and simulator splits are configured but skipped in the baseline
- **Decision:** `configs/eval.yaml` defines all four spec splits. `heldout_family` (optics) and `simulators`
  are skipped with a logged reason, because neither exists before phase 6.
- **Alternative:** implement optics/simulators tonight (phase 6 work, outside tonight's scope).
- **Why:** the untrained model doesn't change, so its numbers on those splits can be added in phase 6 by
  rerunning the same command. Nothing is lost.

### D22. The rollout tape was built in phase 3, as the eval's logging mechanism
- **Decision:** `physcal/tape/tape.py` writes the spec's fields, plus `split`, `n_tokens` and `truncated`, one
  JSON line per completion. A `.meta.json` sidecar stores the env, reward and eval configs, the system prompt
  and the git commit. `replay()` regenerates each problem from (version, family, depth, seed), checks it is
  identical, re-scores it, and requires exact float equality of the reward. Eval lines use `step = 0` and
  `lora_hash = null`.
- **Why:** "every rollout is logged" is a hard constraint, and eval completions are rollouts.

### D23. Metric denominators
- **Decision:** accuracy counts parse failures and unit mismatches as incorrect (denominator = all problems).
  Median relative error is over answers that parsed with valid units. 1σ/2σ coverage, PIT and reliability are
  over answers that also stated a σ. The `n` behind each is reported (`n_answered`, `n_with_sigma`).
  `accuracy_among_answered` is reported too. Mean score per rule counts failures at that rule's floor.
- **Why:** headline accuracy must not improve just because the model stops answering hard problems. Coverage
  is only defined where a σ exists.

### D24. Peak VRAM is recorded two ways
- **Decision:** torch `max_memory_allocated` / `max_memory_reserved` (this process only), plus device-wide
  `nvidia-smi` polling every 2 s. The device-wide number includes ~1.9 GB used by the Windows display/WSL host,
  which this process does not control.

### D25. Eval split construction
- **Decision:** 200 problems per split, assigned round-robin to (family, depth) cells (ID: 3 families × depths
  1–4 = 12 cells, ~17 each; depth extrapolation: 3 cells, ~67 each). Seeds: `eval_seed_offset + split_index ×
  1e6 + i`, so no two splits share a seed and none collides with training seeds (< 1e9).

### D26. Every decoding parameter is explicit; the official baseline uses repetition_penalty = 1.0 **[REVIEW]**
- **Finding:** my first baseline run passed only `do_sample=False`, so `generate()` silently took
  `repetition_penalty: 1.1` from Qwen2.5's `generation_config.json`. HF applies that penalty to every token in
  the context, *including the prompt*, which pushes the model away from reusing the problem's numbers and units.
- **Decision:** `configs/eval.yaml` now sets `repetition_penalty`, `temperature`, `top_p`, `top_k`,
  `num_samples` and `seed` explicitly. The official baseline is pure greedy with `repetition_penalty: 1.0`. The
  first run is kept, clearly labelled, in `results/baseline_qwen_genconfig_rep1.1/` with a NOTE.md.
- **Alternative:** treat the vendor default (1.1) as the baseline.
- **Why:** TRL's GRPO rollouts use repetition_penalty 1.0 by default, so trained checkpoints should be evaluated
  with the same decoding as the baseline. Hidden defaults make runs irreproducible across model versions. The
  two runs are both reported so the effect is visible.

### D27. Sampled group-variance probe (inference only), to size up phase 4
- **Decision:** `configs/eval_group_probe.yaml` samples k = 8 completions at T = 1.0 (top_p 1, top_k off,
  repetition penalty 1.0) for 48 ID problems (4 per family×depth cell). Per GRPO-style group it reports pass@8,
  the mean pass rate, and the fraction of zero-variance groups under the binary reward and under the main CRPS
  reward. Results go to `results/baseline_group_probe/`.
- **Why:** phase 4 is "first GRPO run with the binary reward". With 0–1% greedy accuracy, most binary groups
  will carry no gradient. This measures how bad that is before anyone spends GPU hours on it. No training.

### D28. Prompt variants are diagnostics; the official baseline stays on prompt v1 **[REVIEW]**
- **Decision:** `physcal/prompt.py` holds three system prompts, selected by `prompt:` in the eval config:
  `v1` (official baseline, also the default for training), `v2_reason_first` (asks explicitly for worked
  reasoning first) and `v3_worked_example` (v1's rules plus one fully worked example from a *training*-seed
  problem, embedded verbatim). v2/v3 results live in `results/baseline_prompt_v2|v3/` and are reported as
  prompt-sensitivity diagnostics, not as the baseline.
- **Why:** the baseline was at 0%, so it was worth knowing whether that is a prompting artefact before
  phase 4. Which prompt training uses is a research-design choice, and it's yours.
- **Caveat:** v3's example states "about 1%" uncertainty, which is itself a cue for σ. If v3 is adopted, that
  cue confounds the calibration question (RQ2); a version without a σ heuristic would be cleaner.
- **Related finding (not changed):** the generator's depth-1 sentence "No intermediate values need to be
  reported, so <steps> may be empty." makes the model skip its calculation (96% answer-only at depth 1, 0% at
  depths 2–5). Replacing it (diagnostic, `scripts/diag_depth1_wording.py`) cut answer-only to 34%, but
  accuracy stayed 0%. Changing it would bump the generator version, so I left it for you to decide.

## Phase 3.5 — Fixing the starting point (2026-10-12)

### D29. `crps_log` is the default scoring rule; the spec's relative CRPS is an ablation flag (final)
- **Decision (yours):** `configs/reward.yaml` → `scoring.rule: crps_log`. `crps` (relative, as in the spec),
  `binary`, `log_score` and `abs_rel_error` remain selectable. Failures keep the rule's floor: −5 for
  `crps_log` and `crps`, 0 for `binary` (D13, accepted).
- **Consequence:** tapes logged before this change carry `reward_config.scoring.rule = crps` in their
  meta.json, and `replay()` re-scores them with that logged config, so they still reproduce exactly. New runs
  report `mean_reward` under `crps_log`. Every breakdown still contains the scores under every rule.

### D30. Generator 1.1.0: depth-1 wording asks for the calculation; the wording is keyed by version (final)
- **Decision (yours):** depth-1 problems now end with "Show your calculation first; no intermediate values
  need to be listed in <steps>." (the wording tested in `scripts/diag_depth1_wording.py`) instead of "No
  intermediate values need to be reported, so <steps> may be empty." `configs/env.yaml` version → `1.1.0`, giving
  generator version `1.1.0+c735a0d4`.
- **How old tapes stay replayable:** `render.NO_STEPS_TEMPLATES` maps version → wording. Every tape's
  meta.json carries its env config, so `replay()` regenerates with `1.0.0` and gets the old wording and the old
  version hash (`1.0.0+afcc2019`). All 6 phase-3 tapes (2,368 rollouts) still replay exactly, and a test pins
  this. An unknown version raises instead of silently using some wording.
- **Side effect (by design, D8):** the version string seeds each problem's RNG, so the bump re-draws *every*
  problem, not just the depth-1 text. 1.1.0 eval sets have the same distribution as 1.0.0 but different
  numbers. 1.0.0 and 1.1.0 numbers are therefore compared on different (identically distributed) problem
  sets.
- **Frozen eval sets:** `uv run python -m physcal.eval.export_sets` writes
  `data/eval_sets/<generator_version>/{id,depth_extrapolation}.jsonl` + `manifest.json` (SHA-256 per split). A
  test regenerates them and requires byte equality, so any change in generator output without a version bump
  now fails CI.

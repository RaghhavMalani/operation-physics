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

### D31. Model-probe design (phase 3.5) **[REVIEW]**
- **Candidates (pinned):** `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`,
  `Qwen/Qwen2.5-Math-1.5B-Instruct` @ `aafeb0fc6f22cbf0eaeed126eff8be45b0360a35`, and `Qwen/Qwen3.5-2B` @
  `15852e8c16360a2fea060d615a32b45270f8a8fc`. These are verified on the Hub API (ungated, Apache-2.0). The
  revision is passed to `from_pretrained`, checked after loading, and recorded in every metrics.json.
- **Why Qwen3.5-2B is the "strongest current ≤ 2B instruct model":** released 2026-02-28; non-thinking
  MMLU-Pro 55.3 vs Qwen3-1.7B 40.2 on its card. Using Qwen3-1.7B as the common reference across cards, it
  is well ahead of LiquidAI LFM2.5-1.2B-Instruct (44.4 vs Qwen3-1.7B 42.9 on LiquidAI's card) and
  granite-4.0-1b (33.5 there). Its *text* decoder has 1.88B parameters. The checkpoint also contains a 0.39B
  vision tower and an MTP head, which text-only training doesn't use, and the footprint estimate excludes them.
  Considered and excluded: Gemma-4-E2B (about 5B total parameters), Llama-3.2-1B-Instruct (gated, and weaker),
  DeepSeek-R1-Distill-Qwen-1.5B (a reasoning model whose long traces can't fit 1024 tokens), LFM2.5-1.2B (license
  "other"). I didn't download these.
- **Qwen3.5-2B runs in non-thinking mode** (`enable_thinking: false`, the card's default for 2B). Thinking
  traces would blow the 1024-token budget, and the card warns that 2B tends to loop in thinking mode.
- **Same prompt for all:** `v4_worked_example` = v3 without the sentence "The inputs have 3 significant figures,
  so I estimate a relative uncertainty of about 1%". The example's answer line still shows `± 0.0003 km`
  (≈1%): any concrete example of the format implies *some* σ.
- **Same decoding for all:** our explicit parameters (greedy; or T = 1.0, top_p 1, top_k off, repetition
  penalty 1.0), not each model card's recommended sampling. This keeps the comparison controlled and matches
  what GRPO will use.
- **Out-of-memory handling:** batches back off (prompts split in half, then samples) instead of falling back to a
  smaller model. Backoffs are counted in metrics.json.
- **GRPO footprint is estimated, not measured** (`physcal/eval/footprint.py`, `configs/footprint.yaml`): exact
  parameter counts from safetensors headers; 4-bit decoder linears + bf16 embeddings; LoRA r = 16 on every linear
  with fp32 Adam; rollout KV cache for 8 × (prompt + 1024) tokens; training with micro-batch 2, offloaded
  checkpoints and chunked log-probs (Unsloth-style); +10% fragmentation and 0.45 GiB CUDA context. A
  pessimistic plain-TRL variant is reported too. Sanity check: the same activation model applied to bf16
  inference predicts 4.02 GiB for the phase-3 baseline, which measured 3.76 GiB.
- **Budget:** your "about 6.5 GB" is taken as 6.5 decimal GB (6.05 GiB) for our process. That matches what is
  actually free on this card (8 GiB minus ~1.9 GiB used by the Windows display).

### D32. Qwen2.5-Math-1.5B fails on format, not physics; scored as-is (verifier unchanged)
- **Finding:** it answers in its native `\boxed{}` style and never writes `<answer>`, despite the system prompt
  and worked example. Verifier result: 100% parse failure (greedy) and 99.5% (sampled). As an
  out-of-band diagnostic (`scripts/diag_boxed_answers.py`, not used for selection), reading its last
  `\boxed{}` number in the requested unit gives **35% ID / 8.5% depth-5 within 1%** (greedy), the best
  physics of the three candidates.
- **Decision:** the selection rule uses the verifier as-is (your instruction: don't modify the verifier),
  so it is ranked on its real 97.9% binary zero-variance. The diagnostic goes into the recommendation:
  a format-only warm-up could unlock this model.

### D33. Qwen3.5-2B group probe re-run at batch size 8
- **What happened:** at batch 16 (2 prompts × 8 samples) the sampled run took ~660 s per batch, against
  ~85 s for greedy batches of 16. That points to out-of-memory retries late in decoding: the device sat at
  7.7/8 GB, and greedy at 16 already peaked at 6.9 GiB allocated. At that pace it would have finished
  after your 15:45 GPU cutoff.
- **Decision:** I killed it after 64/384 completions, discarded the partial output, and re-ran the group
  part only with `group_batch_size: 8` (1 prompt × 8 samples). It ran at ~40 s per batch. Batch size
  doesn't change what the probe measures (same 48 problems, k = 8, T = 1.0, seed 0), but the random stream
  differs from a batch-16 run. The OOM backoff now prints a line whenever it triggers, so this is visible
  next time. Greedy results (batch 16, no backoffs) are unchanged.

## Phase 4 — First GRPO run (2026-10-12, unattended per your instructions)

### D34. How the first GRPO run was set up **[REVIEW]**
- **Model and reward:** Qwen3.5-2B (the selection rule's pick). Reward `crps_log` + 0.2·step credit, not
  binary: its probe binary zero-variance was 72.9% > 50% (your rule).
- **Stack:** unsloth 2026.10.3 + trl 1.13.0 + peft 0.21.2 + bitsandbytes 0.50.2. Unsloth requires
  `transformers <= 5.17.0`, so the project now pins `transformers==5.17.0` (the probes ran on 5.19.0).
  Unsloth supported Qwen3.5 on the first attempt (it bundles the Gated-DeltaNet Triton kernels); there was no
  fallback model and no install trouble.
- **GRPO shape:** 1 prompt × 8 completions per optimizer step (micro-batch 2 × 4 accumulation), 1024
  completion tokens, T = 1.0, top_p 1, top_k off, LoRA r = 16 on q/k/v/o/gate/up/down (the Gated-DeltaNet
  projections in the linear-attention layers carry no adapter), lr 5e-6 constant after 3 warmup steps,
  AdamW-8bit, grad-norm clip 0.1, beta 0 (no KL / no reference model). TRL 1.13 defaults kept and recorded:
  `loss_type = dapo`, `scale_rewards = group`. Training problems: train split, seeds 0.., round-robin over
  the 12 training cells (the curriculum sampler is phase 5). The model is loaded text-only (vision tower
  skipped), bf16 compute (Unsloth warns Qwen3.5 GDN layers NaN in fp16).
- **Caps:** your 2 h / 300 steps, plus a hard stop at 14:55 IST so the eval could finish before the 15:44
  GPU cutoff. The hard stop bound first: **49 steps** in 57 min. Because of that, no step-50 checkpoint
  exists; the stop saved `checkpoint-49` and `final_adapter` (both gitignored, kept locally in
  `results/train/phase4-grpo-qwen3.5-2b-crps_log/`).
- **Logging:** every rollout goes to `rollouts.jsonl` (392 = 49 × 8, replays exactly). Per step:
  reward, zero-variance fraction (crps_log and σ-free binary), median/mean predicted relative σ, 1σ/2σ
  coverage, accuracy, parse/unit failures, truncation and completion length go to `train_metrics.jsonl`.
  The curve is `training_curve.png`.
- **Smoke-test fixes:** the step counter read a stale `TrainerState` (TRL replaces it when `train()` starts),
  and `warmup_ratio` is not accepted by this stack (→ `warmup_steps`). The smoke run's own logs keep step = 1
  in every row.
- **Eval of the trained adapter:** same eval sets as the selected model's "before" probe
  (`eval_probe_greedy`: 200 ID + 200 depth-5, prompt v4, greedy, 1024 tokens), run with
  `uv run --with transformers==5.19.0` so the library version matches the "before" numbers. The base is
  loaded text-only (`AutoModelForCausalLM`, because the adapter's keys are `model.layers.*`; CPU-verified that
  all 96 LoRA modules attach), and LoRA is merged into the bf16 base. The "before" probe loaded the VL
  class; its text path is the same computation. Guarded by `timeout` until 15:44 IST.

## Phase 3.6 — SFT warm-up A/B and overnight GRPO (2026-10-13, unattended)

### D35. SFT data and loss masking (your approval: plan as drafted, σ masked)
- **Data:** `physcal/train/sft_data.py` writes one short solution per problem, in the exact output format:
  an SI-values line, a "Not needed:" line naming distractors, one line per chain step (`out = formula =
  substituted numbers = value unit`, 4 s.f.), an optional unit-conversion line, then `<steps>` and
  `<answer>`. Problems come from `generate_problem(..., split="train")` with seeds 1e7.. (disjoint from GRPO
  seeds 0.. and eval seeds ≥ 1e9), round-robin over the 12 training cells. `build_examples` refuses any
  seed range that reaches the eval offset. A test runs 240 targets through our own verifier: all are correct,
  with full step credit.
- **Loss:** prompt tokens are masked, and so are **the tokens of the σ number** (any token overlapping its
  character span). `±`, the unit, `</answer>` and the end-of-turn token are trained, so the model learns the
  format and when to stop. The σ written in the target text is 2% of the value; because it is masked it only
  serves as context, and the model is never trained to produce it.
- **Prompt:** v4 (the same system prompt used by the probes and GRPO), so the warm-up matches inference.
- **Recipe:** LoRA r = 16 / α = 16 on q/k/v/o/gate/up/down, 4-bit base, lr 1e-4 constant after 10 warmup
  steps, micro-batch 2 × 4 accumulation, AdamW-8bit, 30 min of wall clock per model (it stops before one
  epoch of the 3,000 pre-built examples).
- **After SFT:** the adapter is merged into the bf16 base (`physcal/train/merge.py`, peft, CPU) to give an
  "SFT-only" model directory. GRPO then trains a *fresh* LoRA on top of that merged model, so the RL adapter
  isolates what RL added.

### D36. SFT A/B protocol and the overnight GRPO design **[REVIEW]**
- **A/B fairness:** equal wall clock (30 min each, your brief), not equal examples. On the 4060, Qwen3.5-2B's
  hybrid layers train at ~6.7 s per 8 examples, so it sees fewer examples than Qwen2.5-Math in the same time.
  Both use identical data (the same 3,000 training-seed problems, shuffled with the same seed), the same LoRA
  and the same lr. The selection is your rule: lower binary (σ-free) zero-variance in the unchanged k = 8
  group probe (`configs/model_probe_sft.yaml`), with ties broken by the higher pass rate
  (`scripts/ab_select.py`).
- **Token budget rule applied off the 1024-token probe:** "truncated at 512" = no end-of-turn within the first
  512 generated tokens. Sampling is identical up to that point whatever the cap, so no extra run is needed.
- **Curriculum (EMA, your brief; the HMM sampler is not built):** `physcal/env/sampler.py` follows spec §5:
  per-cell EMA of the σ-free binary pass rate (α = 0.1, prior 0.5), weight p(1−p) + 0.02, and only
  unlocked cells are sampled. It starts at depths 1–2 for every family; depth d+1 of a family unlocks once
  depth d's EMA pass rate is ≥ 0.5 after ≥ 8 groups. The 0.5 threshold is my choice, the peak of p(1−p):
  the cell has become informative-but-easy. A pool of 400 training problems per cell (seeds 0.., cell-major)
  is cycled in order.
- **TRL integration:** `CurriculumGRPOTrainer._get_train_sampler` returns a sampler that reproduces TRL's
  `RepeatSampler` pattern exactly (chunk of 4 prompts, each × 8 generations, chunk × steps_per_generation),
  but asks the EMA sampler for the 4 cells at the moment each chunk is requested. The reward function feeds
  each group's results back, so step k's outcomes inform step k+1's cells. Every step's cells and the full
  EMA state are logged to `curriculum.jsonl`.
- **Step shape:** 4 prompts × 8 completions = 32 per optimizer step (micro-batch 4 × 8 accumulation), lr 5e-6,
  DAPO loss (TRL default), no KL, LoRA r = 16 on top of the *merged* SFT model. Hard stop 06:30 IST
  (next-occurrence parsing), checkpoints every 50 steps keeping 3.
- **Final comparison (`physcal/train/compare.py`):** raw base, SFT-only and SFT + RL are all evaluated with
  identical settings: the frozen eval sets, prompt v4, greedy, max_new_tokens = the training budget, text-only
  loading, transformers 5.17 (the current env, for all three). Paired McNemar / Wilcoxon for rl-vs-sft (what RL
  added), rl-vs-base and sft-vs-base. The raw-base numbers here are not the phase-3.5 probe numbers, which
  used 1024 tokens.

### D37. Observation: the 0.1% σ floor makes the crps_log reward flat in σ for this policy **[REVIEW]** (not changed)
- **What I saw overnight:** the SFT'd policy states σ ≈ 0.06–0.08% of the value (median), and that did not move
  across 250+ GRPO steps (1σ coverage stayed at 15–23%). The scoring rule floors the relative σ at 0.1%
  (`scoring.sigma_floor: 1e-3`), so *every* σ below 0.1% scores identically. The reward is locally flat in
  σ, and only sampled jumps of σ above the floor carry any σ signal.
- **Why it matters:** CRPS (unlike the log score) stays finite as σ → 0 (it tends to |error|), so the floor isn't
  needed for numerical stability under `crps_log` / `crps`. The floor is only there to stop the log score
  blowing up. Under crps_log it removes the gradient exactly where an over-confident model needs it.
- **Not changed tonight** (your rule: no verifier changes). **Recommendation:** for the CRPS rules, lower the
  floor to ~1e-6 or apply it only to `log_score`. Bump the reward-config version and re-score old tapes with
  `rescore` (cheap) before the next RL run.

## Phase 3.7 — σ floor removed, matched-step GRPO rerun (2026-10-13/14, unattended)

### D38. Verifier 1.1.0: crps_log gets its own σ floor (1e-6); everything else unchanged (D37 approved)
- **Decision:** `configs/reward.yaml` gains `version: "1.1.0"` and `scoring.crps_log_sigma_floor: 1.0e-6`.
  `scoring.sigma_floor` stays at 1e-3 and still applies to `crps` (the spec's own `max(σ/|y|, 1e-3)`) and
  `log_score` (which diverges as σ → 0). `crps_log` applies its floor in both places it used the shared one (the
  relative σ and σ/μ). Every breakdown now logs `verifier_version`.
- **Old tapes:** a reward config without `version` is verifier 1.0.0, and `crps_log` then falls back to the shared
  floor, so every logged tape still replays exactly under its own meta.json config. `validate_reward_config`
  rejects a config that has a version > 1.0.0 without `crps_log_sigma_floor` (or the reverse), so the fallback
  can't be hit by accident.
- **Unchanged on purpose:** the logged `z` and `rel_sigma_floored` fields still use the shared 1e-3 floor (they
  are descriptive, not part of any reward), and so do `crps`, `log_score`, the clips and the failure floors.
- **Alternatives:** lower the shared `sigma_floor` (would also change the spec's `crps` and blow up `log_score`);
  key the floor on the version string (hides a threshold in code).
- **Tests added (13):** the approval condition (for |error| from 0.1% to 5%, rewards strictly increase as σ rises
  from 0.01% to 0.1% of the value, through the full `reward()` pipeline, and were exactly flat under 1.0.0);
  Monte Carlo propriety at a belief spread of 0.03% (below the old floor: best σ = the true spread, and the
  legacy floor gives a flat curve); `crps`/`log_score` unchanged; σ = 0 scores |ln(μ/y)|; version validation;
  exact replay of last night's SFT-only and SFT+RL eval tapes under their logged 1.0.0 config. All existing
  propriety / MC tests pass unchanged. 251 tests pass.

### D39. Memory fix: micro-batch 2 × 16 accumulation and a 7.5 GiB allocator cap; nothing else changes
- **Cause of last night's spill:** on WSL, Unsloth strips `expandable_segments` from the allocator config (it is
  unsupported there), so fragmentation let torch's *reserved* memory reach 8.96 GiB while only 4.58 GiB was ever
  allocated.
- **Decision:** `configs/grpo_overnight2.yaml` = last night's config with `per_device_train_batch_size: 2`,
  `gradient_accumulation_steps: 16` (was 4 × 8), plus `guards.max_reserved_gib: 7.5`, which calls
  `torch.cuda.set_per_process_memory_fraction` so the caching allocator frees cached blocks and retries before
  it grows past 7.5 GiB. Unsloth's gradient checkpointing (`"unsloth"`) was already on. Per-step peak
  reserved/allocated memory, allocator retries and OOMs go to `memory.jsonl`; the summary reports them too.
- **Why this keeps the optimization identical:** the generation batch is still 32 completions (= 2 × 16 =
  4 prompts × 8), so the same prompts, group-normalized advantages and samples per step. TRL/Unsloth's DAPO loss
  divides the summed token losses by the token count of the *whole* generation batch, so splitting it into 16
  micro-batches instead of 8 gives the same gradient up to float summation order. lr, warmup, optimizer,
  clipping, β, LoRA, seeds, curriculum and token budget are unchanged; a test checks that the two YAML files
  differ only in these keys and the run id.
- **Alternatives:** the cap alone at micro-batch 4 (more allocator retries; riskier); `torch.cuda.empty_cache()`
  every step (doesn't bound the within-step peak); micro-batch 1 (slower, kept as the fallback if the smoke test
  failed).
- **Risk accepted:** with a hard cap, a step that truly needs > 7.5 GiB live would raise OOM instead of
  spilling. The live peak last night was 4.58 GiB, so the margin is large; the smoke test checks it.

### D40. Post-hoc calibration control: one σ scale fitted on training-seed SFT-only completions
- **Calibration set:** 400 problems from the generator's *train* split, seeds 20,000,000–20,000,399, round-robin
  over the 12 training cells (depths 1–4). These are disjoint from the GRPO pool (0–4,799), the SFT data
  (10,000,000–10,002,999) and every eval seed (≥ 1e9). `build_split` gained a `train_seed_offset` option that uses
  the train split (so held-out cells are refused) and refuses to reach the eval range; `posthoc.fit` re-checks
  that no calibration seed is an eval seed.
- **Generation:** the SFT-only merged model, exactly the eval's settings (prompt v4, greedy, 512 tokens,
  text-only, batch 16), so the fitted scale is applied to completions of the same kind. Scored by verifier 1.1.0.
- **Fit (primary):** the single c that maximizes mean crps_log of σ' = c·σ on the calibration set: the same
  objective RL optimizes, so it is the like-for-like "what could a free post-hoc fix do" baseline. Mean crps_log
  is convex in c (CRPS is the perspective of a convex function in σ), so grid + golden-section search finds the
  optimum. Failures keep the floor (−5) and are unaffected by c.
- **Sensitivity variant:** the c that makes 1σ coverage exactly 68.27% on the calibration set (the 68.27th
  percentile of |μ − y|/σ). This is the strongest possible post-hoc baseline *for the primary metric*
  (1σ coverage), so both are reported, with crps-fit as the headline.
- **Not done:** a per-depth or per-family scale (more parameters than "a single multiplicative scale"), and
  depth-5 problems in the fit (training seeds can't produce depth 5; the scale is transferred to depth 5
  unchanged, which is itself a test).
- **Paired comparisons against the control** use a derived JSONL (`sft_eval_scaled_<variant>.jsonl`) with σ, the
  coverage flags, crps_log and reward recomputed under σ' = c·σ. It is not a replayable tape; the replayable source
  is the re-scored SFT-only eval tape.

### D41. Comparison protocol for tonight (matched step 287)
- **Primary:** the new run's permanent step-287 adapter, evaluated first, against (a) SFT-only, (b) last night's
  RL (its final adapter is step 287), (c) the post-hoc control. **Secondary ("longer training"):** the final
  adapter vs step 287. All tapes are on verifier 1.1.0: last night's SFT-only / RL eval tapes were re-scored
  (stage 1), and the new evals are scored live.
- **Tests:** the same paired tests as before (McNemar on strict ≤ 1% correctness, Wilcoxon on reward), plus
  McNemar on 1σ and 2σ coverage (primary metric) and Wilcoxon on the crps_log score alone. For paired coverage a
  failed answer counts as not covered (denominator = all 200 problems per split); the headline coverage numbers
  are over answers with a σ, as in every earlier table. With 0% parse failures the two coincide.
- **Smoke test result (5 steps, `results/train/smoke-overnight2-mb2/`):** 44 s/step, torch peak reserved
  **6.46 GiB** (allocated 4.45 GiB), 0 allocator retries, 0 OOMs, 32 completions per step, and the curriculum
  drew exactly the same cells as last night's first five steps. The overnight run started at 23:57 IST.

### D42. Tonight's training tape is committed gzipped
- **Issue:** the training tape passed 50 MB at step 400 (GitHub warns at 50 MB and rejects files over 100 MB), and
  a full night will roughly double it. Progress commits up to step 400 included the raw tape (several versions
  are now in history; I did not rewrite history).
- **Decision:** from step 400 on, progress commits leave `rollouts.jsonl` out. The final commit untracks it, adds
  it to `.gitignore`, and commits `rollouts.jsonl.gz` (gzip -9) next to `rollouts.meta.json`. To replay:
  `gunzip -k rollouts.jsonl.gz`, then `replay()` as usual (the raw file stays on disk locally).
- **Alternatives:** Git LFS (not set up on this repo, and it's an account-level choice); splitting the tape
  (breaks `replay(path)`).

## Phase 9.5 / 10 setup (2026-10-14, unattended)

### D43. Merging PHASES_10-15.md into CLAUDE.md; FINDINGS numbering
- **Decision:** CLAUDE.md's phase list now runs 1–9, 9.5, 10–15; the old Phase 10 (README) is Phase 15, and the full
  text of phases 9.5–15 (framing, prior work, H1–H5, working rules) is a new "Research extension" section.
  PHASES_10-15.md itself is left untracked and unchanged (its content now lives in CLAUDE.md).
- **Numbering conflict:** PHASES_10-15.md assigns FINDINGS #7 both to the Phase 9.5 probe and to the Phase 10
  rounding audit. Your brief for today says probe = #7, rounding audit = #8, so CLAUDE.md records the audit as #8
  with a note.
- **Ordering:** CLAUDE.md says "do not start phase 8 or 9 until phase 7 has produced numbers". Phase 9.5 borrows the
  probe from phases 9/13 early; your brief approves that, and CLAUDE.md now states the exception explicitly.

### D44. Phase 9.5 step 1: σ = 0 alert and the standing post-hoc baseline (monitoring only)
- **Training:** every step logs `n_sigma_zero`, `sigma_zero_share`, the pooled rolling share over
  `window_steps` (25) and `sigma_zero_alert`. When the rolling share crosses `threshold` (0.5), a line goes to the
  console and `events.jsonl` (and again when it clears). It never changes the reward, sampling or stopping.
  Settings are in the new `configs/monitor.yaml`, shared by training and eval.
- **Eval reports:** `metrics.json` gets `sigma_zero_rate` per split and a `posthoc_baseline` block; `summary.md` gets a
  "Standing calibration baseline" table with (a) this model's stated σ, (b) this model's σ × 18.1, and (c) when
  the eval is on the same frozen split, generator and verifier as the fit, SFT-only unscaled and × 18.1 (the
  baseline RL must beat). `rescore --current-verifier` writes the same block.
- **Why show (b):** it tells apart "RL changed σ's level" (fixable by a scale) from "RL changed σ's ranking"
  (not fixable by a scale). Pooled windows (counts, not a mean of per-step shares) avoid over-weighting steps with
  few parsed answers.

### D45. Probe implementation details (Phase 9.5 step 2, within PREREGISTRATION family P)
- **Fit set size:** 2,000 calibration-split completions (the 400 post-hoc problems + 1,600 new, seeds
  20,000,400–20,001,999), generated with the eval's exact settings. 400 alone is too few for 1,536-dim probes.
- **Token:** the last token whose characters start before `<answer>`, from a teacher-forced pass over the exact
  prompt (chat template as in run_eval) + the logged completion text. Re-tokenizing the decoded completion can
  differ from the sampled ids in rare cases; it is the same text, so this is accepted.
- **Backbone only:** hidden states come from `model.get_decoder()` (no LM head), which saves ~2 GB of logits per
  batch. Layer 0 = embedding output, layer 28 = the last block (after the final norm in HF's convention).
- **Metadata baseline on depth 5:** depth 5 never occurs in the fit set, so its one-hot is unseen and those test
  rows are scored on family alone (`handle_unknown="ignore"`).
- **MLP probes** use the same CV and layer-selection rule as the linear ones, controls included (the
  pre-registration's "same pipeline"); they are secondary and outside the Holm family.

### D46. Split-credit trainer design (Phase 9.5 step 3)
- **Mechanism:** a subclass of the (curriculum) GRPO trainer overrides `_generate_and_score_completions`. After TRL
  has generated, scored and computed its per-sequence advantages, the subclass replaces `advantages` with a
  (batch, completion length) tensor. Unsloth's loss (`grpo_compute_loss_slow` and the chunked
  `grpo_accumulated_loss`) only unsqueezes 1-D advantages and otherwise uses them elementwise, and TRL's
  shuffle/split helpers index dim 0, so per-token advantages pass through unchanged. A test runs Unsloth's real
  loss function on 2-D advantages and checks loss and gradients against a reference DAPO implementation.
- **σ tokens:** the tokens overlapping the σ number after the first ± marker (±, +/-, +-, \pm) in the single answer
  block, the same "σ number only" convention as the SFT masking (D35). The ± sign, units and everything else are
  answer tokens. Token-to-character offsets come from re-encoding the decoded completion when that reproduces
  the sampled ids; otherwise from decoding every prefix. Rows that still can't be matched get no σ tokens and
  are counted (`unmatched_offsets` in `split_credit.jsonl`).
- **Reward parts:** r_answer = w_final·crps_log(μ, σ_ref) + w_steps·steps with σ_ref = 1% of μ, r_sigma =
  w_final·[crps_log(μ, σ) − crps_log(μ, σ_ref)]; they sum to the verifier's reward. Failures keep everything in
  r_answer (r_sigma = 0). Both parts use TRL's exact group normalization (`nanstd`, + 1e-4).
- **Alignment guard:** the reward function stores the parts in completion order; the trainer checks the count and
  that `batch_decode(completion_ids)` equals the texts the reward saw, and raises otherwise. A silent
  misalignment would corrupt training.
- **Unchanged:** the verifier, the logged reward, the curriculum (which still reads σ-free binary correctness),
  every hyperparameter. A test checks that `grpo_split_credit.yaml` differs from `grpo_overnight2.yaml` only in
  `split_credit` and the run id.

### D47. Probe: the pre-registered sanity check failed; exploratory baselines added afterwards
- **What happened:** the random-init probe matched the SFT probe on correctness (0.914 vs 0.917, p = 0.40). Per
  PREREGISTRATION.md, P1–P3 are reported as uninterpretable, even though each passes its Holm-corrected test.
- **Added after seeing results, labelled exploratory:** template one-hot, TF-IDF bag-of-words, length, and the
  real-vs-random-init test on log|error|. They explain the failed sanity check (the signal is readable from the
  completion text); they do not rescue P1–P3.
- **Not done:** re-running with other probe positions or a residualized probe (beyond what a bag-of-words predicts).
  Both would be new, pre-registered experiments if pursued.

### D48. Rounding audit scope (Phase 10 step 2)
- **Finding first:** the truth already uses the displayed givens (D4, verified: relative difference 0.0 on all 400
  eval problems), so the brief's "if so, compute the implied σ*" does not apply literally: σ* = 0 for the problems
  as defined.
- **Reported anyway:** (a) the counterfactual σ* if the 3-s.f. givens were measurements (uniform half-ulp error,
  2,000 Monte Carlo draws per problem), because that is the noise floor the Phase 10 noise mode must dominate;
  (b) the error of a perfect solver that rounds intermediates to 3/4/6 s.f., because that is the only irreducible
  error for the current problems and explains the "exactly right" error spike.
- **Units:** every display unit in the three families is a pure scale of its SI unit, so a relative half-ulp error
  in the display unit equals the same relative error in SI (no offset units such as °C yet).
- **Found by the smoke test (first attempt crashed, log kept as `results/smoke_split_credit_attempt1_crash.log`):**
  Unsloth's fused loss left-packs each micro-batch (every pad token moves right) and scores the last L + M
  columns, M = the largest prompt left-padding in the micro-batch, so row i's completion token j sits at column
  M − leftpad_i + j. 1-D advantages broadcast over that; 2-D ones did not match (163 vs 181 columns). Fix:
  the subclass overrides `compute_loss` and re-lays the advantages into that layout (`pack_like_unsloth`) before
  calling Unsloth. A test builds a batch with mixed padding and checks, with Unsloth's own `left_pack_padding` and
  `create_completion_attention_mask`, that every completion token's advantage lands on that token.
- **Smoke test (10 steps, `results/train/smoke-split-credit/`):** 40.0 s/step (40.1 over steps 2–10 vs 43.0 for the
  uniform-credit run's same steps), peak torch reserved 6.46 GiB, 0 allocator retries, 0 OOMs. σ tokens found
  in 94–100% of completions (2–5 tokens each; the rest are parse failures), 0 offset mismatches, no group with
  zero r_sigma variance, curriculum cells identical to the uniform-credit run for all 10 steps. Tape replays
  exactly. The long run was not started (your instruction).

## Amendment, 2026-10-14 (laptop day): conditional post-hoc baselines

### D49. Two more post-hoc baselines for every calibration comparison; RL calibration claims are judged against the text-feature one
- **Status:** an amendment to the analysis plan, dated 2026-10-14 and committed **before any split-credit result
  exists** (the split-credit long run has not started; only its 10-step smoke test exists, and that was never
  evaluated). PREREGISTRATION.md is not edited: family S and family H run exactly as pre-registered, and
  everything below is reported *next to* them.
- **Why:** FINDINGS #7. A bag-of-words reader of the completion predicts correctness at AUROC 0.906, and the
  standing control (one global σ × 18.1) cannot use any of that. If RL only learns what a text classifier
  fitted after the fact already gets, that is not a calibration result. So the bar is raised to a post-hoc
  method that sees the same text.
- **(a) Text-feature post-hoc ("text post-hoc"), the new control:**
  - **Form:** σ' = c(x)·σ_eff with log c(x) = b + w·φ(x). This is a σ *scale*, so the global scale is the
    intercept-only special case.
  - **σ = 0 handling:** σ_eff = max(σ, f·|μ|), where f is the verifier's crps_log σ floor (1e-6 relative).
    crps_log already scores a stated σ = 0 exactly as this floor, and it matches PREREGISTRATION.md's σ = 0
    convention. Without it, the 23.5% of SFT-only answers that state σ = 0 could never be rescaled.
  - **Features φ(x):** binary bag of words (presence; whitespace tokens; unigrams and bigrams; minimum document
    frequency in `configs/posthoc.yaml`), with separate vocabularies for the problem text and the completion.
  - **Fit:** maximize mean crps_log (the RL objective, as for × 18.1) minus an L2 penalty on w (not on b).
    L-BFGS, started from the global fit.
  - **Fit data:** the calibration split only, i.e. the 2,000 existing SFT-only completions on training seeds
    20,000,000–20,001,999 (the 400-problem post-hoc set plus the 1,600-problem probe fit set). The penalty is
    chosen by 5-fold CV on that split (held-out mean crps_log), with the vocabularies refit inside each fold.
    Eval sets are never used for fitting.
  - **Application:** applied unchanged to any model's completions (SFT-only or RL).
- **(b) Oracle per-cell post-hoc, an upper bound only:**
  - **Form:** for each (family, depth) cell, the scale c on σ_eff that maximizes mean crps_log.
  - **Fit data:** fitted **in-sample on the evaluated completions themselves**, so it uses the eval truths.
    That is what makes it an oracle. Calibration-split data could not cover it anyway: there is no depth-5
    calibration data.
  - **Use:** it is the best mean crps_log any per-cell scale can reach on those exact problems. It is never
    a competitor and never tested.
- **Reporting:** every eval report (`metrics.json` → `posthoc_baseline`, and the `summary.md` table) adds four
  rows: this model × (a), this model × (b), and on the same frozen split SFT-only × (a) and SFT-only × (b).
  The × 18.1 rows stay.
- **Judgement rule:** a claim that RL improved calibration is judged against **SFT-only + (a)**. Three added
  comparisons are run on the pooled frozen eval sets (n = 400). Each uses a paired bootstrap over problems
  (B = 10,000, seed 0), is one-sided, and is Holm-corrected within this amendment family A:
  - **A1:** |1σ coverage − 68.27%| of the RL model's stated σ < that of SFT-only + (a). The denominator is all
    problems, and failures count as not covered, as in family S.
  - **A2:** Spearman ρ(relative σ, |relative error|) of the RL model > that of SFT-only + (a).
  - **A3:** mean crps_log of the RL model > that of SFT-only + (a). This also moves with accuracy, so it is read
    together with A1 and A2.
  - "RL calibration beats post-hoc" requires A1 **and** A2 to be supported.
  - With 3 seeds, the same per-seed and seed-mean rule as family S applies.
  - For family H (phases 10–14), wherever "post-hoc" appears (H2b, H4a, H4b), the pre-registered test runs as
    written, and the same test against (a) runs next to it. (a) is refit on that phase's calibration split.
- **Alternatives considered:**
  - A text model that predicts σ directly and ignores the stated σ. Rejected: the brief asks for a scale, and a
    scale nests the global baseline.
  - TF-IDF features as in the FINDINGS #7 exploratory model. Rejected: its l2 row normalization shrinks a
    single decisive token such as "± 0" in long completions.
  - An oracle per cell fitted on the calibration split. That would be a legitimate baseline, not an oracle,
    and it cannot cover depth 5.
- **Implementation notes (same day, before the commit; still before any split-credit result):**
  - **λ grid widened.** The first λ grid (1e-6 … 1e-1) put the CV optimum at its edge, so it now runs from 1e-5
    to 10 in half-decades. Selection still uses calibration-split CV only. Result: λ = 0.03 for (a), 0.003 for (a′).
  - **(a′) "text direct" added, as a reported sensitivity row only.** It is the same model with σ' = c(x)·|μ|, so it
    ignores the stated σ.
    - **Why:** calibration-split CV showed that the stated σ carries *less* than no information. A single
      constant relative σ (CV mean crps_log −0.2322) beats the global scale on the stated σ (−0.2359). The scale
      form (a) only reaches −0.2331, while (a′) reaches −0.2248.
    - **Status:** it was added after these CV numbers and the SFT-only eval numbers were seen. **The control
      named in the judgement rule stays (a)**, as the brief specifies. The family-A tests are also run against
      (a′) and reported as sensitivity. Promoting (a′) to the control is your call, and it should be made before
      any split-credit result is read.
  - **Oracle (b) search range.** It searches log10 c ∈ [−6, 6], not the global fit's [−2, 6]. energy/1 hit the −2
    bound because its stated σ sits far above its errors.
  - **Not a universal upper bound.** (b) bounds what any *per-cell scale of the stated σ* can reach. It does not
    bound (a′): on the SFT-only eval sets, (a′) already beats (b) on mean crps_log.
  - **L-BFGS settings.** max_iter is 5,000 (the direct form needs ~1,400 iterations at its CV optimum; the
    objective agreed to 6 digits with a 1,000 cap). gtol = 1e-10 and ftol = 1e-12: the gradients scale with the
    relative σ, and scipy's default tolerances would stop at the starting point.

## Amendment, 2026-10-14 (afternoon): external reviews, A1–A8

### D50. Amendments A1–A8 from REVIEW_SYNTHESIS_2026-10-14.md (commit `f8b24c4`)
- **Status:** recorded and pushed before any Phase 10+ result and before any result of a new experiment.
  PREREGISTRATION.md is not edited. Where A1–A8 change a pre-registered test, the pre-registered version is still
  computed and reported next to the amended one, and the write-up says the amendment came after the
  pre-registration.
- **Results that already existed today when this was written:**
  1. **D49** baselines (a), (a′), (b), with SFT-only numbers on the frozen eval sets (commit `6eeef4b`). A5
     supersedes D49's judgement rule, and this amendment came after those numbers were seen. The D49 rows stay
     in every report and the D49 tests stay in the compare script, labelled "superseded".
  2. **Dry run of the D49 compare script** (not committed): uniform-credit RL@287 stood in for the split-credit
     run, and it failed both D49 controls. This is not a split-credit result.
  3. **The Phase 10 self-consistency gate has not run** under any rule (it was paused at your request before
     it ran). It runs once, under A3. The superseded rule ("Kupiec p > 0.05 in every cell") is evaluated on the
     same run and reported, but A3 decides.
  4. **No split-credit result exists.**

**A1. Two estimands.**
- **Epistemic:** P(answer correct within tolerance), stated as a confidence p and scored by Brier.
- **Aleatoric:** σ\* under the measurement model.
- **Proper scores** (crps_log, log score) against the realized truth stay valid on all problems.
- **Computation-correct subset:** every comparison of stated σ with σ\* (H1, H3, the calibration surface) is
  computed only on answers with |ln μ − ln f(x_meas)| ≤ τ. Here τ = 0.1% relative (in config), and
  f(x_meas) is the chain evaluated on the displayed noisy givens, stored per problem by the generator. It is
  not the clean truth. All-problem versions, including the pre-registered "all noise > 0 problems" version of
  H1, are reported as secondary.
- **Regimes:** noise-free and noisy problems are reported as separate regimes.

**A2. σ\* definition.**
- σ\* comes from the Monte Carlo predictive distribution of the clean truth given the displayed givens, in log
  space. Each noisy given is drawn as x = x_meas/(1 + e′), e′ ~ N(0, s²); exact givens and constants are fixed;
  the draws are pushed through the chain.
- **Primary target σ\*:** the CRPS-optimal Gaussian log-scale for that distribution, centred at
  ln f(x_meas).
- **Also stored:** the SD of the log draws (σ\*_sd, the original spec's definition) and the delta-method value.
- **Distribution check:** the PIT of the clean truth under the MC distribution should be close to uniform (KS
  test).

**A3. Generator self-consistency gate** (replaces "Kupiec p > 0.05 in every cell").
- **Run:** 10,000 problems, cells = depth (1–5) × noise level (1, 2, 5, 10%), 500 per cell, families round-robin.
- **Intervals:** ln f(x_meas) ± k·σ\* with k = 1 and 2, at nominal 68.27% and 95.45%.
- **The gate passes iff all four hold:**
  1. Pooled Kupiec at both levels: p > 0.05.
  2. Per-cell Kupiec, Holm-corrected across the 20 cells (separately per level): no rejection.
  3. At each level, ≥ 95% of cells have coverage inside the central 99% binomial interval around nominal.
  4. Pooled PIT KS over all 10,000 problems: p > 0.05.
- **Noise-0 problems:** checked separately for exactness (truth = f(x_meas)), with no coverage test.
- **If it fails:** stop and report, and do no further Phase 10+ work (your brief).

**A4. Statistics.**
- Cross-cell coverage homogeneity is a Pearson chi-square test (CMH when stratified), never called
  Christoffersen. This renames and redefines H2's pre-registered "Christoffersen-style" description.
- Paired bootstrap and permutation tests are primary. Diebold–Mariano (pre-registered in Phase 13) becomes a
  robustness check.
- **Mincer–Zarnowitz:** ln|error| = α + β·ln σ (ideal α = 0, β = 1; R² reported). On computation-correct answers
  regressed on σ\*, it is the formal vol beta for H1. On all answers, it measures total-uncertainty calibration.
- **Also added:** interval score / WIS decomposition, PIT histograms, and coverage stratified by the chain's
  condition number.

**A5. Nested baselines** (supersede D49's judgement rule; D49 rows stay as reported context).
- **Feature sets, fitted on the calibration split:**
  1. metadata only (family, depth);
  2. problem text;
  3. problem text + the generated `<answer>` block;
  4. problem text + answer + reasoning (all completion text before `<answer>`, including `<steps>`).
- **Features:** binary bag of words / TF-IDF over whitespace tokens. No verifier-derived features: no parsed μ,
  σ, units, correctness or step credit. The parsed μ is used only to centre a forecast and to turn a relative σ̂
  into absolute units.
- **Models** (each feature set × {linear, small GBT}):
  - p̂ = P(correct within 1%): logistic regression / GBT classifier.
  - σ̂: ridge / GBT regression of ln max(|ln(μ/y)|, 1e-6), then σ̂_rel = c·exp(ŷ), with c fitted for mean
    crps_log on out-of-fold predictions.
  - Failures are excluded from fitting; they score the floor regardless.
- **Model selection:** the same budget for every baseline: a fixed grid per model type, and 5-fold CV grouped
  by chain template (family | target | set of givens), so templates are held out within the calibration split.
  The depth-5 eval split has unseen templates by construction.
- **"Strongest":** the best CV mean crps_log for σ (best CV Brier for p) on the calibration split, chosen before
  the baselines are applied to any eval set or RL completion.
- **Judgement:** RL (and E1) uncertainty claims are judged against the strongest baseline **applied to the same
  completions**. μ is then identical, so only σ differs, which answers the reviewers' confound. The baseline is
  fitted on SFT-only calibration completions (the only ones that exist) and refit when an RL model's
  calibration completions exist. Tests: paired bootstrap over problems, one-sided:
  - N1: |1σ coverage − 68.27%| is smaller.
  - N2: Spearman ρ(σ, |error|) is higher.
  - N3: mean crps_log is higher.
  - Holm-corrected over {N1, N2, N3}. "Beats the baseline" = N2 and N3 supported; N1 is reported.
- **Hidden-state probes** are reported as a separate source only if they beat baseline 4 (FINDINGS #7 already
  suggests they will not).

**A6. Counterfactual sensitivity pairs** (propagation vs lookup).
- **Construction:** matched pairs with the same chain template, depth, phrasing and stated noise levels, where
  the numbers differ so that σ\* differs.
  - For one chain, draw K candidate worlds and compute each world's delta-method σ\*.
  - Keep the lowest- and highest-σ\* worlds if their ratio is ≥ r_min. K and r_min are in config.
  - σ\* is then computed by MC (A2).
  - The κ ≤ 5 filter still applies, so pairs come from templates whose sensitivity varies with the numbers
    (sums, differences, quadratics). Pure power-law chains cannot form pairs.
- **Primary analysis:** on pairs where both answers are computation-correct (A1): the within-pair OLS slope of
  Δln(stated σ) on Δln σ\*, one-sided paired bootstrap for slope > 0. A surface-feature lookup predicts ≈ 0.

**A7. Cuts.**
- H4b (Kelly growth) is dropped and replaced by the log-score difference against a fixed-width benchmark.
- The market-scoring-rule idea is dropped.
- The Basel traffic light is limited to at most one summary table.
- The stress suite is reduced to the A6 pairs plus unit swaps (H5 reduced accordingly).
- The LoRA ensemble becomes a free appendix from the 3 seeds.
- Phase 13 core: stated σ, the strongest A5 baseline, and k = 8 sampling spread. The probe is included only if
  it beats baseline 4.

**A8. Headline question:** *Can a small language model learn to distinguish uncertainty inherited from the world
(noisy inputs) from uncertainty introduced by its own reasoning, and communicate both in a way that improves
decisions?* The contribution is a controlled testbed that separates the two, plus an analysis of when and why RL
fails or succeeds; a clear mechanistic negative result counts. Where CLAUDE.md's phase text conflicts with A1–A8,
A1–A8 win; CLAUDE.md is to be updated with the write-up framing.

### D51. Generator 2.0.0: the measurement-noise mode (Phase 10 step 3; σ\* per D50/A2) **[REVIEW]**
- **Where it lives:** `configs/env_v2.yaml` (version 2.0.0 = 1.1.0 plus a `generator.noise` block). `configs/env.yaml`
  stays at 1.1.0 and stays the default, so tonight's split-credit run, E1, the frozen 1.1.0 eval sets and every
  existing tape are untouched; Phase 10+ code loads `env_v2` explicitly. 1.1.0 problem dicts are byte-identical
  (the frozen-set test passes); the new `noise` field and a given's `noise_rel` are omitted when absent/zero.
- **Noise dial:** one level per problem, s ∈ {0, 1, 2, 5, 10%}, applied to every given and every distractor (so
  "± p%" does not mark the relevant values). Displayed = round(x_true·(1 + e), 5 s.f.), e ~ N(0, s²), shown as
  "2.4013 kg ± 2%", plus one sentence saying what ± p% means (one-SD relative uncertainty; unmarked values exact).
  Mixed per-given levels are representable (`noise.givens[sym].rel`) but not generated yet.
- **Truth:** the noise-free chain on the unrounded true values. **Noise 0** behaves exactly like 1.1.0 (3 s.f., truth
  from the displayed values, D4), so σ\* = 0 and f(x_meas) = truth there.
- **Paired by construction:** the noise level is not part of the main random stream; e = s·z with z from a side
  stream, and the MC draws use another side stream shared across levels. The same seed at different levels has
  the same chain, world, units, phrasing and clean truth, with the measurement errors scaled by s (tested).
- **Step credit [REVIEW]:** `intermediates` (what the verifier credits within 2%) are the chain on the *displayed*
  values, i.e. what a correct solver computes; the clean intermediates are stored in
  `noise.intermediates_true_si`. Alternative: clean intermediates, which would penalize correct arithmetic by
  ~s·κ and make step credit noisy at 5–10%.
- **Stored per problem (`noise`):** level, f(x_meas) (`point_estimate_si`), true values and realized errors,
  σ\* = CRPS-optimal Gaussian log-scale centred at ln f(x_meas) (A2 primary), the SD of the log draws, the
  delta-method σ, the SD in SI, and the PIT of the truth under the 4,000 MC draws.
- **Rejection:** an attempt is resampled if f(x_meas) or any MC draw leaves the valid domain (on top of the κ ≤ 5
  filter on the true values). This conditions on the noise draw; it is rare with κ ≤ 5, and the A3 gate measures
  the net effect.
- **Cross-check (step 3):** for s ≤ 2%, MC SD vs delta method and CRPS-optimal vs SD agree within 6% on 45 problems
  per level (test).

### D52. Pre-registration of E1, the fixed-μ σ-only pilot (written before any E1 data or result exists)
- **Question:** if μ is held fixed, so that the advantage compares σ values only (the reviewers' clean credit
  assignment), can GRPO teach the SFT model a σ that beats (i) its own stated σ, and (ii) the strongest A5
  surface-feature baseline on the same completions?
- **Model:** the SFT-only merged model (the start of every RL run), with a fresh LoRA (r = 16, α = 16, the same
  target modules), Unsloth 4-bit, generator 1.1.0, verifier 1.1.0.
- **Training prefixes:**
  - The 2,000 existing SFT-only greedy completions on the calibration split (training seeds 20,000,000–20,001,999:
    `results/posthoc_calibration/sft_calibration_set` and `results/probe_sft/sft_calibration_ext`).
  - Each is truncated right after the first "±" (or +/-, +-, \pm) inside its single `<answer>` block, keeping that
    marker.
  - Completions without a parsable answer or without the marker are excluded; the count is reported.
  - Prompt = the exact chat-template prompt (v4) + the truncated completion as the start of the assistant turn.
- **Generation:** 8 samples per prefix, T = 1.0, top_p 1, top_k off, repetition penalty 1, max 24 new tokens. That
  is the review's "about 12" plus margin for long σ strings and units; a truncated suffix is a parse failure.
- **Reward:**
  - **E1a:** the unchanged verifier reward (crps_log + 0.2·steps) on prefix + generated suffix. Steps and μ sit in
    the prefix, so within a group only σ varies. Exception: a suffix whose unit changes μ in SI; this is logged
    as `mu_changed` (at eval it counts as a failure).
  - **E1b:** the same, except that the stated relative σ (σ/|μ|) is first mapped to the nearest of nine bins
    {0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 100%} in log space (σ = 0 maps to 0.1%). This is the review's "verifier
    mapping" option; constrained decoding was not used. The tape logs the unchanged verifier reward (replay stays
    exact), and the mapped training reward is logged next to it.
- **Optimization:** identical to the split-credit run wherever it applies:
  - 4 prefixes × 8 samples per step, micro-batch 2 × 16;
  - lr 5e-6 constant after 5 warmup steps, AdamW-8bit, grad-norm clip 0.1, β = 0, DAPO loss, 7.5 GiB allocator cap.
  - **Budget:** 300 steps (1,200 prefixes, sampled without replacement in a fixed shuffled order, seed 0), plus a
    permanent checkpoint at step 150 and the hard deadline in tonight's script.
  - **Why lr is not raised:** the DAPO loss averages over completion tokens, and a ~12-token suffix already
    gives each σ token a much larger share of the gradient than in the full-completion runs.
- **Evaluation:**
  - **Data:** the SFT-only greedy completions on the frozen 1.1.0 eval sets (verifier-1.1.0 tape, 200 ID +
    200 depth 5), truncated the same way.
  - **Decoding:** the E1 model writes the σ suffix greedily (24 tokens). The full text is scored by the verifier;
    E1b is scored with its bin mapping.
  - **Comparators** (on the same completions, so μ is identical): SFT-only's own stated σ (raw), and the
    strongest A5 σ baseline. The A5 baseline is selected on the calibration split before E1 is evaluated.
  - **Exclusions and failures:** problems whose SFT completion has no ± are excluded from all arms (count
    reported). E1 outputs that fail to parse, mismatch units, or change μ count as failures (crps_log floor −5,
    not covered).
- **Hypotheses (family E):** one-sided paired bootstrap over problems, B = 10,000, seed 0, Holm over the four
  tests:
  - **E1a-P:** mean crps_log of E1a's σ > that of the SFT stated σ.
  - **E1b-P:** the same for E1b.
  - **E1a-N:** mean crps_log of E1a's σ > that of the strongest A5 σ baseline.
  - **E1b-N:** the same for E1b.
- **Reported, not tested:**
  - Spearman ρ(σ, |error|), for E1, SFT and the A5 baseline;
  - σ-bin entropy over the 9 bins (free-form σ mapped to the nearest bin for E1a and SFT);
  - 1σ coverage (target 68%), sharpness (median relative σ) and the σ = 0 share;
  - parse-fail, unit-mismatch and μ-changed rates;
  - per-step training reward and bin entropy;
  - SFT's σ mapped to bins (descriptive);
  - E1a vs E1b at matched steps (300, or 150 if either stops early).
- **Interpretation rule:**
  - E1x-N supported → the LM's own σ adds information beyond surface features.
  - E1x-P supported without E1x-N → σ is learnable with fixed μ, but not beyond surface features. That is the
    clean negative result the review describes.
  - Neither supported → the σ channel did not learn in 300 steps.
- **Pipeline sanity (not a result):** if more than 10% of step-1 suffixes fail to parse, or the run OOMs, the
  run is reported as a pipeline failure.

### D53. A5 and E1 implementation notes (no result had been seen when these were fixed, except where stated)
- **A5 (`physcal/eval/nested_baselines.py`, `configs/nested_baselines.yaml`):**
  - **Features:** TF-IDF unigrams over whitespace tokens (min_df 3, sublinear tf), one vocabulary per field. The
    GBT uses the 1,000 most frequent terms per field as dense input.
  - **Grids, the same for every feature set:** ridge α (5 values), logistic C (5 values), and 4 HistGradientBoosting
    configurations.
  - **CV:** 5-fold GroupKFold by template (865 templates in the 2,000 calibration completions).
  - **σ target:** the σ target is ln max(|ln(μ/y)|, 1e-6) and σ_log = c·exp(ŷ), with c fitted for crps_log on the
    out-of-fold predictions.
  - **Answers with μ ≤ 0** (one SFT eval answer) count as failures for the baselines; crps_log already gives them
    the clip.
  - **Selection** (`selection.json`) was written before any eval-set use. A rerun to fix a pickling path
    (`__main__.Candidate`) reproduced it byte for byte.
- **Superseded, not deleted:** D49's rows and family A remain in the reports and the compare script, labelled as
  superseded. The judgement is now family N (N1–N3) against the strongest A5 σ applied to the *same*
  completions.
- **E1 (`physcal/train/e1.py`, `e1_compare.py`, `configs/e1.yaml`, `configs/grpo_e1{a,b}.yaml`):**
  - **Prefixes:** 1,996 training prefixes (3 completions without ± and 1 unit mismatch excluded) and 400 eval
    prefixes (none excluded).
  - **Token budget:** the longest prompt + prefix is 1,476 tokens, so `max_prompt_length` is 1,536 and never
    truncates. The original σ suffixes are ≤ 15 tokens; the budget is 24.
  - **Training reward:** "SFT completion with a parsed answer" = verifier reason ok.
  - **μ ≤ 0:** an answer with μ ≤ 0 keeps the crps_log clip and is excluded only from ρ.
  - **Code paths:** E1 has its own trainer (it reuses the GRPO callbacks); `grpo.train`, which runs tonight's
    split-credit run, is untouched.
  - **CPU checks:** prefix + the SFT's own suffix reproduces the logged SFT reward exactly on all 400 eval
    problems. E1 tapes replay exactly. An echo model run through eval and compare gives Δcrps_log = 0.
  - **Not testable today:** the Unsloth training step itself has had no GPU smoke test (no GPU today), so the
    first E1 step tonight is its smoke test.

## Decisions after the first Phase 10 gate (2026-10-14, evening)

### D54. Noise dial capped at 5% for the main experiments; 10% becomes a "heavy-tail" regime (your decision)
- **This cap was chosen after seeing the gate result (FINDINGS #9).** It is a post-hoc design change.
- **Main experiments** (Phases 11–14: frozen eval sets, curriculum, H1–H5, calibration surface) use noise levels
  {0, 1%, 2%, 5%}.
- **10% "heavy-tail" regime:**
  - It is scored only by crps_log against the Monte Carlo predictive distribution: the expected CRPS of the
    stated Gaussian over the generator's MC draws (regenerated from the problem's seeded MC stream). It may also
    be scored against the realized truth, which A1 keeps valid everywhere.
  - No coverage claims and no coverage tests are made in this regime. It is always reported separately.
- **The generator is unchanged:** generator 2.0.0 (`configs/env_v2.yaml`, `2.0.0+bdbb2279`) still generates
  10%. Which levels an experiment draws is set in that experiment's config, so problem identities and the
  version string do not change.
- **Gate rerun** on {0, 1, 2, 5}% under the A3 rules:
  - **Sample:** 10,005 noisy problems (15 cells = depth 1–5 × {1, 2, 5}%, 667 per cell, families round-robin)
    plus 500 noise-0 problems for the exactness check.
  - **Fresh seeds:** offset 1.6e9, disjoint from the first gate run (1.5e9). The rerun is therefore an
    independent sample, not a re-score of problems already seen.
  - **Config and output:** `configs/sigma_star_gate_capped.yaml`, `results/sigma_star_gate_capped/`.
  - **Next step:** build A6 if it passes; stop and report if it fails.
- **Stated before the rerun: what A3 means with 15 cells.**
  - "≥ 95% of cells inside the 99% binomial interval" with 15 cells means all 15 (14/15 = 93.3% < 95%).
  - A valid generator passes that criterion with probability ≈ 0.99^15 ≈ 0.86 per level, against ≈ 0.98 with the
    20 cells of the first run.
  - The rule is applied as written. If the rerun fails only on it, that is reported as a failure. The number of
    cells outside is reported either way.

### D55. Step credit on noisy problems uses displayed-value intermediates: approved (closes D51's [REVIEW])
- **Your decision (2026-10-14):** the D51 choice stands. On noisy problems, `intermediates` (what the verifier
  credits within 2%) are the chain evaluated on the *displayed* givens, i.e. what a correct solver computes. The
  clean intermediates stay in `noise.intermediates_true_si` for analysis only.
- **No change:** neither code nor the generator version changes.

### D56. Idea log, NOT pre-registered: E5, distill the A5 calibrator into the model
- **Idea (yours, 2026-10-14):** fine-tune the model by SFT so that its stated σ reproduces the σ of the strongest
  A5 baseline (FINDINGS #10: problem + answer + reasoning, GBT, fitted on the calibration split). Then compare
  out-of-distribution calibration of the distilled model against the external calibrator applied to the same
  completions, on depth 5 and the held-out family.
- **Question:** does a calibrator internalized in the LM generalize OOD better or worse than the same calibrator
  kept outside on surface features? The external A5 model sees only TF-IDF/GBT features. The distilled model may
  learn something that transfers to unseen templates, or it may copy the surface mapping.
- **Status:** an idea only. Nothing is designed, run or pre-registered. Before any E5 result exists, it needs a
  pre-registration entry with:
  - training data (calibration split only, no eval seeds);
  - whether σ tokens are unmasked (they were masked in D35's SFT);
  - the OOD splits (the held-out family needs optics or circuits built first);
  - metrics (crps_log, 1σ coverage, ρ, per split);
  - the control (the external A5 σ on the same completions).

### D57. A3's cell-share rule is replaced by a miss-count test; the next gate rerun is FINAL (your decision, recorded before any rerun)
- **Why:** the share rule ("≥ 95% of cells inside their 99% binomial interval") was a **design error**. With 15
  cells per level it demanded zero misses, so even a valid generator would fail it about 14% of the time per
  level. The error was found before the first rerun (D54 spelled out the zero-miss consequence before that run).
  FINDINGS #11 then confirmed it: that rerun failed only on this rule, with one cell outside.
- **New rule (per level k, replacing the share rule):** under a valid generator, each cell falls outside its
  central 99% binomial interval with probability ≤ 0.01. The number of such cells, X, is therefore (at most)
  Binomial(n_cells, 0.01). The test rejects if P(X ≥ observed) < 0.01.
  - With 15 cells, at most 1 miss is allowed: P(X ≥ 1) = 0.140, P(X ≥ 2) = 0.0096.
  - With 20 cells, at most 2.
  - Because the discrete binomial interval holds at least 99%, the Binomial(n, 0.01) null is conservative.
- **Everything else in A3 is unchanged:**
  - pooled Kupiec at both levels: p > 0.05;
  - per-cell Kupiec with Holm: no rejection;
  - pooled PIT KS: p > 0.05;
  - noise-0 exactness.
- **FINDINGS #9 and #11 stand as reported.** Noted for transparency, not used: under D57, #11's data would have
  passed (one miss at ±2σ\*, all other criteria passing). #9 would still fail, on the Holm criterion. Neither
  verdict is revised; the decision rests on the new run below.
- **The next rerun is FINAL:**
  - **Sample:** fresh seeds (offset 1.7e9, disjoint from both earlier runs) and D54's levels {1, 2, 5}% × depth
    1–5, 667 per cell (10,005 problems), plus 500 noise-0 problems.
  - **Files:** `configs/sigma_star_gate_final.yaml`, output `results/sigma_star_gate_final/`.
  - **If it passes:** build A6.
  - **If it fails:** stop and diagnose. There are no further reruns without a specific, diagnosed cause.

### D58. A6 counterfactual sensitivity pairs: implementation (fixed before any model is evaluated on them)
- **Code:** `physcal/env/pairs.py` (construction), `physcal/eval/a6_pairs.py` (analysis), `configs/pairs.yaml`.
  Built on generator 2.0.0 after the final gate passed (FINDINGS #12).
- **Construction** (per family, depth, noise level ∈ {1, 2, 5}% and pair seed):
  1. Draw a chain, then 24 candidate worlds that pass the generator's validity and κ ≤ 5 filters.
  2. Compute each world's delta-method σ\* at its true givens. Take the lowest and highest.
  3. Keep the pair if σ\*_hi / σ\*_lo ≥ 2. Otherwise draw a new chain (up to 20).
  - **What the two members share:** chain, phrasing (render stream), given and target units (chosen once,
    from the lo world), distractors, noise level and the standardized measurement errors z.
  - **Stored σ\*:** the MC value (A2).
  - **Template bias:** templates whose sensitivity cannot vary (pure power laws) never form pairs. The pair set
    is therefore biased toward sum/difference/quadratic templates, as D50 anticipated; the manifest reports the
    family mix and the failures per cell.
- **Identity and replay:**
  - Member seeds are 2·pair_seed + {0 (lo), 1 (hi)}.
  - `noise.pair` stores the pair seed, member, chain try, delta ratio and the full construction config, so
    `generator.regenerate` replays from the problem itself. This needs no generator version change; the pair
    version is `1.0.0+<hash of the construction>`.
  - A refactor shared the noise-record code between the generator and the pairs. It was checked byte-identical
    on 72 generator-2.0.0 problems, and the frozen 1.1.0 sets still match.
- **Frozen set:**
  - **Size:** 15 cells (depth 1–5 × {1, 2, 5}%) × 34 pairs = 510 pairs, 1,020 problems.
  - **Seeds:** pair seeds from 1.8e9, families round-robin; at most 136 seeds are tried per cell, and shortfalls
    are reported.
  - **Files:** `data/eval_sets/2.0.0+bdbb2279/a6_pairs.jsonl` + `a6_pairs_manifest.json` (SHA-256).
- **Analysis (D50/A6's primary test, made concrete):**
  - **Variables:** x = ln σ\*_hi − ln σ\*_lo; y = ln σ_hi − ln σ_lo.
  - **σ:** the stated *relative* σ (σ/|μ|, crps_log's scale). σ = 0 is floored at 1e-6 relative (the
    PREREGISTRATION convention).
  - **Primary:** pairs where both answers are computation-correct (A1, τ = 0.1%). The OLS slope of y on x with an
    intercept, tested with a one-sided bootstrap over pairs (B = 10,000, seed 0) for slope > 0.
  - **Secondary:** all pairs with both answers parsed.
  - **Sensitivity:** primary pairs without a stated σ = 0.
  - **Reference points:** an honest propagator has slope ≈ 1, a surface lookup ≈ 0 (synthetic tests).
- **Not built yet:** an eval path that runs a model on a frozen problem file (run_eval builds splits from the
  generator). It is needed when a noise-trained model exists (Phase 12). Until then the pair set is used only
  through `a6_pairs.analyze` on an eval tape over it.

### D59. Pre-registration of A6b, invariance pairs (written before A6b is built; no model has been evaluated on A6 or A6b)
- **Purpose:** the second side of the propagation test.
  - **A6** asks whether the stated σ moves when σ\* moves (expected slope ≈ 1).
  - **A6b** asks whether the stated σ stays put when the values change but the relative σ\* does not (expected
    slope ≈ 0).
  - **What each kind of model does:**
    - a propagating model passes both;
    - a surface lookup passes A6b trivially and fails A6;
    - a magnitude-driven model fails A6b. For example, one that quotes a roughly fixed *absolute* σ, so that its
      relative σ shrinks as the values grow, gives a slope ≈ −1.
- **Templates:** power-law chains only, i.e. chains whose log-sensitivities d ln f / d ln x_i agree within 1e-5 at
  6 reference worlds. They exist at every depth. The structural count, a generator property checked before
  building:

  | family | depth 1 | depth 2 | depth 3 | depth 4 | depth 5 |
  |---|---|---|---|---|---|
  | kinematics | 15 / 42 | 18 / 181 | 0 | 0 | 0 |
  | dynamics | 40 / 43 | 142 / 158 | 225 / 290 | 177 / 250 | 12 / 97 |
  | energy | 23 / 30 | 43 / 85 | 39 / 112 | 28 / 89 | 8 / 36 |

  Depths 3–5 therefore come from dynamics and energy only.
- **Construction**, per pair seed and noise level ∈ {1, 2, 5}%:
  1. The family goes round-robin over families that have power-law templates at that depth. The chain is drawn
     uniformly from that cell's power-law templates.
  2. Draw 24 candidate worlds that pass the generator's validity and κ ≤ 5 filters.
  3. lo / hi = the worlds with the smallest / largest clean target value. Keep the pair if the ratio is ≥ 2;
     otherwise redraw the chain (up to 20 times).
  4. **Shared exactly as in A6:** template, phrasing, units (from lo), distractors, noise level, standardized
     errors z, and the MC draws.
  5. **MC check, required for every pair:** |ln(σ\*_hi / σ\*_lo)| ≤ 1e-3 (A2 σ\*). Pairs that fail are discarded
     and counted. For an exact power law the two members' MC log-draws differ by a constant, so the ratio should
     be 1 up to floating point.
- **Frozen set:** depth 1–5 × {1, 2, 5}% × 34 pairs = 510 pairs; pair seeds from 1.9e9.
  `data/eval_sets/2.0.0+bdbb2279/a6b_pairs.jsonl` + manifest.
- **Analysis:**
  - **Variables:** y = Δ ln σ_rel (stated, hi − lo; σ = 0 → the 1e-6 floor); m = ln(truth_hi / truth_lo)
    (≥ ln 2 by construction).
  - **Statistic:** the OLS slope of y on m, with an intercept, on pairs where both answers are
    computation-correct (A1, τ = 0.1%).
  - **A6b test:** equivalence (TOST) of the slope to 0 with margin δ = 0.3. The 90% bootstrap CI (B = 10,000,
    pairs resampled, seed 0) must lie inside [−0.3, 0.3].
    - The margin is a third of the way from the propagation ideal (0) to the absolute-σ habit (−1). It is fixed
      here.
  - **Also reported:** a two-sided bootstrap p for slope ≠ 0, the mean |y|, the slope's distance from −1, and
    the all-answered and no-σ = 0 versions (as for A6).
- **Joint verdict: the two-sided propagation test,** reported in one table with A6.
  - **"Propagates"** iff A6's primary test is supported (slope of y on Δ ln σ\* > 0, one-sided, D58) **and** A6b's
    slope is equivalent to 0.
  - This is an intersection-union test, each part at α = 0.05, so no multiplicity correction is needed.
  - **Labels:**
    - A6 fails → "no propagation (lookup-like)";
    - A6 passes and A6b fails → "magnitude-driven";
    - both pass → "propagates".
  - A6's slope is also reported against 1 (CI).
- **Known confound:** A6 needs templates whose sensitivity varies, and A6b needs constant sensitivity, so the two
  halves use disjoint template classes. They differ in template mix as well as in σ\* variation, and the write-up
  says so.

### D60. Eval path for frozen problem files
- **What changed in `run_eval`:**
  - A split can be `file: <frozen JSONL>` instead of families / depths / n.
  - The eval config's `env_config` (default `configs/env.yaml`) selects the generator. Every problem's
    `generator_version` must match it.
  - The file's SHA-256 is checked against `sha256` or the file's `*_manifest.json`.
  - `verify_regenerate` (N problems, −1 = all) regenerates problems and compares them before evaluating.
  - `--limit` takes the first N lines (use an even N for pair sets).
  - Replay already handles noisy and pair problems (`generator.regenerate`).
- **Config:** `configs/eval_frozen_sets.yaml` runs A6 + A6b with generator 2.0.0, full regeneration checks, greedy
  decoding, 512 tokens and prompt v4. The Phase 11 noise eval sets will be added as further `file:` splits once
  they are frozen; they are not built yet.
- **Report:** when `calibration.computation_correct_tau` is set (0.1%, A1), every split with noise information
  gets a per-noise-level table:
  - computation-correct rate;
  - 1σ coverage over answered and over computation-correct answers;
  - the median log10(stated relative σ / σ\*) on computation-correct answers.

  Pair analyses read one split of the tape (`a6_pairs analyze <tape> --split a6_pairs|a6b_pairs`, then
  `a6_pairs joint`).
- **CPU option:** `model.device: cpu` skips every CUDA call and the nvidia-smi poller. It exists only for
  pipeline checks.
- **CPU check (`tests/test_frozen_eval.py`):**
  - **Setup:** a tiny random 2-layer Qwen2 with the real tokenizer, run in a subprocess with
    `CUDA_VISIBLE_DEVICES=""`, on 4 A6 + 4 A6b problems + a small generator-2.0.0 noise file (levels 0–5%).
  - **Checked:** the tape replays exactly, both pair analyses run, and the per-noise table renders.
  - **Refusals tested:** a version mismatch, a wrong checksum, and a tampered problem.
  - **Not done:** no GPU was used and no real model was evaluated.

### D61. A6 slope labels (refines D59 and D50/A6; written before any model result on A6 or A6b exists)
- **Origin:** proposed in review on 2026-10-14. It was not committed then because the session ended. It is
  committed now (2026-10-16), and no model has been evaluated on A6 or A6b (only the D60 tiny-random-model pipeline
  check, which is not a result).
- **What is reported:** the A6 slope (Δ ln stated σ on Δ ln σ\*, computation-correct pairs, as in D58) with its
  90% CI.
- **Labels:**
  - **"full propagation"** if the CI excludes 0 and contains 1;
  - **"partial propagation"** if the CI excludes 0 but lies entirely below 1;
  - **"no propagation"** if the CI contains 0;
  - **"over-propagation"** if the CI lies entirely above 1.
- **Joint A6/A6b verdict:** uses these labels in place of "slope > 0". The existing pass/fail rules in D59 stay as
  written and are reported alongside.
- **Recording note (Claude, not part of the rule above; open for your decision):** read literally, a CI that lies
  entirely below 0 "excludes 0 but lies entirely below 1" and would be labelled "partial propagation". D59 already
  treats a negative slope as a distinct habit (the absolute-σ pattern gives about −1 on A6b), so this case
  probably needs its own label. It is not resolved here. The labels are not implemented in
  `physcal/eval/a6_pairs.py` yet; that waits until this case is settled.
- **Resolved in D64** (2026-10-17): a CI entirely below 0 is "anti-propagation".

### D62. Split-credit seeds 1 and 2; the seed-mean S1–S3 test; exploratory greedy vs sampled σ (written before any seed-1/2 data or sampled eval exists) **[REVIEW]**
- **Context:** your brief of 2026-10-16 evening. Seed 0's family S results are already known (seed-0 numbers,
  committed in `13da449`). This entry is written after seeing them and before seeds 1 and 2 have started.
- **Seeds 1 and 2:**
  - **Configs:** `configs/grpo_split_credit_seed{1,2}.yaml` are copies of `grpo_split_credit.yaml`. Only `seed`
    and `run_id` differ (a test checks this).
  - **What the seed changes:** `cfg["seed"]` drives the LoRA init, the TRL sampling seed and the EMA curriculum
    sampler. The training problem pool (`train_seed_offset: 0`) is unchanged, as is everything else.
  - **Run:** each goes to step 287 (permanent checkpoint), then gets the step-287 greedy eval on the frozen
    1.1.0 sets, then `queue_sanity.py`, then `phase95_compare.py` (per-seed family S/N).
  - **Gate:** seed 2 runs only if seed 1's `sanity.json` says PASS (your brief).
- **The seed-mean test (PREREGISTRATION.md family S: "a hypothesis is supported only if it holds on the seed-mean
  comparison").** The pre-registration does not say how the seed mean is tested, and McNemar has no direct
  seed-mean form, so this fixes it:
  - **Data:** per problem i (the pooled 400) and seed s ∈ {0, 1, 2}, the step-287 greedy outcomes. The controls
    are as pre-registered: SFT-only, and the single uniform-credit RL@287 run (only one uniform-credit seed
    exists).
  - **S1a / S1b:** statistic = mean over i of (c̄_i − c^ctrl_i), with c̄_i = mean over s of the 1σ-coverage
    indicator. One-sided paired bootstrap over problems: B = 10,000, seed 0; p = share of resamples ≤ 0.
  - **S2:** statistic = (mean over s of ρ_s) − ρ_uniform. ρ_s is Spearman ρ(stated relative σ, |relative
    error|) on seed s's answered problems, the same function as `phase95_compare.rho`. Each resample uses one
    problem index set for all arms. p = share of resamples ≤ 0.
  - **S3:** statistic = seed-mean accuracy (≤ 1%) − uniform accuracy. p = share of resamples ≤ −5 points, as in
    the per-seed test.
  - **Holm** over {S1a, S1b, S2, S3} at α = 0.05. A hypothesis is supported iff it is supported on this
    seed-mean test.
  - **Also reported, as pre-registered:** the per-seed tests (McNemar for S1, bootstrap for S2/S3), plus the
    range across seeds.
  - **Alternatives considered:**
    - McNemar on a majority vote (covered in ≥ 2 of 3 seeds): discards information.
    - McNemar on the 1,200 pooled problem-seed pairs: treats the same problem under 3 seeds as independent.
    - Stouffer or Fisher combination of the per-seed p-values: tests "some seed differs", not the seed mean.
  - **Why this one:** the paired bootstrap over problems is the pre-registration's general rule (§ general
    rules), and McNemar is its binary special case.
  - **Limitation:** inference is over problems, conditional on these three seeds. Seed-to-seed variance shows in
    the per-seed range but is not in the p-value.
  - **Code:** `scripts/phase95_seed_mean.py`, committed before any seed-1 eval exists.
- **Exploratory, no tests: greedy vs sampled σ at eval.**
  - **Why:** in training, split credit's sampled median σ settled near 1.3%, but its greedy eval σ is 0.24%.
  - **Models:** split-credit seed 0 @287, E1a final, and E1b final, on the same frozen eval problems / prefixes
    as their greedy evals.
  - **Sampling:** k = 4 samples per problem at the training settings (T = 1.0, top_p 1, top_k off, repetition
    penalty 1, torch seed 0).
  - **Outputs:** `compare/rl287_sampled` and `e1_eval/final_sampled`.
  - **Reported:** median relative σ, σ = 0 share, 1σ/2σ coverage, mean crps_log, ρ(σ, err) and the failure rate,
    greedy vs sampled. Also the within-problem spread of ln σ across the k samples, and where the greedy σ falls
    within its own samples. For split credit μ also varies between samples, so its sampled rows mix σ and answer
    changes; E1's μ is fixed.
  - **Code:** a `--samples K` option in `physcal.train.compare` and `physcal.train.e1 eval`. With K = 0 (the
    default) the greedy path is unchanged, and a test checks this.
- **Order and deadline:**
  - **Order:** seed 1 → (gate) → seed 2 → the three sampled evals.
  - **Deadline:** GPU work ends by 08:30 IST (2026-10-17).
  - **Waiting:** the queue waits on the previous queue's PID before it starts.
  - **Commits:** it commits and pushes after every stage itself (tapes gzipped), so commits do not depend on a
    watcher.

### D63. Split-credit seed 1 restarted from step 0 after the laptop shut down (no resume)
- **What happened:** the laptop shut down abruptly at about 23:30 IST on 2026-10-16. Seed 1 had started at
  21:06 and had logged 162 of 287 steps.
  - **Checkpoints:** the last complete one was `checkpoints/checkpoint-150` (23:24 IST).
  - **Tape:** it ended cleanly at step 162 (5,184 lines, all parseable).
  - **Damage check:** no stale `index.lock` and no running git process. `git fsck` is clean, local HEAD equals
    `origin/main` (`d371ca2`), and every tracked file matches HEAD. The SFT model, the seed-0 step-287 adapter
    and the E1a/E1b final adapters all load.
- **Your rule:** resume only if optimizer, scheduler, RNG state AND the curriculum/EMA sampler state are
  restored and the tape continues cleanly; otherwise restart from step 0.
- **Decision: restart from step 0.** Resuming fails the rule:
  1. `physcal/train/grpo.py` has no resume path (`trainer.train()` is called without `resume_from_checkpoint`).
  2. The checkpoint holds optimizer.pt, scheduler.pt and rng_state.pth, but not the `EMASampler` state.
     - The sampler's own numpy RNG (`default_rng(seed)`) is saved nowhere.
     - `curriculum.jsonl` logs the EMA pass rates rounded to 4 decimals, so the exact state cannot be rebuilt.
  3. A transformers resume would also replay the dataloader to skip the first 150 steps. Here that would call
     the curriculum's `next_batch` with no matching `observe`, which changes the sampler's path.
  4. The tape, `train_metrics`, `curriculum.jsonl` and `split_credit.jsonl` would need truncating to step 150.
- **The partial run was moved, not deleted:** to `results/discarded/split-credit-seed1-partial-crash-2026-10-16/`
  (git-ignored, about 375 MB), with its training log and the queue log at the crash. It is not a result and is
  not used anywhere. The restarted seed 1 writes a fresh run directory, so it can never append to the old tape.
- **Relaunch:** the same queue (`scripts/queue_split_seeds.sh`, D62).
  - **Order:** seed 1 from step 0 → eval → sanity → (gate) seed 2 → eval. Then the sampled-σ evals, only if
    there is time.
  - **Deadline:** GPU work ends by 08:30 IST.
  - **Changes:** each sampled eval now starts only if enough time is left (`SPLIT_SAMPLED_MIN_S` = 3,600 s,
    `E1_SAMPLED_MIN_S` = 2,100 s, about 1.3× the estimated duration). Otherwise it is logged as skipped and moves
    to tomorrow night (your brief: seeds come first). `LEFTOVERS=0` skips the stage-0 commit, which already
    happened.
- **Expected timing at seed 0's pace** (36 s/step + an 11-min eval): seed 1 ≈ 02:45, seed 2 ≈ 05:50, sampled
  evals ≈ 07:20.
- **Follow-up, not done tonight:** if mid-run resume is wanted in future, the sampler state (including its RNG
  state) has to be checkpointed alongside the adapter and replayed without dataloader skipping.

### D64. A6 slope label for a CI entirely below 0: "anti-propagation" (your decision, 2026-10-17; written before any A6/A6b model result exists)
- **Decision (yours, MORNING.md §6.3):** a 90% CI that lies entirely below 0 is labelled **"anti-propagation"**. This
  closes D61's open point. D61's other four labels are unchanged.
- **Status of the data:** no model has been evaluated on A6 or A6b. The only A6/A6b output so far is D60's
  tiny-random-model pipeline check, which is not a result.
- **The full rule, as implemented** (`physcal/eval/a6_pairs.py: slope_label`, on the primary slope's 90% pairs-bootstrap
  CI; endpoints count as contained):
  1. CI contains 0 → "no propagation";
  2. CI entirely below 0 → "anti-propagation";
  3. CI above 0 and contains 1 → "full propagation";
  4. CI inside (0, 1) → "partial propagation";
  5. CI entirely above 1 → "over-propagation".
- **CI level:** `configs/pairs.yaml: analysis.label_ci_level = 0.90` (D61). The A6 primary result's `ci` field is
  now the 90% CI; `ci95` is still written. Before this change the A6 `ci` field held the 95% CI, but nothing read
  it and no A6 result existed.
- **Joint verdict with labels (D61: "the labels in place of 'slope > 0'"; my implementation, for your review):**
  D59's three branches, with the A6 label substituted for the one-sided test:
  - A6 label is "no propagation" or "anti-propagation" → that label (D59's "A6 fails" branch; A6b is not
    consulted);
  - A6 label is full / partial / over, and A6b is not equivalent to 0 → "magnitude-driven";
  - A6 label is full / partial / over, and A6b is equivalent to 0 → the A6 label itself.
  `joint` writes this as `labelled_verdict`, next to the unchanged D59 `verdict`, the `A6_label`, and the label CI
  in the table.
- **Consistency with D59:** with a percentile bootstrap, a 90% CI whose lower end is above 0 is (up to ties) the same
  event as D58's one-sided p < 0.05. So "A6 fails" under D59 and "no / anti-propagation" under the labels agree on
  the same bootstrap draws.
- **One case the labels leave coarse (no change made; flagged):** a wide CI containing both 0 and 1 is labelled
  "no propagation" under rule 1, although such a result is really inconclusive. D61 says so as written, so it is
  implemented that way. The write-up should report the CI next to the label.
- **Tests:** `tests/test_pairs.py`:
  - `test_slope_labels`: the CI → label table, including a CI touching 0 and one spanning [0, 1].
  - `test_a6_labels_and_labelled_joint_verdict`: synthetic honest (slope 1), half-propagating (slope 0.5),
    lookup (0) and anti (−1) answers get the four expected labels, and the labelled joint verdicts are checked
    against relative-σ and absolute-σ A6b answers.

### D65. Exploratory: RL adapters evaluated on the 4-bit training base (written before any 4-bit eval exists)
- **Origin:** MORNING.md §6.1 (2026-10-17). RL LoRAs are trained on a 4-bit base (`physcal/train/grpo.py`,
  `e1.py`: Unsloth, `load_in_4bit=True`), but every RL eval so far merged them into the bf16 SFT model. Your decision
  (2026-10-17): run the check now, as an exploratory analysis. It changes no pre-registered result: every family S /
  N / E verdict stays on the bf16 evals, as pre-registered.
- **Question:** is the ~4× gap between training σ and eval σ (MORNING.md §5) a train/eval precision artifact?
- **What changes (one variable):** how the adapter is applied. The bf16 evals used a bf16 base with the LoRA merged.
  This check uses the training loader: a 4-bit base with the LoRA unmerged. Everything else is identical: the frozen
  eval problems / prefixes, prompt v4, the token budget (512 / 24), the decoding, and the eval tokenizer.
- **Loader (`model.loader: unsloth_4bit`, `physcal/eval/run_eval.py: load_unsloth_4bit`):**
  - `FastLanguageModel.from_pretrained(<adapter dir>, load_in_4bit=True, dtype=None, fast_inference=False)`, with
    `max_seq_length` and `text_only` taken from the run's own training config. Unsloth quantizes the base the adapter
    names and attaches the adapter unmerged. The loader checks that this base is the SFT model the bf16 eval used.
  - Generation then runs in Unsloth's inference mode. Unsloth's `generate` enters that mode itself, so training
    rollouts used it too.
  - **Prompts:** still built and tokenized by the eval's own tokenizer. The loader checks that Unsloth's tokenizer is
    identical (vocabulary, chat template, EOS id, backend JSON). In the smoke test it was identical in all four.
  - **Adapter check:** the loader checks that the adapter changes the logits (adapter on vs off on one prompt).
  - **Smoke test:** split-credit @287, 4 problems, greedy. It ran, with 1.6 GiB peak and about the bf16 speed.
- **Runs** (this order, one at a time, GPU temp > 87 °C → 10-min pause; `scripts/queue_precision_check.sh`):
  1. E1a final, greedy → `e1_eval/final_4bit`
  2. split-credit seed 0 @287, greedy → `compare/rl287_4bit`
  3. E1a final, k = 4 sampled → `e1_eval/final_4bit_sampled`
  4. split-credit seed 0 @287, k = 4 sampled → `compare/rl287_4bit_sampled`
  - **Sampling** is D62's: T = 1.0, top_p 1, top_k off, repetition penalty 1, torch seed 0.
  - **GPU time:** about 70–75 min at the bf16 durations (11 + 41 min split credit, about 5 + 13 min E1a). This is
    more than the 45 min in MORNING.md, which counted only split credit's greedy run and underestimated its sampled
    one.
- **Report (`scripts/precision_check.py` → `results/precision_check/report.{json,md}`; descriptive, no tests):**
  - **Per arm × decoding × precision**, as in D62: failure rate, median relative σ, σ = 0 share, 1σ/2σ coverage,
    mean crps_log, ρ(σ, err) and accuracy (split credit).
  - **Paired, greedy, 4-bit vs bf16:**
    - the share of byte-identical completions (σ suffixes for E1a);
    - for split credit, the share with the same μ;
    - the median of ln(σ_4bit / σ_bf16) on problems answered in both runs (σ floored at 1e-6).
  - **Training reference:** the median relative σ of training samples, overall and per depth, in the last 25 steps:
    split credit steps 263–287 and E1a steps 276–300. The per-depth table controls for the problem mix.
- **Reading rule, fixed before results:**
  - **The share of the gap closed** is g = ln(σ̃_4bit,sampled / σ̃_bf16,sampled) / ln(σ̃_train / σ̃_bf16,sampled),
    with σ̃ the median relative σ.
  - **Labels:** g ≥ 0.75 → "precision explains most of the gap"; 0.25 ≤ g < 0.75 → "partly"; g < 0.25 → "does not
    explain the gap".
  - g is computed per arm. E1a is the cleaner arm, because μ is fixed and only the σ suffix differs.
  - **If precision explains most of the gap,** every RL eval so far has understated the trained policy's σ. The
    pre-registered verdicts still stand as tests of the bf16-evaluated model, but the write-up must say which model
    was evaluated.

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

### D62. Split-credit seeds 1 and 2; the seed-mean S1–S3 test; exploratory greedy vs sampled σ (written before any seed-1/2 data or sampled eval exists) **[REVIEWED 2026-10-17: approved as written]**
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
- **Review outcome (yours, 2026-10-17, after the results):** D62 is approved as written. No alternative seed-mean
  test replaces it, so there is no deviation to record.
  - **Reporting rule for S2:** "narrowly supported; sensitive to the seed-mean definition", always with the per-seed
    p-values beside it (0.061 / 0.020 / 0.092). It holds on its own only on seed 1.
  - **Applied in:** FINDINGS #13, `results/split_credit_seed_mean/family_S_seed_mean.md` (the script now prints the
    per-seed p next to every seed-mean verdict) and PROGRESS.md.
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
  **Confirmed by you on 2026-10-17 (night), before any A6/A6b model result.**
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

### D66. Curriculum state checkpointed with the adapter; mid-run resume path (follows D63)
- **Origin:** your decision on MORNING.md §6.4 (2026-10-17): checkpoint the curriculum sampler state (including its
  RNG) with the adapter, add a resume path, and test it on CPU where possible. This is code only; no training was
  run.
- **What every rotating checkpoint now also holds** (`checkpoint-N/physcal_state.json`, curriculum runs; written
  atomically by an `on_save` callback right after the trainer's own files):
  - **The EMA sampler's exact state:** `EMASampler.checkpoint()`. It holds the full-precision pass rates, counts,
    unlocked cells, unlock log, observation count and the numpy RNG's bit-generator state. `state()`, the rounded
    per-step log, is unchanged.
  - **The repeat sampler:** the per-cell cursors and every chunk drawn so far (the dataset indices).
  - **The σ = 0 monitor's rolling window** and its alert flag.
  - **Run identity:** `max_steps` and a digest of the run's configs (the tape meta).
- **Finding while building it: the curriculum draws one step ahead.**
  - **What happens:** chunk k + 1, the prompts of step k + 1, is drawn *before* step k's results are observed. The
    trainer fetches all micro-batches of a step before generating, and accelerate prefetches one more batch.
  - **Checked on the logs of all three split-credit seeds:** an `EMASampler(seed)` fed the tape's group outcomes
    reproduces every one of the 287 logged chunks under this order, with 0 mismatches. Under the naive order
    (observe, then draw) 32 of 287 steps differ on seed 0.
  - **Checked in the real trainer** (CPU test below): checkpoint-N holds N + 1 drawn chunks.
  - **What it means:** in every curriculum run so far, the EMA weights used for a draw were one step stale. This is
    deterministic, identical across all arms, and changes no result. It is recorded here because the write-up
    describes the curriculum.
  - **Consequence for resume:** a checkpoint after step N holds a chunk that was drawn but not trained on. It must
    be replayed, not redrawn.
- **Resume** (`python -m physcal.train.grpo --config <cfg> --resume latest|<checkpoint dir>`, `setup_trainer` in
  `physcal/train/grpo.py`):
  1. **Checkpoint:** the newest complete checkpoint (`latest`), or the one named. Complete means the trainer state,
     optimizer, scheduler, RNG, adapter and `physcal_state.json`. Checkpoints written before D66 cannot be
     resumed, as in D63.
  2. **Refusals:** the step in `physcal_state.json` differs from the trainer's; `max_steps` differs; the run's
     configs differ from the saved digest or from `rollouts.meta.json`.
  3. **Log cut-back:** before any log is reopened, the tape, `train_metrics`, `curriculum`, `split_credit` and
     `memory` logs are cut back to steps ≤ N.
     - **Completeness:** every step 1..N must be present with the expected line count (32 tape lines, 1
       otherwise), and steps must never go backwards.
     - **Crash debris:** a partial last line from the crash is dropped. Any other unreadable line refuses the
       resume.
     - **Cut lines are kept** in `resume_discarded/`. `events.jsonl` gets a "resumed from" event.
  4. **Restore:** the EMA sampler, the σ = 0 monitor and the history are restored. The repeat sampler gets its
     cursors and chunks back and replays the drawn-but-untrained chunks before it draws new ones.
  5. **Training:** `trainer.train(resume_from_checkpoint=...)` restores LoRA, optimizer, scheduler and RNG, with
     `ignore_data_skip=True`. The trainer's own data skipping would call `next_batch` with no matching `observe`
     (D63 point 3).
  6. **Summary:** records `resumed_from` / `resumed_at_step`. `train_seconds` is the resumed segment, and
     `seconds_per_step` now divides by that segment's steps.
  - **Scope:** curriculum runs only. A run without a curriculum raises NotImplementedError. E1 (`e1.py`) has no
    curriculum and no resume path.
- **Refactor:** `setup_trainer()` is extracted from `train()`. It covers everything between the loaded model and
  `trainer.train()`, so the CPU test runs the real wiring with plain TRL.
  - **Fresh runs do not change:** `ignore_data_skip` is passed only when resuming, so the GRPOConfig is the one
    built before. The only addition is the `physcal_state.json` written into checkpoints.
  - **Existing tests:** the training, split-credit and E1 tests all pass.
- **What resume guarantees, and what it does not:**
  - **Guaranteed:** a resumed run is a valid continuation of the same process. The sampler, curriculum, optimizer
    and RNG are where an uninterrupted run would be, and given the same observations it makes the same choices.
  - **Not guaranteed:** bit-identical model outputs on GPU. D63 already saw drift from step 3 with identical
    code, so a resumed GPU run will diverge numerically from the uninterrupted one, as any rerun does.
- **Tests (`tests/test_resume.py`, CPU, about 60 s):**
  - **Samplers:** EMA and uniform checkpoints round-trip through JSON and continue identically. `restore` refuses
    another sampler's checkpoint.
  - **The real seed-0 logs:** the logged curriculum is reproduced under the one-step-ahead order. A checkpoint at
    step 150, restored, reproduces steps 151–287.
  - **Repeat sampler:** a restore replays the drawn chunks first.
  - **σ = 0 monitor:** its checkpoint round-trips.
  - **Helpers:** log cut-back (partial line, a missing step, backwards steps, unreadable lines), checkpoint
    selection (newest complete, pre-D66 refused), and refusal of a changed run.
  - **End to end (real TRL `GRPOTrainer`, its own checkpoint resume, a tiny random Qwen2 + LoRA, the real
    `setup_trainer`):**
    - 8 uninterrupted steps vs a "crash" with logs past step 4 and only checkpoint-4 left, resumed with D66;
    - **result:** identical curriculum log, tape problems and final saved state, and the resumed tape replays;
    - **negative control:** a naive resume (fresh curriculum, the trainer's data skipping, D63's failure) draws
      different cells and unlocks different depths, so the test can tell the two apart.
    - **Setup notes:** the model's completions are noise, so the binary outcome the curriculum sees is a fixed
      function of the problem seed. The CPU config uses a fast curriculum (α = 0.5, unlock after 1 group); with
      the real α = 0.1 the weights barely move in 4 steps, and the control could not separate the runs.
- **Not tested: the GPU path.** That is Unsloth's patched trainer, plus resuming a 4-bit PEFT checkpoint through it.
  Your brief allowed no other GPU work today. **Suggested check, about 5 GPU-min:** a 4-step smoke config (save
  every 2 steps). Kill it after step 3, resume with `--resume latest`, and compare `curriculum.jsonl` and
  `physcal_state.json` with an uninterrupted 4-step run.
- **Addendum to D65: post-hoc diagnostics (2026-10-17, 22:40–22:55; added after the 4-bit *greedy* results and
  before the 4-bit sampled split-credit result).** These were not in the plan above. They are exploratory, CPU only,
  and change no rule or verdict.
  - **Why:** the 4-bit greedy evals already stated a σ near the training σ. The open question was whether the
    cause is the base precision (4-bit vs bf16) or the *merge* the bf16 eval applies (`merge_and_unload`).
  - **Merge check** (`scripts/precision_merge_check.py`; CPU, bf16, E1a final adapter):
    - **Data:** the first 3 eval prefixes whose bf16 and 4-bit greedy σ suffixes differ.
    - **Unmerged:** the bf16 base with the adapter attached *unmerged* gives the 4-bit eval's suffix on 3 / 3.
    - **Merged:** after `merge_and_unload` the suffix changes on 2 / 3.
    - **Reading:** the evidence points to the merge, not the base. With n = 3 this is suggestive only.
  - **Merge rounding** (`scripts/precision_merge_rounding.py`; weights only, every adapter).
    - **Method:** per LoRA module, delta = scaling · B @ A in fp32, against PEFT's merge W ← bf16(W + bf16(delta)).
    - **Every RL adapter:** 91–96% of the update's entries round to exactly zero. The merged update's relative
      error is 0.89–0.95, and the median |delta| is 0.03–0.07 of a bf16 step on the base weight.
    - **The SFT adapter, for comparison:** 41% of entries round to zero, relative error 0.28.
    - **Reading:** the bf16 merge erases almost all of an RL update at this learning rate (5e-6) and step count.
  - **4-bit SFT control from existing data:** at training step 1 the LoRA's B is zero, so step-1 rollouts are
    samples of the plain 4-bit SFT model. Their median relative σ is 0.05–0.16% (split credit seeds, uniform credit;
    E1a 0.016%). The bf16 SFT greedy eval gives 0.063%. Quantization alone does not produce a ~1% σ. Caveat: these
    are training problems (depths 1–2) at T = 1, not the eval set.
  - **Consequence (for your review, nothing changed):** every RL *eval-set* result so far evaluated roughly "SFT +
    rounding noise", not the trained policy.
    - **Affected:** FINDINGS #3, #5, #6 (its eval tables), #13 (family S and N, all seeds and the uniform control)
      and #14 (family E). The pre-registered verdicts stand as tests of those merged models, as D65 says. As
      evidence about the RL-trained policies they are not valid.
    - **Not affected:** results from training rollouts, which are generated by the unmerged 4-bit policy. These
      include #6's σ = 0 collapse in training and the training-σ windows.
    - **Self-consistent:** SFT-only results are on the merged SFT model, which is also the start of every RL run.
      The SFT merge itself rounds away 41% of its entries (relative error 0.28), so "the SFT model" means the
      merged one throughout.

### D67. The bf16 merge was an implementation bug: corrected evals (4-bit training base, adapter unmerged) are the PRIMARY results for families S, N and E (your decision, 2026-10-18 00:00 IST; written before any corrected family S/N/E result)
- **Decision (yours):** merging RL adapters into the bf16 SFT model before eval was an implementation bug (FINDINGS
  #15).
  - **Primary results:** the pre-registered analyses of families S, N and E are re-run on corrected evals, and those
    are the primary results.
  - **Merged-eval results:** reported as **superseded**, side by side with the corrected ones. They are not deleted.
- **This is a deviation from the implemented eval path, not from PREREGISTRATION.md.** The hypotheses, tests,
  controls, thresholds, the Holm families and D62's seed-mean rule are unchanged. What changes is the model that is
  evaluated: the policy that was trained, instead of a merge that rounds away 91–96% of its update (FINDINGS #15).
- **What is already known at the time of writing:**
  - D65's descriptive numbers for split credit seed 0 @287 and E1a final on the 4-bit base, greedy and k = 4
    sampled (FINDINGS #15).
  - **Not computed yet:** any corrected family S, N or E test; and any corrected eval of split credit seeds 1 and 2,
    uniform credit, E1a step 150, E1b, or the SFT-only model on the 4-bit base.
- **The corrected eval:**
  - **Loader:** D65's `unsloth_4bit`. The base the adapter names, i.e. the merged SFT model, is quantized as at
    training start, and the LoRA is attached unmerged. `max_seq_length` and `text_only` come from each run's own
    training config.
  - **Everything else as in the superseded evals:** greedy decoding, prompt v4, the frozen 1.1.0 sets (200 ID + 200
    depth 5), a 512-token budget; for E1, the 400 frozen eval prefixes and 24 tokens.
- **Controls on the same 4-bit base:**
  - **SFT-only, full completions:** the merged SFT model on the frozen sets, no adapter (`compare --part sft`). This
    is family S's S1a control. Output: `overnight2…/compare/sft_4bit`, the matched uniform run's dir.
  - **SFT-only, σ suffixes on the E1 eval prefixes:** this is family E's "SFT stated σ" control. Output:
    `results/e1/sft_4bit/`. The prefixes are fixed data (SFT completions truncated after ±), so μ is unchanged.
  - **Uniform credit @287:** `overnight2…/compare/rl287_4bit`. This is the S1b / S2 / S3 control.
- **The baselines:**
  - **Family N:** the same fitted A5 strongest σ (`results/a5_baselines`, unchanged), applied to the corrected
    split-credit completions (`compare/rl287_4bit_a5_sigma.jsonl`).
  - **Family E:** the A5 σ on the fixed prefixes, unchanged.
  - **The standing post-hoc baseline** (one σ scale, c = 18.1, fitted on the calibration split's SFT completions):
    applied to the 4-bit SFT answers, reported and not tested, and labelled as not re-fitted.
  - **D49's family A:** already superseded by D50, so it is not re-run.
- **Which tape is primary when two exist:**
  - The split credit seed 0 @287 and E1a final greedy 4-bit evals already exist from D65, with the identical loader
    and settings, committed before D67. **The D65 tapes are the primary ones.**
  - Tonight's queue reruns both into `*_rerun` dirs as a determinism check. Any difference is reported. The choice
    is fixed here, before either rerun exists.
- **Tonight's queue (your order; GPU deadline 08:30 IST; temp > 87 °C → 10-min pause; commit + push after every
  stage):**
  1. **Greedy:**
     - SFT-only 4-bit (both controls above);
     - split credit seeds 0 (rerun), 1, 2 @287;
     - uniform credit @287;
     - E1a final (rerun) and step 150;
     - E1b final and step 150.
  2. **If time remains,** the adapters of FINDINGS #5 and #6:
     - #5 = the overnight run's `final_adapter` (step 287; that run ended at 287);
     - #6 = overnight2's `final_adapter` (step 759; its step-287 checkpoint is the uniform control in tier 1).
  3. **Then sampled k = 4** for split credit seed 0 and E1a, at the D62 settings. D65 already ran exactly these on
     the 4-bit base, so tonight's runs go to `*_sampled_rerun`. Both are reported, together with the within-problem
     σ spread ("σ instability").
  4. **Then the GPU check of `--resume`** (D66): a 4-step smoke config.
- **Analyses after the evals (CPU):**
  - `phase95_compare.py` per seed (families S and N) and `phase95_seed_mean.py` (family S, D62) on the corrected
    tapes and controls;
  - `e1 compare` (family E) with the 4-bit SFT σ control and the corrected E1 rows.
  - Code is identical apart from the input paths. Outputs go to new `…_corrected` files, and every superseded file
    is kept.
  - A side-by-side table (corrected vs superseded) goes in FINDINGS and MORNING.md.
- **Also decided (yours):** RL adapters are never merged. This becomes the eval default, with a guard (D68).

### D68. "Never merge RL adapters" is the eval default, with a guard (your decision, 2026-10-18)
- **Default loader** (`physcal/eval/run_eval.py: default_loader`): a model config with an adapter loads through
  `unsloth_4bit`, the 4-bit training base with the LoRA unmerged. One without an adapter loads through
  `transformers`, as before.
  - **`compare`:** defaults to `unsloth_4bit` for the sft and rl parts and to `transformers` for the raw base.
  - **`e1 eval`:** defaults to `unsloth_4bit`.
  - **Phase-4 `train.evaluate`:** passes the run's 4-bit loader settings.
  - **Loader settings:** `max_seq_length` and `text_only` always come from the run's own training config.
- **Guard** (`physcal/adapters.py: check_merge_allowed`): called before any merge, and before anything is loaded.
  - **An RL adapter is always refused.** Either of two independent marks makes an adapter RL: the nearest run dir's
    `rollouts.meta.json` (the adapter dir or up to 2 levels above it) holds a GRPO training config; or the adapter's
    model card names GRPO (TRL writes it, so a copied adapter is caught too).
  - **Any other adapter** is merged only on an explicit `merge_adapter: true` in the eval's model config.
  - **`physcal.train.merge`** (the tool that built the SFT model) also refuses RL adapters.
  - **The one deliberate merge left** is `scripts/precision_merge_check.py`, the FINDINGS #15 diagnostic, which
    merges on purpose to demonstrate the bug.
- **Also new:** `e1 eval --sft` gives family E's SFT-only σ control on the same base (D67), with no adapter, scored
  unbinned. The loader settings come from `grpo_e1a` (`configs/e1.yaml: sft_eval`).
- **Tests** (`tests/test_eval_loader.py`):
  - the defaults per part, with settings from the training config;
  - an adapter defaulting to the 4-bit loader;
  - the evidence rules on fake runs, model cards and real adapters (all real RL adapters are flagged, the SFT
    adapter is not);
  - the guard in the backend (an RL adapter is refused even with `merge_adapter: true`; an unrequested merge is
    refused; both before loading) and in the merge tool;
  - the E1 eval's meta, and the SFT-only E1 eval.
  - Two older tests that built RL configs without a run meta now pass the loader explicitly or build a fake meta.
- **Execution notes for D67 (2026-10-18 night).** These are not new decisions.
  - **Ran:** every stage of the queue ran, 00:17–02:20 IST, and none was skipped.
  - **Bug found and fixed in my analysis code (`20ec565`):** in `--corrected` mode the per-seed family S/N outputs
    are named after the run. The run names contain dots (`…v1.1.0…`), so `Path.with_suffix` collapsed all three
    seeds into one file. The seed-mean and family E outputs use a different path and were not affected. Seeds
    0–2 were rerun and a regression test added. Every number in FINDINGS #16 comes from the fixed outputs.
  - **The sampled reruns:** you asked for them as "re-measure σ instability". D65 had already run exactly these
    evals, so tonight's runs went to `*_rerun` dirs as D67 planned. They are byte-identical to D65's, and both are
    reported.


## Decisions on the 2026-10-18 morning report (2026-10-18, daytime; CPU only)

### D69. Both post-hoc baselines are re-fitted on 4-bit SFT completions of the calibration split (your decision, 2026-10-18; written before any 4-bit calibration completion exists)
- **Origin:** MORNING.md §5.3 (2026-10-18). Both baselines were fitted on completions of the bf16 merged SFT model:
  the standing post-hoc scale (×18.1, D40 / D44) and the A5 σ (D50/A5). Every corrected eval (D67) runs that SFT model
  quantized to 4 bits. So the fitted model and the evaluated model differ in one thing, the base precision.
- **GPU, the first stage of tonight's queue:**
  - **Model:** the SFT-only model through the corrected loader. This is D67's SFT control: the merged SFT model,
    quantized to 4 bits at load (`unsloth_4bit`), with no adapter.
  - **Settings:** greedy, prompt v4, a 512-token budget. These are the bf16 fits' settings.
  - **Problems:** the bf16 fits' calibration problems. That is `posthoc.yaml: calibration_split` (400; seeds
    20,000,000–20,000,399) and `probe_sft.yaml: calibration_ext_split` (1,600; seeds 20,000,400–20,001,999).
  - **Outputs:** `results/posthoc_calibration_4bit/sft_calibration_{set,ext}/`. Nothing that exists is overwritten.
- **CPU fits** (nice, n_jobs ≤ 4). The code is unchanged; only the input and output paths change.
  - **Global scale (D40):** c = argmax of mean crps_log on the 400-problem tape, plus the coverage-matched variant,
    exactly as in D40.
    - **Applied to:** the 4-bit SFT frozen eval (`overnight2…/compare/sft_4bit`).
    - **Sensitivity:** the same fit on all 2,000 completions (reported, not used).
  - **A5 (D50/A5):** `nested_baselines fit` on the two 4-bit tapes, with the same feature sets, grids, grouped CV
    and seed.
    - **Order:** `selection.json` (the strongest σ and p) is written before any eval use, as before.
    - **Eval tape:** the 4-bit SFT frozen eval. **Output:** `results/a5_baselines_4bit/`.
- **How the re-fits are used (fixed now, before any result):**
  - **From now on** (corrected evals, X2, Phase 12): the re-fitted global scale is the standing post-hoc baseline
    (D44), and the re-fitted A5 is "the A5 baseline".
  - **FINDINGS #16 (families N and E): the recorded verdicts stand,** because D67 fixed them on the bf16-fitted A5.
    - **Sensitivity rerun:** both families are rerun with the re-fitted A5, reported side by side.
    - **If a verdict changes,** MORNING.md reports it as a flipped verdict for your decision. Nothing is overwritten.
      **[REVIEW]**
  - **The bf16-fitted baselines:** they stay on disk and in FINDINGS #10, labelled "fitted on the bf16 merged SFT".
    The ×18.1 row in FINDINGS #16 keeps its "not re-fitted" label, and the re-fitted row goes next to it.

### D70. Phase 12 trains on the 4-bit base; new SFT is one LoRA adapter that RL keeps training; never merged (your decision, 2026-10-18)
- **Decision (yours):**
  - **Phase 12 RL** runs on the 4-bit base, so training and eval use the same base (D67 / D68).
  - **New SFT (Phase 12, R2-2):** one LoRA adapter on the 4-bit base. SFT trains it first, then RL continues
    training the same adapter. Nothing is ever merged.
- **The Unsloth check: supported, by construction.** This is a reading of the installed sources; nothing was run on
  the GPU today. Versions: unsloth / unsloth-zoo 2026.10.3, peft 0.21.2, trl 1.13.0, transformers 5.17.0,
  bitsandbytes 0.50.2.
  1. **Loading.** `FastLanguageModel.from_pretrained(<adapter dir>, load_in_4bit=True)` reads `adapter_config.json`
     and loads the 4-bit base it names.
     - **Attaching:** it attaches the adapter with `PeftModel.from_pretrained(..., is_trainable=True)`.
     - **Patching:** it then runs `patch_peft_model(model, use_gradient_checkpointing="unsloth")`. That gives the same
       LoRA kernels and gradient checkpointing a fresh adapter gets (`unsloth/models/loader.py`, about L1586–1645).
  2. **`get_peft_model` on that model.** It compares r, α, dropout, bias, target modules, rsLoRA, init and LoftQ
     with the stored config.
     - **If all match,** it logs "Already have LoRA adapters! We shall skip this step." and returns the model
       unchanged. It still re-arms the hooks the RL trainer needs.
     - **If any differs,** it raises TypeError (`llama.py`, about L3724–3840).
     - **For us:** `physcal/train/grpo.py` works unchanged with `model.id` set to the SFT adapter dir, and a LoRA
       config mismatch fails loudly. Our SFT and RL LoRA settings are identical (r 16, α 16, dropout 0, the same 7
       modules).
  3. **TRL `GRPOTrainer`.** A pre-wrapped PEFT model passed with no `peft_config` is trained as it is.
     - **β = 0 (every GRPO config so far):** there is no reference model.
     - **β ≠ 0:** TRL 1.13 copies the loaded adapter into a "ref" adapter (`grpo_trainer.py`, about L448), so the
       reference is the SFT policy. A plain `disable_adapter()` would instead give the pre-SFT base. We do not use
       β ≠ 0; this is noted for completeness.
  4. **dtype.** The SFT adapter and the RL adapters both store fp32 LoRA weights (392 tensors each). The continued
     adapter therefore trains in the same dtype as a fresh RL adapter.
- **Two findings from the check (relevant to Phase 12; no existing result changes):**
  1. **An adapter load does not pin the base revision.** For a PEFT load, Unsloth loads the base with
     `revision=None` (`model_revision = base_revision if not is_peft else None`), i.e. whatever `main` resolves to.
  2. **The existing SFT adapter was trained on Unsloth's pre-quantized mirror, not on the pinned Qwen revision.**
     - **What happened:** `configs/sft.yaml` asks for `Qwen/Qwen2.5-Math-1.5B-Instruct@aafeb0f` with
       `load_in_4bit=True`. Unsloth's name mapper substituted `unsloth/Qwen2.5-Math-1.5B-Instruct-bnb-4bit` and
       dropped the revision (`_revision_for_resolved_repo`: "Ignoring revision"). The adapter config names that
       mirror, and the cache holds one snapshot of it, `e24e953`.
     - **Then the merge:** `train.merge` added the adapter to bf16 `Qwen/…@aafeb0f`. So the SFT update was learned
       against the mirror's 4-bit weights and merged into the full-precision ones.
     - **Effect:** this is a second train/merge mismatch on top of FINDINGS #15's rounding. It changes no result.
       Every existing RL run starts from the merged model and is self-consistent (R2-2).
  - **Proposed for Phase 12** (to be built with the Phase 12 SFT, not now):
    - pin the 4-bit base explicitly, either as the mirror's commit (`use_exact_model_name` + revision) or as a local
      snapshot path;
    - assert the loaded base's commit in both trainers and in the eval loader. R2-1's provenance fields (D71) make
      it visible in every tape.
- **The plan (pre-registered with Phase 12):**
  - **SFT:** `physcal.train.sft` as now (4-bit base, LoRA r 16), saving `adapter/`. There is no merge step.
  - **RL:** `model.id` is that adapter dir, and `get_peft_model` skips (identical LoRA config). The optimizer and
    the LR schedule start fresh; β = 0.
  - **Eval:** the corrected loader with `lora` set to the RL adapter. The SFT-only control is the SFT adapter on the
    same base, so both arms are adapters on one base.
    - **Code to adapt:** `load_unsloth_4bit`'s base check compares local paths today. It must accept a hub id plus a
      revision.
- **Suggested GPU smoke test (about 5 min; not in tonight's queue).** Run SFT for 5 steps, then RL for 2 steps from
  that adapter. Check:
  1. the "skip this step" path is taken;
  2. at RL step 0 the LoRA weights equal the SFT adapter;
  3. they change after the RL steps;
  4. the eval loader passes the training-time fingerprint check (R2-1, D71).

## Round-2 review (REVIEW_SYNTHESIS_ROUND2_2026-10-17.md, committed with this entry)

### D71. Amendments R2-1 to R2-7 (recorded 2026-10-18, before any result they affect)
- **Source:** `REVIEW_SYNTHESIS_ROUND2_2026-10-17.md` (Gemini and ChatGPT). Its header says "R2-1 to R2-9", but it
  numbers seven amendments, R2-1 to R2-7. The experiments X1–X3 are not amendments; each is pre-registered in its
  own entry before its first result (X1: D73, X2: D74; X3 later).
- **State of the data when this is written:**
  - **Exists:** every corrected eval and analysis of D67 (FINDINGS #16).
  - **Does not exist yet:** the uniform-credit seeds 1 and 2; any 4-bit SFT calibration completion (D69); any X1 or
    X2 result; any training-time fingerprint (R2-1).
- **R2-1. Evaluation hygiene.**
  - **The test:** for a fixed set of prompts, the reloaded model must reproduce the training-time logits and the
    training-time σ distribution within tolerance. Any eval loader change must pass it.
  - **Provenance in every eval tape:** base revision, bitsandbytes version, compute dtype, quantization config and
    adapter hash.
  - **Implementation and tolerances:** D72. They are fixed before the first training-time fingerprint exists.
  - **First real use:** tonight's uniform-credit seeds 1 and 2.
- **R2-2. SFT pipeline revalidation.**
  - **Background:** merging the SFT adapter into bf16 rounded away 41% of its entries.
  - **Existing comparisons:** the merged SFT model is the start of every existing RL run, so these comparisons stay
    internally consistent. FINDINGS #4 / #5's SFT numbers describe an attenuated SFT. D70 adds a second mismatch:
    the update was learned on Unsloth's 4-bit mirror and merged into the bf16 Qwen weights.
  - **New SFT (Phase 12):** it is never merged into bf16. D70 chose the second option of R2-2: RL continues training
    the unmerged SFT adapter.
- **R2-3. What σ should and should not predict.**
  - **The rule:** under the A1 decomposition, σ on noise-free problems is not judged by its correlation with total
    error. That is the epistemic channel's job (p).
  - **Family N, reframed:** "a single σ asked to carry epistemic uncertainty fails to rank errors". This motivates
    two channels. It is not a claim that small models cannot express epistemic uncertainty; that claim is
    unsupported.
  - **Where it applies:** the write-up and every new FINDINGS text. FINDINGS #16's numbers do not change.
- **R2-4. Seed-level statistics.** With 3 vs 3 seeds, the smallest exact two-sided permutation p is 0.10 (one-sided:
  0.05).
  - **What is reported:** seed-wise effect sizes and matched-step trajectories, alongside the pre-registered
    problem-level tests.
  - **The wording rule:** pooled problem-level significance is never described as seed-level significance.
  - **The analysis:** fixed below, under R2-6.
- **R2-5. Publication criterion (stated now).**
  - **The positive claim** requires a reproducible, baseline-beating advantage on at least one *conditional,
    held-out* uncertainty task. Pooled coverage alone does not count.
  - **If that fails,** the controlled negative result is the paper.
- **R2-6. Uniform-credit seeds 1 and 2 (tonight's queue, D75).**
  - **Configs:** `configs/grpo_overnight2_seed{1,2}.yaml` are copies of `grpo_overnight2.yaml`. Only `seed` and
    `run_id` differ (a test checks this).
  - **Protocol:** the same SFT start, hyperparameters, problem pool and step 287 as the split-credit seeds. The LR is
    constant after warmup, so a run stopped at 287 matches the seed-0 control's step-287 checkpoint.
  - **Run:** to step 287 (permanent checkpoint), then the corrected eval (`compare --part rl`, `unsloth_4bit`,
    greedy, frozen 1.1.0 sets) and `queue_sanity.py`.
  - **Gate:** seed 2 starts only if seed 1 is sane, as in D62.
  - **One known difference from the seed-0 control:** these runs also write the training-time fingerprint after
    training ends (D72). It runs after the last optimizer step and after the adapter is saved, so it cannot change
    the trained adapter.
- **R2-6 / R2-4: the 3-vs-3 robustness analysis (fixed now; `scripts/robustness_3v3.py`, written before any seed-1/2
  data).**
  - **Status:** a robustness analysis reported next to the pre-registered verdicts. It does not replace them. The
    FINDINGS #16 verdicts (S1b / S2 / S3 against the single uniform seed) stay the pre-registered result.
  - **(A) Problem-level, both arms as seed means.** This is D62's construction with the uniform arm also averaged
    over its 3 seeds.
    - **S1b:** mean over problems of (c̄_i^split − c̄_i^uniform), where c̄_i is the mean over seeds of the 1σ-coverage
      indicator.
    - **S2:** mean over seeds of ρ_s (split) minus the same for uniform.
    - **S3:** seed-mean accuracy difference, with the −5-point margin.
    - **S1a** is unchanged (its control, SFT-only, has no seed).
    - **Inference:** a one-sided paired bootstrap over problems (B = 10,000, seed 0; one index set per resample
      for all arms), with Holm over {S1a, S1b, S2, S3} at α = 0.05, as in D62.
    - **Wording:** "conditional on these 3 + 3 seeds".
  - **(B) Seed-level, descriptive.** For each metric (1σ coverage, ρ(σ, err), accuracy, σ = 0 share, median
    relative σ), the per-run value for all 6 runs.
    - **Effect:** Δ = mean(split) − mean(uniform).
    - **Exact permutation test:** over the 20 ways to split the 6 runs 3 + 3. The one-sided p is the share of
      relabellings with Δ at least as large as observed (in the hypothesis's direction; minimum 0.05). The
      two-sided p is computed the same way on |Δ| (minimum 0.10).
    - **Also reported:** Hedges' g, and the 9 cross-seed differences split_i − uniform_j with the share that has
      the hypothesis's sign.
    - **No verdict is drawn from (B).**
  - **(C) Matched-step trajectories, descriptive.** These come from the training logs of all 6 runs (the
    training-tape σ is the unmerged 4-bit policy's own σ).
    - **Per 25-step window up to 287:** median relative σ of training samples, σ = 0 share, mean reward, binary
      pass rate, and the zero-variance group fraction.
    - **Output:** a table plus one figure.
  - **If only one new uniform seed reaches step 287:** the analysis runs as 3 vs 2 and says so (10 relabellings;
    minimum one-sided p 0.10).
- **R2-7. Citations to verify before use** (not verified yet):
  - Zeng, Xuan, Cui & Voigt, "Thinking Out Loud: Do Reasoning Models Know When They're Right?" (EMNLP 2025,
    arXiv 2504.06564);
  - Leng, Huang, Zhu & Huang, "Taming Overconfidence in LLMs: Reward Calibration in RLHF" (ICLR 2025);
  - Dettmers et al., QLoRA (arXiv 2305.14314);
  - Gneiting, Balabdaoui & Raftery (2007), "Probabilistic forecasts, calibration and sharpness", JRSS-B 69(2).
- **Also adopted from the synthesis's agreed points:**
  - **Merged results:** superseded and kept in a methodology / reproducibility section. They are never called a
    "control", because rounding is a structured perturbation, not controlled noise.
  - **Framing:** the world-versus-reasoning uncertainty decomposition is the headline. Credit assignment and the
    merge bug are supporting results.
  - **Distillation (X3):** secondary.

### D72. R2-1 implemented: the training-time fingerprint, the reloaded-model regression check, and provenance in every eval tape (tolerances fixed before any real fingerprint exists)
- **Code:** `physcal/eval/fingerprint.py` and `configs/fingerprint.yaml`; the tests are `tests/test_fingerprint.py`
  (CPU, tiny random Qwen2 + LoRA).
- **What a fingerprint holds.** The trainers record it right after the final adapter is saved and after the summary's
  memory stats are taken. They use the in-memory policy, put in Unsloth's inference mode as the eval loader does.
  It is written to `final_adapter/physcal_fingerprint.json` (`grpo.train`, `e1.train`).
  - **Fixed prompts:** 16 generator-1.1.0 problems, round-robin over the 12 training cells. They use training-split
    seeds 40,000,000+, a range no other set uses (X2 takes 30,000,000+, D74). Prompt v4.
  - **Teacher-forced log-probs:** for each prompt, every token of a fixed continuation. The continuation is the
    generator-written solution in the output format with σ = 2% (the SFT target format), scored with the adapter on
    and off.
    - **"Off"** measures the adapter's own effect: the mean |Δ log p| between on and off. It is the scale of the
      tolerances.
  - **The first σ token:** the next-token distribution there, as top-64 plus the rest's mass, adapter on and off.
  - **Sampled σ:** k = 4 completions per prompt at the training decoding (T = 1, top_p 1, top_k off, 512 tokens,
    torch seed 0), scored by the verifier for the stated relative σ.
  - **Provenance:** see below.
  - **Cost:** about 32 forward passes and 64 completions, a few GPU-minutes per run.
- **The check.** The eval's adapter loader (`load_unsloth_4bit`) runs `check_adapter` on every adapter.
  - **Lookup:** it finds the fingerprint for those exact weights, matched by a tensor-content hash. It looks in the
    adapter dir and in its run's `final_adapter`, so the step-287 permanent checkpoint of a run that ended at 287
    finds it.
  - **Measurement:** the reloaded model is measured with the same functions, so only the loader differs.
  - **Pass iff all four hold:**
    1. mean |Δ log p| (reloaded vs training) ≤ **0.25** × the adapter's own mean |Δ log p|;
    2. TV distance at the first σ token ≤ **0.25** × the adapter's own TV distance;
    3. |median ln σ_rel, reloaded − training| ≤ **ln 2** over the sampled completions. This is evaluated only when
       both sides parse ≥ 16 σ; otherwise the criterion is "not evaluable" and is ignored;
    4. the provenance fields `base_identity`, `quantization_config`, `compute_dtype` and `bitsandbytes` are equal.
       Other library versions are reported, not failed.
  - **Outcomes:** "pass" or "fail". **"Inconclusive"** if the adapter's own effect is below 0.005 nats/token, because
    then a loader that drops the adapter cannot be told apart. **"no_fingerprint"** for adapters trained before D72
    (every existing adapter).
  - **The thresholds and their reasoning:**
    - **A faithful reload** should reproduce the training-time numbers up to kernel differences. Training-mode and
      inference-mode Unsloth kernels in bf16 give a ratio near 0.
    - **FINDINGS #15's merge** erased > 90% of the update, which gives a ratio near 1.
    - **0.25** sits between the two. **ln 2** is about 3 standard errors of a median over 64 completions, and well
      below the merge's factor of ~5 in σ.
- **Enforcement:** `enforce: false` for now.
  - **Meaning:** a FAIL is recorded in the eval's `loader_check.fingerprint` (tape meta and metrics.json) and
    printed. It does not stop the eval.
  - **Why:** tonight's uniform-credit seeds 1 and 2 are the first real fingerprints. Their eval loader is D67's
    unchanged, already validated loader (byte-identical reruns, g ≈ 1). The check's own tolerances have not yet been
    seen on a real adapter.
  - **From then on:** any change to an eval loader must pass with `enforce: true` (R2-1). The tolerances are not
    tuned after seeing tonight's numbers without a new DECISIONS entry.
- **Provenance in every eval tape (R2-1).** `run_eval`'s meta and `metrics.json`, and the E1 eval's meta, now hold
  `provenance`:
  - `base_identity`: the content hash of a local base, or repo@commit for a hub model;
  - the quantization config and compute dtype (`bnb_4bit_compute_dtype`, the model dtype);
  - bitsandbytes / torch / transformers / peft / trl / accelerate / unsloth / unsloth-zoo versions;
  - the adapter's file hash and tensor hash, the base it names, and r / α / target modules;
  - the loader, and whether the adapter was merged.
- **CPU tests (`tests/test_fingerprint.py`, 8 tests):**
  - **A faithful reload passes.** This is the adapter attached unmerged to the same base: ratio 0, and the samples
    are byte-identical.
  - **A dropped adapter fails,** with ratio 1.0.
  - **The merge case.** The tiny base's weights are bf16-representable, as a fixed low-precision base is.
    - **An update below half a bf16 step** (B ~ N(0, 2e-5), like the RL updates): the bf16 merge rounds it away.
      Ratio 1.09, FAIL. With the configured effect floor the verdict is "inconclusive", which is also tested.
    - **A large update** (B ~ N(0, 0.05)) survives the merge (ratio 0.003). The check judges a loader by its
      outcome, not by the word "merge".
  - **Also tested:** lookup by tensor hash (a permanent checkpoint finds its run's final_adapter; other weights do
    not); the tensor hash ignores file metadata; the σ-distribution and provenance criteria on synthetic inputs
    (5× σ fails; a quantization-config change fails; a version change is reported only); `enforce` raises.
  - **The frozen-set eval test** now also asserts the provenance block in the tape meta.
- **Not tested: the GPU path.** That is Unsloth's inference mode for the teacher-forced forward and for generation,
  the 4-bit provenance fields, and the `UNSLOTH_RETURN_HIDDEN_STATES` guard.
  - **The guard:** GRPO sets this variable for its log-prob pass and does not always reset it. With it on, a forward
    returns hidden states instead of logits. The fingerprint code forces it off and checks the logits' shape.
  - **When it gets tested:** the first GPU use is tonight's two training runs and their evals.
  - **If recording fails:** `record_after_training` logs the error to `events.jsonl` and `summary.json` and never
    fails the run, since the adapter is already saved. The eval then reports "no_fingerprint".
- **Also changed:** `scripts/queue_sanity.py` takes an optional eval dir name (default `rl287`; the corrected evals
  use `rl287_4bit`).

### D73. Pre-registration of X1: a shallow text model of σ\* on the A6 / A6b pairs (written before any X1 model is fitted or any X1 prediction exists)
- **Origin:** REVIEW_SYNTHESIS_ROUND2 X1 (Gemini). CPU only.
- **Question:** can surface features of the problem text propagate measurement uncertainty? The answer sets the bar
  the LLM's σ must clear on A6 / A6b in Phase 12.
- **State of the data:** no model of any kind has been evaluated on A6 or A6b. The only A6/A6b output is D60's
  tiny-random-model pipeline check.
  - **One check before this entry:** the regex parser below was checked on 360 dev problems (seeds 52,000,000+,
    used by no set below). It recovered 100% of the shown measurements with the right symbols, converted 100% of
    the units, and gave values identical to the displayed SI values. That is a property of the parser, not an
    outcome.
- **Fit set (`configs/x1_text_sigma.yaml`):**
  - **Problems:** generator 2.0.0 (`env_v2`), split "train". That is the training families, depths 1–4 (depth 5 is
    held out of every training split, so X1 extrapolates to A6/A6b's depth 5, as any trained model would), and
    noise 1 / 2 / 5% (A6's levels).
  - **Size and order:** 12,024 problems, round-robin over 12 cells × 3 levels; seeds 50,000,000 + i.
  - **Test set (descriptive accuracy only):** 1,224 problems, seeds 51,000,000 + i.
  - **Overlap check:** the code refuses any seed overlap between fit, test and the pair sets.
  - **Target:** y = ln σ\*_log, the stored Monte Carlo σ\* in log space (A2).
- **Inputs: the problem text only.** No metadata and no generator field is used.
  - **TF-IDF:** whitespace tokens, A5's settings (unigrams, min_df 3, sublinear tf), the top 1,000 by document
    frequency.
  - **Numeric features parsed from the text:** every measurement "<symbol> = <number> <unit> ± <p>%" (or ",
    equals"), converted to SI with the project's pint registry.
    - **Statistics of p:** the count, max, mean and root-sum-square.
    - **Statistics of the magnitudes:** min, max, mean and SD of ln|SI value|.
    - **Per symbol:** ln|value| in a column for each symbol seen in ≥ 20 fit problems. A missing symbol is NaN,
      which the GBT handles natively.
  - **Exact values are not parsed.** These are the ones shown without ±, e.g. "g = 9.81 m/s^2".
- **Feature sets, nested:**
  1. noise (the p statistics only);
  2. noise_text (+ TF-IDF);
  3. **noise_text_magnitudes (+ the magnitude features): the PRIMARY.** Sets 1 and 2 are descriptive.
- **Model:** sklearn `HistGradientBoostingRegressor` (squared error), from A5's 4-point GBT grid.
  - **Selection:** per feature set, by 5-fold CV grouped by template (family | target | set of givens, as A5),
    minimizing CV MSE of ln σ\*. Then the model is refit on the whole fit set.
  - **Settings:** seed 0; at most 4 threads / workers; nice.
- **Primary analysis.** It uses the D58 / D59 / D61 / D64 pair analysis, unchanged (`a6_pairs.analyze_rows`,
  `joint`), on the frozen A6 (510 pairs) and A6b (510 pairs) sets, checked against their manifests' SHA-256.
  - **Stated σ:** X1's σ̂ = exp(ŷ) is each member's stated σ.
  - **Membership:** X1 states no answer, so every member counts as computation-correct and all 510 pairs enter
    each primary analysis. This is the one difference from how an LLM's answers enter it.
  - **Reported for the primary set:**
    - A6's slope, its 90% CI and D61/D64 label, and the one-sided test (D58);
    - A6b's slope, 90% CI and TOST equivalence (margin 0.3, D59);
    - D59's verdict and the labelled joint verdict (D64).
- **Pipeline references** (same analysis; sanity checks, not results):
  - **σ̂ = σ\* itself** must give slope 1 on A6 ("full propagation") and be equivalent to 0 on A6b ("propagates").
  - **The noise-only set** is constant within a pair, so it must give slope 0 ("no propagation").
- **Descriptive:**
  - for every feature set: test-set R², median |ŷ − y| and Spearman, and the same on the A6/A6b members;
  - within-pair corr(Δ ln σ̂, Δ ln σ\*), the share with the same sign, and A6b's mean Δ ln σ̂;
  - the primary analysis per depth.
- **Reading rules (fixed now).** These are bars for Phase 12, not hypotheses tested here. X1 is a single descriptive
  fit; no multiplicity correction applies.
  - **X1's labelled verdict is full or partial propagation (A6b equivalent to 0):**
    - **Reading:** surface features can propagate.
    - **The bar:** any claim that the LLM propagates beyond surface features needs its A6 slope to beat X1's on the
      same pairs (a paired bootstrap; specified in Phase 12's pre-registration).
  - **"Magnitude-driven":**
    - **Reading:** text features follow σ\* through the magnitudes, in a way that breaks invariance.
    - **The bar:** the LLM must pass both halves.
  - **"No propagation" or "anti-propagation":**
    - **Reading:** surface features do not propagate.
    - **The bar:** any LLM label of partial propagation or better, with A6b equivalent to 0, is beyond this
      baseline.
- **Code:** `physcal/eval/x1_text_sigma.py`, with tests of the parser, the features and the references in
  `tests/test_x1_text_sigma.py`. Outputs: `results/x1_text_sigma/{selection.json, results.json, summary.md}`.
  `selection.json` (CV table and parse coverage) is written before any prediction on the pairs.

### D75. Tonight's GPU queue, 2026-10-18/19 (prepared, not launched; `scripts/queue_uniform_seeds.sh`)
- **Order:** the GPU part of D69 first, as you asked, then R2-6.
  1. **GPU, 4-bit SFT-only calibration completions (D69).** `posthoc generate-4bit --part set` (400), then
     `--part ext` (1,600).
     - **Settings:** greedy, prompt v4, 512 tokens, batch 16, repetition penalty 1. These are the bf16 tapes' settings.
     - **Model:** the merged SFT model through `unsloth_4bit` with no adapter. `max_seq_length` and `text_only` come
       from the uniform-credit run's training config, exactly as D67's `compare/sft_4bit` control (checked: the
       same model config).
     - **Timing:** about 40 min at D67's 4-bit speed (1.1 s/problem).
  2. **CPU, in the background (nice, 4 threads):** `posthoc refit-4bit` (the D40 fit on the 400, the all-2,000
     sensitivity fit, the scaled eval tapes), then `nested_baselines fit --config nested_baselines_4bit` (A5).
     - **Code change:** the fit functions take paths or a config name. The defaults reproduce every existing number
       (checked: the bf16 re-run gives ×18.1425 exactly).
     - **Not in the queue:** D69's family N/E sensitivity reruns (CPU, tomorrow).
  3. **GPU, uniform-credit seed 1 (R2-6).**
     - **Training:** `grpo --config grpo_overnight2_seed1 --max-steps 287 --stop-at <deadline − 30 min>`. It writes
       the R2-1 fingerprint after the save (D72), and the queue copies it to the run dir for the commit.
     - **Eval:** `compare --part rl --adapter permanent_checkpoints/checkpoint-287 --out-name rl287_4bit` (the
       corrected loader, greedy, frozen 1.1.0 sets), which runs the fingerprint check. Then `queue_sanity.py …
       rl287_4bit`.
  4. **GPU, uniform-credit seed 2:** the same.
     - **Gate:** it starts only if seed 1 is sane (as D62) and at least 3.0 h of training time remain before the
       deadline minus the eval reserve. A run that cannot reach 287 is useless for the 3-vs-3 analysis.
  5. **CPU, after the re-fits finish:** `scripts/robustness_3v3.py` (D71 A/B/C) with whatever uniform seeds reached
     a corrected step-287 eval. With one missing, it runs 3 vs 2 (D71).
- **Guards (as in earlier queues):**
  - GPU deadline 08:30 IST;
  - GPU > 87 °C before a stage → 10-min pause (the trainer also pauses itself);
  - every stage is a child process waited on by PID;
  - commit + push after every stage under a flock, with training tapes gzipped.
- **Time budget:** about 0.7 h calibration + 2 × (2.9–3.2 h training + about 10 min eval + about 5 min fingerprint)
  ≈ 7.2 h.
  - **Launch time:** both seeds fit only if the queue starts by about **00:45 IST**. Started later, seed 2 is skipped
    by the 3.0-h gate and the analysis runs 3 vs 2.
- **Checked on CPU:**
  - `DRY_RUN=1` prints the plan above, and `bash -n` passes;
  - the seed configs equal `grpo_overnight2` apart from seed and run_id (test);
  - the calibration eval config equals D67's SFT control;
  - `robustness_3v3.py` runs end to end on the real corrected evals with a stand-in uniform arm (test). Its (A)
    reproduces D62's statistics when the uniform arm is the single control.
- **Not checked: any GPU stage.**
- **Launch command:**
  `nohup scripts/queue_uniform_seeds.sh > results/queue_uniform_seeds/nohup.log 2>&1 &`

### D74. Pre-registration of X2: within-problem correctness pairs (written before any X2 sample, selection or score exists)
- **Origin:** REVIEW_SYNTHESIS_ROUND2 X2 (ChatGPT), "the decisive epistemic test".
- **Question:** does any confidence source rank a *correct* answer above an *incorrect* answer **to the same
  problem**, with difficulty held fixed?
- **Today (CPU):** this entry; the frozen candidate pool; the selection rule and the metric (`physcal/eval/x2_pairs.py`,
  `configs/x2.yaml`, `tests/test_x2_pairs.py`). Everything else is GPU work for a later night (the schedule says
  Monday).
- **Candidate pool (frozen; `data/eval_sets/1.1.0+c735a0d4/x2_candidates.jsonl` + manifest, SHA-256):**
  - **Problems:** 1,200 generator-1.1.0 problems from the training split, 100 per training cell (round-robin).
  - **Seeds:** 30,000,000 + i. No other set uses them: GRPO uses 0–4,799, SFT 1e7+, calibration 2e7–2e7+1,999, the
    fingerprint prompts 4e7+, X1 5e7–5.2e7.
  - **Sizing:** the mixed rate at k = 8 was 21 / 48 = 44% for the bf16 SFT model (`model_probe_sft`), so about 530
    mixed problems are expected.
- **Sampling (GPU, later):** k = 8 per candidate from the corrected SFT model (D67's SFT control: the merged SFT model
  quantized to 4 bits, no adapter), at the training decoding.
  - **Settings:** T = 1, top_p 1, top_k off, 512 tokens, torch seed 0, prompt v4.
  - **Order:** the frozen order. If the night ends early, the sampled prefix (still stratified, since the order is
    round-robin) is used and its size reported.
  - **Cost:** about 9,600 completions, roughly 4 GPU-hours at the D62/D65 sampled speed.
- **Selection rule.**
  - **Answers:** an answer is a completion with a parsed μ in the right dimension. Parse failures and unit mismatches
    are not answers.
  - **Correct:** |relative error| ≤ 1% (the verifier's binary tolerance); otherwise incorrect.
  - **Kept:** a problem is kept iff it has ≥ 1 correct and ≥ 1 incorrect answer. Every (correct, incorrect) pair
    within a kept problem is formed.
- **Metric: pairwise accuracy.**
  - **Per problem:** the share of its pairs in which the correct answer gets more confidence. Ties, and a missing
    confidence on either side, count 0.5.
  - **Headline:** the mean over kept problems; each problem weighs the same, whatever its number of pairs.
  - **Inference:** a bootstrap over problems (B = 10,000, seed 0) gives the 95% CI and a one-sided p for
    "pairwise accuracy > 0.5".
  - **Holm** at α = 0.05 over the five primary methods.
- **Methods and their confidence.** All are scored on the same answers.
  1. **Stated σ, SFT (primary):** the σ in each sampled completion; confidence = −ln max(σ/|μ|, 1e-6).
  2. **Stated σ, split credit seed 0 @287 (primary).**
     - **The answer is held fixed:** each answer's SFT completion is truncated after "±". Split credit, loaded with
       the corrected loader, then writes a greedy σ suffix (24 tokens, the E1 eval machinery).
     - **Confidence:** as in 1. A suffix that does not parse or that changes μ counts as a tie.
  3. **A5, p channel (primary).** This is the A5 strongest p model, re-fitted on 4-bit SFT completions (D69), using
     problem + answer + reasoning of each sampled completion. Confidence = p(correct). **Secondary:** A5's strongest σ,
     confidence = −ln σ̂.
  4. **Problem-text-only baseline (a sanity check, not tested).** It is a function of the problem text alone (A5's
     "problem" p model), so it is identical within a problem and must give exactly 0.5. Anything else is a pipeline
     bug.
  5. **Answer-conditioned hidden-state probe (primary).** A logistic regression (L2) on the hidden state at the first
     token of "±" in `<answer>`. That position's input contains the complete numeric answer.
     - **Model:** the corrected SFT model (4-bit, as loaded by `unsloth_4bit`). If Unsloth's inference forward does
       not return hidden states, the same merged SFT is quantized by transformers + bitsandbytes with the same
       quantization config, recorded in provenance.
     - **Fit set:** separate training seeds, namely D69's 4-bit SFT greedy calibration completions (2,000, seeds
       2e7+), labelled correct within 1%. Completions without a parsed "±" are dropped.
     - **Fitting:** C from `probe_sft.yaml`'s grid, by 5-fold CV grouped by template. The layer is the one with the
       best CV AUROC on the fit set. Standardization is fitted on the fit set.
     - **Order:** all of this is fixed before any X2 answer is scored.
     - **Confidence:** the predicted P(correct).
  6. **Random-init control (primary; in the Holm family, so that its "success" is tested too):** the same
     architecture with random weights (seed 0), the same extraction, fitting and layer selection.
- **Secondary (reported, no verdict):**
  - A5 σ.
  - **A cross-fitted, answer-specific probe** (exploratory): 5-fold grouped by problem over the kept X2 problems. It is
    fitted on the sampled answers of 4/5 of the problems and scored on the held-out fifth, with the layer chosen by
    inner CV.
  - **Paired differences** with 95% bootstrap CIs over problems: probe − SFT σ, probe − random-init, A5-p − probe.
  - Per-depth pairwise accuracy; the number of pairs; the distribution of correct counts per kept problem.
- **Reading rules (from the synthesis, made concrete; fixed now).**
  - **Definitions:**
    - **"Probe succeeds":** the probe is Holm-supported and probe − random-init has a CI above 0.
    - **"Stated σ fails":** SFT stated σ is not Holm-supported.
  - **The outcomes:**
    - **The probe succeeds and stated σ fails:** the information exists, and the readout or training is deficient.
    - **Only A5 succeeds** (the probe fails, or does not beat random-init): the signal is in the text, not accessible
      to this hidden-state readout.
    - **Every primary method fails:** the earlier baseline advantages (FINDINGS #10, #16) were mostly difficulty
      prediction, not answer-specific awareness.
    - **Any other combination** is reported as observed.
- **Dependencies:**
  - **Fit data:** A5-p and the probe both use tonight's D69 4-bit calibration completions.
  - **GPU parts, in order:** sampling (~4 h); split-credit σ suffixes (~15 min); hidden states for the fit set and
    the X2 answers, for both models (~40 min).

## Night of 2026-10-18/19 (unattended; your instructions at launch, 22:47 IST)

### D76. Changes to tonight's queue at launch (your instructions; recorded before the queue starts)
- **Your instructions:** launch the queue now. A failed stage is logged BLOCKED, and the queue continues with the next
  stage that does not depend on it (the uniform seeds do not depend on the calibration completions or re-fits). If
  time remains after the 3-vs-3 analysis, run the ~5-min D70 smoke test. The usual rules apply (PID watchers, 87 °C
  pause, GPU done by 08:30 IST, commit + push after every stage, MORNING.md at the end).
- **Changes to D75 (`scripts/queue_uniform_seeds.sh`):**
  1. **Seed 2 no longer waits for seed 1's sanity check** (D75 took that gate from D62). Seed 2 does not depend on
     seed 1's outputs, so by your instruction it runs whenever a full run still fits: ≥ 3.0 h of training time
     before the deadline minus the eval reserve.
  2. **Failed stages are logged as BLOCKED** in the chain log. A failed calibration stage skips the D69 re-fits; the
     seeds go on.
  3. **Stage 6, the D70 smoke test** (`scripts/sft_continue_smoke.py`, `HF_HUB_OFFLINE=1`). It runs only if ≥ 25 min
     remain after stage 5, with a timeout of at most 30 min.
     - **Loading checks:** it loads the existing SFT adapter on its 4-bit base. The loaded model must be a PEFT
       model; its trainable tensors must be exactly the LoRA tensors; the loaded weights must equal the saved
       adapter; and `get_peft_model` with the RL LoRA config must take the skip path.
     - **Training checks:** it then runs 2 GRPO steps from that adapter with the real trainer (grpo_overnight2's
       config). Both steps must complete with a finite loss, and the final adapter must have the same tensors and
       base, with changed values.
     - **Status:** a capability check, not a result.
- **Hardening of the eval loader (`run_eval.py`):** the R2-1 check and the provenance block are new and untested on
  the GPU. An exception in either was fatal to the whole eval.
  - **Now:** while `enforce: false`, an exception in the check is recorded as `fingerprint.status = "error"` with its
    traceback, and the eval goes on. With `enforce: true` it raises, as before. A provenance error is recorded in
    `provenance.error`.
  - **Effect on outputs:** none on any eval output. Only a crash of the check itself is caught.

## Decisions on the 2026-10-19 morning report (2026-10-19, daytime; CPU only)

### D77. R2-1: below the log-prob floor the σ-token criterion decides (future fingerprints only); the check is enforced (your decisions, 2026-10-19)
- **Origin:** MORNING.md §4.1–4.2 (2026-10-19). Uniform-credit seed 1's reload was bit-exact, but its check is
  "inconclusive": the adapter's own log-prob effect (0.0047 nats/token) is under D72's floor of 0.005. Its σ-token
  effect (TV 0.018) discriminates clearly.
- **Decision 1 (yours): option (b), for future runs only.**
  - **The rule:** when the adapter's own log-prob effect is below `min_adapter_effect`, the σ-token criterion decides.
    - **Log-prob criterion:** reported, not decisive (`decisive: false` in the criteria).
    - **Still decisive:** the σ-token criterion (TV ratio ≤ 0.25), plus the two checks that do not depend on the
      adapter's size: provenance (`must_match`), and the sampled-σ median when evaluable.
    - **Status:** "pass" if all the decisive criteria pass, otherwise "fail".
    - **The degenerate case:** if the σ-token effect is also exactly 0, neither criterion can discriminate, and the
      status stays "inconclusive". This is not a new threshold.
  - **Not retroactive, by construction:** `compare` uses the config stored *in the fingerprint*.
    - **New fingerprints:** `configs/fingerprint.yaml` gains `below_floor: sigma_token`, so every fingerprint
      recorded from now on carries it.
    - **The two existing fingerprints** (uniform seeds 1 and 2) lack the key and keep D72's rule. Seed 1 stays
      "inconclusive" as recorded, in FINDINGS #18 and in its eval's `metrics.json`. Nothing is re-run or rewritten.
  - **Tolerances unchanged:** 0.25 / 0.25 / ln 2 and the 0.005 floor (D72).
- **Decision 2 (yours): R2-1 is enforced from now on** (`enforce: true`).
  - **"fail":** raises in the eval loader (`load_unsloth_4bit`), so the eval stops before generating anything.
  - **"inconclusive" and "no_fingerprint":** a WARNING on stderr and a `warning` field in the check record. The eval
    goes on. "no_fingerprint" covers every adapter trained before D72, e.g. split credit seed 0 in X2.
  - **An exception inside the check itself:** with `enforce: true` it raises, as D76 specified. A check that could not
    run has certified nothing, so it blocks the same way a "fail" does.
  - **Where `enforce` comes from:** the current config file, not the fingerprint.
- **Code:** `physcal/eval/fingerprint.py` (`compare`, `check_adapter`), `configs/fingerprint.yaml`, the comment in
  `run_eval.load_unsloth_4bit`.
- **Tests (`tests/test_fingerprint.py`):**
  - **On the tiny model**, with an update below the floor: a faithful reload passes by the σ-token criterion with the
    log-prob criterion non-decisive, the bf16 merge fails, and the pre-D77 config still gives "inconclusive".
  - **On synthetic inputs:** a log-prob "failure" (ratio 1) with an identical σ-token distribution passes under D77
    and fails under D72. A zero σ-token effect gives "inconclusive". A provenance mismatch still fails below the
    floor. An unknown rule raises.
  - **Enforcement:** enforced "fail" raises; "no_fingerprint" warns. 25 / 25 fingerprint and eval-loader tests pass.

### D78. The 4-bit base is pinned explicitly for all Phase 12 work (your decision, 2026-10-19; D70's proposal)
- **Origin:** D70's two findings.
  - **An adapter load does not pin the base revision:** Unsloth loads a PEFT adapter's base with `revision=None`,
    i.e. whatever `main` is.
  - **A hub-name load can drop the revision:** the existing SFT adapter asked for `Qwen/…@aafeb0f` but was trained
    on Unsloth's mirror, because Unsloth remapped the name and logged "Ignoring revision".
  - **Not affected:** existing results. Every existing RL run starts from the local merged SFT dir, which R2-1's
    provenance pins by content hash.
- **The pin (`configs/base_4bit.yaml`):**
  - **Base:** `unsloth/Qwen2.5-Math-1.5B-Instruct-bnb-4bit` at commit `e24e9530845f4e3a6648343796883cb509935039`.
    This is the only snapshot in the repo-local cache, and the one the existing SFT adapter was trained on (D70).
  - **Content hash:** `9939448b3a0fbe45`. This is `fingerprint._dir_content_hash` of the snapshot's weights, so it
    equals R2-1's `base_identity` in every Phase 12 eval tape.
  - **Quantization:** nf4, double quant, bf16 compute (the snapshot's own config.json).
- **How it is enforced (`physcal/base.py`):** no loader ever receives a hub name. A loader gets the **local
  snapshot dir** of the pinned commit, named repo-relative, the same convention every RL adapter uses for its base.
  - **Resolving:** `pinned_base_dir` resolves (repo, commit) with `huggingface_hub.snapshot_download`, offline by
    default. A missing snapshot raises; it never falls back to `main`.
  - **Before any load:** `verify_base_dir` checks the commit and the weights hash. `check_model_source` checks the
    base dir, or an adapter's `base_model_name_or_path`, against the pinned dir.
  - **After every load:** `assert_loaded_base` checks the loaded model's `_name_or_path`, or a PEFT model's base's.
- **Where it applies:** a model entry with `pinned_base: base_4bit`, which every Phase 12 config will carry.
  - **SFT** (`physcal.train.sft`): `id` absent means the pinned base itself. The saved SFT adapter then names the
    pinned dir as its base.
  - **RL** (`physcal.train.grpo`): `id` is the SFT adapter dir, which must name the pinned dir (D70: RL continues
    that adapter). The R2-1 fingerprint records the *base* dir's identity, not the adapter's weights.
  - **Eval** (`run_eval.HFBackend`): `id` must be the pinned dir, and any adapter must name it. Checked before
    loading, and again after. The verified record goes into the tape's provenance (`pinned_base`).
  - **A `revision` next to `pinned_base` is an error:** the pinned config is the only place a commit is written.
  - **Entries without `pinned_base`:** unchanged, so every existing config and run behaves exactly as before. The
    GRPO tape meta gains the key only when it is set, so a pre-D78 run's resume digest (D66) is unchanged.
- **Not Phase 12 (noted for consistency): X3's students (D79)** train on the RL runs' base, the local merged SFT dir,
  which is pinned by its content hash `0b435c0ebaed36e2` through R2-1 provenance, as in every existing run.
- **Tests (`tests/test_base_pin.py`, CPU, a fake hub cache):**
  - **Resolution and integrity:** the pinned dir is the commit's snapshot; tampered weights and a non-snapshot dir
    are refused; an uncached commit raises instead of resolving `main`.
  - **Adapters:** an adapter naming the pinned dir passes; one naming the hub name or another dir is refused.
  - **`resolve_model`:** legacy entries unchanged; `pinned_base` + `revision` refused.
  - **After loading:** the loaded-base assertion works for a plain model and a PEFT-like model.
  - **The eval backend** refuses an adapter on another base before loading anything.
  - **The real pin** matches the cached snapshot.
- **Not tested: the GPU path** (Unsloth loading a pre-quantized snapshot from a local dir, then saving and reloading
  an adapter that names it). Tonight's queue ends with a 2-minute GPU check of exactly that, when time allows (D81).

### D79. Pre-registration of X3: the distillation capacity test, pilot (written before any X3 teacher σ, student or score exists)
- **Origin:** REVIEW_SYNTHESIS_ROUND2 X3 (both reviewers; secondary). Your instruction (2026-10-19): pre-register X3, build
  the out-of-fold A5 teacher σ and the true-error-label control, configs; the pilot runs tonight only if ≥ 2 h of GPU
  time remain after X2 (D81).
- **Question:** can the RL runs' own setup express a σ that ranks errors, if it is *told* the target by SFT?
  - **The setup:** the 4-bit merged SFT base plus a fresh LoRA r = 16.
  - **If it can,** split credit's ρ ≈ 0.06 (FINDINGS #16 / #18) is an optimization problem (credit, reward signal),
    not a capacity limit.
  - **What X3 does not test:** self-awareness produced by RL. It tests capacity and transfer only (R2 synthesis).
- **State of the data:** the D69 4-bit calibration completions and the 4-bit A5 fit (`results/a5_baselines_4bit`)
  exist. No X3 teacher σ, label, student or score exists. X2 has not been sampled.
- **Students (one variable between them: the σ target):**
  - **The model:** exactly the RL runs' start and LoRA.
    - **Base:** the merged SFT model (`results/sft/qwen2.5-math-1.5b-instruct/merged`, content hash
      `0b435c0ebaed36e2`), loaded 4-bit by Unsloth (the training / corrected-eval loader).
    - **LoRA:** a fresh adapter with grpo_overnight2's settings (r 16, α 16, dropout 0, the same 7 modules, seed 0).
    - **Never merged (D68):** every eval applies the adapter unmerged, R2-1 enforced (D77). The trainer records the
      fingerprint.
  - **Training rows:** D69's 2,000 4-bit SFT greedy calibration completions (seeds 2e7..2e7+1,999). A row is kept if
    it has a parsed answer with μ > 0 (A5's `ok`) and a "±" in `<answer>`; the excluded count is reported.
    - **Context (no loss):** the prompt (v4) and the completion up to and including "±", as in E1. So the student
      never trains on μ or the reasoning.
    - **Target (loss):** the suffix after "±", with the σ number replaced by the target σ in the answer's own unit,
      written `.2g` (the SFT data's σ format), then the original unit, `</answer>` and the end-of-turn token.
    - **The σ in the target:** σ_display = σ_log · |μ_display|, so crps_log reads it back as σ_log (it scores
      N(ln μ, (σ/μ)²)).
  - **Student (a), the A5 teacher.** σ_log is the strongest 4-bit A5 σ (`problem_answer_reasoning/gbt`, its selected
    hyperparameters), predicted **out of fold**.
    - **The folds:** A5's own 5-fold CV, grouped by template.
    - **The scale:** c × exp(out-of-fold prediction), with c the value A5's selection fitted on those out-of-fold
      predictions (one global scalar).
    - **Why out of fold:** the refit A5 has seen these completions' own errors, so in-sample targets would leak them.
    - **Pipeline check (not a result):** these σ must reproduce A5's recorded CV metrics for that candidate
      (crps_log −0.1972, ρ 0.653, 1σ coverage 69.2%).
  - **Student (b), the true-error-label control.** σ_log is the σ that minimizes this completion's own crps_log given
    its realized error: σ_log = max(|ln(μ / truth)|, 1e-6) / √(ln 2). For one observation the CRPS of N(0, σ²) at
    e is minimized at |e| = σ·√(ln 2); 1e-6 is the verifier's crps_log floor.
    - **What it measures:** no teacher sits between the labels and the student. The student is fitted directly to
      the true errors, and on held-out problems it can only learn E[label | text].
- **Training (both students identical except the target):**
  - **Settings:** `configs/sft.yaml`'s optimizer settings (AdamW 8-bit, LR 1e-4 constant after 10 warmup steps,
    batch 2 × 4 accumulation, grad-norm 1.0, weight decay 0, bf16), seed 0, shuffled once with seed 0.
  - **Budget:** 2 epochs (about 500 optimizer steps).
  - **Time cap:** at launch the queue gives (a) a wall-clock cap, so that both students and their evals fit before
    the deadline. Student (b) then gets exactly (a)'s completed step count, so the budget stays matched.
  - **Reported:** steps, loss on the target tokens, time.
- **Evaluation (greedy σ suffix, 24 tokens, the E1 eval machinery; μ fixed):**
  1. **ID + depth 5 (primary):** the frozen 1.1.0 sets' 400 4-bit SFT greedy completions (`overnight2…/compare/
     sft_4bit`), truncated after "±".
     - **These answers already have** the 4-bit A5 σ (ρ 0.68), the SFT's own σ (ρ 0.05), and the 4-bit SFT corrected
       eval rows of FINDINGS #16.
     - **Failures:** parse failure, unit mismatch, or a suffix that changes μ (E1's `mu_changed`). These score the
       floor, count as not covered, and are excluded from ρ, as in E1.
  2. **X2 pairs (secondary):** every answer of the kept X2 problems (D74) truncated after "±". Pairwise accuracy
     with confidence −ln σ_rel, as X2's method 2; an unusable suffix is a tie.
- **Metrics:**
  - ρ = Spearman(σ, |ln(μ / truth)|) on the pooled 400 (answered, μ unchanged);
  - mean crps_log; 1σ / 2σ coverage; median relative σ; σ = 0 share; the parse / unit / μ-changed rates;
  - fidelity: Spearman(student σ, teacher σ) on the same answers;
  - X2 pairwise accuracy.
  - **Inference:** bootstrap over problems (B = 10,000, seed 0) for the 95% CIs; paired bootstrap for differences.
- **Reading rules (fixed now; a pilot, so labels, not hypothesis tests):**
  - **"Capacity shown":** student (a)'s ρ CI lies above 0.5 (the synthesis's bar). RL's ρ ≈ 0.06 is then an
    optimization problem, not a capacity limit.
  - **"Capacity likely":** the point estimate is ≥ 0.5, but the CI reaches below 0.5.
  - **"Not shown at the pilot budget":** the point estimate is < 0.5. Read with the fidelity: low fidelity means the
    student did not learn the teacher, which is an SFT-budget question, not a capacity verdict.
  - **Student (b), descriptive:** ρ_b − ρ_a with its paired CI. Both are read against the teacher's own ρ on the
    same answers (0.68) and SFT's (0.05).
  - **X2 pairwise accuracy, descriptive:** reported next to X2's methods. A student above 0.5 there would carry
    answer-specific information that A5's text features do or do not have.
  - **What comes next:** a full X3 (3 seeds, a longer budget) is specified separately, if the pilot warrants it.
- **Code and outputs:**
  - **Code:** `physcal/train/x3.py`; config `configs/x3.yaml`; tests `tests/test_x3.py`.
  - **Data:** `results/x3/data/` (rows, targets, the pipeline check). It is built today on CPU and committed before
    any student is trained.

### D80. Phase 12 SFT data: three arms on the same noisy problems (PREP ONLY; not pre-registered; several items [REVIEW])
- **Your instruction (2026-10-19):**
  - **The arms:** SFT data for noisy problems in three arms (implicit σ, an explicit propagation line, a supervised σ\*
    baseline), with p in discrete bins, plus tests.
  - **Not yet:** pre-registration. Phase 12 is pre-registered after X2 reports.
  - **Status of these choices:** nothing here is fixed for Phase 12. The [REVIEW] items are open until the
    pre-registration.
- **Code and outputs:**
  - **Code:** `physcal/train/sft_phase12.py`, `configs/sft_phase12.yaml`, `tests/test_sft_phase12.py` (7 tests).
  - **Outputs:** `results/phase12_sft_data/`.
    - **Committed:** `manifest.json` (counts, sha256 of every file) and `samples.md` (8 problems × 3 arms).
    - **Not committed:** the jsonl files, about 15 MB, rebuilt deterministically by
      `python -m physcal.train.sft_phase12 build`.
- **Problems (the same in every arm):**
  - **Source:** generator 2.0.0 (`env_v2`), training split.
  - **Count:** 3,000, round-robin over the 12 training cells × noise {0, 1, 2, 5}%. 5% is D54's cap; 10% stays the
    heavy-tail regime. That gives 62–63 per cell × level.
  - **Seeds:** 60,000,000–60,002,999, used by no other set.
  - **The answer is f(x_meas)**, the generator's point estimate from the displayed values (4 s.f., the old SFT data's
    format). It is never the noise-free truth.
    - **How:** `sft_data.solution` is reused unchanged on a copy of the problem whose target is the point estimate,
      so every chain line is computed from the displayed values.
- **Format (two channels, per the round-2 synthesis):**
  - **Layout:** `<answer>μ ± σ unit</answer>`, then on the next line `<confidence>c%</confidence>`.
  - **[REVIEW] Why a separate block after `</answer>`:**
    - **The problem with ", confidence p" inside `<answer>`:** the current verifier would parse it as part of the unit
      and fail the answer.
    - **What a separate block keeps:** the verifier's μ / σ parsing and scoring exactly as they are; the existing
      verifier ignores text after `</answer>` (tested: every arm's completion scores "ok").
    - **Still needed:** a verifier extension for c, i.e. its parser and a Brier score on A1's computation-correct
      event. It is additive and needs your approval with the Phase 12 pre-registration.
    - **The parser:** `parse_confidence` is the reference for that extension. It requires exactly one block after
      `</answer>` with a bin value, otherwise None.
  - **[REVIEW] Prompt `v5_two_channel` (draft, `physcal/prompt.py`).** It is v4 with:
    - the `<confidence>` block in the format;
    - σ defined as the 1-SD uncertainty caused by the givens' measurement uncertainties (0 if all exact);
    - c defined as the probability that the calculation is correct for the values as stated, one of the bins;
    - "exactly one <answer> block, then exactly one <confidence> block";
    - the worked example's answer as `± 0 km` (its givens are exact) with `<confidence>90%</confidence>`.
    - **Unchanged:** the existing prompts (their hashes are tested). The prompt is applied at tokenization, so changing
      it needs no data rebuild.
- **Confidence channel ("p in discrete bins"):**
  - **Bins:** c ∈ {10, 20, …, 90, 95, 99}%.
  - **[REVIEW] The draw:** c is drawn uniformly from the bins per problem (its own seeded stream), independent of the
    problem, and the same in every arm. Its tokens are trained.
    - **What it teaches:** the format and the allowed values, with a flat prior and no confidence signal for RL
      (Brier, later) to unlearn.
    - **Realized counts:** 258–295 per bin.
  - **Alternatives:** mask c (the model may then emit non-bin values), or a fixed c (a strong prior).
- **σ channel, per arm:**
  - **implicit:** σ\* (the CRPS-optimal Monte Carlo σ\*, A2) written after "±" and **masked** from the loss, as D35
    masked σ. RL learns σ.
  - **explicit:** one line at the end of `<steps>`: `sigma_rel = sqrt((a1*s1)^2 + …) = X`.
    - **The terms:** a_i = |d ln f / d ln x_i|, the chain's log-sensitivities at the displayed values, written `.2g`;
      s_i is the stated level.
    - **Order:** the givens in the order of the solution's "SI values" line.
    - **X:** computed from the displayed terms, so the line is self-consistent, and written `.2g`. The answer's σ is
      X·|μ|.
    - **Exact givens:** the line is `sigma_rel = 0`.
    - **How X compares:** X agrees with the generator's exact first-order σ (its `sigma_star_log_delta`, tested) up to
      rounding. Against σ\*, the median |ln(X / σ\*)| on noisy problems is 0.013.
    - **The verifier:** it reads the line as an unrequested step, so it has no effect on step credit.
    - **[REVIEW] `train_line_numbers`, default true:** SFT trains the line's numbers and the answer's σ, i.e. it
      teaches first-order propagation. False masks those numbers (the `^2` exponents stay trained): format only, and
      RL must find the sensitivities. The choice decides what "explicit + RL" can add over its SFT-only control.
    - **[REVIEW] The line names no symbols** (as the synthesis's example). Naming them would make it more learnable.
  - **supervised:** σ\* written and **trained**; no RL.
    - **[REVIEW] The loss:** cross-entropy on a 2-significant-figure σ only approximates the synthesis's "log-scale
      loss".
    - **Its text is identical to the implicit arm's;** only the mask differs.
- **Exact givens (noise 0):** σ = 0 (`± 0 unit`) in every arm. Under A1 / R2-3 the aleatoric channel is 0 when every
  given is exact, and epistemic doubt belongs in c.
- **Tests (7):**
  - **Every arm's completion:** scores "ok" with the unchanged verifier, with μ = f(x_meas) (never the truth when
    noisy) and σ / μ read back as intended.
  - **The arms:** implicit and supervised texts are identical; explicit is that text plus the line, with its own σ.
  - **The confidence:** the same in every arm, the span checks, and the parser refuses malformed blocks.
  - **The line:** matches the generator's first-order σ and parses as a step.
  - **Masking per arm:** checked with the real tokenizer, including the format-only and masked-c variants.
  - **Data integrity:** the manifest (seeds, no held-out depth / family, noise levels, sha256), and the existing
    prompts unchanged.
- **Open for the Phase 12 pre-registration (your decisions):**
  - the `<confidence>` block and the v5 prompt;
  - uniform trained c;
  - explicit-arm supervision (`train_line_numbers`) and symbol names;
  - whether the supervised arm's token loss is acceptable as the "log-scale" baseline;
  - n = 3,000 and the noise mix (25% exact problems).

### D81. Tonight's GPU queue, 2026-10-19/20: X2, then the X3 pilot (prepared, NOT launched; `scripts/queue_x2_x3.sh`)
- **Your instructions (2026-10-19):** prepare, do not launch.
  - **X2 sampling** on all 1,200 candidates, at the largest batch that fits;
  - **hidden-state extraction** for the X2 probe;
  - **the X2 analysis** on CPU;
  - **then the X3 distillation pilot,** only if ≥ 2 h remain;
  - **deadline** 08:30 IST.
- **Stages:**
  1. **Batch-size probe** (`scripts/x2_batch_probe.py`).
     - **The model:** X2's exact model and decoding.
     - **The test:** the pool's longest prompts, every sequence forced to the full 512 new tokens (the worst case).
     - **Choice:** candidate sizes 128 / 96 / 64 / 48 / 32 / 16 sequences per call, largest first; the first that runs
       without OOM is the night's size.
     - **Fallback:** D74's 16, if the probe fails.
  2. **X2 sampling:** 6 chunks of 200 candidates in the frozen order. Each chunk is its own committed tape.
     - **Start gate:** a chunk starts only if its worst-case time (the probe's, × 1.3) still leaves 1 h for the
       downstream X2 stages.
     - **If the night ends early,** the sampled prefix is used, as D74 allows (it is round-robin, so stratified).
  3. **Selection** (CPU, seconds).
  4. **Method 2:** split credit seed 0's σ suffixes on the kept answers.
  5. **Hidden states:** SFT and random init × fit set and X2 answers.
  6. **The X2 analysis** (CPU), in the background while the GPU goes on.
  7. **The X3 pilot (D79),** if ≥ 2 h remain after stage 5.
  8. **The D78 pinned-base GPU check** (about 2 min), if ≥ 10 min remain.
- **Guards** (as every queue so far):
  - every stage is a child waited on by PID;
  - GPU > 87 °C before a stage → a 10-minute pause;
  - commit + push after every stage, under a flock;
  - a failed stage is logged BLOCKED, and the stages that do not depend on it go on;
  - `HF_HUB_OFFLINE=1`.
- **How this implements D74 (stated now, before any X2 sample exists):**
  1. **Batch size:** the probe's choice replaces D74's 16, as you asked, and is recorded in every chunk's tape.
     - **Effect:** a different batch draws different random numbers. Every size gives valid samples at the same
       decoding.
  2. **Chunks:** chunk i uses torch seed 0 + i (D74: seed 0); each chunk's call re-seeds.
     - **The prefix check:** `x2_pairs.sampled_chunks` refuses chunks that are not a contiguous prefix of the pool.
       `sampled_records` checks that the sampled seeds are the pool's first candidates.
  3. **Memory cap:** the sampling run caps the allocator at 7.5 GiB, the training runs' value. Past it, torch raises
     OOM; the eval's backoff then halves the batch and retries, instead of spilling into WSL shared memory, which
     only slows. Backoffs are counted in the tape.
  4. **Method 2 is in the queue** although your list did not name it. D74 makes it a primary method, and it costs
     about 15 minutes.
     - **The adapter:** `permanent_checkpoints/checkpoint-287`, the corrected eval's.
     - **R2-1:** it was trained before D72 and has no fingerprint, so the check warns ("no_fingerprint") and does not
       block (D77).
  5. **Hidden states:**
     - **The position:** the token containing the first character of the "±" in `<answer>`. Qwen's tokenizer merges
       it with the preceding space (" ±"); checked on the calibration completions.
     - **The SFT path:** the merged SFT dir loaded 4-bit by Unsloth.
       - **A check on the first sequence:** lm_head of the last returned state, as returned or after the final norm,
         must reproduce the model's own logits within 2% of the largest logit.
       - **Otherwise:** D74's fallback, transformers + bitsandbytes with the same quantization config, recorded in
         `info.json`.
     - **The random-init control:** built today on CPU (seed 0, bf16, `results/x2/random_init_model`, content hash
       `13ededef3f3acbd4`, not committed). It is loaded exactly the same way, so "the same extraction" includes the
       4-bit quantization.
  6. **Method 4's model:** D74 named A5's "problem" p model without its kind. It is `problem/gbt`, the lower CV Brier
     of the two (0.1544 vs 0.1648, `results/a5_baselines_4bit/selection.json`).
  7. **The probe:**
     - **Inside the CV folds:** standardization is fitted on each training fold. The final scaler is fitted on the
       whole fit set (D74).
     - **Selection:** layer and C by CV AUROC; ties go to the earlier layer, then the smaller C.
     - **The cross-fitted secondary:** inner CV (grouped by problem) chooses both layer and C from the same grid
       (D74 named the layer only).
  8. **Missing inputs:** a primary method whose inputs are missing counts as p = 1 in Holm (as E1's compare does),
     so the correction is never weaker than pre-registered.
  9. **Reading rules:** "every primary method fails" means methods 1, 2, 3 and 5 are all unsupported. Method 6 is the
     control.
- **The X3 pilot in the queue (D79):**
  - **Student (a)'s cap:** (time left − 50 min for evals − 2 × 15 min slack) / 2. Two epochs (about 500 steps at
    about 5.5 s/step, about 46 min) fit well inside it if X2 ends on time.
  - **Student (b):** `--max-steps` = (a)'s completed steps.
  - **Each student's evals:** the frozen sets and the X2 answers. The frozen sets give 399 prefixes: one of the 400
    4-bit SFT answers has no "±", so that problem drops out of every arm alike.
  - **The analysis** runs after the X2 analysis has finished.
  - **R2-1 enforced (D77):** each student records a fingerprint, so a failing reload blocks its eval.
- **Time budget (estimates; nothing here has run on the GPU):**
  - **Fixed parts:** probe ≤ 10 min; downstream X2 about 45 min; X3 about 2 h.
  - **Sampling:** about 3.3 h at batch 16 (D62/D65's 1.2 s per completion). With a larger batch it is shorter, by an
    amount the probe will measure.
  - **Launch time:** launched by about 23:00, everything fits.
- **Checked on CPU:**
  - `bash -n`; `DRY_RUN=1` prints the plan above.
  - **New tests:**
    - `tests/test_x2_night.py` (6): the chunk-prefix rules; the per-chunk sample config; the whole D74 analysis end to
      end on synthetic chunks, suffixes and hidden states with the real 4-bit A5 models; missing inputs count as
      p = 1; the reading rules; the hidden-state helpers on a tiny CPU model (positions, states, logit check).
    - `tests/test_x3.py` (10).
  - **Inputs present:** the X3 data and the random-init weights; the X3 eval rows build (399).
- **Not checked: any GPU stage.** That covers:
  - the probe;
  - `min_new_tokens` under Unsloth's generate;
  - Unsloth returning every layer's states;
  - X3 training and its fingerprint;
  - the D78 snapshot load.
- **Launch command:** `nohup scripts/queue_x2_x3.sh > results/queue_x2_x3/nohup.log 2>&1 &`

## Decisions on the 2026-10-19 day report (2026-10-19, afternoon; CPU only)

### D82. X3's control and X2's method-2 stage confirmed (your decisions, 2026-10-19)
- **X3 control (D79):** each answer's own CRPS-optimal σ, |ln(μ/truth)| / √(ln 2), floored at 1e-6, is **confirmed**
  as the reading of "a σ fitted directly to true error labels". The pre-registration and the built data stand
  unchanged.
- **X2 method 2 in tonight's queue (D81):** the split credit seed 0 @287 σ-suffix stage is **approved**. The queue is
  unchanged.

### D83. Verifier 1.2.0: the `<confidence>` channel, Brier on the computation-correct event (your approval, 2026-10-19; no Phase 12 training yet)
- **Your decision:** approve the verifier extension of D80.
  - **The score:** Brier on the computation-correct event.
  - **The version:** a new verifier version, under which old tapes must replay exactly.
  - **Scope:** implement and test, with no Phase 12 training yet.
- **Where it lives:**
  - **Config:** `configs/reward_v12.yaml`, version 1.2.0: 1.1.0's settings plus a `confidence` section and
    `weights.confidence`.
  - **The default is unchanged:** `configs/reward.yaml` stays 1.1.0 and the default of every code path. Every
    existing run and tonight's queue are therefore untouched. Phase 12 code will load
    `reward_config("reward_v12")`.
  - **Code:** `physcal/verify/reward.py` (`confidence_terms`, the validation); `parse_confidence` moved from the
    Phase 12 data module into `physcal/verify/parse.py`.
- **The rule:**
  - **Format:** exactly one `<confidence>c%</confidence>` block after `</answer>`, with c one of {10, 20, …, 90, 95,
    99}.
  - **The event (D50/A1):** computation-correct iff the answer parses, its unit has the target's dimension, and
    |ln μ − ln f(x_meas)| ≤ τ.
    - **f(x_meas):** the generator's point estimate from the displayed values (`noise.point_estimate_si`). For
      problems with exact givens (generator 1.x, noise 0) it is the truth; there the two coincide.
  - **score_confidence = −(c/100 − 1{event})²,** the negative Brier score: strictly proper over the allowed values
    (tested).
  - **The worst Brier score, −1,** goes to: a missing, malformed or non-bin block, two blocks, or a block before
    `</answer>` (`confidence_reason: confidence_missing`); and a completion whose answer does not parse
    (`answer_parse_fail`).
  - **A unit mismatch** is scored normally, with the event false.
  - **The answer's own scoring is unchanged:** score_final is against the realized truth, and the steps are as in
    1.1.0.
  - **Reward:** w_final·score_final + w_steps·score_steps + w_confidence·score_confidence.
  - **Validation:**
    - steps + confidence weights (× the Brier span 1) must stay below the final answer's span;
    - the section and its weight must exist iff version ≥ 1.2.0;
    - bins must lie strictly between 0 and 100, and τ > 0.
- **Two values set here, both [REVIEW] for the Phase 12 pre-registration / Round 3:**
  1. **`weights.confidence = 0`.**
     - **Effect:** the reward value equals 1.1.0's for every completion. score_confidence is in the breakdown for
       training code that credits the c tokens separately (split credit, D46), which is the synthesis's plan: "Brier
       with the answer held fixed".
     - **Why not positive:** a positive weight couples the channels. With weight 1, a wrong answer stated with
       c = 10% (crps −0.3, Brier −0.01: total −0.31) beats a right answer stated with c = 10% (crps −0.05, Brier
       −0.81: total −0.86).
  2. **τ = 0.1%, A1's value. This needs a look before Phase 12.** On D69's 2,000 4-bit SFT calibration answers:

     | τ | 0.05% | **0.1% (A1)** | 0.2% | 0.5% | **1% (verifier "correct")** | 2% |
     |---|---|---|---|---|---|---|
     | share of answers computation-correct | 10.5% | **14.5%** | 19.6% | 30.1% | **39.9%** | 48.8% |

     - **What it means:** at 0.1%, about two thirds of the answers that are right within 1% count as computation-
       incorrect. That is mostly intermediate rounding; the SFT writes 4-significant-figure intermediates.
     - **The risk:** the c channel would then partly score numeric precision rather than correctness.
     - **Unchanged:** A1's τ was set for comparing σ with σ\* on computation-correct answers (D58), and I have not
       changed it. τ is one config value here.
- **Replay (your requirement):**
  - **Unchanged for old configs:** 1.0.0 / 1.1.0 configs carry no `confidence` section, so the scoring path, the
    reward and the breakdown are exactly as before.
  - **Tested:** both committed 1.1.0 tapes checked (400 + 400 rows) re-score to identical rewards **and identical
    breakdowns** (stronger than replay's reward check). The existing 1.0.0 replay tests pass.
  - **Under 1.1.0, a `<confidence>` block is ignored** (tested).
- **Tests (`tests/test_verify_confidence.py`, 19):**
  - the versions and the unchanged default;
  - 1.1.0 ignores the block; the 1.1.0 tapes reproduce exactly;
  - Brier values against f(x_meas), not the truth, on a 5% noisy problem;
  - the noise-free reference is the truth (and an answer correct within 1% but not 0.1% is not computation-correct);
  - six malformed or missing variants get −1 with the answer score untouched; a block before the answer does not
    count;
  - an unparsed answer gets both floors; a unit mismatch makes the event false;
  - the weight enters the reward (0 reproduces 1.1.0's value);
  - propriety over the bins; five invalid configs are refused;
  - every Phase 12 data arm scores "ok" under 1.2.0, with c parsed and the event true (4-significant-figure answers).
- **Not done (as you said):** no Phase 12 training, and no eval report fields for c (calibration of c, Brier
  decomposition). Those come with the Phase 12 pre-registration.

### D84. Phase 12 explicit arm: the propagation numbers are masked in SFT, format only (your decision, 2026-10-19; resolves D80's `train_line_numbers`)
- **Your decision:** mask the propagation numbers in SFT (format only), consistent with the implicit arm's masked σ.
  Rebuild that arm's data.
- **Config (`configs/sft_phase12.yaml`):** the explicit arm gets `train_sigma: false` and `train_line_numbers:
  false`.
  - **Masked:** every number on the `sigma_rel` line, i.e. the sensitivities a_i, the levels s_i and the result X,
    plus the answer's σ.
  - **Still trained:** the line's structure (`sigma_rel = sqrt((`, `*`, `)^2 + (`, `) =`) and the `^2` exponents. The
    exponent regex already excluded them (D80).
  - **The text is unchanged:** the line still shows the correct first-order values, and the answer's σ still equals
    X·|μ|. Only the loss mask changed, so SFT teaches the format and RL must supply the values, as it must for σ in
    the implicit arm.
- **The rebuild:** `python -m physcal.train.sft_phase12 build --arms explicit`.
  - **Rows now carry their loss masks** (`masked`, character spans), so the built data records the decision, and
    `tokenize` refuses a config whose masks differ from the data's.
  - **Only `explicit.jsonl` was rewritten** (sha256 `bc9861ea…`). The other files were checked against the manifest
    and left as built under D80. Their rows carry no `masked` field; their masks come from the config, as before.
  - **The manifest records** which decision built each file, and the mean masked spans per example: implicit 1.0,
    explicit 7.2, supervised 0.
- **Tests:** `tests/test_sft_phase12.py` (7).
  - **Masks with the real tokenizer:** the explicit arm masks every line number and the answer's σ, but never `sqrt`,
    `^` or `)`; D80's trained variant masks nothing.
  - **Consistency:** stale masks are refused.
  - **The rebuilt file:** its row masks match the config.
- **Still open for Round 3 (D80):** uniform trained c, the supervised arm's token loss, n and the noise mix, and symbol
  names in the line.

### D85. Draft prompt v5: the worked example shows a placeholder for c, not 90% (your decision, 2026-10-19)
- **The change:** the example's last line is now `<confidence>c%</confidence>`, the same placeholder as the format
  block. It was `<confidence>90%</confidence>` (D80).
  - **Why:** a value in the example is a cue for c, as the old example's "about 1%" was for σ (Phase 3.5).
- **Unchanged:** the rest of v5, and every other prompt. v1–v4's hashes are tested.
- **Data:** no rebuild is needed; the prompt is applied at tokenization.
- **Still a draft:** v5 is part of the Phase 12 pre-registration (Round 3).
- **Test:** `tests/test_sft_phase12.py` checks that the example ends with the placeholder and that no c value appears
  in the prompt.

## Night of 2026-10-19/20 (unattended; your instructions at launch, ~23:05 IST)

### D86. Launch checks after the WSL restart; X2's allocator cap lowered to 5.75 GiB (decided at launch)
- **Your instructions:**
  - **Before launch** (WSL needed a service restart tonight after a disk access error): check that git status is clean,
    local matches origin/main, and there is no stale `.git/index.lock`.
  - **Then launch** `scripts/queue_x2_x3.sh`. The time gates decide what fits.
  - **Usual rules:** a failed stage is logged BLOCKED, and the next independent stage goes on; PID watchers; 87 °C → a
    10-minute pause; GPU done by 08:30 IST; commit + push after every stage; MORNING.md at the end.
- **Checks (23:04 IST): all passed.**
  - **Git:** clean, apart from the long-untracked `PHASES_10-15.md`; HEAD = origin/main = `971ffc4`; no
    `.git/index.lock`; `git fsck` reports nothing. `.git/queue-commit.lock` is the queue's flock target, an empty
    file that holds no lock.
  - **Tonight's local, untracked inputs:** intact. Content hashes: random-init `13ededef3f3acbd4`, merged SFT
    `0b435c0ebaed36e2`, pinned snapshot `9939448b3a0fbe45`. The X3 rows equal the committed file.
  - **Disk:** 913 GB free.
- **One change, to `configs/x2.yaml` only: `sampling.max_reserved_gib` 7.5 → 5.75.** The queue script is unchanged.
  - **What happened:** a host-side (Windows) process holds about 1.65 GB of VRAM, steady over three readings; this
    morning it was 0.34 GB. WSL cannot see the process.
  - **Why it matters:** D81's cap exists so that an oversized batch raises OOM, which the backoff halves, instead of
    spilling into WSL shared memory, which is only slow. With 6.3 GiB physically free, a 7.5 GiB cap would allow the
    spill. The batch probe could then pick a batch that only "fits" by spilling.
  - **The new value:** 8.0 − 1.65 − a 0.5 GiB margin, rounded down.
  - **Effect:** the probe and the sampling run see the same cap. The likely choice falls to 32; batch 16 peaked at
    3.5 GiB reserved on earlier nights.
  - **Not affected:** the other stages (suffixes, extraction, X3 SFT) need far less memory and set no cap.
- **Deadline (your message at launch, ~23:07):** all GPU work by **09:00 IST** instead of 08:30. The queue takes it
  from `GPU_DEADLINE=09:00` at launch, so the script is unchanged. Its gates (X2 downstream reserve, the X3 pilot's
  ≥ 2 h, the D78 check's ≥ 10 min) measure from that deadline.

### D87. X2 method 2 timed out (BLOCKED): σ-suffix generation spilled into WSL shared memory; fixed for the X3 evals and for a rerun after the queue
- **What happened:** the queue's `x2_suffix` stage (split credit seed 0 @287 writes a greedy σ suffix on each of the
  516 kept problems' answers) hit its 1 h timeout (exit 124, 01:11 → 02:18) without writing output.
  - **The queue's handling, as designed:** it logged BLOCKED (method 2 counts as p = 1) and went on to the
    hidden-state stages.
- **Cause (from the log and the probe):** about 60 generate calls in an hour, roughly 60 s each where about 5 s was
  expected.
  - **The shape of each call:** 32 *distinct* prompts of about 1,300 tokens (prompt + completion up to "±"), one
    prefill. The X2 sampling batches were 4 prompts × 8 samples.
  - **The probe already measured that shape:** 32 sequences × 1,108 tokens peaked at 4.57 GiB reserved.
  - **Free memory tonight:** about 6.3 GiB physically free (D86).
  - **The cap:** E1's eval path passed no allocator cap, so the overflow went to WSL shared memory (slow, no error)
    instead of OOM. D86's claim that the other stages need far less memory was wrong for this stage.
  - **Also:** `x2_pairs.suffix` imported transformers before Unsloth, which Unsloth warns can slow it.
- **Fix (one attempt; it applies to every later stage, since each stage imports the code when it starts):**
  - **Batches:** σ-suffix batches 32 → 16 (`x2.yaml suffix.batch_size`, `x3.yaml eval.batch_size`).
  - **The cap:** `e1._evaluate` takes an optional `max_reserved_gib`, which X2's suffix stage and X3's evals set to
    the sampling cap, 5.75 GiB. An overflow now raises OOM, which the backoff halves.
  - **Import order:** Unsloth is imported before transformers in both.
  - **Progress lines:** `suffixes n/N (s, OOM backoffs)` every 10 batches.
  - **E1's own defaults are unchanged.** 28 CPU tests pass (E1, X3, X2).
- **Plan:**
  - **The X3 evals in the queue** pick the fix up automatically.
  - **After the queue's GPU stages end, if time remains before 09:00:** rerun the method-2 stage by hand
    (`python -m physcal.eval.x2_pairs suffix`, about 15 min expected), then rerun the X2 analysis.
  - **The queue's own X2 analysis** runs without method 2 and is superseded by the rerun. Both are committed. The
    rerun is the complete D74 analysis; Holm's family size does not depend on it.

### D88. Student (a)'s X2-answer eval crashed on an answer of μ = 0 (BLOCKED); fixed before student (b)'s
- **What happened:** `x3_eval_a_x2` generated all 4,116 σ suffixes in about 18 min (D87's fix works: no OOM backoff),
  then crashed while scoring with a `ZeroDivisionError` (exit 1, logged BLOCKED).
  - **The cause:** E1's μ-changed check divides by the original μ, and one of the 4,116 kept X2 answers states
    μ = 0. Six others are negative; those already passed through safely.
  - **Why it was never hit before:** E1's prefixes (SFT greedy answers) never contained μ = 0.
  - **Effect:** the generated texts were lost; the stage must be rerun.
- **Fix (`physcal/train/e1.py`; it changes nothing for any existing E1 row, which never had μ = 0):**
  - **The μ-changed check:** when the original μ is 0, it compares directly (a suffix that moves μ is still caught).
  - **The row:** a suffix with no relative σ (μ = 0) is a failed suffix, so a tie in X2's pairwise accuracy, as D74
    treats an unusable suffix.
  - **Test:** `tests/test_e1.py`.
  - **In time for the rest of the queue:** applied while student (b) trains, so its X2 eval, the method-2 rerun
    (D87) and student (a)'s rerun all use it.
- **Plan:** after the queue's GPU stages, rerun `x3 eval --arm a --set x2` (about 18 min), then `x3 analyze`. The
  queue's own X3 analysis lacks (a)'s X2 numbers and is superseded.

## 2026-10-20 day (CPU only)

### D89. Pre-registration of X4: self-verification on the X2 pairs (written before any X4 verification, reading or score exists)
- **Your instructions (2026-10-20):** show the 4-bit SFT model its own completed answer and ask, with a fixed yes/no
  verification prompt, whether the answer is correct; it may recompute first. Confidence = P("yes") from the
  next-token probabilities. Primary metric: within-problem pairwise accuracy on the X2 pairs (same pairs, same Holm
  conventions as X2), vs 0.5 and vs the X2 probe. Secondary: AUROC across problems; per depth. Controls: the same
  prompt on the random-init model; a shuffled-answer control (the model judges another problem's answer).
- **Question:** X2 (FINDINGS #19) found that no stated σ, text model or hidden-state readout ranks a correct answer
  above an incorrect one to the same problem. Can the model do it when it is *asked*, with room to recompute?
- **What exists before any X4 data:** this entry; `physcal/eval/x4_verify.py`, `physcal/eval/x4_analysis.py`,
  `configs/x4.yaml`, `tests/test_x4.py` (CPU, a tiny model); and the X2 reference confidences (below), computed from X2
  outputs only.
- **The answers and the pairs (exactly X2's):**
  - **The problems:** X2's 516 kept problems (`results/x2/selected.json`; `x2_pairs.kept_groups` checks the keys).
  - **The answers:** every parsed answer of those problems (4,116: μ parsed, right dimension), i.e. every answer that
    enters an X2 pair. **The pairs:** X2's 5,390 (correct, incorrect) pairs; correct = |relative error| ≤ 1%.
  - **Order:** the kept order (the pool's round-robin order), then the sample index.
- **The verification conversation (one per answer):**
  1. system: prompt v4 (the prompt the answer was sampled under);
  2. user: the problem text;
  3. assistant: the answer's completion, verbatim (the model's own sampled text);
  4. user: the fixed verification prompt `x4_v1` (`physcal/prompt.py VERIFY_PROMPTS`; SHA-256 `5f5fa39d…0f34`):
     > Now check your solution above. Is the final value in your <answer> correct to within 1%? You may redo the
     > calculation before you decide. Then end your reply with one line, either "Verdict: yes" (the value is correct)
     > or "Verdict: no" (it is not).
  5. the assistant generation prompt.
- **Recomputation:** greedy decoding, up to 512 new tokens (the answers' own budget; their completions are at most 340
  tokens), stopping at the end-of-turn token; repetition penalty 1.0. Batches of 16 conversations (2 problems × 8
  answers) with left padding and the allocator cap (below); an OOM halves the batch (the eval backoff).
- **The reading position:** the generated text is cut before its **last** occurrence of "verdict" (case-insensitive;
  trailing whitespace and `*`, `_`, `#` dropped), and the canonical line `Verdict:` is appended (after a newline if any
  text remains). If the model wrote no verdict, or hit the 512-token limit, the line is appended after all of its text.
  So every answer is read at the same canonical position, whether the model wrote the verdict itself or not.
- **The confidence:**
  - **Next-token logits at the reading position,** in float32: lm_head applied in float32 to the final (post-norm)
    hidden state of the last token. The model's own logits are bf16, and their rounding would create spurious ties
    within a problem.
    - **Check, on the first answer of every run:** these logits must reproduce the model's own logits there
      (max |diff| / max |logit| ≤ 0.02 and the same argmax), as X2's hidden-state check.
    - **Otherwise:** the model's own logits are used for the whole run, and this is recorded.
  - **"yes" and "no" tokens:** {yes, " yes", Yes, " Yes", YES, " YES"} and {no, " no", No, " No", NO, " NO"}. Each is a
    single Qwen token (tested).
  - **Primary confidence:** the yes-vs-no log-odds, logsumexp(logits[yes]) − logsumexp(logits[no]). It is a monotone
    function of P(yes) renormalized over yes + no, P(yes) / (P(yes) + P(no)), so it ranks exactly as that probability
    does, without float saturation near 0 or 1. Probability mass on other tokens (formatting, say) counts as neither
    answer.
  - **Secondary:** the raw P(yes) over the full vocabulary.
- **Arms (the same conversations, scored on the same pairs):**
  1. **`sft` (primary):** the corrected SFT model of X2 (the merged SFT dir, 4-bit at load by Unsloth, no adapter; the
     X2 sampling model and loader).
  2. **`random_init` (control):** X2's random-init model (`results/x2/random_init_model`, content hash
     `13ededef3f3acbd4`), loaded the same way, with the same recomputation, reading and confidence.
  3. **`shuffled` (control):** the SFT model judges another problem's answer.
     - **The swap:** each answer keeps its own completion and its own correctness label (for its own problem), but the
       user turn shows the text of a **donor** problem.
     - **The donor:** the next kept problem of the same (family, depth) cell in the kept order, cyclically. It is a
       deterministic derangement: every kept problem's text is shown exactly once, never with its own answers, and all
       8 answers of a problem share one donor.
     - **Why the same cell:** the donor asks for the same kind of target and intermediates, so the mismatch is in the
       numbers, not in the format.
     - **What it tests:** whether the ranking depends on checking the answer against the problem it answers. A ranking
       that survives the swap comes from the completion text alone, i.e. its internal consistency or surface features.
- **Primary metric:** X2's within-problem pairwise accuracy.
  - **Computation:** per kept problem, the share of its pairs in which the correct answer gets the higher confidence;
    ties and a missing confidence count 0.5. The headline is the mean over problems.
  - **Inference:** a bootstrap over problems (B = 10,000, seed 0) gives the 95% CI and a one-sided p for "> 0.5".
- **The Holm family (X2's convention: the method and its controls are all tested):** "> 0.5" for `sft`,
  `random_init` and `shuffled`, Holm at α = 0.05 over these three. An arm whose rows are missing counts as p = 1.
- **The comparisons (paired over problems, 95% bootstrap CIs, the same B and seed, X2's `paired`):**
  - **Primary:**
    - `sft − X2 probe` (method 5, the probe on the SFT hidden states at the first "±" token);
    - `sft − random_init`;
    - `sft − shuffled`.
  - **Secondary:** `sft − X2 SFT stated σ` (method 1).
  - **Where the X2 values come from:** `results/x4/x2_reference.json`, computed before any X4 data by rerunning X2's
    method-5 code on the saved X2 hidden states. The analysis refuses to run unless the probe's mean pairwise accuracy
    reproduces `results/x2/results.json` (0.5471) to 1e-9. That check is what makes these "the same pairs".
- **Reading rules (fixed now):**
  - **Definitions:**
    - **"Self-verification succeeds":** `sft` is Holm-supported and `sft − random_init` has a CI above 0 (X2's rule
      for the probe).
    - **"Problem-specific":** `sft − shuffled` has a CI above 0.
    - **"Above the X2 probe":** `sft − X2 probe` has a CI above 0.
  - **The outcomes:**
    1. **Succeeds, problem-specific, above the probe:** asking the model recovers answer-specific self-knowledge that
       no X2 readout found, and it depends on checking the answer against the problem.
    2. **Succeeds and problem-specific, but not above the probe:** self-verification carries answer-specific signal
       that depends on the problem, not detectably more than the X2 probe.
    3. **Succeeds but not problem-specific:** the signal is in the completion text itself (internal consistency or
       surface features), not in checking the answer against the problem.
    4. **`sft` not Holm-supported:** self-verification fails within problems, consistent with X2. Asking the model
       does not rank its right answers above its wrong ones.
    5. **Any other combination** is reported as observed.
- **Secondary (reported, no verdict):**
  - **AUROC across problems:** confidence vs correct, pooled over every verified answer of the analyzed problems,
    with a bootstrap CI over problems. It is reported for the three arms and, as references, the X2 probe and SFT
    stated σ. It mixes between- and within-problem signal, restricted to the mixed problems.
  - **Per depth:** pairwise accuracy of every arm.
  - **The raw P(yes)'s pairwise accuracy.**
  - **Diagnostics:**
    - the share of answers where the model wrote a verdict itself, and its yes rate on correct and on incorrect
      answers;
    - the mean renormalized P(yes) on correct and on incorrect answers;
    - the mean mass on yes + no;
    - the share that hit 512 tokens;
    - the tie share;
    - which logits path was used.
  - **Exploratory, `sft` only:** recomputation agreement. If the verification text contains an `<answer>` that the
    verifier parses (against the shown problem), confidence = 1 when its μ is within 1% of the original μ and 0
    otherwise; no parsed answer is a tie.
- **A night that ends early:**
  - Rows are appended after every batch, and a run resumes where it stopped.
  - **The analyzed problems:** each arm is analyzed on the kept problems it completed (all of their answers). That is
    a prefix of the round-robin kept order, so still stratified; its size is reported.
  - **Paired differences** use the problems that both arms completed.
- **Model and memory settings:**
  - Unsloth 4-bit; max_seq_length 3072 (the longest conversation is about 1,980 tokens; max_position_embeddings is
    4096, and the RoPE is the default, so outputs do not depend on this setting).
  - **The allocator cap** is set per stage at launch: total VRAM − `nvidia-smi` memory.used with nothing of ours
    running − 0.5 GiB, clipped to [4.0, 7.5] GiB (the WSL-spill rule of D86/D87). It is recorded in each run's meta.
- **Order of the GPU runs:** `sft`, then `shuffled`, then `random_init`.
  - **Why `shuffled` first:** it is the more informative control. `random_init` has no fitted readout, so its expected
    pairwise accuracy is 0.5 apart from surface correlations. It is also the most expensive, because a random-init
    model never stops before 512 tokens.
- **[REVIEW] items** (they can still change before launch, since no X4 data exists):
  1. the wording of `x4_v1`;
  2. the renormalized yes-vs-no log-odds as the primary confidence (raw P(yes) as secondary);
  3. the donor rule (next kept problem in the same cell);
  4. the order of the controls.

### D90. Tonight's GPU queue, 2026-10-20/21: X4, then its controls (prepared, NOT launched; `scripts/queue_x4.sh`)
- **Your instructions (2026-10-20):** prepare, do not launch: X4 on all X2 pairs, then the controls. Deadline 09:00
  IST.
- **Stages:**
  0. **Input checks:** the X2 reference, `selected.json`, the random-init and the merged SFT models. A missing input
     is logged BLOCKED. An arm that already has rows is noted; it resumes.
  1. **GPU: `sft`** on all 4,116 X2 answers.
  2. **GPU: `shuffled`.**
  3. **GPU: `random_init`.**
  4. **CPU: the D89 analysis** (`x4_analysis analyze`, nice, 4 threads).
- **The allocator cap, per GPU stage:** measured just before the stage (`x4_verify cap`), as D89 specifies: total −
  `nvidia-smi` memory.used − 0.5 GiB, clipped to [4.0, 7.5].
  - **This morning's reading:** 7.23–7.35 GiB, with 171 MiB used and nothing running.
  - **If a host process holds VRAM tonight** (D86: 1.65 GB), the cap falls with it, and no launch-time edit is needed.
- **Time limits.** Each arm gets a soft limit (`--minutes`: it stops cleanly between batches) and a hard timeout
  10 min later.
  - **The soft limits:** `sft` = time left − 4 h (kept for the two controls); `shuffled` = time left − 2 h (kept for
    `random_init`); `random_init` = time left − 15 min.
  - **A stage that stops early, or fails,** leaves a prefix of the kept order. Its rows are committed. D89 analyzes the
    problems each arm completed. Rerunning the same command resumes it.
- **One runtime detail added to D89's "an OOM halves the batch":** the eval backoff halves only the call that failed.
  X4 then keeps the halved size for the rest of the run, so it does not repeat the OOM on every batch (tested). The
  final size is recorded in the run's meta.
- **Time estimates.** Nothing of X4 has run on the GPU; these come from last night's measurements.
  - **Prefill:** about 3–4 s per batch of 16 conversations of ~1,300–1,470 tokens. X2's σ-suffix rerun did 16
    distinct ~1,300-token prompts at ≈ 3.8 s per batch, including 24 new tokens.
  - **Decoding:** about 55–70 ms per step. X2's batch probe measured 32 sequences × (1,108 + 512) tokens at 35 s, i.e.
    ≈ 68 ms per step. A batch decodes until its longest recomputation ends.
  - **Reading:** about 3 s per batch: 4 forwards of 4 contexts each.
  - **Per arm (258 batches):**
    - **`sft` and `shuffled`:** about 1.7 h if a batch's longest recomputation is ~300 tokens, about 2.7 h if 512.
    - **`random_init`:** about 2.7 h. A random-init model does not stop before 512 tokens.
  - **The total: 6.1–8.1 h.** Launched at about 23:00, all three arms finish before 09:00 even at the upper estimate.
  - **A later launch:** `random_init` is cut short (a prefix, analyzed as D89 says) after about 00:40 at the upper
    estimate, or after about 02:40 at the lower one.
- **Memory, expected:** 16 conversations × (~1,470 + 512) tokens.
  - **The nearest measurements:** the X2 probe's 32 × (1,108 + 512) peaked at 4.6 GiB, and D87's 16 × ~1,300 ran
    under the 5.75 GiB cap. So this should fit under ~7.2 GiB.
  - **If it does not,** OOM halves it (above) instead of spilling into WSL shared memory.
- **Guards** (as every queue so far):
  - every stage is a child waited on by PID;
  - GPU > 87 °C before a stage → a 10-minute pause;
  - commit + push after every stage, under a flock;
  - a failed stage is logged BLOCKED, and the next arm goes on (the arms are independent);
  - `HF_HUB_OFFLINE=1`.
- **Checked on CPU:**
  - `bash -n`; `DRY_RUN=1`, at the real time (the deadline 16 min away, so every limit came out negative, as it
    should) and with a deadline 10 h away (soft limits 360 / 480 / 585 min).
  - `tests/test_x4.py` (14) with the X2 tests (24 pass); the full suite (470 before the last X4 test).
- **Not checked: any GPU stage.** That covers:
  - Unsloth's generate on two-turn conversations;
  - the float32-head check on the 4-bit models. The expected gap is about X2's measured 0.13% (SFT) / 0.47% (random
    init), against the 2% limit; a failure only switches the arm to the slower model-logits path;
  - the memory and the speed above.
- **Launch command (09:00 is taken as the next morning's):** `nohup scripts/queue_x4.sh > results/queue_x4/nohup.log 2>&1 &`

### D91. Amendment to X4 (D89): a `direct` mode, a `perturbed` control, problem-level cluster bootstrap, token cost; tonight's order (your instructions, 2026-10-20 night; written before any X4 data exists)
- **State when written:** `results/x4/` holds only `x2_reference.json` (CPU, from X2 outputs). No X4 verification,
  reading or score exists. Everything in D89 stands unless changed here.
- **a. Mode `direct` (new arm `direct`):** the SFT model, the answer's own problem, the same conversation as D89 (v4
  system prompt, problem, the completion verbatim, `x4_v1`, the generation prompt), but **no generated tokens**:
  - **The reading context** is the generation prompt followed directly by the canonical marker `Verdict:`. That is what
    D89's reading rule gives for an empty generation, so both modes are read at the same canonical position.
  - **Everything else is D89's:** float32 head, the first-row logit check, the yes/no token sets, the primary log-odds
    and the secondary raw P(yes).
  - **The verification prompt is unchanged,** although it says "You may redo the calculation": the only difference
    between the modes is whether the recomputation tokens exist.
  - **Reported side by side with recompute (`sft`):** every D89 metric for both, and the paired difference
    `sft − direct` (what the recomputation tokens add).
- **b. Control `perturbed` (new arm `perturbed`):** the SFT model, recompute mode (as `sft`), the answer's own problem,
  the same steps; only the final value changes.
  - **The edit:** in the completion's single `<answer>` block, the μ number (the first number after the block's last
    `=`, or at its start) is replaced by μ × 1.25. σ, the unit and every other character of the completion stay as
    they were.
  - **The new number is written** with the original's significant digits (at least 3), in the original's notation: a
    plain decimal stays plain; with an exponent (`e`, `× 10^k`), only the mantissa changes.
  - **Checked on every answer:** the edited completion must re-parse (the verifier's parser) to μ′ within that rounding
    of 1.25 μ, with the same σ and unit. An answer that fails the check, or has μ = 0 (no change), is excluded from
    `perturbed`; the count is reported.
  - **Labels:** each answer keeps its original label, as in `shuffled`. Its scores sit on the same X2 pairs. If the
    verifier checks the shown final value, the ranking should collapse: every originally-correct answer is now 25% off.
  - **Also reported (secondary):**
    - **Detection:** for originally-correct answers, the share where the `perturbed` confidence is below the `sft`
      confidence of the same answer (ties 0.5). Mean over problems.
    - **How many perturbed answers are correct** against the truth (an originally-wrong answer can become right).
- **c. Uncertainty for every X4 result: a cluster bootstrap over problems.**
  - **The resampling:** the analyzed problems (516 when complete) are resampled with replacement; B = 10,000, seed 0.
    Each statistic is recomputed from all pairs and answers of the resampled problems. Pairs and answers are never
    resampled on their own.
  - **What changes:** D89's headline (per-problem accuracy, mean over problems) and its AUROC already resample problems.
    This makes the cluster bootstrap the rule for every X4 statistic, the new ones included (detection, diagnostics,
    token counts).
  - **Paired differences** use the same resampled problems for both arms.
  - **Extra inference tokens per mode,** per answer (mean, median, 95th percentile; mean with a cluster-bootstrap CI):
    - generated tokens (`direct`: 0);
    - total extra tokens beyond the answer's own conversation: the verification turn, the generated tokens and the
      marker.
- **The Holm family, extended:** "> 0.5" for `sft`, `direct`, `perturbed`, `shuffled`, `random_init`; Holm at
  α = 0.05 over these five. An arm with no rows counts as p = 1 (D89).
- **Comparisons:** D89's stand (`sft − X2 probe`, `sft − random_init`, `sft − shuffled`; secondary `sft − X2 SFT stated
  σ`). Added:
  - `sft − perturbed` (primary for the new control);
  - `sft − direct`; `direct − X2 probe`; `direct − random_init` (secondary).
    - The last uses the recompute-mode `random_init`. No direct-mode random-init arm runs; this is a limit, stated.
- **Reading rules added (D89's are unchanged):**
  - **"Final-value-sensitive":** `sft − perturbed` has a CI above 0. If not, the `sft` ranking does not depend on the
    shown final value (the model judges the steps or the surface of the completion).
  - **"Direct self-verification succeeds":** `direct` is Holm-supported and `direct − random_init` has a CI above 0.
  - **"Recomputation helps":** `sft − direct` has a CI above 0.
- **Tonight's run order (GPU deadline 11:30 IST, replacing D90's 09:00):** `sft` (recompute, main), `direct`,
  `perturbed`, `shuffled`, `random_init`, then the CPU analysis. Later stages may not fit. Each stage runs behind a time
  gate, and a stage cut short leaves a prefix of the kept order (D89).
  - **`sft`** starts at once with the tested D89/D90 code: a frozen copy of `scripts/queue_x4.sh` with only its `sft`
    stage. Its soft limit is the time left − 1 h (for `direct`).
  - **The rest** are built and CPU-tested while `sft` runs, then chained to start when its PID exits.

### D92. X2 robustness checks: post-hoc and exploratory (your instructions, 2026-10-20 night; CPU only; defined before computing)
- **Status:** not pre-registered. Every number from this entry is labelled post-hoc/exploratory and cannot change any
  D74 (X2) or D89/D91 (X4) verdict.
- **The data:** X2's 516 kept problems and 5,390 pairs; X2's per-answer confidences, recomputed with X2's own code
  (`x2_analysis`). They must reproduce `results/x2/results.json` to 1e-9, or the check refuses to run. Code:
  `physcal/eval/x2_robustness.py`; outputs in `results/x2/robustness/`.
- **a. Clearly separated pairs:**
  - **Which pairs:** the correct answer has |relative error| ≤ 0.25% and the incorrect one ≥ 10%. Both are
    thresholds on the verifier's `abs_rel_error`, in config.
  - **The metric:** pairwise accuracy as in X2. Per problem, the mean over its separated pairs (ties 0.5); the mean
    over the problems that have at least one such pair.
  - **Reported for:** every X2 method, the two new baselines (b, c) and, once their rows exist, the X4 arms.
- **b. Completion length:**
  - **Right vs wrong:** completion length in tokens (the SFT tokenizer), for correct and for incorrect answers.
    Reported: the pooled means, and the within-problem difference (per problem, mean correct − mean incorrect,
    averaged over problems).
  - **The length-only baseline:** confidence = −length (shorter = more confident), X2's pairwise accuracy. The
    opposite direction is 1 − that (the ties are the same).
- **c. Self-consistency baseline:** confidence = the fraction of the problem's other 7 samples whose parsed μ agrees
  with this answer within 1% (|μ_j / μ_i − 1| ≤ 1%, SI). An unparsed sample never agrees. X2's pairwise accuracy.
- **d. X2's CIs, problem-level cluster bootstrap:**
  - **What X2 already did:** X2's published CIs (`x2_pairs.summarize`, `x2_analysis.paired`) already resample
    problems: the per-problem accuracy, mean over problems.
  - **What is recomputed:** those numbers (they must match), and the pooled pair-weighted pairwise accuracy (all
    pairs, each pair weighing the same) with a cluster bootstrap over problems: problems resampled with all their
    pairs.
- **Uncertainty everywhere:** a bootstrap over problems, B = 10,000, seed 0. The one-sided p for "> 0.5" is
  reported, with no Holm family: exploratory.

### D93. Round-3 review amendments R3-1 to R3-10 (`REVIEW_SYNTHESIS_ROUND3_2026-10-20.md`; recorded 2026-10-20 night, before any result they affect)
- **State when recorded:** X4 `sft` rows are being written on the GPU; no X4 analysis, reading or score exists. No
  E1c, E1d, A6c or Phase 12 data exists. X2 robustness (D92) code exists; its full run is in progress.
- **R3-1, X4 additions:** done as **D91** (direct mode, the perturbed control, problem-level cluster bootstrap,
  extra inference tokens per mode), before any X4 data.
  - **Later, not tonight:** a yes/no order variant (affirmative bias), and a deterministic arithmetic checker as a
    ceiling. Each needs its own entry before it runs.
- **R3-2, X2 robustness (post-hoc, exploratory):** done as **D92**.
  - The self-consistency baseline costs 8 samples per problem at inference. It is reported next to its number.
- **R3-3, optimization pilots before Phase 12 RL:** E1c (σ advantages normalized over the batch) and E1d (higher
  learning rate), on the E1 machinery. Pre-registered in **D94**, configs and CPU tests built, NOT launched.
  - **The decision rule:** whichever moves ρ sets the RL configuration for Phase 12 (D94 gives the test).
- **R3-4, Phase 12 primary scope: aleatoric σ propagation only.** Stages:
  - **A (CPU):** X1, plus noise heuristics (max, mean, root-sum-square of the stated noise); a scoring audit; the
    oracle-centred diagnostic (R3-6). The heuristics are built tonight with A6c (**D95**).
  - **B:** a supervised σ\* model, 3 seeds: the propagation ceiling. It comes before any RL night.
  - **C:** implicit σ-RL, 1 pilot with the R3-3 configuration; 3 seeds only if it shows signal.
  - **D:** explicit propagation, 1 diagnostic seed each.
    - **The positive control:** SFT with real propagation numbers, i.e. D80's trained variant, with the line numbers
      in the loss.
    - **The ablation:** format-only SFT, i.e. **D84**'s masked arm.
    - **This amends D84:** D84's masked arm stays, as the ablation, no longer as the only explicit arm. Full seeds
      only if D resolves B vs C.
  - **E:** p (confidence, D83's `<confidence>` channel) is deferred to a follow-up. If X4 works, verification-derived
    p becomes its teacher or baseline.
  - **Not changed tonight:** no Phase 12 code, data or config changes with this entry. Your instruction was "do not
    change anything else in Phase 12".
- **R3-5, a new counterfactual set A6c: noise-allocation pairs.**
  - **The pairs:** the same multiset of noise percentages, with the inputs that receive them swapped (e.g.
    y = a·b², 1% on a and 5% on b, then reversed). The visible percentages are identical; the correct σ\* differs.
  - **Role:** a primary propagation test alongside A6/A6b, compared against X1 and the R3-4 heuristics.
  - Built tonight, frozen like A6/A6b (**D95**); no model is evaluated on it.
- **R3-6, tolerances and selection bias:**
  - **The confidence event:** |ŷ − y_display| / |y_display| ≤ 1%, with y_display computed from the displayed
    givens. Sensitivity analyses at 0.1% and 0.5%.
    - This replaces D83's τ = 0.1% (A1's value) as the primary.
    - It is a `reward_v12.yaml` change, made when p is next worked on (R3-4 E defers it). Verifier 1.2.0 is not
      edited tonight.
  - **σ propagation, primary:** the oracle-centred diagnostic on all problems. The stated σ is scored with μ replaced
    by f(x_meas), which avoids selecting on correctness.
  - **σ propagation, secondary:** the correctly computed answers, with τ_σ = max(0.25%, 0.25·σ\*).
  - **Noise-0 problems** are handled separately (log σ\* is undefined).
- **R3-7, statistics:**
  - Bootstrap over problems or templates, never over overlapping pairs. X4 (D91) and D92 follow it.
  - The credit-assignment result is described as 3 vs 3 seeds, not 9 independent replications. It applies to the
    write-up and every later summary.
- **R3-8, claim strength:** Part II is stated "for the tested estimators and this model", with no claim about model
  size.
  - **The defences:** X4, candidate verification and, optionally later, an inference-only scale check on a 7B model.
- **R3-9, explicit arm mechanics (when stage D is built):**
  - SFT teaches the copy step (answer σ = line result × |μ|).
  - Split credit routes the σ reward to the line's number tokens as well as to the answer's σ.
  - If p is ever trained, its loss coefficient is specified separately. A zero weight in the total reward does not
    mean zero effect through separately normalized token advantages.
- **R3-10, references:** these are to be verified before citing, none cited yet:
  - Huang et al. (arXiv 2310.01798); Zhang et al., Generative Verifiers (arXiv 2408.15240);
  - Tian et al. (arXiv 2305.14975); Weng et al. (arXiv 2212.09561);
  - Xia et al. (arXiv 2601.08064); Ma et al. (arXiv 2608.07827);
  - Zeng et al. (arXiv 2504.06564); Leng et al. (arXiv 2410.09724).
  - Each gets checked against its arXiv page before it enters the write-up. A reviewer's "now verified" does not
    count.
- **The six points both reviewers agree on** (the file's first section) are recorded as given there.
  - **No claim** of a model-size limit, and no claim that RL's failure is purely optimization without the R3-3 test.
  - **Monte Carlo σ\* is not called exact.**

### D94. Pre-registration of E1c and E1d, the R3-3 optimization pilots (written before any E1c/E1d data exists; built, NOT launched)
- **Question (R3-3):** E1 had clean credit (μ fixed, only σ varies within a group), yet learned an almost constant σ.
  - **The corrected 4-bit eval (D67):** ρ(σ, |error|) = 0.013 for E1a and 0.014 for E1b; 0.001 for the SFT σ; 0.70
    for the A5 σ on the same answers.
  - **Supervised training on the same kind of labels** reached ρ ≈ 0.58 (X3, FINDINGS #20).
  - **So:** is E1's failure an optimization artifact? Two candidates: (c) per-prompt advantage normalization;
    (d) the learning rate.
- **The base: E1a, exactly** (`configs/grpo_e1a.yaml`, D52/D53). Free σ; the unchanged verifier reward (crps_log
  + 0.2·steps); the same 1,996 training prefixes in the same shuffled order (seed 0); 300 steps of 4 prefixes × 8
  samples; LoRA r = 16; DAPO loss; β = 0; grad-norm clip 0.1; the 7.5 GiB cap; the same loader. Each pilot changes
  one variable:
  - **E1c (`configs/grpo_e1c.yaml`): `scale_rewards: batch`.** The advantage keeps the prompt's own mean as its
    baseline, (r − group mean), so credit stays within the prompt. It is divided by the std over the step's whole
    batch (all 32 completions: 4 prompts × 8), not by the std within each prompt's 8.
    - This is TRL 1.13's `scale_rewards="batch"`. E1a ran with TRL's default `"group"`.
    - **Checked in source:** Unsloth's compiled GRPOConfig overrides `scale_rewards` only for `loss_type: dr_grpo`;
      E1 uses `dapo`. `summary.json` records the value the trainer actually used.
    - **Why it might matter:** per-prompt scaling gives every prompt unit-variance advantages. Prompts whose μ is
      exact (where σ hardly changes crps_log) then push as hard as prompts whose μ is far off (where a wide σ pays a
      lot). Batch scaling keeps those relative sizes.
    - **[REVIEW]: the other reading of "normalized across the whole batch"** is to center at the batch mean as well.
      That re-mixes μ error (between prompts) into σ's advantage, so it was not chosen. It can still change before
      launch.
  - **E1d (`configs/grpo_e1d.yaml`): learning rate 5e-5,** 10× E1a's 5e-6. Same schedule (constant after 5 warmup
    steps).
    - **A correction to R3-3:** the SFT learning rate is 1e-4 (`configs/sft.yaml`), so RL's is 20× below SFT's, not
      ~40×.
    - **[REVIEW]:** 5e-5 sits between E1a and SFT on a log scale. The alternatives are 2.5e-5 and 1e-4. It can
      change before launch.
- **Evaluation:** E1a's corrected eval, unchanged (D67).
  - **The eval:** `python -m physcal.train.e1 eval --run <run> --adapter final_adapter --out-name final_4bit` (the
    greedy σ suffix on the 400 frozen eval prefixes; 4-bit base, adapter unmerged), and the same for step 150.
  - **The reference** is E1a's existing `final_4bit` rows.
- **Primary metric:** Spearman ρ(relative σ, |log error|) over the eval prefixes. These are `e1_compare.describe`'s
  rows: failures and μ ≤ 0 left out, as for E1a.
- **Tests (family E-opt):** ρ(E1x) > ρ(E1a), for x = c and d.
  - **The p-value:** one-sided, from a paired bootstrap over the 400 problems (B = 10,000, seed 0). Each resample
    recomputes both ρ on the same problems; p = the share of resamples with ρ(E1x) − ρ(E1a) ≤ 0.
  - **Holm** at α = 0.05 over the two tests. A pilot that did not run counts as p = 1.
- **The decision rule (R3-3: "whichever moves ρ sets the RL configuration for Phase 12"):**
  - **Exactly one Holm-supported:** its setting becomes Phase 12's RL setting.
  - **Both supported:** the one with the higher ρ point estimate. Combining them would be a new, untested
    configuration; it needs its own entry.
  - **Neither:** Phase 12 RL keeps E1's settings. The negative result is reported: under these optimizer settings,
    GRPO does not learn a problem-dependent σ with clean credit.
- **Reported, not tested:**
  - D52's family E for both pilots: crps_log vs the SFT σ and vs the strongest A5 σ, paired, on the same answers;
  - 1σ / 2σ coverage, median relative σ, the σ = 0 share, the 9-bin entropy;
  - parse-fail, unit-mismatch and μ-changed rates;
  - per-step training reward;
  - ρ at step 150.
- **Collapse check (E1d mainly):** if one σ string makes up > 90% of the eval suffixes, the run is reported as
  collapsed, whatever its ρ.
- **Pipeline sanity:** D52's rule. If more than 10% of step-1 suffixes fail to parse, or the run OOMs, it is a
  pipeline failure, not a result.
- **Budget:** 1 seed each (pilots), about 2.7 h each (E1a: 300 steps in 2.71 h). The R3 schedule puts them on the
  night of Wed 21.
- **Built (CPU):** the two configs; `e1.grpo_args` (E1a/E1b's arguments are unchanged; `scale_rewards` passes
  through when set); `e1_compare.compare_pilots` (the ρ tests); tests in `tests/test_e1_pilots.py`.

### D95. A6c noise-allocation pairs and the noise heuristics (R3-5, R3-4 A): construction and analysis fixed before the set is built; no model is evaluated on it
- **Purpose (R3-5):** a pair keeps the same multiset of visible noise percentages and swaps which input receives
  them. The visible percentages are identical, but σ\* differs. This defeats "copy the largest (or mean, or RSS)
  percentage", which A6 cannot: within an A6 pair every given has the same level.
- **Construction (`kind: allocation`, `configs/pairs.yaml`; code in `physcal/env/pairs.py`):** per family, depth,
  level pair (l, h) and pair seed:
  1. **The chain:** at least 2 givens. The world is the first of up to 24 draws that passes the generator's validity
     and κ ≤ 5 filters at its true givens.
  2. **The allocation:** for each given j, the delta-method σ\* with h on j and l on every other given (the
     sensitivities at the true givens).
     - **Member lo** puts h on the j with the smallest σ\*, member hi on the j with the largest. Ties go to the
       earlier given.
     - **Kept** if σ\*_hi / σ\*_lo ≥ 1.5 (delta method). Otherwise the next chain, up to 20.
     - **Why 1.5, not A6's 2:** with (1%, 2%) no allocation can reach 2. Before this entry, a CPU feasibility check
       of the construction only (no model) found 11–39% of (1%, 2%) chains at ≥ 1.5, and 35–96% for (1%, 5%).
  3. **Identical displayed values in both members:**
     - Member lo: measured = true · (1 + e_lo), with e_lo = levels_lo ⊙ z.
     - Member hi: the same unrounded measured values, with levels_hi. Its true givens are backed out,
       true_hi = measured / (1 + levels_hi ⊙ z), with the same z. So both members follow the generative model, and
       only the two swapped givens' true values differ.
     - **Checked:** every displayed number string is identical, and the text differs only in those two "± p%".
       Member hi must also pass the validity and κ filters; otherwise the next chain.
  4. **Shared, as in A6:** chain, phrasing, given and target units (from lo), z, the MC standard-normal draws
     (`n_mc` of the generator), and the distractors (shown at level l, the same in both members).
  5. **σ\*:** the MC value (A2) with per-given levels, pushed through the chain, as `noise.predictive` already
     does. Delta-method σ\* is stored too.
- **Identity and replay:**
  - **Members:** seeds 2·pair_seed + {0 (lo), 1 (hi)}.
  - **`noise.pair`:** the construction including `levels: [l, h]`, the member, the chain try, the delta ratio, and
    the given that carries h.
  - **Per-given levels:** in `noise.givens[g].rel`. `noise.level` = h, the largest stated level, used for per-level
    tables and by `generator.regenerate`. The construction must agree with it.
  - **No generator change:** `generator.regenerate` already sends every pair problem to `pairs.generate_pair`,
    which dispatches on the stored construction's kind.
- **Frozen set:**
  - **Size:** depth 1–5 × level pairs {(1%, 5%), (2%, 5%), (1%, 2%)} × 34 pairs = 510 pairs, 1,020 problems.
  - **Seeds:** families round-robin; pair seeds from 2.0e9 (disjoint from A6 1.8e9, A6b 1.9e9); at most 136 seeds
    per cell, shortfalls reported.
  - **Files:** `data/eval_sets/2.0.0+bdbb2279/a6c_pairs.jsonl` + `a6c_pairs_manifest.json` (SHA-256; per cell:
    pairs, seeds tried, the MC σ\* ratio median and min, the family mix).
- **Analysis for a model (later, when one is evaluated):**
  - **Variables:** x = ln σ\*_hi − ln σ\*_lo (MC); y = ln σ_hi − ln σ_lo (stated relative σ; σ = 0 → the 1e-6
    floor).
  - **Primary:** all pairs with both σ parsed, so no selection on correctness (R3-6). The test is A6's statistic:
    the OLS slope of y on x with an intercept, one-sided bootstrap over pairs (B = 10,000, seed 0) for slope > 0.
    Pairs do not overlap, so a pair is the cluster (R3-7).
  - **Secondary:** both answers computation-correct (A1, τ = 0.1%, A6's primary); mean y > 0; the through-origin
    slope Σxy / Σx².
  - **Reported in one table** with A6 and A6b.
- **The noise heuristics (R3-4 A; `physcal/eval/noise_heuristics.py`):** a stated relative σ from the visible
  percentages alone.
  - **The three:** max, mean and root-sum-square (√Σ s_i²) of the levels.
  - **Primary:** over the chain's givens (a solver that knows which values matter). **Secondary:** over every shown
    measurement, distractors included (a surface reader).
  - **Evaluated now, on CPU, on the frozen A6, A6b and A6c sets:**
    - each set's pair statistic (A6 slope, A6b slope, A6c slope);
    - per problem: median |ln(σ_h / σ\*)|, Spearman ρ(σ_h, σ\*), and the slope of ln σ_h on ln σ\* over all
      problems.
  - **What follows by construction:** every heuristic is constant within an A6, A6b and A6c pair (y = 0). So each
    fails A6 and A6c and passes A6b. That is the reference point A6c exists to make.
  - **The comparison with X1 (D73) on A6c** needs X1's fitted model applied to the new set. It is left for the next
    day.

### D96. Tonight's run, 2026-10-20/21: implementation decisions taken during the night (overnight rules: decision, alternatives, why)
1. **Launching `sft` first.**
   - **Problem:** D90's `queue_x4.sh` would have run `shuffled` right after `sft`, not D91's order.
   - **Decision:** `sft` was launched from a frozen copy with only its stage (`scripts/queue_x4_main.sh`, committed),
     at 23:22 IST, with the soft limit = time left − 1 h. `queue_x4.sh` was then rewritten for D91's order
     (`X4_ARMS`, `WAIT_PID`), dry-run, and started as the chain, waiting on the frozen queue's PID.
   - **Why a copy:** bash reads a running script as it goes, so the script a queue runs is never edited while it
     runs. **Alternative:** adding an arm filter to the tested script before launch delays the GPU, and you said
     launch now.
2. **The GPU temperature rule inside a stage (`scripts/gpu_temp_guard.sh`).**
   - **What D90's queues did:** checked temperature only before a stage. Tonight's stages are 15 min – 2.5 h long.
   - **The guard:** polls every 30 s. Above 87 °C it SIGSTOPs the stage's process group (`timeout` → `uv` →
     `python`) for 600 s, then SIGCONTs it. It is PID-based, tested on a dummy stage, and attached to `sft` by hand,
     then to every chained stage by the queue.
   - **It fired once:** 88 °C at 23:42:56; the stage continued at 23:53:52 with the GPU at 56 °C. The stage's soft
     limit counts the pause; that was harmless tonight.
   - **The cause** was mostly my CPU job: the refit below, 2 joblib workers at ~3 threads each. From then on, CPU jobs
     ran on 1 thread at nice 19 while the GPU was busy.
3. **X2 robustness (D92): the refits wait for the end of the queue.**
   - **What happened:** under 1 thread, the cross-fitted probe recomputed 0.5390 vs X2's published 0.5396. D92's
     reproduction check refused it, and the run stopped as designed. The random-init probe refit reproduced exactly.
   - **Why:** the inner-CV (layer, C) selection is sensitive to BLAS summation order, i.e. to the thread count.
   - **Decision:** the refits rerun after the GPU queue under X2's own settings (4 jobs, 4 BLAS threads), and the
     reproduction rule stays as written. The results without the two refitted probes were committed first,
     labelled "refits pending".
   - **Alternative:** relaxing the 1e-9 rule for the cross-fit. Rejected: D92 fixed it before computing.
4. **The A6c feasibility check** (construction only, no model) set `min_ratio` = 1.5. It is recorded in D95.
5. **The noise heuristics are order-invariant:** the levels are sorted before max / mean / RSS.
   - **Why:** the first run showed RSS "supported" on A6c with slope +0.000. Summing the same squares in a
     different order made y ≈ 1e-17 instead of exactly 0. The fix ran before any result was committed.
6. **A dry run of the D91 analysis** on the real `sft`, `direct` and partial `perturbed` rows, into the scratchpad,
   checked the code path on real data before the queue's analysis stage. No file in `results/x4` was written; the
   pre-registered analysis is the queue's.

### D97. Your decisions on the 2026-10-20/21 report, and today's GPU queue (2026-10-21 day; `scripts/queue_e1_pilots.sh`)
- **Approved as written (your decisions, 2026-10-21 morning):**
  1. **D94:** E1c as built (the per-prompt mean stays the baseline; the std is taken over the step's batch) and E1d's
     learning rate 5e-5. D94's two [REVIEW] items are closed; the configs are unchanged.
  2. **D95:** A6c's min ratio 1.5 and identical displayed values in both members. The frozen set is unchanged.
  3. **D96:** the in-stage temperature guard. **Standing rule from now on:** CPU jobs run single-threaded
     (OMP / OPENBLAS / MKL = 1, n_jobs = 1) at nice 19 whenever the GPU is busy.
  4. **Self-consistency as the p baseline and teacher candidate:** recorded as its own entry, **D98**.
- **The queue (written and launched 06:02 IST, GPU deadline 14:00 IST):**
  1. **E1c train** (`grpo_e1c`) → its corrected evals (D67: `final_adapter` → `final_4bit`, `checkpoint-150` →
     `step150_4bit`, 4-bit base, adapter unmerged);
  2. **E1d train** (`grpo_e1d`) → its evals, **only if ≥ 3 h remain** before the deadline when E1c's evals end (your
     rule; otherwise logged SKIP and E1d counts as p = 1, as D94 says);
  3. **compare_pilots** (CPU, 1 thread, nice 19).
  - **Mechanics as D90/D96:** PID-based waits; the in-stage guard on every GPU stage (> 87 °C → the stage's process
    group stops for 10 min) plus the cool check before each; a failed stage is logged BLOCKED and the next
    independent stage goes on (E1d does not depend on E1c; an eval needs its adapter); commit + push after every
    stage under the shared flock (`.git/queue-commit.lock`, which my own commits also take).
  - **Time:** training stops cleanly 20 min before the deadline (two evals at ~2 min each, measured in D67), with a
    hard timeout 15 min later (the final save and fingerprint). E1a's own `wall_clock_limit_minutes: 240` still
    applies.
  - **Tapes:** `rollouts.jsonl` is gitignored and committed gzipped, as E1a's.
- **The allocator cap stays E1a's 7.5 GiB.**
  - **The WSL-spill rule would give** 8.0 − 0.39 (host use at launch) − 0.5 ≈ 7.1 GiB.
  - **What E1a did at 7.5:** its peak allocated memory was 5.63 GiB. After the step-250 checkpoint save, reserved
    memory went from 1.58 to 7.47 GiB, and steps went from ~25 s to ~80 s for the last 48 steps (most likely a
    spill). It still finished 300 steps in 2.71 h.
  - **Decision:** keep 7.5. It is the pre-registered config ("the 7.5 GiB cap"), the cap changes speed, not the
    computation, and the 8 h window holds two E1a-length runs plus evals.
  - **Alternative:** a 7.1 cap through a config change. It would likely avoid the slow end, but it edits a
    pre-registered config and adds a small OOM risk that would kill the run. Step times are logged
    (`memory.jsonl`, `train_metrics.jsonl`) and reported.
- **Two temperature pauses can stack:** the trainer's own callback (E1a's: every 2 steps, 600 s above 87 °C) and the
  in-stage guard are both active. That is harmless: the guard's pause cools the GPU, and the callback then passes.
- **While the queue runs:** no edit to any module or config the stages import (checked by importing the stages'
  entry points: `physcal.env.*`, `verify.*`, `tape`, `prompt`, `config`, `train.e1` / `grpo` / `e1_compare`,
  `eval.run_eval` / `fingerprint` / `posthoc` / `signals` / `stats` / `calibration`, and the lazily imported `base`,
  `adapters`, `train.sft_data`, `env.sampler`, `train.split_credit`). Today's CPU work goes into new files.

### D98. Self-consistency is the p baseline and the teacher candidate for Phase 12 stage E; X4's self-verification is ruled out as a teacher (your decision, 2026-10-21)
- **Background:**
  - **R3-4 E** deferred p (D83's `<confidence>` channel) to a follow-up: "if X4 works, verification-derived p becomes
    its teacher or baseline".
  - **X4 did not work (FINDINGS #21):** within problems, `sft` reached 0.524 [0.498, 0.550], not Holm-supported. The
    model says "yes" to 99.9% of its own answers, and across problems it is anti-informative (AUROC 0.469).
  - **Self-consistency (D92c, FINDINGS #22, post-hoc):** 0.643 [0.616, 0.671] within problems (0.861 on the
    separated pairs). That is the strongest within-problem signal so far, without training.
- **Decision:**
  1. **X4's verification is ruled out as a teacher for p.** It stays a reported baseline row wherever the X2 pairs
     are scored, labelled as failed.
  2. **Self-consistency is the p baseline.** Any p that stage E trains must be compared against it on the same
     answers: X2's within-problem pairwise accuracy and the across-problem AUROC, paired bootstrap over problems.
     The test and its Holm family are fixed in stage E's pre-registration.
  3. **Self-consistency is the teacher candidate.** The target for a trained p is the share of the problem's other
     samples that agree with the answer. Whether it is used as a distillation target, and how, is stage E's
     pre-registration; this entry fixes only that it is the candidate.
- **The definition carried forward:** D92c's. Confidence = the share of the problem's other k − 1 samples whose
  parsed μ agrees with this answer within 1% (|μ_j / μ_i − 1| ≤ 1%, SI); an unparsed sample never agrees; k = 8 as
  in X2. Code: `x2_robustness.self_consistency_conf`.
  - **For noisy problems,** R3-6's event (|ŷ − y_display| / |y_display| ≤ 1%) is the correctness label. The
    agreement rule itself does not change.
- **Costs and caveats, reported with every use:**
  - **Inference cost:** k = 8 samples per problem. A distilled p would remove it at inference, not in data generation.
  - **The number is post-hoc:** FINDINGS #22 was computed after D74's X2 verdict, as D92 labels it. Stage E's
    comparison is the first pre-registered test.
  - **Overlap with Phase 13:** the k-sample agreement is close to DAR-style rollout spread (Phase 13 source 2).
    The write-up must not count them as two independent sources.
- **Not done now:** nothing is built, trained or run. Stage E stays deferred (R3-4).

### D99. X1 on A6c (D95's deferred comparison, R3-5): defined before any X1 prediction on A6c exists
- **The model is X1 exactly as D73 fitted it.** Its fitted models were not saved, so they are refit:
  - the same fit set (regenerated from D73's seeds) and the same features;
  - the hyperparameters D73's grouped CV selected (`results/x1_text_sigma/selection.json`, best grid point per
    feature set). There is no new selection.
- **The reproduction check comes before any A6c prediction.** For every feature set, the refit must reproduce
  `results/x1_text_sigma/results.json` within 1e-9:
  - the test accuracy (R², median abs ln error, Spearman), and the A6 / A6b member accuracy;
  - the A6 and A6b primary pair slopes and their CIs.
  - **If it fails:** nothing on A6c is computed now. The refit reruns after the GPU queue under X1's own settings
    (4 workers, 4 BLAS/OpenMP threads), as D96 did for X2.
- **On A6c** (510 pairs, the manifest's SHA-256 checked):
  - **σ̂:** σ̂ = exp(ŷ) is each member's stated σ (log space).
  - **The analysis:** D95's, through `a6_pairs.analyze_rows`, as for the noise heuristics. X1 states no answer, so
    every pair enters. The primary is the OLS slope of Δ ln σ̂ on Δ ln σ\* with an intercept, a one-sided bootstrap
    over pairs (B = 10,000, seed 0), "supported" at α = 0.05. The secondaries are the mean Δ ln σ̂ > 0 and the
    through-origin slope.
  - **References, through the same analysis:** σ̂ = σ\* (slope 1), and the noise-only feature set.
  - **Breakdowns:** per depth and per level pair.
- **What follows from the construction (stated before computing):**
  - Within an A6c pair, the displayed values and the multiset of "± p%" are identical. X1's noise and magnitude
    features are therefore identical within a pair.
  - Only TF-IDF can differ: whitespace tokens carry attached punctuation, so a swapped "5%" may become "5%." in one
    member and "5%," in the other.
  - **So X1's within-pair Δ ln σ̂ is expected to be ≈ 0.** A nonzero slope would come from token-position artifacts,
    not from propagation.
- **Reported in one table** with the heuristics (D95) and σ\*: the A6, A6b and A6c pair statistics.
- **Settings:** 1 worker, 1 thread, nice 19 (D97's rule while the GPU is busy).
- **Outputs and code:**
  - **Outputs:** `results/x1_text_sigma/a6c/` with `reproduction.json`, `results.json`, `summary.md`, and
    `predictions.jsonl` (σ̂ for every A6 / A6b / A6c member, for stage A, D100).
  - **Code:** `physcal/eval/x1_a6c.py`, a new module. It imports `x1_text_sigma` and does not edit it.

### D100. Phase 12 stage A: the scoring audit and the oracle-centred diagnostic (R3-4 A, R3-6); defined before computing; descriptive, CPU only
- **[REVIEW]: my reading of "scoring audit".** R3-4 names it without a definition.
  - **My reading:** an audit of whether the scoring Phase 12 will use rewards σ propagation on noisy problems,
    before any RL night relies on it.
  - **Which scoring:** the verifier's crps_log realized against the noise-free truth, and the expected CRPS against
    the MC predictive distribution (R3's consensus point 6).
  - **No hypothesis test.** Every number here is descriptive. It informs Phase 12's pre-registration (the primary
    metric and the sample sizes).
- **The oracle-centred diagnostic (R3-6's primary for σ propagation)** is defined for any source of a stated
  relative σ on noisy problems, on all problems (no selection on correctness):
  - **The centre:** μ = f(x_meas), the stored `noise.point_estimate_si`.
  - **Realized score:** the verifier's crps_log of (μ, σ_rel·μ) at the truth: `scoring.score`, unchanged, clip
    included.
  - **Expected score:** the mean CRPS of N(ln μ, σ_rel²) over fresh MC draws of the predictive distribution
    (`noise.predictive`, N = 4,000, a per-problem seed of its own). This is not the generator's own stream, so it
    is an MC estimate, not exact.
  - **ln(σ / σ\*):** median and median absolute value.
  - **Vol beta:** the OLS slope of ln σ on ln σ\*.
  - **Coverage:** of ln truth by ln μ ± k·σ_rel, k = 1, 2, against 68.27% / 95.45%.
  - **Uncertainty:** a bootstrap over clusters (B = 2,000, seed 0). A cluster is a pair on the pair sets (R3-7)
    and a problem otherwise.
  - **Code:** `physcal/eval/oracle_centred.py`, a new module, reusable by Phase 12's evals, with tests.
- **The scoring audit asks four questions:**
  1. **The optimum.** With μ = f(x_meas), is the mean realized crps_log maximized at σ = σ\*?
     - **Data:** the capped gate sample (`results/sigma_star_gate_capped/rows.jsonl`): 10,005 independent noisy
       problems, depth 1–5 × {1, 2, 5}%. Its rows hold ln truth − ln f(x_meas) and σ\*.
     - **The method:** σ = c·σ\* on a log grid, c from 1/8 to 8 (241 points). Reported: the argmax c, pooled and
       per cell, with a bootstrap CI over problems, and the scores at c = 0.5, 1 and 2.
     - **What is already known:** by A2's construction the expected score peaks at c = 1. The realized peak is the
       check; it combines A3's calibration with the score.
  2. **The resolution.** Per problem, the paired score difference between σ\* and each alternative:
     - **The alternatives:** c·σ\* for c = 0.5 and 2; the heuristics (max, mean and RSS over the givens; D95); X1's
       three feature sets (from D99, only if its reproduction passes); and a constant σ = the set's median σ\*
       (an oracle constant).
     - **Reported, realized and expected:** the mean difference, its SD, and n₂ = (2·SD / mean)², the number of
       problems a mean difference needs to sit two standard errors from 0. Also the variance ratio
       Var(realized) / Var(expected).
     - **Data:** the frozen A6, A6b and A6c sets (3,060 noisy problems, checksums checked), pooled and per set.
       They carry the texts and givens that the heuristics and X1 need. Nothing is fitted on them.
  3. **The ranking.** Do the sources ranked by mean realized crps_log follow their ranking by median
     |ln(σ / σ\*)|? Both orders are reported, with Kendall's τ.
  4. **The verifier's details:**
     - the share of oracle-centred scores at the −5 clip, and the share where the 1e-6 floor binds;
     - an identity check: the diagnostic's realized score equals `reward()`'s `score_final` on a rendered answer,
       for a subset of problems (exact).
- **Plus one descriptive table: μ error against σ signal.**
  - **The method:** μ = f(x_meas)·(1 + δ) for δ ∈ {0, 0.1%, 0.5%, 1%, 5%}. Reported: the mean realized crps_log of
    σ\* and of the RSS heuristic, and their difference.
  - **Why:** it shows how quickly μ error swamps the σ signal, i.e. why R3-6 centres at f(x_meas).
- **Settings and output:**
  - **Settings:** 1 thread, nice 19 while the GPU is busy (D97).
  - **Output:** `results/stage_a/{results.json, summary.md}`; config `configs/stage_a.yaml`, a new file.

### D101. D94's decision rule applied: E1d's learning rate (5e-5) is Phase 12's RL setting; E1c's batch scaling is not adopted (2026-10-21, after compare_pilots; FINDINGS #25)
- **The outcome (`results/e1/pilots/`):**
  - **E1d:** ρ 0.586 vs E1a's 0.013, Δρ +0.573 [+0.465, +0.680], p < 1e-4. Holm-supported.
  - **E1c:** ρ 0.002, Δρ −0.011 [−0.092, +0.069], p = 0.62. Not supported.
  - **The rule:** exactly one pilot is supported, so its setting becomes Phase 12's RL setting. Neither run is
    collapsed (top suffix 1.8% / 4.8%).
- **Applied:**
  - **Phase 12 RL** uses learning rate 5e-5 (E1d's), with E1's other settings, `scale_rewards: group` included.
  - **No config is edited now.** Phase 12's RL configs do not exist yet; they get the value when they are written.
- **[REVIEW] for Phase 12's pre-registration (not decided here):**
  1. **The 5e-5 was validated only on σ-only suffix training** (24 tokens, μ fixed).
     - **The risk:** Phase 12's stage C trains whole completions, and a 10× learning rate on reasoning tokens may
       destabilize μ.
     - **The options:** run stage C's pilot at 5e-5 as the rule says, with a μ-accuracy guard; or add a 1e-5 arm.
  2. **E1d is still improving at step 300** (ρ 0.27 at step 150). The step budget for Phase 12 may need to grow.
  3. **E1d's σ is badly overconfident:** 1σ coverage 0.30, median relative σ 0.95%. A post-hoc scale or a
     calibration term belongs in Phase 12's plan.
  4. **One seed.** These were pilots (D94). The R3-7 wording applies.

## Night of 2026-10-21/22 (unattended; your instructions at 23:10 IST, GPU deadline 09:00 IST)

### D102. Pre-registration of the E1d within-problem test on the X2 pairs (written before any E1d suffix on an X2 answer exists)
- **Question:** E1d's σ ranks errors between problems (ρ 0.586, FINDINGS #25). Does it rank a correct answer above an
  incorrect answer **to the same problem**? X3's supervised student (a) does not (0.513 [0.484, 0.543], FINDINGS #20);
  self-consistency does (0.643, post-hoc, FINDINGS #22).
- **Answers:** X2's 516 kept problems and their 4,116 sampled answers (D74), each truncated after "±" by E1's
  `rows_from_records` (prompt v4), exactly as X2's method 2 and X3's X2 eval. No new selection.
- **σ source:** E1d's `final_adapter` (step 300, FINDINGS #25's adapter).
  - **Loading:** the merged SFT model loaded 4-bit by Unsloth, the adapter unmerged (D67 / D68), with E1d's own loader
    settings from its tape's `train_config`. R2-1's check as in every E1 eval (D77).
  - **Decoding:** E1's eval, a greedy σ suffix of ≤ 24 tokens. Batch 16 and the 5.75 GiB allocator cap (D87's X2
    suffix settings: OOM + backoff instead of WSL spill).
- **Confidence:** −ln max(σ_rel, 1e-6), as X2's method 2. A failed suffix (no parse, unit mismatch, μ changed, μ = 0;
  D88) is a tie.
- **Metric:** D74's pairwise accuracy: per problem the share of its (correct, incorrect) pairs where the correct
  answer is more confident (ties 0.5), mean over problems.
  - **Uncertainty:** a problem-level cluster bootstrap (B = 10,000, seed 0): problems are resampled with all their
    pairs (R3-7). Paired differences resample the same problems for both methods.
- **The comparators (scored on the same pairs):**
  - **SFT stated σ:** X2's method 1, the σ in each sampled completion (published 0.499).
  - **X3 student (a):** its σ suffix on the same 4,116 prefixes (`results/x3/a/eval_x2`, published 0.513).
  - **Self-consistency:** D92c / D98's definition, the share of the problem's other 7 samples agreeing within 1%
    (published 0.643).
  - **Reproduction first:** each comparator is recomputed from its rows and must reproduce its published pairwise
    accuracy within 1e-9, or the analysis stops.
- **Tests (family E1d-X2, Holm at α = 0.05 over three, one-sided bootstrap p):**
  - **T1:** E1d's pairwise accuracy > 0.5 (p = the share of resamples ≤ 0.5).
  - **T2:** E1d − SFT stated σ > 0 (p = the share of paired resamples ≤ 0).
  - **T3:** E1d − X3 student (a) > 0.
  - **Not tested:** E1d − self-consistency, reported with its paired CI. Self-consistency is a post-hoc baseline
    (D92) that costs 8 samples per problem; its first test is stage E's (D98).
- **Reading rules (fixed now):**
  - **T1 not supported:** E1d's ρ is between-problem ranking only, the same reading as X3 (FINDINGS #20, #22). RL
    at 5e-5 learned the X3 kind of σ, not answer-specific awareness.
  - **T1 and T3 supported:** RL produced within-problem information that supervised distillation of the A5 teacher
    did not. Reported next to self-consistency's level.
  - **T1 supported, T3 not:** a within-problem signal that is not distinguishable from X3's.
  - **Any other combination** is reported as observed.
- **Secondary control (reported, no test): the SFT model's own greedy σ suffix on the same prefixes** (no adapter;
  the same loader, decoding and cap). It is D67's family-E "SFT stated σ" control, the exact no-RL counterpart of
  E1d's greedy suffix (method 1 is sampled). Reported: its pairwise accuracy and E1d − it, paired. It runs after
  E1d's stage (~18 min) and does not block anything.
- **Reported, not tested (E1d and every comparator):**
  - D92's pooled pair-weighted accuracy (cluster bootstrap) and the separated pairs (correct ≤ 0.25%, incorrect
    ≥ 10%); per depth;
  - the tie / failure share; the collapse check (one σ string in > 90% of suffixes → "collapsed", whatever the
    number); μ-changed and parse rates;
  - across problems, on all 4,116 answers: Spearman ρ(σ_rel, |ln(μ / truth)|) and the AUROC of the confidence for
    "correct", both with the cluster bootstrap.
- **Pipeline sanity:** more than 10% failed suffixes is reported as a pipeline problem next to the number; the test
  is still computed as pre-registered (failures are ties).
- **Code and outputs:** `physcal/eval/e1d_x2.py` (GPU: the suffixes), `physcal/eval/e1d_x2_analysis.py` (CPU),
  `configs/e1d_x2.yaml`, `tests/test_e1d_x2.py`; outputs in `results/e1d_x2/` (`e1d_final/`, `sft_suffix/`,
  `results.json`, `summary.md`).
- **Queue:** `scripts/queue_e1d_x2.sh` (GPU_DEADLINE 09:00): E1d suffixes → SFT suffixes, each with the in-stage
  temperature guard, a PID wait, and commit + push after the stage. The CPU analysis runs when both exist, at
  1 thread and nice 19 (D97).

### D103. Your decisions on the 2026-10-21 day report (recorded 2026-10-21 night)
1. **D100 confirmed:** the scoring audit is "does the scoring Phase 12 will use reward σ propagation on noisy
   problems". The [REVIEW] mark on D100's reading is closed; FINDINGS #24 stands as stage A's audit.
2. **D101 refined: stage C trains σ E1d-style.** The answer is frozen (prefixes truncated after "±", as E1), only σ
   is trained by RL, at learning rate 5e-5, on noisy problems. D101's [REVIEW] item 1 (a 10× learning rate on whole
   completions may destabilize μ) is thereby resolved: μ is never trained in stage C. Items 2–4 (step budget,
   overconfidence, one seed) stay open for stage C's pre-registration.
3. **A 7.1 GiB allocator cap for Phase 12 runs, after an OOM check.**
   - **The value:** the WSL-spill rule (D86/D87), total − host − 0.5 GiB, with the day's host use (0.39 GiB).
   - **Tonight's host use:** 0.59 GiB (601 MiB with nothing running, 23:14 IST), so 7.1 leaves a 0.3 GiB margin
     instead of 0.5. Kept at 7.1 as decided: stage B's SFT peaked at 4.7 GiB device-wide in Phase 3.6 (the same base
     and LoRA), far below either value.
   - **The OOM check (how it is applied):** every Phase 12 training run starts with one forward + backward pass on
     its longest batch under the cap. An OOM there stops the run before step 1 (logged BLOCKED), instead of failing
     or spilling hours in. Evals keep their OOM backoff (the batch halves).
   - **Does not apply to** E1-machinery or X-series stages (their caps are pre-registered), only to Phase 12 runs.

### D104. Pre-registration of Phase 12 stage B: the supervised σ\* ceiling, 3 seeds (written before any stage-B model, completion or score exists)
- **Question (R3-4 B):** if SFT shows the model σ\* itself on noisy problems, does its stated σ propagate the
  measurement noise through the chain on new problems? This is the ceiling stage C's σ-RL is read against.
- **Data: D80's supervised arm,** unchanged (`results/phase12_sft_data/supervised.jsonl`, the manifest's SHA-256
  checked before training).
  - 3,000 generator-2.0.0 training problems (seeds 6e7+), noise {0, 1, 2, 5}%; the answer is f(x_meas); σ\* written
    after "±" and **trained** (`.2g`); the `<confidence>` block (uniform c, trained; D80) as built.
  - **Prompt:** `v5_two_channel` with D85's placeholder. Rows longer than the loader's 2,048 tokens are dropped and
    counted.
- **Model (D70 / D78, your decisions):** a fresh LoRA (r 16, α 16, dropout 0, the 7 modules) on the **pinned 4-bit
  base** (`configs/base_4bit.yaml`, `unsloth/Qwen2.5-Math-1.5B-Instruct-bnb-4bit@e24e953`, content hash
  `9939448b3a0fbe45`), checked before and after loading. Never merged; the eval applies the adapter unmerged. The
  R2-1 fingerprint is recorded after the save (D72 / D77).
  - **[REVIEW] the base.** D70 / D78 make Phase 12 SFT one adapter on the pinned base, which RL continues. The
    alternative is the merged SFT model (E1d's and X3's base); it would make stage B and an E1d-style stage C share a
    base exactly. I followed D70 / D78; stage C's pre-registration must state which base it uses and how the two
    are compared.
- **Training (identical across seeds except the seed):** `configs/sft.yaml`'s optimizer settings (AdamW 8-bit, LR 1e-4
  constant after 10 warmup steps, batch 2 × 4 accumulation, grad-norm 1.0, weight decay 0, bf16).
  - **[REVIEW] the budget: 2 epochs (750 optimizer steps), X3's σ-learning budget.** The Phase 3.6 SFT ran 0.87
    epoch on noise-free data; stage B starts from the base, and σ is a few tokens per example.
  - **Seeds 0, 1, 2:** the seed sets the LoRA init, the shuffle and the trainer's seed.
  - **D103's cap (7.1 GiB) and OOM check;** the trainer's temperature callback (87 °C → 600 s) plus the in-stage
    guard.
- **Eval (per seed):** greedy completions (512 tokens, prompt v5) on the frozen A6, A6b and A6c sets (3,060 problems;
  `run_eval` regenerates each problem and checks it against the file), 4-bit pinned base + adapter unmerged, batch 32
  with the 7.1 GiB cap (OOM → the batch halves). Scored by the verifier 1.1.0 path (`run_eval`'s default; the
  `<confidence>` block is ignored by it, D83).
- **The model's σ for the analyses:** the stated relative σ = σ / |μ| of the parsed answer (crps_log's scale).
  - **Pair statistics (D59 / D95):** `a6_pairs.member_row`'s conventions (σ = 0 → the 1e-6 floor; a parse failure or
    μ ≤ 0 → the member is unanswered).
  - **The oracle-centred score (D100):** the same σ_rel scored at μ = f(x_meas). An unanswered problem gets σ_rel = 0
    (the floor), the verifier's convention for no stated spread, and is counted.
- **The comparators, through the same code:** X1 primary (D73's model as D99 refit it,
  `results/x1_text_sigma/a6c/predictions.jsonl`), the RSS heuristic over the givens (D95's primary scope), and
  σ̂ = σ\* (the ideal). They state no answer, so for the pair statistics they enter on the same pairs as the model.
- **The seed statistic (R2-4 / R3-7):** each primary statistic is the **mean over the 3 seeds** of the per-seed
  estimate. One cluster bootstrap over pairs (B = 10,000, seed 0) resamples the same pairs for all seeds;
  p = the one-sided share of resamples ≤ 0. This is pooled problem-level evidence, never described as seed-level
  significance. Every per-seed estimate is reported with its own CI.
- **Tests (family B, Holm at α = 0.05 over five; a seed that did not run is left out of the mean, and a test with no
  seed counts as p = 1):**
  - **B1, A6 (D59 / D61):** the slope of Δ ln σ on Δ ln σ\* over the computation-correct pairs (A1, τ from
    `configs/pairs.yaml`) > 0.
  - **B2, A6c (D95):** the slope over all answered pairs > 0.
  - **B3, expected score vs RSS (FINDINGS #24):** the mean per-problem difference in oracle-centred **expected**
    crps_log (model − RSS) over A6 + A6b + A6c > 0. The expected score uses fresh MC draws of the predictive
    distribution (N = 4,000, stage A's per-problem streams).
  - **B4, expected score vs X1 primary:** as B3.
  - **B5, A6 vs X1:** model slope − X1 slope on the model's computation-correct pairs > 0. (RSS's slope is 0 by
    construction, so B1 is already "beats RSS on A6"; on A6c X1's slope is −0.002, so B2 ≈ "beats X1 on A6c".)
- **Labels and verdicts (reported per seed and on the seed mean; not in the Holm family):**
  - **A6's label (D61 / D64)** from the 90% CI of the slope;
  - **A6b's TOST (D59)** with `pairs.yaml`'s margin, and D59's joint verdict;
  - **A6c's secondaries (D95):** computation-correct pairs, mean Δ ln σ > 0, the through-origin slope;
  - **R3-6's σ secondary:** A6 and A6c restricted to pairs computation-correct at τ_σ = max(0.25%, 0.25·σ\*).
- **Reported, not tested:**
  - the oracle-centred diagnostic (D100) per seed and source: realized and expected scores, median ln(σ / σ\*) and
    |ln(σ / σ\*)|, the vol beta, 1σ / 2σ coverage, the clip and floor shares; per set;
  - training loss and steps; parse-fail and unit-mismatch rates; the A1 computation-correct rate; σ = 0 share;
    the `<confidence>` block's parse rate (format only).
- **Reading rules (fixed now):**
  - **"The ceiling propagates":** B1 and B2 Holm-supported, A6b equivalent to 0 on the seed mean, and an A6 label of
    partial, full or over-propagation.
  - **"The ceiling beats the surface readers":** B3 and B4 supported.
  - **B2 not supported:** supervised σ\* does not teach noise allocation with this base and budget. Stage C's bar
    is then X1 / RSS, not stage B.
  - **Any other combination** is reported as observed.
- **What stage B does not test:** RL, and whether the model "knows" its σ beyond imitation. It is a capacity and
  transfer ceiling (as X3 was for between-problem ranking).
- **Code and outputs:** `physcal/train/stage_b.py` (train / eval), `physcal/eval/stage_b_analysis.py`,
  `configs/stage_b.yaml`, `tests/test_stage_b.py`; outputs in `results/stage_b/seed{0,1,2}/` and
  `results/stage_b/{results.json, summary.md}`.
- **Queue:** `scripts/queue_stage_b.sh`, chained after stage (a)'s queue by PID: seed 0 → 1 → 2, each training run
  followed by its eval. A seed starts only if its training (2 epochs at the measured speed, else 75 min) plus its eval
  (70 min) plus 15 min of slack fit before 09:00 IST; otherwise it is logged SKIP.

### D105. Verifier 1.3.0: the σ reward against the generator's predictive distribution, centred at f(x_meas) (your approval, 2026-10-21 night; built and tested, nothing trained with it)
- **Your instruction:** a new reward config version whose σ reward is the expected crps_log against the generator's
  predictive distribution, centred at f(x_meas); old tapes must replay unchanged; tests. FINDINGS #24 motivates it:
  one realized truth barely separates σ\* from a 2×-off σ, the expected score does, with 5–23× less variance.
- **The score (`physcal/verify/sigma_reward.py`, `configs/reward_v13.yaml`, version 1.3.0):**
  - **score_sigma** = the mean, over fixed draws y_j of the problem's predictive distribution of ln(target), of the
    verifier's crps_log of N(ln f(x_meas), σ_rel²) at y_j, with σ_rel = σ / |μ| as stated.
  - **It depends on σ_rel and the problem only.** μ's error does not enter (R3-6's oracle centring), so the σ tokens
    are credited without μ's noise or the realized truth's.
  - **Proper:** an average of a proper score under the predictive distribution, so it peaks at that distribution's
    CRPS-optimal scale (A2's σ\*). Tested on a 5% problem: the grid 0.5…2 × the optimum peaks at 1.
  - **The draws:** N = 4,000 (stage A's N), from the problem's own stream, seeded by
    sha256("verifier_1.3.0|family|depth|seed"). Not stage A's tag, so training rewards and the stage A / B eval
    diagnostics never share draws. Cached per problem; deterministic, so 1.3.0 tapes replay exactly (tested).
  - **Invalid draws:** a draw that leaves the valid domain is dropped and counted (`noise.predictive` instead
    returns None for the whole set; that happened on 24 of stage A's 3,060 problems). Below 99% valid draws, the
    draws become N(ln f(x_meas), σ\*²) on the same stream, flagged `gaussian_fallback`. With every draw valid, the
    masked draws equal `noise.predictive`'s exactly (tested).
  - **Exact givens:** the predictive is the truth itself (one draw), so σ_rel = 0 is optimal (D80, R2-3).
  - **Failures:** a parse failure, a unit mismatch, or no relative σ (μ = 0) gets the crps_log floor, −5.
- **[REVIEW] `weights.sigma = 0`.** The 1.3.0 total equals 1.2.0's; score_sigma is in the breakdown for training
  code that credits the σ tokens with it, as D83 did for c.
  - **Stage C (E1d-style, μ frozen)** would use score_sigma as its σ-only reward. That belongs in its own
    pre-registration (D106).
  - **A positive weight next to final = 1** would count σ twice (realized and expected). The validator allows it
    only while steps + confidence + σ·span < final·span, the "never outweighs the answer" rule.
- **No existing module was edited** (tonight's stages import `reward.py`, `tape.py`, `noise.py`; D97's rule).
  - **The entry point:** `sigma_reward.reward(problem, completion, cfg)`. It calls the unchanged `reward.reward` and
    adds the σ term only for configs ≥ 1.3.0. Older configs pass straight through (tested on four completion
    kinds × 1.1.0 / 1.2.0).
  - **Old tapes:** the two committed 1.1.0 tapes D83 used (400 + 400 rows) replay through the new entry point with
    identical rewards **and** breakdowns.
  - **Until `reward.reward` and `tape.replay` dispatch on the version** (a small daytime edit, after the queue),
    1.3.0 configs must be scored and replayed through `sigma_reward` (`reward`, `replay`). `reward.reward` called
    directly with a 1.3.0 config omits the σ term; `tape.replay` would then flag every 1.3.0 row with w_sigma > 0
    as a mismatch, a loud failure, never a silent one. With w_sigma = 0 the totals agree.
- **Tests (`tests/test_verify_sigma_reward.py`, 17):** the versions and the unchanged default; six invalid configs
  refused; pass-through for 1.1.0 / 1.2.0; both 1.1.0 tapes replay exactly; 1.3.0's total = 1.2.0's at w_sigma = 0
  and adds w·score_sigma otherwise; score_sigma ignores μ's error and equals stage A's expected score on
  `noise.predictive`'s draws; the peak at the optimum; exact givens; failures; dropped draws and the fallback; a
  1.3.0 tape replays exactly.

### D106. Pre-registration of the Phase 12 stage C pilot: E1d-style σ-only RL with verifier 1.3.0's σ reward (written before any stage C prefix, rollout or score exists; built and CPU-tested tonight; runs tonight only if ≥ 3.5 h remain after stage B)
- **Question (R3-4 C, D103):** with the answer frozen and the reward on σ alone, does GRPO at learning rate 5e-5,
  rewarded by the expected crps_log against the predictive distribution (D105), learn a σ that propagates the
  measurement noise (A6 / A6b / A6c), beyond the surface readers and beyond its own SFT-only start?
- **Base [REVIEW]: E1d's, exactly** (D101 / D103 "E1d-style"): the merged SFT model (content hash
  `0b435c0ebaed36e2`) loaded 4-bit by Unsloth, plus a fresh LoRA (r 16, α 16, dropout 0, the 7 modules, seed 0).
  - **Not D70 / D78's pinned base,** which stage B uses (D104). So stage B and stage C differ in base. Stage C's
    within-stage control is its SFT-only start (below); its comparison with stage B is descriptive only.
  - **The alternative:** the pinned base with an implicit-arm SFT adapter that RL continues (D70). It adds an SFT
    run (~1.2 h) and is untested on the GPU with GRPO.
- **Prompt [REVIEW]:** `v5_two_channel` (it defines σ as the propagated measurement uncertainty). The merged SFT
  model was trained on v4; v5 is new to it, as it is to every Phase 12 model.
- **Training prefixes:** D80's implicit-arm completions (generator-written; the answer is f(x_meas) at 4 s.f.) of the
  2,250 noisy problems (noise 1, 2, 5%), truncated right after "±" (E1's `truncate_after_pm`). The manifest's
  SHA-256 is checked first. Shuffled once (seed 0); 300 steps × 4 prefixes = 1,200 prefixes (53% of an epoch).
- **RL: E1d's settings unchanged** (`configs/grpo_e1d.yaml`): learning rate 5e-5 constant after 5 warmup steps,
  AdamW 8-bit, grad-norm 0.1, β = 0, `loss_type` dapo, `scale_rewards` group, 8 samples per prefix, 2 × 16
  accumulation (4 prefixes per step), ≤ 24 new tokens, T = 1, top-p 1, top-k off.
  - **Changed (one bundle, stated):** the prefixes and the prompt (above), the reward (below), and D103's 7.1 GiB
    cap with its OOM check (a forward + backward on the longest prompt before step 1) instead of E1a's 7.5.
- **The training reward:** score_sigma of verifier 1.3.0 (`sigma_reward.reward`, `reward_v13`). The floor (−5)
  goes to a suffix that fails to parse, mismatches the unit, or moves μ (E1's μ-changed rule: a same-dimension unit
  change moves μ in SI, which score_sigma alone would not see).
  - **The tape** logs the 1.3.0 verifier reward and breakdown (score_sigma included), replayable with
    `sigma_reward.replay`, plus the training reward and the suffix.
- **Eval (μ fixed):** a greedy σ suffix (24 tokens, E1's eval) on prefixes of the frozen A6, A6b and A6c problems
  (3,060), built exactly like the training prefixes: each problem's implicit-arm completion (answer f(x_meas)),
  truncated after "±".
  - **Three arms on the same prefixes:** stage C's final adapter; its step-150 checkpoint (secondary); and the
    SFT-only control (the merged SFT, no adapter).
  - **Loading:** 4-bit, adapter unmerged (D67 / D68), batch 16, the 7.1 GiB cap. A failed suffix (parse, unit, μ
    changed) leaves the member unanswered: σ_rel = 0 in the oracle-centred score, as D104.
- **Tests (family C, Holm at α = 0.05 over six; the final adapter):**
  - **C1–C5:** D104's B1–B5 with stage C's σ in place of stage B's (one seed): A6 slope > 0 on computation-correct
    pairs; A6c slope > 0 on all answered pairs; expected crps_log − RSS > 0; − X1 > 0; A6 slope − X1's > 0. Same code
    (`stage_b_analysis`), same bootstrap over pairs.
  - **C6, the RL effect:** the mean per-problem expected crps_log, stage C − its SFT-only control, > 0 (paired,
    bootstrap over pairs).
- **Labels and verdicts (reported, as D104):** A6's label, A6b's TOST and D59's joint verdict, A6c's secondaries,
  R3-6's σ secondary, the oracle-centred diagnostic for all three arms.
- **Reading rules (fixed now):**
  - **"σ-RL propagates":** C1 and C2 supported, A6b equivalent to 0, and an A6 label of partial, full or
    over-propagation.
  - **"RL moved σ toward the predictive distribution":** C6 supported.
  - **C6 supported, C2 not:** RL improved the score without learning allocation (e.g. a better global scale).
  - **Stage B (D104), descriptive:** stage C's numbers sit next to stage B's seed mean, with the base caveat.
  - **One seed:** a pilot. "3 seeds only if it shows signal" (R3-4 C): signal = C6 or C2 supported.
- **Sanity (reported, not tested):** the collapse check (one σ string in > 90% of eval suffixes); μ-changed and
  parse rates; > 10% failed suffixes at step 1 is a pipeline failure, not a result; per-step training reward,
  σ = 0 share, zero-variance groups.
- **Budget:** about 2.1 h of training at E1d's speed (25 s/step) and 3 evals of ~12 min (3,060 suffixes at E1d's
  X2 rate of 4.4/s). The queue starts it only if ≥ 3.5 h remain before the deadline (your rule).
- **Code and outputs:** `physcal/train/stage_c.py` (data / train / eval), `physcal/eval/stage_c_analysis.py`,
  `configs/stage_c.yaml`, `tests/test_stage_c.py`; outputs in `results/stage_c/`. The queue hook is
  `scripts/queue_stage_c_pilot.sh`, run by stage B's chain only if `results/queue_stage_b/STAGE_C_READY` exists.

### D107. Tonight's run, 2026-10-21/22: implementation decisions taken during the night (overnight rules: decision, alternatives, why)
1. **Stage (a) launched first, as its own queue; stage B chained by PID.**
   - **Order:** D102 written and committed, then `scripts/queue_e1d_x2.sh` launched at 23:14 IST. Stage B was built
     while (a) ran, and `scripts/queue_stage_b.sh` started at 23:25, waiting on (a)'s queue PID (as D96).
   - **Why:** "get the GPU busy first". The GPU went from (a) to stage B seed 0 without idling (23:47).
2. **The SFT model's own greedy σ suffix as (a)'s no-RL control** (pre-registered in D102, ~15 min). X2's method 1
   is a sampled σ, so it is not the exact counterpart of E1d's greedy suffix. Reported, not tested.
3. **D103's OOM check:** a forward + backward on the longest batch under the cap before step 1 (stage B: 2 × 1,640
   tokens, 2.0 GiB reserved, passed). Alternative: run step 1 and see. Rejected because step 1's batch is not
   necessarily the longest.
4. **Stage B's eval batch is 32 distinct prompts,** under the 7.1 GiB cap with the OOM backoff. D87's 16 was for
   a 5.75 cap and a WSL spill; 32 × ~1,700 tokens fits the 7.1 cap by the KV-cache estimate (~1.7 GiB).
5. **Verifier 1.3.0 lives in a new module** (`physcal/verify/sigma_reward.py`). Editing `reward.py` / `tape.py` would
   break D97's rule (tonight's stages import them). The dispatch edit is left for the day (D105).
6. **Stage C amendment, before any rollout: `max_prompt_length` 1,536 → 2,016** (`configs/stage_c.yaml`
   `grpo_overrides`).
   - **Why:** v5 prompts plus prefixes reach 1,620 tokens (training) and 1,669 (eval), so E1d's 1,536 would cut
     them.
   - **Effect:** none on the computation. 2,016 + 24 new tokens fits E1d's `max_seq_length` 2,048, and nothing is
     truncated, as 1,536 truncated nothing for E1.
7. **Stage C marked ready after its CPU tests** (`results/queue_stage_b/STAGE_C_READY`). Your 3.5 h gate decides; with
   stage B's three seeds at ~2 h each it is expected to be skipped tonight. Its GPU path is untested.
8. **My own commits take the queue's flock** (`.git/queue-commit.lock`), as D97 says.
9. **Stage B's eval loader reports `tokenizer_matches.backend: False`** (every earlier eval on the merged SFT base:
   True). Checked on CPU while seed 0's eval ran:
   - **The training tokenizer** (saved with the adapter by Unsloth) and **the eval's** (the pinned dir's) differ
     only in the post-processor: `TemplateProcessing` with no special tokens vs `ByteLevel`. Neither adds a token.
   - **The vocab, added tokens, chat template, pad (151665) and eos (151645) are equal.** Encodings are identical on
     300 prompt + completion texts of the training data.
   - **Decision:** benign; the eval goes on. Alternative: stop and rebuild the eval tokenizer from the adapter dir.
     Rejected, since nothing it encodes differs.

## Night of 2026-10-22/23 (unattended; your instructions at ~23:25 IST, GPU deadline 09:00 IST)

### D108. Your decisions on the 2026-10-21/22 report (MORNING.md §6), and tonight's launch (recorded 2026-10-22, 23:45 IST)
1. **Stage B is not extended** (D104's [REVIEW] on base and budget). Stage D's positive control is the ceiling test
   (D109).
2. **Stage C's pilot base is approved as a pilot** (D106's [REVIEW]): the merged SFT model, prompt v5, as E1d.
3. **`weights.sigma = 0` is approved** as verifier 1.3.0's default (D105's [REVIEW]).
4. **The verifier hookup (the version dispatch in `reward.reward` / `tape.replay` for 1.3.0) is approved for
   tomorrow only.** Tonight no module a running stage imports is edited. Stage D is new files only
   (`physcal/train/stage_d.py`, `physcal/eval/stage_d_analysis.py`, `configs/stage_d.yaml`, `tests/test_stage_d.py`,
   `scripts/queue_stage_d.sh`); it imports `stage_b`, `stage_b_analysis`, `sft_phase12` and the verifier unchanged.
5. **The stage C pilot was launched at 23:28 IST** (`GPU_DEADLINE=09:00 bash scripts/queue_stage_c_pilot.sh`, queue PID
   6944, training PID 6967 under the in-stage guard).
   - **Host VRAM before launch:** 1,602 MiB, then 1,090, then 622–637 MiB over 15 s, with nothing running in WSL
     (a Windows-side spike). D103's 7.1 GiB cap was kept as pre-registered: the steady host use matches last night's
     0.59 GiB, for which the 0.3 GiB margin was accepted. Alternative: lower the cap to ~5.9 GiB for the 1.6 GiB
     reading. Rejected, because E1d's allocated peak was 5.63 GiB, so ~5.9 risked an OOM on every step for a spike
     that had passed.
   - **The OOM check passed:** 2 × 1,644 tokens, 2.0 GiB reserved. The first training step reserved 6.05 GiB (5.60
     allocated), 7.08 GiB device-wide.
   - **Speed:** steps 1–2 took ~48 s, then ~29–31 s (E1d: 25 s). The first steps include the score_sigma predictive
     cache's warm-up. 300 steps at ~30 s end ~02:05 IST.
   - **σ = 0 share (Phase 9.5's monitoring rule, > 50% alerts):** 0.52–0.97 over steps 1–7, then 0.03–0.69 per step
     through step 20, with 0–75% zero-variance groups. This is logged as an alert for the morning, not acted on (the
     reward is the pre-registered score_sigma; D106's sanity rule is about parse failures, and the parse rate is 1.0).
6. **Stage D (D109) is chained after stage C's queue by PID:** pos, then fmt. Each arm starts only if its training,
   eval and slack fit before 09:00.

### D109. Pre-registration of Phase 12 stage D: explicit propagation, the positive control and the format-only ablation, 1 seed per arm (written before any stage D model, completion or score exists; built and CPU-tested tonight)
- **Question (R3-4 D, R3-9; your decision D108.1, "stage D's positive control is the ceiling test"):**
  - **D-pos (positive control):** if SFT teaches the model to write the first-order propagation line with its numbers
    (log-sensitivities, levels, result X) and to copy X·|μ| into the answer's σ, does the stated σ propagate the
    measurement noise on new problems (A6 / A6b / A6c)? Does it beat the surface readers, stage B's supervised σ\*,
    and the format alone?
  - **D-fmt (ablation):** the same line with its numbers masked (D84's data). What does the format alone do? It is
    also the SFT-only start that a later explicit-arm RL would continue from (R3-9).
- **Data: D80's explicit arm as built under D84** (`results/phase12_sft_data/explicit.jsonl`, SHA-256 `bc9861ea…`, and
  `problems.jsonl`, both checked against the manifest).
  - **The problems:** the same 3,000 as stage B (noise 0 / 1 / 2 / 5%; the answer f(x_meas) at 4 s.f.).
  - **The line, at the end of `<steps>`:** `sigma_rel = sqrt((a1*s1)^2 + …) = X`, with a_i = |d ln f / d ln x_i| at
    the displayed values (`.2g`), s_i the stated levels, the noisy givens in problem order, and X computed from the
    shown terms (`.2g`). Exact givens give `sigma_rel = 0`.
  - **The answer:** σ = X·|f(x_meas)| at `.2g`; the `<confidence>` block (uniform c, trained; D80) as built.
  - **The arms differ only in the loss mask; the text is identical.**
    - **D-pos:** nothing in the σ channel is masked. The line's numbers and the answer's σ are trained
      (`train_line_numbers: true`, `train_sigma: true`: D80's trained variant; R3-9's "SFT teaches the copy step").
      The rows' recorded D84 masks are replaced in memory (all 3,000 rows); the file is not rebuilt.
    - **D-fmt:** D84's masks exactly (every number on the line and the answer's σ masked; `sqrt`, the brackets, `*`,
      `^2` and `=` trained), the rows' recorded masks, checked equal (7.2 spans, 24.2 tokens per row).
  - **Tokenized (prompt v5, the pinned base's tokenizer):** 3,000 rows each, none dropped; sequences ≤ 1,719 tokens
    (mean 1,420; stage B's 1,380). The longest training completion is 460 tokens, under the eval's 512.
- **Model and training: stage B seed 0's exactly, except the data** (`configs/stage_d.yaml`; tested equal to
  `configs/stage_b.yaml`).
  - **The setup:** the pinned 4-bit base (D70 / D78), checked before and after loading; a fresh LoRA (r 16, α 16,
    dropout 0, the 7 modules); seed 0 (LoRA init, shuffle, trainer).
  - **The optimizer and budget:** 2 epochs (750 steps), `configs/sft.yaml`'s optimizer.
  - **The guards:** D103's 7.1 GiB cap and OOM check, the temperature callback and the in-stage guard, the R2-1
    fingerprint after the save.
  - **[REVIEW] the base and budget are stage B's.** So D-pos vs stage B seed 0 changes one thing: how the σ channel is
    supervised (an explicit computation plus a copy, vs the σ\* number). Stage C uses another base (the merged SFT,
    D106), so D vs C is descriptive.
- **Eval: stage B's exactly** (`stage_b.eval_config` with the arm's adapter):
  - greedy, 512 tokens, prompt v5, on the frozen A6 / A6b / A6c sets (3,060);
  - the pinned base with the adapter unmerged, batch 32, the 7.1 GiB cap with OOM backoff;
  - scored by the verifier 1.1.0 path. The line is an unrequested step and does not affect scoring.
- **The model's σ (as D104):**
  - **The σ tested** is the answer's stated σ_rel = σ / |μ|.
  - **Pair statistics:** `a6_pairs.member_row`'s conventions.
  - **The oracle-centred score:** σ_rel scored at μ = f(x_meas). An unanswered problem gets the floor.
- **Tests (bootstrap over pairs, B = 10,000, seed 0; one seed, so problem-level evidence only):**
  - **Family D, the positive control (Holm at α = 0.05 over seven):**
    - **D1–D5:** D104's B1–B5 on D-pos's σ, through stage B's code unchanged (`stage_b_analysis.analyze`):
      - A6 slope > 0 on computation-correct pairs;
      - A6c slope > 0 on all answered pairs;
      - expected crps_log − RSS > 0, and − X1 > 0;
      - A6 slope − X1's > 0.
    - **D6, vs stage B:** the mean per-problem expected crps_log, D-pos − stage B > 0. Stage B's per-problem score is
      the mean over its 3 seeds, held fixed in the bootstrap.
    - **D7, the numbers' training effect:** expected crps_log, D-pos − D-fmt > 0 (paired).
  - **Family F, the ablation (Holm over six):** F1–F5 = B1–B5 on D-fmt's σ; F6 = D-fmt − stage B > 0.
  - **[REVIEW] two families instead of one Holm over 13.** The positive control's family then does not depend on
    whether D-fmt runs tonight.
  - **An arm that did not run:** its tests count as p = 1 (D104's rule).
- **Labels and verdicts (reported per arm, as D104):** A6's label, A6b's TOST and D59's joint verdict, A6c's
  secondaries, R3-6's σ secondary, the oracle-centred diagnostic (expected and realized scores, ln(σ / σ\*), vol beta,
  coverage), the tape's rates.
- **Line diagnostics (your request; reported, not tested):** computed per arm on all eval completions, also by depth
  and set.
  - **The parser (fixed now):** the last line before `<answer>` that starts with `sigma_rel =`. It is parsed if its
    right side is `0`, or `sqrt((a*s)^2 + …) = X` with decimal or scientific numbers. Anything else is "present,
    malformed".
  - **The reference:** the line D80's builder writes for the same eval problem (`sft_phase12.propagation_line`), i.e.
    perfect imitation. It reproduces the stored training lines exactly (tested on 200). On the eval sets its X has
    median |ln(X / σ\*)| 0.013, and every eval problem is noisy.
  - **Copy fidelity** is measured among answered completions with a parsed line.
    - **The test:** is the stated σ a valid 2-s.f. rounding of X·|μ|, both in the answer's unit, widened by μ's own
      4-s.f. rounding (5e-4 relative, from D80's `mu_sig_figs`)? X = 0 requires σ = 0.
    - **Also reported:** the median |ln(σ_rel / X)|, and the share with σ = 0 while X > 0.
    - **Not checkable:** a σ stated in another unit or as a percentage, counted separately.
    - **On the SFT data:** all 3,000 completions pass (tested).
  - **Sensitivity correctness** is measured on lines with the reference's number of terms, matched by position (the
    line names no symbols, D80).
    - **The primary match is exact:** the a_i equal the reference's `.2g` value. Reported per term, and as the share of
      problems with every term exact.
    - **[REVIEW] a lenient match** within 10% relative (`analysis.sensitivity_rel_tol`).
    - **Also reported:** the order-free (multiset) exact share, the levels s_i correct in order, the term count
      correct, and the zero line on noisy problems (a claim of exact givens).
  - **Arithmetic:** is X a valid 2-s.f. rounding of sqrt(Σ(a_i s_i)²) of the line's own terms?
  - **X against the reference X and σ\*:** the median |ln(X / X_ref)| and |ln(X / σ\*)|.
  - **The line's X as the σ (secondary):** X per problem (unparsed → 0, the floor) goes through the comparators' path.
    That gives A6 / A6b / A6c slopes over all pairs and the oracle-centred summary. If X propagates but the answer's σ
    does not, the copy is what fails.
- **Reading rules (fixed now):**
  - **"Trained explicit propagation propagates (the ceiling)":** D1 and D2 supported, A6b equivalent to 0, and an A6
    label of partial, full or over-propagation.
  - **"It beats the surface readers":** D3 and D4 supported.
  - **"It beats stage B's supervised σ\*":** D6 supported.
  - **"The trained numbers matter beyond the format":** D7 supported.
  - **D2 not supported:** trained first-order propagation does not transfer noise allocation to new problems with
    this base and budget.
    - **The line diagnostics then locate the failure** (sensitivities, copy, arithmetic), descriptively. No threshold
      for that attribution is pre-registered.
  - **"The format alone propagates":** F1 and F2 supported, A6b equivalent to 0, and an A6 label that propagates.
  - **Seeds ([REVIEW] my reading of R3-4 D's "full seeds only if it resolves B vs C"):** D-pos gets 3 seeds if D2 or
    D6 is supported, i.e. a ceiling that propagates or differs from stage B. Otherwise stage D stays a one-seed
    diagnostic.
- **Not in stage D:**
  - **RL.** R3-9's split credit (the σ reward routed to the line's number tokens as well as the answer's σ) belongs
    to an explicit-arm RL run, which needs its own entry.
  - **p** is not trained.
- **Budget and queue (`scripts/queue_stage_d.sh`, after stage C's queue by PID; pos, then fmt):**
  - **The gate:** an arm starts only if training + eval + 15 min fit before 09:00 IST, otherwise SKIP.
  - **The estimates:** 85 + 65 min (stage B's measured 74–81 min and 54 min, scaled for the line's tokens), then
    D-pos's measured times.
  - **The training's clean stop:** at the deadline minus the eval estimate and the slack.
  - **The analysis** runs on the CPU (1 thread, nice 19) after both arms.
- **Code and outputs:**
  - **The code:** `physcal/train/stage_d.py` (data / train / eval), `physcal/eval/stage_d_analysis.py`,
    `configs/stage_d.yaml`, `tests/test_stage_d.py` (10 tests).
  - **Outputs:** `results/stage_d/{pos,fmt}/seed0/` (training, eval, per-arm `results.json`) and
    `results/stage_d/{results.json, summary.md}`.

### D110. Tonight's run, 2026-10-22/23: implementation decisions taken during the night (overnight rules: decision, alternatives, why)
1. **Stage C was launched before stage D was written** ("launch NOW"). D109 was written and committed (`ff59b4a`,
   23:43) before any stage D model existed; the stage D chain started at 23:44, waiting on stage C's queue PID.
2. **D103's 7.1 GiB cap was kept despite a 1.6 GB host reading** (D108.5: a transient spike; steady 0.6 GiB).
   Outcome: stage C reserved at most 7.04 GiB with no OOM and no spill; stage D's SFT peaked at 4.2 GiB device-wide.
3. **Stage D is new files only and reuses stage B's code unchanged** (`check_data`, `oom_check`, `fit`, `eval_config`,
   `stage_b_analysis.analyze`). D108.4: no module a running stage imports was edited.
   - **Alternative:** a stage-D option in `stage_b.py`. Rejected: stage B's module is stage B's record, and stage C
     imported its neighbours tonight.
4. **D-pos's masks are applied in memory;** `explicit.jsonl` is not rebuilt. Its text is the same as D-fmt's; only the
   mask differs. Its SHA-256 is checked, and D-fmt must reproduce the recorded D84 masks exactly (tested).
   - **Alternative:** a second data file. Rejected: a second file of identical text, and the manifest would change.
5. **The copy-fidelity band includes μ's own 4-s.f. rounding** (5e-4 relative, derived from D80's `mu_sig_figs`).
   - **When:** decided while building, before D109 and before any stage D completion.
   - **Why:** D80 formats σ from the unrounded f(x_meas), while the check can only use the stated μ. Without the
     slack, 10 of the 3,000 SFT completions (the generator's own text) failed at rounding boundaries; with it, all
     pass.
6. **Stage C's policy turned deterministic from step ~40; nothing was changed** (FINDINGS #28). D106 fixes 300
   steps. The remaining ~250 steps (~2.2 h of GPU) learned almost nothing, but stage D still fit before the
   deadline. Alternative: stop at step ~60 and start stage D earlier. Rejected: a protocol change mid-run.
7. **A preliminary stage D analysis on D-pos alone was run into the scratchpad (05:00)** while D-fmt trained, to
   draft the write-up. Nothing pre-registered changed; the official results are the queue's analysis after both
   arms.
8. **Exploratory checks after the analyses (not pre-registered; in FINDINGS #28 / #29):**
   - what stage C's σ copies;
   - D-pos's line errors by set and by term type;
   - **the training problems never mix levels:** all 2,250 noisy SFT problems state one level for every given, so
     A6c (B2 / C2 / D2 / F2) tests transfer to an unseen level mix.
9. **The full test suite ran on the CPU** (1 thread, nice 19, GPU hidden) during stage C (23:44–23:53): exit 0,
   563 collected. pytest's count line was missing because `pyproject.toml`'s `addopts = "-q"` plus my own `-q`
   gives `-qq`, which suppresses it (found on 2026-10-23; later runs drop the extra `-q`).

## 2026-10-23 morning, after the disk-full crash (your instructions at ~10:37 IST)

### D111. Integrity checks, stage D's ignore patterns and a free-space guard in every queue script (your instructions; recorded 2026-10-23, before anything from last night was committed)
- **What happened (your account):** the MLStorage volume filled (STATUS_DISK_FULL). It has been extended and e2fsck
  re-run.
  - **The volume:** `D:\MLStorage`. It holds the Ubuntu disk image (`wsl_ubuntu\ext4.vhdx`, 53.6 GB; 51.1 GB used
    inside it).
  - **This session's transcript** has no entries between 07:14 and 10:37 IST. The prompt you gave this morning
    (steps 1–6) is not in it or in any other local transcript. Only the steps in today's message were done; steps
    3–6 wait for you.
- **Integrity checks** on the 167 files written since 2026-10-22 17:50 UTC, plus the memory notes
  (`scripts/integrity_check.py`, read-only). Nothing is damaged:
  - **`git fsck --full --strict`:** exit 0. Its one dangling commit is a 2026-10-19 stash, unrelated.
  - **Text and logs:** no NUL bytes, every file valid UTF-8, no truncated last line.
  - **JSONL:** 34,683 lines in 14 files, every one parses.
  - **gzip:** all 6 files pass their CRCs and are byte-identical to their uncompressed copies.
  - **safetensors (6):** headers and offsets consistent with the file size; no all-zero tensor, no NaN / Inf, no
    all-zero 4 KiB block.
  - **torch archives (9):** zip CRCs pass. **PNGs (8):** every chunk CRC matches; IEND present.
  - **Tracked files:** the 77 that git calls unchanged hash to their HEAD blobs.
  - **Not checked:** `/lost+found` (it needs root).
- **`.gitignore`, stage D as stage B (approved):**
  - **Ignored:** the adapter weights, tokenizer files, README and chat template; `checkpoints/`; the uncompressed eval
    tapes; the queue's PID files.
  - **Committed:** adapter configs, fingerprints, gzipped tapes, metrics, summaries, plots.
- **The free-space guard (`scripts/disk_guard.sh`):**
  - **Where it runs:** all 14 `scripts/queue_*.sh` source it right after their `cd`. Every stage launcher calls
    `disk_guard NAME || exit 75` as its first statement (`run_stage`; `run_gpu` / `run_cpu` in
    `queue_corrected_evals.sh`).
  - **The reading:** the free bytes of the one volume labelled `MLStorage`, via `powershell.exe` (`Get-Volume`,
    ~1 s). `df` inside WSL shows the image's virtual 1 TB, not the host's space.
  - **The rule:** below 15 GiB, BLOCKED is logged and the queue exits 75. No new stage starts; running children are
    not touched; nothing is committed. GiB is the stricter reading of "15 GB".
  - **[REVIEW] fail closed:** if the free space cannot be read (3 tries, 5 s apart), that is also BLOCKED.
    - **Alternative:** warn and go on. Rejected because a broken check would then let the failure that corrupts
      files come back unseen.
  - **Overrides:** `DRY_RUN=1` only logs the reading. `DISK_GUARD=off` is an explicit bypass and is logged.
    `MIN_FREE_GB` and `DISK_GUARD_LABEL` can be set.
  - **Not added to the non-queue helpers** (`phase37_morning.sh`, `eval_after_training.sh`, …), which are not
    queues.
  - **Tests (`tests/test_disk_guard.py`, 20):**
    - with a fake `powershell.exe`: enough space, low space (the 15 GiB edge), unreadable, dry run, bypass,
      settable threshold and label;
    - every queue script sources the guard and guards every launcher;
    - the real stage D queue, with 1 GiB reported, stops with BLOCKED and exit 75 before its first stage.
- **Reclaimable space:** listed in MORNING.md; nothing deleted. Deleting inside WSL frees blocks the image reuses, but
  the image on MLStorage does not shrink unless it is compacted.

## 2026-10-25: the Round-4 review (your instructions at ~17:00 IST; GPU deadline 09:00 IST on 2026-10-26)

### D112. Round-4 review decisions R4-1 to R4-9 (`REVIEW_SYNTHESIS_ROUND4_2026-10-25.md`), two seed rules not run, the guard stays fail-closed (your decisions; recorded before any result they affect)
- **The review file** is committed with this entry. R4-1 to R4-9 below are your decisions as written there; the
  specifics R4-1 to R4-5 need before anything is built (data format, tests, criteria, stop/go) are pre-registered in
  D113, written before any D-pos-v2 / B-v2 / MC-A data, model or score exists.
- **R4-1. D-pos-v2 pilot and its matched implicit control B-v2, 1 seed each, tonight.**
  - **The same new data for both:** noise levels 0 / 1 / 2 / 5% assigned independently per given. Every
    value-dependent sensitivity is derived as its own visible step (e.g. `s_a = a/(a−b) = … = 1.50`); power-law
    sensitivities are stated as exponents.
  - **The same SFT recipe, budget and pinned base as D-pos** (D109).
  - **A6c is in-distribution for these arms** (mixed levels are in their training data), unlike every earlier arm
    (FINDINGS #29: no Phase 12 SFT problem mixed levels). Every A6c result for them says so.
  - **This approves a generator change** (MORNING.md 2026-10-23 §6.3): mixed per-given levels are generated, not only
    represented. D113 fixes how, without changing any existing problem.
- **R4-2. Grade the line, not only the answer.** For D-pos-v2, three σ values are scored: (a) the answer's stated σ;
  (b) the σ the line states (its X); (c) a verifier-computed σ from the model's own written sensitivities and levels,
  combined exactly. If (c) propagates and (a) does not, the failure is arithmetic, not propagation.
- **R4-3. Success criteria and stop/go, fixed now** (the exact statistics are in D113):
  - **Propagation:** the A6c slope has a 90% CI lower bound above 0; a point estimate ≥ 0.5 is the practical threshold
    for "meaningful". The A6b slope is equivalent to 0 (D59's margin).
  - **Reporting:** everything split by sensitivity type and by depth.
  - **Stop/go:** 3 seeds of both arms only if D-pos-v2 meets the propagation criterion on (a), (b) or (c); which one is
    recorded.
- **R4-4. MC-A, a diagnostic (inference only).** On A6c first (all 1,020 problems), then A6 / A6b if time allows:
  - each displayed given perturbed by its stated noise, k = 8 fixed seeds, greedy decoding;
  - **variants:** raw (SD of ln answers); robust (runs whose sequence of intermediate step names differs from the
    unperturbed run are dropped, then MAD × 1.4826); oracle (the exact solver on the same perturbations: the attainable
    bound);
  - the drop rate, and the k = 2 / 4 / 8 curve from the same logs;
  - **stopping rule:** if robust MC-A does not beat the RSS heuristic on A6c, k is not increased.
  - **Status:** a diagnostic with oracle and robust variants; promoted to a contribution only if the results justify it.
- **R4-5. A frozen confirmatory set:** new sensitivity, invariance and allocation pairs from fresh seeds, untouched until
  a 3-seed confirmatory run. A6 / A6b / A6c become development sets from now on.
- **R4-6. Three estimands, reported separately from now on:**
  - **aleatoric:** σ vs σ\* on valid computations, oracle-centred;
  - **epistemic:** correctness discrimination;
  - **end-to-end:** the proper score of the delivered answer, unconditional.
- **R4-7. Later, each with its own pre-registration (not tonight):**
  - a cross-family inference replication (X2, X4, MC-A);
  - more E1d seeds, plus one unfrozen joint run at lr 5e-5;
  - an MC-B prototype (the model writes a simulation program; few-shot first). PAL and Program-of-Thoughts limit its
    novelty;
  - compositional transfer (train depths 1–3, test 4–5, held-out optics);
  - X4 with native chain-of-thought.
- **R4-8. The paper's framing** is the review's wording: "Predicting uncertainty is not the same as computing it. In a
  controlled small-model physics benchmark, confidence about whether a particular answer is correct is revealed by
  cross-sample agreement rather than by single-response estimators, while propagation of measurement uncertainty is
  hard to learn implicitly, motivating a controlled comparison with explicit and executable computation."
  - **Not claimed:** "uncertainty is not felt inside one forward pass". E1d's capability appeared after a
    learning-rate change, which argues against intrinsic-capability conclusions.
  - **The headline figure:** "Where does uncertainty information come from?", with panels for correctness
    discrimination vs sampling cost, the sensitivity-pair response, and invariance.
  - **Claims are scoped** to this model and this benchmark.
  - **Write-up items (both reviewers):** a provenance ledger of every pre-registered hypothesis and its outcome; the bf16
    merge pitfall as its own box; k-cost curves; problem-clustered CIs; sensitivities split by type (power law vs
    value-dependent) with transfer to held-out compositions and numerical regimes.
- **R4-9. References to verify before citing (not verified tonight; none is cited yet):** Wang et al., Self-Consistency
  (arXiv 2203.11171); Gao et al., PAL (arXiv 2211.10435); Chen et al., Program of Thoughts (arXiv 2211.12588); Kamath
  et al., Selective Question Answering under Domain Shift (ACL 2020); JCGM 101:2008 (Monte Carlo propagation of
  distributions); Wang and Stengel-Eskin (ICLR 2026) and Jang et al. (ICML 2026), both unverified.
- **Deviation: two pre-registered 3-seed rules fired and are not run** (your decision, MORNING.md 2026-10-23 §6.2).
  - **Stage C (D106's signal rule, on C6 +0.045):** the control is the merged SFT model with prompt v5, which answers
    "± 0" on 79% of these problems, so most of C6 is RL removing "± 0", not learning to propagate.
  - **Stage D (D109's seed rule, on D6 +0.001):** D-pos's sensitivities are memorized exponent templates (77% exact on
    power-law terms, 5% on value-dependent ones; FINDINGS #29), and it never read per-given levels. More seeds of the
    same data would replicate the template.
  - **Instead:** R4-1's corrected data, with its own stop/go (R4-3).
- **D111's [REVIEW] is resolved: the free-space guard stays fail-closed** (an unreadable volume is BLOCKED).
- **Leftovers, done before any of the above was built (GPU, ~3 min):**
  - **The D78 GPU check, re-run after the crash and e2fsck:** PASS in 37 s (`scripts/pinned_base_smoke.py`): the
    pinned base loads 4-bit from its snapshot dir (196 bnb 4-bit linears), a fresh adapter names that dir, and the
    reloaded adapter's provenance gives `base_identity sha256:9939448b3a0fbe45`. It first passed on 2026-10-19.
  - **A trained adapter loads (`scripts/adapter_load_check.py`, new):** D-pos's adapter loads through the eval backend
    on the pinned base, its hash equals the eval tape's (`c631c4e5c216ec54`), and its first eval batch (the 32 longest
    prompts, regenerated as one batch) reproduces the logged greedy completions exactly, 32 of 32, each re-scoring to
    the logged reward. 135 s.
- **Host VRAM with nothing running in WSL:** 2,297 MiB at 17:05 IST, and 2,296 MiB later (higher than the 0.6 GiB of
  earlier nights). Tonight's caps are set from a fresh reading at launch (D115).

### D113. Pre-registration of R4-1 to R4-5: D-pos-v2 and B-v2 on mixed-level data, the three-σ grading, the stop/go, MC-A, the frozen confirmatory set (written before any of their data, model, completion or score exists)
- **Status:** a one-seed pilot on the development sets (A6 / A6b / A6c; R4-5). Nothing here is a confirmatory claim.
  Confirmation, if R4-3's go fires, is a 3-seed run evaluated on the frozen R4-5 set under its own entry.
- **1. The data (R4-1): `physcal/train/sft_r4.py`, `configs/sft_r4.yaml`, `results/phase12_sft_data_r4/`.**
  - **The problems:** D80's 3,000 training seeds (60,000,000 + i) with D80's cell per seed (the same round-robin), so
    each problem keeps D80's chain, world, units and phrasing wherever the generator accepts the same attempt. Only
    the noise changes.
  - **Mixed per-given levels (the approved generator change; an option, not a new version):**
    `generate_problem(..., mixed_levels=[0, 0.01, 0.02, 0.05])`.
    - **The levels:** each chain given's level is drawn uniformly and independently from the list on the problem's
      own `levels` stream (one draw per given per attempt). The main and noise streams are untouched.
    - **The errors:** e_i = level_i · z_i, with the noise stream's existing z.
    - **The display:** every given at the noise mode's 5 s.f.; "± p%" on the noisy ones; level-0 givens are exact
      (the predictive keeps them fixed). Distractors draw their own levels from the same list.
    - **σ\*:** the same Monte Carlo with the per-given levels.
    - **The record:** `noise.level` = the largest level, `noise.mixed` = the list, per-given `rel` as A6c. Replay
      passes the list back.
    - **Unchanged:** the default (`mixed_levels=None`) leaves every existing problem byte-identical, and the generator
      version string (2.0.0+bdbb2279) does not change. Tested on a sample of seeds per family and on the frozen sets'
      regeneration. [REVIEW] An option, not a version bump: a bump would re-key every stream and lose the seed match
      with D80.
  - **Two arms, same problems, same `<confidence>` draw as D80:**
    - **`explicit` (D-pos-v2):** D80's solution text, then the sensitivity derivations, then D80's line, then the
      answer:
      - **Before `<steps>`:** one line per noisy given, in problem order:
        `S_x = <expression> = <the expression with the solution's 4-s.f. SI values substituted> = <value .3g>`.
        - **The expression:** the given's total log-sensitivity d ln f / d ln x, by the chain rule over the steps.
          Each step's local elasticity is the shorter of its implicit form (x·(−∂F/∂x)/(∂F/∂y)/y, with F = lhs − rhs)
          and its explicit form; paths are summed, then `factor(cancel(·))`.
        - **The symbols:** the givens, intermediates, target and constants, i.e. the values the solution already
          wrote.
        - **The value** is computed from the substituted, rounded numbers, so each line is self-consistent.
        - **A power-law term** (a constant expression) is stated as its exponent: `S_m = 1`, `S_t1 = -2`,
          `S_h = 0.5`.
        - **Exact givens get no line;** a problem with no noisy given gets none.
      - **Inside `<steps>`, last:** `sigma_rel = sqrt((S_1*s_1)^2 + ...) = X`.
        - **The terms:** the S values copied as written (signed), and s_i each given's own stated level, in problem
          order.
        - **X:** computed from the shown terms, at `.2g`. `sigma_rel = 0` when every given is exact.
      - **The answer:** σ = X·|μ| at `.2g` (D80's copy).
      - **D-pos-v2's masks:** nothing in the σ channel is masked (S lines, line, σ and c are all trained), as D-pos.
    - **`supervised` (B-v2):** D80's supervised arm on the new problems: no S lines and no line; σ\* (with the
      per-given levels) written after "±" and trained.
  - **The builder's checks (tests):**
    - every S value matches `noise.log_sensitivities` (finite differences at the displayed values) within 1% or
      0.005 absolute;
    - the line parses with D-pos-v2's parser and is a valid `.2g` rounding of its own terms;
    - σ is a valid copy;
    - no row is dropped as too long (2,048 tokens).
- **2. Training and eval: D-pos's recipe exactly, except the data** (`physcal/train/stage_r4.py`,
  `configs/stage_r4.yaml`).
  - **Training:** the pinned 4-bit base (D78); a fresh LoRA (r 16, α 16, the 7 modules); seed 0; 2 epochs (750
    steps); `configs/sft.yaml`'s optimizer; D103's 7.1 GiB cap with the OOM check; the temperature guards; the R2-1
    fingerprint.
  - **Eval:** stage B's (`stage_b.eval_config`): greedy, prompt v5, A6 / A6b / A6c (3,060), the adapter unmerged,
    batch 32.
  - **Two exceptions, fixed now:**
    - **The token budget:** if the longest D-pos-v2 training completion exceeds 480 tokens, both arms are evaluated at
      768 new tokens instead of 512 (parity). Truncation is reported per arm either way.
    - **The cap:** set at launch from tonight's host VRAM reading (D115). The 2026-10-25 afternoon readings were
      2.3 GiB, against 0.6 on earlier nights.
  - **Order:** D-pos-v2 (train, eval), then B-v2 (train, eval).
  - **Labels:** A6c is in-distribution for both arms (both trained on mixed levels). Depth 5 is held out from
    training (the eval split's held-out depth) and is reported separately as transfer.
- **3. The three σ values (R4-2), per D-pos-v2 completion:**
  - **(a):** the answer's stated σ_rel = σ / |μ|.
  - **(b):** the line's stated X.
  - **(c):** sqrt(Σ (a_i·s_i)²) of the line's own written terms, computed exactly by the verifier side (no
    rounding).
  - **Parsing:** a line that does not parse gives (b) = (c) = 0, i.e. the floor (D109's convention). A zero line gives
    0.
  - **Descriptive (c_S):** the model's S lines matched by name, combined with the problem's true stated levels
    (sensitivities alone, level reading excluded). A missing S line counts 0.
  - **Reading:**
    - (c) propagates and (a) does not: the failure is the arithmetic or the copy, not propagation;
    - (b) ≠ (c): arithmetic;
    - (a) ≠ (b): the copy.
- **4. R4-3's criteria and the stop/go, applied to (a), (b) and (c) separately.**
  - **C1:** the A6c slope (D95: OLS of Δ ln σ on Δ ln σ\* within pairs; pairs bootstrap, B = 10,000, seed 0) has a
    90% CI lower bound > 0.
    - **(a)** on all answered pairs (D109's D2 set);
    - **(b) and (c)** on all pairs (unparsed → the floor).
  - **C2:** the A6c point estimate ≥ 0.5 (the practical threshold).
  - **C3:** the A6b slope (on Δ ln truth) has a 90% CI inside (−0.3, +0.3) (D59's margin).
    - **(a)** on computation-correct pairs (D104's convention);
    - **(b) and (c)** on all pairs.
  - **"Meets the propagation criterion" = C1 and C2 and C3.** [REVIEW] I read R4-3's practical threshold as part of
    the criterion. Alternative: C1 and C3 only. Rejected because you declined 3 seeds for effects you judged too weak
    (D112).
  - **Go:** 3 seeds of both arms (D-pos-v2 and B-v2), evaluated on the frozen R4-5 set, if and only if D-pos-v2 meets
    it on (a), (b) or (c). Which ones is recorded.
  - **Otherwise:** no seeds. C1 and C3 without C2 is labelled "propagates below the practical threshold".
  - **The go rule uses the unadjusted criteria, as you wrote them.**
- **5. Tests for the record (one seed, so problem-level evidence only; Holm over R1–R8 at α = 0.05).**
  - **R1 / R2 / R3:** D-pos-v2's A6c slope > 0 for (a) / (b) / (c) (C1's sets; one-sided bootstrap p).
  - **R4:** B-v2's A6c slope > 0 (all answered pairs).
  - **R5:** mean oracle-centred expected crps_log, D-pos-v2 − B-v2 > 0 (paired; cluster bootstrap over pairs, B =
    10,000, seed 0, as D109's D6 / D7).
  - **R6 / R7:** D-pos-v2 − RSS > 0, and D-pos-v2 − X1 > 0.
  - **R8:** B-v2 − RSS > 0.
  - **Also reported, untested:**
    - each arm through stage B's analysis unchanged (`stage_b_analysis.analyze`: B1–B5, labels, D59's verdict, the
      oracle-centred diagnostic);
    - the data effect: D-pos-v2 − D-pos, and B-v2 − stage B's 3-seed mean (expected crps_log);
    - D-pos-v2's line diagnostics, D109's set extended to the S lines: S present / parsed, exact against the
      builder's line for the same problem (`.3g`), lenient within 10%, per term and per problem; levels right in order;
      arithmetic; copy.
- **6. Splits (R4-3, both reviewers):**
  - **What:** every slope and score above, and the line diagnostics, by depth (1–5; depth 5 held out) and by
    sensitivity type.
  - **The type is the builder's own derivation:**
    - **per term:** power law (constant S) or value-dependent;
    - **per A6c pair:** "power-law chain" if every term of the chain is a power law, otherwise "value-dependent
      chain".
  - **By set:** A6 is value-dependent by construction (power-law chains never qualify, D50) and A6b is power-law
    only (D59). So A6c is the set where the split is informative.
- **7. MC-A (R4-4), inference only: `physcal/eval/mc_a.py`.**
  - **The model:** B-v2 (seed 0), with stage B's eval settings (pinned base, adapter unmerged, prompt v5, greedy, the
    same token budget, batch 32).
    - **Why B-v2:** the matched implicit model, trained on mixed-level texts like the perturbed ones. Its completions
      have no S lines, so they cost less GPU than D-pos-v2's.
    - **If B-v2 has no adapter when MC-A starts:** stage B seed 0 instead, logged. [REVIEW]
  - **The perturbations:** for run j = 1..8 and each noisy given i, z_ij ~ N(0, 1) from SHA-256 of
    `mc-a|set|family|depth|seed|j`.
    - **The draw:** x_ij = x_meas,i / (1 + s_i·z_ij), the generator's own predictive draw (so oracle MC-A targets
      σ\*).
    - **The display:** shown in the given's displayed unit at 5 s.f. Exact givens and distractors are unchanged.
    - **The text:** the original, with each noisy given's number replaced ("± p%" stays). Each replaced string must
      occur exactly once in the text; a problem where one does not is excluded and counted.
  - **Run 0** is the unperturbed problem's completion from B-v2's own eval tape (same model, prompt and greedy
    decoding).
  - **Per run:**
    - **μ_j:** the verifier's parse in SI (reason ok and μ > 0; anything else is invalid);
    - **the step-name sequence:** the left-hand names of the lines before `<steps>` matching `^name = `, excluding
      the `SI values:` / `Not needed:` / `In <unit>:` lines.
  - **The σ (log space, used as σ_rel), from the k perturbed runs:**
    - **raw:** the SD (ddof 1) of ln μ_j over the valid runs (≥ 2 needed);
    - **robust:** 1.4826 × the MAD of ln μ_j over the valid runs whose step-name sequence equals run 0's (≥ 2 needed;
      if run 0 is invalid, robust is undefined);
    - **oracle:** the SD (ddof 1) of ln f(x_j) over the same k perturbations, the exact chain on the displayed,
      rounded perturbed values.
    - **Undefined → 0, i.e. the floor** (the comparators' convention).
    - **Also reported:** the oracle's MAD × 1.4826 (estimator parity).
  - **The stopping rule (R4-4), fixed now:** robust MC-A (k = 8) beats RSS on A6c if the mean oracle-centred
    expected crps_log difference (robust − RSS) over A6c's 1,020 problems has a one-sided cluster-bootstrap p < 0.05
    (pairs, B = 10,000, seed 0). If it does not, k is not increased in any later run.
  - **Reported, descriptively:**
    - **the drop rate** (perturbed runs that are invalid, and runs dropped by the step-name filter, separately);
    - **the k = 2 / 4 / 8 curve** (the first k runs of the same logs);
    - **for each variant, next to RSS, X1 and σ\*:** expected crps_log, median ln(σ / σ\*), 1σ / 2σ coverage, and
      the A6c slope over all pairs with its 90% CI, split by depth and type (as §6);
    - **R4-6's three estimands:**
      - aleatoric: oracle-centred, on problems where run 0 is computation-correct (A1's τ);
      - epistemic: AUROC of σ for run 0 not being computation-correct;
      - end-to-end: the verifier's crps_log of run 0's μ with the MC-A σ, against the truth, every problem (unanswered:
        the floor).
  - **Afterwards, if time remains:** A6, then A6b, by the same procedure, descriptive only. The stopping rule is
    A6c's.
- **8. The frozen confirmatory set (R4-5):**
  - **Construction:** `configs/pairs.yaml`'s three constructions unchanged, with fresh pair-seed offsets
    2,100,000,000 (A6), 2,200,000,000 (A6b) and 2,300,000,000 (A6c). The same 15 cells × 34 pairs each (1,020
    problems per set). One worker, nice 19.
  - **Where:** `data/eval_sets/2.0.0+bdbb2279/confirmatory/`, with manifests (SHA-256) and `FROZEN.md`.
  - **The guard:** `run_eval`'s frozen-split loader refuses any file under a `confirmatory/` directory unless
    `PHYSCAL_CONFIRMATORY=1` is set (tested). No model is evaluated on it, and no statistic is computed from it,
    until a 3-seed confirmatory run has its own entry.
  - **From now on, A6 / A6b / A6c are development sets.**
- **9. Budget and queue (`scripts/queue_r4.sh`; GPU_DEADLINE 09:00 IST):** D-pos-v2 train → eval → B-v2 train →
  eval → MC-A A6c → MC-A A6 → MC-A A6b.
  - **The gate:** each stage starts only if its estimate plus 15 min slack fits before the deadline.
  - **The estimates:** D-pos's measured times scaled by the data's token ratio; MC-A from B-v2's measured eval
    seconds per completion × 8.
  - **Every stage:** the free-space guard, the in-stage temperature guard, a PID watcher, BLOCKED-and-continue
    (a later stage that needs a missing output is skipped and logged), and a commit + push.
  - **The analyses:** CPU, 1 thread, nice 19, after each arm's eval and after each MC-A set.

### D114. R4 build: what was built, and two deviations from D113 found while building (before any R4 model, completion or score exists)
- **Built (CPU, 1 thread, nice 19), as D113 specifies:**
  - **The generator's mixed option** (`generate_problem(..., mixed_levels=...)`, `noise_record(levels=...)`,
    `regenerate`). The default path is unchanged: the generator, noise and frozen-set tests pass (75), D80's first 48
    problems regenerate byte-identically, and the frozen A6 / A6b / A6c sets still regenerate.
  - **The R4-1 data** (`physcal/train/sft_r4.py`, `configs/sft_r4.yaml`; `results/phase12_sft_data_r4/`; SHA-256s
    in its manifest):
    - **The givens' levels:** 2,638 at 0, 2,628 at 1%, 2,555 at 2%, 2,611 at 5%.
    - **The problems:** 2,067 of 3,000 mix two or more noisy levels; 78 have every given exact.
    - **Every problem keeps D80's cell;** > 90% keep D80's chain (tested).
    - **Sensitivity terms:** 7,794, of which 3,707 are power-law.
    - **The line's X:** median |ln(X / σ\*)| 0.011 on the noisy problems.
    - **Example lines:** `S_t1 = -t1/t2 = -11.34/24.51 = -0.463`, `S_v1 = -2`.
  - **Stage R4** (`physcal/train/stage_r4.py`, `configs/stage_r4.yaml`): D-pos's procedure; the config is tested
    equal to stage D's except the data, the budget and the cap.
  - **The confirmatory set** (R4-5; `data/eval_sets/2.0.0+bdbb2279/confirmatory/`, `FROZEN.md`):
    - built in ~2 min; 510 pairs per set, all 15 cells full;
    - SHA-256 `e803a567…` (A6), `a4f6d81f…` (A6b), `95ba4280…` (A6c);
    - `run_eval` refuses the directory without `PHYSCAL_CONFIRMATORY=1` (tested).
  - **Tests:** `tests/test_r4.py`, 9.
- **Deviation 1: the eval token budget is 1,280 for both arms, not 768.**
  - **D113's rule fired:** the longest D-pos-v2 training completion is 855 tokens, above 480. 9.3% exceed 512 and
    0.7% exceed 768.
  - **Its fallback, 768, is too small:** the builder's own completion for the eval problems reaches 1,165 tokens (A6c,
    depth 5). 768 would truncate 32 A6c and 20 A6 reference completions; 1,024 would still truncate 12.
  - **So 1,280 for both arms** (B-v2's longest training completion is 391, so the budget only matters for its runaway
    generations).
  - **Why the lines are long:** deep kinematics chains, whose total sensitivities in the problem's symbols run to
    100–170 characters, e.g. `a1*t1*(2*s1 - t1*u)/(2*t_tot*u*(a1*t1 + u))`.
  - **Alternative: shorten the format** (e.g. local chain-rule factors). Rejected: it would change the pre-registered
    data after seeing only its lengths.
- **Deviation 2: 6 of D-pos-v2's 3,000 rows exceed 2,048 tokens and are dropped** by stage B's tokenization, which
  drops and counts as it always has (B-v2 drops none). D113 had listed "no row dropped" among the builder's checks.
  - **Alternative:** raise the pinned base's `max_seq_length`. Rejected: that setting is shared by every Phase 12 load.
  - **The cost:** the 6 are among the longest depth-4 rows (0.2%).

### D115. Tonight's launch, 2026-10-25 (17:32 IST; GPU deadline 09:00 IST on 2026-10-26)
- **Host VRAM with nothing running in WSL:** steady at 1,617–1,625 MiB over 12 s (17:30 IST). That is higher than the
  0.6 GiB of earlier nights; the afternoon's first readings were 2,297 MiB.
- **The cap is 5.9 GiB for training and eval** (`configs/stage_r4.yaml`): 8.0 GiB total − 1.6 host − 0.5, the
  memory note's rule.
  - **Alternative:** keep D103's 7.1 GiB, which would spill silently into shared memory with this host use.
  - **The cost:** stage D's SFT peaked at 4.2 GiB device-wide, so training fits. The eval's OOM backoff halves the
    batch if 1,280-token generations need it.
- **The queue:** `GPU_DEADLINE=09:00 setsid nohup bash scripts/queue_r4.sh`, chain PID 22552, started 17:32 IST.
  - **D-pos-v2's training:** 2,994 rows, 750 steps at ~5.8 s / step, so ~73 min.
- **MC-A (R4-4) is a second chain** (`scripts/queue_mc_a.sh`), built and tested while this one runs, started with
  `WAIT_PID=22552`. The order is your instruction: MC-A after B-v2's eval.
- **Running alongside the GPU:** CPU work only, at 1 thread and nice 19 (D97).

### D116. D-pos-v2's eval was BLOCKED by an OOM in the eval's own backoff; fixed, rerun after B-v2's eval (decided at 19:05 IST, before any R4 completion was scored)
- **What happened:** D-pos-v2 trained normally (750 / 750 steps, loss 0.116, 83 min including one 10-min temperature
  pause at 17:57, 88 °C). Its eval started at 18:55 and died at 19:01 with a CUDA OOM.
  - **The first batch** (the 32 longest prompts) finished in 54 s.
  - **The second batch** OOM'd, and the backoff then failed at every size, 32 → 16 → 8 → 4 → 2 → 1.
  - **Nothing was scored:** no tape is written before generation ends. The R4 chain logged it BLOCKED and went on to
    B-v2's training (19:02), as designed.
- **The cause is a bug in `run_eval.HFBackend._generate`, not real memory pressure.**
  - **The bug:** the retry recursion ran inside the `except` block. There, the failed `generate` call's frames (and
    their tensors) stay referenced by the exception's traceback, so each smaller retry ran with every larger failed
    attempt's memory still held. The log shows 3.95 GiB allocated and 1.95 GiB reserved at the single-prompt failure.
  - **Why it never showed before:** earlier evals used 512 new tokens and a 7.1 GiB cap. Tonight's are 1,280 and 5.9
    (D114, D115).
- **The fix (tested):** the retry now runs after the `except` block. The failed attempt is released (gc, then
  `empty_cache`) before the split.
  - **New test:** `test_oom_retry_runs_after_the_failed_attempt_is_released`. It fails on the old code and passes on
    the new.
  - **Unchanged:** behaviour without an OOM, so every earlier eval is unaffected. The eval tests pass (29).
- **Batch 16 instead of 32 for both R4 arms' evals and MC-A** (`configs/stage_r4.yaml`). This halves the KV cache of
  1,280-token generations under the 5.9 GiB cap, so OOM-and-retry should become rare.
  - **Parity:** B-v2's eval (not yet started) uses the same setting.
  - **The cost:** greedy outputs can differ slightly with batch composition (padding). D-pos (v1) ran at 32.
  - **Alternative:** keep 32 and rely on the fixed backoff. Rejected because each OOM wastes the generation up to the
    failure point.
- **The queue (the running chains cannot be edited):**
  - **The MC-A chain is stopped.** PID 31529 was only waiting and had done no work.
  - **Its replacement, `scripts/queue_r4_tail.sh`** (`WAIT_PID=22552`), started at 19:07. Once the R4 chain's GPU
    stages are done (B-v2's train and eval), it runs D-pos-v2's eval. Then it starts the MC-A chain unchanged and runs
    D113's analysis on the CPU beside it.
  - **The order is therefore** D-pos-v2 train → B-v2 train → B-v2 eval → D-pos-v2 eval → MC-A, instead of your D-pos-v2
    eval before B-v2's training. Nothing depends on the order: each eval uses only its own adapter.
  - **The R4 chain's own analysis** (after B-v2's eval) runs without D-pos-v2. The tail chain's later analysis replaces
    it.

### D117. Tonight's run, 2026-10-25/26: what happened after D116 (overnight rules: decision, alternatives, why)
1. **Your 21:45 message** ("continue exactly as queued … don't start anything that isn't pre-registered") was
   followed. Nothing new was started on the GPU after it.
2. **The R4-3 go fired** (FINDINGS #30: D-pos-v2 meets C1–C3 on (a), (b) and (c)). The pre-registered consequence is
   3 seeds of both arms on the frozen R4-5 set. **It is not started tonight**, because it needs its own entry (D113 §4).
   That entry is the next step, for your approval.
3. **MC-A's stopping rule fired** (FINDINGS #31): robust MC-A does not beat RSS on A6c, so k is not increased in any
   later run.
4. **The gates' MC-A estimate was too high.** It used B-v2's eval rate (1.24 s per completion; 172 min per set), while
   MC-A ran at 0.86–1.06 s (117–144 min). So A6 and A6b both ran, and the GPU finished at 05:54, 3 h before the
   deadline. Alternative: a lower default. Rejected tonight: a pessimistic gate only skips; it never risks the
   deadline.
5. **Three 10-min temperature pauses (88 °C):**
   - **17:57** during D-pos-v2's training, while the full test suite ran on one CPU thread. That is the memory note's
     pattern; one thread was enough to tip it.
   - **19:24** during B-v2's training, GPU load alone.
   - **02:19** during MC-A on A6, GPU load alone.
6. **One exploratory check, after the pre-registered analysis** (FINDINGS #30): the regret of D-pos-v2's σ against
   σ\*, by depth and by tail share. It used the eval tapes only (CPU), no model.

## 2026-10-26: decisions on the Round-4 morning report (your instructions at ~07:20 IST; queue auto-starts 23:00 IST, GPU deadline 09:00 IST on 2026-10-27)

Your decisions on MORNING.md (2026-10-26) §6, as written:
- **§6.1:** pre-register the 3-seed confirmatory run (D118, below).
- **§6.2:** C2 is part of R4-3's criterion (D113 [REVIEW]): confirmed.
- **§6.3 / §6.4:** D114's and D116's deviations are accepted.
- **§6.5:** the mixed-level option is not a version bump, provided every problem record and tape stores the flag and replay
  uses it; otherwise the generator's minor version is bumped (D120).
- **MC-B (R4-7):** pre-registered separately (D119) and built CPU-only, with no training.

### D118. Pre-registration of the R4-3 confirmatory run: D-pos-v2 and B-v2, 3 seeds each, on the frozen R4-5 set (your §6.1 decision; written before any seed-1 or seed-2 model, any completion on the confirmatory set, or any statistic computed from it exists)
- **Status: confirmatory.** FINDINGS #30 (one seed, development sets) motivates this run. Nothing below was chosen from the
  confirmatory set, which no model or statistic has touched yet (D113 §8's guard).
- **1. Arms and seeds.**
  - **The arms:** D-pos-v2 (stage R4 `dpos2`) and B-v2 (`b2`), seeds 0, 1 and 2.
  - **Seed 0:** the existing adapters (`results/stage_r4/<arm>/seed0/final_adapter`, trained 2026-10-25 under D113 /
    D115). They are not retrained.
  - **Seeds 1 and 2:** stage R4's recipe exactly (`configs/stage_r4.yaml`, D113 §2), with only the seed changed.
    - **What the seed sets:** the LoRA init, the data shuffle and the trainer's seed (stage B's convention, D104).
    - **The data:** the same R4-1 files for every seed (manifest SHA-256s checked at load). Seeds vary training, not the
      data. Alternative: redraw the mixed levels per seed. Rejected: stage B's 3 seeds also shared one dataset, and the
      question is the recipe's seed variance.
  - **The cap:** 5.9 GiB (D115) for parity with seed 0. If the launch reading of host VRAM is above 1.6 GiB, the cap is
    lowered to 8.0 − host − 0.5 GiB and logged. It is never raised. The cap changes numerics only if an OOM backoff
    fires, and every backoff is logged.
- **2. The confirmatory eval: the seed-0 development eval's settings exactly, on the frozen R4-5 files.**
  - **The files:** `data/eval_sets/2.0.0+bdbb2279/confirmatory/{a6,a6b,a6c}_pairs.jsonl`, 1,020 problems each,
    checksums from their manifests, every problem regenerated and compared before generation.
  - **The settings:** stage B's eval config (`stage_b.eval_config`), greedy, prompt v5, the pinned 4-bit base, the
    adapter unmerged, 1,280 new tokens (D114), batch 16 (D116).
  - **The guard:** `PHYSCAL_CONFIRMATORY=1` is set only by these six evals and by the analysis.
  - **The output:** `results/stage_r4/<arm>/seed<s>/confirmatory/`, with the split names `a6_pairs` / `a6b_pairs` /
    `a6c_pairs`.
- **3. Comparators on the confirmatory set (CPU, no model):**
  - **σ\***, and **RSS** (D95's heuristic, givens scope);
  - **X1:** D99's exact refit of D73's model, applied to the confirmatory pairs. It is used only if the refit reproduces
    D73's published results (D99's rule, tolerance 1e-9). Otherwise every X1 test is "missing" and enters Holm with
    p = 1.
  - **The oracle-centred terms:** as stage A (4,000 predictive draws per problem, seed tag `stage_a`).
- **4. The σ values:**
  - **D-pos-v2:** D113 §3's three, unchanged: (a) the answer's σ_rel, (b) the line's X, (c) the exact combination of
    the line's own terms. Plus (c_S), descriptive.
  - **B-v2:** (a) only.
- **5. The seed-mean statistics: stage B's joint bootstrap (D104), unchanged.**
  - **The bootstrap:** over pairs, B = 10,000, seed 0. Each resample draws the pairs once, and every seed's statistic
    is recomputed on that resample (`stage_b_analysis.seed_mean_slope` / `expected_diff`).
  - **The estimate:** the mean over the 3 seeds of the per-seed statistic.
  - **The slope:** D95's OLS of Δ ln σ on Δ ln σ\* within pairs, with an intercept.
  - **The pair sets, as D113 §4, per seed:**
    - **A6c (a):** pairs where both members are answered.
    - **A6c (b) and (c):** all pairs (unparsed → the floor).
    - **A6b (a):** computation-correct pairs. **A6b (b) and (c):** all pairs.
- **6. The primary hypothesis P (your wording): D-pos-v2's answer σ (a) propagates on A6c at the 3-seed mean.**
  - **Amended by D125 (2026-10-26, 08:35 IST, before any confirmatory result): P also requires C3. That makes it
    D113's full R4-3 criterion.** The text below is the original.
  - **P is confirmed iff both hold:**
    - **K1:** the seed-mean A6c slope is > 0 (one-sided bootstrap p, Holm-adjusted below, < 0.05);
    - **C2:** the seed-mean point estimate is ≥ 0.5.
  - **Reported with it:**
    - the seed-mean 90% CI (C1's form);
    - C3, the A6b seed-mean slope's 90% CI inside (−0.3, +0.3), D59's margin;
    - the label "meets D113's full criterion" iff C1, C2 and C3 all hold.
  - **[REVIEW]** C3 is reported but is not part of P, because your wording names only the slope and C2. Alternative: P =
    C1 ∧ C2 ∧ C3, D113's full criterion.
- **7. The tests: Holm over K1–K8 at α = 0.05, as D113's R1–R8, at the 3-seed mean.**
  - **K1 / K2 / K3:** D-pos-v2's A6c seed-mean slope > 0 for (a) / (b) / (c). **K1 is the primary.**
  - **K4:** B-v2's A6c seed-mean slope > 0.
  - **K5:** the seed-mean oracle-centred expected crps_log, D-pos-v2 (a) − B-v2 > 0.
  - **K6, secondary, reported prominently:** D-pos-v2 (a) − RSS > 0 ("beats RSS", R6).
  - **K7, secondary, reported prominently:** D-pos-v2 (a) − X1 > 0 ("beats X1", R7).
  - **K8:** B-v2 − RSS > 0.
  - **K5–K8:** per problem, the mean over seeds of the expected-score difference, pooled over the three sets (problems
    with predictive draws). The bootstrap is clustered over pairs (B = 10,000, seed 0), as D113's R5–R8.
- **8. The pre-specified split: training depths 1–4 vs the held-out depth 5.**
  - **On each subset:** K1–K4's seed-mean slopes and K5–K8's differences, each with its 90% / 95% CI and one-sided p.
  - **The 3-seed bootstrap** is the same as above, restricted to the subset's pairs.
  - **These are subgroup estimates, not in the Holm family.** No confirmatory claim is made from a subgroup.
  - **[REVIEW]** Alternative: put the 16 subgroup tests into the Holm family. Rejected: they would dilute the primary
    for estimates whose job is to show where the effect lives (FINDINGS #30: the development set's A6c slope fell from
    1.00 at depth 1 to 0.06 at depth 5).
- **9. Also reported, untested:**
  - per-seed values and the seed SD;
  - each arm through stage B's analysis at the 3-seed mean (B1–B5, D59's labels);
  - D-pos-v2's line diagnostics per seed (D113 §5);
  - the A6c slope by depth 1–5 and by chain type (D113 §6);
  - **pre-specified here, after FINDINGS #30's exploratory look:** regret concentration. That is the share of (a)'s
    expected-score regret against σ\* that sits in its worst 5% of problems, and the mean regret by depth.
- **10. When the analysis runs: once, after all six confirmatory evals exist.**
  - **Interim reports** (MORNING.md before then) give only operational counts per eval: completions, truncation and
    parse-fail rates, OOM backoffs.
  - **What they never give:** slopes, scores or coverage.
  - **If an eval is BLOCKED and cannot be rerun:** the analysis runs on the seeds that exist and is labelled
    "incomplete (n seeds)". It still counts as the confirmatory analysis; it is not rerun when the missing seed appears.
- **11. Order and budget (`scripts/queue_r4_seeds.sh`):**
  - **The order:** SFT of D-pos-v2 seeds 1 and 2, then B-v2 seeds 1 and 2. Then the confirmatory evals: D-pos-v2 s0,
    B-v2 s0, D-pos-v2 s1, B-v2 s1, D-pos-v2 s2, B-v2 s2.
  - **The gate:** each stage starts only if its measured estimate plus 15 min fits before 09:00 IST. A stage that does
    not fit is skipped, and the next one is tried.
  - **The estimates (seed 0's wall times):**
    - SFT: D-pos-v2 83 min, B-v2 80 min, each including one 10-min pause;
    - eval: D-pos-v2 111 min, B-v2 65 min.
  - **Tonight:** the four SFTs, then the evals in order as long as they fit (seed 0 of both arms expected).
  - **The following night:** the remaining evals, then this analysis (CPU).
- **12. The reading, fixed now:**
  - **P confirmed:** "D-pos-v2's explicit sensitivity computation propagates measurement noise on fresh problems,
    across 3 seeds".
  - **K6 / K7:** say whether that also buys a better expected score than RSS / X1.
  - **P not confirmed:** reported as a failed confirmation, with the same prominence as a success. FINDINGS #30's
    one-seed result then stands as development-set evidence only.

### D119. Pre-registration of MC-B (R4-7): the model writes the problem's program, a sandboxed interpreter runs it on MC-A's perturbations (your decision; written before any MC-B code, prompt output or score exists). Build and CPU tests now, with no training; the GPU run comes the following night
- **Status: a prototype, inference only.** It runs on the development sets A6c / A6 / A6b, so nothing here is
  confirmatory. PAL and Program-of-Thoughts (R4-9, unverified) limit its novelty.
  - **Its role:** the "executable computation" arm of R4-8's comparison, next to MC-A's oracle (an exact executor) and
    D-pos-v2 (a learned explicit computation).
- **1. The model: the pinned 4-bit base (Qwen2.5-Math-1.5B-Instruct, D78), no adapter.**
  - **The settings:** the unsloth 4-bit loader, as every Phase 12 eval; greedy; one completion per problem.
  - **Why no adapter:**
    - the SFT arms were trained on the `<steps>` / `<answer>` format without code;
    - the instruct model was trained on tool-integrated reasoning in Python, so few-shot program writing is in its
      distribution;
    - and no training is allowed here.
  - **Alternative 1: Qwen2.5-Math's own TIR format** (code with the values hard-coded, the interpreter's output fed
    back, a boxed answer). Rejected as the primary format, because its programs hard-code the givens. Re-running them
    on perturbed inputs would mean rewriting numeric literals in model-written code, which is ambiguous when a value
    appears twice or is converted before use. A function whose parameters are the givens' own symbols makes the
    substitution exact.
  - **Alternative 2: B-v2 with a program instruction.** Rejected, because it was trained away from code.
- **2. The prompt (fixed in `configs/mc_b.yaml` and `physcal/eval/mc_b.py` at build, before any MC-B generation).**
  - **No tuning:** the prompt is never adjusted on any model output. The only GPU run is the pre-registered one.
  - **The structure:** a system message, two worked examples (user: a problem text; assistant: its program), then the
    problem.
  - **The examples:**
    - **Their source:** training-split generator-2.0.0 problems with mixed levels, from a seed range of their own
      (81,000,000 + i). They are not from any eval or development set.
    - **The selection, fixed now:** the first seed per cell, for cells (kinematics, 2) and (dynamics, 2), whose problem
      has at least one distractor, a given in a non-SI unit, and at least one noisy given.
    - **The assistant turns:** the builder's own reference program for each example (item 3).
  - **The format the system message asks for:**
    - one ```python block defining `solve(...)`;
    - its parameters are the symbols of the givens it uses, named exactly as in the problem, each in the unit the
      problem shows;
    - units are converted inside the function, with the `math` module only;
    - it returns the requested quantity as a float in the unit the problem asks for.
  - **The token budget:** the longest prompt on the three sets, plus 768 new tokens, must fit the pinned base's
    `max_seq_length` (2,048). This is tested. If it does not fit, the examples are switched to depth 1 before any
    generation, and the switch is logged.
- **3. The reference program (the builder's; CPU).**
  - **What it does:**
    - converts each given used from its displayed unit to SI by a pint factor;
    - assigns the family's constants (`g = 9.81`);
    - writes each chain step as `out = <the solved expression>` (sympy's `pycode`, math module);
    - divides the target by the requested unit's SI factor.
  - **What the reference programs are for:**
    - the few-shot examples;
    - a test that the harness reproduces MC-A's oracle exactly when the program is right: for every problem of the
      three sets, the reference program's σ equals MC-A's oracle σ within 1e-9.
- **4. Execution.**
  - **The extraction:** the first fenced ```python block that defines `solve`. If there is none, the program fails with
    `no_program`.
  - **A static AST check:**
    - imports only `math`;
    - no name or attribute that starts with `_`;
    - no `eval` / `exec` / `open` / `compile` / `getattr` / `setattr` / `delattr` / `globals` / `locals` / `vars` /
      `input` / `breakpoint` / `exit` / `quit`;
    - no `class`, `global`, `nonlocal`, `with`, `async` or `lambda` at module level;
    - otherwise the program fails with `rejected`.
  - **The signature:**
    - positional-or-keyword parameters only;
    - each one must be a displayed given's symbol (chain given or distractor); otherwise `signature`;
    - the harness calls `solve(**{p: displayed value as a float})`.
  - **The sandbox:**
    - one separate `python -I` process per problem, with all k + 1 calls in it;
    - restricted builtins with `math` injected;
    - limits RLIMIT_CPU 2 s, RLIMIT_AS 1 GiB, RLIMIT_FSIZE 0, and a 10 s wall timeout;
    - a failure gives `exception` or `timeout`.
  - **The output:** it must be a finite real > 0. It is converted to SI by the problem's target-unit factor. Anything
    else is `bad_output`.
  - **A program fails if any of its k + 1 calls fails.** The failure rate is reported unconditionally: every problem
    enters the denominator, by failure type, set and depth.
- **5. The runs and the σ.**
  - **The runs:**
    - **run 0:** the displayed values;
    - **runs 1–8:** MC-A's perturbations exactly (`mc_a.perturb`, tag `mc-a`, the same z, the perturbed values as
      displayed at 5 s.f.).
  - **Exclusions:** the problems MC-A excluded are excluded identically and counted.
  - **The σ (log space, used as σ_rel), from the first k perturbed runs:**
    - **`mc_b`:** the SD (ddof 1) of ln y_j;
    - **`mc_b_mad`:** 1.4826 × the MAD, for parity with MC-A's robust variant.
  - **μ:** y_0.
  - **A failed program:** σ = 0 (the floor) and μ unanswered.
  - **No step filter:** every run executes the same program, so MC-A's step-name filter has nothing to drop.
  - **The k curve:** k = 2 / 4 / 8, from the first k runs.
- **6. Metrics (R4-6's three estimands, MC-A's analysis).**
  - **The problems:** the problems MC-A used, so the comparison with MC-A is paired.
  - **Aleatoric:**
    - the oracle-centred expected crps_log; the median ln(σ / σ\*); 1σ / 2σ coverage;
    - the slope with its 90% CI (A6c / A6 on Δ ln σ\*, A6b on Δ ln truth), split by depth and chain type;
    - the same on run-0-computation-correct problems.
  - **Epistemic:** the AUROC of σ for run 0 not computation-correct (|ln y_0 − ln f(x_meas)| > τ = 1e-3).
  - **End-to-end:** the verifier's crps_log of y_0 with MC-B's σ against the truth (unanswered → the clip).
  - **The program's own quality:**
    - the share of problems with run 0 computation-correct;
    - the share whose 9 outputs all match the exact chain within τ ("exact program").
- **7. Comparators on the same problems:** σ\*, RSS, X1, MC-A's oracle / robust / raw (k = 8, from its run logs), and
  D-pos-v2 (a), seed 0 (its development eval).
- **8. Tests on A6c (the primary set; the cluster bootstrap over pairs, B = 10,000, seed 0; one-sided; Holm over M1–M5
  at α = 0.05):**
  - **M1:** MC-B − RSS > 0 (mean oracle-centred expected crps_log);
  - **M2:** MC-B − X1 > 0;
  - **M3:** MC-B − D-pos-v2 (a) > 0;
  - **M4:** MC-B − MC-A robust > 0;
  - **M5:** MC-B's A6c slope > 0.
  - **Reported, untested:** MC-B − MC-A oracle (the attainable bound), and everything on A6 / A6b.
- **9. The reading, fixed now:**
  - **If M1 holds:** "an executed model-written program propagates measurement noise better than the heuristic", with
    the failure rate always beside it.
  - **If M1 fails:** the failure rate and the share of wrong programs (by depth) say where the program arm loses.
  - **No k increase and no prompt change follow from this run** without a new entry.
- **10. Budget:** 3,060 generations (one per problem). The estimate is ≤ 75 min, from B-v2's eval rate with shorter
  completions but longer prompts. It runs the following night, after the remaining confirmatory evals. Execution and
  analysis are CPU work at 1 thread, nice 19, after the generation ends.

### D120. Your §6.5 check: the mixed-level flag is stored in every problem record and tape, and replay uses it. No version bump
- **The record:** every mixed-level problem's dict has `noise.mixed` (the list drawn from; `noise.level` is the
  largest). This holds for `results/phase12_sft_data_r4/problems.jsonl` too, which carries `[0, 0.01, 0.02, 0.05]` on
  every row.
- **The tape:** `TapeWriter.log` writes the full problem dict, so the flag is on every tape line.
- **Replay:** `tape.replay` calls `generator.regenerate`, which passes `noise.mixed` back as `mixed_levels`.
- **The new end-to-end test** (`tests/test_tape.py::test_mixed_level_tapes_store_the_flag_and_replay_exactly`):
  - a tape of 12 mixed-level problems replays exactly;
  - with the flag dropped, or with a different list, the problem no longer regenerates, and replay reports it.
- **Result:** the generator's version string stays 2.0.0+bdbb2279, as D113 [REVIEW] proposed and you accepted. No tape
  or eval set changes.
- **Where the flag occurs so far:** only the R4-1 training data. No eval tape or MC-A run has a mixed-level problem (A6c
  varies levels through its pair construction, which `regenerate` already handles).

### D121. D118's build and tonight's armed queue (2026-10-26, 07:30–07:45 IST)
- **Built:**
  - **stage R4** takes `--seed`, `--confirmatory` (the frozen R4-5 files, `<arm>/seed<s>/confirmatory/`, run_id
    `stage-r4-<arm>-seed<s>-confirmatory`) and `--max-reserved-gib` (refused if above the config's 5.9: never raised);
  - **`configs/stage_r4.yaml`** has a `confirmatory` block (seeds [0, 1, 2], the three frozen files with full
    regeneration checks);
  - **the queue:** `scripts/queue_r4_seeds.sh`;
  - **the waiter:** `scripts/run_at.sh`.
- **Tests (`tests/test_r4_confirm.py`, 7):**
  - the confirmatory eval config equals the development one except splits, output dir and run_id;
  - the cap is refused above 5.9;
  - the launch cap with a fake `nvidia-smi`;
  - the dry run's stage order and commands;
  - the waiter execs under its own PID, and can be cancelled before its start time.
- **A fix to an existing test:** `test_disk_guard.py::test_a_real_queue_stops_before_its_first_stage` failed whenever it
  ran less than ~2.75 h before 09:00 IST. Stage D's time gate then skipped the arm before the guard was reached. Its
  deadline is now six hours ahead of the clock. No code change.
- **The launch cap rule:** host VRAM is the median of 8 readings, 2 s apart, re-read up to 5 times while it gives a cap
  below 5.9. The cap is min(5.9, total − host − 0.5) GiB, floored to 0.1. With the D115 reading (1,625 MiB) it gives
  5.9; at 07:30 today the host used 1,034 MiB.
- **GPU smoke tests (07:35–07:41 IST; nothing kept):**
  - **training:** `stage_r4 train --arm b2 --seed 9 --max-steps 2 --max-reserved-gib 5.9` passed (184 s). The summary
    records seed 9 and the cap. The directory was deleted afterwards.
  - **eval:** seed 9's 2-step adapter was refused by the R2-1 reload check (ratio 0.94 > 0.25). That is expected for an
    adapter whose effect on the logits is ~0.01; trained adapters pass it. So the eval path was smoked instead with
    seed 0's trained adapter, on 4 development A6c problems, into the scratchpad: exit 0, 4 / 4 parsed, the adapter
    hash as logged (`24148d53…`).
  - **The confirmatory set was not touched.**
- **The waiter:** `START=23:00 PID_FILE=results/queue_r4_seeds/waiter.pid GPU_DEADLINE=09:00 setsid nohup bash
  scripts/run_at.sh bash scripts/queue_r4_seeds.sh`. **PID 109020**, armed at 07:42 IST.
  - **At 23:00 IST** it execs the queue under the same PID.
  - **To cancel:** `kill 109020`. Before 23:00 nothing has started. After 23:00 the kill stops the chain from starting
    further stages, but does not touch a running stage.
- **The expected night (seed 0's wall times, the gate's 15-min slack):**
  - the four SFTs, 23:00 → ~04:30;
  - D-pos-v2 s0's eval → ~06:20, then B-v2 s0's → ~07:25;
  - B-v2 s1's eval also fits (~08:30). D-pos-v2 s1's (111 min) does not, so the gate skips it, and the next night runs
    it with s2's two evals and MC-B.

### D122. MC-B's build (D119; CPU only, no GPU, no model output), and two deviations found while building
- **Built:**
  - **`physcal/eval/mc_b.py`:**
    - the reference program;
    - the examples and the prompt;
    - generation (GPU, resumable, chunked);
    - execution (CPU, resumable);
    - the estimators;
    - D119 §6–§8's analysis and the summary.
  - **The sandbox:** `physcal/eval/sandbox.py` (extraction, the static check, the signature, the parent with its wall
    timeout) and `sandbox_child.py` (limits set before the program is read, restricted builtins, `import math` the only
    import that resolves).
  - **The config:** `configs/mc_b.yaml`.
  - **Tests:** `tests/test_mc_b.py`, 23.
- **The reference check (D119 §3): the builder's program, run through the same harness on MC-A's inputs, reproduces
  MC-A's oracle on every problem of the three development sets.**
  - **The numbers:** 3,060 of 3,060, no failure and no exclusion. Max |σ_ref − σ_oracle| 3.2e-16; max |ln y − ln f|
    1.8e-15 over all 9 runs. About 20 s per set.
  - **So a model program that computes the right function gets exactly MC-A's oracle σ.**
- **The examples (D119 §2's selection):**
  - kinematics depth 1, seed 81,000,000 (`t2 = s2 / v`, km → m, cm/s → m/s, target in h);
  - dynamics depth 1, seed 81,000,002 (`f_k = F − F_net`, kN → N).
  - Both have a distractor, a non-SI unit and noisy givens.
- **Deviation 1: the token budget.** D119 §2's check failed for the depth-2 examples (the longest prompt is 1,402 tokens;
  + 768 = 2,170 > 2,048). Its pre-registered fallback, depth-1 examples, still failed by 21 tokens (1,301 + 768 =
  2,069).
  - **So the new-token budget is 736:** 2,048 − 1,301, rounded down to a multiple of 16. Truncation is reported.
  - **Why it is enough:** the reference programs are a few lines (the longest, depth 5, is ~20 lines).
  - **[REVIEW]** Alternative: depth-2 examples (they show chaining, which deep problems need) with 640 tokens (1,402
    + 640 = 2,042). I kept D119's fallback order and changed only the budget.
- **Deviation 2: two readings of D119 §4, fixed before any model output exists.**
  - **The forbidden constructs:** class, global, nonlocal, with and async are rejected anywhere in the program; lambda
    only at module level. D119's sentence put "at module level" at its end.
    - **Why:** "anywhere" is stricter for the constructs a computation never needs.
    - **What stays allowed:** a lambda inside `solve`.
  - **Defaulted parameters:** a parameter that is not a displayed symbol is allowed if it has a default, and is then not
    passed (e.g. `g=9.81`). Without a default it fails as `signature`.
    - **Why:** the problem text says "Use g = 9.81 m/s^2", and a program that makes g a defaulted parameter computes
      exactly what one with `g = 9.81` in its body does.
- **Also fixed by construction:**
  - **The problem text the model sees** has the answer format's `<steps>` sentence removed (e.g. "Show your calculation
    first; …"). Nothing else changes (tested on a sample of every set).
  - **`print` is a sink** in the child, so a program that prints still returns its value.
  - **No GPU was used for MC-B,** and the prompt has never produced a model output. The first generation is the
    pre-registered run, the following night.
- **Amendment (your decision, 2026-10-26, ~08:35 IST, before any MC-B generation or result): depth-2 examples with
  640 new tokens.**
  - **Your condition:** the longest reference program on all 3,060 problems fits within ~600 tokens.
  - **Measured** with the pinned base's tokenizer, counting the fenced block plus EOS:
    - longest 266 tokens (A6c, depth 5, seed 4,002,800,065); p99 219; median 125;
    - by depth, the longest is 136 / 178 / 209 / 234 / 266.
  - **So the examples are back to D119 §2's cells** (kinematics 2 and dynamics 2; seeds 81,000,001 / 81,000,001), with
    `max_new_tokens` 640.
  - **The budget check:** the longest depth-2 prompt is 1,402 tokens, and 1,402 + 640 = 2,042 ≤ 2,048 (the test
    passes).
  - **This supersedes Deviation 1's 736 / depth-1 choice.**

### D123. D118's analysis is built and checked before any confirmatory completion exists; X1's confirmatory predictions are computed (D118 §3)
- **X1 on the confirmatory set** (`x1_a6c confirmatory`; CPU, 1 thread, nice 19, 2 min, 07:53–07:55 IST):
  - **The refit:** D99's refit reproduces D73's published numbers exactly (max abs difference 0.0 on all three feature
    sets).
  - **A check added before the frozen files are read:** the refit's σ̂ on the development sets equals the committed
    `results/x1_text_sigma/a6c/predictions.jsonl` exactly. It passed.
  - **The output:** 3,060 predictions (`results/x1_text_sigma/confirmatory/`). These are X1's σ̂ from problem texts
    only, and nothing was scored.
- **The analysis: `physcal/eval/r4_confirm.py`** (D118 §5–§10).
  - **What it computes:**
    - the seed-mean slopes and expected-score differences with stage B's joint bootstrap;
    - K1–K8 with Holm;
    - P = K1 ∧ C2;
    - C1–C3 per source;
    - the depth 1–4 / 5 split;
    - by depth and by chain type;
    - regret concentration;
    - each arm through stage B's analysis.
  - **Its two guards:** it refuses without `PHYSCAL_CONFIRMATORY=1`, and before all six evals exist
    (`--allow-incomplete` only if one is BLOCKED for good, D118 §10).
  - **Stage B's analysis takes three optional arguments** (another stage-A config, each seed's eval subdirectory, the
    output dir). The defaults are unchanged; stage B's, stage D's, R4's and MC-A's tests pass (33).
- **The check (`tests/test_r4_confirm.py`, 3 more tests):**
  - **The setup:** the analysis pointed at the development sets, with seed 0's development tapes copied into three
    "seeds".
  - **The result:** every K point estimate equals D113's R1–R8 (`results/stage_r4/results.json`) to 1e-12, and the seed
    SD is 0.
  - **Also reproduced:** FINDINGS #30's depth curve for (a) (1.00 / 0.97 / 0.76 / 0.38 / 0.06) and its regret
    concentration (80% in the worst 5%).
  - **The confirmatory files were not read** for this: the development-set run used the development files and the
    development X1 predictions.

### D124. The queue's second-night stages, added behind flags; tonight's stages unchanged (2026-10-26, ~08:00 IST)
- **What changed in `scripts/queue_r4_seeds.sh`:** the armed waiter (PID 109020) execs it at 23:00, so this edit
  applies tonight. Tonight's GPU stages are unchanged; the dry-run test checks the stage list exactly.
  - **Step 11, MC-B (D119), only with `MC_B=1`** (the following night). Tonight's waiter does not set it.
    - **On the GPU:** generation on A6c, A6, A6b. Each set is gated on an unmeasured estimate of 40 min, plus the slack.
    - **Then on the CPU:** sandbox execution and D119's analysis per set (1 thread, nice 19).
  - **Step 12, D118's analysis (CPU),** only when all six confirmatory evals exist. Tonight it only logs how many exist.
- **Skips:** trained seeds and finished evals are already skipped. So the second night is the same script with
  `MC_B=1`: D-pos-v2 s1 / s2 and B-v2 s2 evals (whichever remain), then MC-B, then the analysis.
- **Alternative:** a separate script for the second night. Rejected: the shared stages would be duplicated, and a
  dry-run test covers both modes.
- **Housekeeping:** `.gitignore` now covers seeds 1 and 2 as well as seed 0. Ignored: the adapter weights, the
  checkpoints, the raw development and confirmatory rollouts, and the new queue's PID files. Everything the queue
  commits stays tracked (checked with `git check-ignore`).
- **The full test suite (08:03–08:13 IST, CPU, 1 thread, GPU idle): 641 passed, 0 failed, 0 skipped.**

### D125. Amendment of D118 §6: the primary P is D113's full R4-3 criterion (your decision, 2026-10-26, ~08:35 IST; recorded before any confirmatory result exists)
- **Your decision:** "use D113's full R4-3 criterion, i.e. K1 (A6c slope > 0, Holm) AND C2 (3-seed mean ≥ 0.5) AND
  A6b equivalent to 0 (D59 margin) … my earlier wording omitted invariance by mistake."
- **P (D-pos-v2's answer σ, (a)) is confirmed iff all three hold:**
  - **K1:** the A6c seed-mean slope > 0, with its Holm-adjusted one-sided p < 0.05 (Holm over K1–K8, as before);
  - **C2:** the A6c seed-mean point estimate ≥ 0.5;
  - **C3:** the A6b seed-mean slope's 90% CI inside (−0.3, +0.3), D59's margin. It is computed on computation-correct
    pairs (D113 §4), with stage B's joint bootstrap over the 3 seeds.
- **Unchanged:** K1–K8 and their Holm family, the secondaries K6 / K7, the depth split, and everything else in D118.
  The [REVIEW] in D118 §6 is resolved by this entry.
- **The state when amended (08:31 IST):** no confirmatory eval, completion or statistic exists. The waiter (PID 109020)
  is still sleeping until 23:00, and `results/stage_r4/*/seed*/confirmatory/` does not exist.
- **The code:**
  - `configs/stage_r4.yaml` has `confirmatory.primary_criteria: [K1, C2, C3]`;
  - `r4_confirm` computes P from that list and reports each flag;
  - the 3-identical-seeds test checks the new rule, and that P fails without C3.
  - **Only the analysis is touched.** Tonight's stages read only the eval keys of this config (`seeds`, `splits`,
    `output_subdir`), and the queue script and waiter are untouched.

### D126. The 23:00 waiter died with a WSL restart; the queue was launched by hand at 23:36 IST with the 10:00 IST deadline (your instruction at 23:35 IST)
- **What happened:** at 23:36 IST the waiter (PID 109020) was gone. `results/queue_r4_seeds/` had no `chain.log`, and
  `nohup.log` held only the 07:42 "armed" line. So the queue never started.
  - **The cause:** the WSL VM had restarted (`uptime -s`: 2026-10-26 18:06 UTC = 23:36 IST), which killed every WSL
    process, including the waiter and this session's watcher.
  - **The waiter was not at fault:** its test and the dry runs were fine. A process inside WSL cannot outlive a VM
    shutdown.
- **Checked before launching:** no `queue_r4_seeds.sh`, `run_at.sh` or `physcal` process was running; the GPU showed no
  compute process (1,212 MiB host use); and no stage output existed (`results/stage_r4/*/seed1`, `seed2` and every
  `confirmatory/` were absent).
- **Your case 2: launched now.** `GPU_DEADLINE=10:00 setsid nohup bash scripts/queue_r4_seeds.sh` (no waiter). **Chain
  PID 632**, started 23:36:44 IST, 623 min to the deadline.
  - **The launch cap:** 5.9 GiB (host readings 1,157–1,175 MiB).
  - **Free space:** 53 GiB on MLStorage.
  - **The first stage:** `train_dpos2_s1`, PID 1244, started 23:37:03.
- **The new deadline (10:00) is the queue's own `GPU_DEADLINE`.** Every gate and training `--stop-at` uses it, so
  09:45 is the latest stop.
- **The expected night (seed 0's wall times):**
  - the four SFTs end ~05:10;
  - D-pos-v2 s0's eval ~07:05, then B-v2 s0's ~08:10;
  - then B-v2 s1's eval (~09:20) fits. D-pos-v2 s1's (126 min with slack) does not.
- **The risk this exposed:** any WSL restart kills the queue. Nothing inside WSL can guard against that, so MORNING.md
  will report it if it happens again.

### D127. Night 1 of D118 is done (3 of 6 evals); two queue fixes before night 2: MC-B's analysis waits for a finished generation, and the dry-run tests stop reading the real results (2026-10-27, ~09:30 IST)
- **Night 1:** chain PID 632 ran 23:36–09:18 IST. Every stage it started exited 0.
  - **Finished:** the four SFTs (750 / 750 steps each) and three confirmatory evals (D-pos-v2 s0, B-v2 s0, B-v2 s1).
  - **Skipped by the deadline gate:** the D-pos-v2 s1 and s2 evals and the B-v2 s2 eval.
  - **The analysis waits** (D118 §10: 3 of 6 evals exist).
- **Fix 1, MC-B completeness (`mcb_generated` in `scripts/queue_r4_seeds.sh`):**
  - **The problem:** a generation stage stopped by its timeout (2 × MCB_EST = 80 min, and MCB_EST is unmeasured)
    leaves a partial `generations.jsonl`. The queue then executed and analyzed whatever existed. `results.json` would
    have covered a subset (the first problems in set order) without saying so.
  - **Now:** execution and analysis run only if every problem has a program and `meta.json` has `finished`.
    `finished` is written at the end of a generation run; each run rewrites `meta.json` without it at its start.
    Otherwise the queue logs BLOCKED, and the next run resumes the generation (it appends).
  - **Test:** `test_mc_b_analysis_waits_for_a_finished_generation`.
  - **Unchanged:** MC-B's code, prompt, sandbox and analysis (D119, D122).
- **Fix 2, test isolation:**
  - **The problem:** `test_dry_run_order_and_commands` and `test_dry_run_with_mc_b_adds_its_stages_after_the_evals`
    failed after night 1. The dry run read `results/stage_r4`, where the adapters and three evals now exist, so it
    skipped them.
  - **Now:** `OUT` is `${QUEUE_OUT:-results/stage_r4}`, and the tests pass an empty temp dir. Real runs are unchanged.
- **The confirmatory plots:** the queue commits each eval's rollouts, meta, metrics and summary, but not its PNGs
  (calibration per set, reliability). They are committed with this entry, unopened, as the development evals' were.
- **D126's reading, corrected:** `uptime -s` (23:36 IST) is the VM's boot time. So the VM was down before 23:36, and
  the waiter died whenever it stopped (some time between 07:42 and 23:36).
  - **The trigger is unknown:** the app closing, sleep, or Windows stopping the VM.
  - **The conclusion stands:** a waiter inside WSL does not survive a VM stop.
- **Night 2's dry run** (the real state, `MC_B=1`):
  - **Skipped:** the four SFTs and the three existing evals.
  - **Run:** the D-pos-v2 s1 and s2 evals and the B-v2 s2 eval; MC-B generation on A6c, A6 and A6b; MC-B execution and
    analysis on CPU; then `r4_confirm`.
- **Full suite after the fixes:** 641 passed, 0 failed, 1 skipped (the GPU test, with the GPU hidden).
- **Night 2 armed (your item 3, 2026-10-26):** after this morning's report, at 09:34 IST, the same way as D121.
  `START=23:00 PID_FILE=results/queue_r4_seeds/waiter.pid GPU_DEADLINE=09:00 MC_B=1 setsid nohup bash scripts/run_at.sh
  bash scripts/queue_r4_seeds.sh`. **Waiter PID 67051**; the chain will keep that PID.

### D128. A single-instance lock on the queue, and a Windows scheduled task as tonight's second start (your decisions on MORNING.md §6, 2026-10-27, ~10:45 IST)
- **§6.1 and §6.2 approved:** D127's MC-B guard, and its correction of D126.
- **The lock:** `scripts/queue_r4_seeds.sh` takes `flock -n` on `results/queue_r4_seeds/queue.lock` (fd 8; commit_stage's
  git lock is fd 9), before it writes `queue.pid` or logs "chain start".
  - **A second copy** logs "already running: another queue_r4_seeds.sh holds …; this copy (PID n) exits 0" and exits 0.
  - **The stages inherit the fd**, so the lock is held until the chain and every stage it started have ended. A new
    chain cannot start while an orphaned stage still holds the GPU.
  - **Test:** `test_second_copy_exits_0_while_the_lock_is_held` (a held lock gives exit 0, "already running", no
    `queue.pid`; once released, the same log dir runs the chain). The queue tests pass: 37.
  - **Gitignored:** `results/queue_r4_seeds/queue.lock`.
- **The Windows task `physcal-night-queue`,** registered from WSL by `scripts/schedule_night_queue.ps1` (committed, so
  the exact command is in the repo):
  - **Runs:** `wsl.exe -d Ubuntu -e bash -lc "cd ~/operation-physics && GPU_DEADLINE=09:00 MC_B=1 nohup
    scripts/queue_r4_seeds.sh >> results/queue_r4_seeds/nohup.log 2>&1"`.
  - **Trigger:** once, 2026-10-27 23:00 IST. "Run only when user is logged on" (Interactive), as user Raghh.
  - **Settings:** the Windows default "do not start on batteries" is kept. "Stop if going on batteries" is off, so a
    power blip does not kill the night's `wsl.exe`. Time limit 16 h.
  - **Delete:** `schtasks.exe /delete /tn physcal-night-queue /f`, or in PowerShell `Unregister-ScheduledTask
    -TaskName physcal-night-queue -Confirm:$false`.
- **One deviation from your command: no trailing `&`.**
  - **Why:** with `nohup … &`, `wsl.exe` exits at once and the instance has no client left. WSL may then stop it when
    idle, killing the queue. That is the likely way yesterday's waiter died (D127: the VM was down before 23:36).
    Community reports describe the same for current WSL2 (`instanceIdleTimeout` / `vmIdleTimeout`; `systemd=false`
    here).
  - **So:** `wsl.exe` runs the queue in the foreground and stays attached all night. The task shows "Running", and a
    console window stays open until the chain ends. `nohup` is kept, so closing that window does not SIGHUP the chain.
  - **When the lock is held,** the task's copy exits 0 at once.
- **Tested end-to-end through Task Scheduler** with a throwaway task (`physcal-night-queue-test`, deleted after):
  - **The time trigger fired at 10:57:00.** The full dry chain ran via Windows → `wsl.exe` → `bash -lc`, with result 0.
  - **The same task, started while a held lock existed,** logged "already running" and exited 0 (result 0).
  - **The login shell has the same environment as the session:** uv, nvidia-smi, flock and git, and push auth works.
- **The waiter (PID 67051) is kept as the backup.** At 23:00 it execs the current script, lock included.
  - **The race at 23:00:** the waiter wakes at 23:00:00 inside the VM, while the task first has to start `wsl.exe`. So
    if the VM is up, the waiter's copy probably takes the lock, the task's copy exits 0, and the chain again depends
    on something keeping WSL alive (this session, as last night).
  - **If the VM is down, the waiter is dead** and the task's copy runs, attached.
  - **Making the task the primary:** register it at 22:59 instead (`-At 22:59`). Not done: you specified 23:00.
- **Amended (your decision, 2026-10-27, ~11:10 IST):** the task is re-registered at **22:59 IST**
  (`Unregister-ScheduledTask`, then `schedule_night_queue.ps1 -At 22:59`). `schtasks /query`: next run 10/27/2026
  10:59:00 PM, one time only, Interactive only.
  - **The task is now the primary start:** its `wsl.exe` stays attached for the whole chain.
  - **The waiter (PID 67051, 23:00) is the backup:** if the task's chain holds the lock, it logs "already running" and
    exits 0.
  - **The foreground version (no `&`) is approved.**

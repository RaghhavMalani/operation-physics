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

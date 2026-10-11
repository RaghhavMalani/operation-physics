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

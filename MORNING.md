# Morning report: overnight session 2026-10-11

**Phases 1, 2 and 3 are done, tested and committed. Phase 4 was not started.** 210 tests pass
(`uv run pytest`, ~33 s). Nothing is BLOCKED. Nothing was pushed.

**The headline finding:** untrained Qwen2.5-1.5B-Instruct scores **0%** on these problems. When I sampled
8 completions per problem, as a GRPO group would, **100% of groups had zero reward variance under the binary
reward**. Phase 4 as specified (first GRPO run with the binary reward) would therefore get no gradient from this
starting point. The CRPS reward does not have this problem: only 4% of groups were zero-variance. Details and
a recommendation are at the end.

Commits (local `main`):
```
6b80d63 17:47 phase 1: formula-graph problem generator for kinematics, dynamics, energy
487c6cf 17:58 phase 2: verifier (parser, pint units gate, CRPS and ablation scores, step credit)
044ae76 18:19 phase 3: baseline eval of Qwen2.5-1.5B-Instruct, rollout tape, calibration metrics
e9a2fed 19:17 phase 3: explicit greedy baseline, group-variance probe, prompt diagnostics
```
(MORNING.md itself is in a final small commit.) The full log is in [PROGRESS.md](PROGRESS.md). All 28
decisions are in [DECISIONS.md](DECISIONS.md).

---

## 1. What was completed

### Phase 1: Environment (87 tests)
- `physcal/env/`: kinematics, dynamics and energy families. Each family is one coherent multi-stage scenario
  (cart accelerate → cruise → brake; crate pulled, then released; block on ramp → rough patch → spring).
  Formulas are sympy relations with unique positive solve directions, and dimensions are pint-checked.
- Problems are linear chains of depth 1–5 with **honest depth**: the minimal derivation depth over all formulas
  of the family equals the chain length. Givens are **algebraically independent** (Jacobian rank), so no
  problem is over-determined. Givens are shown at 3 s.f. in random units (km/h, g, min, mN, ...), and ground
  truth is recomputed from the displayed numbers. Chains whose condition number exceeds 5 are rejected.
  0–2 irrelevant distractors per problem.
- Valid chain templates per cell (exhaustively enumerated): kinematics 42/181/251/73/24, dynamics
  43/158/290/250/97, energy 30/85/112/89/36 for depths 1–5.
- Tests: 1,000 seeds per family against an independent 30-digit sympy evaluation of the composed chain;
  hypothesis property tests; physics consistency (every formula holds in every sampled world); plus a
  30,000-seed stress run with 0 failures.
- **Hand-check the physics:** 5 random problems with full worked ground truth are in PROGRESS.md
  ("Five random problems for hand-checking"). I re-derived all of them by hand.
- Two real bugs were found and fixed with regression tests: over-determined givens (found while hand-checking)
  and a divide-by-exactly-zero after rounding (found by an eval seed).

### Phase 2: Verifier (107 tests)
- Tolerant parser (±, +/-, \pm, sci notation incl. ×10^k and unicode, LaTeX, `(μ ± σ) unit`, percent σ,
  `[unit]`). It's strict about structure: exactly one `<answer>`, no truncation.
- Pint units gate. Relative Gaussian CRPS as specified, plus `binary`, `log_score`, `abs_rel_error`, and an
  extra `crps_log` (see D15). Step credit within 2%. `reward()` returns `(float, breakdown)`.
- CRPS is verified against its integral definition. Monte Carlo confirms the expected CRPS is minimized at the
  true σ.

### Phase 3: Baseline eval (16 tests + runs)
- Rollout tape with `replay()`. **All 2,368 logged rollouts replay with bit-exact rewards.**
- Eval harness (`uv run python -m physcal.eval.run_eval`): stratified splits with disjoint seeds, PIT and
  reliability plots, per-depth and per-family tables, `--replot`, and `physcal.eval.rescore` to re-score a tape
  under any rule.
- Held-out-family and simulator splits are configured but skipped: they don't exist until phase 6 (D21).

---

## 2. Baseline numbers (`results/baseline/`)

Qwen/Qwen2.5-1.5B-Instruct, bf16, **no OOM** (no fallback needed). Pure greedy, ≤ 512 new tokens, prompt v1.
**Peak VRAM: 3.76 GiB** allocated by torch (3.84 GiB reserved). The device-wide peak was 4.95 GiB of 8 GiB,
including ~1.9 GB used by the Windows display. 400 problems took 451 s.

| split | n | accuracy (≤ 1%) | σ-free accuracy | 1σ coverage | 2σ coverage | parse fail | unit mismatch | median rel. error | median stated rel. σ |
|---|---|---|---|---|---|---|---|---|---|
| ID (depth 1–4) | 200 | **0.0%** | 0.5% | **0.7%** | 1.4% | **26.5%** | **0.0%** | 0.92 | 0.47% |
| depth 5 (extrapolation) | 200 | **0.0%** | 0.0% | **0.0%** | 3.2% | **21.5%** | **0.5%** | 0.93 | 1.0% |

- Coverage target is 68% (1σ) / 95% (2σ). The model is wildly over-confident: it states ~0.5% σ while its
  median error is ~92%.
- Parse failures are mostly answers without ± (20% of all completions), plus some 512-token truncations.
- σ-free accuracy = the same completions re-scored with the `binary` rule, which doesn't require ±.
- Mean CRPS reward: −2.41 (ID), −2.31 (depth 5). Mean step credit ≈ 0.01.

I checked by hand that the zeros come from the model, not the verifier. The model restates the givens and
formulas, then writes a placeholder-like guess (e.g. writes `t2 = t_tot - t1 - t3`, truth 14.1 s, answers
`1.0 s`).

### Diagnostics run for phase 4 (inference only, all logged under `results/`)

| what | result |
|---|---|
| Group probe, k = 8, T = 1.0, prompt v1 (48 ID problems) | 0/384 correct. **Binary: 100% of groups have zero variance.** CRPS: 4% |
| Same, prompt v2 (an instruction to "reason first") | 1/384 correct. Binary: 97.9% zero-variance. CRPS: 14.6% |
| Greedy, prompt v3 (one worked example in the system prompt) | ID accuracy **1.5%** (σ-free 2.0%), but **45% (ID) / 68% (depth 5) of completions hit the 512-token cap** |
| Depth-1 wording | 96–98% of depth-1 answers skip the calculation entirely, caused by the generator's sentence "...<steps> may be empty". Replacing it gives 34% answer-only, but still 0% accuracy |
| Qwen's default `repetition_penalty=1.1` (silently applied in my first run) | slightly worse (35% parse fail, 11% truncated). Kept as `results/baseline_qwen_genconfig_rep1.1/` for reference |

---

## 3. Decisions to review

All 28 are in DECISIONS.md. These are the ones I'd most like you to look at, in priority order:

1. **D13, failures score the rule's floor (−5 under CRPS), not a literal 0.** Under CRPS every valid answer
   scores < 0, so a literal "reward 0" for malformed output would make garbage the best possible output. For
   the binary rule the floor is 0, matching the spec exactly.
2. **D15, the spec's relative CRPS is only approximately proper.** Dividing by the *truth* |y| re-weights
   outcomes. By Monte Carlo, the optimal reported μ is biased by −1% at 10% relative spread and by −11% at 30%,
   with σ inflated up to 7%. This matters for a small model and for RQ2. I kept `crps` as the default, per the
   spec, and added `crps_log` (CRPS on ln y, strictly proper and still scale-free) as a config option, with
   tests. **Decide before phase 5.**
3. **D26, decoding is explicit; the official baseline uses repetition_penalty 1.0.** Qwen's
   `generation_config.json` silently adds 1.1, which also penalizes copying prompt numbers. Both runs are kept.
4. **D28, prompt choice for training.** v1 is official. v3 (worked example) is the only prompt that moves
   accuracy off zero, but its "about 1%" uncertainty line would cue σ (confounding RQ2), and its verbosity
   collides with the 512-token cap. Also: whether to fix the depth-1 sentence (needs a generator version bump).
5. **D3, a family = one coherent multi-stage scenario**, with the full scenario text in every problem. This was
   needed to make depth-5 chains unambiguous. Kinematics depth 5 is narrow: 24 chains over 5 targets (D5).
6. **D8, problem identity is (generator_version, family, depth, seed)**, not just (version, seed), because the
   curriculum picks the cell. The version string includes a hash of the generator config.
7. **D20, greedy eval** (deterministic). Trained checkpoints must be evaluated the same way.

The rest are lower stakes but worth a skim: D1 (Python 3.11 and the uv/HF caches live inside the project),
D4 (truth from displayed values), D6 (Jacobian-rank independence), D7 (condition number ≤ 5), D9 (held-out
cells refused in the generator; eval seeds ≥ 1e9), D10 (dedicated distractors), D11 (plausibility bounds),
D16–D17 (parser leniency; `<steps>` optional), D18 (depth-1 step score = 0), D21 (held-out/simulator splits
skipped until phase 6).

---

## 4. BLOCKED

**Nothing.** No failure persisted after 3 attempts.

One housekeeping item: `uv python install` also created the symlink `~/.local/bin/python3.11`, outside the
project. My attempt to remove it was denied by the permission system, so it is still there. It is harmless and
points into `.cache/uv-python/`. Remove it with:

```bash
rm ~/.local/bin/python3.11
```

---

## 5. Recommended next step

**Fix the starting point before any GRPO.** The phase-4 plan ("binary reward, reward curve goes up") cannot
work from a model with pass@8 = 0. Every group would be zero-variance. I'd spend an hour on this, using only
the tools that already exist:

1. **Choose the prompt and the depth-1 wording.** Write a v4 prompt: v3's worked example without the
   "about 1%" σ hint, and with terser arithmetic so it fits in 512 tokens. Also replace the depth-1
   "<steps> may be empty" sentence, bumping the generator to 1.1.0 and re-running the baseline.
2. **Measure with the group probe** until the binary zero-variance fraction is clearly below 100% on the easy
   cells:
   ```bash
   uv run python -m physcal.eval.run_eval --config eval_group_probe
   ```
   (Edit `prompt:` in `configs/eval_group_probe.yaml`. Each run takes ~8 min.)
3. **If the binary pass rate stays ≈ 0:** either (a) treat "binary cannot bootstrap from zero, CRPS can" as a
   phase-4/RQ1 result, and run the first GRPO pilot with CRPS just to validate the training loop; or (b) do a
   short SFT warm-start on generator-written solutions (`worked_solution()` already produces the steps). (b)
   changes the research story, so it's your call.
4. **Then install phase-4 deps:** `uv add unsloth trl`. Unsloth 2026.10.3 accepts our torch 2.14.1 (it requires
   `<2.15`) and pins `trl<=1.13.0`; I checked PyPI but didn't install. Decide `crps` vs `crps_log` (D15)
   before phase 5.

Note on VRAM: inference peaked at 3.8 GiB, so a 4-bit 1.5B model with LoRA and G = 8 should fit within the
~6 GB available to us. That's still to be confirmed in phase 4.

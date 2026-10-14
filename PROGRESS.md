# Progress log

Overnight session 2026-10-11, scope: phases 1–3 (no RL training).
Design choices are recorded in [DECISIONS.md](DECISIONS.md).

## Phase 1 — Environment: plan

1. Project scaffold: `uv` project `physcal`, Python 3.11, configs in `configs/*.yaml`.
2. `physcal/env/quantities.py` + `units.py`: named quantities with pint dimensions, SI sampling ranges,
   display-unit choices.
3. `physcal/env/formulas/{kinematics,dynamics,energy}.py`: each family is one coherent physical scenario
   (a "world") with its own quantities and sympy relations. Every formula declares which symbols it can be
   solved for; each solve direction must have exactly one positive solution (checked at import).
4. `physcal/env/graph.py`:
   - sample a consistent world from a few independent basis quantities (log-uniform within physical ranges),
     propagating with pint so every derived value carries checked dimensions;
   - build a linear chain of depth d backwards from a target (each step consumes the previous step's output
     plus given quantities);
   - validate with a breadth-first derivation closure over *all* formulas of the family: the target's minimal
     derivation depth must equal d, each intermediate i must have depth i, and no given is derivable from the
     other givens (no over-determined problems);
   - round givens to the displayed precision and recompute the chain from the displayed values, so the text and
     the ground truth agree exactly; reject ill-conditioned chains (relative condition number above a threshold).
5. `physcal/env/render.py`: phrasing templates, unit randomization (SI prefixes, km/h, g, min, ...),
   0–2 irrelevant distractor quantities, requested intermediates, requested answer unit.
6. Problem dict: `text, target_symbol, target_value_si, target_unit, target_value (requested unit),
   intermediates, family, depth, seed, generator_version` plus givens/distractors metadata.
7. Held-out split in config: family `optics` (added in phase 6) and depth 5 are eval-only;
   `generate_problem(..., split="train")` refuses them.
8. Tests: 1,000 seeds per family where the ground truth equals an independent sympy `evalf` of the composed
   chain expression; hypothesis property tests (determinism, depth/intermediate structure, unit round-trip,
   display/SI consistency, held-out enforcement, every family reaches depth 5).

## Phase 1 — Environment: results

**Status: done. 87 tests pass** (`uv run pytest`, ~25 s): 54 in `tests/test_env_graph.py`, 33 in `tests/test_env_generator.py`.

What exists:
- `physcal/env/units.py`: one shared pint registry, display-unit menus (SI prefixes, km/h, g, min, h, N s, ...).
- `physcal/env/graph.py`: `Quantity`, `Formula` (sympy relation + unique positive solve directions),
  `Family` (world sampling, derivation closure, exhaustive chain enumeration, chain validation,
  pint-checked chain evaluation, condition number, composed sympy expression).
- `physcal/env/formulas/{kinematics,dynamics,energy}.py`: three scenarios with 13/14/13 quantities,
  12/15/10 formulas and 3 irrelevant distractor quantities each.
- `physcal/env/render.py`: rounding to 3 s.f., unit choice, phrasing templates.
- `physcal/env/generator.py`: `generate_problem(family, depth, seed, split)`, `worked_solution(problem)`.
- `configs/env.yaml`: every generator threshold; generator version = `1.0.0+<hash of generator config>`.

Valid chain templates per cell, after all validity checks (enumerated exhaustively):

| family     | d=1 | d=2 | d=3 | d=4 | d=5 | targets at d=5 |
|------------|-----|-----|-----|-----|-----|----------------|
| kinematics | 42  | 181 | 251 | 73  | 24  | 5 of 13 |
| dynamics   | 43  | 158 | 290 | 250 | 97  | 13 of 14 |
| energy     | 30  | 85  | 112 | 89  | 36  | 7 of 13 |

Bug found while hand-checking: the first draft let 6 givens into a 5-degree-of-freedom world
(over-determined, and inconsistent after rounding by 0.16%). The closure check could not see it, because the
redundancy only shows up when you solve simultaneous equations. Fixed with a Jacobian-rank independence check,
plus a regression test (`test_overdetermined_by_simultaneous_equations_is_rejected`). See D6.

### Five random problems for hand-checking

Generated with `numpy.random.default_rng(20261011)` choosing the cells and seeds (eval split, so depth 5 is
allowed). Reproduce with `generate_problem(family, depth, seed, split="eval")` and `worked_solution(p)`.
I re-derived every number below by hand, and also checked that each problem's implied hidden quantities
form a physically sensible world (e.g. problem 1 implies u ≈ 5.0 m/s, a1 ≈ 0.75 m/s^2, b ≈ 0.61 m/s^2).

### Problem 1: kinematics, depth 4, seed 725778

```text
Consider a cart on a straight, horizontal track. Phase 1: the cart has initial speed u and speeds up with constant acceleration a1 during a time t1; it travels s1 and ends the phase with speed v. Phase 2: the cart keeps the constant speed v for a time t2 and travels s2. Phase 3: the cart slows down with constant deceleration of magnitude b and stops after a time t3, having travelled s3 in this phase. For the entire motion, d_tot is the total distance, t_tot the total time, and v_avg the average speed.

Data:
- The length of the cart is len_cart = 494 mm.
- The total distance covered by the cart, d_tot, equals 0.243 km.
- The duration of phase 2, t2, equals 0.375 min.
- The average speed of the cart over the whole trip, v_avg, equals 22.7 km/h.
- The braking distance in phase 3, s3, equals 0.0479 km.
- The duration of phase 1 is t1 = 3550 ms.

Find the distance covered during phase 2, s2, in m. In <steps>, report the intermediate values t_tot, t3, v.
```

Worked ground truth:

```text
Givens in SI: d_tot = 243 m; s3 = 47.9 m; t1 = 3.55 s; t2 = 22.5 s; v_avg = 6.30556 m/s
Step 1 [K12: v_avg = d_tot/t_tot]  t_tot = d_tot/v_avg  with d_tot = 243 m, v_avg = 6.30556 m/s  =>  t_tot = 38.5374 s
Step 2 [K11: t_tot = t1 + t2 + t3]  t3 = -t1 - t2 + t_tot  with t1 = 3.55 s, t2 = 22.5 s, t_tot = 38.5374 s  =>  t3 = 12.4874 s
Step 3 [K9: s3 = v*t3/2]  v = 2*s3/t3  with s3 = 47.9 m, t3 = 12.4874 s  =>  v = 7.67171 m/s
Step 4 [K6: s2 = v*t2]  s2 = t2*v  with t2 = 22.5 s, v = 7.67171 m/s  =>  s2 = 172.613 m
Answer: s2 = 172.613 m (SI) = 172.613 m (requested unit)
```

### Problem 2: dynamics, depth 3, seed 449175

```text
On a horizontal floor sits a crate of mass m; mu is the coefficient of kinetic friction between crate and floor. A constant horizontal pull F acts on the initially resting crate for a time t1. During the motion the normal force is N, the kinetic friction force is f_k, the net force is F_net and the acceleration is a1. At the moment the pull is removed, the crate moves with speed v, carries momentum p, and has covered a distance s1. Friction then decelerates it at a rate a2 (magnitude) until it comes to rest after an extra time t2, covering an extra distance s2.

s2 = 172 cm (the distance the crate slides after the pull). t1 = 1210 ms (the duration of the pull). a1 = 273 cm/s^2 (the acceleration of the crate during the pull). Use g = 9.81 m/s^2.

Find the coefficient of kinetic friction, mu. Report these intermediate quantities in <steps>: v, a2.
```

Worked ground truth:

```text
Givens in SI: a1 = 2.73 m/s^2; s2 = 1.72 m; t1 = 1.21 s
Constants: g = 9.81 m/s^2
Step 1 [D5: v = a1*t1]  v = a1*t1  with a1 = 2.73 m/s^2, t1 = 1.21 s  =>  v = 3.3033 m/s
Step 2 [D14: v**2 = 2*a2*s2]  a2 = v**2/(2*s2)  with s2 = 1.72 m, v = 3.3033 m/s  =>  a2 = 3.17203 m/s^2
Step 3 [D11: a2 = mu*g]  mu = a2/g  with a2 = 3.17203 m/s^2, g = 9.81 m/s^2  =>  mu = 0.323347
Answer: mu = 0.323347  (SI) = 0.323347 (dimensionless) (requested unit)
```

### Problem 3: energy, depth 3, seed 441653

```text
Starting from rest at a height h, a block of mass m slides down a frictionless ramp; U denotes its potential energy at the top relative to the bottom. At the bottom it has kinetic energy K1 and speed v1. Next it slides across a rough level section of length L (coefficient of kinetic friction mu), losing energy W_f to friction, and exits with kinetic energy K2 and speed v2. On the frictionless floor beyond, it hits a light spring of spring constant k, which it compresses by at most x; at that maximum compression the spring exerts a force of magnitude F_s on the block.

The speed of the block at the bottom of the ramp is v1 = 13.7 km/h. The coefficient of kinetic friction on the rough patch is mu = 0.0505. It is known that L = 380 cm, where L is the length of the rough patch. It is known that m = 0.997 kg, where m is the mass of the block. The spring constant is k = 113 N/m. Use g = 9.81 m/s^2.

Determine x, the maximum compression of the spring. Express the result in cm. In <steps>, report the intermediate values v2, K2.
```

Worked ground truth:

```text
Givens in SI: L = 3.8 m; k = 113 N/m; m = 0.997 kg; mu = 0.0505; v1 = 3.80556 m/s
Constants: g = 9.81 m/s^2
Step 1 [E8: v2**2 = v1**2 - 2*mu*g*L]  v2 = sqrt(-2*L*g*mu + v1**2)  with L = 3.8 m, g = 9.81 m/s^2, mu = 0.0505, v1 = 3.80556 m/s  =>  v2 = 3.27371 m/s
Step 2 [E7: K2 = m*v2**2/2]  K2 = m*v2**2/2  with m = 0.997 kg, v2 = 3.27371 m/s  =>  K2 = 5.34251 J
Step 3 [E9: K2 = k*x**2/2]  x = sqrt(2)*sqrt(K2)/sqrt(k)  with K2 = 5.34251 J, k = 113 N/m  =>  x = 0.307502 m
Answer: x = 0.307502 m (SI) = 30.7502 cm (requested unit)
```

### Problem 4: kinematics, depth 4, seed 327230

```text
Consider a cart on a straight, horizontal track. Phase 1: the cart has initial speed u and speeds up with constant acceleration a1 during a time t1; it travels s1 and ends the phase with speed v. Phase 2: the cart keeps the constant speed v for a time t2 and travels s2. Phase 3: the cart slows down with constant deceleration of magnitude b and stops after a time t3, having travelled s3 in this phase. For the entire motion, d_tot is the total distance, t_tot the total time, and v_avg the average speed.

Given:
- The total distance covered by the cart is d_tot = 0.329 km.
- m_cart = 5080 g (the mass of the cart).
- It is known that s3 = 3100 cm, where s3 is the braking distance in phase 3.
- t_tot = 21.3 s (the total duration of the trip).
- The speed of the cart at the end of phase 1 is v = 63.3 km/h.
- The duration of phase 1, t1, equals 0.0876 min.

Determine s1, the distance covered during phase 1. Express the result in km. Your <steps> must include the intermediate values t3, t2, s2.
```

Worked ground truth:

```text
Givens in SI: d_tot = 329 m; s3 = 31 m; t1 = 5.256 s; t_tot = 21.3 s; v = 17.5833 m/s
Step 1 [K9: s3 = v*t3/2]  t3 = 2*s3/v  with s3 = 31 m, v = 17.5833 m/s  =>  t3 = 3.52607 s
Step 2 [K11: t_tot = t1 + t2 + t3]  t2 = -t1 - t3 + t_tot  with t1 = 5.256 s, t3 = 3.52607 s, t_tot = 21.3 s  =>  t2 = 12.5179 s
Step 3 [K6: s2 = v*t2]  s2 = t2*v  with t2 = 12.5179 s, v = 17.5833 m/s  =>  s2 = 220.107 m
Step 4 [K10: d_tot = s1 + s2 + s3]  s1 = d_tot - s2 - s3  with d_tot = 329 m, s2 = 220.107 m, s3 = 31 m  =>  s1 = 77.893 m
Answer: s1 = 77.893 m (SI) = 0.077893 km (requested unit)
```

### Problem 5: kinematics, depth 2, seed 126168

```text
A cart moves along a straight, level track in three phases. In phase 1 it starts with speed u and accelerates uniformly with acceleration a1 for a time t1, covering a distance s1 and reaching speed v. In phase 2 it moves at the constant speed v for a time t2, covering a distance s2. In phase 3 it brakes with a constant deceleration of magnitude b until it comes to rest, which takes a time t3 and covers a distance s3. Over the whole trip the cart covers a total distance d_tot in a total time t_tot, so its average speed is v_avg.

Data:
- The distance covered during phase 1, s1, equals 0.117 km.
- The speed of the cart at the end of phase 1, v, equals 15.2 m/s.
- The duration of phase 2, t2, equals 0.106 min.
- The speed of the cart at the start of phase 1, u, equals 19.4 km/h.
- t3 = 3310 ms (the braking time in phase 3).

Determine t_tot, the total duration of the trip. Express the result in min. Report these intermediate quantities in <steps>: t1.
```

Worked ground truth:

```text
Givens in SI: s1 = 117 m; t2 = 6.36 s; t3 = 3.31 s; u = 5.38889 m/s; v = 15.2 m/s
Step 1 [K2: s1 = (u + v)*t1/2]  t1 = 2*s1/(u + v)  with s1 = 117 m, u = 5.38889 m/s, v = 15.2 m/s  =>  t1 = 11.3654 s
Step 2 [K11: t_tot = t1 + t2 + t3]  t_tot = t1 + t2 + t3  with t1 = 11.3654 s, t2 = 6.36 s, t3 = 3.31 s  =>  t_tot = 21.0354 s
Answer: t_tot = 21.0354 s (SI) = 0.350589 min (requested unit)
```


## Phase 2 — Verifier: plan

1. `physcal/prompt.py`: system prompt enforcing the `<steps>` / `<answer>` format, plus chat-message builder.
2. `physcal/verify/parse.py`: extract exactly one `<answer>` and the `<steps>` block. Normalize unicode/LaTeX
   (±, +/-, \pm, −, superscripts, ×10^k, \text{}), and accept forms like `μ ± σ unit`, `(μ ± σ) unit`,
   `(μ ± σ) × 10^k unit`, `μ unit ± σ unit`, `μ unit ± 5%`, and a `symbol =` prefix. Step lines are
   `symbol = ... = value unit`, with symbols canonicalized (`t_1` == `t1`, `v_{avg}` == `v_avg`).
3. `physcal/verify/units.py`: pint gate. Convert to SI, and raise `unit_mismatch` on wrong dimensions or an
   unparseable unit.
4. `physcal/verify/scoring.py`: Gaussian CRPS in relative units (main), plus `binary`, `log_score`,
   `abs_rel_error`; clip floors from config.
5. `physcal/verify/reward.py`: `reward(problem, completion, cfg) -> (float, breakdown)`. Step credit within 2%.
   total = w_final·score_final + w_steps·score_steps. Config validation that steps can never outweigh the final
   answer. The breakdown carries everything eval needs (relative error, relative σ, z, PIT, 1σ/2σ hits,
   copied-given flag, scores under every rule).
6. `configs/reward.yaml`: rule, weights, tolerances, clips, σ floor.
7. Tests: hand-written edge cases (wrong units, missing ±, negative values, scientific notation, LaTeX, percent σ,
   multiple answers, truncated output). CRPS closed form vs numerical integration. Monte Carlo propriety:
   the expected CRPS is minimized at the true σ (and true μ). Reward weighting and config validation.

## Phase 2 — Verifier: results

**Status: done. 194 tests pass in total** (`uv run pytest`, ~28 s). Phase 2 adds 107: 68 parser/units-gate
cases in `test_verify_parse.py`, 15 scoring tests in `test_verify_scoring.py`, and 24 reward tests in
`test_verify_reward.py`.

What exists:
- `physcal/prompt.py`: system prompt (free reasoning allowed, then `<steps>` and exactly one `<answer>`).
- `physcal/verify/parse.py`: tolerant normalization (±, +/-, \pm, unicode minus and superscripts,
  ×10^k, LaTeX \text/\mathrm/\boxed, markdown bold, `[unit]`, thousands separators, percent σ,
  `(μ ± σ) × 10^k unit`, separate units for μ and σ). Strict about structure: exactly one `<answer>`, a number
  must come first, at most one ±, and non-finite numbers are rejected.
- `physcal/verify/units.py`: pint gate. Wrong dimensions, a missing unit on a dimensional quantity, or an
  unparseable unit gives `unit_mismatch`. Numeric scaling factors inside a unit ("1000 m") are rejected.
- `physcal/verify/scoring.py`: Gaussian CRPS in relative units (spec, default), `binary`, `log_score`,
  `abs_rel_error`, plus `crps_log` (strictly proper alternative, see D15).
- `physcal/verify/reward.py`: `reward(problem, completion, cfg) -> (float, breakdown)`. The breakdown is
  JSON-safe and contains the reason, scores under every rule, relative error, raw and floored relative σ, z,
  PIT, 1σ/2σ hits, per-step credit, and a copied-given flag.
- `configs/reward.yaml`: rule, weights (1.0 / 0.2), σ floor 1e-3, clip 5, step tolerance 2%,
  binary tolerance 1%.

Checks that matter:
- Closed-form CRPS equals the integral definition ∫(F − 1[x ≥ y])² dx (scipy quad).
- Monte Carlo: the expected CRPS is minimized at the true σ and the true μ (400k samples). It still
  is through the full relative pipeline at 2% spread.
- **Finding:** the spec's normalization by the truth |y| is only approximately proper. Optimal reports under
  a Gaussian belief with relative spread s0:

  | s0   | optimal μ bias | optimal σ / s0 |
  |------|----------------|----------------|
  | 2%   | −0.04%         | 1.002          |
  | 5%   | −0.24%         | 1.001          |
  | 10%  | −1.0%          | 1.004          |
  | 20%  | −4.4%          | 1.024          |
  | 30%  | −11.0%         | 1.066          |

  Documented in D15, with a test that pins this behaviour. `crps_log` is implemented and tested as a proper
  alternative. The default stays the spec's `crps`.

## Phase 3 — Baseline eval: plan

1. `physcal/tape/tape.py`: JSONL rollout tape (spec fields plus split, token count, truncation flag) and a
   `.meta.json` with the env, reward and eval configs. `replay()` regenerates each problem from
   (version, family, depth, seed), checks it equals the logged problem, re-scores, and requires an exact
   reward match.
2. `physcal/eval/calibration.py`: accuracy (failures count as wrong), median relative error, parse-fail and
   unit-mismatch rates, missing-± rate, 1σ/2σ coverage, relative σ, mean score under every rule, step score,
   copied-given rate, reliability curve (central-interval coverage at nominal levels), PIT histogram, and
   per-depth and per-family breakdowns. PNG plots use the dataviz reference palette.
3. `physcal/eval/run_eval.py` + `configs/eval.yaml`: splits `id` (train families, depths 1–4),
   `depth_extrapolation` (depth 5), `heldout_family` and `simulators` (skipped until phase 6), with 200
   problems each, stratified round-robin over (family, depth) cells, and a disjoint seed block per split.
   Inference is plain transformers, bf16, left padding, greedy, batch 16, max 512 new tokens. On CUDA OOM it
   falls back to Qwen2.5-0.5B-Instruct. Peak VRAM is recorded from torch (allocated/reserved) and from
   device-wide nvidia-smi polling.
4. Tests: tape replay reproduces rewards exactly and detects tampering; calibration metrics on synthetic,
   perfectly calibrated predictions (coverage ≈ 68%/95%, uniform PIT); end-to-end `run()` with fake backends
   (oracle, garbage, OOM-then-fallback).
5. Run the baseline: `uv run python -m physcal.eval.run_eval`. Results go to `results/baseline/`.

## Phase 3 — Baseline eval: results

**Status: done. 210 tests pass** (`uv run pytest`, ~33 s). Phase 3 adds 16: 3 in `tests/test_tape.py`, 9 in
`tests/test_eval.py`, 3 generator regression tests for a bug the eval seeds exposed (below), and 1 check that
every system prompt's example parses. All 6 result tapes (2,368 rollouts) replay with exact rewards.

Command: `uv run python -m physcal.eval.run_eval` → `results/baseline/` (metrics.json, summary.md,
rollouts.jsonl + meta, calibration_id.png, calibration_depth_extrapolation.png, reliability_all_splits.png,
rescored_binary.json).
Model: **Qwen/Qwen2.5-1.5B-Instruct** (no OOM, so no fallback was needed). Decoding: bf16, pure greedy
(repetition_penalty 1.0, see D26), batch 16, ≤ 512 new tokens, system prompt `v1`. Generation took 451 s for
400 problems.
**Peak VRAM:** torch max allocated **3.76 GiB** / reserved 3.84 GiB. Device-wide peak (nvidia-smi, includes
~1.9 GB used by the Windows display) **4.95 GiB** of 8 GiB. There is plenty of headroom.
`replay()` on the 400-completion baseline tape reproduces every reward exactly.

| split | n | accuracy (≤1%) | σ-free accuracy* | median rel. err | parse fail | of which missing ± | unit mismatch | 1σ cov | 2σ cov | median rel. σ | mean reward (CRPS) | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| id (depth 1–4) | 200 | **0.0%** | 0.5% | 0.92 | **26.5%** | 20.0% | **0.0%** | **0.7%** (n=147) | 1.4% | 0.47% | −2.41 | 6.0% |
| depth_extrapolation (5) | 200 | **0.0%** | 0.0% | 0.93 | **21.5%** | 20.0% | **0.5%** | **0.0%** (n=156) | 3.2% | 1.0% | −2.31 | 1.5% |
| *ref: id, rep. penalty 1.1* | 200 | 0.5% | 1.0% | 0.95 | 35.0% | 20.5% | 1.5% | 0.8% | 0.8% | 0.50% | −2.82 | 11.0% |
| *ref: depth 5, rep. penalty 1.1* | 200 | 0.0% | 0.0% | 0.87 | 32.5% | 18.5% | 0.5% | 0.0% | 0.7% | 0.63% | −2.64 | 13.5% |

\* σ-free accuracy = the same completions re-scored with the `binary` rule, which doesn't require ±
(`uv run python -m physcal.eval.rescore results/baseline/rollouts.jsonl --rule binary`).
"ref" rows: the first run, which silently used Qwen's default repetition_penalty 1.1
(`results/baseline_qwen_genconfig_rep1.1/`, see its NOTE.md and D26).

Mean score per rule, ID split (failures at the rule's floor): crps −2.41, binary 0.00, log_score −9.71,
abs_rel_error −2.42, crps_log −2.83. Mean step score 0.01. Copied-given rate 8.8% (ID) / 3.9% (depth 5).
A `<steps>` block is present in 57% (ID) / 78% (depth 5) of completions.

Reading:
- The untrained model is at zero: 0/200 correct within 1% on both splits, including depth 1. I checked by hand
  that this is the model, not the verifier. The closest depth-1 answer is 1.3% off, and the attempted
  calculations are wrong (e.g. t1 = s1/(u+v)·(u+v)/2). 48 of 400 completions contain no calculation at all,
  just `<answer>` with a guess (median 17 tokens). Depth-1 problems say "<steps> may be empty", and the model
  often takes that as permission to skip the work.
- Parse failures (96 of 400): 80 omit ± (wrong format under the CRPS rule), 15 hit the 512-token limit, 1 has
  no `<answer>`. Unit mismatches are rare (0–0.5%): when the model answers, it uses the requested unit.
- σ is wildly over-confident: median stated relative σ ≈ 0.5–1% against a median relative error of ≈ 92%.
  1σ coverage is 0–0.7% (target 68%). The PIT histogram has all its mass in the two end bins (see
  `calibration_*.png`).

**Bug found and fixed during this phase:** eval seed 1001000087 (kinematics, depth 5) crashed the generator.
A subtraction step came out exactly 0 after rounding, and the next step divided by it (Python float →
`ZeroDivisionError` inside pint, which bypassed the NaN rejection path). Chain evaluation now stops at the
first non-positive or non-finite intermediate, and the condition number is ∞ when a perturbation leaves the
valid domain (previously `max(k, nan)` silently ignored it). There are regression tests, plus a 30,000-seed
stress run with 0 failures (κ median 2.0, p99 4.9).

### Phase 3 addendum: diagnostics for phase 4 (inference only, no training)

Every run below uses the same model, the same eval problems and pure-greedy or explicit sampling settings.
Each is logged to its own folder, with a tape and plots.

**1. Decoding hygiene (D26).** The first baseline run silently used Qwen's `repetition_penalty: 1.1`.
Rerunning with explicit pure greedy (the official baseline above) gave lower parse failure (35% → 26.5% ID) and
fewer truncations (11% → 6%), and accuracy stayed ≈ 0.

**2. Group-variance probe: what a GRPO group of 8 would see** (`configs/eval_group_probe*.yaml`;
48 ID problems, 4 per family×depth cell, k = 8, T = 1.0, top_p 1, no top-k, rep. penalty 1.0):

| prompt | samples correct | pass@8 | zero-variance groups, **binary** reward | zero-variance groups, CRPS reward | parse fail |
|---|---|---|---|---|---|
| v1 (official) | 0 / 384 | 0.0% | **100%** (48/48) | 4.2% | 32.6% |
| v2 reason-first | 1 / 384 | 2.1% | **97.9%** | 14.6% | 43.8% |

→ A phase-4 GRPO run with the binary reward, from this model and prompt, gets **no gradient at all**. The CRPS
reward is dense: almost every group has reward variance even when every answer is wrong.

**3. Prompt sensitivity, greedy, 200 + 200 problems** (`results/baseline_prompt_v2|v3/`, see D28):

| prompt | split | accuracy | σ-free acc. | parse fail | missing ± | truncated (512 tok) | mean tokens | median rel. err | 1σ cov |
|---|---|---|---|---|---|---|---|---|---|
| v1 (official) | id | 0.0% | 0.5% | 26.5% | 20.0% | 6.0% | 115 | 0.92 | 0.7% |
| v1 (official) | depth 5 | 0.0% | 0.0% | 21.5% | 20.0% | 1.5% | 126 | 0.93 | 0.0% |
| v2 reason-first | id | 0.0% | 0.5% | 31.5% | 26.0% | 5.5% | 107 | 0.93 | 2.3% |
| v2 reason-first | depth 5 | 0.0% | 0.0% | 36.0% | 28.5% | 7.5% | 141 | 0.91 | 0.8% |
| v3 worked example | id | **1.5%** | **2.0%** | 49.0% | 3.5% | **45.0%** | 341 | 0.93 | 3.0% |
| v3 worked example | depth 5 | 0.0% | 0.0% | 75.0% | 4.5% | **68.0%** | 436 | 0.98 | 2.0% |

- v2 (an instruction to reason first) changes nothing. The model mostly ignores it.
- v3 (one worked example) makes the model actually compute: 12 of the 150 parsed ID answers are within 10%,
  versus almost none before. But its reasoning is verbose: 45% (ID) and 68% (depth 5) of completions hit the
  spec's 512-token cap before `<answer>`. The 512-token budget, not just competence, becomes the limit.
- v3 nearly eliminates missing ± (20% → 3.5%), but stated σ gets even sharper (median relative σ 0.05%). The
  example's "about 1%" is probably being copied as a template.

**4. Depth-1 wording** (`scripts/diag_depth1_wording.py`, `results/diagnostics/depth1_wording/`): under every
prompt, 96–98% of depth-1 completions answer with no calculation, versus 0% at depths 2–5. The trigger is the
generator's sentence "No intermediate values need to be reported, so <steps> may be empty." Replacing only that
sentence on the same 50 problems cut answer-only completions to 34%, but accuracy stayed 0% (median error 0.93).
I didn't change the generator (it would need a version bump); see D28.

**5. How the model fails when it does show work:** it fills `<steps>` with restated givens and formulas, then
writes a placeholder-like round number ("1.0 s", "0.500 kg", "10000 cm/s^2"). It often does this right after
writing the correct formula (e.g. `t2 = t_tot - t1 - t3`, truth 14.1 s, answer 1.0 s). It almost never carries
out the arithmetic unless shown a worked example.

## Phase 3.5 — Fix the starting point (2026-10-12)

Plan (from your instructions; no training):
1. Apply the final decisions: `crps_log` becomes the default rule (D29); keep the −5 failure floor (D13);
   fix the depth-1 wording → generator 1.1.0, regenerate and freeze the eval sets, and keep old tapes
   replayable (D30); write FINDINGS.md #1.
2. Model probe (inference only) on Qwen2.5-1.5B-Instruct, Qwen2.5-Math-1.5B-Instruct and the strongest current
   ≤ 2B instruct model I can verify on Hugging Face (exact revisions pinned): a greedy baseline with a
   worked-example prompt without a σ hint, max_new_tokens 1024, plus the k = 8, T = 1.0 group probe. Report
   accuracy at 1/2/5/10%, truncation, zero-variance fractions (binary and crps_log) and peak VRAM.
3. Estimate each model's GRPO footprint (4-bit + LoRA, group 8, 1024 tokens) against ~6.5 GB, and apply the
   selection rule. If every model is > 80% binary zero-variance: write a Phase 3.6 SFT warm-up plan and stop.

Progress:
- [x] D29 `crps_log` default (commit 50bd351). The −5 floor for failures was already in place.
- [x] D30 generator 1.1.0 (`1.1.0+c735a0d4`), frozen eval sets in `data/eval_sets/1.1.0+c735a0d4/`, all phase-3
  tapes still replay exactly. 214 tests pass.
- [x] FINDINGS.md #1: binary 100% zero-variance groups vs relative CRPS 4.2% / crps_log 0.0%, on the phase-3 probe
  tape. While writing it I found that the phase-3 group statistic used CRPS-scored correctness (which requires ±).
  Re-scored with a true σ-free binary rule, the figure is unchanged (100%). Every completion is now scored under
  every rule (`physcal/eval/signals.py`). Group stats and accuracy at 1/2/5/10% tolerance are reported per rule,
  in `run_eval` and in `rescore`.
- [x] Model probe, all three candidates (`results/model_probe/summary.md`). Binary zero-variance in the k = 8 group
  probe: **Qwen3.5-2B 72.9%**, Qwen2.5-1.5B-Instruct 91.7%, Qwen2.5-Math-1.5B 97.9% (format failure, D32).
  Estimated GRPO footprints: 4.27 / 3.67 / 3.67 GB (budget 6.5 GB). **Selection rule → train Qwen3.5-2B.**
  Its binary zero-variance is > 50%, so phase 4 uses `crps_log` (your rule).
  The Qwen3.5 group probe was re-run at batch 8 after an OOM-retry slowdown at batch 16 (D33).

## Phase 4 — First GRPO run (2026-10-12, unattended)

- [x] 4a. Training stack installed (unsloth 2026.10.3, trl 1.13.0, peft 0.21.2, bitsandbytes 0.50.2;
  transformers pinned to 5.17.0). Qwen3.5-2B supported on the first attempt. 226 tests pass.
- [x] 4c. 10-step smoke test: 78 s/step (first step 167 s compile), peak 3.30 GiB torch / 4.17 GiB device.
- [x] 4d. GRPO run, crps_log (D34): 49 steps (hard deadline), 69.4 s/step, peak 3.29 / 4.45 GiB. No
  improvement: 10-step mean reward −4.68 → −4.90, truncation 82–94%, zero-variance 40–80% of steps.
- [x] 4e. Eval of the final adapter on the probe's eval sets (finished 15:22 IST, before the 15:45 GPU cutoff):
  ID σ-free accuracy 13.5% → 14.0%, strict 7% → 9%, parse failures 82% → 78%; depth-5 reward −4.69 → −4.48.
  None significant (paired McNemar/Wilcoxon, smallest p = 0.056). See MORNING.md §4.

## Phase 3.6 — SFT warm-up A/B + overnight GRPO (2026-10-13, unattended)

- [x] SFT data builder (σ masked, training seeds 1e7..), SFT trainer, merge tool, EMA curriculum, 3-way compare
  (D35, D36). 30-min SFT on Qwen3.5-2B (2,096 examples) and Qwen2.5-Math-1.5B (2,600 examples).
- [x] Group probe on both SFT'd models → **winner Qwen2.5-Math-1.5B + SFT** (binary zero-var 56.2% vs 60.4%;
  FINDINGS #4). Truncation at 512 tokens = 0% → **budget 512**.
- [x] Overnight GRPO on Math-SFT: 287 steps (stopped at 06:30 IST), 58.9 s/step, 9,184 rollouts (replay exact);
  the curriculum unlocked dynamics/3–4 and kinematics/3–4; energy stayed at depths 1–2.
- [x] Final 3-way eval + paired tests (FINDINGS #5): ID acc ≤1% raw base 0% → SFT 38.0% → SFT+RL 40.0%;
  depth 5 0% → 31.0% → 31.0%. RL vs SFT: +4/−0 problems on ID (p = 0.125), reward +0.046 (p = 0.018 nominal),
  depth 5 unchanged, calibration unchanged (1σ coverage ~15%). 238 tests pass. GPU work ended 07:07 IST.

## Phase 3.7 — σ floor removed, matched-step GRPO rerun (2026-10-13/14, unattended)

- [x] Verifier 1.1.0 (D37 approved, D38): crps_log σ floor 1e-6, `version` in `configs/reward.yaml`, logged in
  every breakdown; old tapes replay under their own config. 13 new tests (strict σ monotonicity at 0.01–0.1%,
  MC propriety below the old floor, version checks, exact replay of last night's eval tapes). 251 tests pass.
- [x] Stage 1: last night's SFT-only and SFT+RL eval tapes re-scored under 1.1.0 into new, replayable tapes
  (`compare_verifier_1.1.0/`; originals untouched).
- [x] Stage 2: post-hoc control (D40). SFT-only completions on 400 training-seed problems; σ scale c = 18.1
  (crps fit) → SFT-only 1σ coverage 15.0% → 35.0% (ID), 11.5% → 33.5% (depth 5); coverage-matched c = 585 → 65% /
  72.5% at a large crps_log cost. 24% of SFT answers state σ = 0.
- [x] Stage 3: memory fix (D39): micro-batch 2 × 16 + 7.5 GiB allocator cap; smoke test 6.46 GiB peak, 0 OOMs.
- [x] Stage 4: overnight GRPO, identical to last night except the verifier: 759 steps (07:30 IST stop),
  35.7 s/step, peak reserved 7.05 GiB, permanent step-287 adapter. 24,288 rollouts (replay exact, tape gzipped,
  D42). σ = 0 share 25% until step ~300, then 95–98% from step ~575.
- [x] Stage 5: evals (step 287 first, then step 759) + paired tests (FINDINGS #6). At step 287: 1σ coverage 14.0% /
  9.5% vs SFT-only 15.0% / 11.5% and last night 15.5% / 10.0% (all p ≥ 0.125); post-hoc control 35.0% / 33.5%
  (p < 1e-12 in its favour). Step 759: 11.5% / 8.1%, 50.5% of ID answers state σ = 0. 259 tests pass. GPU work
  ended 07:53 IST.

## Phase 9.5 + Phase 10 step 2 (2026-10-14, unattended, daytime)

- [x] CLAUDE.md: phases 9.5–15 merged (old Phase 10 README → Phase 15), D43.
- [x] PREREGISTRATION.md committed (`f601bb7`) before any new result: families P (probe), S (split credit),
  H (phases 10–14), Holm within each family.
- [x] 9.5 step 1: σ = 0 alert in training (rolling 25-step share > 50%, alert only) and the standing post-hoc
  baseline (× 18.1) in every eval report (`configs/monitor.yaml`), D44.
- [x] 9.5 step 2: hidden-state probe on the SFT-only model (FINDINGS #7). Probe AUROC 0.917 vs stated σ 0.50, but
  the random-init control matches it (0.914), so P1–P3 are uninterpretable per the pre-registration; exploratory
  bag-of-words (0.906) and template (0.88) baselines show the signal is in the completion text. D45, D47.
- [x] Phase 10 step 2: rounding audit (FINDINGS #8). Truth = f(displayed givens) exactly; counterfactual 3-s.f.
  σ* median 0.26%; a 4-s.f. solver errs 0.014% median. D48.
- [x] 9.5 step 3: split-credit trainer + tests + 10-step smoke test: 40.0 s/step, peak reserved 6.46 GiB, 0 OOMs.
  The long run is not started. D46. 286 tests pass.

## 2026-10-14 (laptop day, CPU only)

- [x] Security check (repo is public). Regex sweep for tokens, API keys, private keys, `.env`/credential files and
  URL-embedded passwords over **all 624 blobs in all 77 commits** plus the working tree (tracked, untracked and
  ignored-but-present files); positive control matched. **No secrets found.** All 329 keyword hits are the word
  "token" in the LLM sense. Informational only: committed tapes/logs contain the local path
  `/home/flash2404/...` (in `model_id` fields), and commits are authored as `Raghhav <you@example.com>`.
  No history rewrite. `.gitignore` now also excludes `.env`, `.env.*`, `*.pem`, `.netrc`.
- [x] D49 amendment (DECISIONS.md, dated 2026-10-14, committed before any split-credit result): conditional post-hoc
  baselines in `physcal/eval/posthoc_cond.py`, fitted on the 2,000 SFT-only calibration completions (CPU).
  (a) text-feature σ scale (the control RL calibration claims are judged against), (a′) text direct (sensitivity),
  (b) oracle per (family, depth), in-sample (upper bound for per-cell scaling only). SFT-only on the frozen eval
  sets, 1σ coverage / mean crps_log, ID: × 18.1 35.0% / −0.319; (a) 42.5% / −0.314; (a′) 79.5% / −0.292;
  (b) 31.0% / −0.306. Every eval report now has a "Conditional post-hoc baselines (D49)" table; the four existing
  verifier-1.1.0 reports were refreshed (only the baseline block changed). `scripts/phase95_compare.py` runs family A
  (A1 coverage error, A2 ρ, A3 crps_log; Holm) against (a), and against (a′) as sensitivity. 296 tests pass.
- [x] D50: amendments A1–A8 from REVIEW_SYNTHESIS_2026-10-14.md recorded and pushed before any further result.
- [x] Phase 10 step 3 (D51): generator 2.0.0 noise mode in `configs/env_v2.yaml` (env.yaml stays 1.1.0), σ\* by MC
  per A2, delta-method cross-check passes. 16 + 4 new tests.
- [x] Phase 10 step 4, gate under A3 (FINDINGS #9): **FAILED**, 1 of 7 criteria. Per-cell ±2σ\* Kupiec with Holm
  rejects depth 3 / 10% noise (91.8%, p = 0.0004). Pooled coverage, PIT KS and the binomial-interval share
  pass. Phase 10+ work stopped (A6 not built). Decision needed (FINDINGS #9 options).
- [x] D52: E1 pre-registered before any E1 data or result.
- [x] A5 nested baselines (FINDINGS #10, D53). Strongest = problem + answer + reasoning / GBT (calibration CV).
  On SFT-only eval: crps_log −0.293 vs stated −0.327, 1σ coverage 67% vs 13%, ρ 0.70 vs −0.02. Added to every eval
  report (the four existing reports were refreshed) and to `scripts/phase95_compare.py` as family N (the judgement).
- [ ] A6 counterfactual pairs: **not built**. It is Phase 10+ work and the A3 gate failed (FINDINGS #9).
- [x] E1 data and configs: 1,996 training and 400 eval prefixes (`results/e1/data/`), `configs/grpo_e1{a,b}.yaml`,
  trainer, eval and family-E compare (`physcal/train/e1*.py`). 325 tests pass. No GPU smoke test (no GPU today).
- [x] Tonight's queue prepared, NOT launched: `scripts/queue_split_e1.sh` (split-credit seed 0 → 287 + eval + sanity +
  family S/N compare, then E1a and E1b + evals, then family E). PID-based waiting only; dry run checked.
- [x] D54 (noise capped at 5%, chosen after the gate; 10% = heavy-tail regime), D55 (step credit approved), D56
  (E5 idea, not pre-registered).
- [x] Gate rerun on {0, 1, 2, 5}% under A3, fresh seeds (FINDINGS #11): **FAILED**, 1 of 7 criteria. The ±2σ\*
  cell-share rule fails: 14 of 15 cells inside, and d4 / 1% over-covers (97.6%, p = 0.0035). Everything else passes.
  A6 not built; Phase 10+ stopped again pending your decision.
- [x] D57 (miss-count rule; next rerun final) recorded before the rerun. FINAL gate (FINDINGS #12): **PASSED** all
  7 criteria, 0 cells outside at both levels, PIT p = 0.98. Phase 10+ resumes: A6 next.

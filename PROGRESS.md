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


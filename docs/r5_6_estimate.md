# R5-6 estimate: what fresh counterfactual generalization would take (D146 item 5; nothing built)

R5-6 (`REVIEW_SYNTHESIS_ROUND5_2026-10-28.md`): "Unseen coefficient changes inside value-dependent relations (new
Jacobians, same surface form), plus unseen formula compositions. New frozen set; own pre-registration."

Written 2026-10-30 from CPU checks against the R4-1 SFT data and the generator code. No set and no model output
exists for it.

## What the checks found (CPU, today)

1. **The SFT data already covers almost every formula composition the graph allows.**
   - The R4-1 rows (3,000; 3 families; depths 1–4) contain 203 of the 204 formula-to-formula transitions that occur
     in any valid chain:

     | family | transitions in the graph | seen in SFT |
     |---|---|---|
     | kinematics | 78 | 77 |
     | dynamics | 86 | 86 |
     | energy | 40 | 40 |

   - Only 3 chains, all in kinematics, contain the one unseen transition.
   - **Whole chains:** many are unseen at depths 2–4 (kinematics d3: 148 of 251 seen; dynamics d3: 166 of 290), and
     depth 5 is entirely unseen. Depth 5 is already the held-out depth of A6c / A6b, though.
   - **So "unseen compositions" cannot come from the current graph.** A filter would give only "chains not seen as a
     whole", whose every step and every step-to-step hand-over was trained.
2. **New relations cannot go into the three existing families.**
   - `generator_version` hashes the `generator` section of `configs/env.yaml`, not the formula modules.
   - So adding a formula to kinematics, dynamics or energy would change `Family.chains(depth)` without changing the
     version. Every old seed would then regenerate a different problem, which breaks `replay` and every frozen set.
   - **The safe vehicle is a new family.** `configs/env.yaml` already declares `heldout_families: [optics]` (eval
     only, never sampled for training).
3. **The SFT rows store each chain's per-given log-sensitivities** (`explicit.jsonl`: `sensitivities`, with value and
   power-law flag). The training support of the Jacobians can therefore be read per chain without regenerating
   anything.

## The two parts, and what each changes

### Part J: new Jacobians, same surface form (a pair construction; no generator, formula or verifier change)

- **Literal "coefficient changes"** (e.g. `s1 = u*t1 + a1*t1**2/3`) are physics the model cannot know. The answer
  would be wrong by construction, unless the coefficient is stated in the text, and that changes the surface form.
  **Not recommended.**
- **The proposal: move value-dependent chains into Jacobian regions outside the SFT support,** with the same formulas,
  phrasing and value ranges.
  - **The construction:** a new world filter on A6c's `allocation` construction (`physcal/env/pairs.py`,
    `configs/pairs.yaml`).
    - Only value-dependent chains, i.e. the complement of `power_law_chains`.
    - Of the world draws, keep only those whose per-given log-sensitivities lie outside the range seen for that chain
      in the R4-1 rows, by a margin. One example: a term that is minor in every training world becomes dominant.
    - Every quantity stays inside its existing `bounds_si` (so magnitudes stay in distribution), with κ ≤ 5.
    - Then A6c's allocation pair is built as usual: identical displayed values, the high level swapped, and
      σ\*_hi / σ\*_lo ≥ 1.5.
  - **Replay:** a new construction hash (`pair_version`). `generator_version` stays at 2.0.0, and old sets replay
    unchanged.
  - **Code:** ~150–250 lines + config + ~6–8 tests. The tests:
    - the filter rejects in-support worlds;
    - the pairs replay exactly;
    - the displayed strings are identical within a pair;
    - the bounds are respected.
- **The main risk is feasibility.** Out-of-support Jacobians inside the value bounds may be rare in some cells.
  - **First step:** a CPU feasibility count like D95's (no model): the share of chains and worlds that qualify per
    family × depth × level pair.
  - Cells that cannot reach 34 pairs are reported, not padded.
  - ~1 h.

### Part U: unseen compositions (a new held-out family, optics)

- **The family:** thin lens, mirror, refraction (Snell), magnification, lensmaker.
  - About 10–12 relations, written as the existing family modules are (65–84 lines each): quantities with ranges and
    bounds, intros, distractors, a world sampler with validity (e.g. object outside the focal length for a real image).
  - **Units:** lengths and angles only. No diopter target, so neither pint's registry nor the verifier changes.
  - **No change to the existing families or their seeds:** the family registry gains an entry, and `check_split`
    already excludes held-out families from training.
- **Tests:** the phase-1 set for a family:
  - 1,000 seeds with ground truth equal to sympy numeric evaluation;
  - hypothesis property tests;
  - honest depth and independent givens;
  - chains exist at depths 1–4;
  - allocation pairs build and replay.
  - Plus the Phase 10 generator self-consistency check on its noisy problems (σ\* intervals cover the truth at 68 % /
    95 %, Kupiec per cell).
- **The confound to state in the pre-registration:** the model has never seen optics at all, so its vocabulary and
  every formula are new, not only their composition.
  - A6c's primary statistic uses all pairs with both σ parsed, so propagation can still be measured.
  - A weak result would not separate "new composition" from "new domain".
- **The cheaper alternative,** with no generator change: whole chains unseen in SFT at depths 2–4. It must be labelled
  "unseen chains (every transition trained)", not "unseen compositions".

## The frozen set

- **Cells:** depths 1–4 (the trained depths, so the novelty is the Jacobian or the family, not depth) × A6c's three
  level pairs × 34 pairs.
  - J: 408 pairs = 816 problems.
  - U: 408 pairs = 816 problems.
  - Both: 1,632 problems.
- **Built as R4-5's sets were** (D112 / D113):
  - fresh pair-seed offsets (e.g. 2.4e9, 2.5e9);
  - SHA-256 manifests, FROZEN.md;
  - `load_frozen_split` refuses them unless `PHYSCAL_CONFIRMATORY=1`.
- **CPU:**
  - construction and MC σ\* (N = 2,000 per problem, as the generator already does), plus the self-consistency check:
    ~1–2 h at 1 thread, less with more threads while the GPU is idle;
  - comparators (σ\*, RSS, X1, the heuristics): minutes.

## GPU hours (evaluation only; no training)

R5-6 tests the existing adapters. The natural models are the confirmed Qwen D-pos-v2 (#32, seeds 0–2) and its
implicit control B-v2 (seeds 0–2): greedy, batch 16, prompt v5, the R4 confirmatory eval config.

| | rate (from the R4-5 confirmatory evals, 3,060 problems each) | J + U (1,632 problems) | J only (816) |
|---|---|---|---|
| D-pos-v2, 3 seeds | 2.0–2.3 s / problem (6,162–6,873 s; seed 2's 10,245 s had the old 10-min pauses) | ~3.1 h | ~1.6 h |
| B-v2, 3 seeds | 1.15–1.25 s / problem (3,501–3,771 s) | ~1.7 h | ~0.9 h |
| load + R2-1 reload check | ~5 min per run, 6 runs | ~0.5 h | ~0.5 h |
| temperature (D143: 2–4 % last night) | +5 % | ~0.3 h | ~0.15 h |
| **total** | | **~5.6 h (one night)** | **~3.2 h** |

- **Llama is not included.** Its pilot is NO-GO (#38). At ~6 s / problem it would add ~1.4 h per arm for J alone.

## Build effort (my time) and the order

1. **Pre-registration:** a D-entry before any set exists, fixing the following.
   - **The hypotheses:** the A6c-statistic slope on J > 0, and on U > 0; one-sided; Holm over the two.
   - **The practical threshold:** the slope ≥ 0.5, as C2.
   - **The estimator:** `r4_analysis.slope`, B = 10,000, seed-mean over 3 seeds, as #32.
   - **The comparators:** RSS, X1, σ\*.
   - **What would count as failure.**
2. **Part J:** ~3–4 h (construction, tests, feasibility), then the frozen build.
3. **Part U:** about a day (the family, its tests, the self-consistency check), then the frozen build.
4. **The eval:** one queue (resumable evals, D142), one night.
5. **The analysis:** mostly reuse of `r4_confirm` / `r4_analysis` (~2 h of glue and tests).

## Decision for you

- **J only:** about half a day of code and one ~3.2 h GPU night. It answers "does propagation survive Jacobians the
  model never saw?" cleanly.
- **J + U:** about 1.5–2 days of code and a ~5.6 h night. It adds "unseen compositions", but on a family that is new in
  every respect.
- **The cheap U alternative** (unseen whole chains) adds nothing beyond J's night, but the novelty is weak.
- **R5-7's rule:** optics is cut from the paper unless R5-6 is done.

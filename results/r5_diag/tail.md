# R5-2 tail autopsy and R5-3 component swaps: D-pos-v2, frozen R4-5 set, 3 seeds (D133)

**Kind:** R5-2 descriptive autopsy and R5-3 diagnostic component swaps (D133); not tests, not methods. 3060 problems (3039 with predictive draws).

## R5-2: the worst 5% of problems by regret against σ* (per seed, pooled over seeds)

456 problem-seeds. Share of each seed's regret in its worst 5%: 83%, 69%, 76%. Computation-correct μ: 7%, 9%, 9% in the worst 5% vs 14%, 15%, 15% overall.

| primary cause | worst 5% (pooled) | share | all problems (pooled) |
|---|---|---|---|
| sensitivity: value-dependent: formula | 177 | 39% | 1095 (11.9%) |
| sensitivity: value-dependent: substitution (own wrong intermediates) | 87 | 19% | 733 (8.0%) |
| chain selection: wrong number of terms | 35 | 8% | 245 (2.7%) |
| copy: decimal shift (×10^-1) | 33 | 7% | 58 (0.6%) |
| copy: decimal shift (×10^1) | 31 | 7% | 35 (0.4%) |
| sensitivity: power-law exponent | 28 | 6% | 474 (5.2%) |
| format: no answer σ | 26 | 6% | 43 (0.5%) |
| sensitivity: value-dependent: formula (no expression) | 18 | 4% | 129 (1.4%) |
| format: no parsable σ line | 7 | 2% | 22 (0.2%) |
| aggregation | 5 | 1% | 482 (5.3%) |
| copy: decimal shift (×10^-2) | 5 | 1% | 6 (0.1%) |
| copy: decimal shift (×10^2) | 1 | 0% | 1 (0.0%) |
| sensitivity: transcription (S line right, σ-line coefficient wrong) | 1 | 0% | 13 (0.1%) |
| copy: decimal shift (×10^3) | 1 | 0% | 1 (0.0%) |
| sensitivity: value-dependent: substitution (values) | 1 | 0% | 5 (0.1%) |

**Offsetting errors:** on problems with a material sensitivity error, the model's own aggregation and copy move σ back the other way by a material amount on 8%, 11%, 6% (seeds 0, 1, 2; n = 606, 599, 614); median share of the sensitivity error undone: 1%, -1%, -4%.

**By depth (worst 5%, pooled):**

| cause | d1 | d2 | d3 | d4 | d5 |
|---|---|---|---|---|---|
| sensitivity: value-dependent: formula | 13 | 24 | 33 | 20 | 87 |
| sensitivity: value-dependent: substitution (own wrong intermediates) | 20 | 16 | 29 | 17 | 5 |
| chain selection: wrong number of terms | 0 | 5 | 4 | 6 | 20 |
| copy: decimal shift (×10^-1) | 8 | 4 | 6 | 9 | 6 |
| copy: decimal shift (×10^1) | 4 | 4 | 5 | 11 | 7 |
| sensitivity: power-law exponent | 0 | 1 | 2 | 3 | 22 |
| format: no answer σ | 1 | 12 | 1 | 7 | 5 |
| sensitivity: value-dependent: formula (no expression) | 0 | 0 | 0 | 1 | 17 |
| format: no parsable σ line | 0 | 1 | 2 | 2 | 2 |
| aggregation | 0 | 1 | 1 | 0 | 3 |
| copy: decimal shift (×10^-2) | 0 | 0 | 5 | 0 | 0 |
| copy: decimal shift (×10^2) | 1 | 0 | 0 | 0 | 0 |
| sensitivity: transcription (S line right, σ-line coefficient wrong) | 0 | 0 | 0 | 0 | 1 |
| copy: decimal shift (×10^3) | 1 | 0 | 0 | 0 | 0 |
| sensitivity: value-dependent: substitution (values) | 0 | 0 | 0 | 0 | 1 |

**Trimmed expected score (each source drops its own lowest 5% of problems):**

| source | full | 95%-trimmed |
|---|---|---|
| dpos2_a | -0.0397 | -0.0335 |
| sigma_star | -0.0370 | -0.0317 |
| rss | -0.0399 | -0.0338 |
| x1 | -0.0392 | -0.0333 |

## R5-3: one component exact at a time (diagnostic; seed means; problems that do not decompose keep (a) except under 'exact')

RSS-to-σ* gap: 0.00296. Decomposable: 97.6%, 95.9%, 96.3%.

(a)'s slope here grades every pair (unanswered members at the floor), as for (b) and (c); K1's uses answered pairs only.

| variant | expected crps_log | − (a) [95% CI] | − RSS [95% CI] | − X1 [95% CI] | share of RSS→σ* gap | A6c slope [90% CI] |
|---|---|---|---|---|---|---|
| (a) as stated | -0.0397 | – – | +0.0002 [-0.0003, +0.0007] | -0.0005 [-0.0010, -0.0001] | +7% | +0.837 [+0.715, +0.958] |
| exact sensitivities (model's levels, aggregation, copy) | -0.0384 | +0.0013 [+0.0006, +0.0019] | +0.0015 [+0.0009, +0.0020] | +0.0008 [+0.0001, +0.0013] | +51% | +0.970 [+0.874, +1.066] |
| exact levels (model's sensitivities, aggregation, copy) | -0.0397 | -0.0000 [-0.0000, +0.0000] | +0.0002 [-0.0003, +0.0007] | -0.0005 [-0.0010, -0.0001] | +7% | +0.838 [+0.717, +0.960] |
| exact aggregation (model's terms and copy) | -0.0408 | -0.0010 [-0.0018, -0.0005] | -0.0008 [-0.0018, -0.0000] | -0.0015 [-0.0025, -0.0008] | -28% | +0.806 [+0.694, +0.916] |
| exact copy, i.e. σ = the line's X (= (b)) | -0.0397 | +0.0000 [-0.0005, +0.0005] | +0.0002 [-0.0004, +0.0008] | -0.0005 [-0.0012, +0.0001] | +8% | +0.857 [+0.737, +0.975] |
| model's terms, exact aggregation and copy (= (c)) | -0.0417 | -0.0020 [-0.0047, -0.0003] | -0.0018 [-0.0045, -0.0000] | -0.0025 [-0.0053, -0.0007] | -60% | +0.826 [+0.716, +0.934] |
| fully exact: the builder's line, unrounded (ceiling) | -0.0370 | +0.0028 [+0.0023, +0.0033] | +0.0030 [+0.0028, +0.0031] | +0.0022 [+0.0021, +0.0024] | +100% | +0.998 [+0.996, +1.001] |

**Expected crps_log by depth:**

| source | d1 | d2 | d3 | d4 | d5 |
|---|---|---|---|---|---|
| a | -0.0355 | -0.0372 | -0.0397 | -0.0422 | -0.0440 |
| S | -0.0360 | -0.0360 | -0.0375 | -0.0412 | -0.0412 |
| L | -0.0355 | -0.0372 | -0.0397 | -0.0422 | -0.0440 |
| A | -0.0360 | -0.0374 | -0.0402 | -0.0431 | -0.0470 |
| C | -0.0356 | -0.0368 | -0.0394 | -0.0419 | -0.0447 |
| AC | -0.0404 | -0.0371 | -0.0399 | -0.0431 | -0.0482 |
| exact | -0.0340 | -0.0352 | -0.0367 | -0.0398 | -0.0391 |
| sigma_star | -0.0340 | -0.0352 | -0.0367 | -0.0398 | -0.0391 |
| rss | -0.0371 | -0.0381 | -0.0395 | -0.0426 | -0.0423 |
| x1 | -0.0362 | -0.0374 | -0.0388 | -0.0417 | -0.0419 |

# Update 2, 2026-10-14 night: gate passed, A6 built

- **D57**, recorded before the rerun: A3's cell-share rule is replaced by a miss-count test (reject if
  P(Binomial(n_cells, 0.01) ≥ misses) < 0.01, which allows 1 miss of 15). D57 calls the old rule a design error,
  found before the first rerun and confirmed by #11. FINDINGS #9 and #11 stand. The next rerun was declared final.
- **FINAL gate (FINDINGS #12): PASSED all 7 criteria.** Fresh seeds, {1, 2, 5}%, 10,005 problems:
  - pooled coverage 68.32% / 95.71% (p = 0.92 / 0.20);
  - no Holm rejection, and 0 cells outside at either level;
  - PIT KS p = 0.98;
  - all 500 noise-0 problems exact.
- **A6 built (D58):** matched pairs share the template, phrasing, units, distractors, noise level and measurement
  errors; only the world differs, chosen so that σ\* differs by ≥ 2× (delta method).
  - **Frozen set:** 510 pairs / 1,020 problems; median MC σ\* ratio 2.2–3.6× per cell.
  - **Analysis:** the D50 slope test (Δ ln stated σ on Δ ln σ\*, computation-correct pairs).
  - 332 tests pass.
- **Next for A6:** an eval path that runs a model on a frozen problem file (needed once a noise-trained model
  exists, Phase 12).
- **Tonight's queue is unchanged** (1.1.0 runs).

---

# Update, 2026-10-14 evening (after your decisions)

- **D54:** the main experiments use noise {0, 1, 2, 5}%; 10% is a "heavy-tail" regime scored only by crps_log
  against the MC distribution, with no coverage claims. D54 records that the cap was chosen after seeing the
  gate. The generator (`2.0.0+bdbb2279`) is unchanged.
- **D55:** step credit on displayed-value intermediates is approved.
- **D56:** E5 (distill the A5 calibrator into the model) is logged as an idea, not pre-registered.
- **Gate rerun at ≤ 5%:** 10,005 fresh problems, A3 rules (FINDINGS #11). **FAILED** on one criterion only. At
  ±2σ\*, 14 of 15 cells sit inside their 99% binomial interval; with 15 cells the rule needs all 15, which D54
  flagged before the run. The miss is depth 4 / 1%, which over-covers (97.6%, p = 0.0035; Holm does not reject
  it). Everything else passes:
  - pooled coverage 67.6% / 95.5% (p = 0.16 / 0.92);
  - PIT KS p = 0.28;
  - all 500 noise-0 problems exact.
- **A6 is not built, and Phase 10+ is stopped again.**
- **Your call** (FINDINGS #11 has the options): amend the share rule (e.g. "at most ⌈5% of cells⌉ outside per
  level") and rerun on fresh seeds; redesign the cell grid; or accept this as a false positive, recorded as
  post hoc.
- **Tonight's queue is unaffected:** split credit and E1 use generator 1.1.0.

---

# Report: 2026-10-14 laptop day (CPU only), and tonight's queue

**Short version.** No GPU was used, and every step is committed and pushed. **325 tests pass.**

1. **Security:** the repo is clean. Nothing secret was found in 624 blobs across 77 commits or in the working
   tree. History was not rewritten.
2. **Review amendments A1–A8:** recorded as D50 and pushed *before* any further result.
3. **Baselines:**
   - This morning's text-feature amendment (D49) was built first. A5's nested baselines now supersede it.
   - **A5 (FINDINGS #10):** the SFT model's own text predicts its error far better than its stated σ: 67% 1σ
     coverage and ρ 0.70, against 13% and ρ ≈ 0.
4. **Phase 10 noise mode:** generator 2.0.0 is built, and σ\* matches the delta method.
5. **The A3 gate FAILED (FINDINGS #9), on one criterion:** at depth 3 / 10% noise, the ±2σ\* interval covers
   91.8% (Holm-rejected). Per your rule, **Phase 10+ work stopped here, and A6 was not built.** You need to decide
   what to do with the gate (§4).
6. **E1 is ready:** pre-registered (D52), with data, configs, trainer, eval and tests all in place. Its training
   step has had **no GPU smoke test** (no GPU today).
7. **Tonight's queue is ready and not launched** (§6).

---

## 1. What ran (IST, all CPU, `nice`, ≤ 4 threads)

| step | result | commit |
|---|---|---|
| Security scan: full history + working tree | clean | `604fe41` |
| D49 conditional post-hoc baselines (your morning brief) | built; later superseded by A5 | `6eeef4b` |
| Review synthesis committed | – | `f8b24c4` |
| **D50: A1–A8 recorded, before any further result** | – | `5f2a846` |
| Generator 2.0.0 noise mode + A3 gate code (committed before it ran) | – | `162c573`, `a56eea5` |
| Gate run once, under A3 | **FAILED**, 1 of 7 criteria | `71ca64d`, `0dbc20f` |
| D52: E1 pre-registered (before any E1 data) | – | `fbb29a7` |
| A5 nested baselines; E1 data, configs, trainer, eval, compare | – | `e5d1b6f` |
| Tonight's queue script (not launched) | – | `5ca3347` |

## 2. Security check

- **Scope:** a regex sweep over all 624 blobs in all 77 commits, plus tracked, untracked and ignored-but-present
  files. It looked for tokens (HF, GitHub, OpenAI, Anthropic, AWS, Google, Slack), private keys, `.env` and
  credential files, and passwords embedded in URLs. A positive control confirmed the patterns match.
- **Result: nothing found.** All 329 keyword hits are "token" in the LLM sense.
- **Informational only:**
  - Committed tapes and logs contain your local path `/home/flash2404/…` (in `model_id` fields).
  - Commits are authored as `Raghhav <you@example.com>`.
- **Changes:** `.gitignore` now also excludes `.env*`, `*.pem` and `.netrc`. History was not rewritten.

## 3. Baselines: D49, then A5

**D49 (this morning, before the reviews).** All numbers are SFT-only on the frozen eval sets, fitted on the 2,000
calibration completions only.

| SFT-only σ | ID 1σ cov | ID crps_log | depth-5 1σ cov | depth-5 crps_log |
|---|---|---|---|---|
| stated | 15.0% | −0.320 | 11.5% | −0.357 |
| × 18.1 (standing) | 35.0% | −0.319 | 33.5% | −0.354 |
| (a) text-feature σ scale | 42.5% | −0.314 | 45.5% | −0.341 |
| (a′) text direct (ignores the stated σ) | 79.5% | −0.292 | 77.0% | −0.332 |
| (b) oracle per cell, in-sample | 31.0% | −0.306 | 34.5% | −0.340 |

- **The stated σ is worse than no σ.** On calibration CV, one constant relative σ beats any scale of the stated σ.
- **A5 supersedes D49's judgement rule (D50).** The D49 rows stay in reports, labelled "superseded".

**A5 (FINDINGS #10).**
- **Setup:** four nested feature sets × {linear, GBT}, with 5-fold CV grouped by template on the calibration
  split. The strongest σ was chosen before any eval use: **problem + answer + reasoning, GBT**.
- **On the SFT-only eval (pooled, 399 answered):**

| σ source | mean crps_log | 1σ cov | ρ(σ, err) |
|---|---|---|---|
| stated σ | −0.327 | 13.3% | −0.02 |
| metadata only | −0.303 | 63.2% | 0.49 |
| + problem text | −0.296 | 68.4% | 0.65 |
| **+ answer + reasoning, GBT (strongest)** | **−0.293** | **66.9%** | **0.70** |

- **p channel:** Brier 0.111, AUROC 0.913.
- **Where it shows up:** every eval report now has an A5 table, and the four existing verifier-1.1.0 reports were
  refreshed (only their baseline block changed).
- **How tonight's run is judged (family N):** `scripts/phase95_compare.py` compares the split-credit model's σ
  with the strongest A5 baseline **applied to the split-credit model's own completions**. μ is identical, so
  only σ differs.
  - N1: coverage error; N2: ρ; N3: crps_log. Holm-corrected.
  - "Beats the baseline" = N2 and N3 supported.
- **Context from a dry run:** with uniform-credit RL@287 standing in, the stated σ loses on all three
  (ρ −0.02 vs 0.66, crps_log −0.359 vs −0.327).

## 4. Phase 10: noise mode built, gate FAILED

- **Noise mode (D51):**
  - **Config:** generator 2.0.0 lives in `configs/env_v2.yaml`. `configs/env.yaml` stays at 1.1.0, so tonight's
    runs and the frozen sets are untouched (1.1.0 output is byte-identical).
  - **Display:** noisy givens show 5 s.f. and "± p%", plus a sentence explaining what ± p% means.
  - **Truth:** the clean chain output.
  - **σ\* (A2):** the CRPS-optimal log-scale, centred at ln f(x_meas), from 4,000 MC draws.
  - **Pairing:** problems are paired across noise levels by construction.
  - **Cross-check:** MC agrees with the delta method within 6% at 1–2% noise.
- **Gate (A3), one run of 10,000 problems:**

| criterion | result |
|---|---|
| pooled Kupiec 68% / 95% | 67.97% (p 0.52) / 95.22% (p 0.27), pass |
| per-cell Kupiec + Holm, ±1σ\* | pass |
| per-cell Kupiec + Holm, ±2σ\* | **d3 / 10%: 91.8%, p = 0.0004: FAIL** |
| ≥ 95% of cells inside the 99% binomial interval | 100% / 95% (19 of 20), pass |
| pooled PIT KS | p = 0.57, pass |
| noise 0 exact | 500 of 500 |

- **The superseded every-cell rule fails too** (5 cell tests have p < 0.05). It was never used; A3 decides.
- **Diagnosis (post hoc, same run):** the MC predictive distribution itself is fine (the PIT in that cell is
  uniform, p = 0.64). The Gaussian summary is too thin-tailed at 10% noise:
  - Standardized deviations have excess kurtosis 1.06.
  - The MC's own 95.45% interval covers 93.6% in that cell, and 95.44% pooled over all 10%-noise problems.

**Decision needed** (FINDINGS #9 lists the options):
1. Keep the gate as is: the noise dial is invalid at 10%.
2. Cap the dial at 5%. This is a choice made after seeing the run, so it needs a fresh gate run.
3. Grade against the MC quantiles instead of a Gaussian σ\*. This changes A2 and also needs a fresh gate run.

Until you decide, A6 and all other Phase 10+ work stay stopped.

## 5. E1 (pre-registered D52, prepared)

- **Data:** 1,996 training prefixes (calibration split; 4 excluded) and 400 eval prefixes. Each is the SFT
  completion cut right after "±".
- **Configs:** `grpo_e1a.yaml` (free σ) and `grpo_e1b.yaml` (σ mapped to nine bins). Both use the split-credit run's
  hyperparameters: 4 × 8 per step, lr 5e-6, ≤ 300 steps, permanent checkpoint at 150, 24-token suffix.
- **Tests:** family E = E1a/E1b vs the SFT stated σ (P) and vs the strongest A5 σ on the same completions (N),
  Holm over four.
- **CPU checks:**
  - The prefix plus the SFT's own suffix reproduces the logged SFT reward exactly on all 400 eval problems.
  - E1 tapes replay exactly.
  - An echo model run through eval and compare gives Δ = 0.
- **Risk:** the Unsloth training step for E1 has never run (no GPU today). If it fails, the queue logs it and
  moves on.

## 6. Tonight: launch the queue

```bash
nohup scripts/queue_split_e1.sh > results/queue_split_e1/nohup.log 2>&1 &
```

- **Order:**
  1. Split-credit seed 0 to step 287 (permanent checkpoint, ~3.2 h at the smoke test's 40 s/step).
  2. Its eval, then `sanity.json` (no OOM, parse failure < 5%), then `phase95_compare.py` (family S as
     pre-registered + family N).
  3. E1a, then E1b, splitting the remaining time up to 08:30 IST equally. Each gets its eval at the final step
     and at step 150.
  4. `e1 compare` (family E).
- **E1 runs regardless of the sanity verdict:** your brief replaced "seed 1 only if sane" with E1. The verdict is
  still recorded.
- **Options:** `GPU_DEADLINE=07:30` changes the cutoff, and `DRY_RUN=1` prints the plan.
- **Watching (PIDs only):**
  - `results/queue_split_e1/queue.pid` (the chain) and `<stage>.pid` (the current child), via
    `kill -0 $(cat …)`.
  - The chain log: `results/queue_split_e1/chain.log`.
  - In the split-credit run: the `ALERT` lines in `events.jsonl` and `split_credit.jsonl`.
  - In the E1 runs: `train_metrics.jsonl` (`parse_ok_rate`, `mu_changed_rate`, `bin_entropy_bits`).
- **Morning outputs:**
  - `<split run>/compare/phase95_family_S.md` (S1a–S3, N1–N3, the superseded D49 tests) and `sanity.json`.
  - `results/e1/family_E.md`.
- **Tape size:** E1 tapes store full prefixes and may pass 50 MB; gzip them before committing (D42).

## 7. Decisions to review

1. **Gate (FINDINGS #9):** keep it, cap the dial at 5%, or grade against MC quantiles. This blocks Phase 10+ and A6.
2. **D49 (a′) vs (a):** moot now that A5 decides. Note that A5's strongest baseline also beats (a′).
3. **D51 [REVIEW]:** step credit in noisy problems uses intermediates computed from the *displayed* values; the
   clean ones are stored. Distractors are noisy too.
4. **D51:** 2.0.0 is a separate config file rather than a bump of `env.yaml`, to keep tonight's runs and the
   frozen sets on 1.1.0.
5. **D52 (E1 choices):**
   - lr stays at 5e-6 (the DAPO token averaging already magnifies σ tokens).
   - E1b uses verifier mapping, not constrained decoding.
   - A suffix that changes the unit (and so μ) counts as a failure at eval.
6. **D50/A3 wording:** the per-cell Holm runs separately per level (20 tests each), and the "95% of cells" share
   is also counted per level.

## 8. BLOCKED / housekeeping

- **BLOCKED:** Phase 10+ (including A6) is waiting on your gate decision.
- **No GPU smoke test** of the E1 trainer.
- **Untracked:** `PHASES_10-15.md` is still there, and its content is in CLAUDE.md.
- **Not done:** CLAUDE.md is not yet updated with the A8 framing (noted in D50).

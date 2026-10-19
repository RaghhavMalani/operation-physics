# Report: 2026-10-19 day: your decisions applied (D77, D78), X3 prepared (D79), Phase 12 SFT data (D80), tonight's queue ready (D81)

The previous report (2026-10-18/19 night) is in git history: `git show cf74fb9:MORNING.md`.

**Short version.**
1. **Tests first:** 405 / 405 passed, so there was nothing to fix. After today's work: **437 / 437 pass.**
2. **Your six decisions are in:**
   - **R2-1:** the below-floor rule changed for future fingerprints only (seed 1 stays "inconclusive"), and the check
     is enforced (D77).
   - **FINDINGS #16:** the ×22.1 row is added.
   - **The Phase 12 base is pinned** (D78).
   - **FINDINGS #4 / #5:** they carry the R2-2 pointers.
3. **X3 is pre-registered (D79) and its data is built.**
   - **The teacher:** the out-of-fold A5 σ reproduces A5's own CV metrics exactly (max difference 0.0).
   - **The rows:** all 1,998 answered calibration completions are kept. Both students' targets read back through the
     verifier within 4.7%, with μ unchanged.
4. **Phase 12 SFT data (D80):** 3,000 noisy problems × 3 arms. **Not pre-registered;** six choices await you (§4).
5. **Tonight's queue is ready and NOT launched** (D81, `scripts/queue_x2_x3.sh`). Launch:
   `nohup scripts/queue_x2_x3.sh > results/queue_x2_x3/nohup.log 2>&1 &`

---

## 1. What was done (IST, all CPU)

| step | what | commit |
|---|---|---|
| 6 (first) | full test suite: 405 / 405 pass; nothing to fix | – |
| 1 + 2 | **D77:** R2-1's below-floor rule (future fingerprints only) and enforcement | `42bd61e` |
| 3 + 5 | FINDINGS #16: the ×22.1 row; FINDINGS #4 / #5: the R2-2 pointers | `703f795` |
| 4 | **D78:** the Phase 12 base pinned (snapshot dir + weights hash), checked by SFT, GRPO and eval | `836efb8` |
| a | **D79:** X3 pre-registration (committed before any X3 data) | `de83afd` |
| a | X3 data (teacher σ, control σ), code, config, tests | `86ee092` |
| b | **D80:** Phase 12 SFT data, three arms, draft prompt v5, tests | `c91036d` |
| c | **D81:** tonight's queue, X2 extraction / analysis code, the batch probe, the D78 GPU check, tests | `a25619c` |

## 2. Numbers

**FINDINGS #16, the new row** (pooled 400 problems; the same code as every other row, and the ×18.1 row reproduces
exactly):

| arm | acc | 1σ / 2σ cov | σ = 0 | median rel. σ | ρ | crps_log |
|---|---|---|---|---|---|---|
| SFT + post-hoc ×18.1 (bf16 fit, as recorded) | 36.8% | 34.2 / 42.2% | 16.8% | 1.15% | 0.05 | −0.329 |
| **SFT + post-hoc ×22.1 (4-bit fit, D69; standing baseline)** | 36.8% | **36.0 / 43.5%** | 16.8% | **1.40%** | 0.05 | **−0.328** |

**X3 data (`results/x3/data/data_report.json`):**
- **Teacher (a), the out-of-fold `problem_answer_reasoning/gbt`:** crps_log −0.1972, ρ 0.653, 1σ coverage 69.2%,
  identical to A5's recorded CV.
- **Control (b):** σ_log = |ln(μ/truth)| / √(ln 2), each answer's own CRPS-optimal σ. Its ρ with the error is 1.0 on
  these rows, by construction.
- **Targets:** median σ_log 5.2% (a) and 2.6% (b).

**Phase 12 data (`results/phase12_sft_data/manifest.json`):**
- **Size:** 62–63 problems per cell × noise level.
- **Median σ\*:** 1.8% / 3.5% / 8.7% at 1 / 2 / 5% noise.
- **The explicit line vs σ\*:** median |ln ratio| 0.013.

## 3. Today's code changes

- **R2-1:** `fingerprint.compare` takes the rule stored in each fingerprint, `below_floor: sigma_token` from now on.
  `check_adapter` raises on "fail" and warns on "inconclusive" / "no_fingerprint".
- **New `physcal/base.py` (D78):**
  - **What it does:** resolves the pinned commit to its local snapshot dir, checks the weights hash before a load and
    the loaded base after it.
  - **Where it is used:** `sft.py`, `grpo.py` (the fingerprint now records the base's identity, not an adapter's) and
    `run_eval.HFBackend`.
  - **Unchanged:** configs without `pinned_base` behave exactly as before; the resume digest of older runs is
    unchanged.
- **E1's fixed-prefix eval** takes custom rows. X3 and X2's method 2 use it; E1's defaults are unchanged.
- **`run_eval`:** two opt-in config keys.
  - **A `start` offset** for frozen splits, for X2's chunks.
  - **An allocator cap,** `max_reserved_gib`. Past it, torch raises OOM, which the backoff handles, instead of
    spilling into WSL shared memory.
- **New modules:**
  - `physcal/train/x3.py`, `physcal/train/sft_phase12.py`;
  - `physcal/eval/x2_hidden.py`, `physcal/eval/x2_analysis.py`;
  - `x2_pairs` gained chunked sampling, prefix checks and the method-2 stage.
  - **Prompt `v5_two_channel`** (draft). The old prompts are byte-identical, and a test checks their hashes.

## 4. Decisions to review

**Phase 12 SFT data (D80): open until the Phase 12 pre-registration.**
1. **`<confidence>c%</confidence>` as a separate block after `</answer>`.**
   - **Why:** ", confidence p" inside `<answer>` would break today's unit parsing. A separate block leaves the
     verifier untouched; the extension for c is additive.
   - **Your approval is still needed** for that verifier extension.
2. **Draft prompt v5:**
   - σ is defined as the uncertainty from the measurement errors, and 0 when every given is exact;
   - c is the probability that the calculation is right;
   - its worked example shows `± 0 km` and `90%`. The 90% is a cue, like the old "about 1%".
3. **c drawn uniformly from the bins and trained.** The model learns the allowed values with a flat prior and no
   signal. The alternatives are masking c or a fixed c.
4. **The explicit arm's line numbers are trained (default).** SFT then teaches first-order propagation. Masking them
   (one config flag) leaves the line's format to SFT and the values to RL. This choice decides what "explicit + RL"
   can add over its SFT control.
5. **The supervised σ\* arm uses token cross-entropy on a 2-significant-figure σ.** That approximates the synthesis's
   "log-scale loss"; it is not exactly one.
6. **The mix:** 25% of the problems are exact (σ = 0 in every arm, per A1 / R2-3); n = 3,000.

**X2 implementation choices (D81, fixed before any sample; they fill gaps in D74):**
- **Chunks:** 6 × 200 with torch seed 0 + chunk index (D74: seed 0); the chunks must form a prefix of the pool.
- **Method 2 is in the queue** although your list did not name it: D74 makes it primary, and it takes about 15 min.
- **Method 4's model** is `problem/gbt`, the lower CV Brier.
- **The cross-fitted probe** chooses both layer and C by inner CV.
- **A primary method with missing inputs** counts as p = 1.

**X3 (D79):** the control is each answer's own CRPS-optimal σ, |e| / √(ln 2). This is my reading of "a σ fitted
directly to true error labels". Say so if you meant something else; nothing has been trained yet.

## 5. Not verified (GPU) / housekeeping

- **No GPU stage has run.** Untested:
  - the batch probe, including `min_new_tokens` under Unsloth's generate;
  - Unsloth returning every layer's hidden states. The fallback is D74's transformers + bitsandbytes, triggered by a
    logit check;
  - X3 student training and its fingerprint;
  - Unsloth loading the pinned snapshot dir (the queue's last stage checks this).
- **Two inputs prepared on CPU and not committed** (size): the random-init control's weights
  (`results/x2/random_init_model`, 2.9 GB, content hash `13ededef3f3acbd4`) and the Phase 12 jsonl files (sha256 in the
  manifest).
- **X3's frozen-set eval has 399 prefixes:** one 4-bit SFT answer has no "±". It drops out of every arm alike.
- `PHASES_10-15.md` is still untracked.

## 6. Tonight

- **Expected time:** sampling takes about 3.3 h at batch 16, and less if the probe finds a larger batch. Then about
  45 min of X2 GPU stages, and the X3 pilot (about 2 h) if ≥ 2 h remain.
- **Launch time:** start by about 23:00 IST for everything to fit before 08:30.
- **Recommendation:** launch as prepared. If you want the X3 pilot guaranteed, launch earlier rather than lowering the
  2 h gate.

---

## Addendum, afternoon: your decisions on this report (D82–D85; CPU only)

| your decision | done | commit |
|---|---|---|
| 1, 2 | **D82:** the X3 control (per-answer CRPS-optimal σ) is confirmed; the X2 method-2 stage is approved | `f676435` |
| 3 | **D83:** verifier 1.2.0, `<confidence>` scored by Brier on the computation-correct event. Implemented and tested, no training | `db30a41` |
| 4 | **D84:** the explicit arm masks the propagation numbers and its σ (format only). The arm was rebuilt; its rows now carry their loss masks | `c73500f` |
| 5 | **D85:** v5's worked example shows `<confidence>c%</confidence>` instead of 90% | `bba06b8` |
| 6 | the other Phase 12 choices wait for Round 3: uniform trained c, the supervised arm's token loss, n and the noise mix, symbol names in the line | – |

**Verifier 1.2.0** (`configs/reward_v12.yaml`; the default `reward.yaml` stays 1.1.0):
- **Replay:** old tapes replay exactly. Both 1.1.0 tapes checked (800 rows) re-score to identical rewards **and identical
  breakdowns**, and the 1.0.0 replay tests pass. A `<confidence>` block is ignored under 1.1.0.
- **Two values for Round 3:**
  1. **`weights.confidence = 0`.** The reward equals 1.1.0's, and score_confidence goes to split credit through the
     breakdown.
     - **Why not positive:** it couples the channels. A wrong answer stated with c = 10% can then outscore a right one
       stated with c = 10%.
  2. **τ = 0.1% (A1).** On D69's 2,000 SFT answers, only **14.5%** are computation-correct at 0.1%, against **39.9%**
     correct within the verifier's 1%.
     - **Why:** mostly intermediate rounding (4-significant-figure intermediates).
     - **The risk:** at 0.1% the c channel would partly score precision. I did not change A1's τ.

**Tonight's queue: unchanged.** None of its files changed. Its dry run was re-run and plans the same 36 steps.
- **The one shared code change** is the verifier, which under the default 1.1.0 config behaves byte-identically
  (tested).
- **Tests:** 456 / 456 pass.
- **Launch:** `nohup scripts/queue_x2_x3.sh > results/queue_x2_x3/nohup.log 2>&1 &`

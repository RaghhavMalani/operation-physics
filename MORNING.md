# Report: 2026-10-18 (day, CPU only): your decisions, round-2 review, R2-1, X1, X2, tonight's queue

The previous report (2026-10-18 night, corrected evals) is in git history: `git show 4d73548:MORNING.md`.

**Short version.**
1. **Tonight's GPU queue is ready and NOT launched** (D75). Launch it by about **00:45 IST** if both uniform seeds
   should finish before 08:30:
   `nohup scripts/queue_uniform_seeds.sh > results/queue_uniform_seeds/nohup.log 2>&1 &`
   Order: the 4-bit SFT calibration completions → both D69 re-fits (CPU, in the background) → uniform-credit seeds 1
   and 2 to step 287 (corrected evals) → the 3-vs-3 robustness analysis.
2. **X1 (FINDINGS #17):** a text-only gradient-boosted model of σ\* predicts it well *across* problems (test R² 0.83).
   *Within* A6 pairs, though, its slope is only +0.063 (90% CI [+0.03, +0.10]) against an ideal of 1. It is
   labelled "partial propagation"; A6b is equivalent to 0. This is the bar for the Phase 12 LLM.
3. **R2-1 is built (D72).**
   - **What:** a training-time fingerprint (logits, the σ-token distribution, sampled σ, provenance) and the
     reloaded-model check; provenance is recorded in every eval tape.
   - **On CPU:** the faithful reload passes, and a dropped adapter or a bf16 merge of an RL-sized update fails.
   - **First real use:** tonight's two training runs. It is recorded only (`enforce: false`) until you have seen
     those numbers.
4. **Unsloth supports the Phase 12 plan** (one SFT → RL adapter, never merged; D70), by construction. Two findings
   for Phase 12 are in §4.
5. **X2:** pre-registered (D74); the candidate pool is frozen; the selection rule and metric are coded and tested.
   Its sampling needs about 4 GPU-hours on a later night.
6. **BLOCKED:** none. **One honest note:** after X1 finished at 10:10 the session sat idle until your "Try again" at
   about 22:00. That is why the queue was finished only at 22:10, still inside the launch window.

---

## 1. What ran (IST)

| time | step | commit |
|---|---|---|
| 09:48 | FINDINGS #13 heading annotated; **D69** (re-fit both baselines); **D70** (Phase 12 base, Unsloth check) | `92b868d` |
| 09:50 | Review synthesis committed; **D71** records R2-1 to R2-7, including the 3-vs-3 analysis plan | `55c8947` |
| 10:06 | **R2-1 (D72):** fingerprint + regression check + provenance; uniform seed configs; 392 tests pass | `3f70b9a` |
| 10:07 | **D73:** X1 pre-registered (before any fit) | `3eaab53` |
| 10:08–10:10 | X1 run (CPU, nice, 4 threads) | – |
| 22:00 | X1 results + FINDINGS #17 | `e8a7a62` |
| 22:10 | **D75:** tonight's queue (4-bit calibration generation, re-fit code, robustness script + tests) | `a9fdf28` |
| 22:12 | **D74:** X2 pre-registered (before any sample) | `d9a36c9` |
| 22:13 | X2 CPU parts: frozen pool, selection, metric, sampling config | `8714960` |

CPU only, all day. Fits ran with at most 4 threads under nice. The GPU was not used.

## 2. Results

### X1: can surface features propagate σ\*? Barely (FINDINGS #17, D73)

| σ source (problem text only) | test R² (ln σ\*) | A6 slope [90% CI] | A6 label | A6b slope [90% CI] | labelled verdict |
|---|---|---|---|---|---|
| noise (stated p only) | 0.62 | 0.000 [0.00, 0.00] | no propagation | 0.000 | no propagation |
| + TF-IDF | 0.79 | +0.002 [−0.00, +0.00] | no propagation | 0.000 | no propagation |
| **+ parsed magnitudes (primary)** | **0.83** | **+0.063 [+0.03, +0.10]** | **partial propagation** | −0.000 | **partial propagation** |
| reference: σ̂ = σ\* | – | +1.000 | full propagation | 0.000 | full propagation |

- **What it means:** what a text model knows about σ\* is mostly the noise level and the template. Neither varies
  within a pair.
- **Within-pair agreement:** Δ ln σ̂ correlates with Δ ln σ\* at 0.16, and has the right sign in 48.6% of pairs.
- **By depth:** the CI excludes 0 only at depth 2. At depth 5, which is outside the fit depths, the slope is −0.01.
- **The bar for Phase 12 (D73):** an LLM claim of "propagates beyond surface features" needs a higher A6 slope than
  0.063 on the same pairs (paired bootstrap), with A6b equivalent to 0.
- **Caveat:** X1 states no answer, so all 510 pairs enter its analysis. An LLM's analysis keeps only
  computation-correct pairs.

### R2-1 regression check on CPU (D72, `tests/test_fingerprint.py`)

| reload of a tiny trained LoRA | log-prob ratio (0 = identical to training, 1 = adapter gone) | verdict |
|---|---|---|
| unmerged, same base | 0 | pass |
| adapter dropped | 1.00 | fail |
| bf16 merge, update below half a bf16 step (RL-like) | 1.09 | fail |
| bf16 merge, large update | 0.003 | pass (the update survives the merge) |

## 3. Tonight's queue (D75; prepared, dry-run, not launched)

| # | stage | device | estimate |
|---|---|---|---|
| 1 | 4-bit SFT-only greedy completions, calibration 400 + 1,600 (D69) | GPU | ~40 min |
| 2 | re-fit the global scale (400; all-2,000 sensitivity) and A5 (`nested_baselines_4bit`) | CPU, background | ~20 min |
| 3 | uniform-credit seed 1 → step 287 (+ fingerprint) → corrected eval (+ fingerprint check) → sanity | GPU | ~3.3 h |
| 4 | uniform-credit seed 2, the same. Only if seed 1 is sane **and** ≥ 3.0 h of training time remain | GPU | ~3.3 h |
| 5 | `scripts/robustness_3v3.py`: D71 (A) problem-level seed means, (B) seed-level permutation / Hedges' g, (C) trajectories + figure | CPU | ~5 min |

- **Guards:** deadline 08:30 IST; > 87 °C → 10-min pause; PID-based waits; commit + push after every stage.
- **If launched late:** seed 2 is skipped by its gate, and the analysis runs 3 vs 2 (D71).
- **Checked on CPU:**
  - the dry run and `bash -n`;
  - the seed configs equal the seed-0 control apart from seed and run_id (test);
  - the calibration eval config equals D67's SFT control;
  - the robustness script end to end on the real evals with a stand-in uniform arm (test). Its (A) reproduces D62's
    numbers when the uniform arm is the single control.
- **Not checked:** any GPU stage, and in particular the fingerprint's first GPU run (Unsloth's inference-mode forward
  and generation).
  - **If it errors:** the training run is unaffected (the error goes to `events.jsonl` / `summary.json`), and the
    eval reports "no_fingerprint".

## 4. Decisions to review

1. **D69 [REVIEW]: how the re-fitted A5 is used.** I set it up as follows; tell me if you want the re-fit to be
   primary for #16 instead.
   - **From now on:** the re-fitted baselines are the standing ones (X2, Phase 12).
   - **FINDINGS #16's family N/E verdicts stand,** because D67 fixed them on the bf16-fitted A5. They get a
     sensitivity rerun with the re-fit (CPU, next). A flipped verdict would be reported to you, not silently
     replaced.
2. **D70: two Unsloth findings for Phase 12.** No existing result changes.
   1. **The base revision is not pinned:** an adapter load uses `revision=None` for its base.
   2. **The existing SFT adapter's base:** it was trained on `unsloth/Qwen2.5-Math-1.5B-Instruct-bnb-4bit` (cache
      snapshot `e24e953`). Unsloth's mapper silently dropped the pinned Qwen revision. `train.merge` then added the
      adapter to bf16 `Qwen/…@aafeb0f`.
   - **Proposal:** pin the 4-bit base explicitly in Phase 12, and assert its commit in both trainers and the eval
     loader.
   - **Optional smoke test (~5 GPU-min, not queued):** SFT 5 steps → RL 2 steps from that adapter → the fingerprint
     check.
3. **D72 tolerances.** They were fixed before any real fingerprint: log-prob and σ-token ratios ≤ 0.25 of the
   adapter's own effect, median ln σ within ln 2, effect floor 0.005 nats/token.
   - **Tonight:** `enforce: false`.
   - **After tonight's two runs:** decide whether to switch on enforcement. The tolerances do not change without a
     new DECISIONS entry.
4. **X2 GPU cost (D74).** k = 8 on 1,200 candidates is about 9,600 completions, roughly 4 GPU-hours, which crowds
   Monday's X3 pilot. The options are to sample a prefix (the frozen order is stratified) or to raise the batch size.
   Your call.
5. **R2 numbering:** the synthesis header says R2-1 to R2-9, but there are seven amendments. D71 records R2-1 to R2-7
   and treats X1–X3 as experiments.
6. **R2-2 and FINDINGS #4 / #5:** R2-2 says their SFT numbers describe an attenuated SFT. I recorded this in D71 and
   did not edit #4 / #5. Tell me if you want a "Read with" pointer, like #3 / #5 / #6 got for #15.

## 5. Code changes

- **`physcal/eval/fingerprint.py`** (new, D72):
  - **Hooks:** called by `grpo.train` and `e1.train` after the final save, and by `load_unsloth_4bit` for every
    adapter.
  - **`run_eval` / E1 eval meta:** gain `provenance`.
- **`physcal/eval/posthoc.py`:** `generate-4bit --part set|ext` and `refit-4bit`. `fit` takes paths, and its defaults
  reproduce ×18.1425.
- **`physcal/eval/nested_baselines.py`:** `fit --config`; new `configs/nested_baselines_4bit.yaml`.
- **New, X1:** `physcal/eval/x1_text_sigma.py` + `configs/x1_text_sigma.yaml`.
- **New, X2:** `physcal/eval/x2_pairs.py` + `configs/x2.yaml`; `data/eval_sets/1.1.0+c735a0d4/x2_candidates.jsonl`
  (frozen).
- **New scripts:** `scripts/robustness_3v3.py`, `scripts/queue_uniform_seeds.sh`.
- **`scripts/queue_sanity.py`:** takes an optional eval dir name.
- **Configs:** `grpo_overnight2_seed{1,2}.yaml`, `fingerprint.yaml`.
- **Tests:** +22 (fingerprint 8, uniform seed configs 2, X1 5, X2 4, 3-vs-3 3; the frozen-eval test also checks
  provenance). The full suite result is in §6.

## 6. BLOCKED / housekeeping

- **BLOCKED:** none.
- **Full test suite:** 405 collected, 404 pass, 1 skipped (GPU-only), 0 fail (CPU, CUDA hidden; 383 this morning).
- `PHASES_10-15.md` is still untracked (its content is in CLAUDE.md).

## 7. Recommendation

- **Tonight:** launch the queue soon. The uniform seeds settle R2-6, and the D69 re-fits feed X2 and Phase 12.
- **Next CPU work:**
  - the D69 family N/E sensitivity reruns;
  - the X2 analysis code (the split-credit σ suffixes, the hidden-state extraction at "±", the probe and its
    random-init control).
- **Phase 12's pre-registration:** after X2 reports, as the synthesis's schedule says. X1 already sets its A6 bar at
  a slope of 0.063.

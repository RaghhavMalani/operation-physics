# Findings

Research results worth keeping, each with enough provenance to reproduce it. Newest last.

---

## #1. On the untrained 1.5B model, a binary reward gives no GRPO signal; CRPS does (2026-10-12)

**Claim.** Sample 8 completions per problem from untrained Qwen2.5-1.5B-Instruct, as one GRPO group would.
Under the **binary** reward (correct within 1%), **100% of groups have zero reward variance**, so every
group-normalized advantage is 0 and GRPO receives no gradient. Under the **CRPS** reward only **4.2%** of
groups are zero-variance (relative CRPS, the reward logged at the time), and **0.0%** under `crps_log`, the
default since D29. A dense proper scoring rule still ranks wrong answers by how wrong and how over-confident
they are, so it gives a learning signal even when no answer is correct.

| reward (score_final only) | zero-variance groups (of 48) |
|---|---|
| binary (σ not required) | **48 / 48 = 100%** |
| binary + 0.2·step credit | 46 / 48 = 95.8% |
| crps, relative (spec; the reward logged at the time, incl. steps) | **2 / 48 = 4.2%** |
| crps_log (default since D29) | **0 / 48 = 0.0%** |
| abs_rel_error | 2 / 48 = 4.2% |
| log_score (clipped at −10) | 36 / 48 = 75.0% |

pass@8 = 0 and mean pass rate = 0: none of the 384 samples is within 1%. Only 0.3% / 1.6% / 2.1% are within
2% / 5% / 10%. The result holds at every depth: binary zero-variance is 100% in each of depths 1, 2, 3 and 4.

**Corroboration.** The same probe with prompt `v2_reason_first`
(`results/baseline_group_probe_prompt_v2/`): 1/384 samples correct, pass@8 = 2.1%, binary zero-variance
97.9%, crps 14.6%, crps_log 6.3%.

**Side observation.** `log_score` has much less signal than either CRPS (75% zero-variance). With σ that is
over-confident by orders of magnitude, almost every wrong answer hits the −10 clip. This is relevant if
log_score is kept as a phase-7 ablation.

### Provenance
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct` @ revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, bf16, plain
  transformers 5.19.0, torch 2.14.1+cu130, RTX 4060 Laptop (8 GB).
- **Problems:** generator `1.0.0+afcc2019`, ID split (families kinematics, dynamics, energy; depths 1–4),
  48 problems = 4 per (family, depth) cell, seeds 1000000000–1000000047 (the first 48 problems of the
  phase-3 ID eval set).
- **Sampling:** k = 8 per problem, `do_sample: true`, temperature 1.0, top_p 1.0, top_k 0 (off),
  repetition_penalty 1.0, max_new_tokens 512, torch seed 0, system prompt `v1` (`physcal/prompt.py`).
- **Config:** `configs/eval_group_probe.yaml` as of commit `e9a2fed`. The exact config used is also stored in
  the tape's meta (`eval_config`).
- **Tape (all 384 completions, replayable):** `results/baseline_group_probe/rollouts.jsonl` +
  `results/baseline_group_probe/rollouts.meta.json`. `replay()` reproduces every reward exactly.
- **Metrics as logged:** `results/baseline_group_probe/metrics.json` (`splits.id.groups`: binary 1.0, main
  reward (relative crps + steps) 0.0417).
- **Per-rule re-scoring:** `results/baseline_group_probe/rescored_all_rules.json`, produced by
  `physcal.eval.rescore.rescore("results/baseline_group_probe/rollouts.jsonl", "crps_log")` (commit
  after `8e34524`; uses the tape's logged reward config, with every rule swapped in turn).
- **Reproduce generation:** `git checkout e9a2fed && uv run python -m physcal.eval.run_eval --config eval_group_probe`
  (~8.5 min on the 4060).

---

## #2. Which small model gives GRPO a signal: format-following and length beat raw math skill (2026-10-12)

**Claim.** Among three ≤ 2B instruct models with the same prompt (v4, worked example), decoding and 1024-token
budget, only **Qwen3.5-2B** gets the binary zero-variance fraction under 80% (**72.9%**; pass@8 27.1%).
Qwen2.5-1.5B-Instruct is at 91.7%. **Qwen2.5-Math-1.5B-Instruct is at 97.9% despite having the best latent
physics**: it ignores the `<answer>` contract and writes `\boxed{}`, so 99.5–100% of its completions fail to
parse. Read out-of-band, its boxed answers are correct within 1% on 35% of ID problems (greedy).

| model (pinned revision) | greedy ID acc. ≤1% (σ-free) | ≤10% | depth-5 ≤1% | truncated @1024 (greedy) | binary zero-var (k=8) | crps_log zero-var | pass@8 |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct @ `989aa798` | 2.5% | 5.5% | 1.0% | 42.5% | 91.7% | 2.1% | 8.3% |
| Qwen2.5-Math-1.5B-Instruct @ `aafeb0fc` | 0.0% (boxed: 35%) | 0.0% (boxed: 46%) | 0.0% (boxed: 8.5%) | 28.0% | 97.9% | 95.8% | 2.1% |
| Qwen3.5-2B @ `15852e8c` (non-thinking) | 13.5% | 14.0% | 2.5% | 69.5% | **72.9%** | 56.2% | 27.1% |

**The binding constraint for Qwen3.5-2B is length, not knowledge.** 69.5% of greedy and 76.3% of sampled
completions hit the 1024-token cap before `<answer>`. When it does finish, it is mostly right (13.5% ID
accuracy out of 18% parsed). Truncated completions score the floor under every rule, which is why its
crps_log zero-variance (56%) is so much worse than Qwen2.5-1.5B's (2%). Zero-variance also rises steeply with
depth: 25% / 75% / 100% / 92% at depths 1–4.

### Provenance
- `results/model_probe/summary.{md,json}`; per model `results/model_probe/<label>/{greedy,group}/` (tapes
  replayable, metrics.json). Configs: `configs/model_probe.yaml`, `configs/eval_probe_{greedy,group}.yaml` at
  commit `cbb5a0c`. Generator `1.1.0+c735a0d4`, reward rule `crps_log`, transformers 5.19.0.
- Boxed-answer diagnostic: `scripts/diag_boxed_answers.py` →
  `results/model_probe/qwen2.5-math-1.5b-instruct/{greedy,group}/diag_boxed_answers.json`. Not the verifier.
- Group probe: 48 ID problems (4 per family × depth cell, seeds 1000000000–1000000047), k = 8, T = 1.0,
  top_p 1, top_k off, repetition penalty 1.0, seed 0. Qwen3.5 ran at batch 8 (D33).

---

## #3. First GRPO run: 49 steps of crps_log on Qwen3.5-2B change nothing measurable (2026-10-12)

**Claim.** Qwen3.5-2B, 4-bit + LoRA r = 16, reward crps_log + 0.2·steps, 1 prompt × 8 completions per step,
1024 completion tokens, 49 steps (57 min on the RTX 4060, 69 s/step, 4.45 GiB peak). On the same 400 eval
problems, before vs after: ID accuracy ≤1% 13.5% → 14.0% (σ-free), 7% → 9% (strict); depth-5 reward
−4.69 → −4.48. Paired McNemar / Wilcoxon on identical problems: **no significant change** (p ≥ 0.056). During
training, 82–94% of completions per 10-step window were truncated at 1024 tokens and scored the floor, and
40–80% of steps had zero reward variance.

**Interpretation.** The limiting factor is that the policy rarely produces a finished, parseable answer within
the budget, not the reward's density. Every gradient flows through the few completions that finish.

### Provenance
- Config `configs/grpo_phase4.yaml` (commit `0f88620`); run dir `results/train/phase4-grpo-qwen3.5-2b-crps_log/`
  (train_metrics.jsonl; rollouts.jsonl, 392 rollouts, replays exactly; training_curve.png; summary.json).
  The adapter is local only (gitignored), `lora_hash 5efe8a51708c3eca`.
- Eval: `eval_greedy/` (400 rollouts, replays exactly), `before_after.json`, `before_after_significance.json`
  (`scripts/paired_tests.py`). "Before" = `results/model_probe/qwen3.5-2b/greedy/`.

---

## #4. 30 minutes of σ-masked SFT on generator solutions fixes format and length; the A/B is close, and Math-SFT wins by rule (2026-10-13)

**Claim.** A 30-minute LoRA SFT on short, generator-written solutions (training seeds only; σ number masked
from the loss) turns both models into usable GRPO starting points. Parse failures drop from 88–100% to 0.3%,
completions shrink from ~700–900 to ~200 tokens, and crps_log zero-variance drops to 0%. On the unchanged
k = 8 group probe (48 ID problems, T = 1.0, prompt v4, 1024 tokens):

| model | binary zero-var | crps_log zero-var | pass@8 | mean pass rate | parse fail | truncated | mean tokens | sampled acc ≤1% / ≤10% | 1σ cov | median rel. σ |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-2B base | 72.9% | 56.2% | 27.1% | 7.0% | 87.8% | 76.3% | 885 | 7.0 / 7.6% | 27.9% | 0.2% |
| **Qwen3.5-2B + SFT** | 60.4% | 0.0% | 60.4% | 35.7% | 0.3% | 0.3% | 205 | 35.7 / 60.7% | 30.5% | 0.5% |
| Qwen2.5-Math-1.5B base | 97.9% | 95.8% | 2.1% | 0.3% | 99.5% | 17.2% | 708 | 0.3 / 0.3% | – | – |
| **Qwen2.5-Math-1.5B + SFT** | **56.2%** | 0.0% | 58.3% | 36.2% | 0.3% | 0.0% | 197 | 36.2 / 56.8% | 20.2% | 0.1% |

**A/B decision (your rule: lower binary zero-variance): Qwen2.5-Math-1.5B + SFT, 56.2% vs 60.4%.** This is
27 vs 29 zero-variance groups out of 48: a 2-group difference, well within sampling noise, so the two SFT'd
models are effectively tied on this metric. Math-SFT never truncates at 512 tokens, so the GRPO budget is
**512** (rule: < 20% truncation at 512). Two things favour Math-SFT beyond the rule: it trains faster per
example and is smaller. Against it, it is more over-confident (median stated σ 0.1% of the value, 1σ
coverage 20%, vs 0.5% and 31% for Qwen3.5-SFT). σ was never trained (it was masked), so this reflects each
base model's prior, and it is exactly what RL with crps_log has to fix.

**Why binary zero-variance stays ~60%:** with a ~36% pass rate the per-problem outcome is bimodal. Groups are
mostly all-right or all-wrong, so there is a mix at each depth: Math-SFT 42% / 67% / 67% / 50% zero-variance
at depths 1–4. crps_log, by contrast, has variance in every group, because the error magnitude and σ still
differ even when all 8 answers are right or all wrong.

### Provenance
- SFT: `configs/sft.yaml`, `physcal/train/{sft_data,sft,merge}.py` (commits `0e5c97c`–`eee48d5`). Data:
  3,000 training-seed problems (seeds 10,000,000–10,002,999, 12 training cells, generator `1.1.0+c735a0d4`;
  manifests in `results/sft/<model>/data_manifest.json`). Qwen3.5-2B: 262 steps / 2,096 examples, loss 0.59 → 0.05.
  Qwen2.5-Math: 325 steps / 2,600 examples, loss 0.64 → 0.06. Adapters merged into the bf16 bases (CPU, peft).
  Merged models are local only.
- Probe: `configs/model_probe_sft.yaml` → `results/model_probe_sft/<label>/group/` (tapes replayable);
  decision `results/model_probe_sft/ab_selection.json` (`scripts/ab_select.py`). Base rows:
  `results/model_probe/<label>/group/`.

---

## #5. Overnight GRPO on the SFT'd model: SFT did the heavy lifting; RL added a small, non-significant ID gain and no calibration (2026-10-13)

**Setup.** Qwen2.5-Math-1.5B + SFT (FINDINGS #4), fresh LoRA r = 16, crps_log + 0.2·steps, 4 prompts × 8
completions per step, 512 tokens, EMA curriculum from depths 1–2, lr 5e-6, DAPO loss, no KL. **287 steps**
(9,184 rollouts) in 4 h 42 min, stopped at the 06:30 IST hard deadline.

**Final eval** (frozen generator-1.1.0 eval sets, 200 ID + 200 depth-5, prompt v4, greedy, 512 tokens, all three
models loaded text-only):

| split | model | acc ≤1% (σ-free) | ≤10% | parse fail | 1σ / 2σ coverage | median stated rel. σ | mean crps_log reward |
|---|---|---|---|---|---|---|---|
| ID | raw base | 0.0% | 0.0% | 100% | – | – | −5.00 |
| ID | SFT only | 38.0% | 67.5% | 0% | 15.0% / 20.5% | 0.050% | −0.224 |
| ID | SFT + RL | **40.0%** | **71.0%** | 0% | 15.5% / 20.0% | 0.042% | **−0.177** |
| depth 5 | raw base | 0.0% | 0.0% | 100% | – | – | −5.00 |
| depth 5 | SFT only | 31.0% | 60.0% | 0% | 11.5% / 16.0% | 0.081% | −0.215 |
| depth 5 | SFT + RL | 31.0% | 59.5% | 0% | 10.0% / 15.0% | 0.064% | −0.220 |

**What RL added (RL vs SFT-only, paired on identical problems):** ID strict accuracy 76 → 80 (4 problems flipped
to correct, 0 flipped to wrong; McNemar p = 0.125). Mean reward +0.046 (Wilcoxon p = 0.018: nominal only, not
significant after correcting for the 8 RL-vs-SFT tests). Depth 5: 62 → 62, reward −0.005 (p = 0.37). The
small ID gain sits mostly at depth 4 (22% → 28%). Calibration did not move: stated σ is still ~0.05% of the
value, and 1σ coverage is ~15% against a 68% target.

**SFT vs raw base:** 0 → 76 (ID) and 0 → 62 (depth 5) correct, p < 1e-18. The raw base fails the format 100% of
the time.

**Likely reasons RL did so little** (hypotheses, not tested): (1) the 0.1% relative σ floor makes crps_log flat in
σ for a policy whose σ sits below it (D37), so σ gets almost no gradient; (2) a small update budget (287
steps at lr 5e-6); (3) the curriculum spent many steps on cells with low pass rates; energy never unlocked
depth 3 (energy/2 EMA pass rate 0.16), and energy remains the weak family (14–16% ID, 1.5–3% depth 5).

### Provenance
- Config `configs/grpo_overnight.yaml` (filled by `scripts/ab_select.py`, commit `a2d8bb2`); run dir
  `results/train/overnight-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-512/`: `train_metrics.jsonl`,
  `curriculum.jsonl`, `rollouts.jsonl` (9,184 rollouts, replays exactly), `training_curve.png`, `report.md`,
  `summary.json`. The adapter (`lora_hash 8b4b2376bf92206f`) and checkpoints 200/250/287 are local only.
- Eval: `compare/{base,sft,rl}/` (400 rollouts each, all replay exactly), `compare/paired_tests.json`
  (`physcal/train/compare.py`, `scripts/paired_tests.py`).

---

## #6. Removing the σ floor did not let RL move calibration at matched steps; a free post-hoc σ scale beats it, and longer training collapses σ to zero (2026-10-14)

**Question.** With the crps_log σ floor removed (verifier 1.1.0, D37/D38), does a GRPO run that is otherwise
identical to last night's move calibration by the same step (287)? Does it beat a single post-hoc σ scale fitted
on training-seed problems?

**Answer: no on both counts.** At step 287, 1σ coverage is 14.0% (ID) and 9.5% (depth 5), against 15.0% /
11.5% for SFT-only and 15.5% / 10.0% for last night's RL. Every paired McNemar p is ≥ 0.125, and the direction is
slightly *worse*. A single σ multiplier fitted on 400 training-seed problems lifts SFT-only coverage to 35.0% /
33.5%, which beats RL by 42 / 48 problems (p < 1e-12). Kept training past step 287 (to 759), the policy then
**collapsed σ to exactly zero**: 95–98% of training answers state "± 0" after step ~550, and the final adapter's
greedy 1σ coverage falls to 11.5% / 8.1%. The scoring rule itself does not favour this (see "Why").

### Setup
- **Verifier 1.1.0:** crps_log floors relative σ at 1e-6 instead of 1e-3; nothing else changed. Last night's
  SFT-only and SFT+RL eval tapes were re-scored under 1.1.0 (`compare_verifier_1.1.0/`; both replay exactly),
  so every number below is on the same verifier.
- **Run:** identical to FINDINGS #5: same merged SFT start, fresh LoRA r = 16, lr 5e-6, DAPO, β = 0, EMA
  curriculum from depths 1–2, 4 prompts × 8 completions, 512 tokens, seed 0, same GRPO problem pool. Two
  exceptions: the verifier, and a memory-only micro-batch split (2 × 16 instead of 4 × 8, with the same
  32-completion batch and the same DAPO token normalization, D39). The first five steps drew exactly the same
  curriculum cells as last night. 759 steps in 7 h 32 min (35.7 s/step). Peak torch reserved memory was 7.05 GiB
  (cap 7.5), with 0 OOMs and no thermal pauses.
- **Post-hoc control (D40):** SFT-only greedy completions (eval settings) on 400 training-seed problems (seeds
  20,000,000–20,000,399, train split, 12 training cells). One σ multiplier c, fitted to maximize mean crps_log:
  **c = 18.1**. Sensitivity variant: c matched to 68.27% 1σ coverage on the calibration set, **c = 585**.
- **Eval:** frozen generator-1.1.0 sets (200 ID + 200 depth 5), prompt v4, greedy, 512 tokens. Step-287 adapter
  first, then the final adapter (step 759). All tapes replay exactly.

### Results (verifier 1.1.0)

| split | model | acc ≤1% | 1σ / 2σ cov | median stated rel. σ | answers with σ = 0 | ρ(σ, error) | mean crps_log |
|---|---|---|---|---|---|---|---|
| ID | SFT-only | 38.0% | 15.0 / 20.5% | 0.050% | 25.5% | −0.04 | −0.320 |
| ID | post-hoc, crps fit (c = 18.1) | 38.0% | **35.0 / 40.5%** | 0.91% | 25.5% | −0.04 | −0.319 |
| ID | post-hoc, coverage-matched (c = 585) | 38.0% | 65.0 / 69.5% | 29% | 25.5% | −0.04 | −0.519 |
| ID | last night's RL (step 287) | 40.0% | 15.5 / 20.0% | 0.042% | 27.0% | −0.10 | −0.275 |
| ID | **tonight's RL, step 287** | 39.5% | 14.0 / 18.5% | 0.041% | 27.0% | −0.06 | −0.282 |
| ID | tonight's RL, step 759 (longer training) | 41.0% | 11.5 / 16.0% | 0.000% | **50.5%** | +0.01 | −0.239 |
| depth 5 | SFT-only | 31.0% | 11.5 / 16.0% | 0.081% | 12.0% | +0.06 | −0.357 |
| depth 5 | post-hoc, crps fit | 31.0% | **33.5 / 38.0%** | 1.46% | 12.0% | +0.06 | −0.354 |
| depth 5 | post-hoc, coverage-matched | 31.0% | 72.5 / 76.0% | 47% | 12.0% | +0.06 | −0.506 |
| depth 5 | last night's RL (step 287) | 31.0% | 10.0 / 15.0% | 0.064% | 13.5% | +0.07 | −0.361 |
| depth 5 | **tonight's RL, step 287** | 29.0% | 9.5 / 14.0% | 0.064% | 15.0% | +0.01 | −0.437 |
| depth 5 | tonight's RL, step 759 | 31.0% | 8.1 / 13.1% | 0.026% | 35.9% | −0.12 | −0.344 |

Paired tests on the same 200 problems per split. Counts are problems covered at 1σ (failures count as not
covered); McNemar for coverage and correctness, Wilcoxon for crps_log.

| comparison | split | 1σ covered: before → after (only after / only before) | p | 2σ p | strict ≤1% (p) | crps_log Δ (p) |
|---|---|---|---|---|---|---|
| **RL@287 vs SFT-only** | ID | 30 → 28 (0 / 2) | 0.50 | 0.22 | 76 → 79 (0.38) | +0.038 (0.12) |
| **RL@287 vs SFT-only** | depth 5 | 23 → 19 (0 / 4) | 0.125 | 0.125 | 62 → 58 (0.29) | −0.080 (0.55) |
| **RL@287 vs last night's RL** | ID | 31 → 28 (0 / 3) | 0.25 | 0.51 | 80 → 79 (1.0) | −0.007 (0.65) |
| **RL@287 vs last night's RL** | depth 5 | 20 → 19 (2 / 3) | 1.0 | 0.50 | 62 → 58 (0.22) | −0.076 (0.61) |
| **RL@287 vs post-hoc (crps fit)** | ID | 70 → 28 (0 / 42) | 4.6e-13 | 1.1e-13 | 76 → 79 (0.38) | +0.037 (0.87) |
| **RL@287 vs post-hoc (crps fit)** | depth 5 | 67 → 19 (0 / 48) | 7.1e-15 | 7.1e-15 | 62 → 58 (0.29) | −0.083 (0.014) |
| longer: step 759 vs step 287 | ID | 28 → 23 (2 / 7) | 0.18 | 0.27 | 79 → 82 (0.45) | +0.043 (0.99) |
| longer: step 759 vs step 287 | depth 5 | 19 → 16 (4 / 7) | 0.55 | 0.75 | 58 → 62 (0.39) | +0.093 (0.45) |
| longer: step 759 vs SFT-only | ID | 30 → 23 (2 / 9) | 0.065 | 0.035 | 76 → 82 (0.070) | +0.081 (0.23) |

**Reading.** Calibration is unchanged at matched steps; if anything it is slightly worse, with no significant
difference. On crps_log, RL@287's ID edge over SFT-only and over the post-hoc control comes from its slightly
higher accuracy, not from σ. On depth 5 it is worse than the post-hoc control (p = 0.014): one extra clipped
answer and a few more large misses. Accuracy again barely moves (ID 76 → 79 at step 287, 76 → 82 at step 759,
neither significant).

### Why calibration doesn't move: σ carries almost no reward for this policy
- **The σ term is tiny next to the error term.** On the SFT-only eval, the best single σ scale improves mean
  crps_log by only **+0.001 (ID) / +0.004 (depth 5)** per problem. An oracle σ that knew each problem's own error
  (σ = |ln(μ/y)| / 0.833, the CRPS-optimal σ for that error) would improve it by **+0.12 / +0.14**. Almost all
  of the reward available through σ requires σ that knows *which* answers are wrong. A global over-/under-
  confidence correction, which is what moves coverage, is worth almost nothing in reward.
- **The model's σ does not track its error.** The Spearman correlation between stated σ and actual error is ≈ 0
  for every model (−0.12 to +0.07). Errors are bimodal (right to ~0.1%, or off by 2–50%), and σ is unrelated to
  which mode a completion lands in.
- **Removing the floor restored the gradient, not its size.** Below 0.1% the reward now strictly increases with σ
  toward |error| (tested), but the slope is ≈ 0.56 per unit of relative σ. Widening σ from 0.05% to 0.1% of the
  value earns ~0.0003, while the error term varies by 0.01–0.5 within a GRPO group.
- **Training-side, steps 1–287, the two runs are indistinguishable:** σ = 0 share 24–26% vs 24–25%, median stated σ
  0.055–0.067% vs 0.062–0.074%, 1σ coverage 17–18% vs 18–19%, σ-free accuracy 42–46% vs 42–44% (per 100-step
  window, tonight vs last night).

### Longer training: the σ = 0 collapse
- The share of training answers stating "± 0" (literally; not a parser artifact) stays at 24–26% until step
  ~300, then rises to 33% (steps 301–400), 52–62% (451–500), 89% (526–550) and 94–98% from step 576 to the end.
  Training 1σ coverage falls from ~18% to 2–7%. Greedy, the final adapter states σ = 0 on 50.5% (ID) / 36% (depth
  5) of eval problems, and one depth-5 answer drops the ± altogether (a parse failure).
- **The scoring rule does not favour it.** On the σ = 0 training answers of steps 401–520, stating σ = 1% of the
  value would have scored better on average (mean crps_log −0.184 vs −0.191). The collapse goes *against* the
  reward's σ gradient, so it is not reward hacking of the verifier.
- **Likely mechanism (hypothesis, not tested by intervention):** credit assignment. "± 0" answers are slightly
  more often correct (46–50% vs 40–48% for σ > 0, in every 100-step window up to step 600 of both runs; after
  that only 65–139 σ > 0 answers per window remain, and they are more often correct), so the answer's large
  correctness advantage is credited to the "± 0" tokens too. That outweighs the σ term's own ~0.001-sized
  effect. Once most of a group says
  "± 0", σ no longer varies within the group and gets no gradient at all, so the state is absorbing. Last night's
  run had the same correlation and didn't drift within 287 steps, and neither did tonight's. Whether the drift
  after step 300 needs the floor removed (σ = 0 and 0.1% scored identically under 1.0.0) or only more steps can't
  be separated with these two runs.

### What this means
- RL with crps_log alone does not calibrate this model, at either 287 or 759 steps. The σ signal is swamped by
  correctness and gets credited along with it. A free post-hoc scale is the stronger calibration baseline. It
  still falls short of 68% unless one accepts a large crps_log cost, because the errors are bimodal and σ does
  not separate the modes.
- The useful target is σ that tracks which answers are wrong (worth ~0.12 crps_log per problem here). That needs
  σ to get its own credit, separate from correctness. Options are in MORNING.md; each is a design change.

### Provenance
- Verifier 1.1.0: `configs/reward.yaml`, `physcal/verify/{scoring,reward}.py` (commit `3d65dc1`), D38.
- Re-scored tapes: `results/train/overnight-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-512/compare_verifier_1.1.0/
  {sft,rl}/` (`python -m physcal.eval.rescore TAPE --current-verifier OUT`, commit `2d3ec25`).
- Post-hoc control: `configs/posthoc.yaml`, `physcal/eval/posthoc.py`; calibration tape
  `results/posthoc_calibration/sft_calibration_set/` (400 rollouts, replays exactly), fit
  `results/posthoc_calibration/posthoc_fit.json`, scaled eval records `sft_eval_scaled_{crps_fit,coverage_matched}.jsonl`
  (derived, not replayable), σ headroom `sigma_headroom.json` (`scripts/phase37_sigma_headroom.py`). D40.
- Run: `configs/grpo_overnight2.yaml` (commit `faf1327`; a test checks it differs from `grpo_overnight.yaml` only
  in the allowed keys); run dir `results/train/overnight2-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512/`:
  `train_metrics.jsonl`, `curriculum.jsonl`, `memory.jsonl`, `summary.json`, `rollouts.jsonl.gz` (24,288 rollouts;
  gunzip, then `replay()` reproduces every reward exactly), `training_vs_control.{png,json}`,
  `sigma_windows.txt`. Adapters (local only, gitignored): `permanent_checkpoints/checkpoint-287`
  (`lora_hash 86f7436e726bbaa5`), `final_adapter` (step 759, `lora_hash e33c888028e2ec0a`).
- Eval: `compare/{rl287,rlfinal}/` (400 rollouts each, replay exactly); paired tests and tables
  `compare/phase37_paired.json`, `compare/phase37_tables.md` (`scripts/phase37_compare.py`, which uses
  `scripts/paired_tests.py`); chain `scripts/phase37_morning.sh`. D41.

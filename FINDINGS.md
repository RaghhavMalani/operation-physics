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

> **Read with #15 (2026-10-17):** this entry's RL eval applied the adapter merged into bf16. For every Qwen2.5 RL adapter that merge rounds away > 90% of the update; this Qwen3.5-2B adapter was not measured, but it went through the same eval path at a similar learning rate.

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

> **Read with R2-2 (D71, 2026-10-18; pointer added 2026-10-19):** the "+ SFT" rows here were measured on SFT
> adapters merged into bf16. For Qwen2.5-Math (the model every later run uses), the merge rounded away 41% of the
> adapter's entries, and the adapter had been learned on Unsloth's 4-bit mirror, not on the bf16 weights it was
> merged into (D70); the Qwen3.5-2B adapter was not measured. These numbers describe an SFT attenuated by the bf16
> merge. Every RL run started from this merged model, so comparisons between RL runs stay internally
> consistent. The SFT-only model as the RL runs actually saw it (merged, then quantized to 4 bits): #16. New SFT is
> never merged (D70).

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

> **Read with #15 (2026-10-17):** this entry's RL evals applied the adapter merged into bf16, which rounds away > 90% of the RL update. Its eval-set numbers describe that merged model, not the trained policy. **Corrected results: #16.**
>
> **Read with R2-2 (D71, 2026-10-18; pointer added 2026-10-19):** the "SFT only" rows describe an SFT attenuated by
> the bf16 merge (41% of the SFT adapter's entries rounded away; see the note on #4). The corrected SFT-only rows
> (4-bit) are in #16.

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

> **Read with #15 (2026-10-17):** this entry's RL evals applied the adapter merged into bf16, which rounds away > 90% of the RL update. Its eval-set numbers describe that merged model, not the trained policy. **Corrected results: #16.**

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
- **Training-side, steps 1–287, the two runs are indistinguishable:** σ = 0 share 24–26% vs 23.5–25%, median stated
  σ 0.055–0.067% vs 0.062–0.074%, 1σ coverage 16.5–18.5% vs 17.5–19%, σ-free accuracy 42–46% vs 42–44% (per
  100-step window, tonight vs last night).

### Longer training: the σ = 0 collapse
- The share of training answers stating "± 0" (literally; not a parser artifact) stays at 24–26% until step
  ~300, then rises to 29–36% (steps 301–400), 52–62% (451–500), 89–91% (526–575) and 94–98% from step 576 to the end.
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

---

## #7. A linear readout predicts correctness at AUROC 0.92 where stated σ is at chance, but a random-init model does as well: the signal is in the text, not demonstrably in the model (2026-10-14)

**Question (Phase 9.5 step 2, PREREGISTRATION.md family P).** Does the SFT-only model "know" its own error, even though
its stated σ does not?

**Pre-registered answer: uninterpretable.** The probes beat stated σ by a wide margin (P1–P3 all pass Holm, p < 1e-4),
but the pre-registered sanity condition **fails**. A probe on a *randomly initialized* copy of the model predicts
correctness just as well (AUROC 0.914 vs 0.917, paired bootstrap p = 0.40). The pre-registration says P1–P3 are then
reported as uninterpretable, and they are.

| predictor (fit on 1,996 training-seed completions, tested on the 400 frozen eval completions) | correct ≤1%: test AUROC | log\|error\|: test Spearman ρ |
|---|---|---|
| **stated σ** | **0.50** | **0.02** |
| family + depth (metadata) | 0.76 | 0.49 |
| linear probe, SFT-only model (CV-chosen layer 15 / 16) | **0.917** [0.887, 0.944] | **0.733** [0.676, 0.780] |
| linear probe, random-init model (CV-chosen layer 2 / 1) | 0.914 | 0.689 |
| linear probe, shuffled labels | 0.48 (CI contains 0.5 ✓) | −0.13 |
| MLP probe, SFT-only model (secondary) | 0.930 | 0.676 |
| MLP probe, random-init model | 0.899 | 0.610 |

Per split (pooled numbers above): ID probe AUROC 0.944 vs stated σ 0.494; depth 5 0.899 vs 0.501. Depth 5 is never
seen in the fit set, so the readout transfers to deeper chains.

**Exploratory follow-up (not pre-registered; `scripts/probe_exploratory.py`).** What in the text carries the signal?
- **The completion's words:** a TF-IDF bag-of-words logistic regression on the completion text gets AUROC **0.906**
  (ID 0.925, depth 5 0.883), statistically the same as the SFT probe (difference +0.011, CI [−0.012, 0.035]).
- **Which chain the problem uses:** one-hot (family, target, set of givens) alone gets **0.88** on the ID split (0.90 on
  the 40% of test problems whose template occurs in the fit set). Most of the signal is per-template difficulty: the
  model reliably gets some chains right and others wrong, and the formulas it writes reveal which chain it is on.
- **Length** alone: 0.66.
- **Error magnitude is the one place the trained model adds something:** for log|error| the SFT probe beats the
  random-init probe, ρ 0.733 vs 0.689 (+0.044, CI [0.007, 0.082], one-sided p = 0.010, uncorrected).

### What this means
- **The information an informative σ needs is in the model's own context at answer time.** A linear reader of the
  completion separates right from wrong answers at AUROC ~0.9. The stated σ uses none of it (0.50). The bottleneck
  is learning to *map* that context to σ, not access to the information. That is consistent with the
  credit-assignment diagnosis of FINDINGS #6 and supports trying split credit (Phase 9.5 step 3).
- **But "knowing its error" here mostly means recognizing the problem type and the formulas it used**, which a random
  network or a bag of words does equally well. There is no evidence of introspective knowledge beyond the text,
  except a small, exploratory edge on error magnitude.
- **For Phase 13:** a hidden-state probe as an uncertainty source must be reported next to the random-init and
  bag-of-words controls, or it will overclaim. And a text- or template-conditional post-hoc σ (for example, conformal
  per template) is likely a much stronger baseline than the single global scale: the global scale can't use any of
  the 0.9-AUROC signal.

### Provenance
- Protocol: PREREGISTRATION.md family P (commit `f601bb7`, before any result). Implementation details and deviations:
  D45, D47.
- Fit set: `results/posthoc_calibration/sft_calibration_set/` (400) + `results/probe_sft/sft_calibration_ext/` (1,600,
  replays exactly); SFT-only, greedy, prompt v4, 512 tokens; 4 failed answers (parse failure or unit mismatch)
  excluded (1,996 used). Test set: the
  SFT-only eval tape re-scored under verifier 1.1.0 (400, 0 excluded).
- Code: `physcal/interp/{extract,probes}.py`, `configs/probe_sft.yaml`, tests `tests/test_interp.py`. Hidden states
  (local only, gitignored): `results/probe_sft/hidden/{sft,random_init}/{fit,test}/layer_XX.npy`.
- Results: `results/probe_sft/probe_results.json`, `probe_by_layer.png`; exploratory `results/probe_sft/exploratory.json`.

---

## #8. Rounding audit: ground truth already uses the displayed givens; rounding sets an error floor well inside the 1% band (2026-10-14)

**Question (Phase 10 step 2).** Does the ground truth use unrounded world values while the problem text shows
rounded givens? If so, how large is the implied σ\* relative to the 1% accuracy band?

**Answer: no, the truth is computed from exactly the displayed givens (D4), so for the problems as defined σ\* = 0.**
On all 400 frozen eval problems (generator 1.1.0, 200 ID + 200 depth 5), re-parsing every displayed given, converting
it to SI and re-running the chain reproduces the stored truth with a relative difference of **exactly 0.0**. There
is no hidden rounding noise between the text and the answer key.

Two related quantities are still worth knowing, because they bound what an honest σ can be:

**(a) Counterfactual σ\*: if the 3-s.f. givens were measurements** (true value uniform within half a unit in the
last displayed place), the target would be uncertain by, per problem (Monte Carlo, 2,000 draws):

| depth | median σ\*/value | 10th–90th pct | share with σ\* > 0.1% | share with 2σ\* > 1% |
|---|---|---|---|---|
| 1 | 0.14% | 0.07–0.31% | 66% | 8% |
| 2 | 0.23% | 0.10–0.52% | 90% | 12% |
| 3 | 0.24% | 0.09–0.48% | 86% | 8% |
| 4 | 0.29% | 0.11–0.57% | 92% | 14% |
| 5 | 0.30% | 0.13–0.73% | 96.5% | 26% |
| all | **0.26%** | 0.10–0.63% | 90% | 18% |

That is about a quarter of the 1% band at the median, growing with depth. This is the noise scale the Phase 10
measurement-noise mode must dominate. With 3 s.f. displayed givens, rounding alone would add σ\* ≈ 0.26% (in
quadrature) to a 1% measurement noise. The noise mode should therefore display noisy givens with ≥ 5 significant
figures (half-ulp ≤ 5e-5 relative), as Phase 10.3 already requires ("rounding negligible against s").

**(b) Solver-side rounding: the error of a *perfect* solver that rounds every intermediate and its answer to k
significant figures.** This is the only irreducible error for the problems as defined.

| k s.f. | median | 90th pct | 99th pct | share outside the 1% band |
|---|---|---|---|---|
| 3 | 0.15% | 0.44% | 1.19% | 2.0% overall; 2% at depth 4, 3.5% at depth 5 |
| **4** (the SFT targets' habit) | **0.014%** | 0.041% | 0.11% | 0% |
| 6 | 0.0001% | 0.0004% | 0.0009% | 0% |

**What it explains.**
- The "exactly right" spike in the model's errors is solver rounding. 15.5% (ID) / 8.5% (depth 5) of SFT-only
  answers are within 0.1% of the truth, and their typical error (10th percentile 0.04% ID) matches a 4-s.f. solver.
- An honest σ for a *correct* 4-s.f. chain is ~0.01–0.04% of the value, not 0. The SFT model's median stated σ
  (0.05%) is in the right range **for correct answers**. What it lacks is a different σ for the answers that are
  wrong by 2–50% (FINDINGS #6), which no rounding argument covers.
- "± 0" is never exactly honest (the rounding error is > 0 almost surely), but its crps_log cost on a correct
  answer is tiny (≈ |error|, ~1e-4).
- At 3 s.f., a perfect solver misses the 1% band on 2–3.5% of deep problems. The 1% accuracy metric is safe
  only for solvers that keep ≥ 4 s.f. through the chain, which the SFT targets do.

### Provenance
- `physcal/eval/rounding_audit.py`, `configs/rounding_audit.yaml`, tests `tests/test_rounding_audit.py`.
- Output `results/rounding_audit/rounding_audit.json` (summary + one row per problem). Eval sets
  `data/eval_sets/1.1.0+c735a0d4/{id,depth_extrapolation}.jsonl`. Observed errors from the SFT-only eval tape
  re-scored under verifier 1.1.0.

---

## #9. Phase 10 gate (D50/A3): the generator's σ\* FAILS on one cell, the 2σ tail at depth 3 / 10% noise (2026-10-14)

**Verdict.** Generator 2.0.0 (`2.0.0+bdbb2279`) **fails** the amended self-consistency gate. 6 of 7 criteria pass;
per-cell Kupiec at ±2σ\* fails Holm in one cell. By your brief, Phase 10+ work stops here; A6 (counterfactual
pairs) is not built.

**Run.** 10,000 noisy problems (depth 1–5 × noise 1/2/5/10%, 500 per cell, own seeds per cell) + 500 noise-0
problems; σ\* = CRPS-optimal Gaussian log-scale centred at ln f(x_meas) (A2), intervals ln f(x_meas) ± kσ\*.
`uv run python -m physcal.eval.sigma_star_gate`; `results/sigma_star_gate/gate.{md,json}`, per-problem rows in
`rows.jsonl`. The gate code was committed (`162c573`) before it ran; it ran once.

| criterion (A3) | result |
|---|---|
| pooled Kupiec, ±1σ\*: 67.97% vs 68.27% | p = 0.52, pass |
| pooled Kupiec, ±2σ\*: 95.22% vs 95.45% | p = 0.27, pass |
| per-cell Kupiec + Holm (20 cells), ±1σ\* | no rejection (smallest p = 0.027), pass |
| per-cell Kupiec + Holm (20 cells), ±2σ\* | **d3 / 10%: 91.8%, p = 0.0004 < 0.0025: FAIL** |
| cells inside the 99% binomial interval | 100% (±1σ\*), 95% = 19/20 (±2σ\*, the minimum), pass |
| pooled PIT KS (truth under the MC distribution) | D = 0.0078, p = 0.57, pass |
| noise 0: truth = f(x_meas), σ\* = 0 | all 500 exact |

The superseded rule ("Kupiec p > 0.05 in every cell", replaced by A3 before the gate ran) also fails: 5 cells
have p < 0.05.

**Diagnosis (post hoc, same run, no new data).** The predictive distribution itself looks right. Its Gaussian
summary is too thin in the tails at 10% noise:
- In the failing cell, the PIT is uniform (KS p = 0.64), but the standardized deviations (ln y − ln f)/σ\* have
  SD 1.12 and excess kurtosis 1.06.
- The MC distribution's own central 95.45% interval covers **93.6%** there (vs 91.8% for ±2σ\*). Pooled over all
  10%-noise problems, MC quantile intervals cover 95.44%, against 94.72% for ±2σ\*.
- At 1–5% noise, pooled ±2σ\* coverage is 94.9–96.2%.
- First-order propagation is fine in the median (SD/delta 1.001–1.020), but individual problems deviate by up
  to 74%.
- The misses are spread across families (17 / 13 / 11) and are only mildly more ill-conditioned (median κ 2.0 vs 1.74).

**What this means / options (yours to decide; nothing was changed):**
1. Keep the gate as is: the generator is not valid at 10% noise with a Gaussian σ\*.
2. Cap the noise dial at 5%: all ±2σ\* cells pass Holm at 1–5% in this run. But choosing a cap after seeing this
   run is a forking path, and it needs a fresh gate run.
3. Grade coverage against the MC predictive quantiles rather than a Gaussian σ\*. This would change A2's primary
   target, and also needs a fresh gate run.
4. Treat one Holm rejection among 40 tests as chance: not allowed by A3 as written.

---

## #10. A5 nested baselines: the SFT model's own text predicts its error far better than its stated σ (2026-10-14)

**Claim.** Surface features of the problem and of the SFT-only model's completion, fitted only on the calibration
split, give a σ with ~67% 1σ coverage and ρ(σ, |error|) ≈ 0.70 on the frozen eval sets. The model's stated σ gives
13% coverage and ρ ≈ 0. This is the bar any RL σ is now judged against, on the same completions (D50/A5).

| σ source (SFT-only completions, pooled eval, 399 answered) | mean crps_log | 1σ cov | ρ(σ, err) |
|---|---|---|---|
| stated σ | −0.327 | 13.3% | −0.02 |
| metadata (family, depth), GBT | −0.303 | 63.2% | 0.49 |
| + problem text, GBT | −0.296 | 68.4% | 0.65 |
| + answer, GBT | −0.296 | 69.4% | 0.65 |
| + reasoning, linear | −0.292 | 66.7% | 0.66 |
| **+ reasoning, GBT (strongest by calibration CV)** | **−0.293** | **66.9%** | **0.70** |

- **p channel (correct within 1%):** the strongest model reaches Brier 0.111 and AUROC 0.913 on the eval sets.
  Metadata alone reaches AUROC 0.78.
- **Nesting:** most of the gain comes from the problem text. The answer adds ~nothing, and the reasoning adds a
  little.
- **Provenance:** `results/a5_baselines/` (`selection.json` written before eval use; `summary.md`,
  `eval_sft.json`).
- **Caveat:** the selection used calibration-split CV. The eval-set ranking of the top candidates differs
  slightly (linear vs GBT on crps_log), which is expected at n = 400.

---

## #11. Phase 10 gate rerun with noise capped at 5% (D54): FAILS again, on the cell-share rule only (2026-10-14)

**Verdict.** On fresh seeds and noise levels {1, 2, 5}%, generator 2.0.0 (`2.0.0+bdbb2279`) **fails** the A3 gate.
6 of 7 criteria pass. The failure is the share rule at ±2σ\*: one of 15 cells (depth 4 / 1%) is outside its 99%
binomial interval. It **over**-covers: 97.6% vs 95.45%, Kupiec p = 0.0035. With 15 cells the rule requires all
15 inside (D54 said so before the run). By your instruction, Phase 10+ stops again and **A6 is not built**.

**Run.** 10,005 noisy problems (15 cells × 667, seeds from 1.6e9, disjoint from the first run) + 500 noise-0
problems. Run once: `uv run python -m physcal.eval.sigma_star_gate --config sigma_star_gate_capped`;
`results/sigma_star_gate_capped/gate.{md,json}`, `rows.jsonl`.

| criterion (A3) | result |
|---|---|
| pooled Kupiec ±1σ\*: 67.61% vs 68.27% | p = 0.16, pass |
| pooled Kupiec ±2σ\*: 95.47% vs 95.45% | p = 0.92, pass |
| per-cell Kupiec + Holm (15 cells), ±1σ\* / ±2σ\* | no rejection at either level (smallest p = 0.0035 vs threshold 0.0033), pass |
| cells inside the 99% binomial interval, ±1σ\* | 15 / 15, pass |
| cells inside the 99% binomial interval, ±2σ\* | **14 / 15 = 93% < 95%: FAIL** (d4 / 1%: 97.6%) |
| pooled PIT KS | D = 0.0099, p = 0.28, pass |
| noise 0 exact | 500 / 500 |

**Reading (descriptive, after the run):**
- **Opposite direction from the first run.** That run's failure was **under**-coverage at 10% noise (heavy tails).
  This cell **over**-covers at the lowest noise level, and its ±1σ\* coverage is normal (65.8%).
- **Calibration everywhere else is close to nominal:** pooled ±2σ\* coverage is 95.47%. Per level, the MC SD
  agrees with the delta method to within 0.9% in the median.
- **One cell outside is roughly a chance event:** a valid generator puts at least one of 15 cells outside a 99%
  interval about 14% of the time per level. This is not evidence of a defect, but the gate as written fails.

**Options (yours):**
1. Amend the share rule, for example "at most ⌈5% of cells⌉ outside, per level". That allows one cell here, and
   one in the first run's 20 cells. It is a post-hoc amendment: apply it to a **fresh** run on new seeds, not
   to this one.
2. Keep the rule as is, and redesign the cell grid so that 95% of cells allows a miss (≥ 20 cells per level).
3. Accept this run's failure as a known false positive, and record that it was accepted after the fact.

---

## #12. Phase 10 gate, FINAL run (D57): PASSED; generator 2.0.0's σ\* is self-consistent at noise ≤ 5% (2026-10-14)

**Verdict.** On fresh seeds (offset 1.7e9, disjoint from #9 and #11), with noise levels {1, 2, 5}% (D54) and the A3
criteria with the miss-count rule (D57), generator 2.0.0 (`2.0.0+bdbb2279`) **passes all 7 criteria**. This was the
run D57 declared final. Phase 10+ work resumes with A6.

| criterion | result |
|---|---|
| pooled Kupiec ±1σ\*: 68.32% vs 68.27% | p = 0.92, pass |
| pooled Kupiec ±2σ\*: 95.71% vs 95.45% | p = 0.20, pass |
| per-cell Kupiec + Holm (15 cells), ±1σ\* / ±2σ\* | no rejection, pass |
| miss-count (D57), ±1σ\* / ±2σ\* | 0 of 15 cells outside at both levels, pass |
| pooled PIT KS | D = 0.0047, p = 0.98, pass |
| noise 0 exact | 500 / 500 |

- **Superseded rules:** the old share rule would also have passed (0 outside). The pre-A3 every-cell rule fails
  (d1 / 5% at ±1σ\*: 72.0%, p = 0.038). It is reported only because the report always shows it.
- **Cross-checks:** MC SD vs the delta method agree to 1.0008 / 1.0029 / 1.0082 in the median at 1 / 2 / 5%.
  The CRPS-optimal σ\* sits within 0.4% of the SD.
- **Provenance:** `uv run python -m physcal.eval.sigma_star_gate --config sigma_star_gate_final`;
  `results/sigma_star_gate_final/`. Code and config were committed (`5202b1f`) before the run.
- **The full gate history is part of the result:**
  - #9: 10% noise failed (heavy tails).
  - D54: the cap was set after seeing #9.
  - #11: the share-rule design error.
  - D57: the rule was replaced before this final run.

## #13 (evaluated on merged models; superseded by #16; S2 withdrawn). Split credit (Phase 9.5 step 3), 3 seeds: S1a, S1b and S3 supported on the seed mean, S2 narrowly supported (sensitive to the seed-mean definition); σ still does not track the error (2026-10-17)

> **Read with #15 (2026-10-17):** this entry's RL evals applied the adapter merged into bf16, which rounds away > 90% of the RL update. Its eval-set numbers describe that merged model, not the trained policy. **Corrected results: #16.**

**Verdict (PREREGISTRATION.md family S, evaluated on the seed mean as D62 fixes it; D62 approved as written in
review, 2026-10-17).**
- **S1a, S1b and S3 are supported** under Holm.
- **S2 is narrowly supported; sensitive to the seed-mean definition.** Its per-seed p is 0.061 / 0.020 / 0.092,
  so it holds on its own only on seed 1. The seed mean passes because all three effects point the same way.

| test | split credit, mean of 3 seeds | control | difference [95% CI] | one-sided p | Holm threshold | verdict | per-seed p (seeds 0 / 1 / 2) |
|---|---|---|---|---|---|---|---|
| S1a: 1σ cov > SFT-only | 21.7% | 13.25% | +8.4 pts [+5.8, +11.3] | < 1e-4 | 0.0125 | supported | 4.7e-9 / 6.2e-8 / 1.8e-8 |
| S1b: 1σ cov > uniform RL@287 | 21.7% | 11.75% | +9.9 pts [+7.2, +12.8] | < 1e-4 | 0.0167 | supported | 8.0e-12 / 7.3e-11 / 1.1e-10 |
| S2: ρ(σ, err) > uniform RL@287 | 0.081 | −0.017 | +0.098 [−0.012, +0.208] | 0.040 | 0.05 | narrowly supported; sensitive to the seed-mean definition | 0.061 / 0.020 / 0.092 |
| S3: accuracy not > 5 pts below uniform | 35.7% | 34.25% | +1.4 pts [+0.1, +2.9] | < 1e-4 | 0.025 | supported | < 1e-4 / < 1e-4 / < 1e-4 |

- **Per seed** (seed 0 / 1 / 2):
  - 1σ coverage 22.5 / 21.0 / 21.5%; accuracy 34.5 / 36.2 / 36.2%;
  - ρ 0.078 / 0.099 / 0.066; per-seed S2 p 0.061 / 0.020 / 0.092;
  - every per-seed S1 McNemar p < 1e-7.
- **σ = 0 at eval:** 2.2 / 6.5 / 3.5%, vs 18.8% for SFT and 21.0% for uniform credit.
- **Family N (D50; the strongest A5 σ applied to the same answers) fails on every seed.** All three N tests go
  the wrong way with CIs well clear of 0: the A5 σ has coverage 66–68%, ρ 0.66–0.70, and crps_log better by
  0.03–0.04. RL σ does not beat a surface-feature baseline.
- **What this means:** split credit fixes the sequence-level credit problem behind the σ = 0 collapse (FINDINGS
  #6). σ = 0 almost disappears and coverage rises by about 9 points, at no accuracy cost. But 1σ coverage
  is still 21.7% against a 68.3% target, and σ barely ranks problems by error (ρ ≈ 0.08 vs 0.70 for
  a text model). The narrow S2 pass should not be read as "σ tracks error".
- **Caveats:**
  - Only one uniform-credit seed exists, so S1b, S2 and S3 compare a 3-seed mean with a single control run.
  - The inference is over problems, conditional on these 3 seeds (D62).
  - Seed 1 was restarted from step 0 after a power loss (D63); runs are not bit-reproducible past step 2.
- **Provenance:** `scripts/phase95_seed_mean.py` (committed `873dee5`, before any seed-1/2 data) →
  `results/split_credit_seed_mean/`. Per-seed reports are at
  `results/train/split-credit-…-seed{0,1,2}/compare/phase95_family_S.md`.

## #14. E1, fixed-μ σ-only GRPO: σ is learnable but not beyond surface features (family E, 2026-10-16)

> **Read with #15 (2026-10-17):** this entry's RL evals applied the adapter merged into bf16, which rounds away > 90% of the RL update. Its eval-set numbers describe that merged model, not the trained policy. **Corrected results: #16.**

**Verdict (D52, single seed by design).**
- **E1a-P and E1b-P are supported.** The E1 σ beats SFT's stated σ on the same answers: Δcrps_log +0.0024 and
  +0.0020, both p < 1e-4.
- **E1a-N and E1b-N are not.** The E1 σ loses to the strongest A5 σ by 0.031 crps_log, CI [−0.040, −0.023].
- **D52's interpretation rule:** "σ is learnable with fixed μ, but not beyond surface features". This is the clean
  negative result.

| σ source (same 400 SFT answers) | mean crps_log | 1σ cov | ρ(σ, err) | median rel. σ |
|---|---|---|---|---|
| SFT stated σ | −0.3386 | 13.2% | −0.02 | 0.061% |
| E1a (free σ, 300 steps) | −0.3362 | 29.2% | 0.06 | 0.57% |
| E1b (9 bins, 300 steps) | −0.3366 | 28.2% | 0.08 | 0.50% |
| A5 strongest σ | −0.3051 | 66.8% | 0.70 | 8.0% |

- **Size of the gain:** with credit assignment made clean (μ fixed, only σ varies within a group), 300 steps
  widened σ and removed σ = 0. The gain is about 7% of the SFT-to-A5 gap. E1a and E1b are indistinguishable, and
  most of the gain was there by step 150.
- **Together with #13:** fixing credit assignment, whether by split credit or by fixing μ, is necessary to avoid
  collapse but is not sufficient for an informative σ at this scale and budget.
- **Provenance:** `results/e1/family_E.{json,md}`; evals in `results/train/e1{a,b}-…/e1_eval/`.

## #15. Every RL eval so far measured a bf16 merge that erases > 90% of the RL update; evaluated as trained, RL σ is ~5× wider, but still does not track the error (exploratory, D65, 2026-10-17)

**Question.** RL adapters are trained on a 4-bit base with the LoRA unmerged. Every RL eval merged them into the
bf16 SFT model instead. Is that why eval σ was ~4× narrower than training σ (MORNING.md 2026-10-17, §5)?

**Answer: yes, and the cause is the merge itself.**
- **Evaluated as trained** (Unsloth 4-bit base, LoRA unmerged), both adapters state their training σ:
  - split credit (seed 0 @287): sampled median relative σ 1.73%, against 1.34% in training and 0.37% under the
    bf16 merge;
  - E1a: 3.55%, against 3.56% in training and 0.77% under the merge.
- **D65's pre-stated rule:** gap share g = 1.20 and 1.00, "explains most of the gap". (The rule's label says
  "precision". The diagnostics below place the cause in the merge, not in the 4-bit base.)
- **Why the merge does this:** `merge_and_unload` into bf16 rounds 91–96% of every RL update's entries to exactly
  zero. The updates are about 20× smaller than one bf16 step on the base weights.

| model | applied as | decoding | acc ≤ 1% | median rel. σ | σ = 0 | 1σ / 2σ cov | mean crps_log | ρ(σ, err) |
|---|---|---|---|---|---|---|---|---|
| split credit seed 0 @287 | bf16, merged (all previous evals) | greedy | 34.5% | 0.244% | 2.2% | 22.5 / 29.0% | −0.325 | 0.08 |
| split credit seed 0 @287 | **4-bit, unmerged (as trained)** | greedy | 37.8% | **1.614%** | 0.0% | **41.8 / 49.8%** | −0.316 | 0.08 |
| split credit seed 0 @287 | bf16, merged | sampled k = 4 | 29.6% | 0.367% | 5.5% | 19.9 / 25.7% | −0.517 | 0.08 |
| split credit seed 0 @287 | 4-bit, unmerged | sampled k = 4 | 28.9% | 1.732% | 0.4% | 34.5 / 43.1% | −0.412 | 0.12 |
| split credit, training steps 263–287 | 4-bit, unmerged | sampled | – | 1.341% | – | – | – | – |
| E1a final (μ fixed) | bf16, merged | greedy | – | 0.567% | 1.0% | 29.2 / 35.8% | −0.3362 | 0.06 |
| E1a final (μ fixed) | **4-bit, unmerged** | greedy | – | **3.228%** | 0.0% | **48.5 / 58.2%** | −0.3309 | 0.01 |
| E1a final (μ fixed) | bf16, merged | sampled k = 4 | – | 0.773% | 2.7% | 32.6 / 39.1% | −0.3426 | 0.03 |
| E1a final (μ fixed) | 4-bit, unmerged | sampled k = 4 | – | 3.546% | 0.1% | 49.9 / 59.5% | −0.3426 | 0.01 |
| E1a, training steps 276–300 | 4-bit, unmerged | sampled | – | 3.556% | – | – | – | – |

- **The match holds at every depth.** The 4-bit σ is within 1.1–1.6% (split) and 2.7–4.0% (E1a) of the training
  σ at each depth, so the problem mix does not explain the gap.
- **Paired greedy outputs:** only 1.8% of split-credit completions (21.8% of E1a σ suffixes) are identical between
  the two ways of applying the adapter. The median per-problem σ ratio is ×10 (ln ratio 2.30), with 4-bit wider on
  75–78% of problems.

**Diagnostics (post-hoc, CPU, added after the 4-bit greedy results; D65 addendum).**
- **Merged vs unmerged in bf16** (E1a adapter, the first 3 prefixes where the two evals differ): the bf16 base with
  the adapter *unmerged* reproduces the 4-bit eval's σ suffix on 3 / 3. After `merge_and_unload` it does not
  (2 / 3 differ). With n = 3 this is suggestive only.
- **Merge rounding, weights only** (`results/precision_check/merge_rounding.md`):

  | adapter | update entries rounded to 0 | rel. error of the merged update | median \|Δ\| / bf16 step |
  |---|---|---|---|
  | split credit seeds 0 / 1 / 2 @287 | 93.7 / 93.9 / 93.6% | 0.92 | 0.05 |
  | uniform credit @287 | 95.9% | 0.95 | 0.03 |
  | overnight RL, final (#5) | 95.8% | 0.95 | 0.03 |
  | E1a / E1b | 90.9 / 91.3% | 0.89 | 0.07 |
  | SFT adapter (on its original base) | 41.2% | 0.28 | 0.70 |

- **Quantization alone does not widen σ.** Training step 1, where LoRA B = 0, samples the plain 4-bit SFT model.
  Its median σ is 0.05–0.16% (split-credit seeds, uniform credit; E1a 0.016%), against 0.063% for the bf16 SFT
  greedy eval. Caveat: these are training problems at depths 1–2, at T = 1.

**What this means.**
- **The eval-set results in #3, #5, #6, #13 and #14 measured "SFT + rounding noise", not the RL-trained policies.**
  - The pre-registered verdicts of families S, N and E stand as tests of those merged models.
  - As evidence about what RL learned, they are not valid until the adapters are re-evaluated unmerged.
  - Results from training rollouts (e.g. #6's σ = 0 collapse) are unaffected.
- **Evaluated as trained, split credit roughly doubles coverage** (1σ 41.8% vs 22.5%; 2σ 49.8% vs 29.0%). But σ
  still does not rank problems by error: ρ = 0.08, the same as before, against ≈ 0.7 for the A5 text features.
- **E1a (μ fixed, so the comparison is paired and direct):** on the same answers, the as-trained σ scores crps_log
  −0.3309.
  - **Against SFT's own σ (−0.3386):** Δ = +0.008, against +0.0024 for the merged model.
  - **Against the strongest A5 σ (−0.3051):** still worse, by 0.026.
  - **So:** family E's qualitative reading, "σ is learnable, but not beyond surface features", probably survives
    on the trained policy. This is descriptive, not tested.
- **Accuracy:** 37.8% (4-bit, greedy) vs 34.5% (merged). It cannot be attributed to RL without a 4-bit SFT-only
  eval, which has not been run.
- **Caveats:**
  - one seed per arm;
  - exploratory, no tests;
  - the 4-bit SFT control comes only from training step 1;
  - the merged-vs-unmerged generation check has n = 3, though the weight-rounding measurement covers every module
    of every adapter.

**Provenance:** D65 (plan and reading rule committed in `deff739`, before any 4-bit eval), `scripts/precision_check.py`
→ `results/precision_check/report.{json,md}`; diagnostics `scripts/precision_merge_check.py`,
`scripts/precision_merge_rounding.py`; eval tapes `…/compare/rl287_4bit{,_sampled}/` and
`…/e1_eval/final_4bit{,_sampled}/` (all four replay exactly).

## #16. Corrected evals (D67: adapters unmerged on the 4-bit training base) are the primary results: S1a, S1b, S3 supported, S2 NOT supported; family N fails on every seed; family E unchanged (2026-10-18)

**What changed.** FINDINGS #15 found that every RL eval had merged the adapter into bf16, which erased > 90% of
the update. D67 makes the corrected evals primary.
- **RL adapters:** applied unmerged, on the 4-bit base they were trained on.
- **Controls:** SFT-only and uniform credit, on the same base.
- **Everything else is as pre-registered and as in the superseded runs:** the analysis code (only the input paths
  differ), the frozen 1.1.0 sets (200 ID + 200 depth 5), greedy decoding, the A5 baseline, D62's seed-mean rule
  and Holm.
- **The superseded (bf16-merged) results stay in the repo and are shown beside each corrected one:**
  `results/corrected/side_by_side.md`.

**Family S on the seed mean (pre-registered verdict; split credit seeds 0–2 @287):**

| test | corrected: split / control | difference [95% CI] | p | verdict | superseded verdict |
|---|---|---|---|---|---|
| S1a: 1σ cov > SFT-only | 41.8% / 12.0% | +29.8 pts [+25.5, +34.2] | < 1e-4 | **supported** | supported (+8.4 pts) |
| S1b: 1σ cov > uniform RL@287 | 41.8% / 10.5% | +31.3 pts [+27.0, +35.8] | < 1e-4 | **supported** | supported (+9.9 pts) |
| S2: ρ(σ, err) > uniform RL@287 | 0.061 / 0.015 | +0.046 [−0.095, +0.184] | 0.262 | **not supported** | narrowly supported (p = 0.040) |
| S3: accuracy not > 5 pts below uniform | 37.7% / 37.2% | +0.4 pts [−1.0, +1.8] | < 1e-4 | **supported** | supported |

- **Per-seed p:**
  - S1a: 7e-34 / 3e-32 / 9e-35; S1b: 8e-37 / 5e-35 / 1e-37;
  - **S2: 0.207 / 0.396 / 0.225, so no seed passes S2 on its own;**
  - S3: < 1e-4 on every seed.
- **S2's narrow pass was an artifact of the merged eval.** Evaluated as trained, split credit's σ does not rank
  problems by error better than uniform credit's. ρ is 0.08 / 0.03 / 0.07 across seeds.

**Family N (split credit's σ vs the strongest A5 σ on the same corrected answers): fails on every seed, as before.**

| seed | N1: \|1σ cov − 68.3%\|, RL vs A5 | N2: ρ, RL vs A5 | N3: crps_log, RL vs A5 |
|---|---|---|---|
| 0 | 0.265 vs 0.005 | 0.077 vs 0.676 | −0.316 vs −0.289 |
| 1 | 0.270 vs 0.002 | 0.034 vs 0.674 | −0.317 vs −0.290 |
| 2 | 0.258 vs 0.003 | 0.071 vs 0.644 | −0.307 vs −0.283 |

All nine tests go the wrong way (p = 1).

**Family E (μ fixed; the SFT control is the SFT model's own σ suffix on the same 4-bit base): verdicts unchanged.**

| test | corrected Δcrps_log [95% CI] | verdict | superseded |
|---|---|---|---|
| E1a-P: E1a σ > SFT σ | +0.0077 [+0.0062, +0.0092] | **supported** | +0.0024, supported |
| E1b-P: E1b σ > SFT σ | +0.0075 [+0.0061, +0.0090] | **supported** | +0.0020, supported |
| E1a-N: E1a σ > A5 σ | −0.0258 [−0.0342, −0.0181] | **not supported** | −0.0310, not supported |
| E1b-N: E1b σ > A5 σ | −0.0259 [−0.0345, −0.0181] | **not supported** | −0.0314, not supported |

- **The E1 σ itself:** E1a states a median 3.2% (vs 0.57% merged), with 1σ coverage 48.5% (29.2%) and
  ρ(σ, err) = 0.01. E1b: 2.0%, 45.5%, 0.01.
- **Size of the gain:** about 23% of the SFT-to-A5 gap in crps_log (superseded: about 7%).

**Headline, pooled 400 problems, corrected (greedy):**

| arm | acc ≤ 1% | 1σ / 2σ cov | σ = 0 | median rel. σ | ρ(σ, err) | crps_log |
|---|---|---|---|---|---|---|
| SFT-only (4-bit) | 36.8% | 12.0 / 17.0% | 16.8% | 0.063% | 0.05 | −0.333 |
| SFT + standing post-hoc scale (c = 18.1, not re-fitted) | 36.8% | 34.2 / 42.2% | 16.8% | 1.15% | 0.05 | −0.329 |
| SFT + post-hoc scale re-fitted on 4-bit SFT completions (c = 22.1, D69; the standing baseline from 2026-10-19) | 36.8% | 36.0 / 43.5% | 16.8% | 1.40% | 0.05 | −0.328 |
| SFT + A5 σ | 36.8% | 67.8 / 78.2% | 0.0% | 6.2% | 0.68 | −0.302 |
| uniform credit @287 | 37.2% | 10.5 / 15.0% | 19.5% | 0.052% | 0.01 | −0.324 |
| split credit @287, seeds 0 / 1 / 2 | 37.8 / 36.8 / 38.5% | 41.8–42.5 / 49.8–51.5% | 0.0–0.3% | 1.59–1.64% | 0.08 / 0.03 / 0.07 | −0.316 / −0.317 / −0.307 |

- **The re-fitted post-hoc row (added 2026-10-19, D69 / your decision):** the same 4-bit SFT answers with c = 22.1, the
  D40 fit on the 400 4-bit SFT calibration completions. No verdict in this entry uses the post-hoc scale, so none
  changes. The ×18.1 row stays as recorded
  (fitted on bf16-merged SFT completions). Source: `results/posthoc_calibration_4bit/sft_eval_scaled_crps_fit.jsonl`,
  metrics by `scripts/phase95_compare.headline` (the code of every other row; the ×18.1 row reproduces exactly).
- **Accuracy:** the 4-bit SFT model is about 2 points more accurate than the bf16 merge (36.8% vs 34.5%). Most of
  RL's "as-trained accuracy" in #15 is therefore the base, not RL. S3 (split vs uniform) is unaffected.
- **Uniform credit:** evaluated as trained, it is still collapsed (σ = 0 on 19.5% of answers, median σ 0.05%).
  This is #6's collapse, now confirmed on the eval set.

**What this means.**
- **Split credit's σ:** the coverage effect is real and much larger than the superseded numbers showed (≈ 42% vs
  ≈ 11%). But the σ is informative about nothing: ρ is no better than uniform credit's, and the surface-feature
  baseline wins every N test.
- **"σ widens but does not track the error"** is now the corrected, pre-registered result.
- **The superseded headline "S2 narrowly supported" is withdrawn.**

**Also tonight (descriptive, tier 2):**
- **Determinism:** all four reruns of D65's tapes are byte-identical to the originals (completions and rewards):
  greedy and k = 4 sampled, for split credit seed 0 and E1a. The 4-bit eval is fully deterministic on this GPU.
- **σ instability (D62), evaluated as trained:** the within-problem SD of ln σ over 4 samples is 0.85 for split
  credit (1.01 merged) and 0.35 for E1a (0.94 merged). Much of the "σ moves by a factor of e between samples" in
  MORNING 2026-10-17 §5 came from the merge.
- **The FINDINGS #5 adapter, as trained:** σ is SFT-like (median 0.057%, 1σ cov 12.8%); accuracy 39.0%.
- **The FINDINGS #6 adapter (step 759), as trained:** σ = 0 on 97.5% of eval answers (43.2% merged), with 1σ
  coverage 2.8%. This is the training-log collapse of #6 (95–98% after step ~550), at full strength on the eval
  set. The merge had hidden half of it.
- **E1 at step 150, as trained:** E1a median σ 2.7%, 1σ cov 46.5%, ρ −0.00; E1b 2.0%, 41.2%, 0.02. Most of the
  final effect is there by step 150, as before.
- **D66's resume path on the GPU** (Unsloth, 4-bit PEFT): **PASS** on all three pre-set criteria
  (`results/resume_gpu_check/report.md`).
  - The logs are complete and the tape replays.
  - The curriculum cells for steps 1–4 match the uninterrupted run.
  - The reloaded adapter lands near the uninterrupted step 4, not step 2 (‖B4 − A4‖ = 0.0033 vs ‖B4 − A2‖ = 0.0089).

**Provenance:**
- D67 (committed `bb7ff3c`, before any corrected S/N/E result);
- analysis code `d80ee83`, plus the per-seed output-name fix `20ec565`;
- eval tapes `…/compare/{sft_4bit,rl287_4bit}`, `…/e1_eval/{final_4bit,step150_4bit}`, `results/e1/sft_4bit`;
- analyses in `results/corrected/`.

## #17. X1: a shallow text model of σ\* reaches "partial propagation" on A6, but its slope is 0.06 against an ideal 1 (D73, 2026-10-18)

**Claim.** A gradient-boosted model trained on problem text alone (TF-IDF plus the measurements parsed from the text)
predicts σ\* well *across* problems: test R² 0.83 on ln σ\*. *Within* A6 pairs, where only the values change, it barely
moves its σ: slope +0.063 (90% CI [+0.03, +0.10]) against an ideal of 1. That is D61's "partial propagation".

**What this means.** Its A6b slope is equivalent to 0, so the labelled joint verdict is "partial propagation" (D59's
verdict: "propagates"). This is the bar the Phase 12 LLM must clear (D73's reading rules). Most of what a text model
knows about σ\* is the noise level and the template, which do not vary within a pair.

| σ source (problem text only) | test R² (ln σ\*) | A6 slope [90% CI] | A6 label | A6b slope [90% CI] | labelled verdict |
|---|---|---|---|---|---|
| noise (stated p only) | 0.62 | 0.000 [0.00, 0.00] | no propagation | 0.000 [0.00, 0.00] | no propagation |
| + TF-IDF | 0.79 | +0.002 [−0.00, +0.00] | no propagation | 0.000 [−0.00, +0.00] | no propagation |
| **+ parsed magnitudes (primary)** | **0.83** | **+0.063 [+0.03, +0.10]** | **partial propagation** | −0.000 [−0.00, +0.00] | **partial propagation** |
| reference: σ̂ = σ\* (exact) | – | +1.000 | full propagation | 0.000 | full propagation |

- **One-sided test:** the A6 slope is > 0 at p = 0.0008 (D58).
- **Within pairs:** the primary model's Δ ln σ̂ correlates with Δ ln σ\* at only 0.16, and has the right sign in 48.6%
  of pairs (Δ ln σ\* averages 1.08 within a pair).
- **By depth:** the slope's CI excludes 0 only at depth 2 (+0.11 [+0.04, +0.18]). At depth 5, outside the fit set's
  depths, it is −0.01 [−0.05, +0.03].
- **Nesting:** the noise level explains R² 0.62 of ln σ\* across problems, and the text adds 0.17. Only the parsed
  magnitudes give any within-pair movement.
- **Pipeline references behave as specified:** σ̂ = σ\* gives slope 1 / "full propagation"; the noise-only model is
  constant within pairs (slope 0).
- **Caveats:**
  - **No answers:** X1 states no answer, so all 510 pairs of each set enter its primary analysis. An LLM's analysis
    keeps only pairs where both answers are computation-correct.
  - **One fit:** it is a single fit (seed 0, one grid selection by grouped CV); there is no seed variability.
- **Provenance:** `physcal/eval/x1_text_sigma.py`, `configs/x1_text_sigma.yaml` → `results/x1_text_sigma/`
  (`selection.json`, CV table and parse coverage, written before any pair prediction; `results.json`; `summary.md`).
  Fit set: 12,024 generator-2.0.0 training-split problems (1,349 templates, seeds 50,000,000+). Test set: 1,224
  (51,000,000+).

## #18. Uniform-credit seeds 1 and 2 (R2-6): the corrected family S verdicts hold 3 vs 3; split credit's coverage gain is seed-robust, its ρ gain is consistent but tiny (robustness, D71; 2026-10-19)

**Claim.** With three uniform-credit seeds instead of one, the problem-level seed-mean tests give the same verdicts as
FINDINGS #16: S1a, S1b and S3 supported, S2 not supported. Across seeds, split credit's 1σ coverage beats uniform credit's
in all 9 cross-seed comparisons, by about 31 points. Its ρ(σ, err) is also higher in all 9, but the values are tiny:
0.03–0.08 against 0.01. This is a robustness analysis, not the pre-registered verdict (D71); FINDINGS #16 stands.

**(A) Problem level, both arms as seed means** (one-sided paired bootstrap over 400 problems, Holm; conditional on
these seeds):

| test | split (3-seed mean) | control | difference [95% CI] | p | supported |
|---|---|---|---|---|---|
| S1a: 1σ cov > SFT-only | 41.8% | 12.0% | +29.8 pts [+25.5, +34.2] | < 1e-4 | yes |
| S1b: 1σ cov > uniform (3-seed mean) | 41.8% | 10.8% | +31.0 pts [+26.8, +35.3] | < 1e-4 | yes |
| S2: ρ(σ, err) > uniform (3-seed mean) | 0.061 | 0.012 | +0.048 [−0.085, +0.178] | 0.241 | no |
| S3: accuracy non-inferior (−5 pts) | 37.7% | 37.2% | +0.5 pts [−0.8, +1.8] | < 1e-4 | yes |

**(B) Seed level** (descriptive; exact permutation over the 20 relabellings, so one-sided p ≥ 0.05 and two-sided
p ≥ 0.10 (R2-4); no verdict):

| metric | split seeds 0 / 1 / 2 | uniform seeds 0 / 1 / 2 | Δ | perm. p (one / two-sided) | cross-seed pairs with the expected sign |
|---|---|---|---|---|---|
| 1σ coverage | 41.8 / 41.3 / 42.5% | 10.5 / 11.5 / 10.5% | +31.0 pts | 0.05 / 0.10 | 9 / 9 |
| ρ(σ, err) | 0.077 / 0.034 / 0.071 | 0.015 / 0.014 / 0.009 | +0.048 | 0.05 / 0.10 | 9 / 9 |
| accuracy | 37.8 / 36.8 / 38.5% | 37.3 / 37.5 / 36.8% | +0.5 pts | 0.25 / 0.50 | 6 / 9 |
| σ = 0 share | 0 / 0 / 0.3% | 19.5 / 17.1 / 23.1% | −19.8 pts | – / 0.10 | – |
| median relative σ | 1.61 / 1.64 / 1.59% | 0.052 / 0.057 / 0.041% | ×31 | – / 0.10 | – |

- **The new uniform seeds look like the seed-0 control:**
  - 1σ coverage 11.5% and 10.5%; accuracy 37.5% and 36.8%;
  - σ = 0 on 17–23% of answers; median σ about 0.05%;
  - parse failures 0.5%; no OOM; 40.8 and 40.7 s/step.
- **Trajectories (`results/corrected/robustness_3v3/trajectories.png`):** from about step 100, all three split-credit
  seeds raise their training σ about 20× and drive σ = 0 to zero. All three uniform seeds stay flat at about 0.06%
  with σ = 0 near 25%. Mean reward and pass rate overlap throughout.
- **Reading:** the S1 effect (coverage, σ collapse) does not depend on which uniform seed is the control. S2 keeps
  failing at the problem level. At the seed level every split seed's ρ exceeds every uniform seed's, but the gain
  (+0.05) is a tenth of the A5 text baseline's ρ (≈ 0.65). Per R2-4, this is not a seed-level significance claim.
- **R2-1 on its first GPU use (D72):**
  - **The reloads:** both corrected evals reproduced the training-time fingerprint exactly. Log-prob and σ-token
    differences were 0.0, all 64 sampled completions were byte-identical, and provenance matched.
  - **Labels:** seed 2's check is "pass". Seed 1's is "inconclusive", only because its adapter's own log-prob effect
    on the fixed continuations (0.0047 nats/token) is just under D72's floor of 0.005. Its σ-token effect (TV 0.018)
    discriminates clearly.
- **Provenance:** `results/corrected/robustness_3v3/` (`scripts/robustness_3v3.py`, written before the seed-1/2
  data); runs `results/train/overnight2-…-seed{1,2}/`; corrected evals in `compare/rl287_4bit/`; fingerprints in
  `physcal_fingerprint.json`.

## #19. X2: no stated σ and no text model ranks a correct answer above an incorrect one to the same problem; a hidden-state probe does slightly, but no better than the same probe on a random-init model (D74; 2026-10-20)

**Claim.** With problem difficulty held fixed (516 problems where the corrected SFT model's k = 8 samples are both right
and wrong; 5,390 correct/incorrect pairs), stated σ carries no answer-specific information:
- the SFT model's σ scores 0.499;
- split credit's σ scores 0.491;
- A5's text p (problem + answer + reasoning) scores 0.497, and ties on 64% of pairs.

A linear probe on the SFT model's hidden state at the "±" token ranks the correct answer higher in 54.7% of pairs (Holm
supported). But the same probe on a random-init model of the same architecture reaches 53.0%, and the difference
(+0.017 [−0.015, +0.050]) is not distinguishable from 0. D74's labelled outcome is **"another combination: reported as
observed"**.

| method (D74) | pairwise accuracy [95% CI] | one-sided p (> 0.5) | Holm | ties / missing |
|---|---|---|---|---|
| 1. stated σ, SFT | 0.499 [0.474, 0.524] | 0.54 | not supported | 8.9% |
| 2. stated σ, split credit seed 0 @287 | 0.491 [0.461, 0.521] | 0.73 | not supported | 0.4% |
| 3. A5 p channel (4-bit fit) | 0.497 [0.481, 0.514] | 0.63 | not supported | 64.2% |
| 4. problem-text-only (check) | 0.500 exactly | – | – | 100% |
| **5. hidden-state probe, SFT (4-bit)** | **0.547 [0.520, 0.574]** | 0.0001 | **supported** | 0.5% |
| 6. hidden-state probe, random init | 0.530 [0.503, 0.557] | 0.013 | not supported (threshold 0.0125) | 0.5% |
| A5 σ (secondary) | 0.517 [0.494, 0.538] | 0.07 | – | 37.5% |
| probe cross-fitted on the X2 answers (secondary) | 0.540 [0.514, 0.565] | 0.001 | – | 0.5% |

- **Paired differences (bootstrap over problems):**
  - probe − SFT σ: +0.048 [+0.011, +0.086];
  - probe − random init: +0.017 [−0.015, +0.050], so **"probe succeeds" (D74) is not met**;
  - A5 p − probe: −0.050 [−0.080, −0.020].
- **The probes:**
  - **SFT:** layer 12 was chosen (CV AUROC 0.875 for correctness *across* problems on the 1,998 calibration
    completions).
  - **Random init:** layer 1 was chosen, with CV AUROC 0.863 across problems. That is the FINDINGS #7 pattern again:
    across problems, the signal is in the text.
  - **The cross-fitted probe** chose layer 1 in four of five folds, so what separates answers within a problem looks
    like surface features of the answer tokens.
- **By depth:** no method is above 0.58 at any depth (table in `results/x2/summary.md`).
- **What this means:**
  - **A5's text σ:** its between-problem advantage (ρ ≈ 0.65, FINDINGS #10 / #16) is difficulty prediction. Within a
    problem, A5's p barely varies, and when it does, it is at chance.
  - **The model's own σ** (SFT or split credit) says nothing about which of its answers is right.
  - **The one positive signal** (the probe) is small, and a random-weight network reading the same tokens gets most of
    it.
  - **Against D74's reading rules,** this sits between "every primary method fails" (not met: the probe is Holm
    supported) and "the information exists" (not met: it does not beat random init). Reported as observed (R2-5: no
    positive epistemic claim from X2).
- **Sampling facts:**
  - **Answers:** 1,200 candidates × 8 = 9,600 completions, 99.6% parsed, 33.8% correct within 1%.
  - **The kept set:** 516 mixed problems (43%; D74 expected about 44%), and 4,109 answers with a "±".
  - **Correct answers per kept problem:** 1 → 183 problems, … 7 → 64.
- **Deviations / process (DECISIONS D81, D86–D88):**
  - **Sampling:** batch 32, found by the probe under a 5.75 GiB cap (a host process held 1.65 GB of VRAM). Six chunks
    of 200, torch seed 0 + chunk.
  - **Method 2:** the first run timed out by spilling into WSL shared memory. It was rerun after the queue with
    batch 16 under the cap, and an answer of μ = 0 now counts as a tie. The queue's first analysis (method 2 counted
    as p = 1) is superseded by `results/x2/results.json`. Holm's verdicts are the same in both.
- **Provenance:**
  - **Code:** `physcal/eval/x2_pairs.py`, `x2_hidden.py`, `x2_analysis.py`; `configs/x2.yaml`.
  - **Chunks:** `results/x2/sft_k8/chunk_*`.
  - **Selection and suffixes:** `results/x2/selected.json`, `results/x2/split_seed0_suffix/`.
  - **Hidden states:** `results/x2/hidden/*/*/{index,info}.json`; the arrays are local. The Unsloth path passed the
    logit check (gap ≤ 0.5%).
  - **The random-init weights:** content hash `13ededef3f3acbd4`.

## #20. X3 pilot: the RL runs' own base + LoRA can learn a σ that ranks errors (ρ 0.58, "capacity likely"), but only between problems, not within them (D79; 2026-10-20)

**Claim.** Supervised on the out-of-fold A5 σ (teacher), a fresh LoRA r = 16 on the 4-bit merged SFT base (exactly the
RL runs' setup, 2 epochs, 500 steps) states a σ that tracks the error.
- **On the 399 frozen-set answers:** ρ(σ, |log error|) = **0.575 [0.499, 0.643]**, against split credit's 0.03–0.08
  and SFT's 0.04.
- **Label:** the pre-registered **"capacity likely"**. The CI's lower end is 0.499, against the 0.5 bar.
- **Its σ is also calibrated:** 1σ coverage 66.4%, crps_log −0.306, between the teacher's −0.291 and SFT's −0.322.
- **Within problems** (X2's pairs), the student is at chance: 0.513 [0.484, 0.543].

| σ source (μ fixed) | ρ(σ, err) [95% CI] | crps_log | 1σ / 2σ cov | median rel. σ | fidelity vs teacher | X2 pairwise acc. |
|---|---|---|---|---|---|---|
| student (a), A5 teacher | **+0.575 [+0.499, +0.643]** | −0.306 | 66.4 / 79.4% | 10.8% | 0.716 | 0.513 [0.484, 0.543] |
| student (b), true-error labels | +0.583 [+0.509, +0.650] | −0.348 | 30.8 / 37.6% | 0.3% | 0.601 | 0.488 [0.460, 0.517] |
| A5 teacher σ (4-bit fit) | 0.664 | −0.291 | 66.4 / 76.9% | 6.6% | – | – |
| SFT stated σ (4-bit) | 0.038 | −0.322 | 12.0 / 17.0% | 0.1% | – | – |

- **Paired comparisons:**
  - student (a) − teacher: ρ −0.088 [−0.155, −0.024];
  - student (a) − SFT: +0.537 [+0.402, +0.669];
  - (b) − (a): +0.008 [−0.054, +0.072].
- **Student (b) (D82):**
  - **Ranking:** as good as (a).
  - **Calibration:** badly overconfident (median σ 0.3%, 1σ coverage 31%, crps_log below SFT's).
  - **Why:** its targets are bimodal (tiny for correct answers, large for wrong ones), and greedy decoding takes the
    mode.
- **Failures:** no parse failures, unit errors or changed μ in either student. Both pass the enforced R2-1 reload
  check (D77): log-prob ratios 0.030 / 0.027, σ-token 0.023 / 0.039, and adapter effects 0.077 / 0.074 nats/token,
  well above the floor.
- **What this means (D79's reading):** split credit's ρ ≈ 0.06 is not a capacity limit of this base and LoRA. The
  same setup learns, by SFT, a σ almost as good as its teacher's. But what it learns is the teacher's kind of
  knowledge, i.e. which *problems* are hard (X2: 0.51 within problems). This matches FINDINGS #19. The pilot is one
  seed at one budget; a full X3 needs its own specification (D79).
- **Provenance:**
  - **Code and config:** `physcal/train/x3.py`, `configs/x3.yaml`.
  - **Data:** `results/x3/data/` (committed before training).
  - **Runs:** `results/x3/{a,b}/` (summary, fingerprint, eval tapes).
  - **Results:** `results/x3/pilot.{json,md}`.
  - **Process:** student (a)'s X2 eval was rerun after the μ = 0 fix (D88). Its frozen-set numbers come from the
    first run.

## #21. X4: asked to check its own answer, the SFT model says "yes" to almost everything; its P(yes) does not rank right answers above wrong ones to the same problem, with or without recomputation (D89 + D91; 2026-10-21 night)

**Claim (D89 outcome 4).** `sft` is not Holm-supported: 0.524 [0.498, 0.550] on X2's 5,390 pairs, p = 0.033.
Self-verification fails within problems, consistent with X2 (FINDINGS #19). Asking the model does not rank its right
answers above its wrong ones.
- **Neither Holm family supports any arm:** D91's five arms (thresholds 0.010–0.05), or D89's original three (sft
  0.033 vs 0.0167).
- **None of D91's flags holds:**
  - final-value-sensitive: `sft − perturbed` −0.000 [−0.028, +0.028];
  - recomputation helps: `sft − direct` −0.002 [−0.008, +0.004];
  - direct self-verification succeeds: direct is not Holm-supported, although `direct − random_init` +0.037
    [+0.001, +0.073] has a CI just above 0.

| arm (all 516 problems; CIs: bootstrap over problems) | pairwise accuracy | p (> 0.5) | AUROC across problems | extra tokens / answer |
|---|---|---|---|---|
| `sft`: recompute (primary) | 0.524 [0.498, 0.550] | 0.033 | 0.469 [0.442, 0.496] | 93.4 (18.4 generated) |
| `direct`: no recomputation (D91) | 0.526 [0.500, 0.553] | 0.025 | 0.463 [0.436, 0.490] | 75.0 (0 generated) |
| `perturbed`: final μ × 1.25 (D91) | 0.524 [0.498, 0.551] | 0.038 | 0.450 [0.423, 0.477] | 94.5 |
| `shuffled`: another problem's text | 0.513 [0.488, 0.539] | 0.153 | 0.463 [0.435, 0.491] | 99.9 |
| `random_init` | 0.489 [0.465, 0.514] | 0.81 | 0.517 [0.495, 0.539] | 587 (512 generated) |
| X2 probe (reference) | 0.547 [0.520, 0.574] | 0.0001 | 0.727 [0.708, 0.745] | – |

- **The model barely checks.**
  - **The verdicts:** it writes a verdict on 98% of answers, and that verdict is "yes" on 99.9% of correct and 99.8%
    of incorrect ones.
  - **The recomputation:** the median is 13 tokens, so recompute and direct give the same ranking (difference
    −0.002).
  - **The mean renormalized P(yes)** is slightly *higher* on incorrect answers: 0.820 vs 0.804.
- **It notices an inconsistent final value, but not a wrong one.** Multiplying the final μ by 1.25 lowers P(yes) on
  98.6% [97.7, 99.4] of the originally-correct answers: the mean drops from 0.80 to 0.72. The drop is the same for
  incorrect answers (0.82 → 0.74), so the right-vs-wrong ranking is unchanged (0.524 → 0.524).
  - **So the check behaves like** "does the final line match the steps", not "are the steps right".
  - **Another problem's text** lowers P(yes) further (0.58 / 0.64), again without ranking.
- **Across problems it is anti-informative:** AUROC 0.469 [0.442, 0.496] for `sft`, i.e. more confident on the
  problems it gets wrong. The X2 probe, by contrast, reaches 0.727 across problems.
- **Limits:**
  - **The model and prompt:** one model (4-bit SFT Qwen2.5-Math-1.5B) and one fixed prompt (`x4_v1`), greedy.
  - **Direct mode** keeps the prompt's "You may redo the calculation" (D91).
  - **The random-init model** puts essentially no mass on yes/no (yes + no = 0.000), so its log-odds is a
    near-arbitrary ratio. It is a weak control.
  - **Later (R3-1):** the yes/no order variant (affirmative bias) and a deterministic checker ceiling are not run yet.
- **Provenance:**
  - **Code:** `physcal/eval/x4_verify.py`, `x4_analysis.py`; `configs/x4.yaml`.
  - **Rows:** `results/x4/<arm>/rows.jsonl` and `meta.json`.
  - **Results:** `results/x4/results.json`, `summary.md`.
  - **Queue:** `results/queue_x4/` (D90, D91, D96).

## #22. X2 robustness (post-hoc, exploratory, D92): asking the other 7 samples beats every single-sample readout within problems; clearly separated pairs sharpen the ranking but change no verdict; X2's CIs were already problem-level (2026-10-21 night)

**Status:** post-hoc; it cannot change a D74 or D89/D91 verdict. All eight X2 methods were recomputed with X2's own code
and reproduce `results/x2/results.json` exactly. The random-init and cross-fitted probes needed X2's thread settings
(D96). CIs: bootstrap over problems.

| method | X2 pairwise acc. | pooled pair-weighted | separated pairs only (158 problems, 719 pairs) |
|---|---|---|---|
| **self-consistency:** share of the other 7 samples within 1% | **0.643 [0.616, 0.671]** | 0.681 [0.654, 0.708] | **0.861 [0.820, 0.901]** |
| hidden-state probe (SFT) | 0.547 [0.520, 0.574] | 0.553 [0.526, 0.580] | 0.635 [0.570, 0.697] |
| hidden-state probe (random init) | 0.530 [0.503, 0.557] | 0.535 [0.509, 0.561] | 0.584 [0.523, 0.645] |
| length only (shorter = more confident) | 0.535 [0.511, 0.559] | 0.547 [0.524, 0.569] | 0.607 [0.549, 0.663] |
| X4 `sft` (self-verification) | 0.524 [0.498, 0.550] | 0.513 [0.489, 0.539] | 0.544 [0.485, 0.603] |
| SFT stated σ | 0.499 [0.474, 0.524] | 0.496 [0.472, 0.520] | 0.501 [0.442, 0.558] |
| A5 p | 0.497 [0.481, 0.514] | 0.499 [0.482, 0.515] | 0.516 [0.470, 0.563] |

- **a. Clearly separated pairs** (right ≤ 0.25%, wrong ≥ 10%): every readout with some signal gains, the probe
  0.547 → 0.635. But the probe's lead over the random-init probe stays within noise (0.635 vs 0.584), and the stated σ
  and A5 p stay at 0.50. The X2 reading does not depend on near-miss pairs.
- **b. Length:**
  - **Pooled,** correct answers are 17 tokens shorter (186 vs 203).
  - **Within problems,** the difference is only −1.8 [−2.7, −1.0] tokens, and the correct answers are shorter in 55%
    of problems.
  - **As a baseline,** length alone reaches 0.535, statistically indistinguishable from the probe
    (−0.012 [−0.045, +0.021]).
- **c. Self-consistency** (ChatGPT's "independent re-solve"; it costs 8 samples per problem at inference):
  - **The strongest within-problem signal in the project,** +0.096 [+0.058, +0.133] above the probe.
  - **The likely reason:** right answers cluster; wrong answers scatter.
  - **It needs the other samples:** it is not something the single answer, its σ or its hidden state carries.
- **d. CIs:** X2's published CIs already resampled problems (the recomputed ones match to 3 decimals). The pooled
  pair-weighted version gives the same picture.
- **Provenance:**
  - **Code and config:** `physcal/eval/x2_robustness.py`, `configs/x2_robustness.yaml`.
  - **Results:** `results/x2/robustness/` (`results.json`, `summary.md`, the logs, including the 1-thread run the
    reproduction check refused).

## #23. X1 and the noise heuristics on A6c: no surface reader propagates noise allocation; X1's A6 "partial propagation" does not transfer (D95 + D99; 2026-10-21)

**Status:** descriptive. These are bars for Phase 12, not tests. X1 was refit exactly (D73's fit set and
hyperparameters; every published D73 number reproduced with max difference 0.0) and applied to the frozen A6c set
(510 pairs; checksum checked). Every pair enters, since none of these sources states an answer. Slope: OLS of
Δ ln σ̂ on Δ ln σ\* with an intercept, 90% CI, one-sided bootstrap over pairs (ideal 1).

| σ source | A6 slope (FINDINGS #17) | A6b slope (≡ 0?) | **A6c slope** (supported?) |
|---|---|---|---|
| X1 noise | +0.000 | +0.000 (yes) | +0.000 [+0.000, +0.000] (no) |
| X1 noise_text | +0.002 [−0.000, +0.005] | +0.000 (yes) | −0.002 [−0.007, +0.003] (no) |
| **X1 noise_text_magnitudes (primary)** | **+0.063 [+0.031, +0.097]** | −0.000 (yes) | **−0.002 [−0.003, −0.000] (no)** |
| heuristics max / mean / RSS (givens or shown) | +0.000 | +0.000 (yes) | +0.000 (no) |
| reference σ̂ = σ\* | +1.000 | +0.000 | +1.000 (yes) |

- **What A6c removes:**
  - **The surface:** within an A6c pair the shown values and the multiset of "± p%" are identical, so a reader of
    the surface has nothing to move on.
  - **X1:** its σ̂ is bit-identical within 57.5% of pairs. The rest differ only through TF-IDF tokens with attached
    punctuation ("5%," vs "5%."), and the mean Δ ln σ̂ is +0.0002.
- **X1's A6 slope came from the magnitudes.** On A6 the displayed values differ within a pair, so magnitude
  features can follow σ\*. On A6c they cannot.
- **The bar for Phase 12:** any positive A6c slope beats every surface baseline built so far. A6c is the cleanest
  of the three tests of "the model propagates noise through the chain".
- **A detail:** X1's noise-only feature vectors are bit-identical in only 47.5% of pairs, yet its σ̂ is identical in
  100%. The mean and RSS of the same levels differ in the last bit with summation order (the D96 effect), and the
  GBT's binning absorbs that.
- **Provenance:**
  - **Code:** `physcal/eval/x1_a6c.py`, `tests/test_x1_a6c.py`.
  - **Results:** `results/x1_text_sigma/a6c/`: `reproduction.json`, `results.json`, `summary.md`, and
    `predictions.jsonl` (σ̂ per member).
  - **Heuristics:** `results/noise_heuristics/`.

## #24. Phase 12 stage A: crps_log targets σ\*, but one realized truth barely tells a 2×-off σ from σ\*; the expected score against the known predictive distribution is 5–23× less noisy (D100; 2026-10-21)

**Status:** descriptive; no hypothesis test. Every forecast is centred at μ = f(x_meas) (R3-6's oracle-centred
diagnostic), so no problem is selected on the correctness of an answer. Verifier 1.1.0 crps_log, unchanged; the
diagnostic's realized score equals `reward()`'s `score_final` exactly (300 / 300 rendered answers).

- **1. The optimum (capped gate sample, 10,005 noisy problems):**
  - **Pooled:** the mean realized crps_log over σ = c·σ\* peaks at **c = 1.000 [0.966, 1.017]**, and every one of
    the 15 depth × level cells has a CI containing 1. The verifier's score, realized against the noise-free truth,
    rewards σ\* on average.
  - **The curve is flat and asymmetric:**
    - **c = 0.5** loses 0.0027, about 8% of the score at c = 1 (−0.034);
    - **c = 2** loses 0.0057, about 17%.
    - **This is the Gaussian CRPS itself:** E CRPS = √(2(1 + c²)/π) − c/√π, in units of σ\*. In log terms,
      understating σ by 2× costs half as much as overstating it by 2×.
- **2. The resolution per problem (A6 + A6b + A6c, 3,060 problems):**

  | σ − alternative | realized mean diff | realized SD | σ\* better, realized | σ\* better, expected | Var ratio realized / expected |
  |---|---|---|---|---|---|
  | 0.5·σ\* | +0.0031 | 0.0114 | 56.5% | 100% | 22.7 |
  | 2·σ\* | +0.0057 | 0.0151 | 76.5% | 100% | 10.0 |
  | RSS heuristic | +0.0028 | 0.0125 | 58.0% | 91.3% | 6.4 |
  | X1 primary | +0.0021 | 0.0108 | 60.7% | 97.0% | 7.6 |
  | constant (median σ\*) | +0.0046 | 0.0155 | 65.2% | 98.4% | 5.5 |

  - **On one realized truth,** σ\* beats an alternative off by 2× on only 56–77% of problems. That is the
    per-prompt signal a σ-RL step sees.
  - **Against the predictive distribution** (fresh MC, N = 4,000), σ\* wins on 91–100% of problems, and the score
    differences have 5–23× less variance. This confirms R3's consensus point 6 numerically.
  - **To separate σ\* from RSS by two standard errors** takes ~80 problems realized and ~12 expected.
- **3. The ranking:** ranking the sources by mean realized score agrees with ranking them by median |ln(σ / σ\*)|
  (Kendall τ = 0.85). The disagreements all come from the asymmetry: 0.5·σ\* outscores sources that are closer to
  σ\* in log terms.
- **4. The verifier's details:** no oracle-centred score reaches the −5 clip, and the 1e-6 floor never binds.
- **Oracle-centred baselines (pooled):**
  - **X1 primary:** the best non-oracle source, with median |ln(σ / σ\*)| 0.33, vol beta 0.59 and 1σ coverage 0.64.
  - **RSS heuristic:** median |ln(σ / σ\*)| 0.40, vol beta 0.49, coverage 0.62.
  - **Max:** median |ln(σ / σ\*)| 0.69, coverage 0.45.
  - **The vol beta here is across problems.** Because the noise level drives σ\*, the heuristics reach ~0.5 across
    problems while their within-pair slopes are exactly 0 (#23).
- **μ error against σ signal:** with μ = f(x_meas)·(1 + δ):
  - **δ = 1%** costs σ\* 0.0009, a third of the σ\*-vs-RSS gap.
  - **δ = 5%** costs it 0.016, six times that gap.
  - **So in an RL group whose samples' μ vary by a few percent,** μ dominates the reward, which is why R3-6 centres
    the σ diagnostic at f(x_meas).
- **For Phase 12's pre-registration (my reading, to decide):**
  1. **Report σ propagation primarily with the oracle-centred expected score,** plus A6 / A6b / A6c. Realized
     crps_log needs ~10× more problems for the same resolution.
  2. **For σ-RL,** a reward of the expected score against the generator's predictive distribution would be far less
     noisy than the realized one. That is a verifier change; it is not made, only flagged.
- **MC check:** the fresh draws' CRPS-optimal σ\* is within a median 1.2% (max 6.8%) of the stored one.
- **Provenance:**
  - **Code:** `physcal/eval/oracle_centred.py` (the diagnostic, reusable), `physcal/eval/stage_a.py`,
    `configs/stage_a.yaml`.
  - **Tests:** `tests/test_oracle_centred.py`, `tests/test_stage_a.py`.
  - **Results:** `results/stage_a/`.

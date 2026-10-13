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

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

# Phase 3.6 plan: short SFT warm-up on generator worked solutions (DRAFT, not approved, not run)

**Why:** the phase-4 run (MORNING.md §3) showed that the policy rarely finishes a well-formed answer inside the
token budget (Qwen3.5-2B truncates 70–90% at 1024 tokens; Qwen2.5-Math ignores the `<answer>` format), so RL
sees a gradient on only ~40% of steps. The selection rule did pick a model (Qwen3.5-2B, 72.9% binary
zero-variance), so this plan is a recommendation, not a triggered stop. A short supervised warm-up on
solutions the generator writes itself should teach the format and the arithmetic habit, so that sampled
groups contain a mix of right and wrong answers.

## Data
- Source: `generate_problem(..., split="train")`, seeds 0–N (disjoint from eval seeds ≥ 1e9). Uniform over
  the 12 training cells (kinematics/dynamics/energy × depths 1–4); depth 5 and held-out families are
  excluded by the generator itself.
- Target completion, built from `worked_solution()` and the chain:
  1. SI conversion of each given that is used (distractors are named as not needed);
  2. one line per chain step: formula → solved form → substituted numbers → result with unit;
  3. `<steps>` with the requested intermediates;
  4. `<answer>value ± σ unit</answer>` in the requested unit.
- N ≈ 3,000 examples (~2–2.5k tokens each with the v4 prompt). That's about 1 epoch on the 4060 in under
  an hour.

## The σ question (needs your decision before running)
SFT targets are exact, so any σ written into them is a *taught* σ, not an honest one. Options:
- (a) A constant relative σ (e.g. 2%). Simple, but it teaches a fixed heuristic that RL must then unlearn,
  and it confounds RQ2.
- (b) σ derived from the rounding of the inputs (≈ 0.1–0.5%). Honest for a perfect solver, but teaches
  over-confidence to an imperfect one.
- (c) Mask the σ tokens out of the SFT loss (train on the value only) and let GRPO with `crps_log` shape σ
  from scratch. **Recommended:** it keeps the calibration question clean.

## Training
- LoRA r = 16 (same target modules as GRPO), 4-bit base, lr 1e-4, 1 epoch, max length 2304, loss only on
  the completion (and per (c), not on the σ number).
- Run it on the model the probe found strongest; same pinned revision.

## Success criteria (re-run the existing tools)
- `uv run python -m physcal.eval.model_probe --only <label>` with the adapter merged: binary zero-variance
  < 80% (target < 50%) on depths 1–2; greedy ID accuracy (≤ 1%) clearly above the probe baseline; depth-5
  accuracy reported separately, as a check on generalization rather than template memorization.
- Then phase 4 GRPO starts from the SFT adapter.

## Risks
- Imitation of one solution template: depth-5 extrapolation and phrasing variety will show it.
- Even an SFT'd 1.5–2B model may stay below threshold at depths 3–4, which argues for a curriculum
  (phase 5/8) that starts GRPO on depths 1–2.

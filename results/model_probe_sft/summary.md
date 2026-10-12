# Model probe (phase 3.5)

Inference only. Greedy: 200 ID + 200 depth-5 problems. Group: 48 ID problems × k = 8 at T = 1.0.
Prompt v4 (worked example, no σ hint), max_new_tokens 1024, generator 1.1.0, reward rule crps_log.

## Greedy accuracy (σ-free, failures count as wrong)

| model | split | ≤1% | ≤2% | ≤5% | ≤10% | strict ≤1% | truncated | parse fail | missing ± | 1σ cov | mean tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|

## Group probe (what a GRPO group of 8 sees)

| model | zero-var binary | zero-var binary+steps | zero-var crps_log | zero-var crps | pass@8 | pass rate | truncated |
|---|---|---|---|---|---|---|---|

Binary zero-variance by depth: 

## Memory

Estimated GRPO footprint: 4-bit + LoRA r=16, group 8, 1024 completion tokens, micro-batch 2, Unsloth-style chunked log-probs and offloaded checkpoints (configs/footprint.yaml). Budget 6.5 GB.

| model | text params | prompt tokens (max) | est. GRPO peak | phase at peak | pessimistic (plain TRL) | measured inference peak, greedy (torch / device) | measured, group |
|---|---|---|---|---|---|---|---|

## Selection

```
{
  "decision": "incomplete",
  "missing": [
    "qwen3.5-2b-sft",
    "qwen2.5-math-1.5b-instruct-sft"
  ]
}
```

Revisions: 

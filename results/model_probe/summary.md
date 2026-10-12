# Model probe (phase 3.5)

Inference only. Greedy: 200 ID + 200 depth-5 problems. Group: 48 ID problems × k = 8 at T = 1.0.
Prompt v4 (worked example, no σ hint), max_new_tokens 1024, generator 1.1.0, reward rule crps_log.

## Greedy accuracy (σ-free, failures count as wrong)

| model | split | ≤1% | ≤2% | ≤5% | ≤10% | strict ≤1% | truncated | parse fail | missing ± | 1σ cov | mean tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b-instruct | id | 2.5% | 3.0% | 4.0% | 5.5% | 2.5% | 42.5% | 51.5% | 0.5% | 3.3% | 665 |
| qwen2.5-1.5b-instruct | depth_extrapolation | 1.0% | 1.0% | 1.5% | 2.5% | 1.0% | 51.0% | 54.0% | 0.0% | 1.1% | 735 |
| qwen2.5-math-1.5b-instruct | id | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 28.0% | 100.0% | 0.0% | – | 748 |
| qwen2.5-math-1.5b-instruct | depth_extrapolation | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 59.0% | 100.0% | 0.0% | – | 944 |
| qwen3.5-2b | id | 13.5% | 13.5% | 13.5% | 14.0% | 7.0% | 69.5% | 82.0% | 11.5% | 29.0% | 853 |
| qwen3.5-2b | depth_extrapolation | 2.5% | 2.5% | 2.5% | 2.5% | 2.5% | 87.5% | 90.0% | 2.0% | 6.2% | 983 |

## Group probe (what a GRPO group of 8 sees)

| model | zero-var binary | zero-var binary+steps | zero-var crps_log | zero-var crps | pass@8 | pass rate | truncated |
|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b-instruct | **91.7%** | 81.2% | 2.1% | 4.2% | 8.3% | 1.3% | 5.5% |
| qwen2.5-math-1.5b-instruct | **97.9%** | 95.8% | 95.8% | 97.9% | 2.1% | 0.3% | 17.2% |
| qwen3.5-2b | **72.9%** | 60.4% | 56.2% | 58.3% | 27.1% | 7.0% | 76.3% |

Binary zero-variance by depth: qwen2.5-1.5b-instruct: d1 83.3%, d2 100.0%, d3 100.0%, d4 83.3%; qwen2.5-math-1.5b-instruct: d1 100.0%, d2 91.7%, d3 100.0%, d4 100.0%; qwen3.5-2b: d1 25.0%, d2 75.0%, d3 100.0%, d4 91.7%

## Memory

Estimated GRPO footprint: 4-bit + LoRA r=16, group 8, 1024 completion tokens, micro-batch 2, Unsloth-style chunked log-probs and offloaded checkpoints (configs/footprint.yaml). Budget 6.5 GB.

| model | text params | prompt tokens (max) | est. GRPO peak | phase at peak | pessimistic (plain TRL) | measured inference peak, greedy (torch / device) | measured, group |
|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b-instruct | 1.54 B | 1115 | **3.67 GB** | rollout | 11.2 GB | 4.48 / 6.84 GiB | 4.47 / 6.97 GiB |
| qwen2.5-math-1.5b-instruct | 1.54 B | 1115 | **3.67 GB** | rollout | 11.2 GB | 4.48 / 6.83 GiB | 4.47 / 5.80 GiB |
| qwen3.5-2b | 1.88 B | 1156 | **4.27 GB** | training | 15.4 GB | 6.92 / 7.67 GiB | 5.43 / 6.19 GiB |

## Selection

```
{
  "decision": "train",
  "selected": "qwen3.5-2b",
  "threshold": 0.8,
  "budget_gb": 6.5,
  "ranking": [
    [
      "qwen3.5-2b",
      0.7291666666666666
    ],
    [
      "qwen2.5-1.5b-instruct",
      0.9166666666666666
    ],
    [
      "qwen2.5-math-1.5b-instruct",
      0.9791666666666666
    ]
  ]
}
```

Revisions: `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; `Qwen/Qwen2.5-Math-1.5B-Instruct` @ `aafeb0fc6f22cbf0eaeed126eff8be45b0360a35`; `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc`

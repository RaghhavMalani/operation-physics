# R5-5 feasibility: 4-bit + LoRA on the 8 GB GPU, and the untrained bases on our task (D133)

Not a pre-registration. **Training:** stage R4's own path (LoRA r 16 on all 7 projections, batch 2 × 4, adamw_8bit, gradient checkpointing) on D-pos-v2's 3,000 R4-1 rows, tokenized with each model's template, 6 steps under the launch cap. **Baseline:** 50 development A6c problems (10 per depth), prompt v5, greedy, 1,280 new tokens, the default verifier; OOM halves the batch (run_eval's rule). **Lenient arithmetic** ignores the format: the final number (boxed, else in <answer>, else the last) within 1% of f(x_meas), in the requested unit or SI.

## Training

| model | seq tokens mean / p99 / max | rows > 2,048 | OOM check: longest batch | peak reserved (GiB) | device peak (GiB) | s / step | × reference |
|---|---|---|---|---|---|---|---|
| gemma2_2b | 1565 / 2005 / 2177 | 26 | pass (2023 tokens, 3.85 GiB) | 3.85 | 4.95 | 12.9 | 2.10 |
| llama32_3b | 1394 / 1736 / 1858 | 0 | pass (1858 tokens, 3.21 GiB) | 4.20 | 5.83 | 10.2 | 1.66 |
| llama32_1b | 1394 / 1736 / 1858 | 0 | pass (1858 tokens, 1.90 GiB) | 3.30 | 5.75 | 4.2 | 0.68 |
| qwen25_math_1p5b | 1500 / 1920 / 2082 | 6 | pass (2043 tokens, 2.18 GiB) | 2.58 | 5.13 | 6.2 | 1.00 |

## Untrained base on 50 development problems

| model | parsed | units ok | σ stated | truncated | median tokens | strict computation-correct | lenient arithmetic (by depth 1–5) | eval batch that fits | peak reserved (GiB) | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma2_2b | 82% | 80% | 80% | 8% | 116 | 0% | 0% (0% / 0% / 0% / 0% / 0%) | 5 | 5.89 | 560 |
| llama32_3b | 86% | 74% | 74% | 12% | 154 | 0% | 2% (0% / 10% / 0% / 0% / 0%) | 5 | 5.87 | 509 |
| llama32_1b | 48% | 42% | 42% | 2% | 123 | 0% | 0% (0% / 0% / 0% / 0% / 0%) | 10 | 3.48 | 94 |
| qwen25_math_1p5b | 0% | 0% | 0% | 22% | 921 | 0% | 22% (70% / 40% / 0% / 0% / 0%) | 10 | 3.56 | 252 |

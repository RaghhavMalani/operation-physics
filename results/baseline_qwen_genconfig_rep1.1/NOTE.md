# Reference run: greedy decoding *with Qwen's default repetition_penalty = 1.1*

This was the first baseline run (commit 044ae76). The eval config then left the sampling parameters unset, so
`generate()` silently picked up `repetition_penalty: 1.1` from the model's own `generation_config.json`.
HF applies this penalty to every token already in the context, including the prompt, so the model was nudged
away from reusing the problem's numbers and units.

The official baseline in `results/baseline/` sets every decoding parameter explicitly
(`repetition_penalty: 1.0`, pure greedy). This folder is kept for comparison only. Its `rollouts.meta.json`
does not record the 1.1 penalty, because it wasn't in our config; that gap is why decoding is explicit now.

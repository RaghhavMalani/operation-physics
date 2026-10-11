# physcal

Calibrated physics reasoning via RLVR. See `CLAUDE.md` for the spec, `PROGRESS.md` for status, and
`DECISIONS.md` for design choices. (The full README with results is phase 10.)

```bash
uv run pytest                                   # full test suite
uv run python -m physcal.eval.run_eval          # baseline eval -> results/baseline/
uv run python -m physcal.eval.run_eval --limit 4 --output-dir /tmp/smoke   # quick smoke run
```

```python
from physcal.env.generator import generate_problem, worked_solution
from physcal.verify.reward import reward

p = generate_problem("energy", depth=3, seed=42)          # split="train" refuses held-out cells
print(p.text)
print(worked_solution(p))
r, breakdown = reward(p, "<steps>...</steps><answer>1.2 ± 0.1 m</answer>")
```

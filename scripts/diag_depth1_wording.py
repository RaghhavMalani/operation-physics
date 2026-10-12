"""Diagnostic (phase 3, inference only): does the depth-1 sentence "<steps> may be empty" make the model skip
its calculation?

Takes the 50 depth-1 problems of the baseline ID split under generator 1.0.0 (the version this diagnostic
was run with), replaces only that sentence, and reruns greedy decoding with prompt v1. Generator 1.1.0 adopted
the replacement wording (D30). The rollouts are logged to results/diagnostics/depth1_wording/. They are not
replayable with `replay()`, because the problem text was edited after generation.

    uv run python scripts/diag_depth1_wording.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from physcal.config import REPO_ROOT, load_config
from physcal.env.generator import env_config
from physcal.env.render import NO_STEPS_TEMPLATES
from physcal.eval.run_eval import HFBackend, build_split
from physcal.prompt import build_messages
from physcal.verify.reward import reward, reward_config

NO_STEPS_TEMPLATE = NO_STEPS_TEMPLATES["1.0.0"]
REPLACEMENT = NO_STEPS_TEMPLATES["1.1.0"]


def main() -> None:
    eval_cfg, env_cfg, rw_cfg = load_config("eval"), env_config(), reward_config()
    env_cfg["generator"]["version"] = "1.0.0"  # reproduce the problems this diagnostic was run on
    problems = [p for p in build_split(0, eval_cfg["splits"]["id"], eval_cfg, env_cfg) if p.depth == 1]
    texts = [p.text.replace(NO_STEPS_TEMPLATE, REPLACEMENT) for p in problems]
    assert all(NO_STEPS_TEMPLATE not in t for t in texts)
    backend = HFBackend(eval_cfg["model"]["id"], eval_cfg["model"], eval_cfg["generation"])
    gens = backend([build_messages(t, eval_cfg["prompt"]) for t in texts])

    out = REPO_ROOT / "results" / "diagnostics" / "depth1_wording"
    out.mkdir(parents=True, exist_ok=True)
    rows, reasons = [], Counter()
    for p, text, g in zip(problems, texts, gens):
        r, bd = reward(p, g.text, rw_cfg)
        reasons[bd["reason"]] += 1
        rows.append({"family": p.family, "depth": p.depth, "seed": p.seed, "edited_text": text,
                     "completion": g.text, "n_tokens": g.n_tokens, "reward": r, "breakdown": bd})
    (out / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    ok = [row["breakdown"] for row in rows if row["breakdown"]["reason"] == "ok"]
    summary = {
        "n": len(rows),
        "replacement": REPLACEMENT,
        "original": NO_STEPS_TEMPLATE,
        "answer_only_rate": sum("=" not in row["completion"] for row in rows) / len(rows),
        "accuracy": sum(b["correct"] for b in ok) / len(rows),
        "median_abs_rel_error": sorted(b["abs_rel_error"] for b in ok)[len(ok) // 2] if ok else None,
        "reasons": dict(reasons),
        "mean_tokens": sum(row["n_tokens"] for row in rows) / len(rows),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

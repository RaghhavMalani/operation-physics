"""Diagnostic only (not the verifier, not used for selection): how accurate would a model be if its last
\\boxed{x} were read as the answer in the requested unit? Measures latent physics skill of a model that
ignores the <answer> format (Qwen2.5-Math).

    uv run python scripts/diag_boxed_answers.py results/model_probe/qwen2.5-math-1.5b-instruct/greedy/rollouts.jsonl
"""

import json
import re
import sys
from collections import defaultdict

from physcal.env.units import from_si

BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
NUM = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:\s*\\times\s*10\^\{?(-?\d+)\}?)?")
TOLS = (0.01, 0.02, 0.05, 0.10)


def boxed_value(text: str) -> float | None:
    boxes = BOXED.findall(text)
    if not boxes:
        return None
    m = NUM.search(boxes[-1].replace(",", ""))
    if not m:
        return None
    value = float(m.group(0).split("\\times")[0])
    return value * (10.0 ** int(m.group(1)) if m.group(1) else 1.0)


def main(path: str) -> None:
    by_split: dict[str, list] = defaultdict(list)
    for line in open(path):
        rec = json.loads(line)
        p = rec["problem"]
        value = boxed_value(rec["completion"])
        err = None
        if value is not None:
            truth = from_si(p["target_value_si"], p["target_si_unit"], p["target_unit"])
            err = abs(value - truth) / abs(truth)
        by_split[rec.get("split", "all")].append(err)
    out = {}
    for split, errs in by_split.items():
        n = len(errs)
        out[split] = {"n": n, "boxed_rate": sum(e is not None for e in errs) / n,
                      **{f"acc_{t:g}": sum(e is not None and e <= t for e in errs) / n for t in TOLS}}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])

"""Phase 3.6 final comparison: raw base vs SFT-only vs SFT + RL, same frozen eval sets and settings.

    uv run python -m physcal.train.compare --part base|sft|rl --run results/train/<run_id> --label <label> --budget 512
    uv run python -m physcal.train.compare --part tests --run results/train/<run_id>

Every part uses `eval_probe_greedy` (200 ID + 200 depth-5 problems of generator 1.1.0, prompt v4, greedy) with
max_new_tokens = the training budget, and loads the model text-only (`causal_lm`). `tests` runs the paired
McNemar / Wilcoxon tests (scripts/paired_tests.py) for rl-vs-sft (what RL added), rl-vs-base and sft-vs-base.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

from physcal.config import REPO_ROOT, load_config
from physcal.eval.run_eval import run

PAIRS = (("rl", "sft"), ("rl", "base"), ("sft", "base"))


def eval_config(part: str, label: str, budget: int, run_dir: Path, sft_dir: Path) -> dict:
    probe = load_config("model_probe")
    base = next(c for c in probe["candidates"] if c["label"] == label)
    cfg = copy.deepcopy(load_config(probe["greedy_config"]))
    cfg["generation"]["max_new_tokens"] = budget
    cfg["generation"]["batch_size"] = base.get("batch_size", cfg["generation"]["batch_size"])
    cfg["output_dir"] = str(run_dir / "compare" / part)
    cfg["run_id"] = f"{run_dir.name}-compare-{part}"
    cfg["model"].update(model_class="causal_lm", fallback_id=None,
                        chat_template_kwargs=base.get("chat_template_kwargs", {}))
    if part == "base":
        cfg["model"].update(id=base["id"], revision=base["revision"], lora=None)
    elif part == "sft":
        cfg["model"].update(id=str(sft_dir), revision=None, lora=None)
    else:
        cfg["model"].update(id=str(sft_dir), revision=None, lora=str(run_dir / "final_adapter"))
    return cfg


def paired_tests(run_dir: Path) -> dict:
    out = {}
    for after, before in PAIRS:
        a = run_dir / "compare" / after / "rollouts.jsonl"
        b = run_dir / "compare" / before / "rollouts.jsonl"
        if not (a.exists() and b.exists()):
            out[f"{after}_vs_{before}"] = "missing"
            continue
        target = run_dir / "compare" / f"paired_{after}_vs_{before}.json"
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "paired_tests.py"), str(b), str(a), str(target)],
                       check=True, capture_output=True)
        out[f"{after}_vs_{before}"] = json.loads(target.read_text())
    (run_dir / "compare" / "paired_tests.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", required=True, choices=["base", "sft", "rl", "tests"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--sft-dir", default=None)
    ap.add_argument("--budget", type=int, default=None)
    args = ap.parse_args()
    run_dir = Path(args.run) if Path(args.run).is_absolute() else REPO_ROOT / args.run
    if args.part == "tests":
        print(json.dumps(paired_tests(run_dir), indent=2)[:3000])
        return
    sft_dir = Path(args.sft_dir) if args.sft_dir else REPO_ROOT / "results" / "sft" / args.label / "merged"
    run(eval_config(args.part, args.label, args.budget, run_dir, sft_dir))


if __name__ == "__main__":
    main()

"""Evaluate a trained adapter on exactly the eval config its base model was probed with (phase 3.5 greedy).

    uv run python -m physcal.train.evaluate --run results/train/<run_id> --label <candidate label>

Writes <run>/eval_greedy/ (normal run_eval output) and <run>/before_after.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from physcal.config import REPO_ROOT, load_config
from physcal.eval.model_probe import candidate_eval_config
from physcal.eval.run_eval import run

KEYS = ("parse_fail_rate", "missing_sigma_rate", "unit_mismatch_rate", "coverage_1sigma", "coverage_2sigma",
        "median_rel_sigma", "median_rel_error", "truncation_rate", "mean_completion_tokens", "mean_reward",
        "mean_step_score")


def compare(before: dict, after: dict) -> dict:
    out = {}
    for split in before["splits"]:
        b, a = before["splits"][split], after["splits"][split]
        out[split] = {
            "accuracy_sigma_free": {"before": b["accuracy_at_tolerance"]["sigma_free"],
                                    "after": a["accuracy_at_tolerance"]["sigma_free"]},
            "accuracy_strict": {"before": b["accuracy_at_tolerance"]["strict"],
                                "after": a["accuracy_at_tolerance"]["strict"]},
            **{k: {"before": b.get(k), "after": a.get(k)} for k in KEYS},
            "mean_score_by_rule": {"before": b["mean_score_by_rule"], "after": a["mean_score_by_rule"]},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--adapter", default="final_adapter")
    ap.add_argument("--model-class", default=None,
                    help="override the probe's model class, e.g. causal_lm when the adapter was trained text-only")
    args = ap.parse_args()
    probe = load_config("model_probe")
    cand = next(c for c in probe["candidates"] if c["label"] == args.label)
    run_dir = Path(args.run) if Path(args.run).is_absolute() else REPO_ROOT / args.run
    cfg = candidate_eval_config(load_config(probe["greedy_config"]), cand, "greedy", run_dir)
    cfg["output_dir"] = str(run_dir / "eval_greedy")
    cfg["run_id"] = f"{run_dir.name}-eval-greedy"
    cfg["model"]["lora"] = str(run_dir / args.adapter)
    if args.model_class:
        cfg["model"]["model_class"] = args.model_class
    after = run(cfg)
    before = json.loads((REPO_ROOT / probe["output_dir"] / args.label / "greedy" / "metrics.json").read_text())
    (run_dir / "before_after.json").write_text(json.dumps(compare(before, after), indent=2))
    print(json.dumps(compare(before, after)["id"]["accuracy_sigma_free"], indent=2))


if __name__ == "__main__":
    main()

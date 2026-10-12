"""Re-score a logged rollout tape under another scoring rule (no model needed).

Usage:
    uv run python -m physcal.eval.rescore results/baseline/rollouts.jsonl --rule binary
Writes `rescored_<rule>.json` next to the tape.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

from physcal.config import load_config
from physcal.env.generator import problem_from_dict
from physcal.eval import calibration
from physcal.eval.signals import rule_configs, rule_outcomes
from physcal.tape.tape import read_meta, read_tape
from physcal.verify.reward import reward


def rescore(tape_path: str | Path, rule: str) -> dict[str, dict]:
    """Per split: the usual summary under `rule`, accuracy at several tolerances, and (for multi-sample tapes)
    per-rule group statistics. The tape's own logged reward config supplies every other setting."""
    meta = read_meta(tape_path)
    cfg = copy.deepcopy(meta["reward_config"])
    cfg["scoring"]["rule"] = rule
    cal = {**load_config("eval")["calibration"], **meta["eval_config"]["calibration"]}
    per_rule_cfgs = rule_configs(meta["reward_config"])
    by_split: dict[str, list[dict]] = defaultdict(list)
    for rec in read_tape(tape_path):
        problem = problem_from_dict(rec["problem"])
        _, bd = reward(problem, rec["completion"], cfg)
        by_split[rec.get("split", "all")].append({
            "problem": rec["problem"],
            "breakdown": bd,
            "by_rule": rule_outcomes(problem, rec["completion"], per_rule_cfgs),
        })
    out = {}
    for split, recs in by_split.items():
        summary = calibration.summarize([r["breakdown"] for r in recs], cfg["scoring"],
                                        cal["reliability_levels"], cal["pit_bins"])
        summary["accuracy_at_tolerance"] = calibration.accuracy_at_tolerances(recs, cal["accuracy_tolerances"])
        if len(recs) > len({(r["problem"]["family"], r["problem"]["depth"], r["problem"]["seed"]) for r in recs}):
            summary["groups"] = calibration.group_stats(recs)
        out[split] = summary
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tape")
    ap.add_argument("--rule", required=True)
    args = ap.parse_args()
    result = rescore(args.tape, args.rule)
    out = Path(args.tape).with_name(f"rescored_{args.rule}.json")
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    for split, s in result.items():
        print(f"{split}: accuracy {s['accuracy']:.3f}, answered {s['n_answered']}/{s['n']}, "
              f"parse_fail {s['parse_fail_rate']:.3f}, unit_mismatch {s['unit_mismatch_rate']:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

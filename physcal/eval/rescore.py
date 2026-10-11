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

from physcal.env.generator import problem_from_dict
from physcal.eval import calibration
from physcal.tape.tape import read_meta, read_tape
from physcal.verify.reward import reward


def rescore(tape_path: str | Path, rule: str) -> dict[str, dict]:
    meta = read_meta(tape_path)
    cfg = copy.deepcopy(meta["reward_config"])
    cfg["scoring"]["rule"] = rule
    cal = meta["eval_config"]["calibration"]
    by_split: dict[str, list[dict]] = defaultdict(list)
    for rec in read_tape(tape_path):
        _, bd = reward(problem_from_dict(rec["problem"]), rec["completion"], cfg)
        by_split[rec.get("split", "all")].append(bd)
    return {
        split: calibration.summarize(bds, cfg["scoring"], cal["reliability_levels"], cal["pit_bins"])
        for split, bds in by_split.items()
    }


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

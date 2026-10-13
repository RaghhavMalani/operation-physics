"""Re-score a logged rollout tape under another scoring rule, or under the current verifier (no model needed).

Usage:
    uv run python -m physcal.eval.rescore results/baseline/rollouts.jsonl --rule binary
        writes `rescored_<rule>.json` next to the tape (the tape's logged config, with the rule swapped);
    uv run python -m physcal.eval.rescore TAPE --current-verifier OUT_DIR
        writes a new, replayable tape + metrics.json to OUT_DIR, scored by configs/reward.yaml as it is now.
        The original tape is left untouched.
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
from physcal.verify.reward import reward, reward_config, verifier_version


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


def rescore_to_current(tape_path: str | Path, out_dir: str | Path) -> dict:
    """Copy a tape with every reward, breakdown and by_rule outcome recomputed under the current reward config,
    keeping the logged rule. The new meta.json records the source tape and both verifier versions; the new tape
    replays exactly under its own meta. Returns the new metrics.json (same shape as run_eval's)."""
    from physcal.eval.run_eval import write_metrics
    from physcal.tape.tape import TapeWriter

    tape_path, out = Path(tape_path), Path(out_dir)
    meta = read_meta(tape_path)
    cfg = reward_config()
    cfg["scoring"]["rule"] = meta["reward_config"]["scoring"]["rule"]
    per_rule_cfgs = rule_configs(cfg)
    new_meta = {k: v for k, v in meta.items() if k != "run_id"}
    new_meta.update(reward_config=cfg, rescored_from=str(tape_path),
                    rescored_from_verifier=verifier_version(meta["reward_config"]))
    out.mkdir(parents=True, exist_ok=True)
    records: dict[str, list[dict]] = defaultdict(list)
    keep = ("split", "sample", "n_tokens", "truncated")
    (out / "rollouts.jsonl").unlink(missing_ok=True)  # TapeWriter appends
    with TapeWriter(out / "rollouts.jsonl", meta["run_id"], new_meta) as tape:
        for rec in read_tape(tape_path):
            problem = problem_from_dict(rec["problem"])
            r, bd = reward(problem, rec["completion"], cfg)
            extra = {k: rec[k] for k in keep if k in rec}
            extra["by_rule"] = rule_outcomes(problem, rec["completion"], per_rule_cfgs)
            tape.log(rec["step"], problem, rec["completion"], r, bd, rec["model_id"], rec["lora_hash"], extra)
            records[rec.get("split", "all")].append({"problem": rec["problem"], "breakdown": bd, **extra})
    old = json.loads((tape_path.parent / "metrics.json").read_text())
    metrics = {**{k: v for k, v in old.items() if k != "splits"}, "splits": {},
               "verifier_version": verifier_version(cfg), "rescored_from": str(tape_path)}
    cal = {**load_config("eval")["calibration"], **meta["eval_config"]["calibration"]}
    write_metrics(metrics, dict(records), cfg, cal, meta["eval_config"]["generation"]["num_samples"], out)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tape")
    ap.add_argument("--rule", default=None)
    ap.add_argument("--current-verifier", default=None, metavar="OUT_DIR")
    args = ap.parse_args()
    if args.current_verifier:
        m = rescore_to_current(args.tape, args.current_verifier)
        for split, s in m["splits"].items():
            print(f"{split}: mean reward {s['mean_reward']:.4f}, crps_log {s['mean_score_by_rule']['crps_log']:.4f}, "
                  f"cov1 {s['coverage_1sigma']}, acc {s['accuracy']:.3f}")
        return
    if not args.rule:
        ap.error("--rule or --current-verifier is required")
    result = rescore(args.tape, args.rule)
    out = Path(args.tape).with_name(f"rescored_{args.rule}.json")
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    for split, s in result.items():
        print(f"{split}: accuracy {s['accuracy']:.3f}, answered {s['n_answered']}/{s['n']}, "
              f"parse_fail {s['parse_fail_rate']:.3f}, unit_mismatch {s['unit_mismatch_rate']:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

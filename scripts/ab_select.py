"""Phase 3.6 A/B decision (your rules), then fill configs/grpo_overnight.yaml.

1. Winner = lower binary (σ-free) zero-variance fraction in the k=8 group probe (ties: higher pass rate).
2. Budget: if the winner's completions truncate < 20% at 512 tokens, use 512, else 768. "Truncated at 512" =
   no end-of-turn within the first 512 generated tokens (n_tokens > 512, or cut at the probe's 1024 cap);
   sampling is identical up to that point whatever the cap, so this is read off the 1024-token probe.

    uv run python scripts/ab_select.py
"""

import json
import re
from pathlib import Path

from physcal.config import REPO_ROOT, load_config

TRUNCATION_RULE = 0.20
BUDGETS = (512, 768)


def truncation_at(rollouts: Path, budget: int) -> float:
    rows = [json.loads(line) for line in rollouts.open()]
    return sum(r["truncated"] or r["n_tokens"] > budget for r in rows) / len(rows)


def main() -> None:
    probe = load_config("model_probe_sft")
    root = REPO_ROOT / probe["output_dir"]
    rows = []
    for cand in probe["candidates"]:
        g = json.loads((root / cand["label"] / "group" / "metrics.json").read_text())["splits"]["id"]
        rollouts = root / cand["label"] / "group" / "rollouts.jsonl"
        rows.append({
            "label": cand["label"], "id": cand["id"],
            "binary_zero_variance": g["groups"]["zero_variance_fraction"]["binary"],
            "crps_log_zero_variance": g["groups"]["zero_variance_fraction"]["crps_log"],
            "pass_at_8": g["groups"]["pass_at_k"], "mean_pass_rate": g["groups"]["mean_pass_rate"],
            "binary_zero_variance_by_depth": {d: b["zero_variance_fraction"]["binary"]
                                              for d, b in g["groups"]["by_depth"].items()},
            "parse_fail_rate": g["parse_fail_rate"], "mean_completion_tokens": g["mean_completion_tokens"],
            "truncation_at": {str(b): truncation_at(rollouts, b) for b in (512, 768, 1024)},
            "accuracy_sigma_free": g["accuracy_at_tolerance"]["sigma_free"],
            "coverage_1sigma": g["coverage_1sigma"], "median_rel_sigma": g["median_rel_sigma"],
        })
    winner = sorted(rows, key=lambda r: (r["binary_zero_variance"], -r["mean_pass_rate"]))[0]
    budget = BUDGETS[0] if winner["truncation_at"]["512"] < TRUNCATION_RULE else BUDGETS[1]
    decision = {"candidates": rows, "winner": winner["label"], "budget": budget,
                "rule": f"lower binary zero-variance; budget 512 if truncation@512 < {TRUNCATION_RULE:.0%} else 768"}
    (root / "ab_selection.json").write_text(json.dumps(decision, indent=2))

    cand = next(c for c in probe["candidates"] if c["label"] == winner["label"])
    path = REPO_ROOT / "configs" / "grpo_overnight.yaml"
    text = path.read_text()
    text = re.sub(r"^run_id: .*$", f"run_id: overnight-grpo-{winner['label']}-crps_log-{budget}", text, count=1, flags=re.M)
    text = re.sub(r"^  id: null .*$", f"  id: {cand['id']}   # SFT A/B winner (results/model_probe_sft/ab_selection.json)",
                  text, count=1, flags=re.M)
    kwargs = json.dumps(cand.get("chat_template_kwargs", {})).replace('"', "")
    text = re.sub(r"^  chat_template_kwargs: .*$", f"  chat_template_kwargs: {kwargs}", text, count=1, flags=re.M)
    text = re.sub(r"^  text_only: .*$", f"  text_only: {'true' if 'qwen3.5' in winner['label'] else 'false'}",
                  text, count=1, flags=re.M)
    text = re.sub(r"^  max_completion_length: null .*$",
                  f"  max_completion_length: {budget}   # winner truncates {winner['truncation_at']['512']:.0%} at 512",
                  text, count=1, flags=re.M)
    path.write_text(text)
    print(json.dumps({k: v for k, v in decision.items() if k != "candidates"}, indent=2))
    for r in rows:
        print(r["label"], "binary zv", round(r["binary_zero_variance"], 3), "pass@8", round(r["pass_at_8"], 3),
              "trunc@512", round(r["truncation_at"]["512"], 3), "tokens", round(r["mean_completion_tokens"]))


if __name__ == "__main__":
    main()

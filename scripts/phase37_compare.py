"""Phase 3.7 comparisons on the frozen eval sets (200 ID + 200 depth-5), all scored by verifier 1.1.0.

    uv run python scripts/phase37_compare.py

PRIMARY (matched step 287): new RL @287 vs SFT-only, vs last night's RL (@287, its final step), and vs the
post-hoc control (SFT-only with σ scaled by the crps_log-fitted factor; the coverage-matched factor is a
sensitivity row). SECONDARY ("longer training"): new RL final vs new RL @287.
Writes <new run>/compare/phase37_paired.json and phase37_tables.md.
"""

import importlib.util
import json
import statistics as st
import sys
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/train/overnight-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-512/compare_verifier_1.1.0"
NEW = ROOT / "results/train/overnight2-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512/compare"
POSTHOC = ROOT / "results/posthoc_calibration"

TAPES = {
    "sft": OLD / "sft/rollouts.jsonl",
    "rl_lastnight": OLD / "rl/rollouts.jsonl",
    "posthoc_crps_fit": POSTHOC / "sft_eval_scaled_crps_fit.jsonl",
    "posthoc_coverage_matched": POSTHOC / "sft_eval_scaled_coverage_matched.jsonl",
    "rl287": NEW / "rl287/rollouts.jsonl",
    "rlfinal": NEW / "rlfinal/rollouts.jsonl",
}
PRIMARY = [("rl287", "sft"), ("rl287", "rl_lastnight"), ("rl287", "posthoc_crps_fit"),
           ("rl287", "posthoc_coverage_matched")]
SECONDARY = [("rlfinal", "rl287"), ("rlfinal", "sft"), ("rlfinal", "posthoc_crps_fit")]


def _paired_module():
    spec = importlib.util.spec_from_file_location("paired_tests", ROOT / "scripts/paired_tests.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def headline(path: Path) -> dict:
    by_split: dict[str, list[dict]] = {}
    for line in path.open():
        r = json.loads(line)
        by_split.setdefault(r["split"], []).append(r)
    out = {}
    for split, recs in by_split.items():
        bds = [r["breakdown"] for r in recs]
        ok = [b for b in bds if b["reason"] == "ok" and b.get("sigma_si") is not None]
        n = len(recs)
        out[split] = {
            "n": n,
            "acc_1pct": sum(b["reason"] == "ok" and b["abs_rel_error"] <= 0.01 for b in bds) / n,
            "coverage_1sigma": st.mean(bool(b["within_1sigma"]) for b in ok) if ok else None,
            "coverage_2sigma": st.mean(bool(b["within_2sigma"]) for b in ok) if ok else None,
            "median_rel_sigma": st.median(b["rel_sigma"] for b in ok) if ok else None,
            "mean_crps_log": st.mean(b["scores"]["crps_log"] if b["reason"] == "ok" else -5.0 for b in bds),
            "mean_reward": st.mean(r["reward"] for r in recs),
            "parse_fail": sum(b["reason"] == "parse_fail" for b in bds) / n,
            "sigma_zero_rate": sum(b["sigma_si"] == 0 for b in ok) / len(ok) if ok else None,
            # does stated σ track the actual error? (Spearman over answers with σ; 0 = no information)
            "spearman_sigma_vs_error": float(spearmanr([b["rel_sigma"] for b in ok],
                                                       [b["abs_rel_error"] for b in ok]).statistic) if ok else None,
        }
    return out


def main() -> None:
    pt = _paired_module()
    present = {k: p for k, p in TAPES.items() if p.exists()}
    missing = sorted(set(TAPES) - set(present))
    result = {"missing": missing, "headline": {k: headline(p) for k, p in present.items()}, "paired": {}}
    for group, pairs in (("primary", PRIMARY), ("secondary", SECONDARY)):
        result["paired"][group] = {}
        for after, before in pairs:
            if after in present and before in present:
                out = NEW / f"paired_{after}_vs_{before}.json"
                pt.main(str(present[before]), str(present[after]), str(out))
                result["paired"][group][f"{after}_vs_{before}"] = json.loads(out.read_text())
    NEW.mkdir(parents=True, exist_ok=True)
    (NEW / "phase37_paired.json").write_text(json.dumps(result, indent=2))

    pct = lambda x: "–" if x is None else f"{100 * x:.1f}%"
    pct3 = lambda x: "–" if x is None else f"{100 * x:.3f}%"
    lines = ["| split | model | acc ≤1% | 1σ cov | 2σ cov | median rel. σ | σ = 0 | ρ(σ, err) | mean crps_log | "
             "mean reward |", "|---|---|---|---|---|---|---|---|---|---|"]
    for split in ("id", "depth_extrapolation"):
        for k, h in result["headline"].items():
            s = h.get(split)
            if s:
                lines.append(f"| {split} | {k} | {pct(s['acc_1pct'])} | {pct(s['coverage_1sigma'])} | "
                             f"{pct(s['coverage_2sigma'])} | {pct3(s['median_rel_sigma'])} | "
                             f"{pct(s['sigma_zero_rate'])} | {s['spearman_sigma_vs_error']:.2f} | "
                             f"{s['mean_crps_log']:.4f} | {s['mean_reward']:.4f} |")
    lines += ["", "| group | comparison | split | 1σ covered: before → after (only after / only before) | McNemar p | "
                  "2σ covered | McNemar p | strict ≤1% | McNemar p | crps_log Δ | Wilcoxon p |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for group, pairs in result["paired"].items():
        for name, res in pairs.items():
            for split, r in res.items():
                c1, c2, cs, cr = r["covered_1sigma"], r["covered_2sigma"], r["correct_strict"], r["crps_log"]
                lines.append(
                    f"| {group} | {name} | {split} | {c1['before']} → {c1['after']} ({c1['only_after']} / "
                    f"{c1['only_before']}) | {c1['mcnemar_p']:.3g} | {c2['before']} → {c2['after']} | "
                    f"{c2['mcnemar_p']:.3g} | {cs['before']} → {cs['after']} | {cs['mcnemar_p']:.3g} | "
                    f"{cr['mean_diff']:+.4f} | {cr['wilcoxon_p']:.3g} |")
    (NEW / "phase37_tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if missing:
        print(f"missing: {missing}", file=sys.stderr)


if __name__ == "__main__":
    main()

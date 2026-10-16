"""Exploratory, no tests (DECISIONS.md D62): greedy vs sampled stated σ at eval.

    uv run python scripts/sigma_greedy_vs_sampled.py

Arms (same frozen eval problems / prefixes; sampled = k completions per problem at the training settings):
- split-credit seed 0 @287: <run>/compare/rl287 (greedy) vs <run>/compare/rl287_sampled. μ varies between samples.
- E1a and E1b final: <run>/e1_eval/final (greedy) vs <run>/e1_eval/final_sampled. μ is fixed (SFT prefix).
Per arm: failure rate, median relative σ, σ = 0 share, 1σ / 2σ coverage, mean crps_log (failures at the floor),
Spearman ρ(σ, error) on answered rows. For sampled arms also: the per-problem spread of ln σ (σ floored at 1e-6
relative, the PREREGISTRATION convention) across the k samples, and where the greedy σ falls among them.
Writes results/sigma_greedy_vs_sampled/report.{json,md}.
"""

import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physcal.tape.tape import read_tape  # noqa: E402

SIGMA_FLOOR_REL = 1e-6  # PREREGISTRATION.md: σ = 0 is floored here wherever a log of σ is needed
FAIL_CRPS_LOG = -5.0    # the crps_log clip; failures score it (D41), as in phase95_compare.py
SPLIT = ROOT / "results/train/split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed0"
E1 = {"E1a": ROOT / "results/train/e1a-sigma-only-free-seed0", "E1b": ROOT / "results/train/e1b-sigma-only-bins-seed0"}
OUT = ROOT / "results/sigma_greedy_vs_sampled"


def tape_rows(path: Path) -> list[dict]:
    """One row per completion of a run_eval tape: key, relative σ, coverage, crps_log, relative error."""
    rows = []
    for r in read_tape(path):
        p, b = r["problem"], r["breakdown"]
        ok = b["reason"] == "ok" and b.get("sigma_si") is not None
        rows.append({"key": (r["split"], p["family"], p["depth"], p["seed"]), "fail": not ok,
                     "rel_sigma": b["rel_sigma"] if ok else None, "err": b["abs_rel_error"] if ok else None,
                     "cov1": ok and bool(b["within_1sigma"]), "cov2": ok and bool(b["within_2sigma"]),
                     "crps_log": b["scores"]["crps_log"] if ok else FAIL_CRPS_LOG})
    return rows


def e1_rows(path: Path) -> list[dict]:
    """One row per σ suffix of an E1 eval (rows.jsonl): failures already carry the floor and no σ."""
    out = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out.append({"key": tuple(r["key"]), "fail": r["fail"], "rel_sigma": r["rel_sigma_used"],
                    "err": r["abs_log_error"], "cov1": r["cov1"], "cov2": r["cov2"], "crps_log": r["crps_log"]})
    return out


def describe(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["fail"] and r["rel_sigma"] is not None]
    paired = [r for r in ok if r["err"] is not None]
    return {"n": len(rows), "fail_rate": sum(r["fail"] for r in rows) / len(rows),
            "median_rel_sigma": st.median(r["rel_sigma"] for r in ok) if ok else None,
            "sigma_zero": sum(r["rel_sigma"] == 0 for r in ok) / len(ok) if ok else None,
            "cov1": st.mean(r["cov1"] for r in rows), "cov2": st.mean(r["cov2"] for r in rows),
            "mean_crps_log": st.mean(r["crps_log"] for r in rows),
            "rho_sigma_err": float(spearmanr([r["rel_sigma"] for r in paired], [r["err"] for r in paired]).statistic)
            if len(paired) > 2 else None}


def ln_sigma(r: dict) -> float:
    return math.log(max(r["rel_sigma"], SIGMA_FLOOR_REL))


def within_problem(greedy: list[dict], sampled: list[dict]) -> dict:
    """Spread of ln σ across a problem's samples, and the greedy σ's mid-rank among them (0 = below all)."""
    by_key = defaultdict(list)
    for r in sampled:
        if not r["fail"] and r["rel_sigma"] is not None:
            by_key[r["key"]].append(ln_sigma(r))
    spreads, ranks = [], []
    for g in greedy:
        s = by_key.get(g["key"], [])
        if len(s) >= 2:
            spreads.append(st.pstdev(s))
        if s and not g["fail"] and g["rel_sigma"] is not None:
            x = ln_sigma(g)
            ranks.append((sum(v < x for v in s) + 0.5 * sum(v == x for v in s)) / len(s))
    return {"problems_with_2plus_samples": len(spreads),
            "median_sd_ln_sigma": st.median(spreads) if spreads else None,
            "greedy_mean_midrank": st.mean(ranks) if ranks else None,
            "greedy_below_all_samples": sum(r == 0 for r in ranks) / len(ranks) if ranks else None}


def main() -> dict:
    arms = {"split-credit seed 0 @287": (tape_rows, SPLIT / "compare/rl287/rollouts.jsonl",
                                         SPLIT / "compare/rl287_sampled/rollouts.jsonl")}
    arms |= {name: (e1_rows, run / "e1_eval/final/rows.jsonl", run / "e1_eval/final_sampled/rows.jsonl")
             for name, run in E1.items()}
    out = {}
    for name, (reader, greedy_path, sampled_path) in arms.items():
        if not (greedy_path.exists() and sampled_path.exists()):
            out[name] = {"missing": [str(p.relative_to(ROOT)) for p in (greedy_path, sampled_path) if not p.exists()]}
            continue
        g, s = reader(greedy_path), reader(sampled_path)
        out[name] = {"greedy": describe(g), "sampled": describe(s), "within_problem": within_problem(g, s)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(out, indent=2))
    fmt = lambda x, f: "–" if x is None else format(x, f)
    lines = ["# Greedy vs sampled stated σ at eval (exploratory, D62; no tests)", "",
             "| model | decoding | rows | fail | median rel. σ | σ = 0 | 1σ cov | 2σ cov | mean crps_log | ρ(σ, err) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, a in out.items():
        if "missing" in a:
            lines.append(f"| {name} | missing: {', '.join(a['missing'])} | | | | | | | | |")
            continue
        for mode in ("greedy", "sampled"):
            d = a[mode]
            lines.append(f"| {name} | {mode} | {d['n']} | {d['fail_rate']:.1%} | {fmt(d['median_rel_sigma'], '.3%')} | "
                         f"{fmt(d['sigma_zero'], '.1%')} | {d['cov1']:.1%} | {d['cov2']:.1%} | {d['mean_crps_log']:.4f} | "
                         f"{fmt(d['rho_sigma_err'], '.2f')} |")
    lines += ["", "| model | median within-problem SD of ln σ | greedy σ mean mid-rank among samples | greedy σ below all samples |",
              "|---|---|---|---|"]
    for name, a in out.items():
        if "missing" not in a:
            w = a["within_problem"]
            lines.append(f"| {name} | {fmt(w['median_sd_ln_sigma'], '.3f')} | {fmt(w['greedy_mean_midrank'], '.2f')} | "
                         f"{fmt(w['greedy_below_all_samples'], '.1%')} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


if __name__ == "__main__":
    main()

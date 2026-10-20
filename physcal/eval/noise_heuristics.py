"""Noise heuristics (DECISIONS.md D95, R3-4 stage A): a stated relative σ read off the visible noise percentages alone,
the "copy the percentage" baselines that A6c is built to defeat. CPU only.

    nice uv run python -m physcal.eval.noise_heuristics analyze

Heuristics: the max, the mean and the root-sum-square (√Σ s_i²) of the stated levels s_i. Scope `givens` (primary):
the chain's givens, i.e. a solver that knows which values matter; scope `shown` (secondary): every measurement shown,
distractors included, i.e. a surface reader. Each heuristic is scored on the frozen A6, A6b and A6c sets with the
pair analyses of `a6_pairs` (as if μ = f(x_meas): every member counts as computation-correct), and per problem against
σ* (median |ln σ_h − ln σ*|, Spearman ρ, the slope of ln σ_h on ln σ*). Within every A6, A6b and A6c pair the levels
are the same multiset, so every heuristic gives y = 0: it fails A6 and A6c and passes A6b by construction.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.eval import a6_pairs

HEURISTICS = {"max": max, "mean": lambda s: sum(s) / len(s), "rss": lambda s: math.sqrt(sum(x * x for x in s))}
SCOPES = ("givens", "shown")


def levels(problem: dict, scope: str) -> list[float]:
    """The stated relative levels of the problem's measured values (its givens, or every shown measurement), sorted:
    a permutation of the same levels (an A6c pair) must give a bit-identical heuristic, not one that differs in the
    last float digit from the summation order."""
    shown = problem["givens"] + (problem.get("distractors") or [] if scope == "shown" else [])
    return sorted(float(v["noise_rel"]) for v in shown if v.get("noise_rel"))


def sigma(problem: dict, name: str, scope: str) -> float | None:
    s = levels(problem, scope)
    return HEURISTICS[name](s) if s else None


def row(problem: dict, name: str, scope: str, floor: float) -> dict:
    """An a6_pairs member row with the heuristic as the stated relative σ (oracle-centred: μ = f(x_meas))."""
    n = problem["noise"]
    s = sigma(problem, name, scope)
    return {"pair": (problem["family"], problem["depth"], n["level"], n["pair"]["pair_seed"]),
            "member": n["pair"]["member"], "kind": n["pair"].get("kind", "sensitivity"),
            "sigma_star_log": n["sigma_star_log"], "truth": problem["target_value_si"], "ok": s is not None,
            "computation_correct": s is not None, "stated_log": None if s is None else math.log(max(s, floor)),
            "sigma_zero": None if s is None else s == 0}


def per_problem(problems: list[dict], name: str, scope: str) -> dict:
    from scipy.stats import spearmanr

    pts = [(math.log(s), math.log(p["noise"]["sigma_star_log"])) for p in problems
           if (s := sigma(p, name, scope)) and p["noise"]["sigma_star_log"] > 0]
    h, t = np.array(pts).T
    tc = t - t.mean()
    return {"n": len(pts), "median_abs_log_ratio": float(np.median(np.abs(h - t))),
            "median_log_ratio": float(np.median(h - t)), "spearman": float(spearmanr(h, t).statistic),
            "slope_on_log_sigma_star": float((tc * (h - h.mean())).sum() / (tc * tc).sum())}


def analyze(cfg: dict | None = None) -> dict:
    pc = load_config("pairs")
    cfg = cfg or pc["noise_heuristics"]
    acfg = pc["analysis"]
    floor = acfg["sigma_floor_rel"]
    root = REPO_ROOT / "data" / "eval_sets" / cfg["generator_version"]
    res: dict = {"decision": "D95 (R3-4 A)", "sets": {}}
    for name in cfg["sets"]:
        path = root / f"{name}.jsonl"
        if not path.exists():
            res["sets"][name] = {"missing": str(path)}
            continue
        problems = [json.loads(line) for line in path.read_text().splitlines()]
        out = {}
        for h in HEURISTICS:
            for scope in SCOPES:
                pair = a6_pairs.analyze_rows([row(p, h, scope, floor) for p in problems], acfg, pc["invariance_analysis"])
                out[f"{h}/{scope}"] = {"pairs": pair, "per_problem": per_problem(problems, h, scope)}
        res["sets"][name] = {"n_problems": len(problems), "heuristics": out}
    od = REPO_ROOT / cfg["output_dir"]
    od.mkdir(parents=True, exist_ok=True)
    (od / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (od / "summary.md").write_text(render(res))
    return res


def _pair_stat(pairs: dict) -> str:
    kind = pairs["kind"]
    t = pairs["primary_all_answered"] if kind == "allocation" else pairs["primary_computation_correct"]
    verdict = {"sensitivity": lambda: "supported" if pairs["supported"] else "not supported",
               "allocation": lambda: "supported" if pairs["supported"] else "not supported",
               "invariance": lambda: "equivalent to 0" if pairs["equivalent_to_zero"] else "not equivalent"}[kind]()
    return f"{t['slope']:+.3f} ({verdict}; {t['n_pairs']} pairs)"


def render(res: dict) -> str:
    L = ["# Noise heuristics on A6 / A6b / A6c (D95, R3-4 A)", "",
         "Stated relative σ = max / mean / RSS of the visible levels; `givens` = the chain's givens, `shown` = every "
         "measurement shown. Pair slope: A6 and A6c on Δ ln σ* (ideal 1), A6b on Δ ln truth (ideal 0).", ""]
    for name, s in res["sets"].items():
        if s.get("missing"):
            L += [f"## {name}: missing", ""]
            continue
        L += [f"## {name} ({s['n_problems']} problems)", "",
              "| heuristic | pair slope | median abs ln(σ_h / σ*) | median ln(σ_h / σ*) | Spearman ρ(σ_h, σ*) | "
              "slope of ln σ_h on ln σ* |", "|---|---|---|---|---|---|"]
        for h, r in s["heuristics"].items():
            pp = r["per_problem"]
            L.append(f"| {h} | {_pair_stat(r['pairs'])} | {pp['median_abs_log_ratio']:.3f} | {pp['median_log_ratio']:+.3f} "
                     f"| {pp['spearman']:.3f} | {pp['slope_on_log_sigma_star']:.3f} |")
        L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["analyze"])
    ap.parse_args()
    print(render(analyze()))


if __name__ == "__main__":
    main()

"""D50/A6 and D59/A6b analyses: does a model's stated σ track σ* within pairs? (propagation vs lookup)

    uv run python -m physcal.eval.a6_pairs analyze <eval tape> [--split a6_pairs|a6b_pairs] [--out DIR]
    uv run python -m physcal.eval.a6_pairs joint <A6 analysis json> <A6b analysis json> [--out DIR]

Per pair (same template, depth, phrasing and noise level; worlds chosen so that σ* differs, physcal/env/pairs.py):
x = ln σ*_hi − ln σ*_lo, y = ln σ_hi − ln σ_lo, with σ the stated relative σ (σ/|μ|, crps_log's scale; σ = 0 → the
crps_log floor, the PREREGISTRATION convention). Primary: pairs where both answers are computation-correct (A1:
|ln μ − ln f(x_meas)| ≤ τ); OLS slope of y on x with intercept; one-sided bootstrap over pairs for slope > 0.
A surface-feature lookup predicts slope ≈ 0, honest propagation slope ≈ 1. Secondary: all pairs with both answers
parsed; sensitivity: pairs without a stated σ = 0.

A6b (D59, invariance pairs, relative σ* equal within each pair): the same y regressed on m = ln(truth_hi/truth_lo);
propagation predicts slope ≈ 0 (TOST with margin δ: the 90% bootstrap CI inside [−δ, δ]), a fixed absolute σ ≈ −1.
`joint` gives the two-sided propagation verdict: "propagates" iff A6 is supported and A6b is equivalent to 0.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.tape.tape import read_tape


def member_row(problem: dict, b: dict, tau: float, floor: float) -> dict:
    n = problem["noise"]
    ok = b["reason"] == "ok" and b.get("sigma_si") is not None and b["mu_si"] > 0
    row = {"pair": (problem["family"], problem["depth"], n["level"], n["pair"]["pair_seed"]),
           "member": n["pair"]["member"], "kind": n["pair"].get("kind", "sensitivity"),
           "sigma_star_log": n["sigma_star_log"], "truth": problem["target_value_si"], "ok": ok,
           "computation_correct": False, "stated_log": None, "sigma_zero": None}
    if ok:
        row["computation_correct"] = abs(math.log(b["mu_si"] / n["point_estimate_si"])) <= tau
        row["stated_log"] = math.log(max(b["sigma_si"] / b["mu_si"], floor))
        row["sigma_zero"] = b["sigma_si"] == 0
    return row


def pairs_from_rows(rows: list[dict]) -> list[dict]:
    by: dict = {}
    for r in rows:
        by.setdefault(r["pair"], {})[r["member"]] = r
    return [{"lo": v["lo"], "hi": v["hi"]} for _, v in sorted(by.items()) if set(v) == {"lo", "hi"}]


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


def _x(p: dict, regressor: str) -> float:
    if regressor == "sigma_star":
        return math.log(p["hi"]["sigma_star_log"]) - math.log(p["lo"]["sigma_star_log"])
    return math.log(p["hi"]["truth"]) - math.log(p["lo"]["truth"])


def slope_test(pairs: list[dict], n_boot: int, seed: int, regressor: str = "sigma_star", ci_level: float = 0.95) -> dict:
    """OLS slope of y = Δ ln σ_stated on x (Δ ln σ* for A6, Δ ln truth for A6b), with a pairs bootstrap."""
    if len(pairs) < 3:
        return {"n_pairs": len(pairs), "slope": None, "p_one_sided": None, "ci": None}
    x = np.array([_x(p, regressor) for p in pairs])
    y = np.array([p["hi"]["stated_log"] - p["lo"]["stated_log"] for p in pairs])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0:
            boots.append(ols_slope(x[i], y[i]))
    boots = np.array(boots)
    tail = 100 * (1 - ci_level) / 2
    return {"n_pairs": len(pairs), "regressor": regressor, "slope": ols_slope(x, y),
            "ci95": np.percentile(boots, [2.5, 97.5]).tolist(), "ci_level": ci_level,
            "ci": np.percentile(boots, [tail, 100 - tail]).tolist(),
            "p_one_sided": float(np.mean(boots <= 0)),
            "p_two_sided": float(min(1.0, 2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))),
            "mean_dx": float(x.mean()), "mean_dy": float(y.mean()), "mean_abs_dy": float(np.abs(y).mean()),
            "share_dy_positive": float(np.mean(y > 0))}


def analyze_rows(rows: list[dict], cfg: dict, inv_cfg: dict | None = None) -> dict:
    """A6 (sensitivity pairs) or A6b (invariance pairs, needs `inv_cfg`), chosen by the rows' pair kind."""
    kinds = {r["kind"] for r in rows}
    if len(kinds) != 1:
        raise ValueError(f"mixed pair kinds in one analysis: {kinds}")
    kind = kinds.pop()
    pairs = pairs_from_rows(rows)
    both_ok = [p for p in pairs if p["lo"]["ok"] and p["hi"]["ok"]]
    primary = [p for p in both_ok if p["lo"]["computation_correct"] and p["hi"]["computation_correct"]]
    no_zero = [p for p in primary if not (p["lo"]["sigma_zero"] or p["hi"]["sigma_zero"])]
    nb, seed = cfg["bootstrap"], cfg["seed"]
    out = {"kind": kind, "n_pairs_total": len(pairs), "n_pairs_both_answered": len(both_ok)}
    if kind == "sensitivity":  # A6 (D50/D58): slope on Δ ln σ* > 0
        primary_test = slope_test(primary, nb, seed)
        out.update(primary_computation_correct=primary_test,
                   supported=primary_test["p_one_sided"] is not None and primary_test["p_one_sided"] < cfg["alpha"],
                   secondary_all_answered=slope_test(both_ok, nb, seed),
                   sensitivity_no_sigma_zero=slope_test(no_zero, nb, seed))
        return out
    # A6b (D59): slope on Δ ln truth equivalent to 0 (TOST)
    delta, lvl = inv_cfg["equivalence_margin"], inv_cfg["ci_level"]
    t = lambda ps: slope_test(ps, nb, seed, "truth", lvl)
    primary_test = t(primary)
    ci = primary_test["ci"]
    out.update(primary_computation_correct=primary_test, equivalence_margin=delta,
               equivalent_to_zero=ci is not None and -delta < ci[0] and ci[1] < delta,
               secondary_all_answered=t(both_ok), sensitivity_no_sigma_zero=t(no_zero))
    return out


def joint(a6: dict, a6b: dict) -> dict:
    """D59 two-sided propagation verdict (intersection-union: both parts at α)."""
    if a6.get("kind") != "sensitivity" or a6b.get("kind") != "invariance":
        raise ValueError("joint needs an A6 (sensitivity) and an A6b (invariance) analysis")
    if not a6["supported"]:
        verdict = "no propagation (lookup-like)"
    elif not a6b["equivalent_to_zero"]:
        verdict = "magnitude-driven"
    else:
        verdict = "propagates"
    p6, p6b = a6["primary_computation_correct"], a6b["primary_computation_correct"]
    return {"verdict": verdict, "A6_supported": a6["supported"], "A6b_equivalent_to_zero": a6b["equivalent_to_zero"],
            "table": [{"set": "A6 (sensitivity)", "ideal_slope": 1.0, "slope": p6["slope"], "ci95": p6.get("ci95"),
                       "n_pairs": p6["n_pairs"], "test": "slope > 0 (one-sided)", "p": p6["p_one_sided"]},
                      {"set": "A6b (invariance)", "ideal_slope": 0.0, "slope": p6b["slope"], "ci": p6b.get("ci"),
                       "n_pairs": p6b["n_pairs"], "test": f"TOST |slope| < {a6b['equivalence_margin']}",
                       "p_two_sided_vs_0": p6b.get("p_two_sided")}]}


def analyze(tape: Path, split: str | None = None) -> dict:
    """One pair set's analysis; `split` selects it from a tape that holds several (configs/eval_frozen_sets.yaml)."""
    pc = load_config("pairs")
    cfg = pc["analysis"]
    rows = [member_row(r["problem"], r["breakdown"], cfg["tau"], cfg["sigma_floor_rel"]) for r in read_tape(tape)
            if split is None or r.get("split") == split]
    if not rows:
        raise ValueError(f"no records for split {split!r} in {tape}")
    return {"tape": str(tape), "split": split, "tau": cfg["tau"], **analyze_rows(rows, cfg, pc["invariance_analysis"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["analyze", "joint"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--split", default=None, help="analyze: the split to read from a multi-split tape")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    paths = [Path(p) if Path(p).is_absolute() else REPO_ROOT / p for p in args.paths]
    if args.action == "analyze":
        res = analyze(paths[0], args.split)
        name = f"{'a6' if res['kind'] == 'sensitivity' else 'a6b'}_pairs_analysis.json"
        out = Path(args.out) if args.out else paths[0].parent
    else:
        res = joint(json.loads(paths[0].read_text()), json.loads(paths[1].read_text()))
        name, out = "propagation_joint.json", Path(args.out) if args.out else paths[0].parent
    (out / name).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

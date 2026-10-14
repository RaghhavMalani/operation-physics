"""D50/A6 primary analysis: does a model's stated σ track the within-pair change in σ*? (propagation vs lookup)

    uv run python -m physcal.eval.a6_pairs <eval tape over the A6 pair set> [--out DIR]

Per pair (same template, depth, phrasing and noise level; worlds chosen so that σ* differs, physcal/env/pairs.py):
x = ln σ*_hi − ln σ*_lo, y = ln σ_hi − ln σ_lo, with σ the stated relative σ (σ/|μ|, crps_log's scale; σ = 0 → the
crps_log floor, the PREREGISTRATION convention). Primary: pairs where both answers are computation-correct (A1:
|ln μ − ln f(x_meas)| ≤ τ); OLS slope of y on x with intercept; one-sided bootstrap over pairs for slope > 0.
A surface-feature lookup predicts slope ≈ 0, honest propagation slope ≈ 1. Secondary: all pairs with both answers
parsed; sensitivity: pairs without a stated σ = 0.
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
           "member": n["pair"]["member"], "sigma_star_log": n["sigma_star_log"], "ok": ok,
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


def slope_test(pairs: list[dict], n_boot: int, seed: int) -> dict:
    if len(pairs) < 3:
        return {"n_pairs": len(pairs), "slope": None, "p_one_sided": None}
    x = np.array([math.log(p["hi"]["sigma_star_log"]) - math.log(p["lo"]["sigma_star_log"]) for p in pairs])
    y = np.array([p["hi"]["stated_log"] - p["lo"]["stated_log"] for p in pairs])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0:
            boots.append(ols_slope(x[i], y[i]))
    boots = np.array(boots)
    return {"n_pairs": len(pairs), "slope": ols_slope(x, y), "ci95": np.percentile(boots, [2.5, 97.5]).tolist(),
            "p_one_sided": float(np.mean(boots <= 0)), "mean_dx": float(x.mean()), "mean_dy": float(y.mean()),
            "share_dy_positive": float(np.mean(y > 0))}


def analyze_rows(rows: list[dict], cfg: dict) -> dict:
    pairs = pairs_from_rows(rows)
    both_ok = [p for p in pairs if p["lo"]["ok"] and p["hi"]["ok"]]
    primary = [p for p in both_ok if p["lo"]["computation_correct"] and p["hi"]["computation_correct"]]
    no_zero = [p for p in primary if not (p["lo"]["sigma_zero"] or p["hi"]["sigma_zero"])]
    nb, seed = cfg["bootstrap"], cfg["seed"]
    primary_test = slope_test(primary, nb, seed)
    return {"n_pairs_total": len(pairs), "n_pairs_both_answered": len(both_ok),
            "primary_computation_correct": primary_test,
            "supported": primary_test["p_one_sided"] is not None and primary_test["p_one_sided"] < cfg["alpha"],
            "secondary_all_answered": slope_test(both_ok, nb, seed),
            "sensitivity_no_sigma_zero": slope_test(no_zero, nb, seed)}


def analyze(tape: Path) -> dict:
    cfg = load_config("pairs")["analysis"]
    rows = [member_row(r["problem"], r["breakdown"], cfg["tau"], cfg["sigma_floor_rel"]) for r in read_tape(tape)]
    return {"tape": str(tape), "tau": cfg["tau"], **analyze_rows(rows, cfg)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tape")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tape = Path(args.tape) if Path(args.tape).is_absolute() else REPO_ROOT / args.tape
    res = analyze(tape)
    out = Path(args.out) if args.out else tape.parent
    (out / "a6_pairs_analysis.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

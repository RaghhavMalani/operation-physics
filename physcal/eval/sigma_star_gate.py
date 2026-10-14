"""Phase 10 step 4: the generator's own σ* must cover the clean truth at nominal rates (gate as amended, D50/A3).

    nice uv run python -m physcal.eval.sigma_star_gate

10,000 noisy problems (depth 1–5 × noise 1, 2, 5, 10%; 500 per cell, families round-robin, every cell with its
own seeds). Interval: ln f(x_meas) ± k·σ* (σ* = CRPS-optimal log-scale, A2), k = 1, 2. The gate passes iff
1. pooled Kupiec p > α at both levels;
2. per-cell Kupiec, Holm across the cells (per level): no rejection;
3. per level, ≥ 95% of cells have coverage inside the central 99% binomial interval around nominal;
4. pooled PIT KS (truth under the MC predictive distribution): p > α.
The superseded rule ("Kupiec p > 0.05 in every cell") is evaluated on the same run and reported, not used.
Noise-0 problems are checked for exactness (truth = f(x_meas), σ* = 0). Writes <output_dir>/gate.{json,md}.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import kstest, norm

from physcal.config import REPO_ROOT, load_config
from physcal.env.generator import generate_problem, generator_version
from physcal.eval.stats import binomial_interval, holm, kupiec


def _row(args: tuple) -> dict:
    family, depth, seed, level, env_name = args
    try:
        p = generate_problem(family, depth, seed, split="eval", cfg=load_config(env_name), noise=level)
    except RuntimeError as exc:
        return {"family": family, "depth": depth, "seed": seed, "level": level, "error": str(exc)}
    n = p.noise
    return {"family": family, "depth": depth, "seed": seed, "level": level,
            "dev": math.log(p.target_value_si) - math.log(n["point_estimate_si"]),
            "sigma": n["sigma_star_log"], "sigma_sd": n["sigma_star_log_sd"], "sigma_delta": n["sigma_star_log_delta"],
            "pit": n["pit_truth"], "exact": p.target_value_si == n["point_estimate_si"],
            "condition_number": p.condition_number}


def jobs(cfg: dict) -> tuple[list[tuple], list[tuple]]:
    fams, noisy, zero = cfg["families"], [], []
    cells = [(d, lv) for d in cfg["depths"] for lv in cfg["levels"]]
    for c, (d, lv) in enumerate(cells):
        base = cfg["seed_offset"] + c * cfg["cell_seed_stride"]
        noisy += [(fams[i % len(fams)], d, base + i, lv, cfg["env_config"]) for i in range(cfg["n_per_cell"])]
    base = cfg["seed_offset"] + len(cells) * cfg["cell_seed_stride"]
    zero = [(fams[i % len(fams)], cfg["depths"][(i // len(fams)) % len(cfg["depths"])], base + i, 0.0, cfg["env_config"])
            for i in range(cfg["noise_zero_n"])]
    return noisy, zero


def evaluate(rows: list[dict], cfg: dict, sigma_key: str = "sigma") -> dict:
    """Coverage tests on noisy rows for one σ* definition."""
    alpha, out = cfg["alpha"], {"levels": {}}
    cells = sorted({(r["depth"], r["level"]) for r in rows})
    for k in cfg["interval_k"]:
        nominal = float(norm.cdf(k) - norm.cdf(-k))
        cov = [abs(r["dev"]) <= k * r[sigma_key] for r in rows]
        pooled = kupiec(int(sum(cov)), len(cov), nominal)
        per_cell = {}
        for d, lv in cells:
            c = [abs(r["dev"]) <= k * r[sigma_key] for r in rows if (r["depth"], r["level"]) == (d, lv)]
            t = kupiec(int(sum(c)), len(c), nominal)
            lo, hi = binomial_interval(len(c), nominal, cfg["binomial_interval"])
            per_cell[f"d{d}/s{lv:g}"] = {**t, "binomial_interval": [lo, hi], "inside": lo <= t["coverage"] <= hi}
        h = holm({name: v["p"] for name, v in per_cell.items()}, alpha)
        share_inside = float(np.mean([v["inside"] for v in per_cell.values()]))
        out["levels"][str(k)] = {
            "nominal": nominal, "pooled": pooled, "cells": per_cell, "holm": h,
            "pooled_ok": pooled["p"] > alpha,
            "holm_ok": not any(v["rejected"] for v in h.values()),
            "share_cells_inside": share_inside,
            "share_ok": share_inside >= cfg["min_share_cells_inside"],
            "superseded_every_cell_ok": all(v["p"] > alpha for v in per_cell.values()),
        }
    return out


def main() -> None:
    cfg = load_config("sigma_star_gate")
    env = load_config(cfg["env_config"])
    noisy_jobs, zero_jobs = jobs(cfg)
    with ProcessPoolExecutor(cfg["n_jobs"]) as ex:
        rows = list(ex.map(_row, noisy_jobs, chunksize=50))
        zero = list(ex.map(_row, zero_jobs, chunksize=50))
    errors = [r for r in rows + zero if "error" in r]
    noisy = [r for r in rows if "error" not in r]
    zero = [r for r in zero if "error" not in r]
    alpha = cfg["alpha"]
    pits = np.array([r["pit"] for r in noisy])
    ks = kstest(pits, "uniform")
    pit_by_level = {f"{lv:g}": float(kstest([r["pit"] for r in noisy if r["level"] == lv], "uniform").pvalue)
                    for lv in cfg["levels"]}
    primary = evaluate(noisy, cfg, "sigma")
    a3 = {f"k{k}_{name}": primary["levels"][str(k)][f"{name}_ok"] for k in cfg["interval_k"]
          for name in ("pooled", "holm", "share")}
    a3["pit_ks"] = bool(ks.pvalue > alpha)
    ratio = lambda lv, a, b: np.array([r[a] / r[b] for r in noisy if r["level"] == lv])
    result = {
        "generator_version": generator_version(env),
        "config": cfg,
        "n_noisy": len(noisy), "n_errors": len(errors), "errors": errors[:20],
        "gate_A3": {"criteria": a3, "passed": all(a3.values())},
        "superseded_rule_every_cell_kupiec": {
            "passed": all(primary["levels"][str(k)]["superseded_every_cell_ok"] for k in cfg["interval_k"]),
            "note": "D50/A3 replaced this rule before the gate ran; reported for transparency only"},
        "pit": {"ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue), "ks_p_by_level": pit_by_level,
                "histogram": np.histogram(pits, bins=cfg["pit_bins"], range=(0, 1))[0].tolist()},
        "primary_sigma_crps": primary,
        "secondary_sigma_sd": evaluate(noisy, cfg, "sigma_sd"),
        "mc_vs_delta": {f"{lv:g}": {"median_sd_over_delta": float(np.median(ratio(lv, "sigma_sd", "sigma_delta"))),
                                    "max_abs_dev": float(np.max(np.abs(ratio(lv, "sigma_sd", "sigma_delta") - 1))),
                                    "median_crps_over_sd": float(np.median(ratio(lv, "sigma", "sigma_sd")))}
                        for lv in cfg["levels"]},
        "median_sigma_star_log": {f"{lv:g}": float(np.median([r["sigma"] for r in noisy if r["level"] == lv]))
                                  for lv in cfg["levels"]},
        "noise_zero": {"n": len(zero), "all_exact": all(r["exact"] for r in zero),
                       "max_sigma": max(r["sigma"] for r in zero)},
    }
    out = REPO_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps(result, indent=1))
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in noisy + zero))
    (out / "gate.md").write_text(render(result, cfg))
    print(render(result, cfg))


def render(res: dict, cfg: dict) -> str:
    g = res["gate_A3"]
    lines = [f"# Phase 10 generator self-consistency gate ({res['generator_version']})", "",
             f"**Gate (D50/A3): {'PASSED' if g['passed'] else 'FAILED'}**; superseded every-cell rule: "
             f"{'passed' if res['superseded_rule_every_cell_kupiec']['passed'] else 'failed'} (not used).", "",
             f"{res['n_noisy']} noisy problems, {res['n_errors']} generation errors; noise 0: {res['noise_zero']['n']} "
             f"problems, all exact: {res['noise_zero']['all_exact']}.", "",
             "| criterion | ok |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in g["criteria"].items()]
    lines += ["", f"PIT KS (pooled): D = {res['pit']['ks_stat']:.4f}, p = {res['pit']['ks_p']:.3g}; by level: "
              + ", ".join(f"{k}: p = {v:.3g}" for k, v in res["pit"]["ks_p_by_level"].items()),
              f"PIT histogram ({cfg['pit_bins']} bins): {res['pit']['histogram']}", ""]
    for k, lv in res["primary_sigma_crps"]["levels"].items():
        lines += [f"## ±{k}σ* (nominal {lv['nominal']:.2%}): pooled {lv['pooled']['coverage']:.2%}, "
                  f"Kupiec p = {lv['pooled']['p']:.3g}; cells inside the {cfg['binomial_interval']:.0%} binomial "
                  f"interval: {lv['share_cells_inside']:.0%}", "",
                  "| cell | n | coverage | Kupiec p | Holm rejects | inside |", "|---|---|---|---|---|---|"]
        lines += [f"| {c} | {v['n']} | {v['coverage']:.1%} | {v['p']:.3g} | {lv['holm'][c]['rejected']} | {v['inside']} |"
                  for c, v in lv["cells"].items()]
        lines.append("")
    lines += ["## σ* checks", "", "| level | median σ* (log) | median SD/delta | max |SD/delta − 1| | median CRPS-opt/SD |",
              "|---|---|---|---|---|"]
    lines += [f"| {lv} | {res['median_sigma_star_log'][lv]:.4f} | {v['median_sd_over_delta']:.4f} | {v['max_abs_dev']:.3f} "
              f"| {v['median_crps_over_sd']:.4f} |" for lv, v in res["mc_vs_delta"].items()]
    sec = res["secondary_sigma_sd"]["levels"]
    lines += ["", "Secondary (σ* = SD of the log draws): pooled coverage "
              + ", ".join(f"±{k}: {v['pooled']['coverage']:.2%} (p = {v['pooled']['p']:.3g})" for k, v in sec.items())]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

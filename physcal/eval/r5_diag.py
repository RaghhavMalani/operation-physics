"""Round-5 diagnostics (DECISIONS.md D133-): descriptive, diagnostic and post-hoc analyses of logs that already exist.
CPU only; nothing here is confirmatory, and every output is labelled with its kind.

    OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.eval.r5_diag depth5

`depth5`: why the same seed-0 D-pos-v2 adapter has an A6c depth-5 slope of +0.06 on the development sets (FINDINGS
#30) and +0.80 on the frozen R4-5 set (#32). Both slopes come from #30's estimator (`r4_analysis.slope`: OLS of
Δ ln σ_stated on Δ ln σ* over the pairs whose members are both answered, pairs bootstrap); the rest describes the cells.
Reading the frozen set needs PHYSCAL_CONFIRMATORY=1 (D118's analysis has run; this reads, it does not re-test).
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy import stats

from physcal.config import REPO_ROOT, load_config
from physcal.eval import a6_pairs
from physcal.eval import r4_analysis as R
from physcal.eval import r4_confirm as RC
from physcal.eval import stage_a
from physcal.eval import stage_b_analysis as B

KINDS = ("development", "frozen")


def _config() -> dict:
    return load_config("r5_diag")


# ------------------------------------------------------------------------------------------- inputs
class Ctx:
    """One set family (development or frozen) with stage R4's configs, its problems and the builder's reference lines."""

    def __init__(self, kind: str, cfg: dict | None = None):
        self.cfg = cfg or _config()
        self.r4 = load_config(self.cfg["stage_r4_config"])
        self.ac = self.r4["analysis"]
        bcfg = load_config(self.ac["stage_b_config"])["analysis"]
        self.pc = load_config(bcfg["pairs_config"])
        self.dcfg = load_config(self.r4["sft_phase12_config"])
        if kind == "frozen":
            RC._guard()
            self.sa, self.subdir = RC.stage_a_config(self.r4), self.r4["confirmatory"]["output_subdir"]
        else:
            self.sa, self.subdir = load_config(bcfg["stage_a_config"]), "eval"
        self.kind = kind
        self.problems = stage_a.load_sets(self.sa)
        self.refs = R.references(self.problems, self.dcfg)
        self.types = R.pair_types(self.problems, self.refs)

    def records(self, arm: str, seed: int) -> list[dict]:
        recs = B.read_records(REPO_ROOT / self.r4["output_dir"] / arm / f"seed{seed}" / self.subdir)
        if recs is None:
            raise FileNotFoundError(f"no {self.kind} eval for {arm} seed {seed}")
        return recs


# ------------------------------------------------------------------------------------------- depth 5
def pair_xy(pairs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([a6_pairs._x(p, "sigma_star") for p in pairs])
    y = np.array([p["hi"]["stated_log"] - p["lo"]["stated_log"] for p in pairs])
    return x, y


def influence(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out change of the OLS slope per pair (full slope minus the slope without the pair)."""
    full = a6_pairs.ols_slope(x, y)
    keep = ~np.eye(len(x), dtype=bool)
    return np.array([full - a6_pairs.ols_slope(x[k], y[k]) for k in keep])


def describe_cell(all_pairs: list[dict], used: list[dict], types: dict, ac: dict, dc: dict) -> dict:
    """One depth's A6c cell: #30's slope and CI, n usable, the regressor's spread, gross residuals, the leave-one-out
    influence and the slope without the most influential pairs, Theil-Sen, and the slope by chain type."""
    x, y = pair_xy(used)
    sl = R.slope(used, ac)
    out = {"n_pairs": len(all_pairs), "n_used": len(used),
           "n_unanswered_members": sum((not p["lo"]["ok"]) + (not p["hi"]["ok"]) for p in all_pairs),
           "slope": sl["slope"], "ci90": sl["ci"], "ci95": sl["ci95"], "p_one_sided": sl["p_one_sided"],
           "x_sd": float(np.std(x, ddof=1)), "x_range": [float(x.min()), float(x.max())],
           "y_sd": float(np.std(y, ddof=1)), "corr_xy": float(np.corrcoef(x, y)[0, 1])}
    gross = np.abs(y - x) > math.log(dc["gross_factor"])
    out["n_gross_residual"] = int(gross.sum())
    out["slope_without_gross"] = float(a6_pairs.ols_slope(x[~gross], y[~gross])) if (~gross).sum() > 2 else None
    inf = influence(x, y)
    order = np.argsort(-np.abs(inf))
    out["top_influence"] = [{"pair": list(used[i]["lo"]["pair"]), "x": float(x[i]), "y": float(y[i]),
                             "loo_change": float(inf[i]), "type": types.get(used[i]["lo"]["pair"])}
                            for i in order[:dc["influence_top"]]]
    for k in (1, 3, dc["influence_top"]):
        keep = np.ones(len(x), dtype=bool)
        keep[order[:k]] = False
        out[f"slope_without_top{k}"] = float(a6_pairs.ols_slope(x[keep], y[keep]))
    ts = stats.theilslopes(y, x, alpha=dc["theil_sen_ci"])
    out["theil_sen"] = {"slope": float(ts.slope), "ci": [float(ts.low_slope), float(ts.high_slope)]}
    out["n_reversed"] = int((y < 0).sum())  # x > 0 by construction (the hi member's σ* is ≥ 1.5× the lo member's)
    out["by_level"] = {}
    for lv in sorted({p["lo"]["pair"][2] for p in used}):
        k = np.array([p["lo"]["pair"][2] == lv for p in used])
        out["by_level"][f"{lv:g}"] = {"n_used": int(k.sum()), "mean_x": float(x[k].mean()), "mean_y": float(y[k].mean()), "n_reversed": int((y[k] < 0).sum()),
                                      "slope": float(a6_pairs.ols_slope(x[k], y[k])) if k.sum() > 2 else None}
    out["by_type"] = {}
    for t in ("power_law", "value_dependent"):
        sub = [p for p in used if types.get(p["lo"]["pair"]) == t]
        out["by_type"][t] = {"n_used": len(sub), "n_pairs": sum(types.get(p["lo"]["pair"]) == t for p in all_pairs),
                             "slope": R.slope(sub, ac)["slope"] if len(sub) >= 3 else None}
    return out


def depth5(cfg: dict | None = None) -> dict:
    cfg = cfg or _config()
    dc = cfg["depth5"]
    res: dict = {"kind": "descriptive (D133, item 2); not a test", "arm": "dpos2", "seed": dc["seed"],
                 "sigma": "(a), the answer's σ", "set": "a6c_pairs", "estimator": "r4_analysis.slope (FINDINGS #30)"}
    for kind in KINDS:
        ctx = Ctx(kind, cfg)
        _, bds = B.model_sigma(ctx.problems, ctx.records("dpos2", dc["seed"]))
        rows = B.model_rows(ctx.problems, bds, ctx.pc)
        all_pairs = a6_pairs.pairs_from_rows(rows["a6c_pairs"])
        used = R.model_pairs(ctx.problems, bds, ctx.pc)["a6c"]
        res[kind] = {"by_depth": {}}
        for d in sorted({p["lo"]["pair"][1] for p in all_pairs}):
            cell = describe_cell([p for p in all_pairs if p["lo"]["pair"][1] == d],
                                 [p for p in used if p["lo"]["pair"][1] == d], ctx.types, ctx.ac, dc)
            res[kind]["by_depth"][str(d)] = cell
        manifest = json.loads((REPO_ROOT / "data" / "eval_sets" / ctx.sa["generator_version"] /
                               "a6c_pairs_manifest.json").read_text())
        res[kind]["construction"] = {"construction": manifest["construction"], "export": manifest["export"],
                                     "depth5_cells": {k: v for k, v in manifest["cells"].items() if k.startswith("d5")}}
    return res


def render_depth5(res: dict) -> str:
    f = lambda v: "–" if v is None else f"{v:+.3f}"
    L = ["# The depth-5 discrepancy: D-pos-v2 seed 0, A6c, development vs frozen (descriptive; D133 item 2)", "",
         f"Estimator: {res['estimator']}; σ {res['sigma']}. Same adapter, same code; only the problem set differs.", "",
         "| depth | set | pairs used / all | slope [90% CI] | Theil-Sen [90% CI] | SD Δln σ* | corr | gross (>10×) | "
         "slope w/o gross | w/o top 1 / 3 / 5 influential | power-law n, slope | value-dep. n, slope |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in res["development"]["by_depth"]:
        for kind in KINDS:
            c = res[kind]["by_depth"][d]
            pl, vd = c["by_type"]["power_law"], c["by_type"]["value_dependent"]
            L.append(f"| {d} | {kind} | {c['n_used']} / {c['n_pairs']} | {f(c['slope'])} [{f(c['ci90'][0])}, "
                     f"{f(c['ci90'][1])}] | {f(c['theil_sen']['slope'])} [{f(c['theil_sen']['ci'][0])}, "
                     f"{f(c['theil_sen']['ci'][1])}] | {c['x_sd']:.3f} | {c['corr_xy']:+.2f} | {c['n_gross_residual']} | "
                     f"{f(c['slope_without_gross'])} | {f(c['slope_without_top1'])} / {f(c['slope_without_top3'])} / "
                     f"{f(c['slope_without_top5'])} | {pl['n_used']}, {f(pl['slope'])} | {vd['n_used']}, {f(vd['slope'])} |")
    L += ["", "## By level cell (the pair's larger level; x = Δ ln σ* > 0 by construction, so y < 0 is a reversed "
          "direction)", "", "| depth | set | level | n | mean Δ ln σ* | mean Δ ln σ_stated | reversed (y < 0) | within-cell slope |", "|---|---|---|---|---|---|---|---|"]
    for d in res["development"]["by_depth"]:
        for kind in KINDS:
            for lv, c in res[kind]["by_depth"][d]["by_level"].items():
                L.append(f"| {d} | {kind} | {lv} | {c['n_used']} | {c['mean_x']:+.3f} | {c['mean_y']:+.3f} | {c['n_reversed']} | {f(c['slope'])} |")
    L += ["", "## The most influential depth-5 pairs (leave-one-out change of the slope)", ""]
    for kind in KINDS:
        L.append(f"**{kind}:**")
        for t in res[kind]["by_depth"]["5"]["top_influence"]:
            L.append(f"- {t['pair']} ({t['type']}): Δ ln σ* {t['x']:+.3f}, Δ ln σ_stated {t['y']:+.3f}, "
                     f"LOO change {t['loo_change']:+.3f}")
        L.append("")
    return "\n".join(L) + "\n"


def _write(name: str, res: dict, md: str, cfg: dict) -> None:
    out = REPO_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(res, indent=1, default=B._json_default))
    (out / f"{name}.md").write_text(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["depth5"])
    a = ap.parse_args()
    cfg = _config()
    if a.what == "depth5":
        res = depth5(cfg)
        _write("depth5", res, render_depth5(res), cfg)
        print(render_depth5(res))


if __name__ == "__main__":
    main()

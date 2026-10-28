"""Round-5 diagnostics (DECISIONS.md D133-): descriptive, diagnostic and post-hoc analyses of logs that already exist.
CPU only; nothing here is confirmatory, and every output is labelled with its kind.

    OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.eval.r5_diag depth5

`depth5`: why the same seed-0 D-pos-v2 adapter has an A6c depth-5 slope of +0.06 on the development sets (FINDINGS
#30) and +0.80 on the frozen R4-5 set (#32). Both slopes come from #30's estimator (`r4_analysis.slope`: OLS of
Δ ln σ_stated on Δ ln σ* over the pairs whose members are both answered, pairs bootstrap); the rest describes the cells.
Reading the frozen set needs PHYSCAL_CONFIRMATORY=1 (D118's analysis has run; this reads, it does not re-test).

`tail` (R5-2, descriptive; R5-3, diagnostic): on the frozen D-pos-v2 logs, a decomposable problem's stated σ (a)
factors exactly as  σ_a = copy · aggregation · rss(model sensitivities, model levels), and
rss(model S, model L) = [sensitivity factor] · [level factor] · rss(builder S, builder L). R5-2 names the largest
|log factor| (or the first failure: format, chain selection) on each seed's worst 5% of problems by regret against σ*;
R5-3 sets one factor to its exact value at a time and scores the result (never a deployable method).
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from collections import Counter

import numpy as np
from scipy import stats

from physcal.config import REPO_ROOT, load_config
from physcal.eval import a6_pairs
from physcal.eval import oracle_centred as oc
from physcal.eval import r4_analysis as R
from physcal.eval import r4_confirm as RC
from physcal.eval import stage_a
from physcal.eval import stage_b_analysis as B
from physcal.eval import stage_d_analysis as Dn
from physcal.verify.reward import reward_config

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


# ------------------------------------------------------------------------------------------- the tail (R5-2, R5-3)
FUNCS = {"sqrt": math.sqrt, "exp": math.exp, "log": math.log, "sin": math.sin, "cos": math.cos, "tan": math.tan,
         "Abs": abs}
OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
       ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b}
VARIANTS = ("a", "S", "L", "A", "C", "AC", "exact")
VARIANT_NAMES = {"a": "(a) as stated", "S": "exact sensitivities (model's levels, aggregation, copy)",
                 "L": "exact levels (model's sensitivities, aggregation, copy)",
                 "A": "exact aggregation (model's terms and copy)", "C": "exact copy, i.e. σ = the line's X (= (b))",
                 "AC": "model's terms, exact aggregation and copy (= (c))",
                 "exact": "fully exact: the builder's line, unrounded (ceiling)"}


def safe_eval(text: str, names: dict, max_pow: float) -> float | None:
    """A model-written arithmetic expression: numbers, names from `names`, + - * / ** (^), unary ±, and FUNCS. No
    Python eval; anything else, a complex or non-finite value, or an exponent above `max_pow` gives None."""
    try:
        node = ast.parse(text.replace("^", "**"), mode="eval").body
    except (SyntaxError, ValueError):
        return None

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return float(n.value)
        if isinstance(n, ast.Name) and n.id in names:
            return float(names[n.id])
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > max_pow:
                raise ValueError("exponent")
            return OPS[type(n.op)](a, b)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS and len(n.args) == 1 \
                and not n.keywords:
            return FUNCS[n.func.id](ev(n.args[0]))
        raise ValueError("node")

    try:
        v = ev(node)
        return float(v) if isinstance(v, float) and math.isfinite(v) else None
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None


def s_line_parts(completion: str) -> dict[str, list[str]]:
    """The model's S lines before <answer>: symbol -> the right-hand side split at '=' (the last line wins)."""
    head = completion.split("<answer>", 1)[0]
    return {sym: [x.strip() for x in rhs.split("=")] for sym, rhs in R.S_LINE_RE.findall(head)}


def close(a: float | None, b: float, tol: float) -> bool:
    return a is not None and (abs(a - b) <= tol * abs(b) if b else a == b)


def rss(terms) -> float:
    return math.sqrt(sum((a * s) ** 2 for a, s in terms))


def term_cause(t: dict, a_line: float, parts: list[str] | None, vals: dict, cc: bool, tol: float, max_pow: float) -> str:
    """Why one sensitivity in the σ line is wrong (outside `tol` of the builder's): the S line behind it."""
    if parts is None:
        return "no S line"
    if close(safe_eval(parts[-1], {}, max_pow), t["value"], tol):
        return "transcription (S line right, σ-line coefficient wrong)"
    if t["power_law"]:
        return "power-law exponent"
    if len(parts) < 3:
        return "value-dependent: formula (no expression)"
    if not close(safe_eval(parts[0], vals, max_pow), t["value"], tol):
        return "value-dependent: formula"
    n = safe_eval(parts[-2], {}, max_pow)
    if n is None:
        return "value-dependent: substitution (unparsed numbers)"
    if not close(n, t["value"], tol):
        return "value-dependent: substitution (" + ("own wrong intermediates" if not cc else "values") + ")"
    return "value-dependent: S-line arithmetic"


def decompose(p: dict, ref: dict, rec: dict | None, vals: dict, tol: float, tau: float, max_pow: float,
              material: float = 1.25) -> dict:
    """One problem: the cause class, the log factors, and σ under each R5-3 variant (problems that do not decompose
    keep (a) except under 'exact')."""
    comp = rec["completion"] if rec else ""
    b = rec["breakdown"] if rec else {"reason": "missing"}
    ln = Dn.parse_line(comp)
    answered = b["reason"] == "ok" and b.get("sigma_si") is not None and (b.get("mu_si") or 0) > 0
    sig = b["sigma_si"] / b["mu_si"] if answered else 0.0
    cc = answered and abs(math.log(b["mu_si"] / p["noise"]["point_estimate_si"])) <= tau
    rt, exact = ref["terms"], rss(ref["terms"])
    out = {"depth": p["depth"], "set": p["_set"], "computation_correct": cc, "decomposable": False,
           "sigma": {v: sig for v in VARIANTS}}
    out["sigma"]["exact"] = exact
    if not ln["present"] or not ln["parsed"]:
        out["cause"] = "format: no parsable σ line"
    elif ln["zero"]:
        out["cause"] = "format: σ line = 0 on a noisy problem"
    elif not answered:
        out["cause"] = "format: no answer σ"
    elif len(ln["terms"]) != len(rt):
        out["cause"] = "chain selection: wrong number of terms"
    elif ln["x"] <= 0 or rss(ln["terms"]) <= 0:
        out["cause"] = "format: X or the terms are 0"
    if "cause" in out:
        return out
    mt, x = ln["terms"], ln["x"]
    own = rss(mt)
    own_s = rss([(ra, s) for (_, s), (ra, _) in zip(mt, rt)])
    own_l = rss([(a, rs) for (a, _), (_, rs) in zip(mt, rt)])
    agg, cp = x / own, sig / x
    out["decomposable"] = True
    out["sigma"].update(S=cp * agg * own_s, L=cp * agg * own_l, A=cp * own, C=x, AC=own)
    if sig == 0:
        out["cause"] = "copy: σ = 0 while X > 0"
        return out
    lf = {"sensitivity": math.log(own / own_s) if own_s > 0 else math.inf,
          "level": math.log(own_s / exact) if own_s > 0 else -math.inf, "aggregation": math.log(agg),
          "copy": math.log(cp)}
    out["log_factors"] = lf
    out["log_first_order"] = math.log(exact / p["noise"]["sigma_star_log"])  # the builder's line vs σ*: no model error
    main = max(lf, key=lambda k: abs(lf[k]))
    out["cause"] = main if abs(lf[main]) > math.log(material) else "no material component error (rounding, first order)"
    if out["cause"] == "copy":  # σ = X·|μ| written a power of ten off?
        k = round(lf["copy"] / math.log(10))
        out["cause"] = (f"copy: decimal shift (×10^{k})" if k and abs(lf["copy"] - k * math.log(10)) < math.log(material)
                        else "copy: other")
    if main == "sensitivity":
        syms, parts = list(ref["sens"]), s_line_parts(comp)
        wrong = [j for j, ((a, _), (ra, _)) in enumerate(zip(mt, rt)) if not close(a, ra, tol)]
        if wrong:
            eff = {j: abs(math.log(own / max(rss([(ra if k == j else a, s) for k, ((a, s), (ra, _)) in
                                                  enumerate(zip(mt, rt))]), 1e-300))) for j in wrong}
            j = max(eff, key=eff.get)
            t = ref["sens"][syms[j]]
            out["cause"] = "sensitivity: " + term_cause(t, mt[j][0], parts.get(syms[j]), vals, cc, tol, max_pow)
        else:
            out["cause"] = "sensitivity: within tolerance on every term (rounding)"
    return out


def compensation(dec: list[dict], material: float) -> dict:
    """On decomposable problems whose sensitivity factor is material: how often the model's own aggregation and copy
    move σ the other way by a material amount, and the median share of the sensitivity error they undo."""
    m = math.log(material)
    rows = [d["log_factors"] for d in dec if "log_factors" in d and math.isfinite(d["log_factors"]["sensitivity"])
            and abs(d["log_factors"]["sensitivity"]) > m]
    if not rows:
        return {"n": 0}
    down = [f["aggregation"] + f["copy"] for f in rows]
    undo = [-dn / f["sensitivity"] for f, dn in zip(rows, down)]
    return {"n": len(rows), "share_offsetting": float(np.mean([dn * f["sensitivity"] < 0 and abs(dn) > m
                                                                for f, dn in zip(rows, down)])),
            "median_share_undone": float(np.median(undo))}


def worst(regret: np.ndarray, share: float) -> np.ndarray:
    ok = np.flatnonzero(np.isfinite(regret))
    n = max(1, int(round(share * len(ok))))
    return ok[np.argsort(-regret[ok])[:n]]


def trimmed(v: np.ndarray, share: float) -> float:
    v = np.sort(v[np.isfinite(v)])
    return float(v[int(round(share * len(v))):].mean())


def tail(cfg: dict | None = None) -> dict:
    from physcal.env.generator import problem_from_dict
    from physcal.train.sft_r4 import solution_values

    cfg = cfg or _config()
    tc = cfg["tail"]
    ctx = Ctx("frozen", cfg)
    ac, sa, pc = ctx.ac, ctx.sa, ctx.pc
    floor, tau, tol = pc["analysis"]["sigma_floor_rel"], pc["analysis"]["tau"], ac["sensitivity_rel_tol"]
    sc = reward_config()["scoring"]
    probs = ctx.problems
    terms = [oc.problem_terms(p, sa["mc"]["n_draws"], sa["mc"]["seed_tag"]) for p in probs]
    has = np.array([t["draws"] is not None for t in terms])
    vals = [solution_values(problem_from_dict({k: v for k, v in p.items() if k != "_set"})) for p in probs]
    src, _ = stage_a.sources(probs, sa)
    bcfg = load_config(ac["stage_b_config"])["analysis"]
    comp = {"sigma_star": src[stage_a.STAR], "rss": src[bcfg["rss_source"]], "x1": src[bcfg["x1_source"]]}
    cexp = {n: oc.score_source(terms, s, sc)["expected"] for n, s in comp.items()}
    boot = oc.ClusterBootstrap([(p["_set"], p["noise"]["pair"]["pair_seed"]) for p in probs], ac["bootstrap"], ac["seed"])
    depth = np.array([p["depth"] for p in probs])
    per_seed, exp, rows = {}, {v: {} for v in VARIANTS}, {v: {} for v in VARIANTS}
    for s in tc["seeds"]:
        recs = ctx.records(tc["arm"], s)
        by = {(r["split"], *B._key(r["problem"])): r for r in recs}
        dec = [decompose(p, ref, by.get((p["_set"], *B._key(p))), v, tol, tau, tc["max_pow"], tc["material_factor"])
               for p, ref, v in zip(probs, ctx.refs, vals)]
        for v in VARIANTS:
            sig = np.array([d["sigma"][v] for d in dec])
            exp[v][s] = oc.score_source(terms, sig, sc)["expected"]
            rows[v][s] = B.source_rows(probs, sig, floor)
        regret = cexp["sigma_star"] - exp["a"][s]
        w = worst(regret, tc["worst_share"])
        per_seed[s] = {"n_worst": len(w), "share_decomposable": float(np.mean([d["decomposable"] for d in dec])),
                       "causes_worst": Counter(dec[i]["cause"] for i in w),
                       "causes_worst_by_depth": {str(k): Counter(dec[i]["cause"] for i in w if dec[i]["depth"] == k)
                                                 for k in sorted(set(depth))},
                       "worst_by_depth": Counter(int(depth[i]) for i in w),
                       "worst_regret_share": float(regret[w].sum() / regret[np.isfinite(regret)].sum()),
                       "causes_all": Counter(d["cause"] for d in dec),
                       "worst_computation_correct": float(np.mean([dec[i]["computation_correct"] for i in w])),
                       "all_computation_correct": float(np.mean([d["computation_correct"] for d in dec])),
                       "compensation": compensation(dec, tc["material_factor"])}
    pooled = sum((per_seed[s]["causes_worst"] for s in tc["seeds"]), Counter())
    pooled_depth = {k: sum((per_seed[s]["causes_worst_by_depth"][k] for s in tc["seeds"]), Counter())
                    for k in per_seed[tc["seeds"][0]]["causes_worst_by_depth"]}
    gap = float(np.mean(cexp["sigma_star"][has]) - np.mean(cexp["rss"][has]))
    swaps = {}
    for v in VARIANTS:
        m = float(np.mean([np.mean(exp[v][s][has]) for s in tc["seeds"]]))
        swaps[v] = {"name": VARIANT_NAMES[v], "expected": m,
                    "minus_a": RC.seed_mean_diff(boot, exp[v], exp["a"], has) if v != "a" else None,
                    "minus_rss": RC.seed_mean_diff(boot, exp[v], cexp["rss"], has),
                    "minus_x1": RC.seed_mean_diff(boot, exp[v], cexp["x1"], has),
                    "share_of_rss_to_star_gap": (m - float(np.mean(cexp["rss"][has]))) / gap,
                    "a6c_slope": RC.seed_mean_slope(rows[v], "a6c_pairs", "ok", ac),
                    "expected_by_depth": {str(k): float(np.mean([np.mean(exp[v][s][has & (depth == k)])
                                                                 for s in tc["seeds"]])) for k in sorted(set(depth))}}
    trim = {"dpos2_a": {"full": swaps["a"]["expected"],
                        "trimmed": float(np.mean([trimmed(exp["a"][s][has], tc["trim_share"]) for s in tc["seeds"]]))}}
    for n, e in cexp.items():
        trim[n] = {"full": float(np.mean(e[has])), "trimmed": trimmed(e[has], tc["trim_share"])}
    return {"kind": "R5-2 descriptive autopsy and R5-3 diagnostic component swaps (D133); not tests, not methods",
            "arm": tc["arm"], "seeds": tc["seeds"], "n_problems": len(probs), "n_with_draws": int(has.sum()),
            "sensitivity_rel_tol": tol, "per_seed": per_seed, "causes_worst_pooled": pooled,
            "causes_worst_pooled_by_depth": pooled_depth, "trimmed": trim, "rss_to_star_gap": gap, "swaps": swaps,
            "comparators_by_depth": {n: {str(k): float(np.mean(e[has & (depth == k)])) for k in sorted(set(depth))}
                                     for n, e in cexp.items()}}


def render_tail(res: dict) -> str:
    f = lambda v, d=4: "–" if v is None else f"{v:+.{d}f}"
    ci = lambda d, k="ci95", n=4: "–" if not d or d.get("missing") else f"[{d[k][0]:+.{n}f}, {d[k][1]:+.{n}f}]"
    seeds = res["seeds"]
    n_worst = sum(res["per_seed"][str(s)]["n_worst"] if str(s) in res["per_seed"] else res["per_seed"][s]["n_worst"]
                  for s in seeds)
    ps = lambda s: res["per_seed"][s] if s in res["per_seed"] else res["per_seed"][str(s)]
    L = ["# R5-2 tail autopsy and R5-3 component swaps: D-pos-v2, frozen R4-5 set, 3 seeds (D133)", "",
         f"**Kind:** {res['kind']}. {res['n_problems']} problems ({res['n_with_draws']} with predictive draws).", "",
         "## R5-2: the worst 5% of problems by regret against σ* (per seed, pooled over seeds)", "",
         f"{n_worst} problem-seeds. Share of each seed's regret in its worst 5%: " +
         ", ".join(f"{ps(s)['worst_regret_share']:.0%}" for s in seeds) + ". Computation-correct μ: " +
         ", ".join(f"{ps(s)['worst_computation_correct']:.0%}" for s in seeds) + " in the worst 5% vs " +
         ", ".join(f"{ps(s)['all_computation_correct']:.0%}" for s in seeds) + " overall.", "",
         "| primary cause | worst 5% (pooled) | share | all problems (pooled) |", "|---|---|---|---|"]
    allc = sum((Counter(ps(s)["causes_all"]) for s in seeds), Counter())
    tot_all = sum(allc.values())
    for c, n in Counter(res["causes_worst_pooled"]).most_common():
        L.append(f"| {c} | {n} | {n / n_worst:.0%} | {allc.get(c, 0)} ({allc.get(c, 0) / tot_all:.1%}) |")
    cm = [ps(s)["compensation"] for s in seeds]
    L += ["", "**Offsetting errors:** on problems with a material sensitivity error, the model's own aggregation and "
          "copy move σ back the other way by a material amount on " + ", ".join(f"{c['share_offsetting']:.0%}" for c in cm)
          + " (seeds " + ", ".join(map(str, seeds)) + "; n = " + ", ".join(str(c["n"]) for c in cm) +
          "); median share of the sensitivity error undone: " + ", ".join(f"{c['median_share_undone']:.0%}" for c in cm)
          + "."]
    L += ["", "**By depth (worst 5%, pooled):**", ""]
    depths = sorted(res["causes_worst_pooled_by_depth"], key=int)
    causes = [c for c, _ in Counter(res["causes_worst_pooled"]).most_common()]
    L += ["| cause | " + " | ".join(f"d{d}" for d in depths) + " |", "|---|" + "---|" * len(depths)]
    for c in causes:
        L.append(f"| {c} | " + " | ".join(str(Counter(res["causes_worst_pooled_by_depth"][d]).get(c, 0)) for d in depths)
                 + " |")
    L += ["", "**Trimmed expected score (each source drops its own lowest 5% of problems):**", "",
          "| source | full | 95%-trimmed |", "|---|---|---|"]
    for n, t in res["trimmed"].items():
        L.append(f"| {n} | {t['full']:+.4f} | {t['trimmed']:+.4f} |")
    L += ["", "## R5-3: one component exact at a time (diagnostic; seed means; problems that do not decompose keep (a) "
          "except under 'exact')", "",
          f"RSS-to-σ* gap: {res['rss_to_star_gap']:.5f}. Decomposable: " +
          ", ".join(f"{ps(s)['share_decomposable']:.1%}" for s in seeds) + ".", "",
          "(a)'s slope here grades every pair (unanswered members at the floor), as for (b) and (c); K1's uses "
          "answered pairs only.", "",
          "| variant | expected crps_log | − (a) [95% CI] | − RSS [95% CI] | − X1 [95% CI] | share of RSS→σ* gap | "
          "A6c slope [90% CI] |", "|---|---|---|---|---|---|---|"]
    for v, s in res["swaps"].items():
        sl = s["a6c_slope"]
        L.append(f"| {s['name']} | {s['expected']:+.4f} | {f(s['minus_a']['value']) if s['minus_a'] else '–'} "
                 f"{ci(s['minus_a'])} | {f(s['minus_rss']['value'])} {ci(s['minus_rss'])} | "
                 f"{f(s['minus_x1']['value'])} {ci(s['minus_x1'])} | "
                 f"{s['share_of_rss_to_star_gap']:+.0%} | {f(sl['value'], 3)} {ci(sl, 'ci90', 3)} |")
    L += ["", "**Expected crps_log by depth:**", "", "| source | " + " | ".join(f"d{d}" for d in depths) + " |",
          "|---|" + "---|" * len(depths)]
    for v, s in res["swaps"].items():
        L.append(f"| {v} | " + " | ".join(f"{s['expected_by_depth'][d]:+.4f}" for d in depths) + " |")
    for n, e in res["comparators_by_depth"].items():
        L.append(f"| {n} | " + " | ".join(f"{e[d]:+.4f}" for d in depths) + " |")
    return "\n".join(L) + "\n"


def _write(name: str, res: dict, md: str, cfg: dict) -> None:
    out = REPO_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(res, indent=1, default=B._json_default))
    (out / f"{name}.md").write_text(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["depth5", "tail"])
    a = ap.parse_args()
    cfg = _config()
    if a.what == "depth5":
        res = depth5(cfg)
        _write("depth5", res, render_depth5(res), cfg)
        print(render_depth5(res))
    elif a.what == "tail":
        res = tail(cfg)
        _write("tail", res, render_tail(res), cfg)
        print(render_tail(res))


if __name__ == "__main__":
    main()

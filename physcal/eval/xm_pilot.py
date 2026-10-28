"""The cross-model pilot's analysis (DECISIONS.md D137, pre-registered; R5-5), CPU only, run once after both Llama
evals exist:

    PHYSCAL_CONFIRMATORY=1 OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.eval.xm_pilot analyze

The go rule (D137 §3, R4-3's criterion on D-pos-v2's answer σ (a), one seed): K1 (the A6c slope's one-sided pairs-
bootstrap p < 0.05, i.e. C1's 90% lower bound > 0 at one seed), C2 (the slope ≥ 0.5), C3 (the A6b slope's 90% CI on
computation-correct pairs inside ±0.3; fewer than 3 such pairs = not met). Estimator: `r4_analysis.slope` (D95's OLS
within pairs, B = 10,000, seed 0), as #30. Everything else is descriptive: the other σ sources, B-v2, expected-score
differences against B-v2 / RSS / X1 (cluster bootstrap over pairs), per-depth OLS and Theil-Sen slopes (fragile,
#34), post-SFT format and arithmetic, the R5-2 tail categories, and Qwen seed 0's same statistics on the same sets.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter

import numpy as np
from scipy import stats

from physcal.config import REPO_ROOT, load_config
from physcal.eval import a6_pairs, oracle_centred as oc
from physcal.eval import r4_analysis as R
from physcal.eval import r4_confirm as RC
from physcal.eval import r5_diag as D
from physcal.eval import stage_a
from physcal.eval import stage_b_analysis as B
from physcal.verify.reward import reward_config

SETS = ("a6b_pairs", "a6c_pairs")


def _config() -> dict:
    return load_config("stage_xm")


def models(cfg: dict) -> dict:
    """The runs analysed: Llama (this stage) per arm, and Qwen seed 0 (stage R4's confirmatory eval) for reference."""
    r4 = load_config(cfg["stage_r4_config"])
    sub = cfg["eval"]["output_subdir"]
    return {"llama": {arm: REPO_ROOT / cfg["output_dir"] / arm / f"seed{cfg['seed']}" / sub for arm in r4["arms"]},
            "qwen_seed0": {arm: REPO_ROOT / r4["output_dir"] / arm / "seed0" / r4["confirmatory"]["output_subdir"]
                           for arm in r4["arms"]}}


def go_rule(c: dict, b: dict, ac: dict, margin: float) -> dict:
    """K1 ∧ C2 ∧ C3 on (a): c = the A6c slope test (90% CI), b = the A6b one (computation-correct pairs)."""
    k1 = c["slope"] is not None and c["p_one_sided"] < ac["alpha"]  # one test, no family: K1 = C1 at one seed
    c1 = c["slope"] is not None and c["ci"][0] > 0
    c2 = c["slope"] is not None and c["slope"] >= ac["practical_slope"]
    c3 = b["slope"] is not None and -margin < b["ci"][0] and b["ci"][1] < margin  # < 3 pairs: slope None, not met
    return {"K1": bool(k1), "C1": bool(c1), "C2": bool(c2), "C3": bool(c3), "go": bool(k1 and c2 and c3),
            "a6c_slope": c["slope"], "a6c_ci90": c["ci"], "a6c_p_one_sided": c["p_one_sided"], "a6c_n_pairs": c["n_pairs"],
            "a6b_slope": b["slope"], "a6b_ci90": b["ci"], "a6b_n_pairs": b["n_pairs"]}


def theil_sen_by_depth(pairs: list[dict], alpha: float) -> dict:
    out = {}
    for d in sorted({p["lo"]["pair"][1] for p in pairs}):
        sub = [p for p in pairs if p["lo"]["pair"][1] == d]
        x, y = D.pair_xy(sub)
        if len(sub) >= 3 and np.ptp(x) > 0:
            ts = stats.theilslopes(y, x, alpha=alpha)
            out[str(d)] = {"n": len(sub), "theil_sen": float(ts.slope), "ci": [float(ts.low_slope), float(ts.high_slope)],
                           "n_reversed": int((y < 0).sum())}
    return out


def format_arithmetic(problems: list[dict], recs: list[dict], tau: float) -> dict:
    """Post-SFT format-following and arithmetic: the verifier's reason, the σ, the line (D-pos-v2), μ vs f(x_meas)."""
    from physcal.eval import stage_d_analysis as Dn

    by = {(r["split"], *B._key(r["problem"])): r for r in recs}
    rows = []
    for p in problems:
        r = by.get((p["_set"], *B._key(p)))
        b = r["breakdown"] if r else {"reason": "missing"}
        ln = Dn.parse_line(r["completion"]) if r else {"present": False, "parsed": False}
        mu = b.get("mu_si")
        ok = b["reason"] == "ok" and mu is not None and mu > 0
        rows.append({"depth": p["depth"], "set": p["_set"], "reason": b["reason"], "truncated": bool(r and r.get("truncated")),
                     "sigma": ok and b.get("sigma_si") is not None, "line_parsed": bool(ln.get("parsed")),
                     "cc": ok and abs(math.log(mu / p["noise"]["point_estimate_si"])) <= tau,
                     "truth_1pct": ok and b.get("rel_error") is not None and abs(b["rel_error"]) <= 0.01})
    share = lambda rs, f: float(np.mean([f(r) for r in rs])) if rs else None
    summ = lambda rs: {"n": len(rs), "parsed": share(rs, lambda r: r["reason"] != "parse_fail"),
                       "units_ok": share(rs, lambda r: r["reason"] == "ok"), "sigma_stated": share(rs, lambda r: r["sigma"]),
                       "line_parsed": share(rs, lambda r: r["line_parsed"]), "truncated": share(rs, lambda r: r["truncated"]),
                       "computation_correct": share(rs, lambda r: r["cc"]), "within_1pct_of_truth": share(rs, lambda r: r["truth_1pct"]),
                       "reasons": dict(Counter(r["reason"] for r in rs))}
    return {"all": summ(rows), "by_depth": {str(d): summ([r for r in rows if r["depth"] == d])
                                            for d in sorted({r["depth"] for r in rows})},
            "by_set": {s: summ([r for r in rows if r["set"] == s]) for s in SETS}}


def tail_categories(problems, refs, recs, vals, exp_a, star, tol, tau, tc) -> dict:
    by = {(r["split"], *B._key(r["problem"])): r for r in recs}
    dec = [D.decompose(p, ref, by.get((p["_set"], *B._key(p))), v, tol, tau, tc["max_pow"], tc["material_factor"])
           for p, ref, v in zip(problems, refs, vals)]
    regret = star - exp_a
    w = D.worst(regret, tc["worst_share"])
    return {"n_worst": len(w), "worst_regret_share": float(regret[w].sum() / regret[np.isfinite(regret)].sum()),
            "causes_worst": dict(Counter(dec[i]["cause"] for i in w).most_common()),
            "causes_all": dict(Counter(d["cause"] for d in dec).most_common()),
            "share_decomposable": float(np.mean([d["decomposable"] for d in dec]))}


def analyze_model(name: str, dirs: dict, ctx: dict) -> dict:
    """Everything D137 reports for one model's two arms on the frozen A6c + A6b."""
    probs, terms, has, ac, pc, sc = ctx["problems"], ctx["terms"], ctx["has"], ctx["ac"], ctx["pc"], ctx["sc"]
    floor, tau, margin = pc["analysis"]["sigma_floor_rel"], pc["analysis"]["tau"], pc["invariance_analysis"]["equivalence_margin"]
    out: dict = {"arms": {}}
    exps = {}
    for arm, d in dirs.items():
        recs = B.read_records(d)
        if recs is None:
            out["arms"][arm] = {"missing": str(d)}
            continue
        recs = [r for r in recs if r["split"] in SETS]
        a = {}
        src = RC.seed_sources(probs, recs, arm, ctx["refs"], ctx["dcfg"], ac)
        for s, sig in src["sigma"].items():
            pairs = R.model_pairs(probs, src["bds"], pc) if s == "a" else R.source_pairs(probs, sig, floor)
            c, b = R.slope(pairs["a6c"], ac), R.slope(pairs["a6b"], ac, "truth")
            a[s] = {**go_rule(c, b, ac, margin), "theil_sen_by_depth": theil_sen_by_depth(pairs["a6c"], ac["criterion_ci_level"]),
                    "ols_by_depth": {k: v["slope"] for k, v in R.splits(pairs["a6c"], ctx["types"], ac)["by_depth"].items()}}
            if s == "a":
                b_all = R.slope([p for p in a6_pairs.pairs_from_rows(B.model_rows(probs, src["bds"], pc)["a6b_pairs"])
                                 if p["lo"]["ok"] and p["hi"]["ok"]], ac, "truth")
                a[s]["a6b_answered_pairs_slope"] = {"slope": b_all["slope"], "ci90": b_all["ci"], "n": b_all["n_pairs"]}
        exps[arm] = oc.score_source(terms, src["sigma"]["a"], sc)["expected"]
        out["arms"][arm] = {"sources": a, "format_arithmetic": format_arithmetic(probs, recs, tau),
                            "expected": float(np.mean(exps[arm][has]))}
        if arm == "dpos2":
            out["arms"][arm]["tail"] = tail_categories(probs, ctx["refs"], recs, ctx["vals"], exps[arm],
                                                       ctx["cexp"]["sigma_star"], ac["sensitivity_rel_tol"], tau, ctx["tail"])
    boot, cexp = ctx["boot"], ctx["cexp"]
    gap = float(np.mean(cexp["sigma_star"][has]) - np.mean(cexp["rss"][has]))
    diff = lambda x, y: RC.seed_mean_diff(boot, x, y, has)
    out["differences"] = {}
    if "dpos2" in exps:
        out["differences"]["dpos2_minus_rss"] = diff(exps["dpos2"], cexp["rss"])
        out["differences"]["dpos2_minus_x1"] = diff(exps["dpos2"], cexp["x1"])
        out["differences"]["dpos2_gap_share"] = (float(np.mean(exps["dpos2"][has])) - float(np.mean(cexp["rss"][has]))) / gap
    if "b2" in exps:
        out["differences"]["b2_minus_rss"] = diff(exps["b2"], cexp["rss"])
    if "dpos2" in exps and "b2" in exps:
        out["differences"]["dpos2_minus_b2"] = diff(exps["dpos2"], exps["b2"])
    out["comparators_expected"] = {n: float(np.mean(e[has])) for n, e in cexp.items()}
    return out


def context(cfg: dict) -> dict:
    from physcal.env.generator import problem_from_dict
    from physcal.train.sft_r4 import solution_values

    RC._guard()
    r4 = load_config(cfg["stage_r4_config"])
    ac = r4["analysis"]
    sa = RC.stage_a_config(r4)
    sa["pair_sets"] = [s for s in sa["pair_sets"] if s in SETS]
    bcfg = load_config(ac["stage_b_config"])["analysis"]
    pc = load_config(bcfg["pairs_config"])
    sc = reward_config()["scoring"]
    dcfg = load_config(r4["sft_phase12_config"])
    probs = stage_a.load_sets(sa)
    terms = [oc.problem_terms(p, sa["mc"]["n_draws"], sa["mc"]["seed_tag"]) for p in probs]
    src, note = stage_a.sources(probs, sa)
    comp = {"sigma_star": src[stage_a.STAR], "rss": src[bcfg["rss_source"]], "x1": src[bcfg["x1_source"]]}
    refs = R.references(probs, dcfg)
    return {"problems": probs, "terms": terms, "has": np.array([t["draws"] is not None for t in terms]), "ac": ac, "pc": pc,
            "sc": sc, "dcfg": dcfg, "refs": refs, "types": R.pair_types(probs, refs), "sources_note": note,
            "vals": [solution_values(problem_from_dict({k: v for k, v in p.items() if k != "_set"})) for p in probs],
            "cexp": {n: oc.score_source(terms, s, sc)["expected"] for n, s in comp.items()},
            "boot": oc.ClusterBootstrap([(p["_set"], p["noise"]["pair"]["pair_seed"]) for p in probs], ac["bootstrap"], ac["seed"]),
            "tail": load_config("r5_diag")["tail"]}


def analyze(cfg: dict | None = None, force: bool = False) -> dict:
    cfg = cfg or _config()
    out = REPO_ROOT / cfg["output_dir"] / "pilot"
    if (out / "results.json").exists() and not force:
        raise RuntimeError(f"D137: the pilot analysis runs once; {out / 'results.json'} exists")
    dirs = models(cfg)
    missing = [str(d) for d in dirs["llama"].values() if B.read_records(d) is None]
    if missing:
        raise RuntimeError(f"D137: the analysis runs once both Llama evals exist; missing {missing}")
    ctx = context(cfg)
    res = {"decision": "D137 (R5-5 cross-model pilot, pre-registered)", "n_problems": len(ctx["problems"]),
           "sets": list(SETS), "sources_note": ctx["sources_note"], "models": {}}
    for name, d in dirs.items():
        res["models"][name] = analyze_model(name, d, ctx)
    g = res["models"]["llama"]["arms"]["dpos2"]["sources"]["a"]
    res["primary"] = {k: g[k] for k in ("K1", "C2", "C3", "go")}
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=B._json_default))
    (out / "summary.md").write_text(render(res))
    return res


def render(res: dict) -> str:
    f = lambda v, n=3: "–" if v is None else f"{v:+.{n}f}"
    ci = lambda c, n=3: "–" if not c else f"[{c[0]:+.{n}f}, {c[1]:+.{n}f}]"
    pct = lambda v: "–" if v is None else f"{100 * v:.1f}%"
    p = res["primary"]
    L = ["# The cross-model pilot (D137): D-pos-v2 and B-v2 on Llama-3.2-3B-Instruct, seed 0, frozen A6c + A6b", "",
         f"**Go rule (K1 ∧ C2 ∧ C3 on D-pos-v2's answer σ): {'GO' if p['go'] else 'NO-GO'}** "
         f"(K1 {p['K1']}, C2 {p['C2']}, C3 {p['C3']}). {res['n_problems']} problems.", ""]
    for name, m in res["models"].items():
        L += [f"## {name}", "", "| arm / σ source | A6c slope [90% CI] (n pairs) | p (one-sided) | A6b slope, cc pairs [90% CI] (n) | "
              "K1 / C2 / C3 | Theil-Sen by depth 1–5 (fragile, #34) |", "|---|---|---|---|---|---|"]
        for arm, a in m["arms"].items():
            if "missing" in a:
                L.append(f"| {arm} | missing | | | | |")
                continue
            for s, g in a["sources"].items():
                ts = " / ".join(f"{v['theil_sen']:+.2f}" for v in g["theil_sen_by_depth"].values())
                L.append(f"| {arm} ({s}) | {f(g['a6c_slope'])} {ci(g['a6c_ci90'])} ({g['a6c_n_pairs']}) | "
                         f"{g['a6c_p_one_sided'] if g['a6c_p_one_sided'] is not None else '–'} | {f(g['a6b_slope'])} "
                         f"{ci(g['a6b_ci90'])} ({g['a6b_n_pairs']}) | {g['K1']} / {g['C2']} / {g['C3']} | {ts} |")
        L += ["", "| arm | parsed | units ok | σ stated | σ line parsed | truncated | computation-correct | within 1% of truth | "
              "expected crps_log |", "|---|---|---|---|---|---|---|---|---|"]
        for arm, a in m["arms"].items():
            if "missing" in a:
                continue
            fa = a["format_arithmetic"]["all"]
            L.append(f"| {arm} | {pct(fa['parsed'])} | {pct(fa['units_ok'])} | {pct(fa['sigma_stated'])} | "
                     f"{pct(fa['line_parsed'])} | {pct(fa['truncated'])} | {pct(fa['computation_correct'])} | "
                     f"{pct(fa['within_1pct_of_truth'])} | {a['expected']:+.4f} |")
        L += ["", "Expected crps_log differences (oracle-centred, cluster bootstrap over pairs, 95% CI; descriptive): " +
              "; ".join(f"{k} {f(v['value'], 4)} {ci(v['ci95'], 4)}" for k, v in m["differences"].items()
                        if isinstance(v, dict)) +
              (f"; D-pos-v2's share of the RSS-to-σ* gap {m['differences']['dpos2_gap_share']:+.0%}"
               if "dpos2_gap_share" in m["differences"] else ""), ""]
        t = m["arms"].get("dpos2", {}).get("tail")
        if t:
            L += [f"R5-2 tail (D-pos-v2 (a), worst 5% = {t['n_worst']} problems, {t['worst_regret_share']:.0%} of the regret): " +
                  "; ".join(f"{k} {v}" for k, v in t["causes_worst"].items()), ""]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["analyze"])
    ap.parse_args()
    print(render(analyze()))


if __name__ == "__main__":
    main()

"""D151 item 2 (EXPLORATORY, follow-up project; not the paper): exact arithmetic on D-pos-v2's own lines.

    PHYSCAL_CONFIRMATORY=1 OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.eval.followup_arith

A calculator on the model's own text, nothing from the builder (unlike FINDINGS #35's oracle swaps): every S line's
formula is re-evaluated at the values the model itself wrote (r5_diag.safe_eval, a restricted arithmetic AST), the
σ line is recomputed from those sensitivities and its own levels, and σ = X'·|μ| exactly. The structure (which terms,
which formulas, which levels) is unchanged. Reads the frozen R4-5 D-pos-v2 logs; writes <output_dir>/arith/.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter

import numpy as np

from physcal.config import REPO_ROOT, load_config

NAME_LINE = re.compile(r"^[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*=(.*)$")
LEAD_NUM = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")
VARIANTS = ("a", "c35", "c_own", "E", "E_S", "E_chain")
NAMES = {"a": "(a) as stated", "c35": "(c) #35: model's terms, exact aggregation and copy (anchor)",
         "c_own": "(c') the same on this analysis's problems: model's σ-line terms, exact aggregation and copy",
         "E": "E (primary): S re-evaluated at the model's own values, exact aggregation and copy",
         "E_S": "E_S: S re-evaluated, the model's own aggregation and copy kept",
         "E_chain": "E_chain: intermediates re-derived from the model's own formulas, then S; exact aggregation and copy"}


def _config() -> dict:
    return load_config("followup")


def _lead(text: str) -> float | None:
    m = LEAD_NUM.match(text)
    return float(m.group(1)) if m else None


def model_values(completion: str) -> tuple[dict, list[tuple[str, str]]]:
    """The values the model wrote before <steps>: every `name = ... = v unit` (each ;-part of the "SI values:" line
    and every derivation line; not S_* or sigma_rel), v = the leading number of the last =-part, later lines win.
    Also the derivation lines in order as (name, first =-part), for E_chain."""
    head = completion.split("<steps>", 1)[0]
    vals: dict = {}
    derivations: list[tuple[str, str]] = []
    for line in head.splitlines():
        si = line.strip().startswith("SI values:")
        parts = line.split(":", 1)[1].split(";") if si else [line]
        for part in parts:
            m = NAME_LINE.match(part)
            if not m or m.group(1).startswith("S_") or m.group(1) == "sigma_rel":
                continue
            rhs = [x.strip() for x in m.group(2).split("=")]
            v = _lead(rhs[-1])
            if v is None:
                continue
            vals[m.group(1)] = v
            if not si and len(rhs) >= 2:
                derivations.append((m.group(1), rhs[0]))
    return vals, derivations


def chain_values(completion: str, max_pow: float) -> dict:
    """E_chain: start from the model's written values, then re-derive every derivation line in order from its own
    formula with the values so far (a formula that does not evaluate keeps the written value)."""
    from physcal.eval.r5_diag import safe_eval

    vals, derivations = model_values(completion)
    cur = dict(vals)
    for name, formula in derivations:
        v = safe_eval(formula, cur, max_pow)
        if v is not None:
            cur[name] = v
    return cur


def reevaluate(completion: str, terms: list[tuple[float, float]], names: dict, max_pow: float) -> dict | None:
    """The σ-line terms with every sensitivity re-evaluated from the matching S line (by position) at `names`; None
    if the S lines and the σ terms differ in number. A formula that does not evaluate keeps the σ line's number."""
    from physcal.eval.r5_diag import s_line_parts, safe_eval

    lines = list(s_line_parts(completion).values())
    if len(lines) != len(terms):
        return None
    new, n_eval = [], 0
    for parts, (a, s) in zip(lines, terms):
        v = safe_eval(parts[0], names, max_pow) if len(parts) >= 2 else safe_eval(parts[0], {}, max_pow)
        if v is None:
            v = a
        else:
            n_eval += 1
        new.append((v, s))
    return {"terms": new, "n_evaluated": n_eval}


def sigmas(p: dict, rec: dict | None, d35: dict, ac: dict) -> dict:
    """σ_rel under every variant for one problem; problems that do not decompose keep (a)."""
    from physcal.eval import stage_d_analysis as Dn
    from physcal.eval.r5_diag import rss

    comp = rec["completion"] if rec else ""
    b = rec["breakdown"] if rec else {"reason": "missing"}
    ln = Dn.parse_line(comp)
    answered = b["reason"] == "ok" and b.get("sigma_si") is not None and (b.get("mu_si") or 0) > 0
    a = b["sigma_si"] / b["mu_si"] if answered else 0.0
    out = {"depth": p["depth"], "sigma": {v: a for v in VARIANTS}, "status": None, "n_terms": 0, "n_evaluated": 0}
    out["sigma"]["c35"] = d35["sigma"]["AC"]
    if not ln.get("present") or not ln.get("parsed") or ln.get("zero") or not answered or ln["x"] <= 0 \
            or rss(ln["terms"]) <= 0:
        out["status"] = "does not decompose"
        return out
    own = rss(ln["terms"])
    vals, _ = model_values(comp)
    e = reevaluate(comp, ln["terms"], vals, ac["max_pow"])
    if e is None:
        out["status"] = "S lines and σ terms differ in number"
        return out
    ec = reevaluate(comp, ln["terms"], chain_values(comp, ac["max_pow"]), ac["max_pow"])
    out["status"] = "re-evaluated"
    out["n_terms"], out["n_evaluated"] = len(ln["terms"]), e["n_evaluated"]
    out["sigma"].update(c_own=own, E=rss(e["terms"]), E_S=a * rss(e["terms"]) / own, E_chain=rss(ec["terms"]))
    return out


def analyze(cfg: dict | None = None) -> dict:
    from physcal.env.generator import problem_from_dict
    from physcal.eval import oracle_centred as oc
    from physcal.eval import r4_confirm as RC
    from physcal.eval import r5_diag as RD
    from physcal.eval import stage_a
    from physcal.eval import stage_b_analysis as B
    from physcal.train.sft_r4 import solution_values
    from physcal.verify.reward import reward_config

    cfg = cfg or _config()
    ac = cfg["arith"]
    rcfg = RD._config()
    ctx = RD.Ctx("frozen", rcfg)
    r4ac, sa, pc = ctx.ac, ctx.sa, ctx.pc
    floor, tau, tol = pc["analysis"]["sigma_floor_rel"], pc["analysis"]["tau"], r4ac["sensitivity_rel_tol"]
    sc = reward_config()["scoring"]
    probs = ctx.problems
    terms = [oc.problem_terms(p, sa["mc"]["n_draws"], sa["mc"]["seed_tag"]) for p in probs]
    has = np.array([t["draws"] is not None for t in terms])
    truth_vals = [solution_values(problem_from_dict({k: v for k, v in p.items() if k != "_set"})) for p in probs]
    src, _ = stage_a.sources(probs, sa)
    bcfg = load_config(r4ac["stage_b_config"])["analysis"]
    comp = {"sigma_star": src[stage_a.STAR], "rss": src[bcfg["rss_source"]], "x1": src[bcfg["x1_source"]]}
    cexp = {n: oc.score_source(terms, s, sc)["expected"] for n, s in comp.items()}
    boot = oc.ClusterBootstrap([(p["_set"], p["noise"]["pair"]["pair_seed"]) for p in probs], r4ac["bootstrap"],
                               r4ac["seed"])
    depth = np.array([p["depth"] for p in probs])
    exp, rows, per_seed = {v: {} for v in VARIANTS}, {v: {} for v in VARIANTS}, {}
    tc = rcfg["tail"]
    for s in ac["seeds"]:
        recs = ctx.records(ac["arm"], s)
        by = {(r["split"], *B._key(r["problem"])): r for r in recs}
        got = [by.get((p["_set"], *B._key(p))) for p in probs]
        d35 = [RD.decompose(p, ref, r, v, tol, tau, tc["max_pow"], tc["material_factor"])
               for p, ref, r, v in zip(probs, ctx.refs, got, truth_vals)]
        res = [sigmas(p, r, d, ac) for p, r, d in zip(probs, got, d35)]
        for v in VARIANTS:
            sig = np.array([x["sigma"][v] for x in res])
            exp[v][s] = oc.score_source(terms, sig, sc)["expected"]
            rows[v][s] = B.source_rows(probs, sig, floor)
        re_ = [x for x in res if x["status"] == "re-evaluated"]
        moved = lambda v: float(np.mean([abs(math.log(max(x["sigma"][v], 1e-300) / max(x["sigma"]["a"], 1e-300)))
                                         > math.log(ac["material_factor"]) for x in re_]))
        per_seed[s] = {"status": Counter(x["status"] for x in res),
                       "terms": int(sum(x["n_terms"] for x in re_)),
                       "terms_evaluated_share": sum(x["n_evaluated"] for x in re_) / max(1, sum(x["n_terms"] for x in re_)),
                       "moved_material_vs_a": {v: moved(v) for v in ("c_own", "E", "E_S", "E_chain")},
                       "E_vs_c_own_moved_material": float(np.mean(
                           [abs(math.log(x["sigma"]["E"] / x["sigma"]["c_own"])) > math.log(ac["material_factor"])
                            for x in re_ if x["sigma"]["E"] > 0 and x["sigma"]["c_own"] > 0]))}
    gap = float(np.mean(cexp["sigma_star"][has]) - np.mean(cexp["rss"][has]))
    out = {"kind": "D151 item 2, exploratory: a calculator on the model's own lines (not the paper; kept apart from "
                   "FINDINGS #35's oracle swaps)", "arm": ac["arm"], "seeds": ac["seeds"], "n_problems": len(probs),
           "n_with_draws": int(has.sum()), "per_seed": per_seed, "variants": {},
           "comparators": {n: float(np.mean(e[has])) for n, e in cexp.items()}, "rss_to_star_gap": gap}
    for v in VARIANTS:
        m = float(np.mean([np.mean(exp[v][s][has]) for s in ac["seeds"]]))
        out["variants"][v] = {
            "name": NAMES[v], "expected": m,
            "minus_a": RC.seed_mean_diff(boot, exp[v], exp["a"], has) if v != "a" else None,
            "minus_c_own": RC.seed_mean_diff(boot, exp[v], exp["c_own"], has) if v in ("E", "E_chain") else None,
            "minus_rss": RC.seed_mean_diff(boot, exp[v], cexp["rss"], has),
            "minus_x1": RC.seed_mean_diff(boot, exp[v], cexp["x1"], has),
            "share_of_rss_to_star_gap": (m - float(np.mean(cexp["rss"][has]))) / gap,
            "a6c_slope": RC.seed_mean_slope(rows[v], "a6c_pairs", "ok", r4ac),
            "expected_by_depth": {str(k): float(np.mean([np.mean(exp[v][s][has & (depth == k)]) for s in ac["seeds"]]))
                                  for k in sorted(set(depth))}}
    an = ac["anchors"]
    out["anchors"] = {k: {"expected": out["variants"][k if k != "c" else "c35"]["expected"], "anchor": an[k],
                          "ok": abs(out["variants"][k if k != "c" else "c35"]["expected"] - an[k]) <= ac["anchor_atol"]}
                      for k in an}
    out["comparators_by_depth"] = {n: {str(k): float(np.mean(e[has & (depth == k)])) for k in sorted(set(depth))}
                                   for n, e in cexp.items()}
    return out


def main() -> None:
    cfg = _config()
    res = analyze(cfg)
    out = REPO_ROOT / cfg["output_dir"] / "arith"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"anchors": res["anchors"], "variants": {v: {k: res["variants"][v][k] for k in
                      ("expected", "minus_a", "minus_rss", "minus_x1", "a6c_slope")} for v in res["variants"]}},
                     indent=1, default=float))


if __name__ == "__main__":
    main()

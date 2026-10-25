"""R4-1 to R4-3 (DECISIONS.md D113, pre-registered): the analysis of D-pos-v2 and B-v2, CPU only.

    OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.eval.r4_analysis analyze

Each arm goes through stage B's analysis unchanged (B1-B5, labels, the oracle-centred diagnostic). For D-pos-v2 three
σ values are graded (R4-2): (a) the answer's σ_rel; (b) the line's X; (c) sqrt(Σ (a_i·s_i)²) of the line's own
terms, exact; plus (c_S), the model's S lines by name with the problem's true levels (descriptive). R4-3's criteria
per source: C1 the A6c slope's 90% CI lower bound > 0, C2 the slope ≥ 0.5, C3 the A6b slope's 90% CI inside D59's
margin; the go rule (3 seeds of both arms) fires if D-pos-v2 meets C1-C3 on (a), (b) or (c). R1-R8 (Holm) for the
record. Everything split by depth and by sensitivity type (the builder's own derivation). A6c is in-distribution for
both arms (mixed levels in training).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.eval import a6_pairs, oracle_centred as oc
from physcal.eval import stage_a
from physcal.eval import stage_b_analysis as B
from physcal.eval import stage_d_analysis as Dn
from physcal.eval.stats import holm
from physcal.verify.reward import reward_config

ARMS = ("dpos2", "b2")
TESTS = tuple(f"R{i}" for i in range(1, 9))
S_LINE_RE = re.compile(r"^[ \t]*S_([A-Za-z_][A-Za-z0-9_]*)[ \t]*=(.*)$", re.M)
END_NUM_RE = re.compile(rf"({Dn.NUM})[ \t]*$")
TEST_NAMES = {"R1": "D-pos-v2 (a) answer σ: A6c slope > 0", "R2": "D-pos-v2 (b) line X: A6c slope > 0",
              "R3": "D-pos-v2 (c) exact from terms: A6c slope > 0", "R4": "B-v2: A6c slope > 0",
              "R5": "expected crps_log, D-pos-v2 − B-v2 > 0", "R6": "D-pos-v2 − RSS > 0", "R7": "D-pos-v2 − X1 > 0",
              "R8": "B-v2 − RSS > 0"}


def _config() -> dict:
    return load_config("stage_r4")


# ------------------------------------------------------------------------------------------- parsing
def parse_s_lines(completion: str) -> dict[str, float]:
    """The model's `S_x = ... = value` lines before <answer>: symbol -> the line's last number (the last line wins)."""
    head = completion.split("<answer>", 1)[0]
    out = {}
    for sym, rhs in S_LINE_RE.findall(head):
        m = END_NUM_RE.search(rhs)
        if m:
            out[sym] = float(m.group(1))
    return out


def exact_from_terms(ln: dict) -> float:
    """(c): sqrt(Σ (a_i·s_i)²) of a parsed line's own terms, unrounded; unparsed or zero -> 0 (the floor)."""
    if not ln.get("parsed") or ln.get("zero"):
        return 0.0
    return math.sqrt(sum((a * s) ** 2 for a, s in ln["terms"]))


def from_s_lines(got: dict[str, float], problem: dict) -> float:
    """(c_S): the model's S values by name with the problem's own stated levels (a missing S line counts 0)."""
    return math.sqrt(sum((got.get(g["symbol"], 0.0) * g["noise_rel"]) ** 2 for g in problem["givens"]
                         if g.get("noise_rel")))


def references(problems: list[dict], dcfg: dict) -> list[dict]:
    """Per problem, what the R4-1 builder writes (sft_r4): the parsed line and the S values with their types."""
    from physcal.env.generator import problem_from_dict
    from physcal.train import sft_r4

    out = []
    for p in problems:
        sens = sft_r4.sensitivity_lines(problem_from_dict({k: v for k, v in p.items() if k != "_set"}),
                                        dcfg["sensitivity_format"])
        line, _ = sft_r4.propagation_line(sens, dcfg["sigma_format"])
        out.append({**Dn.parse_line(f"{line}\n<answer>"), "line": line, "sens": {t["symbol"]: t for t in sens},
                    "power_law_chain": all(t["power_law"] for t in sens)})
    return out


# ------------------------------------------------------------------------------------------- S lines
def s_line_rows(problems: list[dict], recs: list[dict], refs: list[dict], tol: float) -> tuple[list[dict], np.ndarray]:
    """One row per reference term (noisy given): present, exact (the reference's .3g value), lenient (within `tol`).
    Also returns (c_S) per problem."""
    by = {(r["split"], *B._key(r["problem"])): r for r in recs}
    rows, cs = [], []
    for p, ref in zip(problems, refs):
        r = by.get((p["_set"], *B._key(p)))
        got = parse_s_lines(r["completion"]) if r else {}
        cs.append(from_s_lines(got, p))
        extra = len(set(got) - set(ref["sens"]))
        for i, (sym, t) in enumerate(ref["sens"].items()):
            v = got.get(sym)
            row = {"set": p["_set"], "depth": p["depth"], "power_law": t["power_law"], "present": v is not None,
                   "exact": v is not None and math.isclose(v, t["value"], rel_tol=1e-9, abs_tol=1e-12),
                   "lenient": v is not None and (abs(v - t["value"]) <= tol * abs(t["value"]) if t["value"] else v == 0),
                   "extra_lines": extra if i == 0 else 0}
            if v and t["value"]:
                row["abs_log"] = abs(math.log(abs(v) / abs(t["value"])))
            rows.append(row)
    return rows, np.array(cs)


def summarize_s(rows: list[dict]) -> dict:
    share = lambda k, rs: float(np.mean([r[k] for r in rs])) if rs else None
    out = {}
    for name, rs in (("all", rows), ("power_law", [r for r in rows if r["power_law"]]),
                     ("value_dependent", [r for r in rows if not r["power_law"]])):
        al = [r["abs_log"] for r in rs if "abs_log" in r]
        out[name] = {"n_terms": len(rs), "present": share("present", rs), "exact": share("exact", rs),
                     "lenient": share("lenient", rs), "median_abs_log": float(np.median(al)) if al else None}
    out["extra_s_lines"] = int(sum(r["extra_lines"] for r in rows))
    return out


def s_diagnostics(rows: list[dict]) -> dict:
    return {"summary": summarize_s(rows),
            "by_depth": {str(d): summarize_s([r for r in rows if r["depth"] == d]) for d in sorted({r["depth"] for r in rows})},
            "by_set": {s: summarize_s([r for r in rows if r["set"] == s]) for s in B.SETS}}


# ------------------------------------------------------------------------------------------- slopes and criteria
def _pair_key(p: dict) -> tuple:
    n = p["noise"]
    return p["family"], p["depth"], n["level"], n["pair"]["pair_seed"]


def pair_types(problems: list[dict], refs: list[dict]) -> dict:
    """Pair key -> "power_law" if every term of the (shared) chain is a power law, else "value_dependent"."""
    return {_pair_key(p): ("power_law" if r["power_law_chain"] else "value_dependent")
            for p, r in zip(problems, refs) if p["noise"]["pair"]["member"] == "lo"}


def slope(pairs: list[dict], ac: dict, regressor: str = "sigma_star") -> dict:
    return a6_pairs.slope_test(pairs, ac["bootstrap"], ac["seed"], regressor, ac["criterion_ci_level"])


def criteria(pairs_c: list[dict], pairs_b: list[dict], ac: dict, margin: float) -> dict:
    """R4-3: C1 (A6c 90% lower bound > 0), C2 (A6c slope ≥ the practical threshold), C3 (A6b 90% CI in the margin)."""
    c, b = slope(pairs_c, ac), slope(pairs_b, ac, "truth")
    ok = c["slope"] is not None and b["slope"] is not None
    out = {"a6c": c, "a6b": b,
           "C1": bool(ok and c["ci"][0] > 0), "C2": bool(ok and c["slope"] >= ac["practical_slope"]),
           "C3": bool(ok and -margin < b["ci"][0] and b["ci"][1] < margin)}
    out["meets"] = out["C1"] and out["C2"] and out["C3"]
    out["label"] = ("meets the propagation criterion" if out["meets"] else
                    "propagates below the practical threshold" if out["C1"] and out["C3"] else
                    "C1 fails: no propagation shown on A6c" if not out["C1"] else "C3 fails: A6b not invariant")
    return out


def splits(pairs: list[dict], types: dict, ac: dict) -> dict:
    """A6c slope (90% CI) by depth and by chain type."""
    out = {"by_depth": {}, "by_type": {}}
    for d in sorted({p["lo"]["pair"][1] for p in pairs}):
        out["by_depth"][str(d)] = slope([p for p in pairs if p["lo"]["pair"][1] == d], ac)
    for t in ("power_law", "value_dependent"):
        out["by_type"][t] = slope([p for p in pairs if types.get(p["lo"]["pair"]) == t], ac)
    return out


def model_pairs(problems: list[dict], bds: list[dict], pc: dict) -> dict:
    """(a)'s pairs: A6c both members answered; A6b and A6 both computation-correct (D104 / D109's sets)."""
    rows = B.model_rows(problems, bds, pc)
    pf = lambda s: a6_pairs.pairs_from_rows(rows[s])
    return {"a6c": [p for p in pf("a6c_pairs") if p["lo"]["ok"] and p["hi"]["ok"]],
            "a6b": [p for p in pf("a6b_pairs") if p["lo"]["computation_correct"] and p["hi"]["computation_correct"]],
            "a6": [p for p in pf("a6_pairs") if p["lo"]["computation_correct"] and p["hi"]["computation_correct"]]}


def source_pairs(problems: list[dict], sigma: np.ndarray, floor: float) -> dict:
    """A source graded on every pair (unparsed -> the floor, D109's line-as-σ convention)."""
    rows = B.source_rows(problems, sigma, floor)
    return {"a6c": a6_pairs.pairs_from_rows(rows["a6c_pairs"]), "a6b": a6_pairs.pairs_from_rows(rows["a6b_pairs"]),
            "a6": a6_pairs.pairs_from_rows(rows["a6_pairs"])}


def by_depth_expected(problems: list[dict], expected: np.ndarray) -> dict:
    d = np.array([p["depth"] for p in problems])
    s = np.array([p["_set"] for p in problems])
    ok = np.isfinite(expected)
    return {st: {str(k): float(np.mean(expected[(d == k) & (s == st) & ok])) for k in sorted(set(d))} for st in B.SETS}


def grade(name: str, pairs: dict, types: dict, ac: dict, margin: float, terms: list[dict], sigma: np.ndarray,
          boot, sc: dict, sa: dict, problems: list[dict]) -> dict:
    scores = oc.score_source(terms, sigma, sc)
    return {"source": name, "criteria": criteria(pairs["a6c"], pairs["a6b"], ac, margin),
            "a6": slope(pairs["a6"], ac), "a6c_splits": splits(pairs["a6c"], types, ac),
            "a6_by_depth": splits(pairs["a6"], types, ac)["by_depth"],
            "oracle_centred": oc.summarize(terms, sigma, scores, boot, sc, sa["coverage_k"]),
            "expected_by_set_depth": by_depth_expected(problems, scores["expected"]), "_expected": scores["expected"]}


# ------------------------------------------------------------------------------------------- the analysis
def analyze(cfg: dict | None = None, root=None) -> dict:
    cfg = cfg or _config()
    ac = cfg["analysis"]
    root = root or REPO_ROOT / cfg["output_dir"]
    bcfg = copy.deepcopy(load_config(ac["stage_b_config"]))
    bcfg["seeds"] = [cfg["seed"]]
    bcfg["analysis"].update(bootstrap=ac["bootstrap"], seed=ac["seed"], alpha=ac["alpha"])
    pc, sa = load_config(bcfg["analysis"]["pairs_config"]), load_config(bcfg["analysis"]["stage_a_config"])
    floor, margin = pc["analysis"]["sigma_floor_rel"], pc["invariance_analysis"]["equivalence_margin"]
    sc = reward_config()["scoring"]
    dcfg = load_config(cfg["sft_phase12_config"])
    problems = stage_a.load_sets(sa)
    terms = [oc.problem_terms(p, sa["mc"]["n_draws"], sa["mc"]["seed_tag"]) for p in problems]
    mask = np.array([t["draws"] is not None for t in terms])
    boot = oc.ClusterBootstrap([(p["_set"], p["noise"]["pair"]["pair_seed"]) for p in problems], ac["bootstrap"], ac["seed"])
    refs = references(problems, dcfg)
    types = pair_types(problems, refs)
    src, note = stage_a.sources(problems, sa)
    comps = {"sigma_star": src[stage_a.STAR], "rss": src[bcfg["analysis"]["rss_source"]],
             "x1": src[bcfg["analysis"]["x1_source"]]}
    G = lambda name, pairs, sigma: grade(name, pairs, types, ac, margin, terms, sigma, boot, sc, sa, problems)

    res: dict = {"decision": "D113 (R4-1 to R4-3, pre-registered; one seed, development sets)",
                 "note": "A6c is in-distribution for both arms (mixed per-given levels in training); depth 5 is held "
                         "out from training", "n_problems": len(problems), "bootstrap": ac["bootstrap"], "sources_note": note,
                 "arms": {},
                 "comparators": {n: G(n, source_pairs(problems, s, floor), s) for n, s in comps.items()}}
    graded: dict = {}
    for arm in ARMS:
        recs = B.read_records(root / arm / f"seed{cfg['seed']}" / "eval")
        if recs is None:
            res["arms"][arm] = None
            continue
        r = B.analyze(bcfg, root / arm)
        sig_a, bds = B.model_sigma(problems, recs)
        arm_res = {"stage_b_analysis": {"tests": r["tests"], "seed_mean": r["seed_mean"],
                                        "seed0": r["seeds"][cfg["seed"]]},
                   "sources": {"a": G("a", model_pairs(problems, bds, pc), sig_a)}}
        graded[(arm, "a")] = arm_res["sources"]["a"]
        if arm == "dpos2":
            diag, xs_b = Dn.line_diagnostics(problems, recs, refs, ac["sensitivity_rel_tol"], Dn.mu_rel_slack(dcfg))
            by = {(x["split"], *B._key(x["problem"])): x for x in recs}
            lines = [Dn.parse_line(by[(p["_set"], *B._key(p))]["completion"]) if (p["_set"], *B._key(p)) in by
                     else {"present": False, "parsed": False} for p in problems]
            xs_c = np.array([exact_from_terms(ln) for ln in lines])
            srows, xs_cs = s_line_rows(problems, recs, refs, ac["sensitivity_rel_tol"])
            for name, xs in (("b", xs_b), ("c", xs_c), ("c_S", xs_cs)):
                arm_res["sources"][name] = G(name, source_pairs(problems, xs, floor), xs)
                graded[(arm, name)] = arm_res["sources"][name]
            arm_res["lines"] = diag
            arm_res["s_lines"] = s_diagnostics(srows)
        res["arms"][arm] = arm_res

    exp = lambda arm, s="a": graded[(arm, s)]["_expected"] if (arm, s) in graded else None
    comp_exp = {n: res["comparators"][n]["_expected"] for n in comps}
    crit = lambda arm, s: graded[(arm, s)]["criteria"]["a6c"] if (arm, s) in graded else None
    t = {}
    for k, (arm, s) in (("R1", ("dpos2", "a")), ("R2", ("dpos2", "b")), ("R3", ("dpos2", "c")), ("R4", ("b2", "a"))):
        c = crit(arm, s)
        t[k] = {"missing": True} if c is None or c["slope"] is None else {
            "value": c["slope"], "ci90": c["ci"], "ci95": c["ci95"], "p_one_sided": c["p_one_sided"],
            "n_pairs": c["n_pairs"]}
    t["R5"] = Dn.paired(boot, exp("dpos2"), exp("b2"), mask)
    t["R6"] = Dn.paired(boot, exp("dpos2"), comp_exp["rss"], mask)
    t["R7"] = Dn.paired(boot, exp("dpos2"), comp_exp["x1"], mask)
    t["R8"] = Dn.paired(boot, exp("b2"), comp_exp["rss"], mask)
    h = holm({k: (1.0 if t[k].get("missing") else t[k]["p_one_sided"]) for k in TESTS}, ac["alpha"])
    for k in TESTS:
        t[k]["holm"] = h[k]
    res["tests"] = t

    # the go rule (R4-3) and the descriptive data effects
    met = [s for s in ("a", "b", "c") if (("dpos2", s) in graded and graded[("dpos2", s)]["criteria"]["meets"])]
    res["go"] = {"three_seeds": bool(met), "met_on": met,
                 "rule": "3 seeds of D-pos-v2 and B-v2 iff D-pos-v2 meets C1-C3 on (a), (b) or (c) (D113)"}
    v1 = B.read_records(REPO_ROOT / ac["stage_d_root"] / "pos" / "seed0" / "eval")
    sb = Dn.stage_b_expected(REPO_ROOT / ac["stage_b_root"], problems, terms, sc, load_config(ac["stage_b_config"])["seeds"])
    res["data_effect"] = {
        "dpos2_minus_dpos": Dn.paired(boot, exp("dpos2"), None if v1 is None else oc.score_source(
            terms, B.model_sigma(problems, v1)[0], sc)["expected"], mask),
        "b2_minus_stage_b_mean": Dn.paired(boot, exp("b2"), None if sb is None else sb[0], mask)}
    res["reading"] = reading(res, graded)
    for g in list(graded.values()) + list(res["comparators"].values()):
        g.pop("_expected", None)
    root.mkdir(parents=True, exist_ok=True)
    (root / "results.json").write_text(json.dumps(res, indent=1, default=B._json_default))
    (root / "summary.md").write_text(render(res))
    return res


def reading(res: dict, graded: dict) -> list[str]:
    out = [f"D-pos-v2 ({s}): {graded[('dpos2', s)]['criteria']['label']}" for s in ("a", "b", "c")
           if ("dpos2", s) in graded]
    if ("b2", "a") in graded:
        out.append(f"B-v2: {graded[('b2', 'a')]['criteria']['label']}")
    m = lambda s: ("dpos2", s) in graded and graded[("dpos2", s)]["criteria"]["meets"]
    if m("c") and not m("a"):
        out.append("(c) propagates and (a) does not: the failure is the arithmetic or the copy, not propagation (R4-2)")
    if m("c") and not m("b"):
        out.append("(c) meets and (b) does not: the line's arithmetic fails (R4-2)")
    out.append(f"go (R4-3): 3 seeds of both arms, met on {', '.join(res['go']['met_on'])}" if res["go"]["three_seeds"]
               else "no go (R4-3): D-pos-v2 meets the propagation criterion on none of (a), (b), (c)")
    return out


# ------------------------------------------------------------------------------------------- report
def render(res: dict) -> str:
    sl = lambda d: "–" if not d or d.get("slope") is None else f"{d['slope']:+.3f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"
    yn = lambda b: "yes" if b else "no"
    L = ["# R4-1 to R4-3 (D113): D-pos-v2 and B-v2 on mixed-level data", "",
         "**Reading:** " + "; ".join(res["reading"]) + ".", "",
         f"One seed per arm, development sets (A6 / A6b / A6c). {res['note']}. Slopes: 90% CIs (R4-3) from a pairs "
         f"bootstrap (B = {res['bootstrap']:,}); paired score differences: 95% CIs, cluster bootstrap over pairs.", "",
         "**R4-3 per σ source** (C1: A6c 90% lower bound > 0; C2: A6c slope ≥ 0.5; C3: A6b 90% CI in ±0.3):", "",
         "| source | A6c slope [90% CI] (n) | A6b slope [90% CI] | C1 | C2 | C3 | criterion | A6 slope | "
         "expected crps_log | median ln(σ/σ*) | 1σ cov. |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for arm, a in res["arms"].items():
        if a:
            rows += [(f"{arm} ({s})", g) for s, g in a["sources"].items()]
    rows += [(n, g) for n, g in res["comparators"].items()]
    for name, g in rows:
        c, o = g["criteria"], g["oracle_centred"]
        L.append(f"| {name} | {sl(c['a6c'])} ({c['a6c']['n_pairs']}) | {sl(c['a6b'])} | {yn(c['C1'])} | {yn(c['C2'])} | "
                 f"{yn(c['C3'])} | {c['label']} | {sl(g['a6'])} | {o['expected'][0]:.4f} | {o['median_log_ratio']:+.3f} | "
                 f"{o['coverage_1'][0]:.1%} |")
    L += ["", "**Tests for the record (Holm over R1–R8):**", "", "| test | estimate | CI (slopes 90%, differences 95%) | p (one-sided) | Holm |",
          "|---|---|---|---|---|"]
    for k in TESTS:
        d = res["tests"][k]
        if d.get("missing"):
            L.append(f"| {k}: {TEST_NAMES[k]} | missing | | 1 | not supported |")
            continue
        ci = d.get("ci90") or d.get("ci95")
        L.append(f"| {k}: {TEST_NAMES[k]} | {d['value']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {d['p_one_sided']:.4g} | "
                 f"{'supported' if d['holm']['rejected'] else 'not supported'} |")
    L += ["", "**A6c slope by depth (depth 5 held out from training) and by chain type:**", "",
          "| source | d1 | d2 | d3 | d4 | d5 | power-law chains | value-dependent chains |", "|---|---|---|---|---|---|---|---|"]
    for name, g in rows:
        s = g["a6c_splits"]
        L.append(f"| {name} | " + " | ".join(sl(s["by_depth"].get(str(d))) for d in range(1, 6)) +
                 f" | {sl(s['by_type']['power_law'])} | {sl(s['by_type']['value_dependent'])} |")
    d = res["arms"].get("dpos2")
    if d:
        f = lambda v: "–" if v is None else f"{v:.1%}"
        L += ["", "**D-pos-v2's S lines against the builder's (per term):**", "",
              "| scope | terms | present | exact | within 10% | median abs ln ratio |", "|---|---|---|---|---|---|"]
        sm = d["s_lines"]
        for scope, v in [(k, sm["summary"][k]) for k in ("all", "power_law", "value_dependent")] + \
                [(f"depth {k}", v["all"]) for k, v in sm["by_depth"].items()] + \
                [(f"depth {k}, value-dependent", v["value_dependent"]) for k, v in sm["by_depth"].items()]:
            mal = "–" if v["median_abs_log"] is None else f"{v['median_abs_log']:.3f}"
            L.append(f"| {scope} | {v['n_terms']} | {f(v['present'])} | {f(v['exact'])} | {f(v['lenient'])} | {mal} |")
        ln = d["lines"]["summary"]
        L += ["", f"Line: present {f(ln['line_present'])}, parsed {f(ln['line_parsed'])}, term count right "
                  f"{f(ln['n_terms_ok'])}, levels right in order {f(ln['levels_ok'])}, arithmetic right "
                  f"{f(ln['arithmetic_ok'])}, copy right {f(ln['copy_ok'])} ({ln['copy_checkable']} checkable), "
                  f"σ = 0 while X > 0 {f(ln['sigma_zero_while_x_positive'])}; extra S lines "
                  f"{sm['summary']['extra_s_lines']}."]
    de = res["data_effect"]
    L += ["", "**Data effect (descriptive, expected crps_log):** " + "; ".join(
        f"{k}: " + ("missing" if v.get("missing") else f"{v['value']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
        for k, v in de.items()) + "."]
    for arm, a in res["arms"].items():
        if a:
            tp = a["stage_b_analysis"]["seed0"]["tape"]
            L.append(f"\n{arm}: answered {tp['answered_rate']:.1%}, parse fail {tp['parse_fail_rate']:.1%}, A1 "
                     f"computation-correct {tp['a1_computation_correct_rate']:.1%}, σ = 0 "
                     f"{tp['sigma_zero_share_answered']:.1%}, D59 verdict "
                     f"{a['stage_b_analysis']['seed0']['joint']['labelled_verdict']}.")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["analyze"])
    ap.parse_args()
    print(render(analyze()))


if __name__ == "__main__":
    main()

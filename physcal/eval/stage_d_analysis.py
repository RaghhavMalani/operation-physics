"""Phase 12 stage D (DECISIONS.md D109, pre-registered): the analysis of the explicit-propagation arms, CPU only.

    OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.eval.stage_d_analysis analyze

Each arm (`pos`: the line's numbers and the answer's σ trained; `fmt`: D84's masks; one seed) goes through stage B's
analysis (D104's code, `stage_b_analysis.analyze`, unchanged) on its stated answer σ. Family D (positive control,
Holm over seven): D1-D5 = pos's B1-B5; D6 expected crps_log pos − stage B (per problem, the mean of stage B's three
seeds) > 0; D7 pos − fmt > 0. Family F (ablation, Holm over six): F1-F5 = fmt's B1-B5; F6 fmt − stage B > 0.
Reported, not tested: the propagation line's diagnostics against the line D80's builder writes for the same problem
(copy fidelity, sensitivities, levels, arithmetic), and the line's X used as the σ.
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
from physcal.eval.stats import holm
from physcal.verify.reward import reward_config

ARMS = ("pos", "fmt")
TESTS_D = ("D1", "D2", "D3", "D4", "D5", "D6", "D7")
TESTS_F = ("F1", "F2", "F3", "F4", "F5", "F6")
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
LINE_RE = re.compile(r"^[ \t]*sigma_rel[ \t]*=[ \t]*(.*?)[ \t]*$", re.M)
SQRT_RE = re.compile(rf"^sqrt\((.*)\)\s*=\s*({NUM})$")
TERM_RE = re.compile(rf"^\(\s*({NUM})\s*\*\s*({NUM})\s*\)\s*\^\s*2$")
ZERO_RE = re.compile(r"^[-+]?0*\.?0+$")


def _config() -> dict:
    return load_config("stage_d")


# ------------------------------------------------------------------------------------------- the line
def parse_line(completion: str) -> dict:
    """The last `sigma_rel = ...` line before <answer>: `0`, or `sqrt((a*s)^2 + ...) = X`; else malformed."""
    head = completion.split("<answer>", 1)[0]
    found = LINE_RE.findall(head)
    if not found:
        return {"present": False, "parsed": False}
    rhs = found[-1]
    if ZERO_RE.match(rhs):
        return {"present": True, "parsed": True, "zero": True, "terms": [], "x": 0.0}
    m = SQRT_RE.match(rhs)
    if not m:
        return {"present": True, "parsed": False}
    terms = []
    for part in re.split(r"\s*\+\s*(?=\()", m.group(1).strip()):
        t = TERM_RE.match(part.strip())
        if not t:
            return {"present": True, "parsed": False}
        terms.append((float(t.group(1)), float(t.group(2))))
    return {"present": True, "parsed": True, "zero": False, "terms": terms, "x": float(m.group(2))}


def valid_2sf(shown: float, value: float, rel_slack: float = 0.0) -> bool:
    """`shown` is `value` rounded to 2 significant figures (either way at a rounding boundary); `rel_slack` widens the
    band for an input of `value` that was itself rounded (μ at 4 s.f.: σ was formatted from the unrounded μ)."""
    if value == 0.0:
        return shown == 0.0
    half = 0.5 * 10.0 ** (math.floor(math.log10(abs(value))) - 1)
    return abs(shown - value) <= half * (1 + 1e-6) + (rel_slack + 1e-12) * abs(value)


def mu_rel_slack(p12: dict) -> float:
    """The largest relative rounding error of μ at D80's significant figures (half a unit in the last place)."""
    return 0.5 * 10.0 ** (1 - p12["mu_sig_figs"])


def reference_lines(problems: list[dict], env_cfg: dict, fmt: str) -> list[dict]:
    """Per problem, the line D80's builder writes (sft_phase12.propagation_line at the displayed values), parsed."""
    from physcal.env.generator import problem_from_dict
    from physcal.train.sft_phase12 import propagation_line

    out = []
    for p in problems:
        line, x, exact = propagation_line(problem_from_dict({k: v for k, v in p.items() if k != "_set"}), env_cfg, fmt)
        out.append({**parse_line(f"{line}\n<answer>"), "line": line, "x_exact_first_order": exact})
    return out


def copy_ok(completion: str, x: float, rel_slack: float) -> bool | None:
    """The answer's σ is X·|μ| at 2 significant figures (both in the answer's unit; X = 0 ⇒ σ = 0; `rel_slack` for
    the stated μ's rounding); None when the answer does not parse or states σ in another unit or as a percentage."""
    from physcal.verify.parse import ParseError, parse_completion

    try:
        a = parse_completion(completion).answer
    except ParseError:
        return None
    if a.sigma is None or a.sigma_is_percent or a.sigma_unit not in ("", a.unit):
        return None
    return valid_2sf(a.sigma, x * abs(a.value), rel_slack)


def _share(v: list) -> float | None:
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def line_diagnostics(problems: list[dict], recs: list[dict], refs: list[dict], tol: float,
                     rel_slack: float) -> tuple[dict, np.ndarray]:
    """Copy fidelity, sensitivities, levels, arithmetic and X against the reference line, per arm (frozen-set order).
    Also returns X per problem (unparsed or missing -> 0, the floor's convention) for the line-as-σ source."""
    by = {(r["split"], *B._key(r["problem"])): r for r in recs}
    rows, xs = [], []
    for p, ref in zip(problems, refs):
        r = by.get((p["_set"], *B._key(p)))
        comp = r["completion"] if r else ""
        ln = parse_line(comp)
        b = r["breakdown"] if r else {"reason": "missing"}
        answered = b["reason"] == "ok" and b.get("sigma_si") is not None and (b.get("mu_si") or 0) > 0
        noisy = not ref["zero"]
        row = {"set": p["_set"], "depth": p["depth"], "noisy": noisy, "present": ln["present"], "parsed": ln["parsed"],
               "zero_line": ln.get("zero"), "answered": answered}
        x = ln["x"] if ln["parsed"] else 0.0
        xs.append(x)
        if ln["parsed"] and noisy and not ln["zero"]:
            rt, mt = ref["terms"], ln["terms"]
            row["n_terms_ok"] = len(mt) == len(rt)
            if row["n_terms_ok"]:
                ex = [math.isclose(a, ra, rel_tol=1e-9, abs_tol=0.0) for (a, _), (ra, _) in zip(mt, rt)]
                le = [ra != 0 and abs(a / ra - 1) <= tol or a == ra for (a, _), (ra, _) in zip(mt, rt)]
                row.update(sens_exact=ex, sens_lenient=le,
                           sens_all_exact=all(ex), sens_all_lenient=all(le),
                           sens_multiset=sorted(a for a, _ in mt) == sorted(a for a, _ in rt),
                           levels_ok=all(math.isclose(s, rs, rel_tol=1e-9) for (_, s), (_, rs) in zip(mt, rt)))
            own = math.sqrt(sum((a * s) ** 2 for a, s in mt))
            row["arithmetic_ok"] = valid_2sf(x, own)
            if x > 0:
                row["abs_log_x_ref"] = abs(math.log(x / ref["x"]))
                row["abs_log_x_star"] = abs(math.log(x / p["noise"]["sigma_star_log"]))
        if ln["parsed"] and answered:
            row["copy_ok"] = copy_ok(comp, x, rel_slack)
            srel = b["sigma_si"] / b["mu_si"]
            row["sigma_zero_x_pos"] = srel == 0 and x > 0
            if srel > 0 and x > 0:
                row["abs_log_sigma_x"] = abs(math.log(srel / x))
        rows.append(row)
    return {"summary": summarize_lines(rows),
            "by_depth": {str(d): summarize_lines([r for r in rows if r["depth"] == d])
                         for d in sorted({r["depth"] for r in rows})},
            "by_set": {s: summarize_lines([r for r in rows if r["set"] == s]) for s in B.SETS}}, np.array(xs)


def summarize_lines(rows: list[dict]) -> dict:
    noisy = [r for r in rows if r["noisy"]]
    exact = [r for r in rows if not r["noisy"]]
    parsed_noisy = [r for r in noisy if r["parsed"]]
    nt = [r for r in parsed_noisy if r.get("n_terms_ok")]
    terms = [e for r in nt for e in r["sens_exact"]]
    lenient = [e for r in nt for e in r["sens_lenient"]]
    med = lambda k: float(np.median([r[k] for r in rows if k in r])) if any(k in r for r in rows) else None
    return {
        "n": len(rows), "n_noisy": len(noisy), "line_present": _share([r["present"] for r in rows]),
        "line_parsed": _share([r["parsed"] for r in rows]),
        "malformed": _share([r["present"] and not r["parsed"] for r in rows]),
        "zero_line_on_noisy": _share([r["zero_line"] for r in parsed_noisy]),
        "zero_line_on_exact": _share([r["zero_line"] for r in exact if r["parsed"]]),
        "n_terms_ok": _share([r.get("n_terms_ok") for r in parsed_noisy if not r["zero_line"]]),
        "levels_ok": _share([r["levels_ok"] for r in nt]),
        "sensitivity_term_exact": _share(terms), "sensitivity_term_lenient": _share(lenient),
        "sensitivity_all_exact": _share([r["sens_all_exact"] for r in nt]),
        "sensitivity_all_lenient": _share([r["sens_all_lenient"] for r in nt]),
        "sensitivity_multiset_exact": _share([r["sens_multiset"] for r in nt]),
        "n_problems_terms_ok": len(nt), "n_terms": len(terms),
        "arithmetic_ok": _share([r.get("arithmetic_ok") for r in parsed_noisy]),
        "median_abs_log_x_ref": med("abs_log_x_ref"), "median_abs_log_x_sigma_star": med("abs_log_x_star"),
        "copy_ok": _share([r.get("copy_ok") for r in rows]),
        "copy_checkable": sum(1 for r in rows if r.get("copy_ok") is not None),
        "copy_not_checkable": sum(1 for r in rows if r["parsed"] and r["answered"] and r.get("copy_ok") is None),
        "sigma_zero_while_x_positive": _share([r.get("sigma_zero_x_pos") for r in rows if "sigma_zero_x_pos" in r]),
        "median_abs_log_sigma_over_x": med("abs_log_sigma_x")}


def line_source(problems: list[dict], xs: np.ndarray, terms: list[dict], boot, sc: dict, sa: dict, floor: float,
                ac: dict) -> dict:
    """The line's X as the stated σ, through the comparators' path (no answer: every member counts)."""
    rows = B.source_rows(problems, xs, floor)
    sl = lambda s, reg="sigma_star": a6_pairs.slope_test(a6_pairs.pairs_from_rows(rows[s]), ac["bootstrap"],
                                                         ac["seed"], reg)
    return {"a6_all_pairs": sl("a6_pairs"), "a6b_all_pairs": sl("a6b_pairs", "truth"), "a6c_all_pairs": sl("a6c_pairs"),
            "oracle_centred": oc.summarize(terms, xs, oc.score_source(terms, xs, sc), boot, sc, sa["coverage_k"])}


# ------------------------------------------------------------------------------------------- the analysis
def stage_b_expected(root, problems: list[dict], terms: list[dict], sc: dict, seeds) -> tuple[np.ndarray, list] | None:
    """Stage B's per-problem expected crps_log, the mean over its seeds that have an eval."""
    exp, ran = [], []
    for s in seeds:
        recs = B.read_records(root / f"seed{s}" / "eval")
        if recs is not None:
            exp.append(oc.score_source(terms, B.model_sigma(problems, recs)[0], sc)["expected"])
            ran.append(s)
    return (np.mean(exp, axis=0), ran) if exp else None


def paired(boot, a: np.ndarray | None, b: np.ndarray | None, mask: np.ndarray) -> dict:
    if a is None or b is None:
        return {"missing": True}
    pt, dist = B.boot_mean_dist(boot, a - b, mask)
    return {"value": pt, "ci95": np.percentile(dist, [2.5, 97.5]).tolist(), "p_one_sided": float(np.mean(dist <= 0)),
            "n_problems": int(mask.sum())}


def reading(td: dict, tf: dict, arms: dict) -> list[str]:
    """D109's reading rules."""
    sd = {k: bool(td[k]["holm"]["rejected"]) for k in TESTS_D}
    sf = {k: bool(tf[k]["holm"]["rejected"]) for k in TESTS_F}
    smp = ((arms.get("pos") or {}).get("seed_mean")) or {}
    smf = ((arms.get("fmt") or {}).get("seed_mean")) or {}
    out = []
    if sd["D1"] and sd["D2"] and smp.get("a6b_equivalent_to_zero") and smp.get("a6_label") in B.PROPAGATING:
        out.append(f"trained explicit propagation propagates, the ceiling (D1, D2 supported; A6b equivalent to 0; "
                   f"A6 label {smp['a6_label']})")
    if sd["D3"] and sd["D4"]:
        out.append("it beats the surface readers (D3, D4 supported)")
    if sd["D6"]:
        out.append("it beats stage B's supervised σ* (D6 supported)")
    if sd["D7"]:
        out.append("the trained numbers matter beyond the format (D7 supported)")
    if not sd["D2"]:
        out.append("D2 not supported: trained first-order propagation does not transfer noise allocation to new "
                   "problems with this base and budget (the line diagnostics locate the failure, descriptively)")
    if sf["F1"] and sf["F2"] and smf.get("a6b_equivalent_to_zero") and smf.get("a6_label") in B.PROPAGATING:
        out.append(f"the format alone propagates (F1, F2 supported; A6 label {smf['a6_label']})")
    out.append("3 seeds of D-pos are warranted (D2 or D6 supported; R3-4 D)" if sd["D2"] or sd["D6"] else
               "D stays a one-seed diagnostic (neither D2 nor D6 supported)")
    return out


def analyze(cfg: dict | None = None, root=None) -> dict:
    cfg = cfg or _config()
    ac = cfg["analysis"]
    root = root or REPO_ROOT / cfg["output_dir"]
    bcfg = copy.deepcopy(load_config(ac["stage_b_config"]))
    bcfg["seeds"] = [cfg["seed"]]
    bcfg["analysis"].update(bootstrap=ac["bootstrap"], seed=ac["seed"], alpha=ac["alpha"])
    pc, sa = load_config(bcfg["analysis"]["pairs_config"]), load_config(bcfg["analysis"]["stage_a_config"])
    floor = pc["analysis"]["sigma_floor_rel"]
    sc = reward_config()["scoring"]
    p12 = load_config(cfg["sft_phase12_config"])
    problems = stage_a.load_sets(sa)
    terms = [oc.problem_terms(p, sa["mc"]["n_draws"], sa["mc"]["seed_tag"]) for p in problems]
    mask = np.array([t["draws"] is not None for t in terms])
    boot = oc.ClusterBootstrap([(p["_set"], p["noise"]["pair"]["pair_seed"]) for p in problems], ac["bootstrap"], ac["seed"])
    refs = reference_lines(problems, load_config(p12["env_config"]), p12["sigma_format"])
    seed = cfg["seed"]

    arms, expected, lines = {}, {}, {}
    for a in ARMS:
        recs = B.read_records(root / a / f"seed{seed}" / "eval")
        if recs is None:
            arms[a] = None
            continue
        r = B.analyze(bcfg, root / a)
        diag, xs = line_diagnostics(problems, recs, refs, ac["sensitivity_rel_tol"], mu_rel_slack(p12))
        arms[a] = {"tests": r["tests"], "seed_mean": r["seed_mean"], "seed0": r["seeds"][seed],
                   "comparators": r["comparators"], "lines": diag,
                   "line_as_sigma": line_source(problems, xs, terms, boot, sc, sa, floor, ac)}
        expected[a] = oc.score_source(terms, B.model_sigma(problems, recs)[0], sc)["expected"]
    sb = stage_b_expected(REPO_ROOT / ac["stage_b_root"], problems, terms, sc, load_config(ac["stage_b_config"])["seeds"])
    sb_exp = sb[0] if sb else None

    td = {f"D{i}": (copy.deepcopy(arms["pos"]["tests"][f"B{i}"]) if arms["pos"] else {"missing": True})
          for i in range(1, 6)}
    td["D6"] = paired(boot, expected.get("pos"), sb_exp, mask)
    td["D7"] = paired(boot, expected.get("pos"), expected.get("fmt"), mask)
    tf = {f"F{i}": (copy.deepcopy(arms["fmt"]["tests"][f"B{i}"]) if arms["fmt"] else {"missing": True})
          for i in range(1, 6)}
    tf["F6"] = paired(boot, expected.get("fmt"), sb_exp, mask)
    for fam, keys in ((td, TESTS_D), (tf, TESTS_F)):
        for k in keys:
            fam[k] = fam[k] if fam[k] and not fam[k].get("missing") else {"missing": True}
        h = holm({k: (1.0 if fam[k].get("missing") else fam[k]["p_one_sided"]) for k in keys}, ac["alpha"])
        for k in keys:
            fam[k]["holm"] = h[k]
    ref_diag = summarize_lines([{"set": p["_set"], "depth": p["depth"], "noisy": not r["zero"], "present": True,
                                 "parsed": True, "zero_line": r["zero"], "answered": False} for p, r in zip(problems, refs)])
    res = {"decision": "D109 (Phase 12 stage D, pre-registered)", "tests_d": td, "tests_f": tf,
           "reading": reading(td, tf, arms), "arms": arms,
           "comparators": next((v["comparators"] for v in arms.values() if v), None),
           "stage_b": None if sb is None else {
               "seeds": sb[1], "expected_crps_log_mean": float(np.mean(sb_exp[mask])),
               "tests": json.loads((REPO_ROOT / ac["stage_b_root"] / "results.json").read_text())["tests"]
               if (REPO_ROOT / ac["stage_b_root"] / "results.json").exists() else None},
           "reference_line": {"n_problems": len(problems), "zero_line_share": ref_diag["zero_line_on_exact"],
                              "n_noisy": ref_diag["n_noisy"],
                              "median_abs_log_x_sigma_star": float(np.median(
                                  [abs(math.log(r["x"] / p["noise"]["sigma_star_log"])) for p, r in zip(problems, refs)
                                   if not r["zero"] and r["x"] > 0]))}}
    root.mkdir(parents=True, exist_ok=True)
    (root / "results.json").write_text(json.dumps(res, indent=1, default=B._json_default))
    (root / "summary.md").write_text(render(res))
    return res


# ------------------------------------------------------------------------------------------- report
def render(res: dict) -> str:
    names = {"1": "A6 slope > 0 (computation-correct pairs)", "2": "A6c slope > 0 (all answered pairs)",
             "3": "expected crps_log − RSS > 0", "4": "expected crps_log − X1 > 0", "5": "A6 slope − X1's > 0",
             "6": "expected crps_log − stage B (3-seed mean) > 0", "7": "expected crps_log, pos − fmt > 0"}
    L = ["# Phase 12 stage D (D109): explicit propagation, positive control (pos) and format-only ablation (fmt)", "",
         "**Reading (D109):** " + "; ".join(res["reading"]) + ".", "",
         "One seed per arm: CIs from a bootstrap over pairs (B = 10,000), problem-level evidence only.", ""]
    for title, t, keys in (("Family D (positive control; Holm over seven)", res["tests_d"], TESTS_D),
                           ("Family F (format-only ablation; Holm over six)", res["tests_f"], TESTS_F)):
        L += [f"**{title}**", "", "| test | estimate [95% CI] | p (one-sided) | Holm |", "|---|---|---|---|"]
        for k in keys:
            d = t[k]
            label = f"{k}: {names[k[1]]}"
            if d.get("missing"):
                L.append(f"| {label} | missing | 1 | not supported |")
                continue
            L.append(f"| {label} | {B._ci(d)} | {d['p_one_sided']:.4g} | "
                     f"{'supported' if d['holm']['rejected'] else 'not supported'} |")
        L.append("")
    L += ["| σ source | A6 slope (label) | A6b slope | A6c slope | expected crps_log | median ln(σ/σ*) | median abs | "
          "vol beta | 1σ / 2σ cov. |", "|---|---|---|---|---|---|---|---|---|"]

    def oc_cells(o):
        return (f"{o['expected'][0]:.4f} | {o['median_log_ratio']:+.3f} | {o['median_abs_log_ratio']:.3f} | "
                f"{o['vol_beta'][0]:.3f} | {o['coverage_1'][0]:.1%} / {o['coverage_2'][0]:.1%}")
    sl = lambda d: "–" if d.get("slope") is None else f"{d['slope']:+.3f}"
    for a, r in res["arms"].items():
        if r is None:
            L.append(f"| {a}: answer σ | not run | | | | | | | |")
            continue
        s = r["seed0"]
        L.append(f"| {a}: answer σ | {sl(s['a6']['primary_computation_correct'])} (n "
                 f"{s['a6']['primary_computation_correct']['n_pairs']}, {s['a6']['label']}) | "
                 f"{sl(s['a6b']['primary_computation_correct'])} | {sl(s['a6c']['primary_all_answered'])} | "
                 f"{oc_cells(s['oracle_centred'])} |")
        x = r["line_as_sigma"]
        L.append(f"| {a}: the line's X (all pairs) | {sl(x['a6_all_pairs'])} | {sl(x['a6b_all_pairs'])} | "
                 f"{sl(x['a6c_all_pairs'])} | {oc_cells(x['oracle_centred'])} |")
    for n, r in (res["comparators"] or {}).items():
        L.append(f"| {n} (all pairs) | {sl(r['a6_all_pairs'])} | {sl(r['a6b_all_pairs'])} | {sl(r['a6c_all_pairs'])} | "
                 f"{oc_cells(r['oracle_centred'])} |")
    if res["stage_b"]:
        L += ["", f"Stage B (D104, seeds {res['stage_b']['seeds']}): mean expected crps_log "
                  f"{res['stage_b']['expected_crps_log_mean']:.4f} (D6 / F6's per-problem reference)."]
    f = lambda v: "–" if v is None else f"{v:.1%}"
    g = lambda v: "–" if v is None else f"{v:.3f}"
    L += ["", "**The propagation line (reported, not tested; against the line D80's builder writes for the same "
              "problem):**", "",
          "| arm | present / parsed | zero line: noisy / exact | terms ok | levels ok | sensitivities exact: per term "
          "/ all (lenient) | arithmetic ok | median abs ln(X/X_ref) | copy ok (n) | σ = 0 while X > 0 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for a, r in res["arms"].items():
        if r is None:
            continue
        d = r["lines"]["summary"]
        L.append(f"| {a} | {f(d['line_present'])} / {f(d['line_parsed'])} | {f(d['zero_line_on_noisy'])} / "
                 f"{f(d['zero_line_on_exact'])} | {f(d['n_terms_ok'])} | {f(d['levels_ok'])} | "
                 f"{f(d['sensitivity_term_exact'])} / {f(d['sensitivity_all_exact'])} "
                 f"({f(d['sensitivity_all_lenient'])}) | {f(d['arithmetic_ok'])} | {g(d['median_abs_log_x_ref'])} | "
                 f"{f(d['copy_ok'])} ({d['copy_checkable']}) | {f(d['sigma_zero_while_x_positive'])} |")
    for a, r in res["arms"].items():
        if r is None:
            continue
        L += ["", f"{a}, by depth (sensitivities all exact / copy ok): " + "; ".join(
            f"depth {k}: {f(v['sensitivity_all_exact'])} / {f(v['copy_ok'])}" for k, v in r["lines"]["by_depth"].items())]
    ran = [a for a, r in res["arms"].items() if r]
    if ran:
        L += ["", "| arm | steps | train loss | answered | parse fail | unit mismatch | A1 comp.-correct | σ = 0 | "
                  "confidence block | D59 verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
        for a in ran:
            s = res["arms"][a]["seed0"]
            d, tr = s["tape"], s["training"]
            L.append(f"| {a} | {tr.get('steps')} ({tr.get('steps_planned')}) | "
                     f"{'–' if tr.get('train_loss') is None else format(tr['train_loss'], '.3f')} | "
                     f"{f(d['answered_rate'])} | {f(d['parse_fail_rate'])} | {f(d['unit_mismatch_rate'])} | "
                     f"{f(d['a1_computation_correct_rate'])} | {f(d['sigma_zero_share_answered'])} | "
                     f"{f(d['confidence_parse_rate'])} | {s['joint']['labelled_verdict']} |")
    rl = res["reference_line"]
    L += ["", f"Reference line on the eval sets: {rl['n_noisy']} noisy of {rl['n_problems']} problems; median "
              f"|ln(X_ref / σ*)| = {rl['median_abs_log_x_sigma_star']:.3f}."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["analyze"])
    ap.parse_args()
    print(render(analyze()))


if __name__ == "__main__":
    main()

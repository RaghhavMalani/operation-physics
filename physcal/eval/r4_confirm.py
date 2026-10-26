"""R4-3's confirmatory run (DECISIONS.md D118, pre-registered): D-pos-v2 and B-v2, seeds 0-2, on the frozen R4-5 set.
CPU only; run once, after all six confirmatory evals exist (D118 §10).

    PHYSCAL_CONFIRMATORY=1 OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.eval.r4_confirm analyze

The σ values are D113 §3's: (a) the answer's σ_rel for both arms; (b) the line's X, (c) the exact combination of the
line's own terms and (c_S) for D-pos-v2. Seed-mean statistics use stage B's joint bootstrap over pairs (D104: every
seed's statistic recomputed on the same resampled pairs; B = 10,000, seed 0). P (primary; D118 §6 as amended by
D125): K1 (D-pos-v2 (a)'s A6c seed-mean slope > 0, Holm-adjusted p < 0.05), C2 (its point estimate ≥ 0.5) and C3
(its A6b seed-mean slope's 90% CI inside D59's margin), i.e. D113's full R4-3 criterion. K1-K8 are D113's R1-R8 at the
3-seed mean, Holm over all eight. The pre-specified split (training depths 1-4 vs the held-out depth 5) repeats every
K statistic on each subset, outside the Holm family. Comparators (σ*, RSS, X1) come from the confirmatory files; X1
from D99's refit (`x1_a6c confirmatory`), used only if its reproduction passed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.eval import oracle_centred as oc
from physcal.eval import r4_analysis as R
from physcal.eval import stage_a
from physcal.eval import stage_b_analysis as B
from physcal.eval import stage_d_analysis as Dn
from physcal.eval.stats import holm
from physcal.verify.reward import reward_config

ARMS = ("dpos2", "b2")
SOURCES = {"dpos2": ("a", "b", "c", "c_S"), "b2": ("a",)}
TESTS = tuple(f"K{i}" for i in range(1, 9))
TEST_NAMES = {"K1": "D-pos-v2 (a) answer σ: A6c seed-mean slope > 0 (primary)",
              "K2": "D-pos-v2 (b) line X: A6c seed-mean slope > 0", "K3": "D-pos-v2 (c) exact from terms: A6c slope > 0",
              "K4": "B-v2: A6c seed-mean slope > 0", "K5": "expected crps_log, D-pos-v2 − B-v2 > 0",
              "K6": "D-pos-v2 − RSS > 0 (beats RSS)", "K7": "D-pos-v2 − X1 > 0 (beats X1)", "K8": "B-v2 − RSS > 0"}
SUBSETS = {"depths_1_4": (1, 2, 3, 4), "depth_5": (5,)}


def _config() -> dict:
    return load_config("stage_r4")


def _guard() -> None:
    from physcal.eval.run_eval import CONFIRMATORY_ENV

    if os.environ.get(CONFIRMATORY_ENV) != "1":
        raise PermissionError(f"the confirmatory set is frozen (D113); set {CONFIRMATORY_ENV}=1 (D118 §10)")


def stage_a_config(cfg: dict | None = None) -> dict:
    """Stage A's config pointed at the frozen R4-5 files and X1's confirmatory predictions."""
    cc = (cfg or _config())["confirmatory"]
    sa = copy.deepcopy(load_config("stage_a"))
    sa["generator_version"] = f"{sa['generator_version']}/{cc['set_subdir']}"
    sa["alternatives"].update(x1_predictions=cc["x1_predictions"], x1_reproduction=cc["x1_reproduction"])
    return sa


# ------------------------------------------------------------------------------------------- per seed
def seed_sources(problems: list[dict], recs: list[dict], arm: str, refs: list[dict], dcfg: dict, ac: dict) -> dict:
    """One seed's σ per source (problem order), its verifier breakdowns, and D-pos-v2's line diagnostics."""
    sig_a, bds = B.model_sigma(problems, recs)
    out = {"sigma": {"a": sig_a}, "bds": bds}
    if arm == "dpos2":
        diag, xs_b = Dn.line_diagnostics(problems, recs, refs, ac["sensitivity_rel_tol"], Dn.mu_rel_slack(dcfg))
        by = {(x["split"], *B._key(x["problem"])): x for x in recs}
        lines = [Dn.parse_line(by[(p["_set"], *B._key(p))]["completion"]) if (p["_set"], *B._key(p)) in by
                 else {"present": False, "parsed": False} for p in problems]
        srows, xs_cs = R.s_line_rows(problems, recs, refs, ac["sensitivity_rel_tol"])
        out["sigma"].update(b=xs_b, c=np.array([R.exact_from_terms(ln) for ln in lines]), c_S=xs_cs)
        out["lines"], out["s_lines"] = diag, R.s_diagnostics(srows)
    return out


def _rows(problems: list[dict], source: str, sigma: np.ndarray, bds: list[dict], pc: dict, floor: float) -> dict:
    """(a) through the verifier's breakdowns (answered / computation-correct); a line source on every pair."""
    return B.model_rows(problems, bds, pc) if source == "a" else B.source_rows(problems, sigma, floor)


# ------------------------------------------------------------------------------------------- seed-mean statistics
def seed_mean_slope(rows_by_seed: dict, set_name: str, mask: str, ac: dict, regressor: str = "x") -> dict | None:
    if not rows_by_seed:
        return None
    arr = B.PairArrays(next(iter(rows_by_seed.values()))[set_name])
    if not arr.keys:
        return None
    C = oc.ClusterBootstrap(arr.keys, ac["bootstrap"], ac["seed"]).counts
    per = {s: arr.of(r[set_name]) for s, r in rows_by_seed.items()}
    out = B.seed_mean_slope(arr, per, mask, C, regressor)
    vals = [v["value"] for v in out["per_seed"].values()]
    out["seed_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
    return out


def seed_mean_diff(boot: oc.ClusterBootstrap, a, b, mask: np.ndarray) -> dict:
    """Mean over seeds of the per-problem expected crps_log of a, minus b's (a dict of seeds or one array)."""
    mean = lambda v: None if v is None or (isinstance(v, dict) and not v) else (
        np.mean([v[s] for s in sorted(v)], axis=0) if isinstance(v, dict) else v)
    ma, mb = mean(a), mean(b)
    if ma is None or mb is None:
        return {"missing": True}
    pt, dist = B.boot_mean_dist(boot, ma - mb, mask)
    out = {"value": pt, "ci95": np.percentile(dist, [2.5, 97.5]).tolist(), "ci90": np.percentile(dist, [5, 95]).tolist(),
           "p_one_sided": float(np.mean(dist <= 0)), "n_problems": int(mask.sum())}
    if isinstance(a, dict):
        out["per_seed"] = {s: float(np.mean(((a[s] - (b[s] if isinstance(b, dict) else mb)))[mask])) for s in sorted(a)}
    return out


def k_statistics(idx: np.ndarray, data: dict, ac: dict, pc: dict, floor: float, margin: float) -> dict:
    """K1-K8, C2, C3 and the criteria labels on the problems `idx` (a whole-pair subset)."""
    problems = [data["problems"][i] for i in idx]
    terms = [data["terms"][i] for i in idx]
    has = np.array([t["draws"] is not None for t in terms])
    boot = oc.ClusterBootstrap([(p["_set"], p["noise"]["pair"]["pair_seed"]) for p in problems], ac["bootstrap"],
                               ac["seed"])
    slopes, exp = {}, {}
    for arm in ARMS:
        for src in SOURCES[arm]:
            seeds = data["arms"][arm]
            rows = {s: _rows(problems, src, d["sigma"][src][idx], [d["bds"][i] for i in idx], pc, floor)
                    for s, d in seeds.items()}
            a6c_mask = "ok"
            a6b_mask = "cc" if src == "a" else "ok"
            slopes[(arm, src)] = {"a6c": seed_mean_slope(rows, "a6c_pairs", a6c_mask, ac),
                                  "a6b": seed_mean_slope(rows, "a6b_pairs", a6b_mask, ac, "m"),
                                  "a6": seed_mean_slope(rows, "a6_pairs", "cc", ac)}
            exp[(arm, src)] = {s: d["expected"][src][idx] for s, d in seeds.items()}
    comp = {n: v[idx] for n, v in data["comparator_expected"].items()}
    k = {}
    for name, key in (("K1", ("dpos2", "a")), ("K2", ("dpos2", "b")), ("K3", ("dpos2", "c")), ("K4", ("b2", "a"))):
        s = slopes.get(key, {}).get("a6c")
        k[name] = {"missing": True} if s is None else s
    k["K5"] = seed_mean_diff(boot, exp.get(("dpos2", "a")), exp.get(("b2", "a")), has)
    k["K6"] = seed_mean_diff(boot, exp.get(("dpos2", "a")), comp.get("rss"), has)
    k["K7"] = seed_mean_diff(boot, exp.get(("dpos2", "a")), comp.get("x1"), has)
    k["K8"] = seed_mean_diff(boot, exp.get(("b2", "a")), comp.get("rss"), has)
    crit = {}
    for key, sl in slopes.items():
        c, b = sl["a6c"], sl["a6b"]
        ok = c is not None and b is not None
        crit[f"{key[0]}_{key[1]}"] = {
            "C1": bool(ok and c["ci90"][0] > 0), "C2": bool(ok and c["value"] >= ac["practical_slope"]),
            "C3": bool(ok and -margin < b["ci90"][0] and b["ci90"][1] < margin),
            "a6c": c, "a6b": b, "a6": sl["a6"]}
        crit[f"{key[0]}_{key[1]}"]["meets_full_d113_criterion"] = all(crit[f"{key[0]}_{key[1]}"][x] for x in ("C1", "C2", "C3"))
    return {"n_problems": len(problems), "n_pairs": boot.k, "tests": k, "criteria": crit}


def regret(data: dict, arm: str = "dpos2", top: float = 0.05) -> dict:
    """D118 §9 (pre-specified after FINDINGS #30): per seed, the share of (a)'s expected-score regret against σ* in its
    worst `top` of problems, and the mean regret by depth."""
    star = data["comparator_expected"]["sigma_star"]
    depth = np.array([p["depth"] for p in data["problems"]])
    out = {}
    for s, d in data["arms"].get(arm, {}).items():
        r = star - d["expected"]["a"]
        ok = np.isfinite(r)
        r, dp = r[ok], depth[ok]
        srt = np.sort(r)[::-1]
        n = max(1, int(round(top * len(srt))))
        out[s] = {"share_in_worst": float(srt[:n].sum() / srt.sum()) if srt.sum() > 0 else None,
                  "mean_by_depth": {str(k): float(r[dp == k].mean()) for k in sorted(set(dp))}}
    return out


# ------------------------------------------------------------------------------------------- the analysis
def analyze(cfg: dict | None = None, root=None, allow_incomplete: bool = False) -> dict:
    _guard()
    cfg = cfg or _config()
    ac, cc = cfg["analysis"], cfg["confirmatory"]
    root = root or REPO_ROOT / cfg["output_dir"]
    sa = stage_a_config(cfg)
    pc = load_config(load_config(ac["stage_b_config"])["analysis"]["pairs_config"])
    floor, margin = pc["analysis"]["sigma_floor_rel"], pc["invariance_analysis"]["equivalence_margin"]
    sc = reward_config()["scoring"]
    dcfg = load_config(cfg["sft_phase12_config"])
    problems = stage_a.load_sets(sa)
    recs = {arm: {s: B.read_records(root / arm / f"seed{s}" / cc["output_subdir"]) for s in cc["seeds"]} for arm in ARMS}
    missing = [f"{arm} seed {s}" for arm in ARMS for s in cc["seeds"] if recs[arm][s] is None]
    if missing and not allow_incomplete:
        raise RuntimeError(f"D118 §10: the analysis runs once all six evals exist; missing {missing}")
    terms = [oc.problem_terms(p, sa["mc"]["n_draws"], sa["mc"]["seed_tag"]) for p in problems]
    refs = R.references(problems, dcfg)
    types = R.pair_types(problems, refs)
    src, note = stage_a.sources(problems, sa)
    bcfg = load_config(ac["stage_b_config"])["analysis"]
    comps = {"sigma_star": src[stage_a.STAR], "rss": src[bcfg["rss_source"]]}
    if bcfg["x1_source"] in src:
        comps["x1"] = src[bcfg["x1_source"]]
    data = {"problems": problems, "terms": terms, "arms": {},
            "comparator_expected": {n: oc.score_source(terms, s, sc)["expected"] for n, s in comps.items()}}
    per_seed_desc: dict = {}
    for arm in ARMS:
        data["arms"][arm] = {}
        for s, r in recs[arm].items():
            if r is None:
                continue
            d = seed_sources(problems, r, arm, refs, dcfg, ac)
            d["expected"] = {k: oc.score_source(terms, v, sc)["expected"] for k, v in d["sigma"].items()}
            data["arms"][arm][s] = d
            per_seed_desc.setdefault(arm, {})[s] = {
                "tape": B.describe_tape(problems, r, d["bds"], pc["analysis"]["tau"], dcfg["confidence"]["bins"]),
                **({"lines": d["lines"], "s_lines": d["s_lines"]} if arm == "dpos2" else {})}
    n_seeds = {arm: len(data["arms"][arm]) for arm in ARMS}
    all_idx = np.arange(len(problems))
    res: dict = {"decision": "D118 (R4-3 confirmatory run, pre-registered)", "n_problems": len(problems),
                 "seeds": {arm: sorted(data["arms"][arm]) for arm in ARMS}, "sources_note": note,
                 "complete": not missing, "label": "complete" if not missing else
                 f"incomplete ({', '.join(f'{a} {n} seeds' for a, n in n_seeds.items())}); still the confirmatory analysis",
                 "bootstrap": ac["bootstrap"]}
    main = k_statistics(all_idx, data, ac, pc, floor, margin)
    h = holm({k: (1.0 if main["tests"][k].get("missing") else main["tests"][k]["p_one_sided"]) for k in TESTS},
             ac["alpha"])
    for k in TESTS:
        main["tests"][k]["holm"] = h[k]
    res["main"] = main
    k1 = main["tests"]["K1"]
    flags = {"K1": bool(h["K1"]["rejected"]),
             "C2": bool(not k1.get("missing") and k1["value"] >= ac["practical_slope"]),
             "C3": bool(main["criteria"].get("dpos2_a", {}).get("C3", False))}
    res["primary"] = {**flags, "K1_holm_rejected": flags["K1"], "criteria": list(cc["primary_criteria"]),
                      "confirmed": all(flags[c] for c in cc["primary_criteria"]),
                      "rule": "P confirmed iff " + " and ".join(cc["primary_criteria"]) + " (D118 §6 as amended by "
                              "D125: K1 Holm-rejected, C2 seed-mean A6c slope of (a) >= 0.5, C3 its A6b seed-mean 90% "
                              "CI inside D59's margin)",
                      "meets_full_d113_criterion": main["criteria"].get("dpos2_a", {}).get("meets_full_d113_criterion")}
    depth = np.array([p["depth"] for p in problems])
    res["subsets"] = {name: k_statistics(all_idx[np.isin(depth, ds)], data, ac, pc, floor, margin)
                      for name, ds in SUBSETS.items()}
    res["by_depth"] = {str(d): {f"{a}_{s}": k_statistics(all_idx[depth == d], data, ac, pc, floor, margin)["criteria"]
                                .get(f"{a}_{s}", {}).get("a6c") for a in ARMS for s in SOURCES[a]} for d in sorted(set(depth))}
    pair_type = np.array([types.get(R._pair_key(p)) for p in problems])  # both members share the pair's key
    res["by_type"] = {t: {f"{a}_{s}": k_statistics(all_idx[pair_type == t], data, ac, pc, floor, margin)["criteria"]
                          .get(f"{a}_{s}", {}).get("a6c") for a in ARMS for s in SOURCES[a]}
                      for t in ("power_law", "value_dependent") if (pair_type == t).any()}
    res["regret"] = regret(data)
    res["per_seed"] = per_seed_desc
    out = REPO_ROOT / cc["analysis_output"]
    out.mkdir(parents=True, exist_ok=True)
    stage_b = {}
    bfull = copy.deepcopy(load_config(ac["stage_b_config"]))
    bfull["analysis"].update(bootstrap=ac["bootstrap"], seed=ac["seed"], alpha=ac["alpha"])
    for arm in ARMS:
        bfull["seeds"] = sorted(data["arms"][arm])
        try:
            r = B.analyze(bfull, root / arm, sa=sa, eval_subdir=cc["output_subdir"], out_dir=out / f"stage_b_{arm}")
            stage_b[arm] = {"tests": r["tests"], "seed_mean": r["seed_mean"]}
        except RuntimeError as exc:  # X1 unavailable: stage B's analysis needs both comparators
            stage_b[arm] = {"error": str(exc)}
    res["stage_b_analysis"] = stage_b
    (out / "results.json").write_text(json.dumps(res, indent=1, default=B._json_default))
    (out / "summary.md").write_text(render(res))
    return res


# ------------------------------------------------------------------------------------------- the summary
def _ci(d: dict | None, key: str = "ci90", fmt: str = "+.3f") -> str:
    if not d or d.get("missing") or d.get("value") is None:
        return "–"
    lo, hi = d[key]
    return f"{d['value']:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"


def render(res: dict) -> str:
    pr = res["primary"]
    m = res["main"]
    L = ["# R4-3 confirmatory run: D-pos-v2 and B-v2, 3 seeds, frozen R4-5 set (D118)", "",
         f"Seeds: D-pos-v2 {res['seeds']['dpos2']}, B-v2 {res['seeds']['b2']} ({res['label']}). Problems "
         f"{res['n_problems']}. Bootstrap: stage B's joint pairs bootstrap, B = {res['bootstrap']}. Comparators: "
         f"{res['sources_note']}.", "",
         f"**Primary hypothesis P (D-pos-v2's answer σ propagates on A6c at the 3-seed mean): "
         f"{'CONFIRMED' if pr['confirmed'] else 'NOT CONFIRMED'}** (D125: K1 Holm-rejected: {pr['K1']}; C2, "
         f"seed-mean A6c slope ≥ 0.5: {pr['C2']}; C3, A6b seed-mean 90% CI inside ±0.3: {pr['C3']}). "
         f"D113's full criterion with C1's unadjusted 90% bound: {pr['meets_full_d113_criterion']}.", "",
         "| test | seed-mean value [90% CI] | 95% CI | p (one-sided) | Holm | per seed |", "|---|---|---|---|---|---|"]
    for k in TESTS:
        t = m["tests"][k]
        if t.get("missing"):
            L.append(f"| {k}: {TEST_NAMES[k]} | missing | | | | |")
            continue
        per = ", ".join(f"{s}: {v['value'] if isinstance(v, dict) else v:+.4f}" for s, v in (t.get("per_seed") or {}).items())
        L.append(f"| {k}: {TEST_NAMES[k]} | {_ci(t, fmt='+.4f')} | [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] "
                 f"| {t['p_one_sided']:.4g} | {'rejected' if t['holm']['rejected'] else '–'} | {per} |")
    L += ["", "## The pre-specified split (training depths 1–4 vs the held-out depth 5; outside the Holm family)", "",
          "| test | all | depths 1–4 | depth 5 |", "|---|---|---|---|"]
    for k in TESTS:
        L.append(f"| {k} | {_ci(m['tests'][k], fmt='+.4f')} | " +
                 " | ".join(_ci(res["subsets"][n]["tests"][k], fmt="+.4f") for n in SUBSETS) + " |")
    L += ["", "## Criteria per σ source (seed means)", "",
          "| source | A6c slope [90% CI] | C1 | C2 | C3 (A6b in ±0.3) | A6b slope | A6 slope (cc) | full D113 |",
          "|---|---|---|---|---|---|---|---|"]
    for name, c in m["criteria"].items():
        L.append(f"| {name} | {_ci(c['a6c'])} | {c['C1']} | {c['C2']} | {c['C3']} | {_ci(c['a6b'])} | {_ci(c['a6'])} "
                 f"| {c['meets_full_d113_criterion']} |")
    L += ["", "## A6c seed-mean slope by depth (depth 5 held out from training)", "",
          "| source | " + " | ".join(f"d{d}" for d in res["by_depth"]) + " |", "|---|" + "---|" * len(res["by_depth"])]
    for src in next(iter(res["by_depth"].values())):
        L.append(f"| {src} | " + " | ".join(_ci(res["by_depth"][d].get(src)) for d in res["by_depth"]) + " |")
    if res["by_type"]:
        L += ["", "| source | " + " | ".join(res["by_type"]) + " |", "|---|" + "---|" * len(res["by_type"])]
        for src in next(iter(res["by_type"].values())):
            L.append(f"| {src} | " + " | ".join(_ci(res["by_type"][t].get(src)) for t in res["by_type"]) + " |")
    L += ["", "## Regret concentration (D-pos-v2 (a), per seed; pre-specified in D118 §9)", ""]
    for s, r in res["regret"].items():
        share = "–" if r["share_in_worst"] is None else f"{r['share_in_worst']:.0%}"
        L.append(f"- seed {s}: worst 5% hold {share} of the regret; by depth " +
                 ", ".join(f"{d}: {v:.4f}" for d, v in r["mean_by_depth"].items()))
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["analyze"])
    ap.add_argument("--allow-incomplete", action="store_true", help="D118 §10: only if an eval is BLOCKED for good")
    args = ap.parse_args()
    analyze(allow_incomplete=args.allow_incomplete)
    print((REPO_ROOT / _config()["confirmatory"]["analysis_output"] / "summary.md").read_text())


if __name__ == "__main__":
    main()

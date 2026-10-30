"""R5-6 part J (DECISIONS.md D147, pre-registered): D-pos-v2 and B-v2, seeds 0-2, on the frozen J set (Jacobians
outside the training range of trained chains). CPU only; run once, after all six J evals exist (D147 §11).

    PHYSCAL_CONFIRMATORY=1 OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.eval.r5_6 analyze

R4-3's confirmatory machinery (`r4_confirm`) on the J files: the same σ sources (D113 §3), the same joint bootstrap
over pairs (D118 §5), the same criteria. The J sensitivity pairs (`a6c_pairs`) take A6c's place and the J invariance
pairs (`a6b_pairs`) A6b's.
- **P_J (D147 §8):** J1 (D-pos-v2 (a)'s seed-mean J-sensitivity slope > 0, Holm over J1-J6) and C2 (its point
  estimate ≥ 0.5) and C3 (its J-invariance seed-mean 90% CI inside ±0.3, computation-correct pairs).
- **J2:** the explicit − implicit seed-mean slope, both arms on the same resamples. **J3-J6:** expected crps_log
  against RSS and X1 (D147 §9).
- **Descriptive (D147 §10):** the distances per pair, the clamped comparator, the splits, D-pos-v2's S lines on the
  out-of-range terms, and R4-3's depths 1-4 slopes as context.
"""

from __future__ import annotations

import argparse
import copy
import json

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.env import noise as noise_mode
from physcal.eval import oracle_centred as oc
from physcal.eval import r4_analysis as R
from physcal.eval import r4_confirm as RC
from physcal.eval import stage_a
from physcal.eval import stage_b_analysis as B
from physcal.eval.stats import holm
from physcal.verify.reward import reward_config

ARMS, SOURCES = RC.ARMS, RC.SOURCES
TESTS = tuple(f"J{i}" for i in range(1, 7))
TEST_NAMES = {"J1": "D-pos-v2 (a) answer σ: J-sensitivity seed-mean slope > 0 (primary)",
              "J2": "explicit − implicit: D-pos-v2 (a) slope − B-v2 slope > 0",
              "J3": "D-pos-v2 − RSS > 0 (expected crps_log)", "J4": "D-pos-v2 − X1 > 0 (expected crps_log)",
              "J5": "B-v2 − RSS > 0 (expected crps_log)", "J6": "B-v2 − X1 > 0 (expected crps_log)"}
COMPARATORS = ("sigma_star", "rss", "x1", "clamped")


def _config() -> dict:
    return load_config("r5_6")


def stage_a_config(cfg: dict | None = None) -> dict:
    """Stage A's config pointed at the frozen J files (A6b / A6c layout) and X1's J predictions."""
    cfg = cfg or _config()
    sa = copy.deepcopy(load_config("stage_a"))
    sa["generator_version"] = f"{sa['generator_version']}/{cfg['set_subdir']}"
    sa["pair_sets"] = list(cfg["pair_sets"])
    sa["alternatives"].update(x1_predictions=cfg["x1_predictions"], x1_reproduction=cfg["x1_reproduction"])
    return sa


def clamped_sigma(problems: list[dict]) -> np.ndarray:
    """D147 §10's clamped comparator: σ_δ with every sensitivity clipped into the chain's training range, at the
    member's own levels (NaN for a problem without J metadata)."""
    out = []
    for p in problems:
        j = p["noise"]["pair"].get("j")
        lv = {g: v["rel"] for g, v in p["noise"]["givens"].items()}
        out.append(np.nan if j is None else noise_mode.delta_sigma_log(j["clamped"], lv))
    return np.array(out)


# ------------------------------------------------------------------------------------------- the statistics
def _a_rows(data: dict, arm: str, idx: np.ndarray, pc: dict, floor: float) -> dict:
    problems = [data["problems"][i] for i in idx]
    return {s: RC._rows(problems, "a", d["sigma"]["a"][idx], [d["bds"][i] for i in idx], pc, floor)
            for s, d in data["arms"][arm].items()}


def slope_difference(data: dict, idx: np.ndarray, ac: dict, pc: dict, floor: float) -> dict:
    """J2: seed-mean slope of D-pos-v2 (a) minus B-v2's on the J sensitivity pairs; each arm on its own answered pairs,
    both recomputed on the same resampled pairs (the count matrix J1's seed_mean_slope uses)."""
    rows = {arm: _a_rows(data, arm, idx, pc, floor) for arm in ARMS}
    if not all(rows.values()):
        return {"missing": True}
    arr = B.PairArrays(next(iter(rows["dpos2"].values()))["a6c_pairs"])
    if not arr.keys:
        return {"missing": True}
    C = oc.ClusterBootstrap(arr.keys, ac["bootstrap"], ac["seed"]).counts
    one = np.ones((1, len(arr.keys)))

    def arm_stat(by_seed: dict) -> tuple[float, np.ndarray]:
        pts, boots = [], []
        for r in by_seed.values():
            d = arr.of(r["a6c_pairs"])
            pts.append(B.wslope(one, d["ok"], arr.x, d["y"])[0])
            boots.append(B.wslope(C, d["ok"], arr.x, d["y"]))
        return float(np.mean(pts)), np.nanmean(np.array(boots), axis=0)

    (pe, be), (pi, bi) = arm_stat(rows["dpos2"]), arm_stat(rows["b2"])
    dist = be - bi
    return {"value": pe - pi, "explicit": pe, "implicit": pi, "ci95": np.nanpercentile(dist, [2.5, 97.5]).tolist(),
            "ci90": np.nanpercentile(dist, [5, 95]).tolist(), "p_one_sided": float(np.mean(~(dist > 0))),
            "n_pairs": len(arr.keys)}


def comparator_stats(data: dict, idx: np.ndarray, ac: dict, floor: float) -> dict:
    """Each comparator's slopes (J sensitivity on Δ ln σ*, J invariance on Δ ln truth; every pair) and mean expected
    crps_log on these problems."""
    problems = [data["problems"][i] for i in idx]
    has = np.array([data["terms"][i]["draws"] is not None for i in idx])
    out = {}
    for name, sig in data["comparator_sigma"].items():
        rows = {"-": B.source_rows(problems, sig[idx], floor)}
        exp = data["comparator_expected"][name][idx]
        out[name] = {"sensitivity": RC.seed_mean_slope(rows, "a6c_pairs", "ok", ac),
                     "invariance": RC.seed_mean_slope(rows, "a6b_pairs", "ok", ac, "m"),
                     "mean_expected": float(np.mean(exp[has])) if has.any() else None}
    return out


def j_statistics(idx: np.ndarray, data: dict, ac: dict, pc: dict, floor: float, margin: float) -> dict:
    """J1-J6, the criteria per source (C1-C3 on the J sets) and the comparators on the problems `idx` (whole pairs).
    R4-3's k_statistics on the J files: its K1 is J1, K6 / K7 / K8 are J3 / J4 / J5."""
    base = RC.k_statistics(idx, data, ac, pc, floor, margin)
    problems = [data["problems"][i] for i in idx]
    has = np.array([data["terms"][i]["draws"] is not None for i in idx])
    boot = oc.ClusterBootstrap([(p["_set"], p["noise"]["pair"]["pair_seed"]) for p in problems], ac["bootstrap"],
                               ac["seed"])
    exp_b2 = {s: d["expected"]["a"][idx] for s, d in data["arms"]["b2"].items()}
    x1 = data["comparator_expected"].get("x1")
    k = base["tests"]
    tests = {"J1": k["K1"], "J2": slope_difference(data, idx, ac, pc, floor), "J3": k["K6"], "J4": k["K7"],
             "J5": k["K8"], "J6": RC.seed_mean_diff(boot, exp_b2, None if x1 is None else x1[idx], has)}
    return {"n_problems": base["n_problems"], "n_pairs": base["n_pairs"], "tests": tests,
            "descriptive_tests": {"K2_line_X": k["K2"], "K3_exact_from_terms": k["K3"], "K4_b2_slope": k["K4"],
                                  "K5_dpos2_minus_b2_expected": k["K5"]},
            "criteria": base["criteria"], "comparators": comparator_stats(data, idx, ac, floor)}


def _slopes(st: dict) -> dict:
    """A split's per-source slopes: J sensitivity and J invariance, and each comparator's."""
    out = {name: {"sensitivity": c["a6c"], "invariance": c["a6b"]} for name, c in st["criteria"].items()}
    out.update({f"comparator_{n}": {"sensitivity": c["sensitivity"], "invariance": c["invariance"]}
                for n, c in st["comparators"].items()})
    return {"n_problems": st["n_problems"], "n_pairs": st["n_pairs"], "slopes": out}


def _j(p: dict) -> dict:
    return p["noise"]["pair"].get("j") or {}


def splits(data: dict, ac: dict, pc: dict, floor: float, margin: float, scfg: dict) -> dict:
    """D147 §10's pre-specified splits (descriptive): depth, exceedance (per set, at its median), the chain's training
    count, strict, rank flip (sensitivity pairs)."""
    problems = data["problems"]
    all_idx = np.arange(len(problems))
    sets = np.array([p["_set"] for p in problems])
    exc = np.array([_j(p).get("exceedance", np.nan) for p in problems])
    med = {s: float(np.nanmedian(exc[sets == s])) if np.isfinite(exc[sets == s]).any() else None for s in set(sets)}
    above = np.array([med[s] is not None and e > med[s] for s, e in zip(sets, exc)])
    ntr = np.array([_j(p).get("n_training", 0) for p in problems])
    strict = np.array([bool(_j(p).get("strict")) for p in problems])
    flip = np.array([bool(_j(p).get("rank_flip")) for p in problems]) & (sets == "a6c_pairs")
    depth = np.array([p["depth"] for p in problems])
    groups = {**{f"depth_{d}": depth == d for d in sorted(set(depth))},
              "exceedance_below_median": ~above & np.isfinite(exc), "exceedance_above_median": above,
              f"training_count_1_to_{scfg['training_count_min'] - 1}": (ntr >= 1) & (ntr < scfg["training_count_min"]),
              f"training_count_{scfg['training_count_min']}_plus": ntr >= scfg["training_count_min"],
              "strict": strict, "not_strict": ~strict & np.isfinite(exc),
              "rank_flip_sensitivity": flip, "no_rank_flip_sensitivity": (sets == "a6c_pairs") & ~flip & np.isfinite(exc)}
    out = {"exceedance_medians": med}
    for name, m in groups.items():
        if m.any():
            out[name] = _slopes(j_statistics(all_idx[m], data, ac, pc, floor, margin))
    return out


def s_lines_by_range(data: dict, recs: dict, refs: list[dict], tol: float) -> dict:
    """D147 §10: D-pos-v2's S lines (D113 §5's diagnostics), per seed, split by whether the term's given is out of its
    chain's training range. s_line_rows emits one row per reference term in (problem, term) order."""
    problems = data["problems"]
    flags = [sym in set(_j(p).get("out_of_range_givens", [])) for p, ref in zip(problems, refs) for sym in ref["sens"]]
    out = {}
    for s, r in recs["dpos2"].items():
        if r is None:
            continue
        rows, _ = R.s_line_rows(problems, r, refs, tol)
        out[s] = {"out_of_range": R.summarize_s([x for x, f in zip(rows, flags) if f]),
                  "in_range": R.summarize_s([x for x, f in zip(rows, flags) if not f])}
    return out


def distance_rows(problems: list[dict]) -> list[dict]:
    """D147 §10: how far outside the training range each pair is (one row per pair, generator only)."""
    by: dict = {}
    for p in problems:
        by.setdefault((p["_set"], p["noise"]["pair"]["pair_seed"]), {})[p["noise"]["pair"]["member"]] = p
    rows = []
    for (set_name, seed), m in sorted(by.items()):
        lo, hi = m["lo"], m["hi"]
        j = _j(lo)
        rows.append({"set": set_name, "family": lo["family"], "depth": lo["depth"], "pair_seed": seed,
                     "level": lo["noise"]["pair"]["construction"].get("levels", lo["noise"]["level"]),
                     "chain": lo["chain"], "exceedance": j.get("exceedance"), "exceedance_given": j.get("exceedance_given"),
                     "nearest_training_linf": j.get("nearest_training_linf"), "n_training": j.get("n_training"),
                     "out_of_range_givens": j.get("out_of_range_givens"), "strict": j.get("strict"),
                     "rank_flip": j.get("rank_flip"), "clamped_log_ratio_lo": j.get("clamped_log_ratio"),
                     "clamped_log_ratio_hi": _j(hi).get("clamped_log_ratio"),
                     "d_ln_sigma_star": float(np.log(hi["noise"]["sigma_star_log"] / lo["noise"]["sigma_star_log"])),
                     "d_ln_truth": float(np.log(hi["target_value_si"] / lo["target_value_si"]))})
    return rows


def context(cfg: dict) -> dict | None:
    """R4-3's confirmatory depths 1-4 slopes (#32), beside J's; not a test (the sets differ in more than the
    Jacobian)."""
    p = REPO_ROOT / cfg["r4_context"]
    if not p.exists():
        return None
    crit = json.loads(p.read_text())["subsets"]["depths_1_4"]["criteria"]
    pick = lambda d: None if not d else {k: d[k] for k in ("value", "ci90")}
    return {name: {"a6c": pick(c.get("a6c")), "a6b": pick(c.get("a6b"))} for name, c in crit.items()}


# ------------------------------------------------------------------------------------------- the analysis
def analyze(cfg: dict | None = None, root=None, allow_incomplete: bool = False) -> dict:
    RC._guard()
    cfg = cfg or _config()
    r4 = load_config(cfg["stage_r4_config"])
    ac, jc = r4["analysis"], r4["r5_6"]
    root = root or REPO_ROOT / r4["output_dir"]
    sa = stage_a_config(cfg)
    pc = load_config(load_config(ac["stage_b_config"])["analysis"]["pairs_config"])
    floor, margin = pc["analysis"]["sigma_floor_rel"], pc["invariance_analysis"]["equivalence_margin"]
    sc = reward_config()["scoring"]
    dcfg = load_config(r4["sft_phase12_config"])
    problems = stage_a.load_sets(sa)
    recs = {arm: {s: B.read_records(root / arm / f"seed{s}" / jc["output_subdir"]) for s in jc["seeds"]} for arm in ARMS}
    missing = [f"{arm} seed {s}" for arm in ARMS for s in jc["seeds"] if recs[arm][s] is None]
    if missing and not allow_incomplete:
        raise RuntimeError(f"D147 §11: the analysis runs once all six evals exist; missing {missing}")
    data, per_seed_desc, refs, note = RC.load_data(problems, recs, sa, ac, pc, dcfg, sc)
    clamp = clamped_sigma(problems)
    if np.isfinite(clamp).all():
        data["comparator_sigma"]["clamped"] = clamp
        data["comparator_expected"]["clamped"] = oc.score_source(data["terms"], clamp, sc)["expected"]
    n_seeds = {arm: len(data["arms"][arm]) for arm in ARMS}
    all_idx = np.arange(len(problems))
    res: dict = {"decision": "D147 (R5-6 part J, pre-registered)", "n_problems": len(problems),
                 "seeds": {arm: sorted(data["arms"][arm]) for arm in ARMS}, "sources_note": note,
                 "comparators": sorted(data["comparator_sigma"]), "complete": not missing,
                 "label": "complete" if not missing else
                 f"incomplete ({', '.join(f'{a} {n} seeds' for a, n in n_seeds.items())}); still the one-time analysis",
                 "bootstrap": ac["bootstrap"]}
    main = j_statistics(all_idx, data, ac, pc, floor, margin)
    h = holm({k: (1.0 if main["tests"][k].get("missing") else main["tests"][k]["p_one_sided"]) for k in TESTS}, ac["alpha"])
    for k in TESTS:
        main["tests"][k]["holm"] = h[k]
    res["main"] = main
    j1 = main["tests"]["J1"]
    flags = {"J1": bool(h["J1"]["rejected"]),
             "C2": bool(not j1.get("missing") and j1["value"] >= ac["practical_slope"]),
             "C3": bool(main["criteria"].get("dpos2_a", {}).get("C3", False))}
    res["primary"] = {**flags, "criteria": list(cfg["primary_criteria"]),
                      "confirmed": all(flags[c] for c in cfg["primary_criteria"]),
                      "rule": "P_J confirmed iff " + " and ".join(cfg["primary_criteria"]) + " (D147 §8: J1 Holm-rejected, "
                              "C2 seed-mean J-sensitivity slope of (a) >= 0.5, C3 its J-invariance seed-mean 90% CI inside "
                              "D59's margin)",
                      "reading": reading(flags)}
    res["splits"] = splits(data, ac, pc, floor, margin, cfg["splits"])
    res["s_lines_by_range"] = s_lines_by_range(data, recs, refs, ac["sensitivity_rel_tol"])
    dist = distance_rows(problems)
    res["distances"] = {s: {k: (np.percentile([r[k] for r in dist if r["set"] == s and r[k] is not None], [25, 50, 75]).tolist()
                                if any(r["set"] == s and r[k] is not None for r in dist) else None)
                            for k in ("exceedance", "nearest_training_linf", "n_training", "clamped_log_ratio_hi")}
                        for s in cfg["pair_sets"]}
    res["context_r4_depths_1_4"] = context(cfg)
    res["per_seed"] = per_seed_desc
    out = REPO_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "pair_distances.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dist))
    (out / "results.json").write_text(json.dumps(res, indent=1, default=B._json_default))
    (out / "summary.md").write_text(render(res))
    return res


def reading(flags: dict) -> str:
    """D147 §12's reading, fixed before any result."""
    if flags["J1"] and flags["C2"] and flags["C3"]:
        return ("P_J confirmed: D-pos-v2's explicit sensitivity computation propagates measurement noise through "
                "Jacobians outside its training range (same chains, same value ranges), across 3 seeds, and is invariant "
                "to a change of scale there")
    if flags["J1"] and flags["C3"]:
        return "propagates below the practical threshold outside the training range (J1 and C3 without C2)"
    if flags["J1"] and flags["C2"]:
        return "tracks σ* but is not scale-invariant outside the training range (J1 and C2 without C3)"
    return ("P_J not confirmed: a failed generalization, reported with the same prominence as a success; #32's result "
            "stands for in-range Jacobians only")


# ------------------------------------------------------------------------------------------- the summary
_ci = RC._ci


def render(res: dict) -> str:
    pr, m = res["primary"], res["main"]
    L = ["# R5-6 part J: Jacobians outside the training range (D147), D-pos-v2 and B-v2, 3 seeds", "",
         f"Seeds: D-pos-v2 {res['seeds']['dpos2']}, B-v2 {res['seeds']['b2']} ({res['label']}). Problems "
         f"{res['n_problems']}. Bootstrap: stage B's joint pairs bootstrap, B = {res['bootstrap']}. Comparators: "
         f"{', '.join(res['comparators'])} ({res['sources_note']}).", "",
         f"**P_J: {'CONFIRMED' if pr['confirmed'] else 'NOT CONFIRMED'}** (J1 Holm-rejected: {pr['J1']}; C2, seed-mean "
         f"J-sensitivity slope ≥ 0.5: {pr['C2']}; C3, J-invariance seed-mean 90% CI inside ±0.3: {pr['C3']}).", "",
         f"Reading (fixed in D147 §12): {pr['reading']}.", "",
         "| test | seed-mean value [90% CI] | 95% CI | p (one-sided) | Holm | per seed |", "|---|---|---|---|---|---|"]
    for k in TESTS:
        t = m["tests"][k]
        if t.get("missing"):
            L.append(f"| {k}: {TEST_NAMES[k]} | missing | | | | |")
            continue
        per = ", ".join(f"{s}: {v['value'] if isinstance(v, dict) else v:+.4f}" for s, v in (t.get("per_seed") or {}).items())
        if k == "J2":
            per = f"explicit {t['explicit']:+.4f}, implicit {t['implicit']:+.4f}"
        L.append(f"| {k}: {TEST_NAMES[k]} | {_ci(t, fmt='+.4f')} | [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] "
                 f"| {t['p_one_sided']:.4g} | {'rejected' if t['holm']['rejected'] else '–'} | {per} |")
    L += ["", "## Criteria per σ source (seed means; J sensitivity in A6c's place, J invariance in A6b's)", "",
          "| source | J-sensitivity slope [90% CI] | C1 | C2 | C3 (J-invariance in ±0.3) | J-invariance slope |",
          "|---|---|---|---|---|---|"]
    for name, c in m["criteria"].items():
        L.append(f"| {name} | {_ci(c['a6c'])} | {c['C1']} | {c['C2']} | {c['C3']} | {_ci(c['a6b'])} |")
    for name, c in m["comparators"].items():
        L.append(f"| comparator {name} | {_ci(c['sensitivity'])} | | | | {_ci(c['invariance'])} |")
    L += ["", "Mean oracle-centred expected crps_log: " + ", ".join(
        f"{n} {c['mean_expected']:+.4f}" for n, c in m["comparators"].items() if c["mean_expected"] is not None) + ".",
          "", "## Descriptive (D147 §10; outside the Holm family)", "",
          "| split | pairs | D-pos-v2 (a) J-sens | B-v2 (a) J-sens | D-pos-v2 (a) J-inv | σ* J-sens | clamped J-sens |",
          "|---|---|---|---|---|---|---|"]
    for name, st in res["splits"].items():
        if name == "exceedance_medians":
            continue
        s = st["slopes"]
        g = lambda src, kind: _ci((s.get(src) or {}).get(kind))
        L.append(f"| {name} | {st['n_pairs']} | {g('dpos2_a', 'sensitivity')} | {g('b2_a', 'sensitivity')} "
                 f"| {g('dpos2_a', 'invariance')} | {g('comparator_sigma_star', 'sensitivity')} "
                 f"| {g('comparator_clamped', 'sensitivity')} |")
    L += ["", "**D-pos-v2's S lines, out-of-range vs in-range terms (per seed):**", ""]
    for s, d in res["s_lines_by_range"].items():
        f = lambda x: "–" if x is None else f"{x:.1%}"
        o, i = d["out_of_range"]["all"], d["in_range"]["all"]
        L.append(f"- seed {s}: out of range {o['n_terms']} terms, present {f(o['present'])}, within 10% {f(o['lenient'])}; "
                 f"in range {i['n_terms']} terms, present {f(i['present'])}, within 10% {f(i['lenient'])}")
    L += ["", "**How far outside (quartiles; per pair in pair_distances.jsonl):**", ""]
    for s, d in res["distances"].items():
        L.append(f"- {s}: " + "; ".join(f"{k} {', '.join(f'{v:.3g}' for v in q)}" for k, q in d.items() if q))
    if res["context_r4_depths_1_4"]:
        L += ["", "**Context, not a test: R4-3's confirmatory depths 1–4 (#32), in-range Jacobians, other chain mix:**", ""]
        for name, c in res["context_r4_depths_1_4"].items():
            L.append(f"- {name}: A6c {_ci(c['a6c'])}, A6b {_ci(c['a6b'])}")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["analyze"])
    ap.add_argument("--allow-incomplete", action="store_true", help="D147 §11: only if an eval is BLOCKED for good")
    args = ap.parse_args()
    analyze(allow_incomplete=args.allow_incomplete)
    print((REPO_ROOT / _config()["output_dir"] / "summary.md").read_text())


if __name__ == "__main__":
    main()

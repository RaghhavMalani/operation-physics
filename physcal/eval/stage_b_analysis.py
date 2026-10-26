"""Phase 12 stage B (DECISIONS.md D104, pre-registered): the analysis of the supervised σ* ceiling, CPU only.

    OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.eval.stage_b_analysis analyze

Per seed, the model's stated relative σ on the frozen A6 / A6b / A6c sets (its greedy eval tape) goes through D59 /
D61 / D64 / D95's pair analysis (`a6_pairs.analyze_rows`, unchanged) and D100's oracle-centred diagnostic
(`oracle_centred`, stage A's MC streams), next to X1 primary, the RSS heuristic and σ̂ = σ*. The primary statistics
are seed means; one bootstrap over pairs (the same resampled pairs for every seed) gives their CIs and one-sided p:
B1 A6 slope > 0 (computation-correct pairs), B2 A6c slope > 0 (all answered pairs), B3 / B4 expected crps_log
minus RSS / X1 > 0, B5 A6 slope minus X1's > 0 (the model's computation-correct pairs). Holm over B1-B5.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.eval import a6_pairs, oracle_centred as oc
from physcal.eval import stage_a
from physcal.eval.stats import holm
from physcal.verify.reward import reward_config

SETS = ("a6_pairs", "a6b_pairs", "a6c_pairs")
TESTS = ("B1", "B2", "B3", "B4", "B5")
PROPAGATING = ("partial propagation", "full propagation", "over-propagation")


def _config() -> dict:
    return load_config("stage_b")


def _key(p: dict) -> tuple:
    return p["family"], p["depth"], p["seed"]


# ------------------------------------------------------------------------------------------- inputs
def read_records(tape_dir: Path) -> list[dict] | None:
    """A seed's eval records (rollouts.jsonl, else its .gz), or None if the seed has no eval."""
    for name, opener in (("rollouts.jsonl", open), ("rollouts.jsonl.gz", gzip.open)):
        p = tape_dir / name
        if p.exists():
            with opener(p, "rt") as fh:
                return [json.loads(line) for line in fh if line.strip()]
    return None


def model_sigma(problems: list[dict], recs: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Per problem (the frozen sets' order): the stated relative σ (0 = no stated spread or unanswered: the floor,
    D104) and the verifier breakdown (a missing record is unanswered)."""
    by = {(r["split"], *_key(r["problem"])): r for r in recs}
    sig, bds = [], []
    for p in problems:
        r = by.get((p["_set"], *_key(p)))
        b = r["breakdown"] if r else {"reason": "missing"}
        ok = b["reason"] == "ok" and b.get("sigma_si") is not None and b.get("mu_si") is not None and b["mu_si"] > 0
        sig.append(b["sigma_si"] / b["mu_si"] if ok else 0.0)
        bds.append(b)
    return np.array(sig), bds


def model_rows(problems: list[dict], bds: list[dict], pc: dict, r36: tuple[float, float] | None = None) -> dict:
    """a6_pairs.member_row per set; with `r36` = (floor, share), computation-correct is R3-6's
    |ln μ − ln f(x_meas)| <= max(floor, share·σ*) instead of A1's τ."""
    a = pc["analysis"]
    out: dict = {s: [] for s in SETS}
    for p, b in zip(problems, bds):
        row = a6_pairs.member_row(p, b, a["tau"], a["sigma_floor_rel"])
        if r36 is not None and row["ok"]:
            tau = max(r36[0], r36[1] * p["noise"]["sigma_star_log"])
            row["computation_correct"] = abs(math.log(b["mu_si"] / p["noise"]["point_estimate_si"])) <= tau
        out[p["_set"]].append(row)
    return out


def source_rows(problems: list[dict], sigma: np.ndarray, floor: float) -> dict:
    """A source that states no answer (X1, a heuristic, σ*): every member answered and computation-correct."""
    out: dict = {s: [] for s in SETS}
    for p, s in zip(problems, sigma):
        n = p["noise"]
        out[p["_set"]].append({"pair": (p["family"], p["depth"], n["level"], n["pair"]["pair_seed"]),
                               "member": n["pair"]["member"], "kind": n["pair"].get("kind", "sensitivity"),
                               "sigma_star_log": n["sigma_star_log"], "truth": p["target_value_si"], "ok": True,
                               "computation_correct": True, "stated_log": math.log(max(float(s), floor)),
                               "sigma_zero": float(s) == 0.0})
    return out


# ------------------------------------------------------------------------------------------- the joint bootstrap
class PairArrays:
    """One set's pairs (frozen, so the same for every seed): x = Δ ln σ*, m = Δ ln truth; per source the masks and y."""

    def __init__(self, rows: list[dict]):
        self.pairs = a6_pairs.pairs_from_rows(rows)
        self.keys = [p["lo"]["pair"] for p in self.pairs]
        self.x = np.array([a6_pairs._x(p, "sigma_star") for p in self.pairs])
        self.m = np.array([a6_pairs._x(p, "truth") for p in self.pairs])

    def of(self, rows: list[dict]) -> dict:
        pairs = {p["lo"]["pair"]: p for p in a6_pairs.pairs_from_rows(rows)}
        ps = [pairs[k] for k in self.keys]
        ok = np.array([p["lo"]["ok"] and p["hi"]["ok"] for p in ps])
        cc = ok & np.array([p["lo"]["computation_correct"] and p["hi"]["computation_correct"] for p in ps])
        y = np.array([p["hi"]["stated_log"] - p["lo"]["stated_log"] if o else 0.0 for p, o in zip(ps, ok)])
        return {"ok": ok, "cc": cc, "y": y}


def wslope(C: np.ndarray, w: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS slope (with intercept) of y on x over the pairs with w, per row of the count matrix C (pairs × weights)."""
    w = w.astype(float)
    n, sx, sy = C @ w, C @ (w * x), C @ (w * y)
    sxy, sxx = C @ (w * x * y), C @ (w * x * x)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (n * sxy - sx * sy) / (n * sxx - sx * sx)


def seed_mean_slope(arr: PairArrays, per_seed: dict, mask: str, C: np.ndarray, regressor: str = "x",
                    minus: dict | None = None) -> dict | None:
    """Seed mean of the per-seed slopes (optionally minus another source's slope on the same pairs), with the
    joint bootstrap's 95% / 90% CIs and one-sided p (share of resamples <= 0)."""
    if not per_seed:
        return None
    reg = arr.x if regressor == "x" else arr.m
    one = np.ones((1, len(arr.keys)))
    pts, boots, per = [], [], {}
    for s, d in per_seed.items():
        w = d[mask]
        pt = wslope(one, w, reg, d["y"])[0] - (wslope(one, w, reg, minus["y"])[0] if minus else 0.0)
        bt = wslope(C, w, reg, d["y"]) - (wslope(C, w, reg, minus["y"]) if minus else 0.0)
        pts.append(pt)
        boots.append(bt)
        per[s] = {"value": float(pt), "n_pairs": int(w.sum())}
    b = np.nanmean(np.array(boots), axis=0)
    return {"seeds": sorted(per_seed), "value": float(np.mean(pts)), "ci95": np.nanpercentile(b, [2.5, 97.5]).tolist(),
            "ci90": np.nanpercentile(b, [5, 95]).tolist(), "p_one_sided": float(np.mean(~(b > 0))), "per_seed": per}


def boot_mean_dist(boot: oc.ClusterBootstrap, v: np.ndarray, mask: np.ndarray) -> tuple[float, np.ndarray]:
    num, den = boot.sums(np.where(mask, v, 0.0)), boot.sums(mask.astype(float))
    return float(num.sum() / den.sum()), (boot.counts @ num) / (boot.counts @ den)


def expected_diff(boot: oc.ClusterBootstrap, model: dict, other: np.ndarray, mask: np.ndarray) -> dict | None:
    """B3 / B4: the seed mean of per-problem expected crps_log (model − other), clusters = pairs."""
    if not model:
        return None
    d = np.mean([model[s] - other for s in sorted(model)], axis=0)
    pt, dist = boot_mean_dist(boot, d, mask)
    per = {s: float(np.mean((model[s] - other)[mask])) for s in sorted(model)}
    return {"seeds": sorted(model), "value": pt, "ci95": np.percentile(dist, [2.5, 97.5]).tolist(),
            "p_one_sided": float(np.mean(dist <= 0)), "per_seed": per}


# ------------------------------------------------------------------------------------------- descriptives
def describe_tape(problems: list[dict], recs: list[dict], bds: list[dict], tau: float, bins) -> dict:
    from physcal.verify.parse import parse_confidence

    reasons = [b["reason"] for b in bds]
    ok = [b for b in bds if b["reason"] == "ok" and b.get("mu_si") is not None and b["mu_si"] > 0]
    cc = [abs(math.log(b["mu_si"] / p["noise"]["point_estimate_si"])) <= tau
          for p, b in zip(problems, bds) if b["reason"] == "ok" and b.get("mu_si") and b["mu_si"] > 0]
    return {"n_records": len(recs), "n_problems": len(problems), "missing": reasons.count("missing"),
            "parse_fail_rate": reasons.count("parse_fail") / len(bds),
            "unit_mismatch_rate": reasons.count("unit_mismatch") / len(bds),
            "answered_rate": len(ok) / len(bds), "a1_computation_correct_rate": float(np.mean(cc)) if cc else None,
            "sigma_zero_share_answered": float(np.mean([b.get("sigma_si") == 0 for b in ok])) if ok else None,
            "confidence_parse_rate": float(np.mean([parse_confidence(r["completion"], bins) is not None for r in recs])),
            "mean_verifier_reward": float(np.mean([r["reward"] for r in recs]))}


# ------------------------------------------------------------------------------------------- the analysis
def analyze(cfg: dict | None = None, seeds_root: Path | None = None, sa: dict | None = None, eval_subdir: str = "eval",
            out_dir: Path | None = None) -> dict:
    """`sa` / `eval_subdir` / `out_dir` (D118): another stage-A config (the confirmatory files and their X1), each
    seed's eval directory name, and where results.json / summary.md go. The defaults are stage B's."""
    cfg = cfg or _config()
    ac = cfg["analysis"]
    pc, sa = load_config(ac["pairs_config"]), sa or load_config(ac["stage_a_config"])
    sc = reward_config()["scoring"]
    floor = pc["analysis"]["sigma_floor_rel"]
    root = seeds_root or REPO_ROOT / cfg["output_dir"]
    problems = stage_a.load_sets(sa)
    terms = [oc.problem_terms(p, sa["mc"]["n_draws"], sa["mc"]["seed_tag"]) for p in problems]
    src, note = stage_a.sources(problems, sa)
    for need in (ac["rss_source"], ac["x1_source"]):
        if need not in src:
            raise RuntimeError(f"comparator {need} unavailable: {note}")
    comps = {"sigma_star": src[stage_a.STAR], "rss": src[ac["rss_source"]], "x1": src[ac["x1_source"]]}
    boot = oc.ClusterBootstrap([(p["_set"], p["noise"]["pair"]["pair_seed"]) for p in problems], ac["bootstrap"], ac["seed"])
    has_draws = np.array([t["draws"] is not None for t in terms])
    comp_scores = {n: oc.score_source(terms, s, sc) for n, s in comps.items()}
    res: dict = {"decision": "D104 (Phase 12 stage B, pre-registered)", "n_problems": len(problems),
                 "n_pairs": boot.k, "sources_note": note, "seeds": {}, "comparators": {}}
    bins = load_config(cfg["sft_phase12_config"])["confidence"]["bins"]

    seeds, sig, bds = [], {}, {}
    for s in cfg["seeds"]:
        recs = read_records(root / f"seed{s}" / eval_subdir)
        if recs is None:
            res["seeds"][s] = {"ran": False}
            continue
        seeds.append(s)
        sig[s], bds[s] = model_sigma(problems, recs)
        rows = model_rows(problems, bds[s], pc)
        a6 = a6_pairs.analyze_rows(rows["a6_pairs"], pc["analysis"])
        a6b = a6_pairs.analyze_rows(rows["a6b_pairs"], pc["analysis"], pc["invariance_analysis"])
        a6c = a6_pairs.analyze_rows(rows["a6c_pairs"], pc["analysis"])
        r36 = model_rows(problems, bds[s], pc, (ac["r36_tau_floor"], ac["r36_tau_sigma_share"]))
        scores = oc.score_source(terms, sig[s], sc)
        summ = root / f"seed{s}" / "summary.json"
        tr = json.loads(summ.read_text()) if summ.exists() else {}
        res["seeds"][s] = {
            "ran": True, "a6": a6, "a6b": a6b, "a6c": a6c, "joint": a6_pairs.joint(a6, a6b),
            "r36_secondary": {"a6": a6_pairs.analyze_rows(r36["a6_pairs"], pc["analysis"])["primary_computation_correct"],
                              "a6c": a6_pairs.analyze_rows(r36["a6c_pairs"], pc["analysis"])["secondary_computation_correct"]},
            "oracle_centred": oc.summarize(terms, sig[s], scores, boot, sc, sa["coverage_k"]),
            "tape": describe_tape(problems, recs, bds[s], pc["analysis"]["tau"], bins),
            "training": {k: tr.get(k) for k in ("steps", "steps_planned", "epochs_completed", "train_loss", "seconds",
                                                "oom_check", "peak_vram", "fingerprint", "data")}}
        res["seeds"][s]["_scores"] = scores
    for n, s in comps.items():
        rows = source_rows(problems, s, floor)
        res["comparators"][n] = {
            "a6_all_pairs": a6_pairs.slope_test(a6_pairs.pairs_from_rows(rows["a6_pairs"]), ac["bootstrap"], ac["seed"]),
            "a6b_all_pairs": a6_pairs.slope_test(a6_pairs.pairs_from_rows(rows["a6b_pairs"]), ac["bootstrap"], ac["seed"],
                                                 "truth"),
            "a6c_all_pairs": a6_pairs.slope_test(a6_pairs.pairs_from_rows(rows["a6c_pairs"]), ac["bootstrap"], ac["seed"]),
            "oracle_centred": oc.summarize(terms, s, comp_scores[n], boot, sc, sa["coverage_k"])}

    # the seed-mean tests (joint bootstrap over pairs)
    model_rows_by_seed = {s: model_rows(problems, bds[s], pc) for s in seeds}
    x1_rows = source_rows(problems, comps["x1"], floor)
    arr, C, per = {}, {}, {}
    for st in SETS:
        arr[st] = PairArrays(next(iter(model_rows_by_seed.values()))[st] if seeds else x1_rows[st])
        C[st] = oc.ClusterBootstrap(arr[st].keys, ac["bootstrap"], ac["seed"]).counts
        per[st] = {s: arr[st].of(model_rows_by_seed[s][st]) for s in seeds}
    x1_a6 = arr["a6_pairs"].of(x1_rows["a6_pairs"])
    expected = {s: res["seeds"][s]["_scores"]["expected"] for s in seeds}
    t = {"B1": seed_mean_slope(arr["a6_pairs"], per["a6_pairs"], "cc", C["a6_pairs"]),
         "B2": seed_mean_slope(arr["a6c_pairs"], per["a6c_pairs"], "ok", C["a6c_pairs"]),
         "B3": expected_diff(boot, expected, comp_scores["rss"]["expected"], has_draws),
         "B4": expected_diff(boot, expected, comp_scores["x1"]["expected"], has_draws),
         "B5": seed_mean_slope(arr["a6_pairs"], per["a6_pairs"], "cc", C["a6_pairs"], minus=x1_a6)}
    h = holm({k: (v["p_one_sided"] if v else 1.0) for k, v in t.items()}, ac["alpha"])
    for k in TESTS:
        t[k] = (t[k] or {"missing": True}) | {"holm": h[k]}
    res["tests"] = t
    margin = pc["invariance_analysis"]["equivalence_margin"]
    a6b_mean = seed_mean_slope(arr["a6b_pairs"], per["a6b_pairs"], "cc", C["a6b_pairs"], regressor="m")
    r36 = {s: model_rows(problems, bds[s], pc, (ac["r36_tau_floor"], ac["r36_tau_sigma_share"])) for s in seeds}
    res["seed_mean"] = {
        "a6_label": a6_pairs.slope_label(t["B1"].get("ci90")) if not t["B1"].get("missing") else None,
        "a6b_slope": a6b_mean,
        "a6b_equivalent_to_zero": bool(a6b_mean and -margin < a6b_mean["ci90"][0] and a6b_mean["ci90"][1] < margin),
        "a6c_computation_correct": seed_mean_slope(arr["a6c_pairs"], per["a6c_pairs"], "cc", C["a6c_pairs"]),
        "r36_a6": seed_mean_slope(arr["a6_pairs"], {s: arr["a6_pairs"].of(r36[s]["a6_pairs"]) for s in seeds}, "cc",
                                  C["a6_pairs"]),
        "r36_a6c": seed_mean_slope(arr["a6c_pairs"], {s: arr["a6c_pairs"].of(r36[s]["a6c_pairs"]) for s in seeds}, "cc",
                                   C["a6c_pairs"])}
    res["reading"] = reading(t, res["seed_mean"])
    for s in seeds:
        res["seeds"][s].pop("_scores")
    out = out_dir or (REPO_ROOT / cfg["output_dir"] if seeds_root is None else seeds_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=_json_default))
    (out / "summary.md").write_text(render(res))
    return res


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def reading(t: dict, sm: dict) -> list[str]:
    """D104's reading rules."""
    sup = {k: bool(t[k]["holm"]["rejected"]) for k in TESTS}
    out = []
    if sup["B1"] and sup["B2"] and sm["a6b_equivalent_to_zero"] and sm["a6_label"] in PROPAGATING:
        out.append("the ceiling propagates (B1, B2 supported; A6b equivalent to 0; A6 label "
                   f"{sm['a6_label']})")
    if sup["B3"] and sup["B4"]:
        out.append("the ceiling beats the surface readers (B3, B4 supported)")
    if not sup["B2"]:
        out.append("B2 not supported: supervised σ* does not teach noise allocation with this base and budget; stage "
                   "C's bar is X1 / RSS, not stage B")
    if not out:
        out.append("another combination: reported as observed")
    return out


def _ci(d: dict | None, key: str = "ci95", signed: bool = True) -> str:
    if not d or d.get("missing"):
        return "–"
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(d['value'])} [{f.format(d[key][0])}, {f.format(d[key][1])}]"


def render(res: dict) -> str:
    t, sm = res["tests"], res["seed_mean"]
    ran = [s for s, v in res["seeds"].items() if v.get("ran")]
    names = {"B1": "B1: A6 slope > 0 (computation-correct pairs)", "B2": "B2: A6c slope > 0 (all answered pairs)",
             "B3": "B3: expected crps_log − RSS > 0", "B4": "B4: expected crps_log − X1 > 0",
             "B5": "B5: A6 slope − X1's > 0"}
    L = [f"# Phase 12 stage B (D104): the supervised σ* ceiling, seeds {ran}", "",
         "**Reading (D104):** " + "; ".join(res["reading"]) + ".", "",
         f"{res['n_problems']} problems, {res['n_pairs']} pairs (A6 + A6b + A6c). Seed means; CIs from one bootstrap "
         "over pairs shared by every seed (B = 10,000). Pooled problem-level evidence, not seed-level significance.", "",
         "| test | seed mean [95% CI] | per seed | p (one-sided) | Holm |", "|---|---|---|---|---|"]
    for k in TESTS:
        d = t[k]
        ps = "–" if d.get("missing") else ", ".join(f"{v['value'] if isinstance(v, dict) else v:+.3f}"
                                                       for v in d["per_seed"].values())
        L.append(f"| {names[k]} | {_ci(d)} | {ps} | {'–' if d.get('missing') else format(d['p_one_sided'], '.4g')} | "
                 f"{'supported' if d['holm']['rejected'] else 'not supported'} |")
    L += ["", f"- A6 label (seed mean, 90% CI {_ci(t['B1'], 'ci90')}): **{sm['a6_label']}**",
          f"- A6b slope on Δ ln truth (seed mean): {_ci(sm['a6b_slope'])}, 90% CI {_ci(sm['a6b_slope'], 'ci90')}; "
          f"equivalent to 0 (TOST): **{sm['a6b_equivalent_to_zero']}**",
          f"- A6c on computation-correct pairs (seed mean): {_ci(sm['a6c_computation_correct'])}",
          f"- R3-6 σ secondary (τ_σ = max(0.25%, 0.25·σ*)): A6 {_ci(sm['r36_a6'])}; A6c {_ci(sm['r36_a6c'])}", ""]
    L += ["| source | A6 slope | A6b slope | A6c slope | expected crps_log | realized | median ln(σ/σ*) | "
          "median abs | vol beta | 1σ / 2σ cov. |", "|---|---|---|---|---|---|---|---|---|---|"]

    def oc_cells(o):
        return (f"{o['expected'][0]:.4f} | {o['realized'][0]:.4f} | {o['median_log_ratio']:+.3f} | "
                f"{o['median_abs_log_ratio']:.3f} | {o['vol_beta'][0]:.3f} | {o['coverage_1'][0]:.1%} / "
                f"{o['coverage_2'][0]:.1%}")
    for s in ran:
        r = res["seeds"][s]
        L.append(f"| stage B seed {s} | {r['a6']['primary_computation_correct']['slope'] or float('nan'):+.3f} "
                 f"(n {r['a6']['primary_computation_correct']['n_pairs']}, {r['a6']['label']}) | "
                 f"{r['a6b']['primary_computation_correct']['slope'] or float('nan'):+.3f} | "
                 f"{r['a6c']['primary_all_answered']['slope'] or float('nan'):+.3f} | {oc_cells(r['oracle_centred'])} |")
    for n, r in res["comparators"].items():
        L.append(f"| {n} (all pairs) | {r['a6_all_pairs']['slope']:+.3f} | {r['a6b_all_pairs']['slope']:+.3f} | "
                 f"{r['a6c_all_pairs']['slope']:+.3f} | {oc_cells(r['oracle_centred'])} |")
    L += ["", "| seed | steps (planned) | train loss | answered | parse fail | unit mismatch | A1 comp.-correct | "
          "σ = 0 | confidence block | D59 verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in ran:
        r, d, tr = res["seeds"][s], res["seeds"][s]["tape"], res["seeds"][s]["training"]
        f = lambda v: "–" if v is None else f"{v:.1%}"
        L.append(f"| {s} | {tr.get('steps')} ({tr.get('steps_planned')}) | "
                 f"{'–' if tr.get('train_loss') is None else format(tr['train_loss'], '.3f')} | {f(d['answered_rate'])} | "
                 f"{f(d['parse_fail_rate'])} | {f(d['unit_mismatch_rate'])} | {f(d['a1_computation_correct_rate'])} | "
                 f"{f(d['sigma_zero_share_answered'])} | {f(d['confidence_parse_rate'])} | "
                 f"{r['joint']['labelled_verdict']} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["analyze"])
    ap.parse_args()
    print(render(analyze()))


if __name__ == "__main__":
    main()

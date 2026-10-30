"""X1 on A6c (DECISIONS.md D99): D73's shallow text model of σ*, refit exactly, applied to the noise-allocation pairs.

    OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.eval.x1_a6c run [--n-jobs 1]
    PHYSCAL_CONFIRMATORY=1 OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.eval.x1_a6c confirmatory   # D118 §3
    PHYSCAL_CONFIRMATORY=1 OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.eval.x1_a6c r5_6           # D147 §7

X1's fitted models were not saved, so they are refit: the same fit set (regenerated from D73's seeds), the same
features and the hyperparameters D73's grouped CV selected (results/x1_text_sigma/selection.json); no new selection.
Before anything touches A6c, the refit must reproduce results/x1_text_sigma/results.json within `REPRO_TOL` (test
and member accuracy, the A6 / A6b primary slopes and CIs). Otherwise the run stops and writes only reproduction.json.

On A6c, σ̂ = exp(ŷ) is each member's stated σ and every pair enters D95's analysis (`a6_pairs.analyze_rows`), as for
the noise heuristics. Within a pair the displayed values and the multiset of "± p%" are identical, so the noise and
magnitude features are identical and only TF-IDF (tokens with attached punctuation) can differ.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.eval import x1_text_sigma as x1
from physcal.eval.a6_pairs import analyze_rows

REPRO_TOL = 1e-9  # D99, as D92's reproduction rule
OUT = "a6c"       # results/x1_text_sigma/a6c


def refit(n_jobs: int) -> tuple[dict, list[dict]]:
    """D73's fit set and test set, regenerated; one model per feature set at its D73-selected grid point."""
    cfg = x1._config()
    fit_set = [x1._problem_record(d) for d in x1.build_set({**cfg["train"]}, cfg["env_config"], n_jobs)]
    test_set = [x1._problem_record(d) for d in x1.build_set({**cfg["train"], **cfg["test"]}, cfg["env_config"], n_jobs)]
    table = json.loads((REPO_ROOT / cfg["output_dir"] / "selection.json").read_text())["cv"]
    return x1.fit_models(fit_set, table, cfg), test_set


def _diffs(new, old, path: str = "") -> list[tuple[str, float]]:
    """Every numeric leaf of `old` against the same leaf of `new`: (path, |difference|); a missing leaf is inf."""
    if isinstance(old, dict):
        return [d for k, v in old.items() for d in _diffs(new.get(k) if isinstance(new, dict) else None, v, f"{path}/{k}")]
    if isinstance(old, (list, tuple)):
        new = new if isinstance(new, (list, tuple)) and len(new) == len(old) else [None] * len(old)
        return [d for i, v in enumerate(old) for d in _diffs(new[i], v, f"{path}[{i}]")]
    if isinstance(old, bool) or old is None or isinstance(old, str):
        return [(path, 0.0 if new == old else math.inf)]
    if new is None:
        return [(path, math.inf)]
    return [(path, abs(float(new) - float(old)))]


def reproduction(models: dict, test_set: list[dict], a6: list[dict], a6b: list[dict], published: dict) -> dict:
    """The refit's D73 numbers against the published ones, per feature set (D99's check)."""
    p6, p6b = [x1._problem_record(d) for d in a6], [x1._problem_record(d) for d in a6b]
    out = {"tolerance": REPRO_TOL, "feature_sets": {}}
    for fs, model in models.items():
        l6, l6b = x1.predict(model, p6), x1.predict(model, p6b)
        pa = x1.pair_analysis(a6, a6b, l6, l6b)
        new = {"test_accuracy": x1.accuracy(test_set, x1.predict(model, test_set)),
               "a6_member_accuracy": x1.accuracy(p6, l6), "a6b_member_accuracy": x1.accuracy(p6b, l6b),
               "A6": pa["A6"]["primary_computation_correct"], "A6b": pa["A6b"]["primary_computation_correct"]}
        old = published["feature_sets"][fs]
        old = {"test_accuracy": old["test_accuracy"], "a6_member_accuracy": old["a6_member_accuracy"],
               "a6b_member_accuracy": old["a6b_member_accuracy"], "A6": old["pairs"]["A6"]["primary_computation_correct"],
               "A6b": old["pairs"]["A6b"]["primary_computation_correct"]}
        d = _diffs(new, old)
        worst = max(d, key=lambda t: t[1])
        out["feature_sets"][fs] = {"n_compared": len(d), "max_abs_diff": worst[1], "worst": worst[0],
                                   "ok": worst[1] <= REPRO_TOL}
    out["ok"] = all(v["ok"] for v in out["feature_sets"].values())
    return out


def level_pair(d: dict) -> tuple[float, float]:
    return tuple(d["noise"]["pair"]["construction"]["levels"])


def identical_share(model: tuple, problems: list[dict]) -> dict:
    """Share of pairs whose two members get bit-identical feature vectors, and bit-identical σ̂."""
    recs = [x1._problem_record(d) for d in problems]
    X = model[0].transform(recs)
    yhat = model[1].predict(X)
    by: dict = {}
    for i, d in enumerate(problems):
        by.setdefault(d["noise"]["pair"]["pair_seed"], {})[d["noise"]["pair"]["member"]] = i
    same_x = [np.array_equal(X[v["lo"]], X[v["hi"]], equal_nan=True) for v in by.values()]
    same_y = [yhat[v["lo"]] == yhat[v["hi"]] for v in by.values()]
    return {"n_pairs": len(by), "share_identical_features": float(np.mean(same_x)),
            "share_identical_sigma_hat": float(np.mean(same_y))}


def analyze_a6c(models: dict, a6c: list[dict]) -> dict:
    pc = load_config("pairs")
    rec = [x1._problem_record(d) for d in a6c]
    res = {"feature_sets": {}}
    for fs, model in models.items():
        ln = x1.predict(model, rec)
        res["feature_sets"][fs] = {
            "pairs": analyze_rows(x1.pair_rows(a6c, ln), pc["analysis"], pc["invariance_analysis"]),
            "within_pair": x1.within_pair(a6c, ln), "member_accuracy": x1.accuracy(rec, ln),
            "identical": identical_share(model, a6c)}
    ln_star = np.array([r["y"] for r in rec])
    res["reference_exact_sigma_star"] = analyze_rows(x1.pair_rows(a6c, ln_star), pc["analysis"], pc["invariance_analysis"])
    prim = models[x1._config()["primary"]]
    for key, name in ((lambda d: d["depth"], "by_depth"), (level_pair, "by_level_pair")):
        groups: dict = {}
        for i, d in enumerate(a6c):
            groups.setdefault(key(d), []).append(i)
        res[name] = {}
        for g, idx in sorted(groups.items()):
            sub = [a6c[i] for i in idx]
            ln = x1.predict(prim, [rec[i] for i in idx])
            r = analyze_rows(x1.pair_rows(sub, ln), pc["analysis"], pc["invariance_analysis"])
            res[name][str(g)] = {"primary_all_answered": r["primary_all_answered"], "supported": r["supported"],
                                 "label": r["label"]}
    return res


def predictions(models: dict, sets: dict[str, list[dict]]) -> list[dict]:
    """σ̂ (ln, log-space scale) of every feature set for every member of the frozen pair sets (stage A, D100)."""
    rows = []
    for name, problems in sets.items():
        rec = [x1._problem_record(d) for d in problems]
        preds = {fs: x1.predict(m, rec) for fs, m in models.items()}
        for i, d in enumerate(problems):
            pr = d["noise"]["pair"]
            rows.append({"set": name, "family": d["family"], "depth": d["depth"], "seed": d["seed"],
                         "pair_seed": pr["pair_seed"], "member": pr["member"],
                         "ln_sigma_hat": {fs: float(v[i]) for fs, v in preds.items()}})
    return rows


def run(n_jobs: int = 1) -> dict:
    cfg = x1._config()
    out = REPO_ROOT / cfg["output_dir"] / OUT
    out.mkdir(parents=True, exist_ok=True)
    published = json.loads((REPO_ROOT / cfg["output_dir"] / "results.json").read_text())
    a6, a6b = x1.load_pairs(cfg["pairs"]["a6"]), x1.load_pairs(cfg["pairs"]["a6b"])
    models, test_set = refit(n_jobs)
    rep = reproduction(models, test_set, a6, a6b, published)
    (out / "reproduction.json").write_text(json.dumps(rep, indent=1, default=str))
    if not rep["ok"]:
        return {"decision": "D99", "refused": "the refit does not reproduce D73's results", "reproduction": rep}
    a6c_path = cfg["pairs"]["a6"].replace("a6_pairs", "a6c_pairs")
    a6c = x1.load_pairs(a6c_path)
    res = {"decision": "D99 (X1 on A6c)", "primary": cfg["primary"], "a6c": a6c_path, "reproduction": rep,
           **analyze_a6c(models, a6c)}
    (out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in
                                                   predictions(models, {"a6": a6, "a6b": a6b, "a6c": a6c})))
    heur = REPO_ROOT / load_config("pairs")["noise_heuristics"]["output_dir"] / "results.json"
    res["heuristics"] = json.loads(heur.read_text()) if heur.exists() else None
    res["x1_a6_a6b"] = {fs: e["pairs"] for fs, e in published["feature_sets"].items()}
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "summary.md").write_text(render(res))
    return res


CONF_OUT = "confirmatory"  # results/x1_text_sigma/confirmatory (D118 §3)
# target: (output subdir, the frozen files' subdir under data/eval_sets/<version>/, the sets, the decision)
TARGETS = {"confirmatory": (CONF_OUT, CONF_OUT, ("a6", "a6b", "a6c"), "D118 §3"),
           "r5_6": ("r5_6", "confirmatory/r5_6", ("a6b", "a6c"), "D147 §7")}


def run_confirmatory(n_jobs: int = 1, target: str = "confirmatory") -> dict:
    """D118 §3: the same exact refit, D99's reproduction gate, then two checks before the frozen R4-5 files are read:
    the refit's σ̂ on the development sets equals the committed predictions.jsonl exactly. Writes σ̂ for every member
    of the confirmatory A6 / A6b / A6c (needs PHYSCAL_CONFIRMATORY=1). `target` "r5_6" (D147 §7): the same gate, then
    the frozen J files (A6b / A6c layout) into results/x1_text_sigma/r5_6/."""
    import os

    from physcal.eval.run_eval import CONFIRMATORY_ENV

    if os.environ.get(CONFIRMATORY_ENV) != "1":
        raise PermissionError(f"the confirmatory set is frozen (D113); set {CONFIRMATORY_ENV}=1 (D118 §3)")
    out_name, subdir, names, tag = TARGETS[target]
    cfg = x1._config()
    out = REPO_ROOT / cfg["output_dir"] / out_name
    out.mkdir(parents=True, exist_ok=True)
    published = json.loads((REPO_ROOT / cfg["output_dir"] / "results.json").read_text())
    a6, a6b = x1.load_pairs(cfg["pairs"]["a6"]), x1.load_pairs(cfg["pairs"]["a6b"])
    models, test_set = refit(n_jobs)
    rep = reproduction(models, test_set, a6, a6b, published)
    a6c = x1.load_pairs(cfg["pairs"]["a6"].replace("a6_pairs", "a6c_pairs"))
    dev = predictions(models, {"a6": a6, "a6b": a6b, "a6c": a6c})
    committed = [json.loads(line) for line in (REPO_ROOT / cfg["output_dir"] / OUT / "predictions.jsonl").open()]
    rep["development_predictions_identical"] = dev == committed
    rep["ok"] = rep["ok"] and rep["development_predictions_identical"]
    (out / "reproduction.json").write_text(json.dumps(rep, indent=1, default=str))
    if not rep["ok"]:
        return {"decision": tag, "refused": "the refit does not reproduce D73 / D99", "reproduction": rep}
    conf = {s: x1.load_pairs(cfg["pairs"]["a6"].replace("a6_pairs", f"{subdir}/{s}_pairs")) for s in names}
    rows = predictions(models, conf)
    (out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"decision": f"{tag} (X1 on {subdir})", "reproduction": rep, "n_predictions": len(rows)}


def _slope(t: dict) -> str:
    if t.get("slope") is None:
        return "–"
    ci = t.get("ci")
    return f"{t['slope']:+.3f}" + ("" if ci is None else f" [{ci[0]:+.3f}, {ci[1]:+.3f}]")


def render(res: dict) -> str:
    L = ["# X1 on A6c, the noise-allocation pairs (D99)", "",
         f"X1 (D73) refit exactly: reproduction of D73's published numbers "
         f"{'passed' if res['reproduction']['ok'] else 'FAILED'} (max abs difference "
         f"{max(v['max_abs_diff'] for v in res['reproduction']['feature_sets'].values()):.1e}, tolerance "
         f"{res['reproduction']['tolerance']:.0e}). Every pair enters (X1 states no answer). Slope: OLS of Δ ln σ̂ on "
         "Δ ln σ* with an intercept, 90% CI, one-sided bootstrap over pairs (D95); ideal 1.", "",
         "| σ source | A6c slope [90% CI] | p (> 0) | supported | mean Δ ln σ̂ | through-origin slope | identical features "
         "| identical σ̂ |", "|---|---|---|---|---|---|---|---|"]
    for fs, e in res["feature_sets"].items():
        t, m, idn = e["pairs"]["primary_all_answered"], e["pairs"]["secondary_mean"], e["identical"]
        L.append(f"| X1 {fs}{' (primary)' if fs == res['primary'] else ''} | {_slope(t)} | {t['p_one_sided']} "
                 f"| {e['pairs']['supported']} | {m['mean_dy']:+.4f} | {m['through_origin_slope']:+.4f} "
                 f"| {idn['share_identical_features']:.1%} | {idn['share_identical_sigma_hat']:.1%} |")
    r = res["reference_exact_sigma_star"]
    L += [f"| reference σ̂ = σ* | {_slope(r['primary_all_answered'])} | {r['primary_all_answered']['p_one_sided']} "
          f"| {r['supported']} | {r['secondary_mean']['mean_dy']:+.4f} | {r['secondary_mean']['through_origin_slope']:+.4f} "
          "| – | – |", ""]
    if res.get("heuristics"):
        L += ["## One table: A6 / A6b / A6c pair statistics (X1, the noise heuristics, σ*)", "",
              "A6 and A6c: slope on Δ ln σ* (ideal 1; A6 primary on computation-correct pairs, here all). A6b: slope on "
              "Δ ln truth (ideal 0, TOST margin 0.3).", "",
              "| σ source | A6 slope | A6b slope (≡ 0?) | A6c slope (supported?) |", "|---|---|---|---|"]
        for fs, e in res["feature_sets"].items():
            p = res["x1_a6_a6b"][fs]
            L.append(f"| X1 {fs} | {_slope(p['A6']['primary_computation_correct'])} "
                     f"| {_slope(p['A6b']['primary_computation_correct'])} ({p['A6b']['equivalent_to_zero']}) "
                     f"| {_slope(e['pairs']['primary_all_answered'])} ({e['pairs']['supported']}) |")
        sets = res["heuristics"]["sets"]
        for h in sets["a6_pairs"]["heuristics"]:
            g = {k: sets[k]["heuristics"][h]["pairs"] for k in ("a6_pairs", "a6b_pairs", "a6c_pairs")}
            L.append(f"| heuristic {h} | {_slope(g['a6_pairs']['primary_computation_correct'])} "
                     f"| {_slope(g['a6b_pairs']['primary_computation_correct'])} ({g['a6b_pairs']['equivalent_to_zero']}) "
                     f"| {_slope(g['a6c_pairs']['primary_all_answered'])} ({g['a6c_pairs']['supported']}) |")
        L.append(f"| σ̂ = σ* | +1.000 | +0.000 | {_slope(r['primary_all_answered'])} ({r['supported']}) |")
        L.append("")
    for name, title in (("by_depth", "depth"), ("by_level_pair", "level pair (l, h)")):
        L += [f"## Primary feature set by {title}", "", f"| {title} | pairs | slope [90% CI] | supported |", "|---|---|---|---|"]
        for g, e in res[name].items():
            t = e["primary_all_answered"]
            L.append(f"| {g} | {t['n_pairs']} | {_slope(t)} | {e['supported']} |")
        L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["run", "confirmatory", "r5_6"])
    ap.add_argument("--n-jobs", type=int, default=1, help="problem generation workers (D97: 1 while the GPU is busy)")
    args = ap.parse_args()
    if args.action in TARGETS:
        print(json.dumps(run_confirmatory(args.n_jobs, args.action), indent=1, default=str))
        return
    res = run(args.n_jobs)
    print(json.dumps(res["reproduction"], indent=1) if "refused" in res else render(res))


if __name__ == "__main__":
    main()

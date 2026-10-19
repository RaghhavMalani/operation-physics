"""X2 analysis (D74, pre-registered): which confidence sources rank a correct answer above an incorrect answer to the
same problem?

    nice uv run python -m physcal.eval.x2_analysis analyze      # CPU, after the GPU stages (D81)

Methods, all scored on the same answers of the kept problems (confidence; a missing value makes the pair a tie):
1. stated σ, SFT: −ln max(σ/|μ|, floor) from each sampled completion;
2. stated σ, split credit seed 0 @287: the same of its greedy σ suffix on the answer (x2_pairs.suffix); a suffix that
   fails or changes μ is a tie;
3. A5 p channel (the strongest 4-bit A5 p model on problem + answer + reasoning): P(correct);
4. problem-text-only (A5's "problem" p model): identical within a problem, so it must give exactly 0.5 (a check);
5. hidden-state probe: L2 logistic regression on the state at the first "±" token, fitted on D69's calibration
   completions (5-fold CV grouped by template chooses C and the layer by AUROC; standardization inside the folds),
   applied to the answers: P(correct);
6. the same on a random-init model of the same architecture.
Primary: 1, 2, 3, 5, 6 with Holm (one-sided bootstrap p for "pairwise accuracy > 0.5"). Secondary: A5 σ; a probe
cross-fitted on the X2 answers themselves (5 folds grouped by problem; layer and C by inner CV); paired differences;
per depth. A method whose inputs are missing counts as p = 1. Writes results/x2/results.json and summary.md.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.eval import x2_pairs

PRIMARY = ("sft_sigma", "split_sigma", "a5_p", "probe", "random_init_probe")
LABELS = {"sft_sigma": "1. stated σ, SFT", "split_sigma": "2. stated σ, split credit seed 0 @287",
          "a5_p": "3. A5 p channel (4-bit)", "problem_only": "4. problem-text-only (check: must be 0.5)",
          "probe": "5. hidden-state probe (SFT, 4-bit)", "random_init_probe": "6. hidden-state probe, random init",
          "a5_sigma": "A5 σ (secondary)", "probe_crossfit": "probe cross-fitted on X2 answers (secondary)"}


# ------------------------------------------------------------------------------------------- probes
def _cv_auroc(X: np.ndarray, y: np.ndarray, folds: list, C: float, max_iter: int) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    oof = np.zeros(len(y))
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C, max_iter=max_iter).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return float(roc_auc_score(y, oof))


def select_probe(layers: list[np.ndarray], y: np.ndarray, groups: list, pc: dict, n_jobs: int | None = None) -> dict:
    """CV AUROC for every (layer, C); the best pair (ties: the earlier layer, then the smaller C)."""
    from joblib import Parallel, delayed
    from sklearn.model_selection import GroupKFold

    folds = list(GroupKFold(pc["folds"]).split(layers[0], y, groups))
    grid = [(k, C) for k in range(len(layers)) for C in pc["C"]]
    scores = Parallel(n_jobs=n_jobs if n_jobs is not None else pc["n_jobs"])(
        delayed(_cv_auroc)(layers[k], y, folds, C, pc["max_iter"]) for k, C in grid)
    table = {f"{k}|{C}": s for (k, C), s in zip(grid, scores)}
    (k, C), best = max(zip(grid, scores), key=lambda t: (t[1], -t[0][0], -t[0][1]))
    return {"layer": k, "C": C, "cv_auroc": best, "table": table}


def fit_probe(X: np.ndarray, y: np.ndarray, C: float, max_iter: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=C, max_iter=max_iter).fit(sc.transform(X), y)
    return lambda Z: m.predict_proba(sc.transform(Z))[:, 1]


def probe_confidence(fit_layers, fit_index, x2_layers, x2_index, pc: dict) -> tuple[dict, dict]:
    """Method 5 / 6: select on the fit set, refit there, score the X2 answers. Returns ({(key, sample): p}, info)."""
    y = np.array([r["correct"] for r in fit_index], dtype=int)
    sel = select_probe(fit_layers, y, [r["template"] for r in fit_index], pc)
    f = fit_probe(fit_layers[sel["layer"]], y, sel["C"], pc["max_iter"])
    p = f(x2_layers[sel["layer"]])
    return {(tuple(r["key"]), r["sample"]): float(v) for r, v in zip(x2_index, p)}, \
        {k: v for k, v in sel.items() if k != "table"} | {"n_fit": len(y), "fit_share_correct": float(y.mean())}


def crossfit_confidence(layers, index, pc: dict) -> tuple[dict, dict]:
    """Secondary: 5 folds grouped by problem over the X2 answers; in each, (layer, C) by inner grouped CV on the
    training part, refit there, score the held-out answers."""
    from sklearn.model_selection import GroupKFold

    y = np.array([r["correct"] for r in index], dtype=int)
    groups = ["|".join(map(str, r["key"])) for r in index]  # sklearn wants scalar group labels
    out, chosen = {}, []
    for tr, te in GroupKFold(pc["folds"]).split(layers[0], y, groups):
        sel = select_probe([L[tr] for L in layers], y[tr], [groups[i] for i in tr], pc)
        f = fit_probe(layers[sel["layer"]][tr], y[tr], sel["C"], pc["max_iter"])
        for i, v in zip(te, f(layers[sel["layer"]][te])):
            out[(tuple(index[i]["key"]), index[i]["sample"])] = float(v)
        chosen.append({"layer": sel["layer"], "C": sel["C"], "inner_cv_auroc": sel["cv_auroc"]})
    return out, {"folds": chosen}


# ------------------------------------------------------------------------------------------- the other methods
def a5_confidences(records: list[dict], cfg: dict) -> dict[str, dict]:
    """Methods 3 and 4 and A5 σ for every sampled answer, keyed (key, sample)."""
    from physcal.eval import nested_baselines as nb

    ac = cfg["analysis"]
    models, sel = nb.load_models(ac["a5_config"])
    tol = cfg["selection"]["tolerance"]
    recs = [nb.record(r["problem"], r["completion"], r["breakdown"], tol) for r in records]
    keys = [((r["problem"]["family"], r["problem"]["depth"], r["problem"]["seed"]), r.get("sample", 0)) for r in records]
    p = models["p"][sel["strongest"]["p"]].raw(recs)
    po = models["p"][ac["problem_only_p"]].raw(recs)
    ok = [i for i, r in enumerate(recs) if r["ok"]]
    sig = models["sigma"][sel["strongest"]["sigma"]].sigma_log([recs[i] for i in ok]) if ok else []
    s = {keys[i]: -math.log(max(float(v), ac["sigma_floor_rel"])) for i, v in zip(ok, sig)}
    return {"a5_p": {k: float(v) for k, v in zip(keys, p)}, "problem_only": {k: float(v) for k, v in zip(keys, po)},
            "a5_sigma": s, "_names": {"a5_p": sel["strongest"]["p"], "a5_sigma": sel["strongest"]["sigma"],
                                      "problem_only": ac["problem_only_p"]}}


def suffix_confidence(path: Path, floor: float) -> dict:
    """Method 2 from x2_pairs.suffix's rows: a failed suffix (parse, unit, μ changed) is a tie (None)."""
    out = {}
    for line in path.open():
        r = json.loads(line)
        ok = not r["fail"] and r["rel_sigma_used"] is not None
        out[(tuple(r["key"]), r["source_sample"])] = -math.log(max(r["rel_sigma_used"], floor)) if ok else None
    return out


# ------------------------------------------------------------------------------------------- statistics
def per_problem(groups: dict, keys: list, conf: dict) -> np.ndarray:
    return x2_pairs.pairwise_accuracy(groups, keys, lambda k, a: conf.get((tuple(k), a["sample"])))


def paired(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict:
    d = a - b
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return {"mean_diff": float(d.mean()), "ci95": np.percentile(boots, [2.5, 97.5]).tolist()}


def reading(holm: dict, diffs: dict) -> dict:
    """D74's reading rules."""
    sup = {m: holm[m]["rejected"] for m in PRIMARY}
    probe_ok = sup["probe"] and diffs["probe - random_init_probe"]["ci95"][0] > 0
    stated_fails = not sup["sft_sigma"]
    if probe_ok and stated_fails:
        outcome = "the probe succeeds and stated σ fails: the information exists; the readout or training is deficient"
    elif sup["a5_p"] and not probe_ok:
        outcome = ("only A5 succeeds: the signal is in the text, not accessible to this hidden-state readout"
                   if not any(sup[m] for m in ("sft_sigma", "split_sigma")) else
                   "A5 succeeds and the probe does not (stated σ also supported): reported as observed")
    elif not any(sup[m] for m in ("sft_sigma", "split_sigma", "a5_p", "probe")):
        outcome = ("every primary method fails: the earlier baseline advantages were mostly difficulty prediction, "
                   "not answer-specific awareness")
    else:
        outcome = "another combination: reported as observed"
    return {"supported": sup, "probe_succeeds": probe_ok, "stated_sigma_fails": stated_fails, "outcome": outcome}


def analyze(cfg: dict | None = None) -> dict:
    from physcal.eval import x2_hidden
    from physcal.eval.stats import holm

    cfg = cfg or x2_pairs._config()
    ac = cfg["analysis"]
    n_boot, seed = ac["bootstrap"], ac["seed"]
    records, n_sampled = x2_pairs.sampled_records(cfg)
    groups, keys = x2_pairs.kept_groups(cfg)
    conf: dict[str, dict | None] = {}
    conf["sft_sigma"] = {(k, a["sample"]): x2_pairs.stated_sigma_conf(k, a) for k in keys for a in groups[k]}
    sfx = REPO_ROOT / cfg["suffix"]["output_dir"] / "rows.jsonl"
    conf["split_sigma"] = suffix_confidence(sfx, ac["sigma_floor_rel"]) if sfx.exists() else None
    a5 = a5_confidences(records, cfg)
    names = a5.pop("_names")
    conf.update(a5)
    info: dict = {"a5_models": names}
    for name, model in (("probe", "sft"), ("random_init_probe", "random_init")):
        try:
            fl, fi = x2_hidden.load(model, "fit", cfg)
            xl, xi = x2_hidden.load(model, "x2", cfg)
        except FileNotFoundError as exc:
            conf[name], info[name] = None, {"missing": str(exc)}
            continue
        conf[name], info[name] = probe_confidence(fl, fi, xl, xi, ac["probe"])
        if model == "sft":
            conf["probe_crossfit"], info["probe_crossfit"] = crossfit_confidence(xl, xi, ac["probe"])
    pa = {m: per_problem(groups, keys, c) for m, c in conf.items() if c is not None}
    res: dict = {"decision": "D74 (X2)", "n_candidates_sampled": n_sampled, "n_kept": len(keys),
                 "n_pairs": int(sum(sum(a["correct"] for a in groups[k]) * sum(not a["correct"] for a in groups[k])
                                    for k in keys)),
                 "correct_count_distribution": {}, "methods": {}, "info": info}
    counts = [sum(a["correct"] for a in groups[k]) for k in keys]
    res["correct_count_distribution"] = {str(c): counts.count(c) for c in sorted(set(counts))}
    for m in LABELS:
        if m not in pa:
            res["methods"][m] = {"missing": True}
            continue
        r = x2_pairs.summarize(pa[m], n_boot, seed)
        r["share_pairs_tied_or_missing"] = tie_share(groups, keys, conf[m])
        r["by_depth"] = {str(d): x2_pairs.summarize(pa[m][[i for i, k in enumerate(keys) if k[1] == d]], n_boot, seed)
                         for d in sorted({k[1] for k in keys})}
        res["methods"][m] = r
    p = {m: res["methods"][m].get("p_one_sided_gt_half", 1.0) if not res["methods"][m].get("missing") else 1.0
         for m in PRIMARY}
    res["holm"] = holm(p, ac["alpha"])
    zero = np.full(len(keys), 0.5)
    get = lambda m: pa.get(m, zero)
    res["paired"] = {"probe - sft_sigma": paired(get("probe"), get("sft_sigma"), n_boot, seed),
                     "probe - random_init_probe": paired(get("probe"), get("random_init_probe"), n_boot, seed),
                     "a5_p - probe": paired(get("a5_p"), get("probe"), n_boot, seed)}
    po = res["methods"].get("problem_only", {})
    res["problem_only_check"] = {"pairwise_accuracy": po.get("pairwise_accuracy"),
                                 "pass": po.get("pairwise_accuracy") == 0.5}
    res["reading"] = reading(res["holm"], res["paired"])
    out = REPO_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "summary.md").write_text(render(res))
    return res


def tie_share(groups: dict, keys: list, conf: dict) -> float:
    """Share of all pairs that are ties or have a missing confidence."""
    t = n = 0
    for k in keys:
        c = [conf.get((tuple(k), a["sample"])) for a in groups[k] if a["correct"]]
        w = [conf.get((tuple(k), a["sample"])) for a in groups[k] if not a["correct"]]
        for x in c:
            for y in w:
                n += 1
                t += x is None or y is None or x == y
    return t / n if n else float("nan")


def render(res: dict) -> str:
    L = [f"# X2: within-problem correctness pairs (D74)", "",
         f"{res['n_kept']} kept problems of {res['n_candidates_sampled']} sampled candidates; {res['n_pairs']} "
         f"(correct, incorrect) pairs. Correct answers per kept problem: {res['correct_count_distribution']}.", "",
         f"**Reading (D74):** {res['reading']['outcome']}.", "",
         "| method | pairwise accuracy [95% CI] | p (> 0.5) | Holm | ties / missing |", "|---|---|---|---|---|"]
    for m, label in LABELS.items():
        r = res["methods"][m]
        if r.get("missing"):
            L.append(f"| {label} | missing | – | {'not supported (p = 1)' if m in PRIMARY else '–'} | – |")
            continue
        h = res["holm"].get(m)
        hs = "–" if h is None else ("supported" if h["rejected"] else "not supported")
        L.append(f"| {label} | {r['pairwise_accuracy']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] | "
                 f"{r['p_one_sided_gt_half']:.3g} | {hs} | {r['share_pairs_tied_or_missing']:.1%} |")
    L += ["", "| paired difference (over problems) | mean | 95% CI |", "|---|---|---|"]
    for name, d in res["paired"].items():
        L.append(f"| {name} | {d['mean_diff']:+.3f} | [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] |")
    depths = sorted({d for r in res["methods"].values() if not r.get("missing") for d in r["by_depth"]})
    L += ["", "| method | " + " | ".join(f"depth {d}" for d in depths) + " |", "|---|" + "---|" * len(depths)]
    for m, label in LABELS.items():
        r = res["methods"][m]
        if not r.get("missing"):
            L.append(f"| {label} | " + " | ".join(
                f"{r['by_depth'][d]['pairwise_accuracy']:.3f} (n {r['by_depth'][d]['n_problems']})"
                if d in r["by_depth"] else "–" for d in depths) + " |")
    pi = res["info"].get("probe", {})
    if "layer" in pi:
        L += ["", f"Probe (fit set, {pi['n_fit']} calibration completions): layer {pi['layer']}, C {pi['C']}, CV AUROC "
                  f"{pi['cv_auroc']:.3f}. Random init: layer {res['info']['random_init_probe'].get('layer')}, CV AUROC "
                  f"{res['info']['random_init_probe'].get('cv_auroc', float('nan')):.3f}."]
    L.append(f"Problem-only check (must be exactly 0.5): {res['problem_only_check']}.")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["analyze"])
    ap.parse_args()
    print(render(analyze()))


if __name__ == "__main__":
    main()

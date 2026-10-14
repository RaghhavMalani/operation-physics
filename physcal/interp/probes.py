"""Linear (and MLP) probes on pre-answer hidden states: does the SFT-only model know its own error?

    uv run python -m physcal.interp.probes            # CPU; needs physcal.interp.extract's output

Implements PREREGISTRATION.md family P exactly: per layer, standardize on the fit set, choose the regularization by
5-fold CV on the fit set, choose the layer by its CV score, refit on the whole fit set, then score the test set
(the frozen eval completions, never used for fitting). Labels: (i) correct within 1% (AUROC), (ii) log10|error|
(Spearman ρ). Baselines: stated σ, (family, depth) metadata. Controls: shuffled fit labels, random-init model.
Paired bootstrap on test problems; Holm over P1–P3. Writes `<output_dir>/probe_results.json` and
`probe_by_layer.png`.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from physcal.config import REPO_ROOT, load_config

TASKS = ("correct", "log_error")


def rho(a, b) -> float:
    return float(spearmanr(a, b).statistic)


def metric(task: str, y, score) -> float:
    return float(roc_auc_score(y, score)) if task == "correct" else rho(score, y)


def load(root: Path, model: str, set_name: str) -> tuple[list[np.ndarray], list[dict]]:
    d = root / "hidden" / model / set_name
    index = json.loads((d / "index.json").read_text())
    layers = [np.load(p).astype(np.float32) for p in sorted(d.glob("layer_*.npy"))]
    return layers, index


def _estimator(task: str, kind: str, reg: float, seed: int, pc: dict):
    if kind == "linear":
        return (LogisticRegression(C=reg, max_iter=2000) if task == "correct" else Ridge(alpha=reg))
    cls = MLPClassifier if task == "correct" else MLPRegressor
    return cls(hidden_layer_sizes=(pc["mlp_hidden"],), alpha=reg, early_stopping=True, max_iter=500,
               random_state=seed)


def _scores(est, task: str, X) -> np.ndarray:
    return est.predict_proba(X)[:, 1] if task == "correct" else est.predict(X)


def _grid(task: str, kind: str, pc: dict) -> list[float]:
    if kind == "mlp":
        return pc["mlp_alpha"]
    return pc["logistic_C"] if task == "correct" else pc["ridge_alpha"]


def cv_layer(X: np.ndarray, y: np.ndarray, task: str, kind: str, seed: int, pc: dict) -> dict:
    """Best regularization for one layer by k-fold CV on the fit set; returns it with its CV score."""
    folds = (StratifiedKFold if task == "correct" else KFold)(pc["cv_folds"], shuffle=True, random_state=seed)
    best = None
    for reg in _grid(task, kind, pc):
        oof = np.zeros(len(y))
        for tr, va in folds.split(X, y):
            sc = StandardScaler().fit(X[tr])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                est = _estimator(task, kind, reg, seed, pc).fit(sc.transform(X[tr]), y[tr])
            oof[va] = _scores(est, task, sc.transform(X[va]))
        cv = metric(task, y, oof)
        if best is None or cv > best["cv"]:
            best = {"reg": reg, "cv": cv}
    return best


def fit_predict(Xf, yf, Xt, task: str, kind: str, reg: float, seed: int, pc: dict) -> np.ndarray:
    sc = StandardScaler().fit(Xf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        est = _estimator(task, kind, reg, seed, pc).fit(sc.transform(Xf), yf)
    return _scores(est, task, sc.transform(Xt))


def probe(fit_layers, test_layers, yf, yt, task: str, kind: str, seed: int, pc: dict,
          layers: list[int] | None = None) -> dict:
    """CV-select per layer, pick the best layer by CV, refit, score the test set. Per-layer test scores are
    reported for the figure only; selection never looks at them."""
    layers = list(range(len(fit_layers))) if layers is None else layers
    cvs = Parallel(n_jobs=pc["n_jobs"])(delayed(cv_layer)(fit_layers[k], yf, task, kind, seed, pc) for k in layers)
    preds = Parallel(n_jobs=pc["n_jobs"])(
        delayed(fit_predict)(fit_layers[k], yf, test_layers[k], task, kind, c["reg"], seed, pc)
        for k, c in zip(layers, cvs))
    per_layer = [{"layer": k, "reg": c["reg"], "cv": c["cv"], "test": metric(task, yt, p)}
                 for k, c, p in zip(layers, cvs, preds)]
    i = int(np.argmax([c["cv"] for c in cvs]))
    return {"selected_layer": layers[i], "reg": cvs[i]["reg"], "cv": cvs[i]["cv"],
            "test": metric(task, yt, preds[i]), "test_scores": preds[i].tolist(), "per_layer": per_layer}


def metadata_probe(idx_fit, idx_test, yf, yt, task: str, seed: int, pc: dict) -> dict:
    enc = OneHotEncoder(handle_unknown="ignore").fit([[r["family"], r["depth"]] for r in idx_fit])
    Xf = enc.transform([[r["family"], r["depth"]] for r in idx_fit]).toarray()
    Xt = enc.transform([[r["family"], r["depth"]] for r in idx_test]).toarray()  # depth 5 is unseen -> family only
    best = cv_layer(Xf, yf, task, "linear", seed, pc)
    scores = fit_predict(Xf, yf, Xt, task, "linear", best["reg"], seed, pc)
    return {"reg": best["reg"], "cv": best["cv"], "test": metric(task, yt, scores), "test_scores": scores.tolist()}


def stated_sigma_scores(idx: list[dict], task: str, floor: float) -> np.ndarray:
    """Correctness: −log10 σ (more confident = higher). log|error|: σ itself (Spearman is rank-based)."""
    s = np.array([max(r["rel_sigma"] or 0.0, floor) for r in idx])
    return -np.log10(s) if task == "correct" else s


def paired_bootstrap(task: str, y, a, b, n_boot: int, seed: int) -> dict:
    """One-sided p for metric(a) > metric(b) on resampled problems, plus percentile CIs."""
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    da, db = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if task == "correct" and len(np.unique(y[i])) < 2:
            continue
        da.append(metric(task, y[i], a[i]))
        db.append(metric(task, y[i], b[i]))
    da, db = np.array(da), np.array(db)
    diff = da - db
    return {"a": metric(task, y, a), "b": metric(task, y, b), "diff": metric(task, y, a) - metric(task, y, b),
            "diff_ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
            "a_ci": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
            "p_one_sided": float(np.mean(diff <= 0)), "n_boot": int(len(diff))}


def holm(pvals: dict[str, float], alpha: float) -> dict[str, dict]:
    order = sorted(pvals, key=pvals.get)
    out, still = {}, True
    for rank, name in enumerate(order):
        thr = alpha / (len(order) - rank)
        still = still and pvals[name] <= thr
        out[name] = {"p": pvals[name], "holm_threshold": thr, "reject_null": still}
    return out


def plot(res: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from physcal.eval.calibration import GRID, REFERENCE, SERIES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, _style

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    titles = {"correct": "correct within 1%: test AUROC by layer", "log_error": "log|error|: test Spearman ρ by layer"}
    lines = (("sft", "SFT-only model", SERIES[0]), ("random_init", "random-init model (control)", SERIES[1]),
             ("sft_shuffled", "shuffled labels (control)", SERIES[2]))
    for ax, task in zip(axes, TASKS):
        _style(ax)
        for key, label, color in lines:
            r = res["linear"][task].get(key)
            if not r:
                continue
            xs = [p["layer"] for p in r["per_layer"]]
            ax.plot(xs, [p["test"] for p in r["per_layer"]], color=color, linewidth=2, label=label)
            sel = r["selected_layer"]
            ax.plot([sel], [r["test"]], "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        for key, label in (("stated_sigma", "stated σ"), ("metadata", "family + depth")):
            v = res["baselines"][task][key]["test"]
            ax.axhline(v, color=REFERENCE, linewidth=1)
            ax.text(0.2, v, f" {label} {v:.2f}", color=TEXT_SECONDARY, fontsize=8, va="bottom")
        ax.axhline(0.5 if task == "correct" else 0.0, color=GRID, linewidth=1)
        ax.set_title(titles[task], color=TEXT_PRIMARY, fontsize=10, loc="left")
        ax.set_xlabel("layer (0 = embeddings)", color=TEXT_SECONDARY, fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY, loc="upper right", ncol=3)
    fig.suptitle("Linear probes on the pre-answer hidden state (dot = layer chosen by CV on the fit set)",
                 color=TEXT_PRIMARY, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130, facecolor=SURFACE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-mlp", action="store_true")
    args = ap.parse_args()
    cfg = load_config("probe_sft")
    pc = cfg["probes"]
    seed, nb = pc["seed"], pc["bootstrap"]
    root = REPO_ROOT / cfg["output_dir"]
    fit, idx_fit = load(root, "sft", "fit")
    test, idx_test = load(root, "sft", "test")
    rfit, ridx_fit = load(root, "random_init", "fit")
    rtest, _ = load(root, "random_init", "test")
    assert [r["key"] for r in ridx_fit] == [r["key"] for r in idx_fit]
    labels = {t: (np.array([r[t] for r in idx_fit]), np.array([r[t] for r in idx_test])) for t in TASKS}
    perm = np.random.default_rng(seed).permutation(len(idx_fit))
    res: dict = {"n_fit": len(idx_fit), "n_test": len(idx_test),
                 "fit_correct_rate": float(labels["correct"][0].mean()),
                 "test_correct_rate": float(labels["correct"][1].mean()),
                 "test_splits": {s: sum(r["split"] == s for r in idx_test) for s in {r["split"] for r in idx_test}},
                 "linear": {}, "mlp": {}, "baselines": {}, "tests": {}}
    for task in TASKS:
        yf, yt = labels[task]
        print(f"[{task}] linear probes", flush=True)
        res["linear"][task] = {"sft": probe(fit, test, yf, yt, task, "linear", seed, pc),
                               "random_init": probe(rfit, rtest, yf, yt, task, "linear", seed, pc),
                               "sft_shuffled": probe(fit, test, yf[perm], yt, task, "linear", seed, pc)}
        res["baselines"][task] = {
            "stated_sigma": {"test": metric(task, yt, stated_sigma_scores(idx_test, task, pc["log_error_floor"]))},
            "metadata": metadata_probe(idx_fit, idx_test, yf, yt, task, seed, pc)}
        if not args.skip_mlp:
            print(f"[{task}] MLP probes", flush=True)
            sel = res["linear"][task]["sft"]["selected_layer"]
            mlp = probe(fit, test, yf, yt, task, "mlp", seed, pc)
            k = mlp["selected_layer"]
            res["mlp"][task] = {"sft": mlp,
                                "random_init": probe(rfit, rtest, yf, yt, task, "mlp", seed, pc, layers=[k]),
                                "sft_shuffled": probe(fit, test, yf[perm], yt, task, "mlp", seed, pc, layers=[k]),
                                "linear_selected_layer": sel}

    # pre-registered tests (pooled test set)
    yc, ye = labels["correct"][1], labels["log_error"][1]
    lin_c, lin_e = res["linear"]["correct"]["sft"], res["linear"]["log_error"]["sft"]
    sig_c = stated_sigma_scores(idx_test, "correct", pc["log_error_floor"])
    sig_e = stated_sigma_scores(idx_test, "log_error", pc["log_error_floor"])
    t = res["tests"]
    t["P1_probe_vs_stated_sigma_auroc"] = paired_bootstrap("correct", yc, lin_c["test_scores"], sig_c, nb, seed)
    t["P2_probe_vs_metadata_auroc"] = paired_bootstrap("correct", yc, lin_c["test_scores"],
                                                       res["baselines"]["correct"]["metadata"]["test_scores"], nb, seed)
    t["P3_probe_vs_stated_sigma_spearman"] = paired_bootstrap("log_error", ye, lin_e["test_scores"], sig_e, nb, seed)
    t["holm"] = holm({k: t[k]["p_one_sided"] for k in ("P1_probe_vs_stated_sigma_auroc", "P2_probe_vs_metadata_auroc",
                                                         "P3_probe_vs_stated_sigma_spearman")}, 0.05)
    shuf = res["linear"]["correct"]["sft_shuffled"]["test_scores"]
    t["sanity_shuffled_auroc"] = paired_bootstrap("correct", yc, shuf, np.zeros(len(yc)) + 0.5, nb, seed)
    t["sanity_real_vs_random_init_auroc"] = paired_bootstrap(
        "correct", yc, lin_c["test_scores"], res["linear"]["correct"]["random_init"]["test_scores"], nb, seed)
    # secondary: per test split
    t["per_split"] = {}
    for split in res["test_splits"]:
        m = np.array([r["split"] == split for r in idx_test])
        t["per_split"][split] = {
            "probe_auroc": metric("correct", yc[m], np.array(lin_c["test_scores"])[m]),
            "stated_sigma_auroc": metric("correct", yc[m], sig_c[m]),
            "metadata_auroc": metric("correct", yc[m], np.array(res["baselines"]["correct"]["metadata"]["test_scores"])[m]),
            "probe_spearman": metric("log_error", ye[m], np.array(lin_e["test_scores"])[m]),
            "stated_sigma_spearman": metric("log_error", ye[m], sig_e[m]),
            "correct_rate": float(yc[m].mean()), "n": int(m.sum())}
    (root / "probe_results.json").write_text(json.dumps(res, indent=2))
    plot(res, root / "probe_by_layer.png")
    print(json.dumps({k: v for k, v in t.items() if k != "per_split"}, indent=1))


if __name__ == "__main__":
    main()

"""EXPLORATORY (not pre-registered): where does the probe's correctness signal come from? (FINDINGS #7)

    uv run python scripts/probe_exploratory.py

The pre-registered sanity check failed: a probe on a randomly initialized model is as good as one on the SFT-only
model. These baselines ask what in the *text* predicts correctness, all fit on the fit set and scored on the same
test set:
- template: one-hot (family, target symbol, set of given symbols), i.e. which chain the problem uses;
- bag of words: TF-IDF over the completion's whitespace tokens, logistic regression;
- length: completion length in characters (sign chosen on the fit set);
- real vs random-init on log|error| (the sanity check only covered correctness).
Writes results/probe_sft/exploratory.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physcal.config import load_config  # noqa: E402
from physcal.interp.probes import paired_bootstrap  # noqa: E402


def tape_records(paths):
    out = {}
    for p in paths:
        for line in (ROOT / p).open():
            r = json.loads(line)
            pr = r["problem"]
            out[(pr["family"], pr["depth"], pr["seed"])] = {
                "template": f"{pr['family']}|{pr['target_symbol']}|{','.join(sorted(g['symbol'] for g in pr['givens']))}",
                "completion": r["completion"]}
    return out


def best_logistic(Xf, yf, grid, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    scores = {C: roc_auc_score(yf, cross_val_predict(LogisticRegression(C=C, max_iter=3000), Xf, yf, cv=cv,
                                                     method="predict_proba")[:, 1]) for C in grid}
    return max(scores, key=scores.get), scores


def main():
    cfg = load_config("probe_sft")
    seed, grid, nb = cfg["probes"]["seed"], cfg["probes"]["logistic_C"] + [10.0], cfg["probes"]["bootstrap"]
    root = ROOT / cfg["output_dir"]
    idx_fit = json.loads((root / "hidden/sft/fit/index.json").read_text())
    idx_test = json.loads((root / "hidden/sft/test/index.json").read_text())
    res = json.loads((root / "probe_results.json").read_text())
    meta = tape_records(cfg["fit_tapes"] + cfg["test_tapes"])
    fit = [meta[tuple(r["key"])] for r in idx_fit]
    test = [meta[tuple(r["key"])] for r in idx_test]
    yf = np.array([r["correct"] for r in idx_fit])
    yt = np.array([r["correct"] for r in idx_test])
    split = np.array([r["split"] for r in idx_test])
    out = {"note": "exploratory, not pre-registered"}

    enc = OneHotEncoder(handle_unknown="ignore").fit([[f["template"]] for f in fit])
    Xf, Xt = enc.transform([[f["template"]] for f in fit]), enc.transform([[t["template"]] for t in test])
    C, _ = best_logistic(Xf, yf, grid, seed)
    s_tmpl = LogisticRegression(C=C, max_iter=3000).fit(Xf, yf).predict_proba(Xt)[:, 1]
    seen = np.array([t["template"] in {f["template"] for f in fit} for t in test])
    out["template"] = {"C": C, "n_templates_fit": len(enc.categories_[0]),
                       "test_share_template_seen": float(seen.mean()),
                       "auroc_id": float(roc_auc_score(yt[split == "id"], s_tmpl[split == "id"])),
                       "auroc_id_seen_only": float(roc_auc_score(yt[(split == "id") & seen], s_tmpl[(split == "id") & seen]))}

    vec = TfidfVectorizer(token_pattern=r"\S+", min_df=3, sublinear_tf=True).fit([f["completion"] for f in fit])
    Xf, Xt = vec.transform([f["completion"] for f in fit]), vec.transform([t["completion"] for t in test])
    C, _ = best_logistic(Xf, yf, grid, seed)
    s_bow = LogisticRegression(C=C, max_iter=3000).fit(Xf, yf).predict_proba(Xt)[:, 1]
    out["bag_of_words"] = {"C": C, "auroc": float(roc_auc_score(yt, s_bow)),
                           **{f"auroc_{s}": float(roc_auc_score(yt[split == s], s_bow[split == s])) for s in set(split)}}

    lf = np.array([len(f["completion"]) for f in fit])
    sign = 1.0 if roc_auc_score(yf, lf) >= 0.5 else -1.0
    lt = sign * np.array([len(t["completion"]) for t in test])
    out["length"] = {"sign": sign, "auroc": float(roc_auc_score(yt, lt))}

    probe = np.array(res["linear"]["correct"]["sft"]["test_scores"])
    out["probe_vs_bag_of_words_auroc"] = paired_bootstrap("correct", yt, probe, s_bow, nb, seed)
    ye = np.array([r["log_error"] for r in idx_test])
    out["real_vs_random_init_log_error_spearman"] = paired_bootstrap(
        "log_error", ye, res["linear"]["log_error"]["sft"]["test_scores"],
        res["linear"]["log_error"]["random_init"]["test_scores"], nb, seed)
    (root / "exploratory.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

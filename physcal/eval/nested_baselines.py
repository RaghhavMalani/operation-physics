"""D50/A5: nested surface-feature baselines for stated uncertainty (supersede D49's judgement rule).

    nice uv run python -m physcal.eval.nested_baselines fit

Feature sets, nested: (1) metadata (family, depth); (2) problem text; (3) + the `<answer>` block; (4) + the
reasoning (completion text before `<answer>`). TF-IDF over whitespace tokens, one vocabulary per text field. For
each set, a linear model and a small GBT predict
- σ: regression of y = ln max(|ln(μ/truth)|, floor), then σ_log = c·exp(ŷ) with c fitted for mean crps_log on
  out-of-fold predictions (σ in SI = σ_log·|μ|; crps_log scores ln truth against N(ln μ, σ_log²));
- p = P(correct within the verifier's binary tolerance): classification, scored by Brier.
Selection: 5-fold CV grouped by chain template on the calibration split; "strongest" = best CV mean crps_log (σ)
or Brier (p), chosen before anything is applied to an eval set. Every candidate is then refit on the whole
calibration split and saved; `apply_sigma` gives any completions the strongest σ (RL, E1, SFT eval).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

from physcal.config import REPO_ROOT, load_config
from physcal.eval import posthoc
from physcal.eval.posthoc_cond import _global_log_scale, _log_terms
from physcal.tape.tape import read_meta, read_tape
from physcal.verify.reward import verifier_version
from physcal.verify.scoring import sigma_floor

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
FIELDS = {"metadata": (), "problem": ("problem",), "problem_answer": ("problem", "answer"),
          "problem_answer_reasoning": ("problem", "answer", "reasoning")}


def _config() -> dict:
    return load_config("nested_baselines")


def record(problem: dict, completion: str, b: dict, tolerance: float, split: str = "all") -> dict:
    """Text fields and labels of one completion. `ok` = parsed answer with a positive μ (σ-model rows)."""
    m = ANSWER_RE.search(completion)
    ok = b["reason"] == "ok" and b.get("mu_si") is not None and b["mu_si"] > 0
    return {"split": split, "key": (problem["family"], problem["depth"], problem["seed"]),
            "family": problem["family"], "depth": problem["depth"],
            "template": f"{problem['family']}|{problem['target_symbol']}|"
                        f"{','.join(sorted(g['symbol'] for g in problem['givens']))}",
            "problem": problem["text"], "answer": m.group(1) if m else "",
            "reasoning": completion[:m.start()] if m else completion,
            "ok": ok, "mu": b["mu_si"] if ok else None, "truth": b["truth_si"],
            "stated_sigma": b.get("sigma_si") if ok else None,
            "correct": bool(ok and b["abs_rel_error"] <= tolerance)}


def tape_records(tape: Path, tolerance: float) -> list[dict]:
    return [record(r["problem"], r["completion"], r["breakdown"], tolerance, r.get("split", "all"))
            for r in read_tape(tape)]


def log_error(recs: list[dict], floor: float) -> np.ndarray:
    return np.array([math.log(max(abs(math.log(r["mu"] / r["truth"])), floor)) for r in recs])


class Featurizer:
    """One encoder per field of a feature set; `dense_max` limits each text vocabulary (for the GBT)."""

    def __init__(self, feature_set: str, text_cfg: dict, dense: bool):
        self.feature_set, self.dense = feature_set, dense
        self.fields = FIELDS[feature_set]
        self.meta = OneHotEncoder(handle_unknown="ignore") if feature_set == "metadata" else None
        self.vecs = {f: TfidfVectorizer(token_pattern=text_cfg["token_pattern"], lowercase=False,
                                        ngram_range=tuple(text_cfg["ngram_range"]), min_df=text_cfg["min_df"],
                                        sublinear_tf=text_cfg["sublinear_tf"],
                                        max_features=text_cfg["gbt_max_features"] if dense else None)
                     for f in self.fields}

    def fit(self, recs: list[dict]) -> "Featurizer":
        if self.meta is not None:
            self.meta.fit([[r["family"], r["depth"]] for r in recs])
        for f, v in self.vecs.items():
            v.fit([r[f] for r in recs])
        return self

    def transform(self, recs: list[dict]):
        if self.meta is not None:
            X = self.meta.transform([[r["family"], r["depth"]] for r in recs])
        else:
            X = sp.hstack([v.transform([r[f] for r in recs]) for f, v in self.vecs.items()]).tocsr()
        return X.toarray() if self.dense else X


def _model(kind: str, target: str, hp, seed: int):
    if kind == "linear":
        return Ridge(alpha=hp) if target == "sigma" else LogisticRegression(C=hp, max_iter=5000)
    cls = HistGradientBoostingRegressor if target == "sigma" else HistGradientBoostingClassifier
    return cls(random_state=seed, **hp)


class Candidate:
    """A fitted (feature set, model kind, target) baseline. For σ, `c` scales exp(ŷ) to σ_log."""

    def __init__(self, feature_set: str, kind: str, target: str, hp, cfg: dict):
        self.feature_set, self.kind, self.target, self.hp, self.cfg = feature_set, kind, target, hp, cfg
        self.c: float | None = None

    def fit(self, recs: list[dict], y: np.ndarray) -> "Candidate":
        self.feat = Featurizer(self.feature_set, self.cfg["text"], dense=self.kind == "gbt").fit(recs)
        self.model = _model(self.kind, self.target, self.hp, self.cfg["seed"]).fit(self.feat.transform(recs), y)
        return self

    def raw(self, recs: list[dict]) -> np.ndarray:
        X = self.feat.transform(recs)
        return self.model.predict(X) if self.target == "sigma" else self.model.predict_proba(X)[:, 1]

    def sigma_log(self, recs: list[dict]) -> np.ndarray:
        return self.c * np.exp(self.raw(recs))

    @property
    def name(self) -> str:
        return f"{self.feature_set}/{self.kind}"


def _hps(kind: str, target: str, grids: dict) -> list:
    if kind == "gbt":
        return grids["gbt"]
    return grids["ridge_alpha"] if target == "sigma" else grids["logistic_C"]


def _fit_scale(recs: list[dict], pred: np.ndarray, sc: dict, fit_cfg: dict) -> float:
    rows = [{"mu": r["mu"], "truth": r["truth"], "sigma": float(np.exp(p)) * r["mu"]} for r, p in zip(recs, pred)]
    L, a, live = _log_terms(rows)
    return math.exp(_global_log_scale(L, a, live, sigma_floor("crps_log", sc), sc["crps_clip"], fit_cfg))


def sigma_summary(recs: list[dict], sigma_log: np.ndarray, sc: dict) -> dict:
    """crps_log, coverage and ρ(σ, |error|) of σ_log on answered rows (failures excluded here; see `score_all`)."""
    rows = [{"mu": r["mu"], "truth": r["truth"], "sigma": s * r["mu"]} for r, s in zip(recs, sigma_log)]
    out = posthoc.summary(rows, 1.0, sc)
    err = [abs(math.log(r["mu"] / r["truth"])) for r in recs]
    out["spearman_sigma_error"] = float(spearmanr(sigma_log, err).statistic)
    out["median_sigma_log"] = float(np.median(sigma_log))
    return out


def p_summary(y: np.ndarray, p: np.ndarray) -> dict:
    return {"brier": float(brier_score_loss(y, p)), "auroc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else None}


def select(recs: list[dict], sc: dict, cfg: dict, fit_cfg: dict) -> dict:
    """CV over every (feature set, model kind, hyperparameter) for both targets; returns the CV table."""
    ok = [r for r in recs if r["ok"]]
    y_sig = log_error(ok, cfg["log_error_floor"])
    y_p = np.array([r["correct"] for r in recs], dtype=int)
    groups = {"sigma": [r["template"] for r in ok], "p": [r["template"] for r in recs]}
    data = {"sigma": (ok, y_sig), "p": (recs, y_p)}
    table: dict = {"sigma": {}, "p": {}}
    for target in ("sigma", "p"):
        rs, y = data[target]
        folds = list(GroupKFold(cfg["cv"]["folds"]).split(rs, y, groups[target]))
        for fs in cfg["feature_sets"]:
            for kind in ("linear", "gbt"):
                best = None
                for hp in _hps(kind, target, cfg["grids"]):
                    oof = np.zeros(len(rs))
                    for tr, te in folds:
                        cand = Candidate(fs, kind, target, hp, cfg).fit([rs[i] for i in tr], y[tr])
                        oof[te] = cand.raw([rs[i] for i in te])
                    if target == "sigma":
                        c = _fit_scale(rs, oof, sc, fit_cfg)
                        metric = sigma_summary(rs, c * np.exp(oof), sc)
                        score, entry = metric["mean_crps_log"], {"hp": hp, "c": c, "cv": metric}
                    else:
                        metric = p_summary(y, oof)
                        score, entry = -metric["brier"], {"hp": hp, "cv": metric}
                    if best is None or score > best[0]:
                        best = (score, entry)
                table[target][f"{fs}/{kind}"] = best[1]
    table["strongest"] = {
        "sigma": max(table["sigma"], key=lambda k: table["sigma"][k]["cv"]["mean_crps_log"]),
        "p": min(table["p"], key=lambda k: table["p"][k]["cv"]["brier"])}
    return table


def refit_all(recs: list[dict], table: dict, cfg: dict) -> dict[str, dict[str, Candidate]]:
    ok = [r for r in recs if r["ok"]]
    y_sig = log_error(ok, cfg["log_error_floor"])
    y_p = np.array([r["correct"] for r in recs], dtype=int)
    out: dict = {"sigma": {}, "p": {}}
    for target, (rs, y) in (("sigma", (ok, y_sig)), ("p", (recs, y_p))):
        for name, entry in table[target].items():
            fs, kind = name.split("/")
            cand = Candidate(fs, kind, target, entry["hp"], cfg).fit(rs, y)
            cand.c = entry.get("c")
            out[target][name] = cand
    return out


def load_models() -> tuple[dict, dict]:
    out = REPO_ROOT / _config()["output_dir"]
    return joblib.load(out / "models.joblib"), json.loads((out / "selection.json").read_text())


def apply_sigma(recs: list[dict], name: str | None = None) -> np.ndarray:
    """σ_log of the strongest (or the named) σ baseline for answered records (NaN for failures)."""
    models, sel = load_models()
    cand = models["sigma"][name or sel["strongest"]["sigma"]]
    out = np.full(len(recs), np.nan)
    idx = [i for i, r in enumerate(recs) if r["ok"]]
    if idx:
        out[idx] = cand.sigma_log([recs[i] for i in idx])
    return out


def report_rows(records_by_split: dict[str, list[dict]], scoring_cfg: dict, generator_version: str,
                verifier: str) -> dict:
    """Eval-report rows (D50/A5): the strongest σ baseline applied to this model's completions, and to the SFT-only
    completions of the same frozen split (when generator and verifier versions match). Same denominators as the
    other baseline rows: failures (and μ <= 0) score the floor and count as not covered."""
    out_dir = REPO_ROOT / _config()["output_dir"]
    if not (out_dir / "models.joblib").exists():
        return {"available": False}
    models, sel = load_models()
    name = sel["strongest"]["sigma"]
    cand = models["sigma"][name]
    tol = scoring_cfg["binary_tolerance"]

    def summ(recs: list[dict]) -> dict:
        idx = [i for i, r in enumerate(recs) if r["ok"]]
        sig = cand.sigma_log([recs[i] for i in idx]) if idx else []
        rows = [{"mu": None, "sigma": None, "truth": r["truth"]} for r in recs]
        for i, s_ in zip(idx, sig):
            rows[i] = {"mu": recs[i]["mu"], "sigma": float(s_) * recs[i]["mu"], "truth": recs[i]["truth"]}
        return posthoc.summary(rows, 1.0, scoring_cfg)

    cfg = _config()
    eval_meta = read_meta(REPO_ROOT / cfg["eval_tape"])
    comparable = (eval_meta["generator_version"] == generator_version
                  and verifier_version(eval_meta["reward_config"]) == verifier)
    sft = tape_records(REPO_ROOT / cfg["eval_tape"], tol) if comparable else []
    out = {"available": True, "strongest_sigma": name, "baseline_comparable": comparable, "splits": {}}
    for split, rs in records_by_split.items():
        recs = [record(r["problem"], r["completion"], r["breakdown"], tol, split) for r in rs]
        entry = {"this_model_a5": summ(recs)}
        if comparable and any(r["split"] == split for r in sft):
            entry["sft_only_a5"] = summ([r for r in sft if r["split"] == split])
        out["splits"][split] = entry
    return out


def eval_report(recs: list[dict], models: dict, sc: dict) -> dict:
    """Every candidate on these completions, per split and pooled; failures reported separately (they score the
    floor whatever the σ)."""
    out = {}
    for split in sorted({r["split"] for r in recs}) + ["pooled"]:
        rs = recs if split == "pooled" else [r for r in recs if r["split"] == split]
        ok = [r for r in rs if r["ok"]]
        y = np.array([r["correct"] for r in rs], dtype=int)
        stated = np.array([max(r["stated_sigma"], 0.0) / r["mu"] for r in ok])
        entry = {"n": len(rs), "n_ok": len(ok), "stated_sigma": sigma_summary(ok, stated, sc),
                 "sigma": {n: sigma_summary(ok, c.sigma_log(ok), sc) for n, c in models["sigma"].items()},
                 "p": {n: p_summary(y, c.raw(rs)) for n, c in models["p"].items()}}
        out[split] = entry
    return out


def write_scaled_tape(eval_tape: Path, cand: Candidate, out_path: Path) -> None:
    """Derived JSONL (as posthoc.scaled_tape): the eval completions with σ replaced by the baseline's σ_log·|μ|."""
    recs = tape_records(eval_tape, read_meta(eval_tape)["reward_config"]["scoring"]["binary_tolerance"])
    scales = np.ones(len(recs))
    idx = [i for i, r in enumerate(recs) if r["ok"]]
    scales[idx] = cand.sigma_log([recs[i] for i in idx])
    posthoc.scaled_tape(eval_tape, scales, out_path, anchor="answer")


def fit() -> dict:
    cfg = _config()
    fit_cfg = load_config("posthoc")["fit"]
    tapes = [REPO_ROOT / p for p in cfg["fit_tapes"]]
    metas = [read_meta(t) for t in tapes]
    eval_tape = REPO_ROOT / cfg["eval_tape"]
    eval_meta = read_meta(eval_tape)
    versions = {verifier_version(m["reward_config"]) for m in metas + [eval_meta]}
    if len(versions) != 1:
        raise ValueError(f"tapes scored by different verifier versions: {versions}")
    sc = metas[0]["reward_config"]["scoring"]
    tol = sc["binary_tolerance"]
    recs = [r for t in tapes for r in tape_records(t, tol)]
    if max(r["key"][2] for r in recs) >= metas[0]["env_config"]["splits"]["eval_seed_offset"]:
        raise ValueError("calibration set contains eval seeds")
    table = select(recs, sc, cfg, fit_cfg)
    out = REPO_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    selection = {"decision": "D50/A5", "verifier_version": versions.pop(), "n_calibration": len(recs),
                 "n_calibration_ok": sum(r["ok"] for r in recs), "n_templates": len({r["template"] for r in recs}),
                 "cv_table": table, "strongest": table["strongest"]}
    (out / "selection.json").write_text(json.dumps(selection, indent=1, default=float))  # before any eval use
    models = refit_all(recs, table, cfg)
    joblib.dump(models, out / "models.joblib", compress=3)
    evals = tape_records(eval_tape, tol)
    report = {"eval_tape": cfg["eval_tape"], "eval_generator_version": eval_meta["generator_version"],
              "strongest": table["strongest"], "sft_eval": eval_report(evals, models, sc)}
    (out / "eval_sft.json").write_text(json.dumps(report, indent=1, default=float))
    write_scaled_tape(eval_tape, models["sigma"][table["strongest"]["sigma"]], out / "sft_eval_a5_sigma.jsonl")
    (out / "summary.md").write_text(render(selection, report))
    return {"selection": selection, "report": report}


def render(selection: dict, report: dict) -> str:
    t, st = selection["cv_table"], selection["strongest"]
    lines = ["# A5 nested baselines (D50)", "",
             f"Calibration split: {selection['n_calibration']} completions ({selection['n_calibration_ok']} answered), "
             f"{selection['n_templates']} templates; 5-fold CV grouped by template. Strongest σ: **{st['sigma']}**; "
             f"strongest p: **{st['p']}** (chosen on CV before any eval use).", "",
             "| baseline | CV crps_log (σ) | CV 1σ cov | CV ρ(σ, err) | CV Brier (p) | CV AUROC (p) |", "|---|---|---|---|---|---|"]
    for name in t["sigma"]:
        s, p = t["sigma"][name]["cv"], t["p"][name]["cv"]
        lines.append(f"| {name} | {s['mean_crps_log']:.4f} | {s['coverage_1sigma']:.1%} | {s['spearman_sigma_error']:.2f} "
                     f"| {p['brier']:.4f} | {p['auroc']:.3f} |")
    for split, e in report["sft_eval"].items():
        lines += ["", f"## SFT-only frozen eval, {split} (n = {e['n']}, answered {e['n_ok']}; answered rows only)", "",
                  "| σ source | mean crps_log | 1σ cov | 2σ cov | ρ(σ, err) | Brier (p) | AUROC (p) |",
                  "|---|---|---|---|---|---|---|"]
        s = e["stated_sigma"]
        lines.append(f"| stated σ | {s['mean_crps_log']:.4f} | {s['coverage_1sigma']:.1%} | {s['coverage_2sigma']:.1%} "
                     f"| {s['spearman_sigma_error']:.2f} | – | – |")
        for name, s in e["sigma"].items():
            p = e["p"][name]
            lines.append(f"| {name}{' (strongest σ)' if name == st['sigma'] else ''} | {s['mean_crps_log']:.4f} "
                         f"| {s['coverage_1sigma']:.1%} | {s['coverage_2sigma']:.1%} | {s['spearman_sigma_error']:.2f} "
                         f"| {p['brier']:.4f} | {p['auroc'] if p['auroc'] is None else round(p['auroc'], 3)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["fit"])
    ap.parse_args()
    # Call through the package so pickled classes are importable outside `python -m` (not as __main__.Candidate).
    from physcal.eval import nested_baselines

    res = nested_baselines.fit()
    print(render(res["selection"], res["report"]))


if __name__ == "__main__":
    main()

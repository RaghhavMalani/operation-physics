"""Conditional post-hoc σ baselines (D49, amendment of 2026-10-14).

(a) Text-feature post-hoc: σ' = c(x)·σ_eff with log c(x) = b + w·φ(x), φ = binary bag of words of the problem
    text and of the completion (separate vocabularies). Fitted on the calibration split only by maximizing mean
    crps_log − λ/2·|w|², λ chosen by K-fold CV. RL calibration claims are judged against this one.
(a') The same model with σ' = c(x)·|μ| ("direct": it ignores the stated σ). Reported as a sensitivity row.
(b) Oracle per-cell post-hoc: per (family, depth), the crps_log-optimal scale on σ_eff, fitted in-sample on the
    evaluated completions themselves. An upper bound only.

σ_eff = max(σ, f·|μ|), f = the verifier's crps_log σ floor, which is how crps_log already scores a stated σ = 0.

    nice uv run python -m physcal.eval.posthoc_cond fit --eval <SFT-only eval tape>
    nice uv run python -m physcal.eval.posthoc_cond report <eval dir> [<eval dir> ...]

`fit` writes the models (configs/posthoc.yaml → text_feature.model_file), the SFT-only eval numbers under (a), (a')
and (b), and derived JSONLs for paired tests next to it (sft_eval_text_<form>.jsonl, sft_eval_oracle_cell.jsonl).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import ndtr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold

from physcal.config import REPO_ROOT, load_config
from physcal.eval import posthoc
from physcal.tape.tape import read_meta, read_tape
from physcal.verify.reward import verifier_version
from physcal.verify.scoring import sigma_floor

RULE = posthoc.RULE
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _config() -> dict:
    return load_config("posthoc")


def record(problem: dict, completion: str, b: dict, split: str = "all") -> dict:
    """One completion as the post-hoc code sees it, plus the text and the cell; mu/sigma None for failures."""
    ok = b["reason"] == "ok" and b.get("sigma_si") is not None
    return {"split": split, "key": (problem["family"], problem["depth"], problem["seed"]),
            "cell": (problem["family"], problem["depth"]), "problem_text": problem["text"], "completion": completion,
            "mu": b["mu_si"] if ok else None, "sigma": b["sigma_si"] if ok else None, "truth": b["truth_si"]}


def tape_records(tape: Path) -> list[dict]:
    return [record(r["problem"], r["completion"], r["breakdown"], r.get("split", "all")) for r in read_tape(tape)]


def with_sigma_eff(records: list[dict], rel_floor: float) -> list[dict]:
    """Copies with σ raised to rel_floor·|μ| (answered records only)."""
    return [r if r["mu"] is None else {**r, "sigma": max(r["sigma"], rel_floor * abs(r["mu"]))} for r in records]


def anchored(records: list[dict], anchor: str, rel_floor: float) -> list[dict]:
    """What c(x) multiplies: σ_eff ("stated_sigma", form (a)) or |μ| ("answer", form (a'))."""
    if anchor == "stated_sigma":
        return with_sigma_eff(records, rel_floor)
    if anchor == "answer":
        return [r if r["mu"] is None else {**r, "sigma": abs(r["mu"])} for r in records]
    raise ValueError(f"unknown anchor {anchor!r}")


# ---- (a) text-feature scale ----------------------------------------------------------------------------------

def _vectorizer(cfg: dict, vocabulary: list[str] | None = None) -> CountVectorizer:
    return CountVectorizer(binary=True, lowercase=False, token_pattern=cfg["token_pattern"],
                           ngram_range=tuple(cfg["ngram_range"]), min_df=1 if vocabulary else cfg["min_df"],
                           vocabulary=vocabulary)


def _log_terms(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For answered records: L = ln(μ/y), a = σ_eff/μ (crps_log's relative σ before scaling), and a mask of
    records whose score depends on the scale (μ/y > 0; otherwise crps_log gives the clip)."""
    mu = np.array([r["mu"] for r in records], dtype=float)
    y = np.array([r["truth"] for r in records], dtype=float)
    sig = np.array([r["sigma"] for r in records], dtype=float)
    ratio = mu / y
    live = ratio > 0
    L = np.where(live, np.log(np.where(live, ratio, 1.0)), 0.0)
    a = np.where(live, sig / np.where(mu != 0, np.abs(mu), 1.0), 1.0)
    return L, a, live


def loss_and_grad(theta: np.ndarray, X, L: np.ndarray, a: np.ndarray, live: np.ndarray, lam: float,
                  rel_floor: float, clip: float) -> tuple[float, np.ndarray]:
    """Mean crps_log loss (= −score, clipped at `clip`) of σ' = exp(b + X·w)·σ_eff, plus λ/2·|w|², and its gradient
    in θ = (b, w). ∂CRPS/∂s = 2φ(z) − 1/√π, so ∂CRPS/∂ln s = s·(2φ(z) − 1/√π)."""
    n = len(L)
    b, w = theta[0], theta[1:]
    eta = b + X @ w
    s_raw = np.exp(np.clip(eta, -700, 700)) * a
    s = np.maximum(s_raw, rel_floor)
    z = L / s
    pdf = _INV_SQRT_2PI * np.exp(-0.5 * z * z)
    crps = s * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * pdf - _INV_SQRT_PI)
    loss = np.where(live, np.minimum(crps, clip), clip)
    active = live & (crps < clip) & (s_raw > rel_floor)
    g_eta = np.where(active, s * (2.0 * pdf - _INV_SQRT_PI), 0.0) / n
    grad = np.concatenate([[g_eta.sum()], X.T @ g_eta + lam * w])
    return float(loss.mean() + 0.5 * lam * w @ w), grad


class TextScale:
    """log c(x) = b + w_p·φ_problem(x) + w_c·φ_completion(x)."""

    def __init__(self, cfg: dict, vocab_problem: list[str], vocab_completion: list[str], b: float,
                 w: np.ndarray, lam: float):
        self.cfg, self.b, self.w, self.lam = cfg, float(b), np.asarray(w, dtype=float), lam
        self.vocab_problem, self.vocab_completion = vocab_problem, vocab_completion
        self._vp, self._vc = _vectorizer(cfg, vocab_problem), _vectorizer(cfg, vocab_completion)

    def features(self, records: list[dict]):
        return sp.hstack([self._vp.transform([r["problem_text"] for r in records]),
                          self._vc.transform([r["completion"] for r in records])]).tocsr().astype(float)

    def scales(self, records: list[dict]) -> np.ndarray:
        if not records:
            return np.zeros(0)
        return np.exp(self.b + self.features(records) @ self.w)

    def to_json(self) -> dict:
        k = len(self.vocab_problem)
        return {"b": self.b, "lam": self.lam, "settings": self.cfg,
                "problem": dict(zip(self.vocab_problem, self.w[:k].tolist())),
                "completion": dict(zip(self.vocab_completion, self.w[k:].tolist()))}

    @classmethod
    def from_json(cls, d: dict) -> "TextScale":
        vp, vc = list(d["problem"]), list(d["completion"])
        w = np.array([d["problem"][t] for t in vp] + [d["completion"][t] for t in vc])
        return cls(d["settings"], vp, vc, d["b"], w, d["lam"])


def _global_log_scale(L, a, live, rel_floor: float, clip: float, fit_cfg: dict) -> float:
    """Intercept-only start: the best ln c on the posthoc fit grid (vectorized)."""
    X0 = sp.csr_matrix((len(L), 0))
    grid = np.linspace(*fit_cfg["log10_scale_bounds"], fit_cfg["grid_points"]) * math.log(10)
    return float(grid[int(np.argmin([loss_and_grad(np.array([g]), X0, L, a, live, 0.0, rel_floor, clip)[0]
                                     for g in grid]))])


def fit_text_scale(records: list[dict], sc: dict, cfg: dict, fit_cfg: dict, lam: float) -> TextScale:
    """Fit (a) on answered records (σ already σ_eff). Failures don't depend on the scale and are skipped."""
    recs = [r for r in records if r["mu"] is not None]
    vp = _vectorizer(cfg).fit([r["problem_text"] for r in recs])
    vc = _vectorizer(cfg).fit([r["completion"] for r in recs])
    model = TextScale(cfg, vp.get_feature_names_out().tolist(), vc.get_feature_names_out().tolist(), 0.0,
                      np.zeros(len(vp.vocabulary_) + len(vc.vocabulary_)), lam)
    X = model.features(recs)
    L, a, live = _log_terms(recs)
    f, clip = sigma_floor(RULE, sc), sc["crps_clip"]
    theta0 = np.concatenate([[_global_log_scale(L, a, live, f, clip, fit_cfg)], np.zeros(X.shape[1])])
    res = minimize(loss_and_grad, theta0, args=(X, L, a, live, lam, f, clip), jac=True, method="L-BFGS-B",
                   options={"maxiter": cfg["max_iter"], "gtol": cfg["lbfgs_gtol"], "ftol": cfg["lbfgs_ftol"]})
    model.b, model.w = float(res.x[0]), res.x[1:]
    model.fit_info = {"converged": bool(res.success), "iterations": int(res.nit), "objective": float(res.fun),
                      "message": str(res.message)}
    return model


def cv_select(records: list[dict], sc: dict, cfg: dict, fit_cfg: dict) -> tuple[float, dict]:
    """λ with the best K-fold held-out mean crps_log (exact verifier scoring, answered records). Vocabularies are
    refit inside each fold. The table also holds the intercept-only model ("intercept_only", i.e. λ = ∞) for
    reference; it is not a candidate."""
    recs = [r for r in records if r["mu"] is not None]
    folds = list(KFold(cfg["cv_folds"], shuffle=True, random_state=cfg["seed"]).split(recs))
    f, clip = sigma_floor(RULE, sc), sc["crps_clip"]
    table, held0 = {}, []
    for tr, te in folds:
        c0 = math.exp(_global_log_scale(*_log_terms([recs[i] for i in tr]), f, clip, fit_cfg))
        held0 += [o["score"] for o in posthoc.scaled_outcomes([recs[i] for i in te], c0, sc)]
    for lam in cfg["l2_grid"]:
        held = []
        for tr, te in folds:
            m = fit_text_scale([recs[i] for i in tr], sc, cfg, fit_cfg, lam)
            test = [recs[i] for i in te]
            held += [o["score"] for o in posthoc.scaled_outcomes(test, m.scales(test), sc)]
        table[str(lam)] = float(np.mean(held))
    best = max(cfg["l2_grid"], key=lambda lam: table[str(lam)])
    return best, {**table, "intercept_only": float(np.mean(held0))}


# ---- (b) oracle per cell --------------------------------------------------------------------------------------

def oracle_cell_scales(records: list[dict], sc: dict, fit_cfg: dict) -> dict[tuple, float]:
    """Per cell, argmax_c mean crps_log of c·σ_eff, in-sample (σ already σ_eff). Cells with no answer get 1.
    The search range is oracle_cell.log10_scale_bounds (the grid density of `fit_cfg` is kept)."""
    oc = _config()["oracle_cell"]
    lo, hi = fit_cfg["log10_scale_bounds"]
    fit_cfg = {**fit_cfg, "log10_scale_bounds": oc["log10_scale_bounds"],
               "grid_points": int(round((fit_cfg["grid_points"] - 1) * (oc["log10_scale_bounds"][1]
                                        - oc["log10_scale_bounds"][0]) / (hi - lo))) + 1}
    out = {}
    for cell in sorted({r["cell"] for r in records}):
        recs = [r for r in records if r["cell"] == cell]
        out[cell] = posthoc.fit_crps_scale(recs, sc, fit_cfg) if any(r["mu"] is not None for r in recs) else 1.0
    return out


def oracle_scales(records: list[dict], sc: dict, fit_cfg: dict) -> np.ndarray:
    cells = oracle_cell_scales(records, sc, fit_cfg)
    return np.array([cells[r["cell"]] for r in records])


# ---- reports ---------------------------------------------------------------------------------------------------

def conditional_summaries(records: list[dict], sc: dict, models: dict[str, TextScale]) -> dict:
    """Per split: (b) and each text form (`text_<form>`) applied to these completions."""
    cfg_ph = _config()
    f = sigma_floor(RULE, sc)
    eff = with_sigma_eff(records, f)
    forms = {form: anchored(records, cfg_ph["text_feature"]["forms"][form], f) for form in models}
    out = {}
    for split in sorted({r["split"] for r in eff}):
        idx = [i for i, r in enumerate(eff) if r["split"] == split]
        recs = [eff[i] for i in idx]
        out[split] = {"oracle_cell": posthoc.summary(recs, oracle_scales(recs, sc, cfg_ph["fit"]), sc)}
        for form, m in models.items():
            frecs = [forms[form][i] for i in idx]
            out[split][f"text_{form}"] = posthoc.summary(frecs, m.scales(frecs), sc)
    return out


def load_models() -> tuple[dict[str, TextScale], dict | None]:
    path = REPO_ROOT / _config()["text_feature"]["model_file"]
    if not path.exists():
        return {}, None
    d = json.loads(path.read_text())
    return {form: TextScale.from_json(m) for form, m in d["models"].items()}, d


def baseline_rows(records_by_split: dict[str, list[dict]], scoring_cfg: dict, generator_version: str,
                  verifier: str) -> dict:
    """The D49 rows of an eval report: this model × (a) and × (b), and the SFT-only rows on the same frozen split
    when the generator and verifier versions match the stored SFT-only numbers."""
    models, fitted = load_models()
    recs = [record(r["problem"], r["completion"], r["breakdown"], split)
            for split, rs in records_by_split.items() for r in rs]
    own = conditional_summaries(recs, scoring_cfg, models)
    out = {"text_forms": sorted(models), "splits": {}}
    comparable = False
    if fitted is not None:
        out.update(model_file=_config()["text_feature"]["model_file"], primary=fitted["primary"],
                   lam={form: m.lam for form, m in models.items()})
        comparable = (fitted["eval_generator_version"] == generator_version and fitted["verifier_version"] == verifier)
    out["baseline_comparable"] = comparable
    for split, e in own.items():
        entry = {f"this_model_{k}": v for k, v in e.items()}
        if comparable and split in fitted["eval"]:
            entry.update({f"sft_only_{k}": v for k, v in fitted["eval"][split].items() if k != "unscaled"})
        out["splits"][split] = entry
    return out


def fit(eval_tape: Path) -> dict:
    cfg_ph = _config()
    cfg = cfg_ph["text_feature"]
    tapes = [REPO_ROOT / p for p in cfg["fit_tapes"]]
    metas = [read_meta(t) for t in tapes]
    eval_meta = read_meta(eval_tape)
    versions = {verifier_version(m["reward_config"]) for m in metas + [eval_meta]}
    if len(versions) != 1:
        raise ValueError(f"calibration and eval tapes were scored by different verifier versions: {versions}")
    sc = metas[0]["reward_config"]["scoring"]
    f = sigma_floor(RULE, sc)
    calib = [r for t in tapes for r in tape_records(t)]
    eval_offset = metas[0]["env_config"]["splits"]["eval_seed_offset"]
    seeds = [r["key"][2] for r in calib]
    if max(seeds) >= eval_offset:
        raise ValueError("calibration set contains eval seeds")
    if len(set(r["key"] for r in calib)) != len(calib):
        raise ValueError("calibration tapes overlap")
    eff = with_sigma_eff(calib, f)
    glob = math.exp(_global_log_scale(*_log_terms([r for r in eff if r["mu"] is not None]), f, sc["crps_clip"],
                                      cfg_ph["fit"]))
    models, per_form = {}, {}
    for form, anchor in cfg["forms"].items():
        recs = anchored(calib, anchor, f)
        lam, cv_table = cv_select(recs, sc, cfg, cfg_ph["fit"])
        models[form] = fit_text_scale(recs, sc, cfg, cfg_ph["fit"], lam)
        per_form[form] = {"anchor": anchor, "cv": {"folds": cfg["cv_folds"], "mean_crps_log_by_lam": cv_table,
                                                   "lam": lam},
                          "fit_info": models[form].fit_info,
                          "n_features": {"problem": len(models[form].vocab_problem),
                                         "completion": len(models[form].vocab_completion)},
                          "calibration_set_in_sample": posthoc.summary(recs, models[form].scales(recs), sc)}
    evals = tape_records(eval_tape)
    out_dir = (REPO_ROOT / cfg["model_file"]).parent
    result = {
        "decision": "D49",
        "primary": cfg["primary"],
        "verifier_version": versions.pop(),
        "calibration_tapes": cfg["fit_tapes"],
        "calibration_seeds": [min(seeds), max(seeds)],
        "n_calibration": len(calib), "n_fit": sum(r["mu"] is not None for r in calib),
        "sigma_rel_floor": f,
        "forms": per_form,
        "global_scale_same_data": glob,
        "calibration_set": {"unscaled": posthoc.summary(calib, 1.0, sc),
                            "global_scale_same_data": posthoc.summary(eff, glob, sc)},
        "eval_tape": str(Path(eval_tape).resolve().relative_to(REPO_ROOT)),
        "eval_generator_version": eval_meta["generator_version"],
        "eval": {split: {"unscaled": posthoc.summary([r for r in evals if r["split"] == split], 1.0, sc), **e}
                 for split, e in conditional_summaries(evals, sc, models).items()},
        "eval_oracle_cell_scales": {f"{c[0]}/{c[1]}": v for c, v in
                                    oracle_cell_scales(with_sigma_eff(evals, f), sc, cfg_ph["fit"]).items()},
        "models": {form: m.to_json() for form, m in models.items()},
    }
    (REPO_ROOT / cfg["model_file"]).write_text(json.dumps(result, indent=1))
    for form, m in models.items():
        anchor = cfg["forms"][form]
        posthoc.scaled_tape(eval_tape, m.scales(anchored(evals, anchor, f)), out_dir / f"sft_eval_text_{form}.jsonl",
                            f, anchor)
    posthoc.scaled_tape(eval_tape, oracle_scales(with_sigma_eff(evals, f), sc, cfg_ph["fit"]),
                        out_dir / "sft_eval_oracle_cell.jsonl", f)
    return result


def refresh_report(eval_dir: Path) -> dict:
    """Recompute the `posthoc_baseline` block (global × 18.1 and the D49 rows) of an existing eval report from its
    tape, and rewrite metrics.json and summary.md. Nothing else in the report changes."""
    from physcal.eval.run_eval import render_summary

    meta = read_meta(eval_dir / "rollouts.jsonl")
    metrics = json.loads((eval_dir / "metrics.json").read_text())
    if metrics.get("verifier_version", verifier_version(meta["reward_config"])) != verifier_version(meta["reward_config"]):
        raise ValueError(f"{eval_dir}: metrics.json and the tape disagree on the verifier version")
    records: dict[str, list[dict]] = {}
    for r in read_tape(eval_dir / "rollouts.jsonl"):
        records.setdefault(r.get("split", "all"), []).append(r)
    metrics["posthoc_baseline"] = posthoc.baseline_report(records, meta["reward_config"]["scoring"],
                                                          meta["generator_version"],
                                                          verifier_version(meta["reward_config"]))
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    (eval_dir / "summary.md").write_text(render_summary(metrics))
    return metrics["posthoc_baseline"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["fit", "report"])
    ap.add_argument("eval_dirs", nargs="*", help="report: eval directories (rollouts.jsonl + metrics.json)")
    ap.add_argument("--eval", help="fit: SFT-only eval tape scored by the same verifier version")
    args = ap.parse_args()
    if args.action == "report":
        for d in args.eval_dirs:
            pb = refresh_report(REPO_ROOT / d)
            for split, e in pb["conditional"]["splits"].items():
                print(d, split, {k: round(v["coverage_1sigma"], 3) for k, v in e.items()})
        return
    if not args.eval:
        ap.error("fit needs --eval")
    res = fit(REPO_ROOT / args.eval)
    print(json.dumps({k: res[k] for k in ("n_fit", "global_scale_same_data", "calibration_set", "eval")}, indent=1))
    print(json.dumps({form: {k: v for k, v in e.items()} for form, e in res["forms"].items()}, indent=1))


if __name__ == "__main__":
    main()

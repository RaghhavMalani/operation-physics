"""X1 (D73): a shallow text model of σ* on the frozen A6 / A6b propagation pairs. Can surface features propagate?

    OMP_NUM_THREADS=4 nice uv run python -m physcal.eval.x1_text_sigma run

Fit set: generator-2.0.0 problems from the training split (training families, depths 1–4, noise 1 / 2 / 5%; a seed
range of its own), target y = ln σ*_log (the stored Monte Carlo σ* in log space). Inputs: the problem text only.
- TF-IDF over whitespace tokens (A5's text settings, the top 1,000 tokens by document frequency);
- numeric features parsed from the text: every measurement "<symbol> = <number> <unit> ± <p>%" (or ", equals"),
  converted to SI with pint. Per symbol, ln|value| (a column per symbol seen >= symbols_min_count times; missing
  otherwise, which the GBT handles natively), plus summary statistics of the values and of the stated p.
Feature sets, nested: noise (p statistics only), noise_text (+ TF-IDF), noise_text_magnitudes (+ magnitudes; the
primary). Each is a HistGradientBoostingRegressor chosen from A5's GBT grid by 5-fold CV grouped by template.

The fitted model's σ̂ = exp(ŷ) is the "stated σ" of every pair member. X1 states no answer, so every member counts as
computation-correct and every pair enters the primary A6 / A6b analysis (physcal/eval/a6_pairs.py, D58 / D59 / D61 /
D64, unchanged). Two references run through the same analysis: σ̂ = σ* itself (an exact propagator: slope 1 on A6,
0 on A6b) and the noise-only set (constant within a pair: slope 0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from physcal.config import REPO_ROOT, load_config

NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
UNIT = r"[A-Za-z][A-Za-z0-9/^*]*(?: [A-Za-z][A-Za-z0-9/^*]*)?"
MEAS_RE = re.compile(rf"(?P<sym>[A-Za-z][A-Za-z0-9_]*),?\s+(?:=|equals)\s+(?P<val>{NUM})\s*(?P<unit>{UNIT})?\s*±\s*"
                     rf"(?P<pct>\d+(?:\.\d+)?)\s*%")
STATS = ("n_meas", "p_max", "p_mean", "p_rss")
MAG_STATS = ("lnv_min", "lnv_max", "lnv_mean", "lnv_sd")


def _config() -> dict:
    return load_config("x1_text_sigma")


def parse_measurements(text: str) -> list[dict]:
    """Every '<symbol> = <number> <unit> ± <p>%' in the text: symbol, SI value (None if the unit does not convert)
    and p as a fraction. Exact values (no ±) are not measurements."""
    from physcal.env.units import to_si

    out = []
    for m in MEAS_RE.finditer(text):
        unit = (m.group("unit") or "").strip()
        try:
            si = to_si(float(m.group("val")), unit)
        except Exception:  # noqa: BLE001 (an unparseable unit is a missing feature, not an error)
            si = None
        out.append({"symbol": m.group("sym"), "si": si, "p": float(m.group("pct")) / 100})
    return out


def numeric_features(meas: list[dict], symbols: list[str], with_magnitudes: bool) -> list[float]:
    ps = [x["p"] for x in meas]
    row = [len(meas), max(ps, default=math.nan), float(np.mean(ps)) if ps else math.nan,
           math.sqrt(sum(p * p for p in ps)) if ps else math.nan]
    if not with_magnitudes:
        return row
    lnv = [math.log(abs(x["si"])) for x in meas if x["si"] not in (None, 0.0)]
    row += [min(lnv, default=math.nan), max(lnv, default=math.nan), float(np.mean(lnv)) if lnv else math.nan,
            float(np.std(lnv)) if len(lnv) > 1 else math.nan]
    by_sym = {}
    for x in meas:
        if x["si"] not in (None, 0.0):
            by_sym.setdefault(x["symbol"], math.log(abs(x["si"])))
    return row + [by_sym.get(s, math.nan) for s in symbols]


class Featurizer:
    def __init__(self, feature_set: str, cfg: dict):
        self.fs, self.cfg = feature_set, cfg
        self.symbols: list[str] = []
        self.vec = None

    def fit(self, recs: list[dict]) -> "Featurizer":
        from sklearn.feature_extraction.text import TfidfVectorizer

        if self.fs != "noise":
            t = self.cfg["text"]
            self.vec = TfidfVectorizer(token_pattern=t["token_pattern"], lowercase=False,
                                       ngram_range=tuple(t["ngram_range"]), min_df=t["min_df"],
                                       sublinear_tf=t["sublinear_tf"], max_features=t["max_features"])
            self.vec.fit([r["text"] for r in recs])
        if self.fs == "noise_text_magnitudes":
            c = Counter(x["symbol"] for r in recs for x in {m["symbol"]: m for m in r["meas"]}.values())
            self.symbols = sorted(s for s, n in c.items() if n >= self.cfg["symbols_min_count"])
        return self

    def transform(self, recs: list[dict]) -> np.ndarray:
        mags = self.fs == "noise_text_magnitudes"
        X = np.array([numeric_features(r["meas"], self.symbols, mags) for r in recs], dtype=float)
        if self.vec is not None:
            X = np.hstack([X, self.vec.transform([r["text"] for r in recs]).toarray()])
        return X


def _problem_record(d: dict) -> dict:
    n = d["noise"]
    return {"text": d["text"], "meas": parse_measurements(d["text"]), "y": math.log(n["sigma_star_log"]),
            "sigma_star_log": n["sigma_star_log"], "family": d["family"], "depth": d["depth"], "level": n["level"],
            "seed": d["seed"], "template": f"{d['family']}|{d['target_symbol']}|"
                                          f"{','.join(sorted(g['symbol'] for g in d['givens']))}",
            "n_noisy_shown": len(d["givens"]) + len(d.get("distractors") or [])}


def _gen(args: tuple) -> dict:
    from physcal.env.generator import generate_problem

    family, depth, seed, level, env_name = args
    return generate_problem(family, depth, seed, split="train", cfg=load_config(env_name), noise=level).to_dict()


def build_set(part: dict, env_name: str, n_jobs: int) -> list[dict]:
    """Round-robin over cells (outer) and levels: problem i is cell i mod 12, level (i // 12) mod 3, seed offset + i."""
    cells = [(f, d) for f in part["families"] for d in part["depths"]]
    lv = part["levels"]
    jobs = [(*cells[i % len(cells)], part["seed_offset"] + i, lv[(i // len(cells)) % len(lv)], env_name)
            for i in range(part["n"])]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(_gen, jobs, chunksize=64))


def load_pairs(path: str) -> list[dict]:
    p = REPO_ROOT / path
    blob = p.read_text()
    man = json.loads(p.with_name(p.stem + "_manifest.json").read_text())
    if hashlib.sha256(blob.encode()).hexdigest() != man["sha256"]:
        raise ValueError(f"{p}: checksum mismatch")
    return [json.loads(line) for line in blob.splitlines() if line.strip()]


def select(recs: list[dict], cfg: dict) -> dict:
    """Per feature set: the GBT grid point with the lowest grouped-CV MSE of ln σ* (fit set only)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    y = np.array([r["y"] for r in recs])
    groups = [r["template"] for r in recs]
    folds = list(GroupKFold(n_splits=cfg["cv"]["folds"]).split(np.zeros(len(recs)), y, groups))
    table = {}
    for fs in cfg["feature_sets"]:
        rows = []
        for hp in cfg["gbt_grid"]:
            oof = np.empty(len(recs))
            for tr, te in folds:
                feat = Featurizer(fs, cfg).fit([recs[i] for i in tr])
                m = HistGradientBoostingRegressor(random_state=cfg["seed"], **hp)
                m.fit(feat.transform([recs[i] for i in tr]), y[tr])
                oof[te] = m.predict(feat.transform([recs[i] for i in te]))
            rows.append({"hp": hp, "cv_mse": float(np.mean((oof - y) ** 2)),
                         "cv_median_abs_log_error": float(np.median(np.abs(oof - y)))})
        best = min(rows, key=lambda r: r["cv_mse"])
        table[fs] = {"grid": rows, "best": best}
    return table


def fit_models(recs: list[dict], table: dict, cfg: dict) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor

    y = np.array([r["y"] for r in recs])
    out = {}
    for fs, entry in table.items():
        feat = Featurizer(fs, cfg).fit(recs)
        out[fs] = (feat, HistGradientBoostingRegressor(random_state=cfg["seed"], **entry["best"]["hp"])
                   .fit(feat.transform(recs), y))
    return out


def predict(model: tuple, recs: list[dict]) -> np.ndarray:
    feat, m = model
    return m.predict(feat.transform(recs))


def accuracy(recs: list[dict], yhat: np.ndarray) -> dict:
    y = np.array([r["y"] for r in recs])
    ss = float(np.sum((y - y.mean()) ** 2))
    return {"n": len(recs), "r2": 1 - float(np.sum((y - yhat) ** 2)) / ss if ss > 0 else None,
            "median_abs_log_error": float(np.median(np.abs(yhat - y))),
            "spearman": float(spearmanr(yhat, y)[0]) if len(recs) > 2 else None}


def pair_rows(problems: list[dict], ln_sigma: np.ndarray) -> list[dict]:
    """a6_pairs member rows with σ̂ as the stated relative σ; no answer, so every member is computation-correct."""
    rows = []
    for d, ls in zip(problems, ln_sigma):
        n = d["noise"]
        rows.append({"pair": (d["family"], d["depth"], n["level"], n["pair"]["pair_seed"]), "member": n["pair"]["member"],
                     "kind": n["pair"].get("kind", "sensitivity"), "sigma_star_log": n["sigma_star_log"],
                     "truth": d["target_value_si"], "ok": True, "computation_correct": True,
                     "stated_log": float(ls), "sigma_zero": False})
    return rows


def pair_analysis(a6: list[dict], a6b: list[dict], ln6: np.ndarray, ln6b: np.ndarray) -> dict:
    from physcal.eval.a6_pairs import analyze_rows, joint

    pc = load_config("pairs")
    r6 = analyze_rows(pair_rows(a6, ln6), pc["analysis"])
    r6b = analyze_rows(pair_rows(a6b, ln6b), pc["analysis"], pc["invariance_analysis"])
    return {"A6": r6, "A6b": r6b, "joint": joint(r6, r6b)}


def within_pair(problems: list[dict], ln_sigma: np.ndarray) -> dict:
    """Δ ln σ̂ against Δ ln σ* within pairs (hi − lo): correlation and the share with the same sign."""
    by: dict = {}
    for d, ls in zip(problems, ln_sigma):
        n = d["noise"]
        by.setdefault((d["family"], d["depth"], n["level"], n["pair"]["pair_seed"]), {})[n["pair"]["member"]] = \
            (float(ls), math.log(n["sigma_star_log"]))
    dx = np.array([v["hi"][1] - v["lo"][1] for v in by.values()])
    dy = np.array([v["hi"][0] - v["lo"][0] for v in by.values()])
    return {"n_pairs": len(dx), "mean_d_ln_sigma_star": float(dx.mean()), "mean_d_ln_sigma_hat": float(dy.mean()),
            "corr": float(np.corrcoef(dx, dy)[0, 1]) if np.std(dy) > 0 and np.std(dx) > 0 else None,
            "share_same_sign": float(np.mean(np.sign(dx) == np.sign(dy)))}


def parse_coverage(recs: list[dict]) -> dict:
    """Parsed measurements vs the measured quantities shown (givens + distractors): the parser's recall proxy."""
    shown = np.array([r["n_noisy_shown"] for r in recs])
    got = np.array([len(r["meas"]) for r in recs])
    conv = [x["si"] is not None for r in recs for x in r["meas"]]
    return {"share_problems_all_parsed": float(np.mean(got == shown)), "share_problems_over": float(np.mean(got > shown)),
            "mean_parsed": float(got.mean()), "mean_shown": float(shown.mean()),
            "share_units_converted": float(np.mean(conv)) if conv else None}


def run() -> dict:
    cfg = _config()
    out = REPO_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    env_name = cfg["env_config"]
    fit_set = [_problem_record(d) for d in build_set({**cfg["train"]}, env_name, cfg["n_jobs"])]
    test_part = {**cfg["train"], **cfg["test"]}
    test_set = [_problem_record(d) for d in build_set(test_part, env_name, cfg["n_jobs"])]
    a6, a6b = load_pairs(cfg["pairs"]["a6"]), load_pairs(cfg["pairs"]["a6b"])
    seeds = {r["seed"] for r in fit_set} | {r["seed"] for r in test_set}
    if seeds & {d["seed"] for d in a6 + a6b} or len(seeds) != len(fit_set) + len(test_set):
        raise ValueError("fit / test / pair seeds overlap")
    coverage = {"fit": parse_coverage(fit_set), "test": parse_coverage(test_set)}
    table = select(fit_set, cfg)
    (out / "selection.json").write_text(json.dumps({"decision": "D73 (X1)", "n_fit": len(fit_set),
                                                    "n_templates": len({r["template"] for r in fit_set}),
                                                    "parse_coverage": coverage, "cv": table}, indent=1))
    models = fit_models(fit_set, table, cfg)
    p6, p6b = [_problem_record(d) for d in a6], [_problem_record(d) for d in a6b]
    res = {"decision": "D73 (X1)", "primary": cfg["primary"], "parse_coverage": coverage, "feature_sets": {}}
    for fs, model in models.items():
        l6, l6b = predict(model, p6), predict(model, p6b)
        res["feature_sets"][fs] = {
            "cv_best": table[fs]["best"], "test_accuracy": accuracy(test_set, predict(model, test_set)),
            "a6_member_accuracy": accuracy(p6, l6), "a6b_member_accuracy": accuracy(p6b, l6b),
            "within_pair_a6": within_pair(a6, l6), "within_pair_a6b": within_pair(a6b, l6b),
            "pairs": pair_analysis(a6, a6b, l6, l6b)}
    ln_star6, ln_star6b = np.array([r["y"] for r in p6]), np.array([r["y"] for r in p6b])
    res["reference_exact_sigma_star"] = {"pairs": pair_analysis(a6, a6b, ln_star6, ln_star6b)}
    by_depth = {}
    prim = models[cfg["primary"]]
    for dpt in sorted({d["depth"] for d in a6}):
        i6 = [i for i, d in enumerate(a6) if d["depth"] == dpt]
        i6b = [i for i, d in enumerate(a6b) if d["depth"] == dpt]
        l6, l6b = predict(prim, [p6[i] for i in i6]), predict(prim, [p6b[i] for i in i6b])
        by_depth[dpt] = pair_analysis([a6[i] for i in i6], [a6b[i] for i in i6b], l6, l6b)
    res["primary_by_depth"] = by_depth
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "summary.md").write_text(render(res))
    return res


def _fmt_ci(ci) -> str:
    return "–" if ci is None else f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"


def render(res: dict) -> str:
    lines = ["# X1: a shallow text model of σ* on the A6 / A6b pairs (D73)", "",
             f"Primary feature set: **{res['primary']}**. Parse coverage (fit set): "
             f"{res['parse_coverage']['fit']['share_problems_all_parsed']:.1%} of problems with every shown "
             f"measurement parsed; units converted {res['parse_coverage']['fit']['share_units_converted']:.1%}.", "",
             "| σ source | test R² (ln σ*) | test median abs ln error | A6 slope [90% CI] | A6 label | A6b slope [90% CI] "
             "| A6b ≡ 0 | D59 verdict | labelled verdict |", "|---|---|---|---|---|---|---|---|---|"]

    def row(name, acc, pa):
        a6, a6b, j = pa["A6"]["primary_computation_correct"], pa["A6b"]["primary_computation_correct"], pa["joint"]
        r2 = "–" if acc is None else f"{acc['r2']:.3f}"
        mae = "–" if acc is None else f"{acc['median_abs_log_error']:.3f}"
        return (f"| {name} | {r2} | {mae} | {a6['slope']:+.3f} {_fmt_ci(a6['ci'])} | {j['A6_label']} "
                f"| {a6b['slope']:+.3f} {_fmt_ci(a6b['ci'])} | {pa['A6b']['equivalent_to_zero']} | {j['verdict']} "
                f"| {j['labelled_verdict']} |")

    for fs, e in res["feature_sets"].items():
        lines.append(row(fs + (" (primary)" if fs == res["primary"] else ""), e["test_accuracy"], e["pairs"]))
    lines.append(row("reference: σ̂ = σ* (exact propagation)", None, res["reference_exact_sigma_star"]["pairs"]))
    lines += ["", "## Primary, by depth", "", "| depth | A6 pairs | A6 slope [90% CI] | A6 label | A6b pairs "
              "| A6b slope [90% CI] | labelled verdict |", "|---|---|---|---|---|---|---|"]
    for dpt, pa in res["primary_by_depth"].items():
        a6, a6b = pa["A6"]["primary_computation_correct"], pa["A6b"]["primary_computation_correct"]
        s6 = "–" if a6["slope"] is None else f"{a6['slope']:+.3f} {_fmt_ci(a6['ci'])}"
        s6b = "–" if a6b["slope"] is None else f"{a6b['slope']:+.3f} {_fmt_ci(a6b['ci'])}"
        lines.append(f"| {dpt} | {a6['n_pairs']} | {s6} | {pa['joint']['A6_label']} | {a6b['n_pairs']} | {s6b} "
                     f"| {pa['joint']['labelled_verdict']} |")
    lines += ["", "## Within-pair agreement (descriptive)", "",
              "| σ source | A6 corr(Δ ln σ̂, Δ ln σ*) | A6 same sign | A6b mean Δ ln σ̂ | A6 member R² | A6b member R² |",
              "|---|---|---|---|---|---|"]
    for fs, e in res["feature_sets"].items():
        w6, w6b = e["within_pair_a6"], e["within_pair_a6b"]
        c = "–" if w6["corr"] is None else f"{w6['corr']:.3f}"
        lines.append(f"| {fs} | {c} | {w6['share_same_sign']:.1%} | {w6b['mean_d_ln_sigma_hat']:+.3f} "
                     f"| {e['a6_member_accuracy']['r2']:.3f} | {e['a6b_member_accuracy']['r2']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["run"])
    ap.parse_args()
    print(render(run()))


if __name__ == "__main__":
    main()

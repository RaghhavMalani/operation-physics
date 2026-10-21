"""E1d on the X2 pairs (DECISIONS.md D102): the pre-registered analysis, CPU only.

    OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.eval.e1d_x2_analysis analyze

On X2's 516 kept problems (D74): D74's pairwise accuracy of E1d's greedy σ suffix (confidence −ln σ_rel, a failed
suffix a tie), with a problem-level cluster bootstrap. T1: E1d > 0.5; T2: E1d − SFT stated σ (X2 method 1) > 0; T3:
E1d − X3 student (a) > 0; Holm over the three. Reported without a test: E1d − self-consistency, the SFT model's own
greedy suffix (the no-RL control), D92's pooled and separated-pair accuracies, per depth, and across problems ρ(σ,
|ln error|) and the AUROC for "correct". Each comparator first reproduces its published pairwise accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.eval import x2_pairs
from physcal.eval import x2_robustness as XR
from physcal.eval.stats import holm
from physcal.eval.x2_analysis import per_problem, suffix_confidence

LABELS = {"e1d": "E1d σ suffix (lr 5e-5, step 300)", "sft_sigma": "SFT stated σ (X2 method 1, sampled)",
          "x3_a": "X3 student (a) σ suffix", "self_consistency": "self-consistency (D92c, k = 8; post-hoc)",
          "sft_suffix": "SFT greedy σ suffix (no-RL control)"}
TESTS = {"T1": ("e1d", None), "T2": ("e1d", "sft_sigma"), "T3": ("e1d", "x3_a")}
REPORTED = {"e1d - self_consistency": ("e1d", "self_consistency"), "e1d - sft_suffix": ("e1d", "sft_suffix")}


def _config() -> dict:
    return load_config("e1d_x2")


# ------------------------------------------------------------------------------------------- statistics
def boot_idx(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, (n_boot, n))


def accuracy(pa: np.ndarray, n_boot: int, seed: int) -> dict:
    """Mean over problems, 95% CI and one-sided p (> 0.5) from the cluster bootstrap over problems."""
    boots = pa[boot_idx(len(pa), n_boot, seed)].mean(axis=1)
    return {"n_problems": int(len(pa)), "pairwise_accuracy": float(pa.mean()),
            "ci95": np.percentile(boots, [2.5, 97.5]).tolist(), "p_one_sided_gt_half": float(np.mean(boots <= 0.5))}


def paired(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict:
    """a − b per problem, the same resampled problems for both; one-sided p = share of resamples <= 0."""
    d = a - b
    boots = d[boot_idx(len(d), n_boot, seed)].mean(axis=1)
    return {"mean_diff": float(d.mean()), "ci95": np.percentile(boots, [2.5, 97.5]).tolist(),
            "p_one_sided_gt_zero": float(np.mean(boots <= 0))}


def tests(pa: dict, n_boot: int, seed: int, alpha: float) -> dict:
    """T1-T3 (D102), Holm over the three. A method whose rows are missing counts as p = 1."""
    out, p = {}, {}
    for t, (a, b) in TESTS.items():
        if a not in pa or (b is not None and b not in pa):
            out[t], p[t] = {"missing": True}, 1.0
            continue
        out[t] = accuracy(pa[a], n_boot, seed) if b is None else paired(pa[a], pa[b], n_boot, seed)
        p[t] = out[t]["p_one_sided_gt_half" if b is None else "p_one_sided_gt_zero"]
    h = holm(p, alpha)
    for t in out:
        out[t]["holm"] = h[t]
    return out


def reading(t: dict) -> str:
    """D102's reading rules."""
    s = {k: v["holm"]["rejected"] for k, v in t.items()}
    if not s["T1"]:
        return "T1 not supported: E1d's ρ is between-problem ranking only (the X3 reading)"
    if s["T3"]:
        return "T1 and T3 supported: RL produced within-problem information that X3's supervised student did not"
    return "T1 supported, T3 not: a within-problem signal not distinguishable from X3's"


def across(groups: dict, keys: list, conf: dict, err: dict, n_boot: int, seed: int) -> dict:
    """Across all answers with a confidence: Spearman ρ(−confidence, |ln(μ / truth)|) (for a σ source, ρ(σ_rel,
    error)) and the AUROC of the confidence for "correct"; 95% CIs from the cluster bootstrap over problems."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    per = []
    for k in keys:
        v = [(conf[(k, a["sample"])], err[(k, a["sample"])], a["correct"]) for a in groups[k]
             if conf.get((k, a["sample"])) is not None and err.get((k, a["sample"])) is not None]
        per.append(np.array(v, dtype=float).reshape(-1, 3))

    def stats(blocks):
        x = np.concatenate(blocks)
        rho = float(spearmanr(-x[:, 0], x[:, 1]).statistic)
        auc = float(roc_auc_score(x[:, 2], x[:, 0])) if 0 < x[:, 2].sum() < len(x) else float("nan")
        return rho, auc

    rho, auc = stats(per)
    boots = np.array([stats([per[i] for i in row]) for row in boot_idx(len(per), n_boot, seed)])
    return {"n_answers": int(sum(len(b) for b in per)),
            "spearman_sigma_error": {"value": rho, "ci95": np.nanpercentile(boots[:, 0], [2.5, 97.5]).tolist()},
            "auroc_correct": {"value": auc, "ci95": np.nanpercentile(boots[:, 1], [2.5, 97.5]).tolist()}}


# ------------------------------------------------------------------------------------------- inputs
def abs_log_errors(recs: list[dict]) -> dict:
    """(key, sample) -> |ln(μ / truth)| for parsed answers with μ / truth > 0."""
    out = {}
    for r in recs:
        b = r["breakdown"]
        if b["reason"] == "ok" and b.get("mu_si") is not None and b.get("truth_si") and b["mu_si"] / b["truth_si"] > 0:
            out[XR._key(r)] = abs(math.log(b["mu_si"] / b["truth_si"]))
    return out


def suffix_summary(d: Path, collapse_share: float) -> dict:
    """Failure counts and the collapse check (D94's rule) of one suffix eval."""
    from physcal.train.e1_compare import collapse

    rows = [json.loads(line) for line in (d / "rows.jsonl").open()]
    out = {"n_rows": len(rows), "n_failed": sum(r["fail"] for r in rows), "mu_changed": sum(r["mu_changed"] for r in rows),
           "fail_reasons": dict(Counter(r["reason"] for r in rows if r["fail"])),
           "sigma_zero_share": float(np.mean([r["rel_sigma_used"] == 0 for r in rows if not r["fail"]]))}
    tape = d / "rollouts.jsonl"
    if tape.exists():
        out["collapse"] = collapse(tape, collapse_share)
    out["fail_share"] = out["n_failed"] / len(rows)
    return out


def published(cfg: dict) -> dict:
    p = cfg["published"]
    x2 = json.loads((REPO_ROOT / p["sft_sigma"]).read_text())["methods"]["sft_sigma"]["pairwise_accuracy"]
    x3 = json.loads((REPO_ROOT / p["x3_a"]).read_text())["arms"]["a"]["x2"]["pairwise_accuracy"]
    sc = json.loads((REPO_ROOT / p["self_consistency"]).read_text())["methods"]["self_consistency"]
    return {"sft_sigma": x2, "x3_a": x3, "self_consistency": sc["pairwise_accuracy"]["pairwise_accuracy"]}


def confidences(cfg: dict, groups: dict, keys: list, recs: list[dict]) -> tuple[dict, dict]:
    """Every method's (key, sample) -> confidence that has rows, and the suffix evals present."""
    x2 = load_config(cfg["x2_config"])
    rc = load_config(cfg["x2_robustness_config"])
    floor = x2["analysis"]["sigma_floor_rel"]
    conf = {"sft_sigma": {(k, a["sample"]): x2_pairs.stated_sigma_conf(k, a) for k in keys for a in groups[k]},
            "x3_a": suffix_confidence(REPO_ROOT / cfg["x3_rows"], floor),
            "self_consistency": XR.self_consistency_conf(recs, rc["self_consistency"]["agree_tolerance"])}
    present = {}
    for m, name in (("e1d", "e1d_final"), ("sft_suffix", "sft_suffix")):
        d = REPO_ROOT / cfg["output_dir"] / name
        if (d / "rows.jsonl").exists():
            conf[m] = suffix_confidence(d / "rows.jsonl", floor)
            present[m] = d
    return conf, present


def check_reproduces(pa: dict, pub: dict, atol: float) -> dict:
    gaps = {}
    for m, v in pub.items():
        gaps[m] = abs(float(pa[m].mean()) - v)
        if gaps[m] > atol:
            raise RuntimeError(f"{m}: recomputed {pa[m].mean()}, published {v}")
    return gaps


# ------------------------------------------------------------------------------------------- the analysis
def analyze(cfg: dict | None = None) -> dict:
    cfg = cfg or _config()
    x2 = load_config(cfg["x2_config"])
    rc = load_config(cfg["x2_robustness_config"])
    n_boot, seed = cfg["bootstrap"], cfg["seed"]
    groups, keys, recs = XR.answer_records(x2)
    conf, present = confidences(cfg, groups, keys, recs)
    pa = {m: per_problem(groups, keys, c) for m, c in conf.items()}
    res: dict = {"decision": "D102 (pre-registered)", "n_kept": len(keys),
                 "reproduction_gaps": check_reproduces(pa, published(cfg), cfg["reproduce_atol"]),
                 "suffix_evals": {m: suffix_summary(d, cfg["collapse_share"]) for m, d in present.items()}}
    for m, s in res["suffix_evals"].items():
        s["pipeline_flag"] = s["fail_share"] > cfg["fail_share_flag"]
    if "e1d" not in pa:
        raise FileNotFoundError("E1d's suffix rows do not exist yet")
    res["tests"] = tests(pa, n_boot, seed, cfg["alpha"])
    res["reading"] = reading(res["tests"])
    res["reported_paired"] = {n: paired(pa[a], pa[b], n_boot, seed) for n, (a, b) in REPORTED.items()
                              if a in pa and b in pa}
    err_rel, err_log = XR.errors(recs), abs_log_errors(recs)
    sp = rc["separated"]
    res["methods"] = {}
    for m, c in conf.items():
        by_depth = {}
        for d in sorted({k[1] for k in keys}):
            idx = [i for i, k in enumerate(keys) if k[1] == d]
            by_depth[str(d)] = {"pairwise_accuracy": float(pa[m][idx].mean()), "n": len(idx)}
        n_no_conf = sum(1 for k in keys for a in groups[k] if c.get((k, a["sample"])) is None)
        res["methods"][m] = {
            "pairwise_accuracy": accuracy(pa[m], n_boot, seed),
            "pooled": XR.pooled(groups, keys, c, n_boot, seed),
            "separated": XR.separated(groups, keys, c, err_rel, sp["correct_max_error"], sp["incorrect_min_error"],
                                      n_boot, seed),
            "by_depth": by_depth, "answers_without_confidence": n_no_conf,
            "across_problems": across(groups, keys, c, err_log, n_boot, seed)}
    out = REPO_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "summary.md").write_text(render(res))
    return res


def _ci(v: float, ci: list, signed: bool = False) -> str:
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(v)} [{f.format(ci[0])}, {f.format(ci[1])}]"


def render(res: dict) -> str:
    t = res["tests"]
    L = [f"# E1d on the X2 pairs (D102, pre-registered): {res['n_kept']} kept problems", "",
         f"**Reading (D102):** {res['reading']}.", "",
         "Every comparator reproduces its published pairwise accuracy (max gap "
         f"{max(res['reproduction_gaps'].values()):.1e}). All CIs: cluster bootstrap over problems (B = 10,000).", "",
         "| test | estimate [95% CI] | p (one-sided) | Holm |", "|---|---|---|---|"]
    names = {"T1": "T1: E1d > 0.5", "T2": "T2: E1d − SFT stated σ > 0", "T3": "T3: E1d − X3 student (a) > 0"}
    for k, r in t.items():
        if r.get("missing"):
            L.append(f"| {names[k]} | missing | 1 | not supported |")
            continue
        v, s = (r["pairwise_accuracy"], False) if k == "T1" else (r["mean_diff"], True)
        p = r["p_one_sided_gt_half"] if k == "T1" else r["p_one_sided_gt_zero"]
        L.append(f"| {names[k]} | {_ci(v, r['ci95'], s)} | {p:.4g} | "
                 f"{'supported' if r['holm']['rejected'] else 'not supported'} |")
    for n, r in res["reported_paired"].items():
        L.append(f"| {n} (reported, no test) | {_ci(r['mean_diff'], r['ci95'], True)} | – | – |")
    L += ["", "| method | pairwise acc. [95% CI] | pooled pair-weighted | separated pairs | ρ(σ, error) across "
          "answers | AUROC (correct) | answers without confidence |", "|---|---|---|---|---|---|---|"]
    for m, r in res["methods"].items():
        a, po, se, ac = r["pairwise_accuracy"], r["pooled"], r["separated"], r["across_problems"]
        L.append(f"| {LABELS.get(m, m)} | {_ci(a['pairwise_accuracy'], a['ci95'])} | "
                 f"{_ci(po['pooled_pairwise_accuracy'], po['ci95'])} | {_ci(se['pairwise_accuracy'], se['ci95'])} | "
                 f"{_ci(ac['spearman_sigma_error']['value'], ac['spearman_sigma_error']['ci95'], True)} | "
                 f"{_ci(ac['auroc_correct']['value'], ac['auroc_correct']['ci95'])} | {r['answers_without_confidence']} |")
    depths = sorted(next(iter(res["methods"].values()))["by_depth"])
    L += ["", "| method | " + " | ".join(f"depth {d}" for d in depths) + " |", "|---|" + "---|" * len(depths)]
    for m, r in res["methods"].items():
        L.append(f"| {LABELS.get(m, m)} | " + " | ".join(
            f"{r['by_depth'][d]['pairwise_accuracy']:.3f} (n {r['by_depth'][d]['n']})" for d in depths) + " |")
    L += ["", "| suffix eval | rows | failed | μ changed | σ = 0 share | top suffix share | collapsed | pipeline flag |",
          "|---|---|---|---|---|---|---|---|"]
    for m, s in res["suffix_evals"].items():
        c = s.get("collapse", {})
        L.append(f"| {LABELS.get(m, m)} | {s['n_rows']} | {s['n_failed']} | {s['mu_changed']} | "
                 f"{s['sigma_zero_share']:.1%} | {c.get('top_share', float('nan')):.1%} (`{c.get('top_suffix', '')}`) | "
                 f"{c.get('collapsed')} | {s['pipeline_flag']} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["analyze"])
    ap.parse_args()
    print(render(analyze()))


if __name__ == "__main__":
    main()

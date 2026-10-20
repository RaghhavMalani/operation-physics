"""X2 robustness checks (DECISIONS.md D92): post-hoc and exploratory, CPU only.

    nice uv run python -m physcal.eval.x2_robustness analyze

On X2's 516 kept problems and their pairs, with X2's confidences recomputed by X2's own code (each must reproduce
results/x2/results.json):
  a. pairwise accuracy on clearly separated pairs only (the correct answer |error| <= 0.25%, the incorrect >= 10%),
     for every X2 method, the two baselines below and the complete X4 arms;
  b. completion length (tokens), right vs wrong, and a length-only pairwise baseline (shorter = more confident);
  c. a self-consistency baseline: the share of the problem's other samples whose μ agrees with this answer within 1%;
  d. X2's CIs recomputed (they already resample problems), and the pooled pair-weighted pairwise accuracy with a
     cluster bootstrap over problems.
Every interval: a bootstrap over problems. No Holm family: nothing here is pre-registered.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.eval import x2_pairs
from physcal.eval.x2_analysis import LABELS as X2_LABELS
from physcal.eval.x2_analysis import paired, per_problem

LABELS = X2_LABELS | {"length": "baseline: shorter completion (D92)",
                      "self_consistency": "baseline: agreement with the other samples (D92)"}


def _config() -> dict:
    return load_config("x2_robustness")


def _cfg_of(x) -> dict:
    return x if isinstance(x, dict) else load_config(x)


# ------------------------------------------------------------------------------------------- inputs
def answer_records(x2: dict) -> tuple[dict, list, list]:
    """(groups, kept keys as tuples, every sampled record of the kept problems)."""
    groups, keys = x2_pairs.kept_groups(x2)
    recs, _ = x2_pairs.kept_records(x2)
    return {tuple(k): v for k, v in groups.items()}, [tuple(k) for k in keys], recs


def _key(r: dict) -> tuple:
    p = r["problem"]
    return (p["family"], p["depth"], p["seed"]), r.get("sample", 0)


def errors(recs: list[dict]) -> dict:
    """(key, sample) -> the verifier's |relative error| (parsed answers only)."""
    return {_key(r): r["breakdown"]["abs_rel_error"] for r in recs if r["breakdown"].get("abs_rel_error") is not None}


def x2_confidences(x2: dict, rc: dict) -> tuple[dict, dict]:
    """X2's per-answer confidences, by x2_analysis.analyze's own steps. The SFT probe comes from X4's X2 reference
    (the same code, checked there); the random-init and cross-fitted probes are refitted only with `refits`."""
    from physcal.eval import x2_analysis as XA
    from physcal.eval import x2_hidden

    ac = x2["analysis"]
    groups, keys, _ = answer_records(x2)
    records, _ = x2_pairs.sampled_records(x2)
    conf = {"sft_sigma": {(k, a["sample"]): x2_pairs.stated_sigma_conf(k, a) for k in keys for a in groups[k]}}
    sfx = REPO_ROOT / x2["suffix"]["output_dir"] / "rows.jsonl"
    if sfx.exists():
        conf["split_sigma"] = XA.suffix_confidence(sfx, ac["sigma_floor_rel"])
    a5 = XA.a5_confidences(records, x2)
    a5.pop("_names")
    conf.update(a5)
    ref = json.loads((REPO_ROOT / _cfg_of(rc["x4_config"])["output_dir"] / "x2_reference.json").read_text())
    conf["probe"] = {(tuple(k), s): v for k, s, v in ref["probe"]}
    info = {"probe": "from results/x4/x2_reference.json (D89)"}
    if rc["refits"]:
        pc = ac["probe"] | ({} if rc["n_jobs"] is None else {"n_jobs": rc["n_jobs"]})  # X2's own by default
        fl, fi = x2_hidden.load("random_init", "fit", x2)
        xl, xi = x2_hidden.load("random_init", "x2", x2)
        conf["random_init_probe"], info["random_init_probe"] = XA.probe_confidence(fl, fi, xl, xi, pc)
        xl, xi = x2_hidden.load("sft", "x2", x2)
        conf["probe_crossfit"], info["probe_crossfit"] = XA.crossfit_confidence(xl, xi, pc)
    return conf, info


def check_reproduces(groups: dict, keys: list, conf: dict, published: dict, atol: float) -> dict:
    """Every recomputed X2 method must give X2's published pairwise accuracy (the same pairs, the same confidences)."""
    out = {}
    for m, c in conf.items():
        mine = float(per_problem(groups, keys, c).mean())
        gap = abs(mine - published[m]["pairwise_accuracy"])
        if gap > atol:
            raise RuntimeError(f"X2 method {m}: recomputed {mine}, published {published[m]['pairwise_accuracy']}")
        out[m] = gap
    return out


# ------------------------------------------------------------------------------------------- the baselines
def length_conf(recs: list[dict]) -> dict:
    """b. Confidence = −completion length in tokens (shorter = more confident)."""
    return {_key(r): -float(r["n_tokens"]) for r in recs}


def self_consistency_conf(recs: list[dict], tol: float) -> dict:
    """c. Per parsed answer: the share of its problem's other samples whose parsed μ (SI) agrees within `tol`
    (|μ_j / μ_i − 1| <= tol). The denominator is the problem's other samples, parsed or not. An answer of μ = 0 has
    no relative agreement: no confidence (its pairs are ties)."""
    by: dict = {}
    for r in recs:
        by.setdefault(_key(r)[0], []).append(r)
    out = {}
    for k, rs in by.items():
        mus = [(r.get("sample", 0), r["breakdown"].get("mu_si") if r["breakdown"]["reason"] == "ok" else None)
               for r in rs]
        for s, m in mus:
            if m is None or m == 0:
                continue
            agree = sum(1 for t, x in mus if t != s and x is not None and abs(x / m - 1) <= tol)
            out[(k, s)] = agree / (len(mus) - 1)
    return out


# ------------------------------------------------------------------------------------------- statistics
def boot_mean(v: np.ndarray, n_boot: int, seed: int) -> dict:
    """Mean over problems, 95% CI from a bootstrap over problems."""
    if not len(v):
        return {"mean": None, "ci95": [None, None], "n_problems": 0}
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return {"mean": float(v.mean()), "ci95": np.percentile(boots, [2.5, 97.5]).tolist(), "n_problems": int(len(v))}


def pair_scores(groups: dict, k: tuple, conf: dict, keep=lambda a: True) -> list[float]:
    """X2's pair scores in one problem (1 if the correct answer is more confident, ties and missing 0.5), over the
    (correct, incorrect) pairs whose two answers `keep` accepts."""
    c = [conf.get((k, a["sample"])) for a in groups[k] if a["correct"] and keep(a)]
    w = [conf.get((k, a["sample"])) for a in groups[k] if not a["correct"] and keep(a)]
    return [0.5 if (x is None or y is None or x == y) else float(x > y) for x in c for y in w]


def separated(groups: dict, keys: list, conf: dict, err: dict, lo: float, hi: float, n_boot: int, seed: int) -> dict:
    """a. Pairwise accuracy over the clearly separated pairs: per problem, the mean over its separated pairs; the
    mean over the problems that have one; bootstrap over those problems."""
    per = []
    for k in keys:
        keep = lambda a: err[(k, a["sample"])] <= lo if a["correct"] else err[(k, a["sample"])] >= hi
        s = pair_scores(groups, k, conf, keep)
        if s:
            per.append(float(np.mean(s)))
    return x2_pairs.summarize(np.array(per), n_boot, seed) | {"n_pairs": _n_separated(groups, keys, err, lo, hi)}


def _n_separated(groups: dict, keys: list, err: dict, lo: float, hi: float) -> int:
    return int(sum(sum(a["correct"] and err[(k, a["sample"])] <= lo for a in groups[k])
                   * sum(not a["correct"] and err[(k, a["sample"])] >= hi for a in groups[k]) for k in keys))


def pooled(groups: dict, keys: list, conf: dict, n_boot: int, seed: int) -> dict:
    """d. The pooled pair-weighted pairwise accuracy (every pair weighs the same) with a cluster bootstrap over
    problems (problems resampled with all their pairs)."""
    s = np.array([sum(pair_scores(groups, k, conf)) for k in keys])
    n = np.array([len(pair_scores(groups, k, conf)) for k in keys])
    idx = np.random.default_rng(seed).integers(0, len(keys), (n_boot, len(keys)))
    boots = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return {"pooled_pairwise_accuracy": float(s.sum() / n.sum()), "ci95": np.percentile(boots, [2.5, 97.5]).tolist(),
            "p_one_sided_gt_half": float(np.mean(boots <= 0.5)), "n_pairs": int(n.sum())}


def lengths(groups: dict, keys: list, recs: list[dict], n_boot: int, seed: int) -> dict:
    """b. Completion tokens of correct and incorrect answers: pooled means (cluster bootstrap) and the within-problem
    difference (per problem, mean correct − mean incorrect; mean over problems)."""
    from physcal.eval.x4_analysis import cluster_mean

    n_tok = {_key(r): r["n_tokens"] for r in recs}
    c = [[n_tok[(k, a["sample"])] for a in groups[k] if a["correct"]] for k in keys]
    w = [[n_tok[(k, a["sample"])] for a in groups[k] if not a["correct"]] for k in keys]
    diff = np.array([np.mean(x) - np.mean(y) for x, y in zip(c, w)])
    return {"correct_mean_tokens": cluster_mean(c, n_boot, seed), "incorrect_mean_tokens": cluster_mean(w, n_boot, seed),
            "within_problem_difference": boot_mean(diff, n_boot, seed),
            "share_problems_correct_shorter": float(np.mean(diff < 0))}


def x4_confidences(rc: dict, groups: dict, keys: list) -> dict:
    """The X4 arms whose every kept problem is complete: confidence = the yes-vs-no log-odds (D89)."""
    from physcal.eval import x4_analysis as A
    from physcal.eval import x4_verify as V

    x4 = _cfg_of(rc["x4_config"])
    out = {}
    for arm in A.ARMS:
        rows = V.read_rows(arm, x4)
        meta = A._meta_raw(arm, x4) or {}
        excl = {(tuple(k), s) for k, s, _ in (meta.get("excluded") or [])}
        if rows and len(A.complete_keys(rows, groups, keys, excl)) == len(keys):
            out[f"x4_{arm}"] = A.conf_of(rows, "log_odds")
    return out


# ------------------------------------------------------------------------------------------- the analysis
def analyze(rc: dict | None = None) -> dict:
    rc = rc or _config()
    x2 = _cfg_of(rc["x2_config"])
    n_boot, seed = rc["bootstrap"], rc["seed"]
    groups, keys, recs = answer_records(x2)
    published = json.loads((REPO_ROOT / x2["output_dir"] / "results.json").read_text())
    conf, info = x2_confidences(x2, rc)
    gaps = check_reproduces(groups, keys, conf, published["methods"], rc["reproduce_atol"])
    conf["length"] = length_conf(recs)
    conf["self_consistency"] = self_consistency_conf(recs, rc["self_consistency"]["agree_tolerance"])
    conf |= x4_confidences(rc, groups, keys)
    err = errors(recs)
    sp = rc["separated"]
    res: dict = {"decision": "D92 (post-hoc, exploratory)", "n_kept": len(keys), "reproduction_gaps": gaps,
                 "info": info, "methods": {}}
    for m, c in conf.items():
        pa = per_problem(groups, keys, c)
        r = {"pairwise_accuracy": x2_pairs.summarize(pa, n_boot, seed), "pooled": pooled(groups, keys, c, n_boot, seed),
             "separated": separated(groups, keys, c, err, sp["correct_max_error"], sp["incorrect_min_error"],
                                    n_boot, seed)}
        if m in published["methods"]:
            pub = published["methods"][m]
            r["published"] = {"pairwise_accuracy": pub["pairwise_accuracy"], "ci95": pub["ci95"]}
        res["methods"][m] = r
    res["lengths"] = lengths(groups, keys, recs, n_boot, seed)
    pa = {m: per_problem(groups, keys, c) for m, c in conf.items()}
    res["paired"] = {name: paired(pa[a], pa[b], n_boot, seed) | {"published": published["paired"].get(name)}
                     for name, (a, b) in {"probe - sft_sigma": ("probe", "sft_sigma"),
                                          "probe - random_init_probe": ("probe", "random_init_probe"),
                                          "a5_p - probe": ("a5_p", "probe"),
                                          "self_consistency - probe": ("self_consistency", "probe"),
                                          "length - probe": ("length", "probe")}.items()
                     if a in pa and b in pa}
    out = REPO_ROOT / rc["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "summary.md").write_text(render(res, rc))
    return res


def _ci(r: dict, field: str = "pairwise_accuracy") -> str:
    v = r.get(field, r.get("mean"))
    return "–" if v is None else f"{v:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"


def render(res: dict, rc: dict) -> str:
    sp = rc["separated"]
    L = ["# X2 robustness checks (D92: post-hoc, exploratory)", "",
         f"X2's {res['n_kept']} kept problems. Every recomputed X2 method reproduces results/x2/results.json "
         f"(max gap {max(res['reproduction_gaps'].values()):.1e}). All CIs: bootstrap over problems.", "",
         f"Separated pairs: the correct answer |error| <= {sp['correct_max_error']:.2%}, the incorrect >= "
         f"{sp['incorrect_min_error']:.0%}.", "",
         "| method | pairwise accuracy (X2) | pooled pair-weighted (cluster bootstrap) | separated pairs only | "
         "separated: problems / pairs |", "|---|---|---|---|---|"]
    for m, r in res["methods"].items():
        s = r["separated"]
        L.append(f"| {LABELS.get(m, m)} | {_ci(r['pairwise_accuracy'])} | {_ci(r['pooled'], 'pooled_pairwise_accuracy')} "
                 f"| {_ci(s)} | {s['n_problems']} / {s['n_pairs']} |")
    ln = res["lengths"]
    L += ["", "| completion length (tokens) | value [95% CI] |", "|---|---|",
          f"| correct answers, mean | {_ci(ln['correct_mean_tokens'])} |",
          f"| incorrect answers, mean | {_ci(ln['incorrect_mean_tokens'])} |",
          f"| within problem, correct − incorrect | {_ci(ln['within_problem_difference'])} |",
          f"| problems where the correct answers are shorter on average | {ln['share_problems_correct_shorter']:.3f} |"]
    L += ["", "| paired difference (over problems) | mean [95% CI] | published |", "|---|---|---|"]
    for name, d in res["paired"].items():
        pub = d.get("published")
        ps = "–" if not pub else f"{pub['mean_diff']:+.3f} [{pub['ci95'][0]:+.3f}, {pub['ci95'][1]:+.3f}]"
        L.append(f"| {name} | {d['mean_diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | {ps} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["analyze"])
    ap.parse_args()
    print(render(analyze(), _config()))


if __name__ == "__main__":
    main()

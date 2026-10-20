"""X4 analysis (pre-registered in DECISIONS.md D89): does the model, asked to check its own answer, rank a correct
answer above an incorrect answer to the same problem?

    nice uv run python -m physcal.eval.x4_analysis analyze      # CPU, after the GPU runs

Arms (confidence: the yes-vs-no log-odds at the canonical "Verdict:" position): `sft` (primary), `shuffled` (the same
model, another problem's text), `random_init`. Primary: X2's within-problem pairwise accuracy on X2's pairs, each
arm vs 0.5 (Holm over the three; a missing arm counts as p = 1), and the paired differences sft − X2 probe,
sft − random_init, sft − shuffled (95% bootstrap CIs over problems). Secondary: sft − X2 SFT stated σ; AUROC across
problems; per depth; the raw P(yes); diagnostics; exploratory recomputation agreement (sft). Each arm is analyzed on
the kept problems it completed; paired differences on the problems both completed. Writes results/x4/results.json
and summary.md.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from physcal.config import REPO_ROOT
from physcal.eval import x2_pairs
from physcal.eval import x4_verify as V
from physcal.eval.x2_analysis import paired, per_problem, tie_share

ARMS = ("sft", "shuffled", "random_init")
LABELS = {"sft": "X4 self-verification, SFT (4-bit)", "shuffled": "control: another problem's text (SFT)",
          "random_init": "control: random-init model", "x2_probe": "X2 method 5: hidden-state probe (reference)",
          "x2_sft_sigma": "X2 method 1: SFT stated σ (reference)"}


# ------------------------------------------------------------------------------------------- inputs
def load_reference(cfg: dict, groups: dict, keys: list) -> dict:
    """X2's per-answer confidences (x4_verify.reference), after checking that the probe still reproduces X2's
    published pairwise accuracy on these pairs (the same pairs)."""
    ref = json.loads((REPO_ROOT / cfg["output_dir"] / "x2_reference.json").read_text())
    x2 = V._x2(cfg)
    out = {name: {(tuple(k), s): v for k, s, v in ref[field]}
           for name, field in (("x2_probe", "probe"), ("x2_sft_sigma", "sft_sigma"))}
    mine = float(per_problem(groups, keys, out["x2_probe"]).mean())
    published = json.loads((REPO_ROOT / x2["output_dir"] / "results.json").read_text())["methods"]["probe"]
    if abs(mine - published["pairwise_accuracy"]) > cfg["analysis"]["probe_check_atol"]:
        raise RuntimeError(f"X2 reference probe gives {mine} on these pairs, X2 published {published['pairwise_accuracy']}")
    return out


def complete_keys(rows: list[dict], groups: dict, keys: list) -> list:
    """The kept problems (in the kept order) whose every answer has a row."""
    have = {(tuple(r["key"]), r["sample"]) for r in rows}
    return [k for k in keys if all((tuple(k), a["sample"]) in have for a in groups[k])]


def conf_of(rows: list[dict], field: str) -> dict:
    return {(tuple(r["key"]), r["sample"]): r[field] for r in rows}


# ------------------------------------------------------------------------------------------- statistics
def auroc(groups: dict, keys: list, conf: dict, n_boot: int, seed: int) -> dict:
    """AUROC of the confidence for "correct", pooled over the answers of `keys` (answers without a confidence left
    out), with a bootstrap over problems."""
    from sklearn.metrics import roc_auc_score

    per = []
    for k in keys:
        ys, cs = [], []
        for a in groups[k]:
            c = conf.get((tuple(k), a["sample"]))
            if c is not None:
                ys.append(int(a["correct"]))
                cs.append(float(c))
        per.append((np.array(ys, dtype=int), np.array(cs)))
    y = np.concatenate([p[0] for p in per])
    c = np.concatenate([p[1] for p in per])
    if not 0 < y.sum() < len(y):
        return {"auroc": None, "ci95": [None, None], "n_answers": int(len(y))}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(per), len(per))
        yb = np.concatenate([per[i][0] for i in idx])
        if 0 < yb.sum() < len(yb):
            boots.append(roc_auc_score(yb, np.concatenate([per[i][1] for i in idx])))
    return {"auroc": float(roc_auc_score(y, c)), "ci95": np.percentile(boots, [2.5, 97.5]).tolist(),
            "n_answers": int(len(y))}


def method_summary(groups: dict, keys: list, conf: dict, n_boot: int, seed: int) -> dict:
    pa = per_problem(groups, keys, conf)
    r = x2_pairs.summarize(pa, n_boot, seed)
    r["share_pairs_tied_or_missing"] = tie_share(groups, keys, conf)
    r["by_depth"] = {str(d): x2_pairs.summarize(pa[[i for i, k in enumerate(keys) if k[1] == d]], n_boot, seed)
                     for d in sorted({k[1] for k in keys})}
    r["auroc_across_problems"] = auroc(groups, keys, conf, n_boot, seed)
    return r


def diagnostics(rows: list[dict], keys: list) -> dict:
    """Per arm, over the answers of `keys`: verdicts the model wrote itself, P(yes) by correctness, yes + no mass,
    truncations, the reading path."""
    kset = {tuple(k) for k in keys}
    rs = [r for r in rows if tuple(r["key"]) in kset]
    c = [r for r in rs if r["correct"]]
    w = [r for r in rs if not r["correct"]]
    mean = lambda xs: float(np.mean(xs)) if xs else None
    nat = lambda xs: [r for r in xs if r["natural_verdict"] in ("yes", "no")]
    return {"n_answers": len(rs), "share_wrote_verdict": mean([r["natural_verdict"] is not None for r in rs]),
            "share_wrote_yes_or_no": mean([r["natural_verdict"] in ("yes", "no") for r in rs]),
            "natural_yes_rate_correct": mean([r["natural_verdict"] == "yes" for r in nat(c)]),
            "natural_yes_rate_incorrect": mean([r["natural_verdict"] == "yes" for r in nat(w)]),
            "mean_p_yes_correct": mean([r["p_yes"] for r in c]), "mean_p_yes_incorrect": mean([r["p_yes"] for r in w]),
            "mean_yes_no_mass": mean([r["p_yes_raw"] + r["p_no_raw"] for r in rs]),
            "share_truncated": mean([r["truncated"] for r in rs]),
            "median_gen_tokens": float(np.median([r["gen_tokens"] for r in rs])) if rs else None,
            "paths": sorted({r["path"] for r in rs})}


def recompute_agreement(rows: list[dict], groups: dict, keys: list, tol: float, x2cfg: dict) -> dict:
    """Exploratory (sft): confidence 1 if the verification text holds an <answer> the verifier parses (against the
    shown problem) whose μ is within `tol` of the original μ, 0 if it parses and disagrees, None (a tie) otherwise."""
    from physcal.env.generator import problem_from_dict
    from physcal.verify.reward import reward, reward_config

    rw = reward_config()
    recs, _ = x2_pairs.kept_records(x2cfg)
    probs = {(r["problem"]["family"], r["problem"]["depth"], r["problem"]["seed"]): r["problem"] for r in recs}
    mu = {(tuple(k), a["sample"]): a["mu_si"] for k in keys for a in groups[k]}
    out = {}
    for r in rows:
        key = (tuple(r["key"]), r["sample"])
        if key not in mu:
            continue
        _, bd = reward(problem_from_dict(probs[tuple(r["shown_key"])]), r["generation"], rw)
        m0 = mu[key]
        ok = bd["reason"] == "ok" and bd.get("mu_si") is not None and m0 != 0
        out[key] = (1.0 if abs(bd["mu_si"] / m0 - 1) <= tol else 0.0) if ok else None
    return out


def reading(holm: dict, diffs: dict) -> dict:
    """D89's reading rules."""
    sup = {a: holm[a]["rejected"] for a in ARMS}
    lo = lambda name: diffs[name]["ci95"][0] if diffs.get(name) and not diffs[name].get("missing") else None
    above = lambda name: lo(name) is not None and lo(name) > 0
    succeeds = sup["sft"] and above("sft - random_init")
    specific = above("sft - shuffled")
    over_probe = above("sft - x2_probe")
    if succeeds and specific and over_probe:
        outcome = ("succeeds, problem-specific, above the probe: asking the model recovers answer-specific "
                   "self-knowledge that no X2 readout found, and it depends on checking the answer against the problem")
    elif succeeds and specific:
        outcome = ("succeeds and problem-specific, not above the probe: self-verification carries answer-specific "
                   "signal that depends on the problem, not detectably more than the X2 probe")
    elif succeeds:
        outcome = ("succeeds but not problem-specific: the signal is in the completion text itself (internal "
                   "consistency or surface features), not in checking the answer against the problem")
    elif not sup["sft"]:
        outcome = ("sft not Holm-supported: self-verification fails within problems, consistent with X2; asking the "
                   "model does not rank its right answers above its wrong ones")
    else:
        outcome = "another combination: reported as observed"
    return {"supported": sup, "self_verification_succeeds": succeeds, "problem_specific": specific,
            "above_x2_probe": over_probe, "outcome": outcome}


# ------------------------------------------------------------------------------------------- the analysis
def analyze(cfg: dict | None = None) -> dict:
    from physcal.eval.stats import holm

    cfg = cfg or V._config()
    ac = cfg["analysis"]
    n_boot, seed = ac["bootstrap"], ac["seed"]
    groups, keys = x2_pairs.kept_groups(V._x2(cfg))
    ref = load_reference(cfg, groups, keys)
    rows = {a: V.read_rows(a, cfg) for a in ARMS}
    done = {a: complete_keys(r, groups, keys) if r else [] for a, r in rows.items()}
    res: dict = {"decision": "D89 (X4)", "n_kept": len(keys), "methods": {}, "secondary": {}, "diagnostics": {},
                 "analyzed_problems": {a: len(done[a]) for a in ARMS}}
    pa = {}
    for a in ARMS:
        if not done[a]:
            res["methods"][a] = {"missing": True}
            continue
        conf = conf_of(rows[a], "log_odds")
        res["methods"][a] = method_summary(groups, done[a], conf, n_boot, seed)
        res["methods"][a]["n_pairs"] = int(sum(sum(x["correct"] for x in groups[k]) * sum(not x["correct"] for x in groups[k])
                                               for k in done[a]))
        res["secondary"][f"{a} raw P(yes)"] = method_summary(groups, done[a], conf_of(rows[a], "p_yes_raw"), n_boot, seed)
        res["diagnostics"][a] = diagnostics(rows[a], done[a])
        pa[a] = dict(zip(map(tuple, done[a]), per_problem(groups, done[a], conf)))
    base = done["sft"] or keys
    for name, conf in ref.items():
        res["methods"][name] = method_summary(groups, base, conf, n_boot, seed)
        pa[name] = dict(zip(map(tuple, base), per_problem(groups, base, conf)))
    p = {a: res["methods"][a].get("p_one_sided_gt_half", 1.0) if not res["methods"][a].get("missing") else 1.0
         for a in ARMS}
    res["holm"] = holm(p, ac["alpha"])
    res["paired"] = {}
    for other in ("x2_probe", "random_init", "shuffled", "x2_sft_sigma"):
        name = f"sft - {other}"
        common = [tuple(k) for k in keys if tuple(k) in pa.get("sft", {}) and tuple(k) in pa.get(other, {})]
        if not common:
            res["paired"][name] = {"missing": True}
            continue
        res["paired"][name] = paired(np.array([pa["sft"][k] for k in common]), np.array([pa[other][k] for k in common]),
                                     n_boot, seed) | {"n_problems": len(common),
                                                      "role": "secondary" if other == "x2_sft_sigma" else "primary"}
    if rows["sft"]:
        conf = recompute_agreement(rows["sft"], groups, done["sft"], ac["recompute_tolerance"], V._x2(cfg))
        r = method_summary(groups, done["sft"], conf, n_boot, seed)
        r["share_answers_with_parsed_recomputation"] = float(np.mean([v is not None for v in conf.values()]))
        res["secondary"]["sft recomputation agreement (exploratory)"] = r
    res["meta"] = {a: _meta(a, cfg) for a in ARMS}
    res["reading"] = reading(res["holm"], res["paired"])
    out = REPO_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "summary.md").write_text(render(res))
    return res


def _meta(arm: str, cfg: dict) -> dict | None:
    p = V.arm_dir(arm, cfg) / "meta.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return {"complete": m.get("complete"), "n_rows": m.get("n_rows"), "model_content_sha256": m.get("model_content_sha256"),
            "segments": [{k: s.get(k) for k in ("started", "seconds", "path", "logit_check", "max_reserved_gib",
                                                "oom_backoffs", "stopped_by_time")} for s in m.get("segments", [])]}


def render(res: dict) -> str:
    L = ["# X4: self-verification on the X2 pairs (D89)", "",
         f"X2's {res['n_kept']} kept problems; problems analyzed per arm: {res['analyzed_problems']}.", "",
         f"**Reading (D89):** {res['reading']['outcome']}.", "",
         "| method | pairwise accuracy [95% CI] | p (> 0.5) | Holm | ties / missing | AUROC across problems [95% CI] |",
         "|---|---|---|---|---|---|"]
    for m, label in LABELS.items():
        r = res["methods"].get(m, {"missing": True})
        if r.get("missing"):
            L.append(f"| {label} | missing | – | {'not supported (p = 1)' if m in ARMS else '–'} | – | – |")
            continue
        h = res["holm"].get(m)
        hs = "–" if h is None else ("supported" if h["rejected"] else "not supported")
        au = r["auroc_across_problems"]
        aus = "–" if au["auroc"] is None else f"{au['auroc']:.3f} [{au['ci95'][0]:.3f}, {au['ci95'][1]:.3f}]"
        L.append(f"| {label} | {r['pairwise_accuracy']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] | "
                 f"{r['p_one_sided_gt_half']:.3g} | {hs} | {r['share_pairs_tied_or_missing']:.1%} | {aus} |")
    L += ["", "| paired difference (over problems) | role | mean | 95% CI | problems |", "|---|---|---|---|---|"]
    for name, d in res["paired"].items():
        L.append(f"| {name} | missing | – | – | – |" if d.get("missing") else
                 f"| {name} | {d['role']} | {d['mean_diff']:+.3f} | [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | "
                 f"{d['n_problems']} |")
    present = {m: r for m, r in res["methods"].items() if not r.get("missing")}
    depths = sorted({d for r in present.values() for d in r["by_depth"]})
    L += ["", "| method | " + " | ".join(f"depth {d}" for d in depths) + " |", "|---|" + "---|" * len(depths)]
    for m, r in present.items():
        L.append(f"| {LABELS[m]} | " + " | ".join(
            f"{r['by_depth'][d]['pairwise_accuracy']:.3f} (n {r['by_depth'][d]['n_problems']})"
            if d in r["by_depth"] else "–" for d in depths) + " |")
    if res["secondary"]:
        L += ["", "| secondary (no verdict) | pairwise accuracy [95% CI] | ties / missing |", "|---|---|---|"]
        for name, r in res["secondary"].items():
            L.append(f"| {name} | {r['pairwise_accuracy']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] | "
                     f"{r['share_pairs_tied_or_missing']:.1%} |")
    if res["diagnostics"]:
        L += ["", "| diagnostic | " + " | ".join(res["diagnostics"]) + " |", "|---|" + "---|" * len(res["diagnostics"])]
        for f in next(iter(res["diagnostics"].values())):
            L.append(f"| {f} | " + " | ".join(_fmt(d[f]) for d in res["diagnostics"].values()) + " |")
    return "\n".join(L) + "\n"


def _fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
    return "–" if x is None else str(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["analyze"])
    ap.parse_args()
    print(render(analyze()))


if __name__ == "__main__":
    main()

"""D151 item 3c (EXPLORATORY, follow-up project; not the paper): the within-problem readouts of the 7B and the native
1.5B control, against the SFT 1.5B's X2 numbers (published, and restricted to the same pool prefix).

    nice uv run python -m physcal.eval.followup_scale analyze     # CPU, after the GPU queue (the probe uses 4 jobs)

Readouts per answer of a kept problem: self-consistency (share of the other k − 1 samples within 1%), direct P(yes)
(log-odds), length (shorter = more confident), the cross-fitted hidden-state probe (x2_analysis.crossfit_confidence).
Scored by X2's pairwise accuracy (ties 0.5), mean over kept problems, bootstrap over problems. Writes
<output_dir>/scale/results.json and summary.md.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.eval import followup_scale as F
from physcal.eval import x2_pairs

READOUTS = ("self_consistency", "yes", "probe", "length")
LABELS = {"self_consistency": "self-consistency (k − 1 others within 1%)", "yes": "direct P(yes) (X4 direct)",
          "probe": "hidden-state probe, cross-fitted", "length": "length only (shorter = more confident)"}


def graded(model: str, cfg: dict) -> tuple[dict, list, list[dict]]:
    """Re-grade every sampled row with the current parser; (groups of answers, kept keys, rows)."""
    probs = F.pool(cfg)
    tol = cfg["scale"]["tolerance"]
    rows = F.sampled_prefix(model, cfg)
    for r in rows:
        r.update(F.grade(probs[r["index"]], r["completion"], r["truncated"], tol))
    groups: dict = {}
    for r in rows:
        groups.setdefault(tuple(r["key"]), [])
        if r["value"] is not None:
            groups[tuple(r["key"])].append({"correct": bool(r["correct"]), "sample": r["sample"]})
    keys = [k for k, a in groups.items() if any(x["correct"] for x in a) and any(not x["correct"] for x in a)]
    return groups, keys, rows


def self_consistency(rows: list[dict], tol: float) -> dict:
    """x2_robustness's definition: the share of the problem's other samples (parsed or not) agreeing within tol."""
    by: dict = {}
    for r in rows:
        by.setdefault(tuple(r["key"]), []).append(r)
    out = {}
    for k, rs in by.items():
        for r in rs:
            m = r["value"]
            if m is None or m == 0:
                continue
            agree = sum(1 for x in rs if x["sample"] != r["sample"] and x["value"] is not None
                        and abs(x["value"] / m - 1) <= tol)
            out[(k, r["sample"])] = agree / (len(rs) - 1)
    return out


def model_confidences(model: str, cfg: dict) -> tuple[dict, list, dict, dict]:
    from physcal.eval import x2_analysis as XA

    sc = cfg["scale"]
    groups, keys, rows = graded(model, cfg)
    conf: dict = {"self_consistency": self_consistency(rows, sc["tolerance"]),
                  "length": {(tuple(r["key"]), r["sample"]): -float(r["n_tokens"]) for r in rows}}
    info: dict = {"n_candidates": len({r["index"] for r in rows}), "n_answers": len(rows),
                  "answered_share": float(np.mean([r["value"] is not None for r in rows])),
                  "accuracy_per_sample": float(np.mean([bool(r["correct"]) for r in rows])),
                  "truncated_share": float(np.mean([r["truncated"] for r in rows])),
                  "reasons": {k: sum(r["reason"] == k for r in rows) for k in sorted({r["reason"] for r in rows})},
                  "mean_tokens": float(np.mean([r["n_tokens"] for r in rows]))}
    y = F.out_dir(model, cfg) / "yes.jsonl"
    if y.exists():
        conf["yes"] = {(tuple(r["key"]), r["sample"]): r["log_odds"] for r in map(json.loads, y.open())}
        info["yes_rows"] = len(conf["yes"])
    try:
        layers, idx = F.load_hidden(model, cfg)
    except (TypeError, ValueError):
        layers, idx = None, []
    if layers:
        corr = {(tuple(r["key"]), r["sample"]): bool(r["correct"]) for r in rows}
        index = [{"key": list(k), "sample": s, "correct": corr[(k, s)]} for k, s in idx]
        pc = dict(sc["probe"])
        if pc.pop("even_layers_only"):
            layers = layers[::2]
        c, pinfo = XA.crossfit_confidence(layers, index, pc)
        conf["probe"] = {(tuple(k), s): v for (k, s), v in c.items()}
        info["probe"] = pinfo | {"n_layers": len(layers), "n": len(index)}
    return groups, keys, conf, info


def x2_on_prefix(n_candidates: int, cfg: dict) -> dict:
    """The SFT 1.5B's X2 readouts restricted to the pool's first n candidates: SC, X4 direct and length from the
    existing rows (the probe's per-answer values need X2's refit: reported as missing here)."""
    from physcal.eval import x2_robustness as XR

    x2 = load_config("x2")
    rc = load_config("x2_robustness")
    groups, keys, recs = XR.answer_records(x2)
    seeds = {p["seed"] for p in F.pool(cfg)[:n_candidates]}
    pk = [k for k in keys if k[2] in seeds]
    conf = {"self_consistency": XR.self_consistency_conf(recs, rc["self_consistency"]["agree_tolerance"]),
            "length": XR.length_conf(recs)}
    x4 = XR.x4_confidences(rc, groups, keys)
    if "x4_direct" in x4:
        conf["yes"] = x4["x4_direct"]
    n_boot, seed = cfg["scale"]["bootstrap"], cfg["scale"]["seed"]
    out = {"n_kept": len(pk), "methods": {}}
    for m, c in conf.items():
        pa = x2_pairs.pairwise_accuracy(groups, pk, lambda k, a: c.get((tuple(k), a["sample"])))
        out["methods"][m] = x2_pairs.summarize(pa, n_boot, seed) if len(pk) else None
        out["methods"][m] and out["methods"][m].update(per_problem=pa.tolist())
    return out


def unpaired(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), (n_boot, len(b)))].mean(axis=1)
    return {"diff": float(a.mean() - b.mean()), "ci95": np.percentile(da - db, [2.5, 97.5]).tolist()}


def analyze(cfg: dict | None = None) -> dict:
    cfg = cfg or F._config()
    sc = cfg["scale"]
    n_boot, seed = sc["bootstrap"], sc["seed"]
    res: dict = {"kind": "D151 item 3, exploratory (not the paper)", "models": {}, "x2_published": sc["x2_published"]}
    per: dict = {}
    for model in sc["models"]:
        if not (F.out_dir(model, cfg) / "samples").exists():
            continue
        groups, keys, conf, info = model_confidences(model, cfg)
        m = {"info": info, "n_kept": len(keys), "kept_share": len(keys) / max(1, info["n_candidates"]),
             "n_pairs": int(sum(sum(a["correct"] for a in groups[k]) * sum(not a["correct"] for a in groups[k])
                                for k in keys)), "readouts": {}}
        per[model] = {}
        for r in READOUTS:
            if r not in conf or not keys:
                m["readouts"][r] = {"missing": True}
                continue
            pa = x2_pairs.pairwise_accuracy(groups, keys, lambda k, a, c=conf[r]: c.get((tuple(k), a["sample"])))
            per[model][r] = (keys, pa)
            s = x2_pairs.summarize(pa, n_boot, seed)
            s["by_depth"] = {str(d): float(pa[[i for i, k in enumerate(keys) if k[1] == d]].mean())
                             for d in sorted({k[1] for k in keys})}
            m["readouts"][r] = s
        res["models"][model] = m
    if "7b" in res["models"]:
        n = res["models"]["7b"]["info"]["n_candidates"]
        x2p = x2_on_prefix(n, cfg)
        res["x2_sft_on_prefix"] = {"n_candidates": n, "n_kept": x2p["n_kept"],
                                   "methods": {k: ({kk: vv for kk, vv in v.items() if kk != "per_problem"} if v else None)
                                               for k, v in x2p["methods"].items()}}
        res["differences"] = {}
        for r in READOUTS:
            if r not in per.get("7b", {}):
                continue
            k7, a7 = per["7b"][r]
            d = {}
            if r in per.get("1.5b", {}):
                k1, a1 = per["1.5b"][r]
                d["7b - native 1.5b (unpaired)"] = unpaired(a7, a1, n_boot, seed)
                common = [k for k in k7 if k in set(k1)]
                if common:
                    i7, i1 = [k7.index(k) for k in common], [k1.index(k) for k in common]
                    diff = a7[i7] - a1[i1]
                    rng = np.random.default_rng(seed)
                    bs = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
                    d["7b - native 1.5b (paired, common kept problems)"] = {
                        "n": len(common), "diff": float(diff.mean()), "ci95": np.percentile(bs, [2.5, 97.5]).tolist()}
            xm = x2p["methods"].get(r)
            if xm:
                d["7b - SFT 1.5b X2 (same prefix, unpaired)"] = unpaired(a7, np.array(xm["per_problem"]), n_boot, seed)
            res["differences"][r] = d
    out = REPO_ROOT / cfg["output_dir"] / "scale"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "summary.md").write_text(render(res))
    print(render(res))
    return res


def render(res: dict) -> str:
    ci = lambda r: f"{r['pairwise_accuracy']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"
    L = ["# Item 3: within-problem readouts, 7B vs 1.5B (D151 / D152; exploratory, not the paper)", "",
         "| readout | " + " | ".join(f"{m} ({res['models'][m]['n_kept']} kept)" for m in res["models"]) +
         " | SFT 1.5B X2, same prefix | SFT 1.5B X2 published (516 kept) |", "|---|" + "---|" * (len(res["models"]) + 2)]
    pub = {"self_consistency": "self_consistency", "yes": "x4_direct", "probe": "probe_crossfit", "length": "length"}
    for r in READOUTS:
        cells = [("–" if res["models"][m]["readouts"][r].get("missing") else ci(res["models"][m]["readouts"][r]))
                 for m in res["models"]]
        xp = (res.get("x2_sft_on_prefix") or {}).get("methods", {}).get(r)
        L.append(f"| {LABELS[r]} | " + " | ".join(cells) + f" | {ci(xp) if xp else '–'} | "
                 f"{res['x2_published'][pub[r]]:.3f} |")
    for m, v in res["models"].items():
        i = v["info"]
        L += ["", f"**{m}:** {i['n_candidates']} candidates, {i['n_answers']} samples; answered "
                  f"{i['answered_share']:.1%}, correct per sample {i['accuracy_per_sample']:.1%}, truncated "
                  f"{i['truncated_share']:.1%}, mean {i['mean_tokens']:.0f} tokens; kept {v['n_kept']} "
                  f"({v['kept_share']:.1%}), {v['n_pairs']} pairs. Reasons: {i['reasons']}."]
        if "probe" in i:
            L.append(f"Probe folds: {[(f['layer'], f['C']) for f in i['probe']['folds']]} ({i['probe']['n_layers']} layers).")
    if res.get("differences"):
        L += ["", "| readout | comparison | difference [95% CI] |", "|---|---|---|"]
        for r, d in res["differences"].items():
            for name, x in d.items():
                L.append(f"| {r} | {name} | {x['diff']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}] |")
    return "\n".join(L) + "\n"

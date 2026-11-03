"""D151 item 3c (EXPLORATORY, follow-up project; not the paper): the within-problem readouts of the 7B and the native
1.5B control, against the SFT 1.5B's X2 numbers (published, and restricted to the same pool prefix).

    nice uv run python -m physcal.eval.followup_scale analyze     # CPU, after the GPU queue (the probe uses 4 jobs)

Readouts per answer of a kept problem: self-consistency (share of the other k − 1 samples within 1%), direct P(yes)
(log-odds), length (shorter = more confident), the cross-fitted hidden-state probe (x2_analysis.crossfit_confidence).
Scored by X2's pairwise accuracy (ties 0.5), mean over kept problems, bootstrap over problems. Writes
<output_dir>/scale/results.json and summary.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.eval import followup_scale as F
from physcal.eval import x2_pairs

READOUTS = ("self_consistency", "yes", "probe", "probe_random", "length")
LABELS = {"self_consistency": "self-consistency (k − 1 others within 1%)", "yes": "direct P(yes) (X4 direct)",
          "probe": "hidden-state probe, cross-fitted", "probe_random": "the same probe on random weights (D160)",
          "length": "length only (shorter = more confident)"}


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
    corr = {(tuple(r["key"]), r["sample"]): bool(r["correct"]) for r in rows}
    for name, kind in (("probe", "hidden"), ("probe_random", "hidden_random")):
        got = probe_confidence(model, kind, corr, cfg)
        if got:
            conf[name], info[name] = got
    return groups, keys, conf, info


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def probe_inputs(model: str, kind: str, corr: dict, cfg: dict) -> tuple[list[dict], str, Path] | None:
    """The probe's answers in the parts' order (key, sample, label), the sha256 of its inputs (the probe settings,
    the answers and their labels, every part's bytes) and its cache path; None without parts."""
    parts = sorted((F.out_dir(model, cfg) / kind).glob("part_[0-9][0-9][0-9][0-9].npz"))
    if not parts:
        return None
    idx = [tuple(x) for f in parts for x in json.loads(str(np.load(f)["index"]))]
    index = [{"key": [a, b, c], "sample": s, "correct": corr[((a, b, c), s)]} for a, b, c, s in idx]
    h = hashlib.sha256(json.dumps({"probe": cfg["scale"]["probe"], "index": index}, sort_keys=True).encode())
    for f in parts:
        h.update(f"{f.name}:{_file_sha256(f)}".encode())
    return index, h.hexdigest(), F.out_dir(model, cfg) / f"probe_{kind}.json"


def probe_confidence(model: str, kind: str, corr: dict, cfg: dict) -> tuple[dict, dict] | None:
    """The cross-fitted probe (x2_analysis.crossfit_confidence) on <model>/<kind> (hidden, or D160's hidden_random),
    cached in <model>/probe_<kind>.json under the sha256 of its inputs (`probe_inputs`); a cache with other inputs is
    refit. None without parts."""
    from physcal.eval import x2_analysis as XA

    if (got := probe_inputs(model, kind, corr, cfg)) is None:
        return None
    index, sha, cache = got
    if cache.exists() and (c := json.loads(cache.read_text()))["inputs_sha256"] == sha:
        return {(tuple(k), s): v for k, s, v in c["conf"]}, c["info"]
    layers, _ = F.load_hidden(model, cfg, kind)
    pc = dict(cfg["scale"]["probe"])
    if pc.pop("even_layers_only"):
        layers = layers[::2]
    t0 = time.time()
    conf, pinfo = XA.crossfit_confidence(layers, index, pc)
    info = pinfo | {"n_layers": len(layers), "n": len(index), "seconds": time.time() - t0,
                    "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    cache.write_text(json.dumps({"decision": "D160: the cross-fitted probe's per-answer values (cache)",
                                 "inputs_sha256": sha, "info": info,
                                 "conf": [[list(k), s, v] for (k, s), v in conf.items()]}))
    return {(tuple(k), s): v for (k, s), v in conf.items()}, info


def refit_path(cfg: dict) -> Path:
    return REPO_ROOT / cfg["output_dir"] / "scale" / "x2_sft_probe_refit.json"


def x2_probe_refit(cfg: dict | None = None) -> dict:
    """D160 item 3 (D151 (2)): X2's cross-fitted SFT probe, refit under X2's own settings (x2.yaml analysis.probe; the
    thread environment is recorded). Its per-answer values are kept only if it reproduces X2's published pairwise
    accuracy within D92's tolerance (x2_robustness reproduce_atol); otherwise the refit is recorded as refused."""
    from physcal.eval import x2_analysis as XA
    from physcal.eval import x2_hidden
    from physcal.eval import x2_robustness as XR

    cfg = cfg or F._config()
    x2, rc = load_config("x2"), load_config("x2_robustness")
    groups, keys, _ = XR.answer_records(x2)
    layers, index = x2_hidden.load("sft", "x2", x2)
    t0 = time.time()
    conf, info = XA.crossfit_confidence(layers, index, x2["analysis"]["probe"])
    published = json.loads((REPO_ROOT / x2["output_dir"] / "results.json").read_text())["methods"]["probe_crossfit"]
    mine = float(XA.per_problem(groups, keys, conf).mean())
    gap = abs(mine - published["pairwise_accuracy"])
    ok = gap <= rc["reproduce_atol"]
    res = {"decision": "D160 item 3 (D151 (2)): X2's cross-fitted SFT probe refit (exploratory)", "reproduced": ok,
           "recomputed": mine, "published": published["pairwise_accuracy"], "gap": gap, "atol": rc["reproduce_atol"],
           "n_kept": len(keys), "probe": x2["analysis"]["probe"], "folds": info["folds"], "seconds": time.time() - t0,
           "threads": {"cpu_count": os.cpu_count(), **{v: os.environ.get(v) for v in (
               "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
           "conf": [[list(k), s, v] for (k, s), v in sorted(conf.items())] if ok else None}
    path = refit_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "conf"}, indent=1))
    if not ok:
        raise SystemExit(f"the refit does not reproduce X2 ({mine} vs {published['pairwise_accuracy']}): refused")
    return res


def x2_prefix_data(n_candidates: int, cfg: dict) -> tuple[dict, list, dict, list[dict]]:
    """The SFT 1.5B's X2 answers on the pool's first n candidates: (groups, kept keys on the prefix, confidences, the
    kept problems' records). SC, X4 direct and length from the existing rows; the probe from D160's refit
    (x2_probe_refit) when it reproduced X2, else missing."""
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
    rf = refit_path(cfg)
    if rf.exists() and (r := json.loads(rf.read_text()))["reproduced"]:
        conf["probe"] = {(tuple(k), s): v for k, s, v in r["conf"]}
    return groups, pk, conf, recs


def x2_on_prefix(n_candidates: int, cfg: dict) -> dict:
    """The SFT 1.5B's X2 readouts restricted to the pool's first n candidates (`x2_prefix_data`)."""
    groups, pk, conf, _ = x2_prefix_data(n_candidates, cfg)
    n_boot, seed = cfg["scale"]["bootstrap"], cfg["scale"]["seed"]
    out = {"n_kept": len(pk), "methods": {}}
    for m, c in conf.items():
        pa = x2_pairs.pairwise_accuracy(groups, pk, lambda k, a: c.get((tuple(k), a["sample"])))
        out["methods"][m] = x2_pairs.summarize(pa, n_boot, seed) if len(pk) else None
        out["methods"][m] and out["methods"][m].update(per_problem=pa.tolist())
    return out


def paired(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict:
    """a − b on the same problems: the mean difference and a bootstrap over problems."""
    diff = a - b
    rng = np.random.default_rng(seed)
    bs = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
    return {"n": len(diff), "diff": float(diff.mean()), "ci95": np.percentile(bs, [2.5, 97.5]).tolist()}


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
    for model, want in sc["reproduce_probe"].items():   # D160: a refit must give the earlier result exactly
        got = res["models"].get(model, {}).get("readouts", {}).get("probe", {}).get("pairwise_accuracy")
        if got is not None and abs(got - want) > sc["reproduce_atol"]:
            raise SystemExit(f"the {model} probe gives {got}, not {want} (D160): stopped before writing results")
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
                    d["7b - native 1.5b (paired, common kept problems)"] = paired(a7[i7], a1[i1], n_boot, seed)
            xm = x2p["methods"].get(r)
            if xm:
                d["7b - SFT 1.5b X2 (same prefix, unpaired)"] = unpaired(a7, np.array(xm["per_problem"]), n_boot, seed)
                if r in per.get("1.5b", {}):   # D160: format, training and decoding at a fixed size
                    d["native 1.5b - SFT 1.5b X2 (same prefix, unpaired)"] = unpaired(
                        per["1.5b"][r][1], np.array(xm["per_problem"]), n_boot, seed)
            res["differences"][r] = d
    res["controls"] = {}   # D160: paired within a model (the same kept problems), and the probe's net scale effect
    for model, rs in per.items():
        c = {}
        if "probe" in rs and "probe_random" in rs:
            c["probe - probe_random (paired)"] = paired(rs["probe"][1], rs["probe_random"][1], n_boot, seed)
        for r in ("yes", "probe"):
            if r in rs and "length" in rs:
                c[f"{r} - length (paired)"] = paired(rs[r][1], rs["length"][1], n_boot, seed)
        res["controls"][model] = c
    if all("probe" in per.get(m, {}) and "probe_random" in per.get(m, {}) for m in ("7b", "1.5b")):
        net = {m: per[m]["probe"][1] - per[m]["probe_random"][1] for m in ("7b", "1.5b")}
        res["controls"]["7b vs native 1.5b"] = {
            "(probe - probe_random): 7b - native 1.5b (unpaired)": unpaired(net["7b"], net["1.5b"], n_boot, seed)}
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
        published = res["x2_published"].get(pub.get(r))
        L.append(f"| {LABELS[r]} | " + " | ".join(cells) + f" | {ci(xp) if xp else '–'} | "
                 f"{'–' if published is None else f'{published:.3f}'} |")
    for m, v in res["models"].items():
        i = v["info"]
        L += ["", f"**{m}:** {i['n_candidates']} candidates, {i['n_answers']} samples; answered "
                  f"{i['answered_share']:.1%}, correct per sample {i['accuracy_per_sample']:.1%}, truncated "
                  f"{i['truncated_share']:.1%}, mean {i['mean_tokens']:.0f} tokens; kept {v['n_kept']} "
                  f"({v['kept_share']:.1%}), {v['n_pairs']} pairs. Reasons: {i['reasons']}."]
        for name in ("probe", "probe_random"):
            if name in i:
                L.append(f"{name} folds (layer, C): {[(f['layer'], f['C']) for f in i[name]['folds']]} "
                         f"({i[name]['n_layers']} layers).")
    if res.get("differences") or res.get("controls"):
        L += ["", "| readout / model | comparison | difference [95% CI] |", "|---|---|---|"]
        for group in ("differences", "controls"):
            for r, d in res.get(group, {}).items():
                for name, x in d.items():
                    L.append(f"| {r} | {name} | {x['diff']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}] |")
    return "\n".join(L) + "\n"

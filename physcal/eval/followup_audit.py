"""D164 (EXPLORATORY, follow-up project; not the paper): a zero-GPU confound audit of D162's within-problem table.

    T="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
    env $T nice -n 19 uv run python -m physcal.eval.followup_audit text --model 7b     # then --model 1.5b
    env $T nice -n 19 uv run python -m physcal.eval.followup_audit analyze

1. Truncation: (a) truncated samples dropped (from the answers, the kept rule and self-consistency's denominator);
   (b) problems with any truncated sample dropped.
2. The problems kept by both Kaggle models, stratified by each model's pass count, with a pass-count-standardized
   7B − native difference.
3. An A5-style text classifier (nested_baselines' problem + answer + reasoning TF-IDF, A5's p grid), cross-fitted on
   the probe's folds; probe − text; and every readout on length-matched pairs.
The per-answer readouts are D162's (the probes from their caches, never refit); only the scored problems or pairs
change. The unchanged recompute must reproduce D162's results.json. Writes <output_dir>/audit/results.json and
summary.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.eval import followup_scale as F
from physcal.eval import followup_scale_analysis as A
from physcal.eval import x2_pairs

READOUTS = A.READOUTS + ("text",)
LABELS = A.LABELS | {"text": "A5-style text classifier (D164)"}
CONTROLS = (("probe", "probe_random"), ("yes", "length"), ("probe", "length"),   # D162's
            ("probe", "text"), ("text", "probe_random"), ("text", "length"))     # D164's
SFT = "sft"
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def audit_dir(cfg: dict) -> Path:
    return REPO_ROOT / cfg["output_dir"] / cfg["audit"]["output_dir"]


def _threads() -> dict:
    return {v: os.environ.get(v) for v in THREAD_VARS}


# ------------------------------------------------------------------------------------------- the answers
def group_rows(rows: list[dict]) -> tuple[dict, list]:
    """followup_scale_analysis.graded's grouping: each problem's parsed answers, and the kept problems (>= 1 correct
    and >= 1 incorrect answer)."""
    groups: dict = {}
    for r in rows:
        groups.setdefault(tuple(r["key"]), [])
        if r["value"] is not None:
            groups[tuple(r["key"])].append({"correct": bool(r["correct"]), "sample": r["sample"]})
    keys = [k for k, a in groups.items() if any(x["correct"] for x in a) and any(not x["correct"] for x in a)]
    return groups, keys


def cached_probe(model: str, kind: str, corr: dict, cfg: dict) -> dict:
    """D162's cross-fitted probe values from their cache. A cache whose inputs changed is refused: D164 never refits."""
    got = A.probe_inputs(model, kind, corr, cfg)
    if got is None:
        raise SystemExit(f"{model}/{kind}: no hidden-state parts")
    _, sha, path = got
    c = json.loads(path.read_text())
    if c["inputs_sha256"] != sha:
        raise SystemExit(f"{path}: the cache does not match its inputs; D164 does not refit")
    return {(tuple(k), s): v for k, s, v in c["conf"]}


def _data(groups: dict, keys: list, conf: dict, ntok: dict, truncated: set) -> dict:
    return {"groups": groups, "keys": keys, "conf": conf, "ntok": ntok, "truncated": truncated,
            "pass": {k: sum(a["correct"] for a in groups[k]) for k in keys}}


def kaggle_data(model: str, cfg: dict) -> tuple[dict, dict]:
    """(D162's answers and readouts, the same with truncated samples dropped: 1(a)). Self-consistency is recomputed
    on the remaining samples; the other readouts are per answer and unchanged."""
    groups, keys, rows = A.graded(model, cfg)
    if group_rows(rows) != (groups, keys):
        raise SystemExit(f"{model}: group_rows disagrees with graded")
    tol = cfg["scale"]["tolerance"]
    corr = {(tuple(r["key"]), r["sample"]): bool(r["correct"]) for r in rows}
    fixed = {"yes": {(tuple(r["key"]), r["sample"]): r["log_odds"]
                     for r in map(json.loads, (F.out_dir(model, cfg) / "yes.jsonl").open())},
             "probe": cached_probe(model, "hidden", corr, cfg),
             "probe_random": cached_probe(model, "hidden_random", corr, cfg)}
    if (text := text_confidence(model, cfg, fit=False, rows=rows)) is None:
        raise SystemExit(f"no text classifier for {model}: run `followup_audit text --model {model}` first")
    fixed["text"] = text
    ntok = {(tuple(r["key"]), r["sample"]): r["n_tokens"] for r in rows}
    truncated = {tuple(r["key"]) for r in rows if r["truncated"]}

    def build(rs: list[dict]) -> dict:
        g, k = group_rows(rs)
        conf = {"self_consistency": A.self_consistency(with_others(rs, lambda r: tuple(r["key"])), tol),
                "length": {(tuple(r["key"]), r["sample"]): -float(r["n_tokens"]) for r in rs}} | fixed
        return _data(g, k, conf, ntok, truncated)

    return build(rows), build([r for r in rows if not r["truncated"]])


def sft_data(n_candidates: int, cfg: dict) -> tuple[dict, dict]:
    """The SFT 1.5B's X2 answers on the same prefix (D162's column), and the same with truncated samples dropped."""
    from physcal.eval import x2_robustness as XR

    x2, rc = load_config("x2"), load_config("x2_robustness")
    groups, pk, conf, recs = A.x2_prefix_data(n_candidates, cfg)
    key = lambda r: (r["problem"]["family"], r["problem"]["depth"], r["problem"]["seed"])  # noqa: E731
    ntok = {(key(r), r.get("sample", 0)): r["n_tokens"] for r in recs}
    truncated = {key(r) for r in recs if r["truncated"]}
    base = _data(groups, pk, conf, ntok, truncated)
    kept = [r for r in recs if not r["truncated"]]
    g = {tuple(k): v for k, v in x2_pairs.group(kept, x2).items()}
    still = set(x2_pairs.select(g, x2))
    conf_a = conf | {"self_consistency": XR.self_consistency_conf(with_others(kept, key),
                                                                  rc["self_consistency"]["agree_tolerance"]),
                     "length": XR.length_conf(kept)}
    return base, _data(g, [k for k in pk if k in still], conf_a, ntok, truncated)


def with_others(rows: list[dict], key) -> list[dict]:
    """The rows of problems with >= 2 rows: a lone sample has no others for self-consistency (and is never kept)."""
    n = Counter(key(r) for r in rows)
    return [r for r in rows if n[key(r)] > 1]


def restrict(data: dict, keep) -> dict:
    """Every model's data with its kept problems filtered by keep(model, key), order kept."""
    return {m: d | {"keys": [k for k in d["keys"] if keep(m, k)]} for m, d in data.items()}


# ------------------------------------------------------------------------------------------- pairs
def length_matched(n_a: int, n_b: int, rel: float) -> bool:
    return abs(n_a - n_b) <= rel * max(n_a, n_b)


def _pairs(d: dict, k: tuple, rel: float | None) -> list[tuple[dict, dict]]:
    g = d["groups"][k]
    return [(a, b) for a in g if a["correct"] for b in g if not b["correct"]
            if rel is None or length_matched(d["ntok"][(k, a["sample"])], d["ntok"][(k, b["sample"])], rel)]


def per_problem(d: dict, keys: list, conf: dict, rel: float | None = None) -> np.ndarray:
    """X2's per-problem pairwise accuracy (ties and missing 0.5). With `rel`, over the length-matched pairs only; a
    problem without one is NaN (callers drop such problems first)."""
    if rel is None:
        return x2_pairs.pairwise_accuracy(d["groups"], keys, lambda k, a: conf.get((tuple(k), a["sample"])))
    out = []
    for k in keys:
        s = [0.5 if (x is None or y is None or x == y) else float(x > y)
             for x, y in ((conf.get((k, a["sample"])), conf.get((k, b["sample"]))) for a, b in _pairs(d, k, rel))]
        out.append(float(np.mean(s)) if s else np.nan)
    return np.array(out)


def compare(data: dict, sc: dict, rel: float | None = None) -> dict:
    """D162's table on `data` ({7b, 1.5b, sft}): every readout per model, 7B − native (unpaired, and paired on the
    problems both keep), each Kaggle model − SFT, the paired controls and the probe's net scale effect. With `rel`,
    over length-matched pairs, on the problems that have one."""
    n_boot, seed = sc["bootstrap"], sc["seed"]
    res: dict = {"models": {}, "differences": {}, "controls": {}}
    per: dict = {}
    for m, d in data.items():
        keys = d["keys"] if rel is None else [k for k in d["keys"] if _pairs(d, k, rel)]
        per[m] = {"keys": keys}
        mr = {"n_kept": len(keys), "n_pairs": sum(len(_pairs(d, k, rel)) for k in keys), "readouts": {}}
        for r in READOUTS:
            if r not in d["conf"] or not keys:
                continue
            pa = per_problem(d, keys, d["conf"][r], rel)
            per[m][r] = pa
            s = x2_pairs.summarize(pa, n_boot, seed)
            s["by_depth"] = {str(dep): float(pa[[i for i, k in enumerate(keys) if k[1] == dep]].mean())
                             for dep in sorted({k[1] for k in keys})}
            mr["readouts"][r] = s
        res["models"][m] = mr
    k7, k1 = per["7b"]["keys"], per["1.5b"]["keys"]
    common = [k for k in k7 if k in set(k1)]
    i7, i1 = [k7.index(k) for k in common], [k1.index(k) for k in common]
    res["n_common"] = len(common)
    for r in READOUTS:
        dd = {}
        if r in per["7b"] and r in per["1.5b"]:
            dd["7b - native 1.5b (unpaired)"] = A.unpaired(per["7b"][r], per["1.5b"][r], n_boot, seed)
            if common:
                dd["7b - native 1.5b (paired, common kept problems)"] = A.paired(
                    per["7b"][r][i7], per["1.5b"][r][i1], n_boot, seed)
        for m, name in (("7b", "7b - SFT 1.5b X2 (same prefix, unpaired)"),
                        ("1.5b", "native 1.5b - SFT 1.5b X2 (same prefix, unpaired)")):
            if r in per.get(SFT, {}) and r in per[m]:
                dd[name] = A.unpaired(per[m][r], per[SFT][r], n_boot, seed)
        if dd:
            res["differences"][r] = dd
    for m in ("7b", "1.5b"):
        res["controls"][m] = {f"{a} - {b} (paired)": A.paired(per[m][a], per[m][b], n_boot, seed)
                              for a, b in CONTROLS if a in per[m] and b in per[m]}
    if all("probe" in per[m] and "probe_random" in per[m] for m in ("7b", "1.5b")):
        net = {m: per[m]["probe"] - per[m]["probe_random"] for m in ("7b", "1.5b")}
        res["controls"]["7b vs native 1.5b"] = {
            "(probe - probe_random): 7b - native 1.5b (unpaired)": A.unpaired(net["7b"], net["1.5b"], n_boot, seed)}
    return res


def d162_gap(mine: dict, d162: dict) -> float:
    """The largest absolute difference between this recompute and D162's results.json, over every number they share
    (pairwise accuracies, differences, controls and their CIs)."""
    gaps = [0.0]

    def add(a: dict, b: dict, keys: tuple):
        for k in keys:
            gaps.extend(abs(np.subtract(a[k], b[k])).ravel().tolist())

    for m, v in d162["models"].items():
        for r, s in v["readouts"].items():
            add(mine["models"][m]["readouts"][r], s, ("pairwise_accuracy", "ci95"))
    for r, s in d162["x2_sft_on_prefix"]["methods"].items():
        add(mine["models"][SFT]["readouts"][r], s, ("pairwise_accuracy", "ci95"))
    for group in ("differences", "controls"):
        for r, d in d162[group].items():
            for name, s in d.items():
                add(mine[group][r][name], s, ("diff", "ci95"))
    return max(gaps)


# ------------------------------------------------------------------------------------------- pass counts
def by_pass_count(data: dict, sc: dict) -> dict:
    """Item 2 on problems both Kaggle models keep (the same keys, same order): each readout per model by its own pass
    count c (correct of k), and a pass-count-standardized 7B − native difference (pooled weights, paired bootstrap;
    a stratum empty in a replicate gives its weight to the others in proportion)."""
    n_boot, seed, k = sc["bootstrap"], sc["seed"], sc["generation"]["k"]
    d7, d1 = data["7b"], data["1.5b"]
    if d7["keys"] != d1["keys"]:
        raise ValueError("by_pass_count needs the same problems, in the same order, for both models")
    keys, cs = d7["keys"], np.arange(1, k)
    c = {m: np.array([data[m]["pass"][q] for q in keys]) for m in ("7b", "1.5b")}
    pooled = np.concatenate([c["7b"], c["1.5b"]])
    w = np.array([np.mean(pooled == s) for s in cs])
    out: dict = {"pass_counts": cs.tolist(), "weights": w.tolist(),
                 "n": {m: {str(s): int(np.sum(c[m] == s)) for s in cs} for m in c}, "strata": {}, "standardized": {}}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), (n_boot, len(keys)))

    def standardized(a: np.ndarray, cc: np.ndarray, ix: np.ndarray | None = None) -> np.ndarray:
        A_, C_ = (a[None, :], cc[None, :]) if ix is None else (a[ix], cc[ix])   # (replicates, problems)
        num, den = np.zeros(len(A_)), np.zeros(len(A_))
        for s, ws in zip(cs, w):
            m = C_ == s
            cnt = m.sum(1)
            num += np.where(cnt > 0, ws * np.where(m, A_, 0.0).sum(1) / np.maximum(cnt, 1), 0.0)
            den += ws * (cnt > 0)
        return num / den

    for r in READOUTS:
        if r not in d7["conf"] or r not in d1["conf"]:
            continue
        pa = {m: per_problem(data[m], keys, data[m]["conf"][r]) for m in ("7b", "1.5b")}
        out["strata"][r] = {m: {str(s): x2_pairs.summarize(pa[m][c[m] == s], n_boot, seed) for s in cs
                                if np.any(c[m] == s)} for m in pa}
        v = {m: float(standardized(pa[m], c[m])[0]) for m in pa}
        boots = standardized(pa["7b"], c["7b"], idx) - standardized(pa["1.5b"], c["1.5b"], idx)
        out["standardized"][r] = {"7b": v["7b"], "1.5b": v["1.5b"], "diff": v["7b"] - v["1.5b"],
                                  "ci95": np.percentile(boots, [2.5, 97.5]).tolist()}
    return out


# ------------------------------------------------------------------------------------------- the text classifier
def text_fields(problem_text: str, completion: str) -> dict:
    """A5's problem + answer + reasoning fields for a native answer: the native problem text, the last box's content,
    and the completion before that box. What the probe's state at the box's closing brace can see; nothing after."""
    box = F.last_boxed(completion)
    if box is None:
        raise ValueError("an answered sample without a closed box")
    return {"problem": F.native_text(problem_text), "answer": box[0],
            "reasoning": completion[:completion.rfind("\\boxed{")]}


def text_items(model: str, cfg: dict, rows: list[dict] | None = None) -> tuple[list[dict], np.ndarray, list[str],
                                                                             list[dict]]:
    """The probe's answers in the probe's order: (text records, labels, problem groups, the probe's index). `rows`:
    the model's graded rows, if already read."""
    rows = rows if rows is not None else A.graded(model, cfg)[2]
    corr = {(tuple(r["key"]), r["sample"]): bool(r["correct"]) for r in rows}
    index, _, _ = A.probe_inputs(model, "hidden", corr, cfg)
    probs = F.pool(cfg)
    by = {(tuple(r["key"]), r["sample"]): r for r in rows}
    recs = []
    for it in index:
        r = by[(tuple(it["key"]), it["sample"])]
        recs.append(text_fields(probs[r["index"]]["text"], r["completion"]))
    y = np.array([it["correct"] for it in index], dtype=int)
    return recs, y, ["|".join(map(str, it["key"])) for it in index], index


def text_grid(a5: dict) -> list[tuple[str, object]]:
    """A5's p grid, in order: logistic regression by C, then the GBT configurations."""
    return [("linear", C) for C in a5["grids"]["logistic_C"]] + [("gbt", hp) for hp in a5["grids"]["gbt"]]


def _folds(groups: list[str], n_folds: int) -> list:
    from sklearn.model_selection import GroupKFold

    return list(GroupKFold(n_folds).split(np.zeros((len(groups), 1)), None, groups))


def select_text(recs: list[dict], y: np.ndarray, groups: list[str], feature_set: str, inner_folds: int,
                a5: dict) -> dict:
    """The grid's pooled out-of-fold AUROC under GroupKFold by problem; the best candidate (ties: the earlier)."""
    from sklearn.metrics import roc_auc_score

    from physcal.eval import nested_baselines as nb

    folds, grid, scores = _folds(groups, inner_folds), text_grid(a5), []
    for kind, hp in grid:
        oof = np.zeros(len(y))
        for tr, te in folds:
            cand = nb.Candidate(feature_set, kind, "p", hp, a5).fit([recs[i] for i in tr], y[tr])
            oof[te] = cand.raw([recs[i] for i in te])
        scores.append(float(roc_auc_score(y, oof)))
    j = max(range(len(grid)), key=lambda i: (scores[i], -i))
    return {"kind": grid[j][0], "hp": grid[j][1], "inner_cv_auroc": scores[j], "table": scores}


def crossfit_text(recs: list[dict], y: np.ndarray, groups: list[str], outer_folds: int, feature_set: str,
                  inner_folds: int, a5: dict) -> tuple[np.ndarray, list[dict]]:
    """P(correct) for every answer from a model fit on the other folds' problems (crossfit_confidence's outer folds),
    with the candidate chosen by inner CV on the training part."""
    from physcal.eval import nested_baselines as nb

    p, chosen = np.full(len(y), np.nan), []
    for tr, te in _folds(groups, outer_folds):
        sel = select_text([recs[i] for i in tr], y[tr], [groups[i] for i in tr], feature_set, inner_folds, a5)
        cand = nb.Candidate(feature_set, sel["kind"], "p", sel["hp"], a5).fit([recs[i] for i in tr], y[tr])
        p[te] = cand.raw([recs[i] for i in te])
        chosen.append(sel)
    return p, chosen


def text_confidence(model: str, cfg: dict, fit: bool = True, rows: list[dict] | None = None) -> dict | None:
    """The text classifier's per-answer P(correct), cached in audit/text_<model>.json under the sha256 of its inputs
    (settings, A5's features and grid, the answers, labels and texts). Without a matching cache: fit it, or None."""
    tc = cfg["audit"]["text"]
    a5 = load_config(tc["a5_config"])
    recs, y, groups, index = text_items(model, cfg, rows)
    h = hashlib.sha256(json.dumps({"text": tc, "outer_folds": cfg["scale"]["probe"]["folds"],
                                   "a5": {k: a5[k] for k in ("text", "grids", "seed")},
                                   "index": index, "recs": recs}, sort_keys=True).encode()).hexdigest()
    path = audit_dir(cfg) / f"text_{model}.json"
    if path.exists() and (c := json.loads(path.read_text()))["inputs_sha256"] == h:
        return {(tuple(k), s): v for k, s, v in c["conf"]}
    if not fit:
        return None
    t0 = time.time()
    p, chosen = crossfit_text(recs, y, groups, cfg["scale"]["probe"]["folds"], tc["feature_set"],
                              tc["inner_folds"], a5)
    info = {"n": len(y), "folds": chosen, "seconds": time.time() - t0, "threads": _threads()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"decision": "D164: the A5-style text classifier's per-answer values (cache)",
                                "inputs_sha256": h, "info": info,
                                "conf": [[it["key"], it["sample"], float(v)] for it, v in zip(index, p)]}))
    print(json.dumps(info, default=str))
    return {(tuple(it["key"]), it["sample"]): float(v) for it, v in zip(index, p)}


# ------------------------------------------------------------------------------------------- the audit
def _excludes_zero(x: dict) -> int:
    lo, hi = x["ci95"]
    return 1 if lo > 0 else (-1 if hi < 0 else 0)


SENSITIVE = (("differences", "yes", "7b - native 1.5b (unpaired)"),
             ("differences", "probe", "7b - native 1.5b (unpaired)"),
             ("controls", "7b vs native 1.5b", "(probe - probe_random): 7b - native 1.5b (unpaired)"),
             ("controls", "7b", "probe - probe_random (paired)"),
             ("controls", "1.5b", "probe - probe_random (paired)"))


def sensitivity(base: dict, other: dict) -> dict:
    """D164's reading: for D162's conclusions, whether the sign or the CI's exclusion of 0 changes."""
    out = {}
    for group, r, name in SENSITIVE:
        a, b = base[group][r][name], other[group][r][name]
        out[f"{r}: {name}"] = {"d162": a["diff"], "here": b["diff"], "d162_ci_side": _excludes_zero(a),
                               "here_ci_side": _excludes_zero(b),
                               "changed": bool(np.sign(a["diff"]) != np.sign(b["diff"])
                                               or _excludes_zero(a) != _excludes_zero(b))}
    return out


def analyze(cfg: dict | None = None) -> dict:
    cfg = cfg or F._config()
    sc, ac = cfg["scale"], cfg["audit"]
    t0 = time.time()
    base, drop_samples = {}, {}
    for m in ac["models"]:
        base[m], drop_samples[m] = kaggle_data(m, cfg)
    n = len({k[2] for k in base["7b"]["groups"]})
    base[SFT], drop_samples[SFT] = sft_data(n, cfg)
    d162 = json.loads((REPO_ROOT / cfg["output_dir"] / "scale" / "results.json").read_text())
    res: dict = {"decision": "D164: a zero-GPU confound audit of D162 (exploratory, not the paper)",
                 "threads": _threads(), "n_candidates": n, "variants": {}}
    res["variants"]["d162"] = compare(base, sc)
    gap = d162_gap(res["variants"]["d162"], d162)
    res["d162_reproduction_gap"] = gap
    if gap > sc["reproduce_atol"]:
        raise SystemExit(f"the unchanged recompute differs from D162 by {gap}: stopped before any variant")
    res["counts"] = {m: {"kept": len(d["keys"]), "kept_with_truncated": sum(k in d["truncated"] for k in d["keys"])}
                     for m, d in base.items()}
    res["variants"]["1a_truncated_samples_dropped"] = compare(drop_samples, sc)
    res["variants"]["1a_truncated_samples_dropped"]["gap_to_d162"] = d162_gap(
        res["variants"]["1a_truncated_samples_dropped"], d162)
    res["variants"]["1b_truncated_problems_dropped"] = compare(
        restrict(base, lambda m, k: k not in base[m]["truncated"]), sc)
    common = set(base["7b"]["keys"]) & set(base["1.5b"]["keys"])
    order = [k for k in base["7b"]["keys"] if k in common]
    both = {m: d | {"keys": [k for k in order if k in set(d["keys"])]} for m, d in base.items()}
    res["variants"]["2_common"] = compare(both, sc)
    res["common_by_pass_count"] = by_pass_count({m: both[m] for m in ("7b", "1.5b")}, sc)
    res["variants"]["3_length_matched"] = compare(base, sc, ac["length_match_rel"])
    res["length_match_rel"] = ac["length_match_rel"]
    res["sensitivity"] = {v: sensitivity(res["variants"]["d162"], res["variants"][v])
                          for v in ("1a_truncated_samples_dropped", "1b_truncated_problems_dropped", "2_common",
                                    "3_length_matched")}
    res["text"] = {m: json.loads((audit_dir(cfg) / f"text_{m}.json").read_text())["info"] for m in ac["models"]}
    res["seconds"] = time.time() - t0
    out = audit_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "summary.md").write_text(render(res))
    print(render(res))
    return res


# ------------------------------------------------------------------------------------------- the report
def _ci(x: dict | None, key: str = "pairwise_accuracy", signed: bool = False) -> str:
    if not x:
        return "–"
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(x[key])} [{f.format(x['ci95'][0])}, {f.format(x['ci95'][1])}]"


def render_variant(name: str, v: dict) -> list[str]:
    ms = v["models"]
    head = (f"| readout | 7B ({ms['7b']['n_kept']}) | native 1.5B ({ms['1.5b']['n_kept']}) | 7B − native, unpaired "
            f"| paired ({v['n_common']} common) | SFT 1.5B ({ms[SFT]['n_kept']}) |")
    L = [f"### {name}", "", head, "|---|---|---|---|---|---|"]
    for r in READOUTS:
        d = v["differences"].get(r, {})
        L.append(f"| {LABELS[r]} | {_ci(ms['7b']['readouts'].get(r))} | {_ci(ms['1.5b']['readouts'].get(r))} "
                 f"| {_ci(d.get('7b - native 1.5b (unpaired)'), 'diff', True)} "
                 f"| {_ci(d.get('7b - native 1.5b (paired, common kept problems)'), 'diff', True)} "
                 f"| {_ci(ms[SFT]['readouts'].get(r))} |")
    L += ["", "| control (paired within a model) | 7B | native 1.5B |", "|---|---|---|"]
    for a, b in CONTROLS:
        k = f"{a} - {b} (paired)"
        L.append(f"| {k} | {_ci(v['controls']['7b'].get(k), 'diff', True)} "
                 f"| {_ci(v['controls']['1.5b'].get(k), 'diff', True)} |")
    for k, x in v["controls"].get("7b vs native 1.5b", {}).items():
        L.append(f"| {k} | {_ci(x, 'diff', True)} | |")
    L += ["", "pairs: " + ", ".join(f"{m} {x['n_pairs']}" for m, x in ms.items())]
    return L + [""]


def render(res: dict) -> str:
    V = res["variants"]
    L = ["# D164: a zero-GPU confound audit of D162 (exploratory, not the paper)", "",
         f"The unchanged recompute reproduces D162's results.json: largest gap {res['d162_reproduction_gap']:.1e}. "
         f"1(a)'s largest gap to D162: {V['1a_truncated_samples_dropped']['gap_to_d162']:.1e}.", "",
         "Kept problems with ≥ 1 truncated sample: " + ", ".join(
             f"{m} {c['kept_with_truncated']} of {c['kept']}" for m, c in res["counts"].items()), ""]
    for key, title in (("d162", "D162, recomputed (with the text classifier added)"),
                       ("1a_truncated_samples_dropped", "1(a) truncated samples dropped"),
                       ("1b_truncated_problems_dropped", "1(b) problems with any truncated sample dropped"),
                       ("2_common", "2. the problems kept by both Kaggle models"),
                       ("3_length_matched", f"3. length-matched pairs (|Δn| ≤ {res['length_match_rel']:.0%} "
                                            "of the longer)")):
        L += render_variant(title, V[key])
    P = res["common_by_pass_count"]
    cs = P["pass_counts"]
    L += ["### 2. by pass count (correct of 8; each model's own count)", "",
          "| readout | model | " + " | ".join(f"c = {c}" for c in cs) + " |", "|---|---|" + "---|" * len(cs)]
    for m in ("7b", "1.5b"):
        L.append(f"| n problems | {m} | " + " | ".join(str(P["n"][m][str(c)]) for c in cs) + " |")
    for r, st in P["strata"].items():
        for m in ("7b", "1.5b"):
            L.append(f"| {LABELS[r]} | {m} | " + " | ".join(
                f"{st[m][str(c)]['pairwise_accuracy']:.3f}" if str(c) in st[m] else "–" for c in cs) + " |")
    L += ["", f"weights (pooled): {', '.join(f'{w:.3f}' for w in P['weights'])}", "",
          "| readout | 7B, standardized | native 1.5B, standardized | 7B − native [95% CI] |", "|---|---|---|---|"]
    for r, s in P["standardized"].items():
        L.append(f"| {LABELS[r]} | {s['7b']:.3f} | {s['1.5b']:.3f} | {_ci(s, 'diff', True)} |")
    L += ["", "### Sensitivity of D162's conclusions (sign or CI side changed)", "",
          "| variant | quantity | D162 | here | changed |", "|---|---|---|---|---|"]
    for v, s in res["sensitivity"].items():
        for q, x in s.items():
            L.append(f"| {v} | {q} | {x['d162']:+.3f} | {x['here']:+.3f} | {'**yes**' if x['changed'] else 'no'} |")
    L += ["", "### The text classifier's folds (kind, hyperparameters, inner-CV AUROC)", ""]
    for m, info in res["text"].items():
        L.append(f"- {m} ({info['n']} answers, {info['seconds'] / 60:.0f} min): " + "; ".join(
            f"{f['kind']} {f['hp']} {f['inner_cv_auroc']:.3f}" for f in info["folds"]))
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["text", "analyze"])
    ap.add_argument("--model", choices=["7b", "1.5b"], help="text: the model whose answers are read")
    args = ap.parse_args()
    cfg = F._config()
    if args.action == "text":
        text_confidence(args.model, cfg)
    else:
        analyze(cfg)


if __name__ == "__main__":
    main()

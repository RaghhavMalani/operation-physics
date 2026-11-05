"""D173 (EXPLORATORY, follow-up project, paper #2): two controls for R6-5 at 1.5B, run locally as a recorded exception
to D171. Plan and readings: docs/exploratory/DECISIONS_FOLLOWUP.md D173, recorded before this code existed.

    uv run python -m physcal.eval.followup_r6_controls items      # CPU: freeze swap_same_1.5b (its own manifest)
    uv run python -m physcal.eval.followup_r6_controls stages     # the GPU stages, in the queue's order
    uv run python -m physcal.eval.followup_r6_controls analyze    # CPU, after the stages (4 jobs, the GPU idle)

The GPU stages are `followup_r6 states` on these sets (scripts/queue_r6_controls.sh). Control 1 holds the reasoning
fixed: a <- d is d's completion with a's box, so e(a <- d) = p(a <- d) - p_orig(d) is the effect of replacing the box.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from physcal.config import load_config
from physcal.eval import followup_audit as AU
from physcal.eval import followup_r6 as R6
from physcal.eval import followup_scale as F
from physcal.eval import followup_scale_analysis as A

MODEL = "1.5b"
SET = f"swap_same_{MODEL}"
KINDS = ("correct", "same_value", "diff_value")
GPU_STAGES = (("states", SET, MODEL), ("states", f"answer_only_{MODEL}", f"random_{MODEL}"))


def same_value(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b))


def transplant(a: dict, d: dict, kind: str) -> dict:
    """a's box after d's reasoning, built as followup_r6.swap_set builds it."""
    content, _ = F.last_boxed(a["completion"])
    comp = d["completion"][:d["completion"].rfind("\\boxed{")] + "\\boxed{" + content + "}"
    p1, p2 = R6.native_positions(comp)
    return a | {"id": f"{a['id']}|same|{kind}", "completion": comp, "p1": p1, "p2": p2, "orig_id": a["id"],
                "donor_sample": d["sample"], "donor_id": d["id"], "kind": kind}


def same_label_set(items: list[dict], value: dict, tol: float) -> list[dict]:
    """D173's texts: for each kept answer, in the probe's order, one text per kind it has a donor for. `value`: item id
    -> the parsed box value."""
    by = R6.by_problem(items)
    out = []
    for a in items:
        others = [d for d in by[tuple(a["key"])] if d["correct"] == a["correct"] and d["id"] != a["id"]]
        if a["correct"]:
            found = {"correct": others}
        else:
            same = [d for d in others if same_value(value[a["id"]], value[d["id"]], tol)]
            found = {"same_value": same, "diff_value": [d for d in others if d not in same]}
        out += [transplant(a, min(ds, key=lambda d: d["sample"]), k) for k, ds in found.items() if ds]
    return out


def _kept(cfg: dict) -> tuple[list[dict], list[dict]]:
    rows = R6.kaggle_rows(MODEL, cfg)
    return rows, R6.probe_order_items(MODEL, rows, cfg)


def build_items(cfg: dict | None = None) -> dict:
    """Freeze the set in manifest_d173.json; D167's manifest is not touched. Refuses a different freeze."""
    cfg = cfg or F._config()
    rows, items = _kept(cfg)
    by = {(tuple(r["key"]), r["sample"]): r for r in rows}
    value = {it["id"]: by[R6._key_sample(it)]["value"] for it in items}
    its = same_label_set(items, value, cfg["scale"]["tolerance"])
    blob = "".join(json.dumps(it, sort_keys=True) + "\n" for it in its)
    manifest = {"decision": "D173", "sets": {SET: {"n": len(its), "sha256": R6._sha(blob)}},
                "kinds": {k: sum(it["kind"] == k for it in its) for k in KINDS}}
    d = R6.r6_dir(cfg) / "items"
    path = d / R6.EXTRA_MANIFEST
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise SystemExit(f"{path}: a different freeze exists; D173's set is not rebuilt")
    (d / f"{SET}.jsonl").write_text(blob)
    path.write_text(json.dumps(manifest, indent=1))
    return manifest


# ------------------------------------------------------------------------------------------- the quantities
def box_effects(texts: list[dict], p_text: dict, p_orig: dict) -> dict:
    """e(a <- d) per text id. `p_text`: text id -> p(a <- d); `p_orig`: item id -> p of its own text."""
    return {t["id"]: p_text[t["id"]] - p_orig[t["donor_id"]] for t in texts}


def mismatch_per_problem(texts: list[dict], e: dict) -> tuple[list, np.ndarray]:
    """(e)'s M_a = e(a <- d_diff) - e(a <- d_same) over incorrect answers with both kinds, averaged within each problem."""
    by: dict = {}
    for t in texts:
        if t["kind"] != "correct":
            by.setdefault(t["orig_id"], {})[t["kind"]] = t
    per: dict = {}
    for ts in by.values():
        if len(ts) == 2:
            k = tuple(ts["diff_value"]["key"])
            per.setdefault(k, []).append(e[ts["diff_value"]["id"]] - e[ts["same_value"]["id"]])
    keys = list(per)
    return keys, np.array([float(np.mean(per[k])) for k in keys])


def fixed_reasoning_per_problem(opposite: list[dict], texts: list[dict], p_opp: dict, p_same: dict) -> tuple[list, np.ndarray]:
    """(f): per problem, the correct boxes after its lowest-sample incorrect answer d0 (R6-5's swap texts) against the
    different-valued wrong boxes after d0, scored as X2 does (ties 0.5)."""
    right: dict = {}
    for t in opposite:
        if t["correct"]:
            right.setdefault((tuple(t["key"]), t["donor_sample"]), []).append(p_opp[t["id"]])
    wrong: dict = {}
    for t in texts:
        if t["kind"] == "diff_value":
            wrong.setdefault((tuple(t["key"]), t["donor_sample"]), []).append(p_same[t["id"]])
    keys = [k for k in right if k in wrong]
    return [k[0] for k in keys], np.array([float(np.mean([R6._score(a, b) for a in right[k] for b in wrong[k]]))
                                           for k in keys])


def text_pairs(texts: list[dict], kinds: tuple[str, str]) -> set:
    """The orig ids with a text of kinds[0] (correct answers) or kinds[1] (incorrect ones)."""
    return {t["orig_id"] for t in texts if t["kind"] in kinds}


def mean_box_effects(texts: list[dict], e: dict) -> dict:
    return {k: {"n": sum(t["kind"] == k for t in texts), "mean": float(np.mean([e[t["id"]] for t in texts if t["kind"] == k]))}
            for k in KINDS}


# ------------------------------------------------------------------------------------------- the analysis
def _ordered(name: str, reader: str, cfg: dict) -> tuple[list[dict], list[np.ndarray]]:
    its = R6.load_set(name, cfg)
    layers, ids, _ = R6.load_states(name, reader, "p2", cfg)
    if ids != [t["id"] for t in its]:
        raise SystemExit(f"{name}/{reader}: states not in the set's order")
    return its, layers


def score_probe(items, index, probe_cache, opp, opp_layers, texts, tx_layers, cfg) -> tuple[dict, dict, dict]:
    """R6-5 (d)'s scoring: each text by the fold model holding its problem out. Gate: the refit equals the cache."""
    orig_layers, _ = F.load_hidden(MODEL, cfg)
    row = {it["id"]: i for i, it in enumerate(items)}
    tx_rows = np.array([row[t["orig_id"]] for t in texts])
    po, pp = np.full(len(items), np.nan), np.full(len(items), np.nan)
    pt = np.full(len(texts), np.nan)
    for te, layer, f in R6.refit_fold_probes(orig_layers, index, probe_cache["info"]["folds"], cfg):
        po[te], pp[te] = f(orig_layers[layer][te]), f(opp_layers[layer][te])
        sel = np.flatnonzero(np.isin(tx_rows, te))
        pt[sel] = f(tx_layers[layer][sel])
    if np.isnan(pt).any() or np.isnan(pp).any():
        raise SystemExit("a text has no fold model")
    cached = {(tuple(k), s): v for k, s, v in probe_cache["conf"]}
    gap = float(np.abs(po - np.array([cached[R6._key_sample(it)] for it in items])).max())
    if gap > cfg["r6"]["swap_refit_atol"]:
        raise SystemExit(f"refit fold probes differ from the cache by {gap}")
    return ({it["id"]: float(v) for it, v in zip(items, po)}, {t["id"]: float(v) for t, v in zip(opp, pp)},
            {t["id"]: float(v) for t, v in zip(texts, pt)} | {"_gap": gap})


def score_text(items, opp, texts, cfg) -> tuple[dict, dict]:
    """D164's text classifier (its fold models, refit with its recorded choices) on the originals and the D173 texts."""
    from physcal.eval import nested_baselines as nb

    sc, tc = cfg["scale"], cfg["audit"]["text"]
    a5 = load_config(cfg["audit"]["text"]["a5_config"])
    y = np.array([it["correct"] for it in items], dtype=int)
    groups = [R6._kid(it["key"]) for it in items]
    recs = [AU.text_fields(it["text"], it["completion"]) for it in items]
    tx_recs = [AU.text_fields(t["text"], t["completion"]) for t in texts]
    row = {it["id"]: i for i, it in enumerate(items)}
    tx_rows = np.array([row[t["orig_id"]] for t in texts])
    tf = json.loads((AU.audit_dir(cfg) / f"text_{MODEL}.json").read_text())["info"]["folds"]
    po, pt = np.full(len(items), np.nan), np.full(len(texts), np.nan)
    for (tr, te), fsel in zip(AU._folds(groups, sc["probe"]["folds"]), tf):
        cand = nb.Candidate(tc["feature_set"], fsel["kind"], "p", fsel["hp"], a5).fit([recs[i] for i in tr], y[tr])
        po[te] = cand.raw([recs[i] for i in te])
        sel = np.flatnonzero(np.isin(tx_rows, te))
        pt[sel] = cand.raw([tx_recs[i] for i in sel])
    return {it["id"]: float(v) for it, v in zip(items, po)}, {t["id"]: float(v) for t, v in zip(texts, pt)}


def control_1(items, index, probe_cache, cfg) -> dict:
    sc = cfg["scale"]
    opp, opp_layers = _ordered(f"swap_{MODEL}", MODEL, cfg)
    if [t["orig_id"] for t in opp] != [it["id"] for it in items]:
        raise SystemExit(f"swap_{MODEL}: not in the probe's order")
    texts, tx_layers = _ordered(SET, MODEL, cfg)
    p_orig, p_opp, p_same = score_probe(items, index, probe_cache, opp, opp_layers, texts, tx_layers, cfg)
    gap = p_same.pop("_gap")
    y = np.array([it["correct"] for it in items], dtype=bool)
    delta = np.array([p_opp[t["id"]] - p_orig[t["orig_id"]] for t in opp])
    saved = json.loads((R6.r6_dir(cfg) / "results.json").read_text())[f"R6-5 {MODEL}"]["swap_movement"]
    move = {"median_abs": float(np.median(np.abs(delta))), "mean_signed_correct": float(delta[y].mean()),
            "mean_signed_incorrect": float(delta[~y].mean())}
    if move != saved:
        raise SystemExit(f"R6-5's swap_movement is not reproduced: {move} against {saved}")
    e = box_effects(texts, p_same, p_orig)
    keys, m = mismatch_per_problem(texts, e)
    res: dict = {"refit_gap": gap, "swap_movement_reproduced": True, "n_texts": len(texts),
                 "box_effect_by_kind": mean_box_effects(texts, e),
                 "(e) mismatch cost M": A.paired(m, np.zeros_like(m), sc["bootstrap"], sc["seed"]) | {"n_problems": len(keys)}}
    t_orig, t_same = score_text(items, opp, texts, cfg)
    et = box_effects(texts, t_same, t_orig)
    _, mt = mismatch_per_problem(texts, et)
    res["text classifier"] = {"box_effect_by_kind": mean_box_effects(texts, et),
                              "mismatch cost M": A.paired(mt, np.zeros_like(mt), sc["bootstrap"], sc["seed"])}
    fk, f = fixed_reasoning_per_problem(opp, texts, p_opp, p_same)
    res["(f) box with the reasoning fixed"] = R6._summ(f, sc)
    # (g) descriptive: correct against diff_value texts, with the original and R6-5's swap on the same pairs
    groups = R6.by_problem(items)
    has = text_pairs(texts, ("correct", "diff_value"))
    ok = lambda a, b: a["id"] in has and b["id"] in has  # noqa: E731
    ks = [k for k, n in zip(groups, R6.n_pairs(groups, list(groups), ok)) if n > 0]
    conf = {"original": p_orig, "opposite": {t["orig_id"]: p_opp[t["id"]] for t in opp},
            "same_label": {t["orig_id"]: p_same[t["id"]] for t in texts if t["kind"] in ("correct", "diff_value")}}
    pa = {c: R6.per_problem(groups, ks, v, ok) for c, v in conf.items()}
    d_id = {R6._key_sample(it): it["id"] for it in items}
    split = {}
    for name, sel in (("correct boxes", y), ("incorrect boxes", ~y)):
        ts = [t for t, s in zip(opp, sel) if s]
        donor = [p_orig[d_id[(tuple(t["key"]), t["donor_sample"])]] for t in ts]
        split[name] = {"n": len(ts), "box_effect": float(np.mean([p_opp[t["id"]] - v for t, v in zip(ts, donor)])),
                       "reasoning_effect": float(np.mean([v - p_orig[t["orig_id"]] for t, v in zip(ts, donor)]))}
    res["(g) descriptive"] = {"pairs": {c: R6._summ(v, sc) for c, v in pa.items()},
                              "original - same_label": A.paired(pa["original"], pa["same_label"], sc["bootstrap"], sc["seed"]),
                              "R6-5 change split": split}
    side = R6._side(res["(e) mismatch cost M"])
    lo, hi = res["(f) box with the reasoning fixed"]["ci95"]
    res["reading"] = {"(e)": {-1: "a mismatch alone lowers the probe", 0: "no mismatch cost is shown",
                              1: "no reading (M above 0)"}[side],
                      "(f)": ("with the reasoning fixed, the probe prefers the correct box" if lo > 0.5 else
                              "it prefers the wrong box" if hi < 0.5 else "neither")}
    return res


def control_2(items, index, probe_cache, cfg) -> dict:
    sc, rel = cfg["scale"], cfg["audit"]["length_match_rel"]
    ao, trained_layers = _ordered(f"answer_only_{MODEL}", MODEL, cfg)
    _, random_layers = _ordered(f"answer_only_{MODEL}", f"random_{MODEL}", cfg)
    if [t["orig_id"] for t in ao] != [it["id"] for it in items]:
        raise SystemExit("the answer-only set is not in the probe's order")
    name_t, name_r = f"answer_only_probe_{MODEL}", f"answer_only_probe_random_{MODEL}"
    c_t, _ = R6.cached(cfg, name_t, R6._probe_inputs(name_t, trained_layers, ao, cfg),
                       lambda: (_ for _ in ()).throw(SystemExit(f"{name_t}: R6-5's cached fit does not match its states")))
    c_r, info = R6.cached(cfg, name_r, R6._probe_inputs(name_r, random_layers, ao, cfg),
                          lambda: R6.crossfit_items(random_layers, ao, cfg))
    corr = {R6._key_sample(it): it["correct"] for it in items}
    random_full, _ = A.probe_confidence(MODEL, "hidden_random", corr, cfg)
    conf = {"trained": {t["orig_id"]: c_t[t["id"]] for t in ao}, "random": {t["orig_id"]: c_r[t["id"]] for t in ao},
            "random_full": {it["id"]: random_full[R6._key_sample(it)] for it in items}}
    groups = R6.by_problem(items)
    keys = list(groups)
    lm = lambda a, b: AU.length_matched(a["n_tokens"], b["n_tokens"], rel)  # noqa: E731
    out = {}
    for variant, ok in (("all pairs", None), ("length-matched", lm)):
        ks = [k for k, n in zip(keys, R6.n_pairs(groups, keys, ok)) if n > 0]
        pa = {c: R6.per_problem(groups, ks, v, ok) for c, v in conf.items()}
        d = lambda a, b: A.paired(pa[a], pa[b], sc["bootstrap"], sc["seed"])  # noqa: E731
        out[variant] = {"n_problems": len(ks), "readouts": {c: R6._summ(v, sc) for c, v in pa.items()},
                        "(h) trained - random": d("trained", "random"),
                        "random_full - random": d("random_full", "random")}
    a = out["all pairs"]
    side = R6._side(a["(h) trained - random"])
    reading = ("the trained weights add to the box alone" if side > 0 else
               "the box's signal is surface" if side == 0 and a["readouts"]["random"]["ci95"][0] > 0.5 else "no reading")
    return {"by_variant": out, "fit": info, "reading": {"(h)": reading}}


def analyze(cfg: dict | None = None) -> dict:
    cfg = cfg or F._config()
    rows, items = _kept(cfg)
    corr = {R6._key_sample(it): it["correct"] for it in items}
    index, _, cache = A.probe_inputs(MODEL, "hidden", corr, cfg)
    probe_cache = json.loads(cache.read_text())
    res: dict = {"decision": "D173 (exploratory, paper #2; an exception to D171)", "threads": AU._threads()}
    path = R6.r6_dir(cfg) / "controls_d173.json"

    def save():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1, default=float))
        tmp.rename(path)

    res["control 1"] = control_1(items, index, probe_cache, cfg)
    save()
    res["control 2"] = control_2(items, index, probe_cache, cfg)
    save()
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["items", "stages", "analyze"])
    args = ap.parse_args()
    cfg = F._config()
    if args.action == "items":
        print(json.dumps(build_items(cfg), indent=1))
    elif args.action == "stages":
        print("\n".join(" ".join(s) for s in GPU_STAGES))
    else:
        analyze(cfg)


if __name__ == "__main__":
    main()

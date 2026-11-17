#!/usr/bin/env python
"""D184: the post-hoc diagnosis of R6-6a's E - A check (D175: "the setup leaks: stop and diagnose"), and descriptive
comparisons with the references. EXPLORATORY, post-hoc; nothing here is a pre-registered reading. CPU only.

    uv run python scripts/r6_6a_diagnosis.py        # -> results/followup/r6_6a/diagnosis_d184.json

(a) Leakage: do any training items share a problem (key or text) with the validation or test problems?
(b) The length channel: D175's pairwise accuracy, recomputed on all pairs, on length-matched pairs (|ln ratio| <= 0.1),
    and on length-discordant pairs (the correct answer is the longer one), with paired contrasts.
(c) Descriptive: the trained arms against self-consistency (k = 8), the probe and the rich-text reader.
Per-problem values as D175 (ties 0.5; an arm with seeds is the mean of its seeds); bootstrap over problems, B = 2,000,
seed 0 (smaller than D175's 10,000: these are diagnostics).
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict

import numpy as np

from physcal.config import REPO_ROOT
from physcal.eval import r6_6a as M

RUN = REPO_ROOT / "results" / "followup" / "r6_6a" / "run"
OUT = REPO_ROOT / "results" / "followup" / "r6_6a" / "diagnosis_d184.json"
ARMS = {"A": ["A"], "B": ["B_0", "B_1", "B_2"], "C": ["C_0", "C_1", "C_2"], "D": ["D_0", "D_1", "D_2"], "E": ["E_0"],
        "probe": ["probe"], "rich": ["rich"], "self_consistency": ["self_consistency"], "length": ["length"]}
CONTRASTS = [("E", "A"), ("C", "E"), ("C", "D"), ("D", "E"), ("B", "E"), ("B", "self_consistency"),
             ("C", "self_consistency"), ("C", "probe"), ("D", "rich"), ("B", "probe")]
SUBSETS = {"all": lambda a, b: True,
           "length_matched": lambda a, b: abs(np.log(a["n_tokens"] / b["n_tokens"])) <= 0.1,
           "length_discordant": lambda a, b: a["n_tokens"] > b["n_tokens"]}
B, SEED = 2000, 0


def leakage(rc: dict) -> dict:
    rows = [json.loads(line) for line in gzip.open(RUN / "completions.jsonl.gz")]
    split = M.split_of(rc)
    its = M.items_of(rows, rc, split)
    train = M.in_split(its, "train")[: rc["train"]["items"]]
    test = M.in_split(its, "validation", "id_test", "held_out")
    tk, ek = {tuple(i["key"]) for i in train}, {tuple(i["key"]) for i in test}
    text = {r["index"]: r["problem"]["text"] for r in rows}
    tt = {text[r["index"]] for r in rows if split[r["index"]] == "train"}
    et = {text[r["index"]] for r in rows if split[r["index"]] in ("validation", "id_test", "held_out")}
    return {"training_items": len(train), "training_problems": len(tk), "evaluation_problems": len(ek),
            "shared_keys": len(tk & ek), "shared_texts": len(tt & et)}


def per_problem(rows: list[dict], keep) -> dict[str, dict]:
    by = defaultdict(list)
    for r in rows:
        by[tuple(r["key"])].append(r)
    out: dict = {a: {} for a in ARMS}
    for k, g in sorted(by.items()):
        pairs = [(c, w) for c in g if c["correct"] for w in g if not w["correct"] and keep(c, w)]
        if not pairs:
            continue
        for a, cols in ARMS.items():
            out[a][k] = float(np.mean([np.mean([0.5 if c["conf"][col] == w["conf"][col] else float(c["conf"][col] > w["conf"][col])
                                                for c, w in pairs]) for col in cols]))
        out.setdefault("_pairs", {})[k] = len(pairs)
    return out


def block(rows: list[dict], keep, rng) -> dict:
    pp = per_problem(rows, keep)
    keys = sorted(pp["A"])
    X = {a: np.array([pp[a][k] for k in keys]) for a in ARMS}
    boot = rng.integers(0, len(keys), (B, len(keys)))

    def est(v):
        bs = v[boot].mean(1)
        return {"value": float(v.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    return {"n_problems": len(keys), "n_pairs": int(sum(pp["_pairs"].values())),
            "arms": {a: est(X[a]) for a in ARMS},
            "contrasts": {f"{a} - {b}": est(X[a] - X[b]) for a, b in CONTRASTS}}


def main() -> None:
    rc = M.mode_config("full")
    S = [json.loads(line) for line in gzip.open(RUN / "scores.jsonl.gz")]
    rng = np.random.default_rng(SEED)
    res = {"decision": "D184 (post-hoc, exploratory)", "bootstrap": B, "seed": SEED, "leakage": leakage(rc)}
    for sp in ("id_test", "held_out"):
        rows = [r for r in S if r["split"] == sp]
        res[sp] = {name: block(rows, keep, rng) for name, keep in SUBSETS.items()}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["leakage"]))
    for sp in ("id_test", "held_out"):
        for name in SUBSETS:
            b = res[sp][name]
            c = b["contrasts"]
            print(f"{sp:9} {name:17} {b['n_problems']:4} problems {b['n_pairs']:5} pairs | " + "; ".join(
                f"{k} {v['value']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" for k, v in c.items()
                if k in ("E - A", "C - E", "C - D", "D - E")))


if __name__ == "__main__":
    main()

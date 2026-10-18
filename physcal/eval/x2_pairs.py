"""X2 (D74): within-problem correctness pairs, the decisive test of answer-specific confidence.

    uv run python -m physcal.eval.x2_pairs build            # CPU: freeze the candidate pool (done 2026-10-18)
    uv run python -m physcal.eval.x2_pairs sample [--limit N]  # GPU (a later night): k = 8 SFT samples per candidate
    uv run python -m physcal.eval.x2_pairs select <tape>    # CPU: the mixed problems, from the k = 8 sampled tape

A candidate problem is kept iff the k sampled answers of the corrected SFT model include at least one correct
(|relative error| <= 1%) and at least one incorrect parsed answer (parse failures and unit mismatches are not
answers). Within a kept problem every (correct, incorrect) pair is formed, so difficulty is held fixed. A method's
pairwise accuracy on a problem is the share of its pairs where the correct answer gets more confidence (ties 0.5);
the headline is the mean over problems (each problem weighs the same), with a bootstrap over problems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config


def _config() -> dict:
    return load_config("x2")


def build() -> dict:
    """The frozen candidate pool: problem i is cell i mod 12 (families x depths), seed offset + i."""
    from physcal.env.generator import generate_problem, generator_version

    cfg = _config()
    c, env = cfg["candidates"], load_config(cfg["env_config"])
    cells = [(f, d) for f in c["families"] for d in c["depths"]]
    probs = [generate_problem(*cells[i % len(cells)], c["seed_offset"] + i, split="train", cfg=env)
             for i in range(c["n"])]
    path = REPO_ROOT / c["file"]
    version = generator_version(env)
    if path.parent.name != version:
        raise ValueError(f"{path} is not under the generator version's dir {version}")
    blob = "".join(json.dumps(p.to_dict(), sort_keys=True) + "\n" for p in probs)
    path.write_text(blob)
    manifest = {"decision": "D74 (X2)", "generator_version": version, "n_problems": len(probs),
                "seeds": [c["seed_offset"], c["seed_offset"] + c["n"] - 1], "split": "train",
                "cells": {f"{f}/{d}": sum(1 for p in probs if (p.family, p.depth) == (f, d)) for f, d in cells},
                "sha256": hashlib.sha256(blob.encode()).hexdigest()}
    path.with_name(path.stem + "_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def sample_config() -> dict:
    """The k = 8 sampled eval of the corrected SFT model on the frozen pool (D74): D67's SFT control (merged SFT, 4-bit,
    no adapter; loader settings from the uniform-credit run, as posthoc.refit_4bit) at the training decoding."""
    from physcal.train.compare import eval_config

    cfg, ph = _config(), load_config("posthoc")
    s = cfg["sampling"]
    ev = eval_config("sft", ph["label"], s["max_new_tokens"], REPO_ROOT / ph["refit_4bit"]["settings_run"],
                     REPO_ROOT / "results" / "sft" / ph["label"] / "merged", samples=s["k"], loader="unsloth_4bit")
    ev["generation"].update(temperature=s["temperature"], top_p=s["top_p"], top_k=s["top_k"], seed=s["seed"],
                            batch_size=s["batch_size"], max_new_tokens=s["max_new_tokens"])
    ev["env_config"] = cfg["env_config"]
    ev["splits"] = {"x2": {"file": cfg["candidates"]["file"], "verify_regenerate": -1}}
    ev["output_dir"] = str(REPO_ROOT / cfg["output_dir"] / "sft_k8")
    ev["run_id"] = "x2-sft-4bit-k8"
    return ev


def answer(b: dict, tolerance: float, floor: float) -> dict | None:
    """One sampled completion's answer from its verifier breakdown: None for a parse failure or unit mismatch."""
    if b["reason"] != "ok" or b.get("mu_si") is None:
        return None
    rel = b["sigma_si"] / abs(b["mu_si"]) if b.get("sigma_si") is not None and b["mu_si"] != 0 else None
    return {"correct": b["abs_rel_error"] <= tolerance, "rel_sigma": None if rel is None else max(rel, floor),
            "mu_si": b["mu_si"]}


def group(records: list[dict], cfg: dict | None = None) -> dict:
    """Per problem key (family, depth, seed): its parsed answers (sample index kept)."""
    cfg = cfg or _config()
    tol, floor = cfg["selection"]["tolerance"], cfg["analysis"]["sigma_floor_rel"]
    out: dict = {}
    for r in records:
        p = r["problem"]
        key = (p["family"], p["depth"], p["seed"])
        out.setdefault(key, [])
        a = answer(r["breakdown"], tol, floor)
        if a is not None:
            out[key].append({**a, "sample": r.get("sample", 0)})
    return out


def select(groups: dict, cfg: dict | None = None) -> list:
    """Kept problem keys, in the pool's order: >= min_correct correct and >= min_incorrect incorrect answers."""
    s = (cfg or _config())["selection"]
    return [k for k, ans in groups.items()
            if sum(a["correct"] for a in ans) >= s["min_correct"] and sum(not a["correct"] for a in ans) >= s["min_incorrect"]]


def pairwise_accuracy(groups: dict, keys: list, conf) -> np.ndarray:
    """Per kept problem: the share of (correct, incorrect) pairs where conf(correct) > conf(incorrect), ties 0.5.
    `conf(key, answer)` is the method's confidence (e.g. −ln relative σ, or p); None makes the pair a tie."""
    out = []
    for k in keys:
        c = [conf(k, a) for a in groups[k] if a["correct"]]
        w = [conf(k, a) for a in groups[k] if not a["correct"]]
        s = [0.5 if (x is None or y is None or x == y) else float(x > y) for x in c for y in w]
        out.append(float(np.mean(s)))
    return np.array(out)


def summarize(pa: np.ndarray, n_boot: int, seed: int) -> dict:
    """Mean over problems, 95% bootstrap CI over problems, one-sided p for > 0.5."""
    rng = np.random.default_rng(seed)
    boots = np.array([pa[rng.integers(0, len(pa), len(pa))].mean() for _ in range(n_boot)])
    return {"n_problems": len(pa), "pairwise_accuracy": float(pa.mean()),
            "ci95": np.percentile(boots, [2.5, 97.5]).tolist(), "p_one_sided_gt_half": float(np.mean(boots <= 0.5))}


def stated_sigma_conf(key, a):
    return None if a["rel_sigma"] is None else -float(np.log(a["rel_sigma"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["build", "sample", "select"])
    ap.add_argument("tape", nargs="?")
    ap.add_argument("--limit", type=int, default=None, help="sample: the first N candidates (a smoke test)")
    args = ap.parse_args()
    if args.action == "build":
        print(json.dumps(build(), indent=1))
        return
    if args.action == "sample":
        from physcal.eval.run_eval import run

        run(sample_config(), limit=args.limit)
        return
    from physcal.tape.tape import read_tape

    cfg = _config()
    g = group(list(read_tape(Path(args.tape))), cfg)
    keys = select(g, cfg)
    out = REPO_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "selected.json").write_text(json.dumps({"tape": args.tape, "n_candidates_sampled": len(g),
                                                   "n_selected": len(keys), "keys": keys}, indent=1))
    print(f"{len(keys)} of {len(g)} sampled candidates have mixed outcomes")


if __name__ == "__main__":
    main()

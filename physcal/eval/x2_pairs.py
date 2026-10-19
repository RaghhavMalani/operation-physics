"""X2 (D74): within-problem correctness pairs, the decisive test of answer-specific confidence.

    uv run python -m physcal.eval.x2_pairs build            # CPU: freeze the candidate pool (done 2026-10-18)
    uv run python -m physcal.eval.x2_pairs sample --start S --count N [--batch-size B] [--chunk I]   # GPU (D81)
    uv run python -m physcal.eval.x2_pairs select           # CPU: the mixed problems, from the sampled chunks
    uv run python -m physcal.eval.x2_pairs suffix [--batch-size B]   # GPU: method 2, split credit's σ suffixes
    uv run python -m physcal.eval.x2_analysis analyze       # CPU: the D74 analysis (physcal/eval/x2_analysis.py)

Sampling runs in chunks of the frozen order (D81): chunk i covers candidates [start, start + count) with torch seed
seed + i and is its own tape (`<sampling.output_dir>/chunk_<start>/`). The chunks present must form a prefix of the
pool (checked), so a night that ends early still yields a round-robin-stratified prefix (D74).

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


def sample_config(start: int = 0, batch_size: int | None = None, chunk: int = 0) -> dict:
    """The k = 8 sampled eval of the corrected SFT model on the frozen pool (D74): D67's SFT control (merged SFT, 4-bit,
    no adapter; loader settings from the uniform-credit run, as posthoc.refit_4bit) at the training decoding. D81: one
    chunk of the frozen order from `start`, torch seed + `chunk`, the night's batch size, the allocator cap."""
    from physcal.train.compare import eval_config

    cfg, ph = _config(), load_config("posthoc")
    s = cfg["sampling"]
    ev = eval_config("sft", ph["label"], s["max_new_tokens"], REPO_ROOT / ph["refit_4bit"]["settings_run"],
                     REPO_ROOT / "results" / "sft" / ph["label"] / "merged", samples=s["k"], loader="unsloth_4bit")
    bs = batch_size or s["batch_size"]
    if bs % s["k"]:
        raise ValueError(f"batch size {bs} is not a multiple of k = {s['k']}")
    ev["generation"].update(temperature=s["temperature"], top_p=s["top_p"], top_k=s["top_k"], seed=s["seed"] + chunk,
                            batch_size=bs, max_new_tokens=s["max_new_tokens"])
    ev["model"]["max_reserved_gib"] = s["max_reserved_gib"]
    ev["env_config"] = cfg["env_config"]
    ev["splits"] = {"x2": {"file": cfg["candidates"]["file"], "verify_regenerate": -1, "start": start}}
    ev["output_dir"] = str(chunk_dir(start))
    ev["run_id"] = f"x2-sft-4bit-k8-chunk{chunk}"
    return ev


def chunk_dir(start: int) -> Path:
    return REPO_ROOT / _config()["sampling"]["output_dir"] / f"chunk_{start:05d}"


def sampled_chunks(cfg: dict | None = None) -> list[tuple[int, Path]]:
    """(start, tape) of every finished chunk, in order; raises unless they form a prefix of the pool, each chunk
    complete (k rows per candidate, consecutive candidates)."""
    from physcal.tape.tape import read_meta

    cfg = cfg or _config()
    root = REPO_ROOT / cfg["sampling"]["output_dir"]
    out, expect = [], 0
    for d in sorted(root.glob("chunk_*")):
        tape = d / "rollouts.jsonl"
        if not tape.exists():
            continue
        start = int(d.name.split("_")[1])
        if start != expect:
            raise ValueError(f"X2 chunks are not a prefix of the pool: {d.name} starts at {start}, expected {expect}")
        n = sum(1 for _ in tape.open())
        k = read_meta(tape)["generation"]["num_samples"]
        if n % k:
            raise ValueError(f"{tape}: {n} rows is not a multiple of k = {k}")
        out.append((start, tape))
        expect = start + n // k
    return out


def sampled_records(cfg: dict | None = None) -> tuple[list[dict], int]:
    """Every sampled record of the chunk prefix, in pool order, and the number of candidates sampled. The records'
    seeds must be exactly the pool's first candidates (checked)."""
    from physcal.tape.tape import read_tape

    cfg = cfg or _config()
    recs = [r for _, tape in sampled_chunks(cfg) for r in read_tape(tape)]
    pool = [json.loads(line)["seed"] for line in (REPO_ROOT / cfg["candidates"]["file"]).open()]
    seeds = list(dict.fromkeys(r["problem"]["seed"] for r in recs))
    if sorted(seeds) != sorted(pool[:len(seeds)]):
        raise ValueError("the sampled problems are not the pool's first candidates")
    return recs, len(seeds)


def kept_groups(cfg: dict | None = None) -> tuple[dict, list]:
    """(groups, kept keys) of the sampled prefix; must agree with selected.json when that exists."""
    cfg = cfg or _config()
    recs, _ = sampled_records(cfg)
    g = group(recs, cfg)
    keys = select(g, cfg)
    sel = REPO_ROOT / cfg["output_dir"] / "selected.json"
    if sel.exists() and [list(k) for k in keys] != json.loads(sel.read_text())["keys"]:
        raise ValueError("the kept problems differ from selected.json")
    return g, keys


def kept_records(cfg: dict | None = None) -> tuple[list[dict], str]:
    """Every sampled record (answered or not) of the kept problems, and a description of the source tapes."""
    cfg = cfg or _config()
    recs, n = sampled_records(cfg)
    g = group(recs, cfg)
    kept = set(select(g, cfg))
    keep = [r for r in recs if (r["problem"]["family"], r["problem"]["depth"], r["problem"]["seed"]) in kept]
    return keep, f"{cfg['sampling']['output_dir']}/chunk_* ({n} candidates)"


def suffix(batch_size: int | None = None, backend=None) -> dict:
    """Method 2 (D74): each kept answer truncated after "±"; split credit seed 0 @287 (corrected loader, unmerged)
    writes a greedy σ suffix (E1's eval machinery). Rows carry (key, source_sample)."""
    from transformers import AutoTokenizer

    from physcal.tape.tape import read_meta
    from physcal.train import e1

    cfg = _config()
    sc = cfg["suffix"]
    run = REPO_ROOT / sc["run"]
    e1cfg = e1._config()
    tok = AutoTokenizer.from_pretrained(str(REPO_ROOT / e1cfg["model_id"]))
    recs, source = kept_records(cfg)
    rows, excluded = e1.rows_from_records(recs, tok, e1cfg["prompt"], e1cfg.get("chat_template_kwargs", {}))
    model = read_meta(run / "rollouts.jsonl")["train_config"]["model"]
    out = REPO_ROOT / sc["output_dir"]
    res = e1._evaluate(out, "x2-split-seed0-287-suffix", "free", str(run / sc["adapter"]), model, backend, 0,
                       "unsloth_4bit", rows=rows, source_tapes=[source], batch_size=batch_size or sc["batch_size"])
    (out / "excluded.json").write_text(json.dumps(excluded))
    return {**res, "excluded": excluded}


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
    ap.add_argument("action", choices=["build", "sample", "select", "suffix"])
    ap.add_argument("--start", type=int, default=0, help="sample: first candidate of the chunk")
    ap.add_argument("--count", type=int, default=None, help="sample: candidates in the chunk")
    ap.add_argument("--chunk", type=int, default=0, help="sample: chunk index (torch seed = seed + chunk)")
    ap.add_argument("--batch-size", type=int, default=None, help="sample / suffix: sequences per generate call")
    args = ap.parse_args()
    if args.action == "build":
        print(json.dumps(build(), indent=1))
        return
    if args.action == "sample":
        from physcal.eval.run_eval import run

        run(sample_config(args.start, args.batch_size, args.chunk), limit=args.count)
        return
    if args.action == "suffix":
        print(json.dumps(suffix(args.batch_size), indent=1))
        return
    cfg = _config()
    recs, n = sampled_records(cfg)
    g = group(recs, cfg)
    keys = select(g, cfg)
    out = REPO_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    counts = [sum(a["correct"] for a in g[k]) for k in keys]
    (out / "selected.json").write_text(json.dumps({
        "chunks": [str(t.relative_to(REPO_ROOT)) for _, t in sampled_chunks(cfg)], "n_candidates_sampled": n,
        "n_selected": len(keys), "n_pairs": int(sum(c * (len(g[k]) - c) for c, k in zip(counts, keys))),
        "correct_count_distribution": {str(c): counts.count(c) for c in sorted(set(counts))},
        "keys": [list(k) for k in keys]}, indent=1))
    print(f"{len(keys)} of {n} sampled candidates have mixed outcomes")


if __name__ == "__main__":
    main()

"""D151 item 3 (EXPLORATORY, follow-up project; not the paper): does the epistemic wall move with scale?
Qwen2.5-Math-7B-Instruct (4-bit, inference only) samples k = 8 answers in its native \\boxed{} format on a prefix of
X2's frozen candidate pool; on the problems with right and wrong answers, three readouts are scored by X2's
within-problem pairwise accuracy: self-consistency, X4-style direct P(yes), and a cross-fitted hidden-state probe.

    uv run python -m physcal.eval.followup_scale feasibility --model 7b --cap G        # GPU: memory, batch, s/problem
    uv run python -m physcal.eval.followup_scale sample --model 7b --cap G --stop-at T # GPU: chunks until T (epoch s)
    uv run python -m physcal.eval.followup_scale yes --model 7b --cap G                # GPU: direct P(yes), kept answers
    uv run python -m physcal.eval.followup_scale hidden --model 7b --cap G             # GPU: every layer, kept answers
    uv run python -m physcal.eval.followup_scale analyze                               # CPU

Every stage is resumable: sampling writes one file per chunk of the pool; the readouts append rows / parts and skip
what exists. Outputs: <output_dir>/scale/<model>/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config


def _config() -> dict:
    return load_config("followup")


def out_dir(model: str, cfg: dict | None = None) -> Path:
    cfg = cfg or _config()
    return REPO_ROOT / cfg["output_dir"] / "scale" / model


# ------------------------------------------------------------------------------------------- the pool and prompts
def pool(cfg: dict | None = None) -> list[dict]:
    """X2's frozen candidate pool, in order (configs/x2.yaml candidates.file)."""
    cfg = cfg or _config()
    x2 = load_config(cfg["scale"]["pool_config"])
    return [json.loads(line) for line in (REPO_ROOT / x2["candidates"]["file"]).open()]


def _steps_patterns() -> list[re.Pattern]:
    from physcal.env.render import NO_STEPS_TEMPLATES, STEPS_TEMPLATES

    pats = [re.escape(t).replace(re.escape("{syms}"), r"[^.\n]+") for t in STEPS_TEMPLATES]
    pats += [re.escape(t) for t in set(NO_STEPS_TEMPLATES.values())]
    return [re.compile(r"[ \t]*" + p) for p in pats]


def native_text(text: str) -> str:
    """The problem text without its <steps> sentence (the question sentence and its unit stay)."""
    for p in _steps_patterns():
        text = p.sub("", text)
    if "<steps>" in text:
        raise ValueError(f"a <steps> sentence survived: {text[-300:]!r}")
    return text.rstrip()


def messages(text: str, cfg: dict) -> list[dict]:
    return [{"role": "system", "content": cfg["scale"]["system_prompt"]},
            {"role": "user", "content": native_text(text)}]


def verify_text(cfg: dict) -> str:
    from physcal.prompt import VERIFY_PROMPTS

    sc = cfg["scale"]
    old, new = sc["verify_replace"]
    base = VERIFY_PROMPTS[sc["verify_from"]]
    if base.count(old) != 1:
        raise ValueError(f"{sc['verify_from']} does not contain {old!r} exactly once")
    return base.replace(old, new)


# ------------------------------------------------------------------------------------------- the answer
def last_boxed(s: str) -> tuple[str, int] | None:
    """The content of the last \\boxed{...} (balanced braces) and the index of its closing brace; None if absent or
    unbalanced (a truncated box)."""
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j = k = i + len("\\boxed{")
    depth = 1
    while k < len(s):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j:k], k
        k += 1
    return None


_TEXT_CMD = re.compile(r"\\(?:text|mathrm|textrm|mbox|operatorname|mathit|textit|rm)\s*\{([^{}]*)\}")
_FRAC = re.compile(r"\\[dt]?frac\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}")
_NUMBER = re.compile(r"^\s*([-+]?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|[-+]?\s*\.\d+)"
                     r"(?:\s*(?:×|\*|x)\s*10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?|\s*[eE]\s*([-+]?\d+))?")


def _clean(c: str) -> str:
    prev = None
    while prev != c:
        prev, c = c, _TEXT_CMD.sub(r" \1 ", c)
    for a, b in (("\\left", ""), ("\\right", ""), ("$", ""), ("\\,", " "), ("\\;", " "), ("\\:", " "), ("\\!", ""),
                 ("\\ ", " "), ("~", " "), ("\\quad", " "), ("\\qquad", " "), ("\\approx", "="), ("\\sim", "="),
                 ("\\cdot", "*"), ("\\times", "×"), ("−", "-"), ("\\%", "%"), ("^\\circ", " degree"),
                 ("\\circ", " degree")):
        c = c.replace(a, b)
    c = _FRAC.sub(lambda m: repr(float(m.group(1)) / float(m.group(2))), c)
    return c.split("=")[-1].strip()


def parse_boxed(content: str, target_unit: str) -> dict:
    """D151's answer rule: the first number (optional ×10^n / e exponent; thousands commas dropped) of the box,
    converted from a pint-parsable unit in the box to the requested unit; no unit -> the requested unit; a dimension
    mismatch -> no answer. Returns {value (in target_unit) | None, reason, unit_text}."""
    from physcal.env.units import ureg

    c = _clean(content)
    m = _NUMBER.match(c)
    if not m:
        return {"value": None, "reason": "no_number", "unit_text": None}
    v = float(m.group(1).replace(",", "").replace(" ", ""))
    exp = m.group(2) or m.group(3)
    if exp is not None:
        v *= 10.0 ** int(exp)
    rest = c[m.end():].strip().strip(".").replace("{", "").replace("}", "").replace("\\", "").strip()
    if not rest:
        return {"value": v, "reason": "ok_no_unit", "unit_text": ""}
    try:
        u = ureg.parse_units(rest.replace(" ", ""))
    except Exception:
        return {"value": v, "reason": "ok_unit_unparsed", "unit_text": rest}
    try:
        tu = ureg.parse_units(target_unit) if target_unit else ureg.dimensionless
        conv = float((v * u).to(tu).magnitude)
    except Exception:
        return {"value": None, "reason": "unit_mismatch", "unit_text": rest}
    return {"value": conv, "reason": "ok_converted", "unit_text": rest}


def grade(problem: dict, completion: str, truncated: bool, tol: float) -> dict:
    """One completion: the parsed value in the requested unit, correctness (within `tol`), the box's closing brace."""
    box = last_boxed(completion)
    if truncated or box is None:
        return {"value": None, "reason": "truncated" if truncated else "no_box", "correct": None, "brace": None}
    p = parse_boxed(box[0], problem["target_unit"])
    if p["value"] is None:
        return {**p, "correct": None, "brace": box[1]}
    t = problem["target_value"]
    rel = abs(p["value"] / t - 1) if t else (0.0 if p["value"] == 0 else math.inf)
    return {**p, "correct": bool(rel <= tol), "rel_error": rel, "brace": box[1]}


# ------------------------------------------------------------------------------------------- GPU
def make_backend(model: str, cfg: dict, cap: float, batch_size: int | None = None):
    from physcal.eval.run_eval import HFBackend

    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
    sc = cfg["scale"]
    g = sc["generation"]
    model_cfg = {"id": str(REPO_ROOT / sc["models"][model]["dir"]), "loader": "unsloth_4bit", "lora": None,
                 "max_seq_length": sc["max_seq_length"], "text_only": False, "dtype": "bfloat16",
                 "model_class": "causal_lm", "chat_template_kwargs": {}, "max_reserved_gib": cap}
    gen = {"do_sample": True, "num_samples": g["k"], "max_new_tokens": g["max_new_tokens"],
           "batch_size": batch_size or g["batch_size"], "repetition_penalty": g["repetition_penalty"],
           "seed": g["seed"], "temperature": g["temperature"], "top_p": g["top_p"], "top_k": g["top_k"]}
    return HFBackend(model_cfg["id"], model_cfg, gen)


def _identity(model: str, cfg: dict) -> dict:
    from physcal.eval.fingerprint import _dir_content_hash

    m = cfg["scale"]["models"][model]
    return {"model": model, "repo_id": m["repo_id"], "revision": m["revision"],
            "content_sha256": _dir_content_hash(REPO_ROOT / m["dir"])}


def generate_problems(backend, probs: list[dict], cfg: dict, seed: int) -> tuple[list[dict], float]:
    """k samples per problem (the backend's batching); rows carry the pool index, sample index and the grade."""
    sc = cfg["scale"]
    backend.gen_cfg["seed"] = seed
    t0 = time.time()
    gens = backend([messages(p["text"], cfg) for p in probs])
    dt = time.time() - t0
    k = sc["generation"]["k"]
    rows = []
    for i, p in enumerate(probs):
        for s in range(k):
            g = gens[i * k + s]
            gr = grade(p, g.text, g.truncated, sc["tolerance"])
            rows.append({"index": p["_index"], "key": [p["family"], p["depth"], p["seed"]], "sample": s,
                         "completion": g.text, "n_tokens": g.n_tokens, "truncated": g.truncated, **gr})
    return rows, dt


def feasibility(model: str, cap: float, cfg: dict | None = None, max_new_tokens: int | None = None,
                n: int | None = None, batch16: bool = True, tag: str = "") -> dict:
    """D151 3a: load under the cap, k = 8 on each dev problem (batch 8), then 2 problems at once (batch 16), and the
    readout costs (direct P(yes) and hidden states) on a few dev answers. D153: a budget override, the number of dev
    problems, no batch-16 test, and a file tag."""
    import torch

    cfg = cfg or _config()
    sc = cfg["scale"]
    if max_new_tokens:
        sc["generation"]["max_new_tokens"] = max_new_tokens
    lo, hi = sc["dev_indices"]
    probs = [{**p, "_index": i} for i, p in enumerate(pool(cfg)) if lo <= i < hi][:n or sc["feasibility_n"]]
    out = out_dir(model, cfg)
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"decision": "D151 item 3a (exploratory)" + (f"; D153 ({tag})" if tag else ""),
                 "identity": _identity(model, cfg), "cap_gib": cap, "dev_indices": [lo, hi],
                 "generation": sc["generation"], "n_problems": len(probs)}
    t0 = time.time()
    backend = make_backend(model, cfg, cap)
    res["load_s"] = time.time() - t0
    res["after_load_reserved_gib"] = torch.cuda.memory_reserved() / 1024**3
    rows, times = [], []
    for j, p in enumerate(probs):
        r, dt = generate_problems(backend, [p], cfg, sc["generation"]["seed"] + 10_000 + j)
        rows += r
        times.append(dt)
        print(f"  dev {p['_index']} ({p['family']} d{p['depth']}): {dt:.1f}s, tokens "
              f"{[x['n_tokens'] for x in r]}, correct {[x['correct'] for x in r]}", flush=True)
    res["s_per_problem_batch8"] = times
    res["peak_reserved_gib_batch8"] = torch.cuda.max_memory_reserved() / 1024**3
    res["oom_backoffs_batch8"] = backend.oom_backoffs
    r2 = []
    if batch16:
        backend.gen_cfg["batch_size"] = 2 * sc["generation"]["k"]
        before = backend.oom_backoffs
        r2, dt2 = generate_problems(backend, probs[:2], cfg, sc["generation"]["seed"] + 20_000)
        res["s_two_problems_batch16"] = dt2
        res["batch16_oom_backoffs"] = backend.oom_backoffs - before
        res["peak_reserved_gib_batch16"] = torch.cuda.max_memory_reserved() / 1024**3
        backend.gen_cfg["batch_size"] = sc["generation"]["batch_size"]
    res["by_problem"] = [{"index": p["_index"], "family": p["family"], "depth": p["depth"], "seconds": t,
                          "tokens": [r["n_tokens"] for r in rows if r["index"] == p["_index"]],
                          "correct": [r["correct"] for r in rows if r["index"] == p["_index"]]}
                         for p, t in zip(probs, times)]
    (out / f"feasibility{tag and '_' + tag}_completions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows + r2))
    answered = [r for r in rows if r["value"] is not None]
    res["dev_summary"] = {"n": len(rows), "answered": len(answered),
                          "correct": sum(bool(r["correct"]) for r in answered),
                          "truncated": sum(r["truncated"] for r in rows),
                          "reasons": {k: sum(r["reason"] == k for r in rows) for k in {r["reason"] for r in rows}},
                          "tokens_mean": float(np.mean([r["n_tokens"] for r in rows])),
                          "tokens_max": int(max(r["n_tokens"] for r in rows))}
    # readout costs on the first 8 answered dev completions
    items = [{**r, "text": next(p["text"] for p in probs if p["_index"] == r["index"])} for r in answered[:8]]
    if items:
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        yes_rows, check = read_yes(backend, items, cfg)
        res["yes_s_per_answer"] = (time.time() - t1) / len(items)
        res["yes_check"] = check
        res["yes_sample"] = [round(y["log_odds"], 3) for y in yes_rows]
        t2 = time.time()
        states, hcheck = hidden_states(backend, items, cfg)
        res["hidden_s_per_answer"] = (time.time() - t2) / len(items)
        res["hidden_check"] = hcheck
        res["hidden_layers"] = len(states)
        res["peak_reserved_gib_readouts"] = torch.cuda.max_memory_reserved() / 1024**3
    res["seconds"] = time.time() - t0
    (out / f"feasibility{tag and '_' + tag}.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("s_per_problem_batch8",)}, indent=1, default=str))
    return res


def chunk_path(model: str, start: int, cfg: dict) -> Path:
    return out_dir(model, cfg) / "samples" / f"chunk_{start:05d}.jsonl"


def sampled_prefix(model: str, cfg: dict | None = None) -> list[dict]:
    """Every row of the completed chunks, which must form a prefix of the pool (checked)."""
    cfg = cfg or _config()
    cs, k = cfg["scale"]["chunk_size"], cfg["scale"]["generation"]["k"]
    rows, start = [], 0
    while (f := chunk_path(model, start, cfg)).exists():
        r = [json.loads(line) for line in f.open()]
        if len(r) != cs * k or sorted({x["index"] for x in r}) != list(range(start, start + cs)):
            raise ValueError(f"{f} is not a complete chunk")
        rows += r
        start += cs
    extra = [p for p in (out_dir(model, cfg) / "samples").glob("chunk_*.jsonl") if int(p.stem.split("_")[1]) >= start]
    if extra:
        raise ValueError(f"chunks beyond the prefix: {extra}")
    return rows


def plan(deadline: float, cfg: dict | None = None) -> dict:
    """D151's stop time for 7B sampling: the time left minus the slack is shared so that the 7B readouts (expected
    kept answers x per-answer cost x margin) and, if its feasibility exists, the whole native 1.5B control fit on the
    same prefix. Writes <scale>/plan.json; returns it (t_stop in epoch seconds)."""
    cfg = cfg or _config()
    sc = cfg["scale"]
    now = time.time()
    cost, parts = 0.0, {}
    for m in sc["models"]:
        f = out_dir(m, cfg) / "feasibility.json"
        if not f.exists():
            continue
        fe = json.loads(f.read_text())
        g = float(np.mean(fe["s_per_problem_batch8"]))
        r = sc["expected_mixed_share"] * sc["generation"]["k"] * (fe["yes_s_per_answer"] + fe["hidden_s_per_answer"]) \
            * sc["readout_margin"]
        parts[m] = {"generate_s_per_problem": g, "readout_s_per_problem": r}
        cost += g + r
    avail = deadline - now - sc["stop_slack_s"] - sc["load_overhead_s"]
    n = int(max(0.0, avail) / cost) // sc["chunk_size"] * sc["chunk_size"] if cost else 0
    n = min(n, sc["dev_indices"][0])
    res = {"now": now, "deadline": deadline, "available_s": avail, "per_problem_s": parts, "n_planned": n,
           "t_stop": now + n * parts["7b"]["generate_s_per_problem"] + sc["load_overhead_s"] / 4,
           "control_planned": "1.5b" in parts}
    (REPO_ROOT / cfg["output_dir"] / "scale" / "plan.json").write_text(json.dumps(res, indent=1))
    return res


def sample(model: str, cap: float, stop_at: float, cfg: dict | None = None, limit: int | None = None) -> dict:
    """D151 3b: chunks of the pool in order from the first missing one; a chunk starts only if its projected end
    (the running mean chunk time, the feasibility's before the first) is before `stop_at` (epoch seconds). `limit`:
    no chunk beyond the first `limit` candidates (the 1.5B control samples the 7B's prefix, no further)."""
    cfg = cfg or _config()
    sc = cfg["scale"]
    cs = sc["chunk_size"]
    probs = [{**p, "_index": i} for i, p in enumerate(pool(cfg))]
    lo = min(sc["dev_indices"][0], limit if limit is not None else sc["dev_indices"][0])
    (out_dir(model, cfg) / "samples").mkdir(parents=True, exist_ok=True)
    start = len(sampled_prefix(model, cfg)) // sc["generation"]["k"]
    feas = json.loads((out_dir(model, cfg) / "feasibility.json").read_text())
    est = float(np.mean(feas["s_per_problem_batch8"])) * cs
    backend, times, done = None, [], 0
    meta_path = out_dir(model, cfg) / "samples" / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"decision": "D151 item 3b", "segments": []}
    while start + cs <= lo:
        proj = float(np.mean(times)) if times else est
        if time.time() + proj > stop_at:
            print(f"stop: chunk {start} projected {proj:.0f}s would end after the stop time", flush=True)
            break
        if backend is None:
            backend = make_backend(model, cfg, cap)
            meta["segments"].append({"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cap_gib": cap,
                                     "identity": _identity(model, cfg), "first_chunk": start,
                                     "provenance": getattr(backend, "provenance", None)})
        rows, dt = generate_problems(backend, probs[start:start + cs], cfg, sc["generation"]["seed"] + start // cs)
        tmp = chunk_path(model, start, cfg).with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        tmp.rename(chunk_path(model, start, cfg))
        times.append(dt)
        done += 1
        acc = np.mean([bool(r["correct"]) for r in rows])
        print(f"  chunk {start}: {dt:.0f}s ({dt / cs:.1f} s/problem), accuracy {acc:.3f}, oom backoffs "
              f"{backend.oom_backoffs}", flush=True)
        start += cs
    if backend is not None:
        meta["segments"][-1].update(chunks=done, seconds=float(sum(times)), oom_backoffs=backend.oom_backoffs,
                                    memory=backend.memory_stats(), last_chunk_end=start)
        meta_path.write_text(json.dumps(meta, indent=1, default=str))
    return {"chunks_done": done, "next_start": start}


# ------------------------------------------------------------------------------------------- readouts
def kept(rows: list[dict]) -> tuple[dict, list]:
    """Per problem key: its answered rows; and the kept keys (>= 1 correct and >= 1 incorrect answer), pool order."""
    by: dict = {}
    for r in rows:
        by.setdefault(tuple(r["key"]), []).append(r)
    keys = [k for k, rs in by.items() if any(r["correct"] is True for r in rs) and any(r["correct"] is False for r in rs)]
    return by, keys


def kept_items(model: str, cfg: dict) -> list[dict]:
    rows = sampled_prefix(model, cfg)
    by, keys = kept(rows)
    text = {i: p["text"] for i, p in enumerate(pool(cfg))}
    return [{**r, "text": text[r["index"]]} for k in keys for r in by[k] if r["value"] is not None]


def read_yes(backend, items: list[dict], cfg: dict) -> tuple[list[dict], dict]:
    """X4's direct arm in the native format: system, problem, the completion, the verification turn, then the
    generation prompt + "Verdict:"; the yes-vs-no log-odds of the next token (float32 head when it checks out)."""
    from physcal.eval import x4_verify as X4

    rc = cfg["scale"]["reading"]
    tok, model = backend.tok, backend.model
    vt = verify_text(cfg)
    yes, no = X4.token_ids(tok, rc["yes_tokens"]), X4.token_ids(tok, rc["no_tokens"])

    def ctx(it):
        conv = messages(it["text"], cfg) + [{"role": "assistant", "content": it["completion"]},
                                             {"role": "user", "content": vt}]
        return tok(tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True) + rc["marker"],
                   add_special_tokens=False)["input_ids"]

    ids = [ctx(it) for it in items]
    check = X4.logit_check(model, ids[0], backend.pad_id, backend.device, rc)
    reads = X4.read_out(model, ids, backend.pad_id, backend.device, rc, check["path"], yes, no, rc["score_batch_size"])
    return [{"key": it["key"], "sample": it["sample"], **r, "context_tokens": len(c)}
            for it, r, c in zip(items, reads, ids)], check | {"verify_sha256": hashlib.sha256(vt.encode()).hexdigest()}


def hidden_states(backend, items: list[dict], cfg: dict) -> tuple[list[np.ndarray], dict]:
    """Every layer's state at the token holding the closing brace of the last \\boxed{} (teacher-forced prompt +
    completion, right padding); x2_hidden's logit check on the first item."""
    import torch

    from physcal.eval import x2_hidden as XH

    sc = cfg["scale"]
    tok, model = backend.tok, backend.model
    tok.padding_side = "right"
    try:
        prompts = [tok.apply_chat_template(messages(it["text"], cfg), tokenize=False, add_generation_prompt=True)
                   for it in items]
        enc0 = tok([prompts[0] + items[0]["completion"]], return_tensors="pt", add_special_tokens=False)
        check = XH.logit_check(model, enc0, backend.device)
        if not XH._passes(check, sc["logit_check_rtol"]):
            raise RuntimeError(f"hidden states fail x2_hidden's logit check: {check}")
        layers: list[list[np.ndarray]] | None = None
        bs = sc["hidden_batch_size"]
        for i in range(0, len(items), bs):
            batch, pr = items[i:i + bs], prompts[i:i + bs]
            enc = tok([p + it["completion"] for p, it in zip(pr, batch)], return_tensors="pt", padding=True,
                      add_special_tokens=False, return_offsets_mapping=True)
            pos = [XH.token_at(enc["offset_mapping"][j].tolist(), len(pr[j]) + it["brace"])
                   for j, it in enumerate(batch)]
            enc.pop("offset_mapping")
            st = XH._states(model, enc, pos, backend.device)
            if st is None:
                raise RuntimeError("the model returned no hidden states")
            if layers is None:
                layers = [[] for _ in st]
            for k, s in enumerate(st):
                layers[k].append(s.cpu().numpy().astype(np.float16))
            del st
            torch.cuda.empty_cache()
    finally:
        tok.padding_side = "left"
    return [np.concatenate(L) for L in layers], check


def run_yes(model: str, cap: float, cfg: dict | None = None) -> dict:
    cfg = cfg or _config()
    items = kept_items(model, cfg)
    path = out_dir(model, cfg) / "yes.jsonl"
    done = {(tuple(r["key"]), r["sample"]) for r in map(json.loads, path.open())} if path.exists() else set()
    todo = [it for it in items if (tuple(it["key"]), it["sample"]) not in done]
    print(f"yes: {len(items)} kept answers, {len(todo)} to read", flush=True)
    if not todo:
        return {"n": len(items), "new": 0}
    backend = make_backend(model, cfg, cap)
    t0, step = time.time(), 32
    for i in range(0, len(todo), step):
        rows, check = read_yes(backend, todo[i:i + step], cfg)
        with path.open("a") as fh:
            fh.write("".join(json.dumps({**r, "check": check if j == 0 else None}) + "\n" for j, r in enumerate(rows)))
        print(f"  yes {i + len(rows)}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    return {"n": len(items), "new": len(todo), "seconds": time.time() - t0}


def run_hidden(model: str, cap: float, cfg: dict | None = None) -> dict:
    cfg = cfg or _config()
    items = kept_items(model, cfg)
    d = out_dir(model, cfg) / "hidden"
    d.mkdir(parents=True, exist_ok=True)
    ps = cfg["scale"]["hidden_part_size"]
    backend, t0 = None, time.time()
    for n, i in enumerate(range(0, len(items), ps)):
        f = d / f"part_{n:04d}.npz"
        part = items[i:i + ps]
        idx = [[*it["key"], it["sample"]] for it in part]
        if f.exists() and json.loads(str(np.load(f)["index"])) == idx:
            continue
        backend = backend or make_backend(model, cfg, cap)
        layers, check = hidden_states(backend, part, cfg)
        tmp = d / f"part_{n:04d}.tmp.npz"
        np.savez(tmp, index=json.dumps(idx), check=json.dumps(check, default=str),
                 **{f"layer_{k:02d}": a for k, a in enumerate(layers)})
        tmp.rename(f)
        print(f"  hidden part {n}: {i + len(part)}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    info = {"decision": "D151 item 3c (hidden states; the arrays are local, gitignored)", "n": len(items),
            "parts": len(list(d.glob("part_[0-9][0-9][0-9][0-9].npz"))), "identity": _identity(model, cfg),
            "position": "the token holding the closing brace of the last \\boxed{}", "seconds": time.time() - t0}
    (out_dir(model, cfg) / "hidden_info.json").write_text(json.dumps(info, indent=1))
    return info


def load_hidden(model: str, cfg: dict) -> tuple[list[np.ndarray], list[tuple]]:
    d = out_dir(model, cfg) / "hidden"
    parts = sorted(d.glob("part_[0-9][0-9][0-9][0-9].npz"))
    idx, layers = [], None
    for f in parts:
        z = np.load(f)
        idx += [tuple(x) for x in json.loads(str(z["index"]))]
        ls = sorted(k for k in z.files if k.startswith("layer_"))
        layers = layers or [[] for _ in ls]
        for k, name in enumerate(ls):
            layers[k].append(z[name])
    return [np.concatenate(L) for L in layers], [((a, b, c), s) for a, b, c, s in idx]


# ------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["feasibility", "plan", "sample", "yes", "hidden", "analyze"])
    ap.add_argument("--model", default="7b")
    ap.add_argument("--cap", type=float, default=None, help="allocator cap, GiB")
    ap.add_argument("--stop-at", type=float, default=None, help="sample: epoch seconds")
    ap.add_argument("--limit", type=int, default=None, help="sample: pool candidates at most (the control)")
    ap.add_argument("--deadline", type=float, default=None, help="plan: epoch seconds")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="feasibility: budget override (D153)")
    ap.add_argument("--n", type=int, default=None, help="feasibility: dev problems (D153)")
    ap.add_argument("--no-batch16", action="store_true", help="feasibility: skip the 2-problem test (D153)")
    ap.add_argument("--tag", default="", help="feasibility: output file tag (D153)")
    a = ap.parse_args()
    if a.action == "plan":
        print(json.dumps(plan(a.deadline), indent=1))
        return
    if a.action == "feasibility":
        feasibility(a.model, a.cap, max_new_tokens=a.max_new_tokens, n=a.n, batch16=not a.no_batch16, tag=a.tag)
    elif a.action == "sample":
        print(json.dumps(sample(a.model, a.cap, a.stop_at, limit=a.limit)))
    elif a.action == "yes":
        print(json.dumps(run_yes(a.model, a.cap)))
    elif a.action == "hidden":
        print(json.dumps(run_hidden(a.model, a.cap)))
    else:
        from physcal.eval.followup_scale_analysis import analyze

        analyze()


if __name__ == "__main__":
    main()

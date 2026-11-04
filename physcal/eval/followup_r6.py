"""D167 (EXPLORATORY, follow-up project, paper #2): R6-3 (a common candidate bank), R6-4 (a format × weights factorial
at 1.5B) and R6-5 (probe controls). Pre-registered in D167 before this code existed.

    uv run python -m physcal.eval.followup_r6 items          # CPU: build and freeze the item sets (+ manifest)
    uv run python -m physcal.eval.followup_r6 estimate       # CPU: GPU time per stage on the laptop (measured rates)
    uv run python -m physcal.eval.followup_r6 stages         # the GPU stages, in the queue's order
    uv run python -m physcal.eval.followup_r6 states --set S --reader R --cap C    # GPU (scripts/queue_followup_r6.sh)
    uv run python -m physcal.eval.followup_r6 yes --set S --reader R --cap C       # GPU
    uv run python -m physcal.eval.followup_r6 analyze        # CPU, after every GPU stage (4 jobs, the GPU idle)

Item sets (<output_dir>/r6/items/<set>.jsonl, frozen by manifest.json):
- R6-3: bank_7b, bank_1.5b (each model's answers on the 599 problems both keep), truncated (their truncated samples),
  spot_7b, spot_1.5b (the first own answers, re-read to check the reused D160 states);
- R6-4: fact_native (the native 1.5B's traces) and fact_terse (the SFT model's completions) on the 347 problems both
  keep, and fact_candidates.json (the identical-candidate subset);
- R6-5: answer_only_<m> (the box alone) and swap_<m> (another answer's reasoning, this answer's box), m = 7b, 1.5b.
Readers: 7b, 1.5b (the local 4-bit copies), sft (the merged SFT model, 4-bit at load), random_1.5b (D160's copy).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.eval import followup_audit as AU
from physcal.eval import followup_scale as F
from physcal.eval import followup_scale_analysis as A
from physcal.eval import x2_pairs

READERS = ("7b", "1.5b", "sft", "random_1.5b")
POS = ("p2", "p1")
# D167's GPU work, in the queue's order (the short 1.5B stages first): (kind, item set, reader)
GPU_STAGES = (
    ("states", "spot_1.5b", "1.5b"), ("states", "bank_7b", "1.5b"), ("yes", "bank_7b", "1.5b"),
    ("yes", "truncated", "1.5b"),
    ("states", "fact_native", "1.5b"), ("states", "fact_terse", "1.5b"),
    ("states", "fact_native", "sft"), ("states", "fact_terse", "sft"),
    ("states", "fact_native", "random_1.5b"), ("states", "fact_terse", "random_1.5b"),
    ("states", "answer_only_1.5b", "1.5b"), ("states", "swap_1.5b", "1.5b"),
    ("states", "spot_7b", "7b"), ("states", "bank_1.5b", "7b"), ("yes", "bank_1.5b", "7b"), ("yes", "truncated", "7b"),
    ("states", "answer_only_7b", "7b"), ("states", "swap_7b", "7b"),
)
EXPERIMENT = {"spot": "R6-3", "bank": "R6-3", "truncated": "R6-3", "fact": "R6-4", "answer": "R6-5", "swap": "R6-5"}


def r6_dir(cfg: dict) -> Path:
    return REPO_ROOT / cfg["output_dir"] / cfg["r6"]["output_dir"]


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ------------------------------------------------------------------------------------------- positions
def native_positions(completion: str) -> tuple[int, int] | None:
    """(p1, p2) in a native completion: the last character of the last box's content (trailing spaces left out) and
    its closing brace (D151's position). None without a closed, non-empty box."""
    box = F.last_boxed(completion)
    if box is None:
        return None
    content, brace = box
    start = brace - len(content)
    end = start + len(content.rstrip())
    return (end - 1, brace) if end > start else None


def terse_positions(completion: str) -> tuple[int, int] | None:
    """(p1, p2) in a terse completion: the last character of the single <answer> block's content and the `<` of its
    `</answer>`. None unless there is exactly one block, with content."""
    if completion.count("<answer>") != 1 or completion.count("</answer>") != 1:
        return None
    a = completion.index("<answer>") + len("<answer>")
    b = completion.index("</answer>")
    end = a + len(completion[a:b].rstrip())
    return (end - 1, b) if b > a and end > a else None


# ------------------------------------------------------------------------------------------- item sets (CPU)
def _kid(key) -> str:
    return "|".join(map(str, key))


def native_item(source: str, r: dict, completion: str | None = None, tag: str = "") -> dict:
    """An answered (or, with no box, truncated) native sample as an item; `completion` replaces the text read."""
    comp = r["completion"] if completion is None else completion
    pos = None if r["truncated"] else native_positions(comp)
    if not r["truncated"] and pos is None:
        raise ValueError(f"an answered sample without a box: {r['key']} {r['sample']}")
    return {"id": f"{source}|{_kid(r['key'])}|{r['sample']}{tag}", "key": list(r["key"]), "sample": r["sample"],
            "source": source, "frame": "native", "text": r["text"], "completion": comp,
            "p1": pos[0] if pos else None, "p2": pos[1] if pos else None,
            "correct": None if r["value"] is None else bool(r["correct"]), "n_tokens": r["n_tokens"],
            "truncated": bool(r["truncated"])}


def bank_sets(rows: dict[str, list[dict]], spot_n: int) -> dict[str, list[dict]]:
    """R6-3: each model's answered samples on the problems both keep (the 7B's kept order), its truncated samples,
    and the spot-check prefixes. `rows`: each model's graded rows with their problem text."""
    kept = {m: AU.group_rows(rows[m])[1] for m in ("7b", "1.5b")}
    common = [k for k in kept["7b"] if k in set(kept["1.5b"])]
    order = {k: i for i, k in enumerate(common)}
    out: dict = {"truncated": []}
    for m in ("7b", "1.5b"):
        rs = sorted((r for r in rows[m] if tuple(r["key"]) in order), key=lambda r: (order[tuple(r["key"])], r["sample"]))
        out[f"bank_{m}"] = [native_item(m, r) for r in rs if r["value"] is not None]
        out["truncated"] += [native_item(m, r) for r in rs if r["truncated"]]
        out[f"spot_{m}"] = out[f"bank_{m}"][:spot_n]
    out["truncated"].sort(key=lambda it: (order[tuple(it["key"])], it["source"], it["sample"]))
    return out


def terse_item(r: dict, tol: float) -> dict | None:
    """An SFT completion with a parsed answer and one <answer> block, as a terse item; None otherwise."""
    b, p = r["breakdown"], r["problem"]
    pos = terse_positions(r["completion"])
    if b["reason"] != "ok" or b.get("mu_si") is None or pos is None:
        return None
    key = [p["family"], p["depth"], p["seed"]]
    s = r.get("sample", 0)
    return {"id": f"sft|{_kid(key)}|{s}", "key": key, "sample": s, "source": "sft", "frame": "terse", "text": p["text"],
            "completion": r["completion"], "p1": pos[0], "p2": pos[1], "correct": bool(b["abs_rel_error"] <= tol),
            "n_tokens": r["n_tokens"], "truncated": bool(r.get("truncated", False)),
            "ratio": b["mu_si"] / b["truth_si"] if b["truth_si"] else None}


def match_candidates(native: list[dict], terse: list[dict], tol: float) -> list[dict]:
    """D167's identical candidates in one problem: correct native traces paired with correct terse traces in sample
    order (as many as the smaller count); each incorrect native trace with the first unmatched incorrect terse trace
    whose value agrees within `tol` (greedy, sample order). Items carry `ratio` (value / truth)."""
    nc, tc = [a for a in native if a["correct"]], [b for b in terse if b["correct"]]
    out = [{"native_id": a["id"], "terse_id": b["id"], "correct": True} for a, b in zip(nc, tc)]
    used: set = set()
    for a in (x for x in native if not x["correct"]):
        for b in (y for y in terse if not y["correct"]):
            if b["id"] in used or not a["ratio"] or not b["ratio"]:
                continue
            if abs(a["ratio"] / b["ratio"] - 1) <= tol:
                used.add(b["id"])
                out.append({"native_id": a["id"], "terse_id": b["id"], "correct": False})
                break
    return out


def factorial_sets(native_rows: list[dict], sft_recs: list[dict], target: dict, tol: float,
                   seeds: set) -> tuple[list[dict], list[dict], list[dict]]:
    """R6-4: the native 1.5B's answered traces and the SFT model's terse completions on the problems both keep (the
    native kept order; SFT problems restricted to `seeds`, the same candidate prefix), and the identical candidates.
    `target`: pool seed -> the problem's target value in the requested unit."""
    groups_n, kept_n = AU.group_rows(native_rows)
    terse_all = [t for t in (terse_item(r, tol) for r in sft_recs) if t is not None]
    tg: dict = {}
    for t in terse_all:
        tg.setdefault(tuple(t["key"]), []).append(t)
    kept_t = {k for k, ts in tg.items() if k[2] in seeds and any(t["correct"] for t in ts)
              and any(not t["correct"] for t in ts)}
    problems = [k for k in kept_n if k in kept_t]
    order = {k: i for i, k in enumerate(problems)}
    nat = sorted((r for r in native_rows if tuple(r["key"]) in order and r["value"] is not None),
                 key=lambda r: (order[tuple(r["key"])], r["sample"]))
    fn = [native_item("1.5b", r) | {"ratio": r["value"] / target[r["key"][2]] if target[r["key"][2]] else None}
          for r in nat]
    ft = sorted((t for k in problems for t in tg[k]), key=lambda t: (order[tuple(t["key"])], t["sample"]))
    cands = []
    for k in problems:
        cs = match_candidates([a for a in fn if tuple(a["key"]) == k], [b for b in ft if tuple(b["key"]) == k], tol)
        if any(c["correct"] for c in cs) and any(not c["correct"] for c in cs):
            cands += [c | {"key": list(k)} for c in cs]
    return fn, ft, cands


def answer_only_set(m: str, items: list[dict]) -> list[dict]:
    """R6-5 (c): each kept answer with its completion replaced by its box alone."""
    out = []
    for it in items:
        content, _ = F.last_boxed(it["completion"])
        comp = "\\boxed{" + content + "}"
        p1, p2 = native_positions(comp)
        out.append(it | {"id": it["id"] + "|answer_only", "completion": comp, "p1": p1, "p2": p2, "orig_id": it["id"]})
    return out


def swap_set(m: str, items: list[dict]) -> list[dict]:
    """R6-5 (d): each kept answer's box after the reasoning of the same problem's lowest-sample answered sample with
    the opposite label (the donor's completion up to its last \\boxed{)."""
    by: dict = {}
    for it in items:
        by.setdefault(tuple(it["key"]), []).append(it)
    out = []
    for it in items:
        donor = min((d for d in by[tuple(it["key"])] if d["correct"] != it["correct"]), key=lambda d: d["sample"])
        content, _ = F.last_boxed(it["completion"])
        comp = donor["completion"][:donor["completion"].rfind("\\boxed{")] + "\\boxed{" + content + "}"
        p1, p2 = native_positions(comp)
        out.append(it | {"id": it["id"] + "|swap", "completion": comp, "p1": p1, "p2": p2, "orig_id": it["id"],
                         "donor_sample": donor["sample"]})
    return out


def kaggle_rows(model: str, cfg: dict) -> list[dict]:
    text = {i: p["text"] for i, p in enumerate(F.pool(cfg))}
    return [r | {"text": text[r["index"]]} for r in A.graded(model, cfg)[2]]


def probe_order_items(model: str, rows: list[dict], cfg: dict) -> list[dict]:
    """D162's kept answers as items, in the cross-fitted probe's order (checked against its cache's inputs)."""
    corr = {(tuple(r["key"]), r["sample"]): bool(r["correct"]) for r in rows}
    index, _, _ = A.probe_inputs(model, "hidden", corr, cfg)
    by = {(tuple(r["key"]), r["sample"]): r for r in rows}
    return [native_item(model, by[(tuple(i["key"]), i["sample"])]) for i in index]


def build_items(cfg: dict | None = None) -> dict:
    """Build every item set and freeze it (JSONL + manifest with sha256). Refuses to overwrite a different freeze."""
    cfg = cfg or F._config()
    rc, tol = cfg["r6"], cfg["scale"]["tolerance"]
    rows = {m: kaggle_rows(m, cfg) for m in ("7b", "1.5b")}
    sets = bank_sets(rows, rc["spot_check_n"])
    x2 = load_config("x2")
    pool = F.pool(cfg)
    n = len({r["index"] for r in rows["7b"]})
    seeds = {p["seed"] for p in pool[:n]}
    target = {p["seed"]: p["target_value"] for p in pool}
    sets["fact_native"], sets["fact_terse"], cands = factorial_sets(rows["1.5b"], x2_pairs.kept_records(x2)[0],
                                                                   target, tol, seeds)
    for m in ("7b", "1.5b"):
        kept = probe_order_items(m, rows[m], cfg)
        sets[f"answer_only_{m}"] = answer_only_set(m, kept)
        sets[f"swap_{m}"] = swap_set(m, kept)
    d = r6_dir(cfg) / "items"
    d.mkdir(parents=True, exist_ok=True)
    blobs = {name: "".join(json.dumps(it, sort_keys=True) + "\n" for it in its) for name, its in sets.items()}
    cand_blob = json.dumps(cands, indent=0, sort_keys=True)
    manifest = {"decision": "D167", "sets": {k: {"n": len(sets[k]), "sha256": _sha(b)} for k, b in blobs.items()},
                "fact_candidates": {"n": len(cands), "problems": len({_kid(c["key"]) for c in cands}),
                                    "pairs": _n_candidate_pairs(cands), "sha256": _sha(cand_blob)}}
    mpath = d / "manifest.json"
    if mpath.exists() and json.loads(mpath.read_text()) != manifest:
        raise SystemExit(f"{mpath}: a different freeze exists; D167's item sets are not rebuilt")
    for k, b in blobs.items():
        (d / f"{k}.jsonl").write_text(b)
    (d / "fact_candidates.json").write_text(cand_blob)
    mpath.write_text(json.dumps(manifest, indent=1))
    return manifest


def _n_candidate_pairs(cands: list[dict]) -> int:
    by: dict = {}
    for c in cands:
        by.setdefault(_kid(c["key"]), []).append(c["correct"])
    return sum(sum(v) * sum(not x for x in v) for v in by.values())


def load_set(name: str, cfg: dict) -> list[dict]:
    """A frozen item set; its sha256 must match the manifest."""
    d = r6_dir(cfg) / "items"
    blob = (d / f"{name}.jsonl").read_text()
    want = json.loads((d / "manifest.json").read_text())["sets"][name]["sha256"]
    if _sha(blob) != want:
        raise SystemExit(f"items/{name}.jsonl does not match the manifest")
    return [json.loads(line) for line in blob.splitlines()]


def load_candidates(cfg: dict) -> list[dict]:
    d = r6_dir(cfg) / "items"
    blob = (d / "fact_candidates.json").read_text()
    if _sha(blob) != json.loads((d / "manifest.json").read_text())["fact_candidates"]["sha256"]:
        raise SystemExit("items/fact_candidates.json does not match the manifest")
    return json.loads(blob)


# ------------------------------------------------------------------------------------------- GPU readouts
def prompt_text(it: dict, tok, cfg: dict) -> str:
    """The reader's chat prompt for an item: D151's native frame, or X2's prompt for the terse frame."""
    from physcal.prompt import build_messages

    msgs = F.messages(it["text"], cfg) if it["frame"] == "native" else build_messages(it["text"], cfg["r6"]["terse_prompt_id"])
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def states_at(model, enc, pos: dict[str, list[int]], device: str) -> dict[str, list] | None:
    """One forward pass; every layer's state at each named list of positions (one per row)."""
    import torch

    with torch.inference_mode():
        out = model.model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device),
                          output_hidden_states=True, use_cache=False)
    hs = getattr(out, "hidden_states", None)
    if hs is None or len(hs) != model.config.num_hidden_layers + 1:
        return None
    rows = torch.arange(len(next(iter(pos.values()))), device=device)
    return {name: [h[rows, torch.tensor(p, device=device)].float() for h in hs] for name, p in pos.items()}


def positions_for(name: str, cfg: dict) -> tuple[str, ...]:
    """The positions a set's states keep: p2 always; p1 only where D167 uses it (R6-4)."""
    return POS if name in cfg["r6"]["p1_sets"] else ("p2",)


def item_states(backend, items: list[dict], cfg: dict, positions: tuple[str, ...] = POS
                ) -> tuple[dict[str, list[np.ndarray]], dict, list[str], list[bool]]:
    """Teacher-forced prompt + completion (right padding); every layer's state at each of `positions`; x2_hidden's
    logit check on the first item. Also the sha256 of each full text read (readers must read the same text) and, per
    item, whether p1 and p2 fall on one token (D167 counts it; empty without p1)."""
    import torch

    from physcal.eval import x2_hidden as XH

    tok, model = backend.tok, backend.model
    tok.padding_side = "right"
    try:
        prompts = [prompt_text(it, tok, cfg) for it in items]
        enc0 = tok([prompts[0] + items[0]["completion"]], return_tensors="pt", add_special_tokens=False)
        check = XH.logit_check(model, enc0, backend.device)
        if not XH._passes(check, cfg["scale"]["logit_check_rtol"]):
            raise RuntimeError(f"hidden states fail x2_hidden's logit check: {check}")
        out: dict = {p: None for p in positions}
        same: list[bool] = []
        bs = cfg["scale"]["hidden_batch_size"]
        for i in range(0, len(items), bs):
            batch, pr = items[i:i + bs], prompts[i:i + bs]
            enc = tok([p + it["completion"] for p, it in zip(pr, batch)], return_tensors="pt", padding=True,
                      add_special_tokens=False, return_offsets_mapping=True)
            offs = enc.pop("offset_mapping")
            pos = {p: [XH.token_at(offs[j].tolist(), len(pr[j]) + it[p]) for j, it in enumerate(batch)]
                   for p in positions}
            same += [a == b for a, b in zip(pos["p1"], pos["p2"])] if "p1" in pos else []
            st = states_at(model, enc, pos, backend.device)
            if st is None:
                raise RuntimeError("the model returned no hidden states")
            for p in positions:
                out[p] = out[p] or [[] for _ in st[p]]
                for k, s in enumerate(st[p]):
                    out[p][k].append(s.cpu().numpy().astype(np.float16))
            del st
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        tok.padding_side = "left"
    return {p: [np.concatenate(L) for L in out[p]] for p in positions}, check, \
        [_sha(p + it["completion"]) for p, it in zip(prompts, items)], same


def reader_dir(reader: str, cfg: dict) -> Path:
    if reader in ("7b", "1.5b"):
        return REPO_ROOT / cfg["scale"]["models"][reader]["dir"]
    if reader == "sft":
        return REPO_ROOT / cfg["r6"]["sft_dir"]
    if reader == "random_1.5b":
        return F.random_init_dir("1.5b", cfg)
    raise ValueError(f"unknown reader {reader}")


def reader_backend(reader: str, cfg: dict, cap: float):
    """The reader, loaded as every earlier readout: Unsloth 4-bit (the SFT and random copies quantized at load)."""
    if reader in ("7b", "1.5b"):
        return F.make_backend(reader, cfg, cap)
    d = reader_dir(reader, cfg)
    if reader == "random_1.5b" and not (d / "physcal_random_init.json").exists():
        raise SystemExit(f"no random-weights copy at {d}: run `followup_scale random_init --model 1.5b` first")
    return F.make_backend("1.5b", cfg, cap, model_dir=d)


def states_dir(name: str, reader: str, cfg: dict) -> Path:
    return r6_dir(cfg) / "states" / name / reader


def run_states(name: str, reader: str, cap: float, cfg: dict | None = None) -> dict:
    """Every layer's state at p1 and p2 for an item set, in resumable parts (states/<set>/<reader>/part_NNNN.npz)."""
    cfg = cfg or F._config()
    items = load_set(name, cfg)
    if any(it["p2"] is None for it in items):
        raise SystemExit(f"{name}: items without a position (truncated) have no states")
    d = states_dir(name, reader, cfg)
    d.mkdir(parents=True, exist_ok=True)
    ps, backend, t0 = cfg["r6"]["part_size"], None, time.time()
    for n, i in enumerate(range(0, len(items), ps)):
        f, part = d / f"part_{n:04d}.npz", items[i:i + ps]
        ids = [it["id"] for it in part]
        if f.exists() and json.loads(str(np.load(f)["ids"])) == ids:
            continue
        backend = backend or reader_backend(reader, cfg, cap)
        st, check, shas, same = item_states(backend, part, cfg, positions_for(name, cfg))
        tmp = d / f"part_{n:04d}.tmp.npz"
        np.savez(tmp, ids=json.dumps(ids), text_sha256=json.dumps(shas), check=json.dumps(check, default=str),
                 p1_is_p2=np.array(same, dtype=bool),
                 **{f"{p}_layer_{k:02d}": a for p in st for k, a in enumerate(st[p])})
        tmp.rename(f)
        print(f"  {name}/{reader} part {n}: {i + len(part)}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    info = {"decision": "D167", "set": name, "reader": reader, "reader_dir": str(reader_dir(reader, cfg)),
            "n": len(items), "parts": len(list(d.glob("part_[0-9][0-9][0-9][0-9].npz"))), "seconds": time.time() - t0,
            "positions": {"p2": "the first token after the final answer", "p1": "the final answer's last token"
                          if "p1" in positions_for(name, cfg) else "not kept (D167 uses p1 in R6-4 only)"}}
    (d.parent / f"{reader}_info.json").write_text(json.dumps(info, indent=1))
    return info


def load_states(name: str, reader: str, pos: str, cfg: dict) -> tuple[list[np.ndarray], list[str], list[str]]:
    """(layers at `pos`, item ids, text sha256s) of a finished states stage, in item order."""
    parts = sorted(states_dir(name, reader, cfg).glob("part_[0-9][0-9][0-9][0-9].npz"))
    ids, shas, layers = [], [], None
    for f in parts:
        z = np.load(f)
        ids += json.loads(str(z["ids"]))
        shas += json.loads(str(z["text_sha256"]))
        ks = sorted(k for k in z.files if k.startswith(f"{pos}_layer_"))
        layers = layers or [[] for _ in ks]
        for k, name_ in enumerate(ks):
            layers[k].append(z[name_])
    return [np.concatenate(L) for L in layers], ids, shas


def p1_is_p2_share(name: str, reader: str, cfg: dict) -> float:
    """The share of a states stage's items whose p1 and p2 fall on one token."""
    parts = sorted(states_dir(name, reader, cfg).glob("part_[0-9][0-9][0-9][0-9].npz"))
    return float(np.concatenate([np.load(f)["p1_is_p2"] for f in parts]).mean())


def yes_path(name: str, reader: str, cfg: dict) -> Path:
    return r6_dir(cfg) / "yes" / name / f"{reader}.jsonl"


def run_yes(name: str, reader: str, cap: float, cfg: dict | None = None) -> dict:
    """Direct P(yes) (D151 / D157) on an item set in the native frame, appended in chunks (resumable)."""
    cfg = cfg or F._config()
    items = load_set(name, cfg)
    if any(it["frame"] != "native" for it in items):
        raise SystemExit(f"{name}: P(yes) is defined in the native frame only")
    path = yes_path(name, reader, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["id"] for x in path.open()} if path.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    if not todo:
        return {"n": len(items), "new": 0}
    backend, t0, step = reader_backend(reader, cfg, cap), time.time(), cfg["r6"]["yes_step"]
    for i in range(0, len(todo), step):
        chunk = todo[i:i + step]
        rows, check = F.read_yes(backend, chunk, cfg)
        with path.open("a") as fh:
            fh.write("".join(json.dumps({"id": it["id"], **r, "check": check if j == 0 else None}) + "\n"
                             for j, (it, r) in enumerate(zip(chunk, rows))))
        print(f"  yes {name}/{reader} {i + len(chunk)}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    return {"n": len(items), "new": len(todo), "seconds": time.time() - t0}


# ------------------------------------------------------------------------------------------- GPU time
def estimate(cfg: dict | None = None) -> dict:
    """Laptop GPU time per stage: context tokens (the reader's prompt + completion, + the verification turn for P(yes))
    × the measured s / token (configs r6.rates; batch 1, pauses included), a per-item floor, one load per stage."""
    from transformers import AutoTokenizer

    cfg = cfg or F._config()
    rates = cfg["r6"]["rates"]
    tok = AutoTokenizer.from_pretrained(str(REPO_ROOT / cfg["scale"]["models"]["1.5b"]["dir"]))
    vt = F.verify_text(cfg)
    marker = cfg["scale"]["reading"]["marker"]
    cache: dict = {}

    def ctx(it, kind):
        key = (it["id"], kind)
        if key not in cache:
            p = prompt_text(it, tok, cfg)
            n = len(tok(p + it["completion"], add_special_tokens=False)["input_ids"])
            if kind == "yes":
                conv = F.messages(it["text"], cfg) + [{"role": "assistant", "content": it["completion"]},
                                                      {"role": "user", "content": vt}]
                n = len(tok(tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True) + marker,
                            add_special_tokens=False)["input_ids"])
            cache[key] = n
        return cache[key]

    dims = {}
    for reader in READERS:
        c = json.loads((reader_dir(reader, cfg) / "config.json").read_text())
        dims[reader] = (c["num_hidden_layers"] + 1) * c["hidden_size"] * 2       # bytes per position (fp16)
    out: dict = {"stages": [], "by_experiment": {}, "by_size": {}, "disk_gb": 0.0}
    for kind, name, reader in GPU_STAGES:
        items = load_set(name, cfg)
        size = "7b" if reader == "7b" else "1.5b"
        r = rates[size]
        toks = np.array([ctx(it, "yes" if kind == "yes" else "states") for it in items])
        s = float(np.maximum(r[kind if kind == "yes" else "hidden"] * toks, r["floor_s"]).sum()) + rates["load_s"]
        exp = EXPERIMENT[name.split("_")[0]]
        gb = len(items) * dims[reader] * len(positions_for(name, cfg)) / 1e9 if kind == "states" else 0.0
        out["disk_gb"] += gb
        out["stages"].append({"kind": kind, "set": name, "reader": reader, "n": len(items),
                              "mean_tokens": float(toks.mean()), "max_tokens": int(toks.max()), "hours": s / 3600,
                              "disk_gb": gb, "experiment": exp})
        out["by_experiment"][exp] = out["by_experiment"].get(exp, 0.0) + s / 3600
        out["by_size"][size] = out["by_size"].get(size, 0.0) + s / 3600
    out["total_hours"] = sum(x["hours"] for x in out["stages"])
    path = r6_dir(cfg) / "gpu_estimate.json"
    path.write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------------------- pairs (CPU)
def by_problem(items: list[dict]) -> dict[tuple, list[dict]]:
    out: dict = {}
    for it in items:
        out.setdefault(tuple(it["key"]), []).append(it)
    return out


def _score(x, y) -> float:
    return 0.5 if (x is None or y is None or x == y) else float(x > y)


def per_problem(groups: dict, keys: list, conf: dict, ok=None) -> np.ndarray:
    """Per problem: the share of its (correct, incorrect) item pairs passing ok(a, b) where the correct item gets more
    confidence (ties and missing 0.5); NaN for a problem without such a pair. `conf`: item id -> value."""
    out = []
    for k in keys:
        s = [_score(conf.get(a["id"]), conf.get(b["id"])) for a in groups[k] if a["correct"] for b in groups[k]
             if b["correct"] is False and (ok is None or ok(a, b))]
        out.append(float(np.mean(s)) if s else np.nan)
    return np.array(out)


def n_pairs(groups: dict, keys: list, ok=None) -> np.ndarray:
    return np.array([sum(1 for a in groups[k] if a["correct"] for b in groups[k]
                         if b["correct"] is False and (ok is None or ok(a, b))) for k in keys], dtype=float)


def pair_filters(rel: float) -> dict:
    """R6-3's pair sets: within one source (primary), each source, across sources, all, length-matched within."""
    within = lambda a, b: a["source"] == b["source"]  # noqa: E731
    return {"within_source": within, "source_7b": lambda a, b: a["source"] == b["source"] == "7b",
            "source_1.5b": lambda a, b: a["source"] == b["source"] == "1.5b",
            "cross_source": lambda a, b: a["source"] != b["source"], "all": None,
            "within_source_length_matched": lambda a, b: within(a, b) and AU.length_matched(a["n_tokens"],
                                                                                            b["n_tokens"], rel)}


def boot_paired(arrays: dict[str, np.ndarray], fn, n_boot: int, seed: int) -> dict:
    """A statistic fn({name: per-problem array}) -> float, and its bootstrap over problems (the same resamples for
    every array; problems are the arrays' common rows)."""
    n = len(next(iter(arrays.values())))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))
    reps = np.array([fn({k: v[i] for k, v in arrays.items()}) for i in idx])
    return {"value": float(fn(arrays)), "ci95": np.percentile(reps, [2.5, 97.5]).tolist(), "n_problems": n}


def _mean_diff(a: str, b: str):
    return lambda d: float(np.mean(d[a]) - np.mean(d[b]))


def _summ(pa: np.ndarray, sc: dict) -> dict:
    return x2_pairs.summarize(pa, sc["bootstrap"], sc["seed"])


def _side(x: dict) -> int:
    lo, hi = x["ci95"]
    return 1 if lo > 0 else (-1 if hi < 0 else 0)


def cached(cfg: dict, name: str, inputs: list, compute) -> tuple[dict, object]:
    """compute() -> ({item id: value}, info), cached in <r6>/cache/<name>.json under the sha256 of `inputs` (arrays by
    their bytes, anything else as sorted JSON), so a re-run of the analysis does not refit what it already has."""
    h = hashlib.sha256()
    for x in inputs:
        if isinstance(x, np.ndarray):
            h.update(np.ascontiguousarray(x).tobytes())
        else:
            h.update(json.dumps(x, sort_keys=True, default=str).encode())
    sha, path = h.hexdigest(), r6_dir(cfg) / "cache" / f"{name}.json"
    if path.exists() and (c := json.loads(path.read_text()))["inputs_sha256"] == sha:
        print(f"  {name}: from the cache", flush=True)
        return c["conf"], c["info"]
    t0 = time.time()
    conf, info = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"decision": "D167 (analysis cache)", "inputs_sha256": sha, "info": info, "conf": conf},
                              default=float))
    tmp.rename(path)
    print(f"  {name}: fitted in {time.time() - t0:.0f}s", flush=True)
    return conf, info


def _probe_inputs(name: str, layers: list[np.ndarray], items: list[dict], cfg: dict) -> list:
    return [name, [[it["id"], it["key"], it["correct"]] for it in items], cfg["scale"]["probe"], *layers]


# ------------------------------------------------------------------------------------------- probes (CPU)
def crossfit_items(layers: list[np.ndarray], items: list[dict], cfg: dict) -> tuple[dict, dict]:
    """x2_analysis.crossfit_confidence on items (folds by problem, (layer, C) by inner CV); item id -> P(correct)."""
    from physcal.eval import x2_analysis as XA

    index = [{"key": it["key"], "sample": i, "correct": it["correct"]} for i, it in enumerate(items)]
    pc = dict(cfg["scale"]["probe"])
    pc.pop("even_layers_only")
    conf, info = XA.crossfit_confidence(layers, index, pc)
    return {items[s]["id"]: v for (_, s), v in conf.items()}, info


def stored_states(model: str, cfg: dict) -> tuple[list[np.ndarray], dict]:
    """D160's stored states of a model's kept answers at the closing brace (= p2), and (key, sample) -> row."""
    layers, idx = F.load_hidden(model, cfg)
    return layers, {(tuple(k), s): i for i, (k, s) in enumerate(idx)}


def spot_check(states: list[np.ndarray], ref: list[np.ndarray], rtol: float) -> dict:
    """max |diff| / max |ref| per layer between re-extracted and stored states; passes if every layer is <= rtol."""
    gaps = [float(np.abs(s.astype(np.float32) - r.astype(np.float32)).max() / max(np.abs(r).max(), 1e-12))
            for s, r in zip(states, ref)]
    return {"max_gap": max(gaps), "rtol": rtol, "passes": max(gaps) <= rtol, "n": int(len(states[0]))}


def _key_sample(it: dict) -> tuple:
    return tuple(it["key"]), it["sample"]


def own_and_cross_states(reader: str, bank: list[dict], cfg: dict) -> list[np.ndarray]:
    """A reader's states for the whole bank in bank order: its own answers from D160's stored states, the other
    model's from R6-3's cross stage (p2)."""
    other = "1.5b" if reader == "7b" else "7b"
    own_layers, row = stored_states(reader, cfg)
    cross_layers, cross_ids, _ = load_states(f"bank_{other}", reader, "p2", cfg)
    crow = {i: j for j, i in enumerate(cross_ids)}
    out = []
    for L_own, L_cross in zip(own_layers, cross_layers):
        X = np.empty((len(bank), L_own.shape[1]), dtype=np.float16)
        for j, it in enumerate(bank):
            X[j] = L_own[row[_key_sample(it)]] if it["source"] == reader else L_cross[crow[it["id"]]]
        out.append(X)
    return out


def yes_conf(reader: str, bank: list[dict], cfg: dict) -> dict:
    """A reader's P(yes) log-odds per bank item: own answers from D162's yes.jsonl, the other's from R6-3's stage."""
    other = "1.5b" if reader == "7b" else "7b"
    own = {(tuple(r["key"]), r["sample"]): r["log_odds"] for r in map(json.loads, (F.out_dir(reader, cfg) / "yes.jsonl").open())}
    cross = {r["id"]: r["log_odds"] for r in map(json.loads, yes_path(f"bank_{other}", reader, cfg).open())}
    return {it["id"]: (own[_key_sample(it)] if it["source"] == reader else cross[it["id"]]) for it in bank}


# ------------------------------------------------------------------------------------------- R6-3
def analyze_bank(cfg: dict) -> dict:
    sc, rc = cfg["scale"], cfg["r6"]
    bank = load_set("bank_7b", cfg) + load_set("bank_1.5b", cfg)
    groups, keys = by_problem(bank), list(by_problem(bank))
    res: dict = {"n_items": len(bank), "n_problems": len(keys), "spot_check": {}, "readers": {}}
    conf: dict = {"length": {it["id"]: -float(it["n_tokens"]) for it in bank}}
    recs = [AU.text_fields(it["text"], it["completion"]) for it in bank]
    tc = cfg["audit"]["text"]
    def text_fit():
        p, folds = AU.crossfit_text(recs, np.array([it["correct"] for it in bank], dtype=int),
                                    [_kid(it["key"]) for it in bank], sc["probe"]["folds"], tc["feature_set"],
                                    tc["inner_folds"], load_config(tc["a5_config"]))
        return {it["id"]: float(v) for it, v in zip(bank, p)}, folds
    conf["text"], res["text_folds"] = cached(cfg, "bank_text", ["bank_text", [it["id"] for it in bank], recs, tc,
                                                                [it["correct"] for it in bank]], text_fit)
    for reader in ("7b", "1.5b"):
        spot = load_set(f"spot_{reader}", cfg)
        st, ids, _ = load_states(f"spot_{reader}", reader, "p2", cfg)
        own_layers, row = stored_states(reader, cfg)
        res["spot_check"][reader] = spot_check(st, [L[[row[_key_sample(it)] for it in spot]] for L in own_layers],
                                               rc["spot_check_rtol"])
        if not res["spot_check"][reader]["passes"]:
            raise SystemExit(f"R6-3 spot check failed for {reader}: {res['spot_check'][reader]} (re-read own answers)")
        conf[f"yes_{reader}"] = yes_conf(reader, bank, cfg)
        layers = own_and_cross_states(reader, bank, cfg)
        conf[f"probe_{reader}"], res["readers"][reader] = cached(
            cfg, f"bank_probe_{reader}", _probe_inputs(f"bank_probe_{reader}", layers, bank, cfg),
            lambda: crossfit_items(layers, bank, cfg))
        del layers
    filt = pair_filters(cfg["audit"]["length_match_rel"])
    res["by_filter"] = {}
    pa = {}
    for fname, ok in filt.items():
        ks = [k for k, n in zip(keys, n_pairs(groups, keys, ok)) if n > 0]
        pa[fname] = {c: per_problem(groups, ks, v, ok) for c, v in conf.items()}
        res["by_filter"][fname] = {"n_problems": len(ks), "n_pairs": int(n_pairs(groups, ks, ok).sum()),
                                   "readouts": {c: _summ(v, sc) for c, v in pa[fname].items()},
                                   "7b - 1.5b reader": {r: A.paired(pa[fname][f"{r}_7b"], pa[fname][f"{r}_1.5b"],
                                                                    sc["bootstrap"], sc["seed"]) for r in ("yes", "probe")}}
    res["primary"] = res["by_filter"]["within_source"]["7b - 1.5b reader"]
    s7, s1 = pa["source_7b"], pa["source_1.5b"]
    res["reader_x_source"] = {r: {"table": {f"reader {a} / answers {b}": float(np.mean(pa[f"source_{b}"][f"{r}_{a}"]))
                                            for a in ("7b", "1.5b") for b in ("7b", "1.5b")},
                                  "interaction": boot_paired({"a": s7[f"{r}_7b"], "b": s7[f"{r}_1.5b"],
                                                              "c": s1[f"{r}_7b"], "d": s1[f"{r}_1.5b"]},
                                                             lambda d: float(np.mean(d["a"] - d["b"]) - np.mean(d["c"] - d["d"])),
                                                             sc["bootstrap"], sc["seed"])} for r in ("yes", "probe")}
    nw = n_pairs(groups, keys, filt["within_source"])
    rng = np.random.default_rng(sc["seed"])
    ix = rng.integers(0, len(keys), (sc["bootstrap"], len(keys)))
    res["pair_weighted_within_source"] = {
        r: (lambda a, b: {"diff": float(AU._pw(a, nw)[0] - AU._pw(b, nw)[0]),
                          "ci95": np.percentile(AU._pw(a, nw, ix) - AU._pw(b, nw, ix), [2.5, 97.5]).tolist()})(
            pa["within_source"][f"{r}_7b"], pa["within_source"][f"{r}_1.5b"]) for r in ("yes", "probe")}
    res["truncated"] = truncated_readout(bank, load_set("truncated", cfg), cfg)
    res["reading"] = {"verdict size effect holds on identical inputs": _side(res["primary"]["yes"]) > 0,
                      "probe": {1: "the 7B's states carry more", -1: "the 7B's states carry less",
                                0: "no reader difference on identical inputs"}[_side(res["primary"]["probe"])]}
    return res


def truncated_readout(bank: list[dict], trunc: list[dict], cfg: dict) -> dict:
    """R6-3's separate report: each reader's P(yes) on truncated samples, against the answered ones in their problem."""
    sc = cfg["scale"]
    out = {}
    for reader in ("7b", "1.5b"):
        yb = yes_conf(reader, bank, cfg)
        yt = {r["id"]: r["log_odds"] for r in map(json.loads, yes_path("truncated", reader, cfg).open())}
        tb = by_problem(trunc)
        bb = by_problem(bank)
        res = {}
        for cls in (True, False):
            pa = np.array([np.mean([_score(yb[a["id"]], yt[t["id"]]) for a in bb[k] if a["correct"] is cls
                                    for t in tb[k]]) for k in tb if any(a["correct"] is cls for a in bb[k])])
            res["correct vs truncated" if cls else "incorrect vs truncated"] = _summ(pa, sc)
        res["mean_log_odds"] = {"correct": float(np.mean([yb[a["id"]] for a in bank if a["correct"]])),
                                "incorrect": float(np.mean([yb[a["id"]] for a in bank if not a["correct"]])),
                                "truncated": float(np.mean(list(yt.values())))}
        out[reader] = res
    return out


# ------------------------------------------------------------------------------------------- R6-4
def factorial_effects(pa: dict, sc: dict) -> dict:
    """D167's effects from per-problem pairwise accuracies pa[(reader, frame)] on the same problems."""
    arrays = {f"{r}/{f}": v for (r, f), v in pa.items()}
    W_N = lambda d: d["sft/native"] - d["1.5b/native"]  # noqa: E731
    W_T = lambda d: d["sft/terse"] - d["1.5b/terse"]  # noqa: E731
    P_n = lambda d: d["1.5b/terse"] - d["1.5b/native"]  # noqa: E731
    P_s = lambda d: d["sft/terse"] - d["sft/native"]  # noqa: E731
    stats = {"W_native": lambda d: float(np.mean(W_N(d))), "W_terse": lambda d: float(np.mean(W_T(d))),
             "P_native_weights": lambda d: float(np.mean(P_n(d))), "P_sft_weights": lambda d: float(np.mean(P_s(d))),
             "W": lambda d: float(np.mean((W_N(d) + W_T(d)) / 2)), "P": lambda d: float(np.mean((P_n(d) + P_s(d)) / 2)),
             "interaction": lambda d: float(np.mean(W_T(d) - W_N(d))),
             "|W| - |P|": lambda d: abs(float(np.mean((W_N(d) + W_T(d)) / 2))) - abs(float(np.mean((P_n(d) + P_s(d)) / 2)))}
    if "random_1.5b/native" in arrays:
        S = lambda d: d["random_1.5b/terse"] - d["random_1.5b/native"]  # noqa: E731
        stats["surface (random/terse - random/native)"] = lambda d: float(np.mean(S(d)))
        stats["P net of surface"] = lambda d: float(np.mean((P_n(d) + P_s(d)) / 2 - S(d)))
    out = {k: boot_paired(arrays, f, sc["bootstrap"], sc["seed"]) for k, f in stats.items()}
    return _factorial_reading(out)


def _factorial_reading(out: dict) -> dict:
    """D167's interpretation table, added to a dict of effects."""
    w, p = _side(out["W"]) != 0, _side(out["P"]) != 0
    out["reading"] = {(True, False): "the weights factor dominates: fine-tuning is implicated",
                      (False, True): "the presentation factor dominates: format / readout",
                      (True, True): "both factors (the larger by |W| - |P|, descriptive)",
                      (False, False): "neither factor is shown"}[(w, p)]
    out["the factors interact"] = _side(out["interaction"]) != 0
    return out


def pair_weighted_effects(pa: dict, n: dict, sc: dict) -> dict:
    """D172: D167's effects with each pair weighing the same. Every cell is AU._pw's Σ n·pa / Σ n, with n[frame] the
    pairs per problem in that frame (the two banks hold different pairs), and the effects are the same contrasts of
    those cells. Bootstrap over problems, the ratio recomputed per replicate (R6-3's secondary)."""
    m = len(next(iter(pa.values())))
    ix = np.random.default_rng(sc["seed"]).integers(0, m, (sc["bootstrap"], m))
    point = {k: float(AU._pw(v, n[k[1]])[0]) for k, v in pa.items()}
    reps = {k: AU._pw(v, n[k[1]], ix) for k, v in pa.items()}

    def effects(c: dict) -> dict:
        w_n, w_t = c[("sft", "native")] - c[("1.5b", "native")], c[("sft", "terse")] - c[("1.5b", "terse")]
        p_n, p_s = c[("1.5b", "terse")] - c[("1.5b", "native")], c[("sft", "terse")] - c[("sft", "native")]
        w, p = (w_n + w_t) / 2, (p_n + p_s) / 2
        e = {"W_native": w_n, "W_terse": w_t, "P_native_weights": p_n, "P_sft_weights": p_s, "W": w, "P": p,
             "interaction": w_t - w_n, "|W| - |P|": abs(w) - abs(p)}
        if ("random_1.5b", "native") in c:
            s = c[("random_1.5b", "terse")] - c[("random_1.5b", "native")]
            e |= {"surface (random/terse - random/native)": s, "P net of surface": p - s}
        return e

    est, boot = effects(point), effects(reps)
    out = {k: {"value": float(v), "ci95": np.percentile(boot[k], [2.5, 97.5]).tolist(), "n_problems": m}
           for k, v in est.items()}
    return {"cells": {f"{r}/{f}": v for (r, f), v in point.items()}, "effects": _factorial_reading(out)}


def candidate_pa(cands: list[dict], conf: dict, frame: str) -> tuple[list, np.ndarray]:
    """Per problem of the identical-candidate subset: pairwise accuracy of its candidates, each scored by its trace in
    `frame`."""
    by: dict = {}
    for c in cands:
        by.setdefault(tuple(c["key"]), []).append(c)
    keys = list(by)
    idk = f"{frame}_id"
    return keys, np.array([np.mean([_score(conf.get(a[idk]), conf.get(b[idk])) for a in by[k] if a["correct"]
                                    for b in by[k] if not b["correct"]]) for k in keys])


def analyze_factorial(cfg: dict) -> dict:
    sc, rc = cfg["scale"], cfg["r6"]
    banks = {f: load_set(f"fact_{f}", cfg) for f in ("native", "terse")}
    cands = load_candidates(cfg)
    res: dict = {"n": {f: len(b) for f, b in banks.items()}, "n_candidates": len(cands), "checks": {}, "cells": {}}
    conf: dict = {}
    for f, items in banks.items():
        shas = {}
        for reader in ("1.5b", "sft", "random_1.5b"):
            for pos in POS:
                layers, ids, sh = load_states(f"fact_{f}", reader, pos, cfg)
                if ids != [it["id"] for it in items]:
                    raise SystemExit(f"fact_{f}/{reader}: states are not in item order")
                shas[reader] = sh
                nm = f"fact_{f}_{reader}_{pos}"
                conf[(reader, f, pos)], res["cells"][f"{reader}/{f}/{pos}"] = cached(
                    cfg, nm, _probe_inputs(nm, layers, items, cfg), lambda: crossfit_items(layers, items, cfg))
        res["checks"][f"{f}: every reader read the same text"] = len({tuple(v) for v in shas.values()}) == 1
        if not res["checks"][f"{f}: every reader read the same text"]:
            raise SystemExit(f"fact_{f}: the readers' texts differ")
        conf[("length", f, "p2")] = {it["id"]: -float(it["n_tokens"]) for it in items}
    st, ids, _ = load_states("fact_native", "1.5b", "p2", cfg)
    own, row = stored_states("1.5b", cfg)
    n = rc["spot_check_n"]
    anchor = spot_check([L[:n] for L in st], [L[[row[_key_sample(it)] for it in banks["native"][:n]]] for L in own],
                        rc["spot_check_rtol"])
    res["checks"]["anchor (D160's states)"] = anchor
    if not anchor["passes"]:
        raise SystemExit(f"R6-4 anchor failed: {anchor}")
    readers = ("1.5b", "sft", "random_1.5b")
    res["primary"], res["full_banks"] = {}, {}
    for pos in POS:
        pa = {}
        for r in readers:
            for f in ("native", "terse"):
                keys, pa[(r, f)] = candidate_pa(cands, conf[(r, f, pos)], f)
        res["primary"][pos] = {"n_problems": len(keys), "cells": {f"{r}/{f}": _summ(v, sc) for (r, f), v in pa.items()},
                               "length": {f: _summ(candidate_pa(cands, conf[("length", f, "p2")], f)[1], sc)
                                          for f in ("native", "terse")},
                               "effects": factorial_effects(pa, sc)}
        groups = {f: by_problem(b) for f, b in banks.items()}
        keys_f = list(groups["native"])
        full = {}
        for variant, ok in (("all pairs", None),
                            ("length-matched", lambda a, b: AU.length_matched(a["n_tokens"], b["n_tokens"],
                                                                              cfg["audit"]["length_match_rel"]))):
            pa_f = {(r, f): per_problem(groups[f], keys_f, conf[(r, f, pos)], ok) for r in readers for f in groups}
            keep = np.all([~np.isnan(v) for v in pa_f.values()], axis=0)
            pa_f = {k: v[keep] for k, v in pa_f.items()}
            full[variant] = {"n_problems": int(keep.sum()), "cells": {f"{r}/{f}": _summ(v, sc) for (r, f), v in pa_f.items()},
                             "effects": factorial_effects(pa_f, sc)}
        res["full_banks"][pos] = full
    res["p1_is_p2_share"] = {f"{f}/{r}": p1_is_p2_share(f"fact_{f}", r, cfg) for f in banks for r in readers}
    return res


def analyze_factorial_pair_weighted(cfg: dict) -> dict:
    """D172: the pair-weighting secondary D167 lists for R6-4, which analyze_factorial never computed. It reads the
    cached fits only and writes its own file, so results.json stays as the registered run left it. Gate: the same
    fits, per-problem weighted, must give back results.json's cells exactly."""
    sc, rel = cfg["scale"], cfg["audit"]["length_match_rel"]
    readers, frames = ("1.5b", "sft", "random_1.5b"), ("native", "terse")
    saved = json.loads((r6_dir(cfg) / "results.json").read_text())["R6-4"]
    banks = {f: load_set(f"fact_{f}", cfg) for f in frames}
    cands = load_candidates(cfg)
    groups = {f: by_problem(b) for f, b in banks.items()}
    keys_f = list(groups["native"])
    by: dict = {}
    for c in cands:
        by.setdefault(tuple(c["key"]), []).append(c)
    n_cand = np.array([sum(c["correct"] for c in v) * sum(not c["correct"] for c in v) for v in by.values()], dtype=float)

    def gate(pa: dict, ref: dict) -> None:
        for (r, f), v in pa.items():
            if float(np.mean(v)) != ref["cells"][f"{r}/{f}"]["pairwise_accuracy"]:
                raise SystemExit(f"R6-4 pair weighting: {r}/{f} does not give back results.json's cell")

    res: dict = {"decision": "D172 (exploratory, paper #2)", "primary": {}, "full_banks": {}}
    for pos in POS:
        conf = {(r, f): json.loads((r6_dir(cfg) / "cache" / f"fact_{f}_{r}_{pos}.json").read_text())["conf"]
                for r in readers for f in frames}
        pa = {k: candidate_pa(cands, c, k[1])[1] for k, c in conf.items()}
        gate(pa, saved["primary"][pos])
        res["primary"][pos] = {"n_problems": len(n_cand), "n_pairs": int(n_cand.sum()),
                               **pair_weighted_effects(pa, {f: n_cand for f in frames}, sc)}
        res["full_banks"][pos] = {}
        for variant, ok in (("all pairs", None),
                            ("length-matched", lambda a, b: AU.length_matched(a["n_tokens"], b["n_tokens"], rel))):
            pa_f = {k: per_problem(groups[k[1]], keys_f, c, ok) for k, c in conf.items()}
            keep = np.all([~np.isnan(v) for v in pa_f.values()], axis=0)
            pa_f = {k: v[keep] for k, v in pa_f.items()}
            gate(pa_f, saved["full_banks"][pos][variant])
            n = {f: n_pairs(groups[f], keys_f, ok)[keep] for f in frames}
            res["full_banks"][pos][variant] = {"n_problems": int(keep.sum()),
                                               "n_pairs": {f: int(n[f].sum()) for f in frames},
                                               **pair_weighted_effects(pa_f, n, sc)}
    path = r6_dir(cfg) / "r6_4_pair_weighted.json"
    path.write_text(json.dumps(res, indent=1, default=float))
    return res


# ------------------------------------------------------------------------------------------- R6-5
def hand_features(rec: dict, completion: str, value: float | None, depth: int, n_tokens: int, rt: dict) -> list[float]:
    """D167's 15 hand features from the text the probe can see (problem, answer = the box's content, reasoning)."""
    reason, ans = rec["reasoning"], rec["answer"]
    nums = re.findall(r"\d+(?:\.\d+)?", reason)
    lead = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", ans)
    digits = len(re.sub(r"[^\d]", "", lead.group(0)).lstrip("0")) if lead else 0
    low = reason.lower()
    box_start = completion.rfind("\\boxed{")
    return [float(n_tokens), float(len(reason)), float(reason.count("\n") + 1), float(len(nums)), float(len(set(nums))),
            float(digits), math.log10(abs(value)) if value else rt["log10_zero"],
            float(bool(re.search(r"[A-Za-z]", re.sub(r"\\[A-Za-z]+", "", ans)))), float(reason.count("\\text{")),
            float(reason.count("=")), float(sum(reason.count(o) for o in rt["operators"])),
            float(reason.count("\\boxed") + 1),
            float(sum(low.count(h) for h in rt["hesitation"])), box_start / max(len(completion), 1), float(depth)]


class RichCandidate:
    """R6-5 (a): A5's per-field TF-IDF with D167's n-gram range, plus the hand features (standardized for the linear
    model, raw for the GBT); A5's models."""

    def __init__(self, kind: str, hp, a5: dict, rt: dict):
        from physcal.eval import nested_baselines as nb

        self.text_cfg = a5["text"] | {"ngram_range": rt["ngram_range"]}
        self.kind, self.hp, self.seed, self.nb = kind, hp, a5["seed"], nb

    def _X(self, recs, H, fit: bool):
        import scipy.sparse as sp
        from sklearn.preprocessing import StandardScaler

        if fit:
            self.feat = self.nb.Featurizer("problem_answer_reasoning", self.text_cfg, dense=self.kind == "gbt").fit(recs)
            self.scale = StandardScaler().fit(H)
        T = self.feat.transform(recs)
        if self.kind == "gbt":
            return np.hstack([T, H])
        return sp.hstack([T, sp.csr_matrix(self.scale.transform(H))]).tocsr()

    def fit(self, recs, H, y):
        self.model = self.nb._model(self.kind, "p", self.hp, self.seed).fit(self._X(recs, H, True), y)
        return self

    def raw(self, recs, H):
        return self.model.predict_proba(self._X(recs, H, False))[:, 1]


def crossfit_rich(recs, H, y, groups, outer: int, inner: int, a5: dict, rt: dict) -> tuple[np.ndarray, list]:
    """D164's cross-fit (outer folds = the probe's, inner selection over A5's grid by pooled AUROC) with RichCandidate."""
    from sklearn.metrics import roc_auc_score

    grid, p, chosen = AU.text_grid(a5), np.full(len(y), np.nan), []
    for tr, te in AU._folds(groups, outer):
        r_tr, H_tr, g_tr = [recs[i] for i in tr], H[tr], [groups[i] for i in tr]
        scores = []
        for kind, hp in grid:
            oof = np.zeros(len(tr))
            for a, b in AU._folds(g_tr, inner):
                c = RichCandidate(kind, hp, a5, rt).fit([r_tr[i] for i in a], H_tr[a], y[tr][a])
                oof[b] = c.raw([r_tr[i] for i in b], H_tr[b])
            scores.append(float(roc_auc_score(y[tr], oof)))
        j = max(range(len(grid)), key=lambda i: (scores[i], -i))
        c = RichCandidate(*grid[j], a5, rt).fit(r_tr, H_tr, y[tr])
        p[te] = c.raw([recs[i] for i in te], H[te])
        chosen.append({"kind": grid[j][0], "hp": grid[j][1], "inner_cv_auroc": scores[j], "table": scores})
    return p, chosen


def refit_fold_probes(layers, index: list[dict], folds_info: list[dict], cfg: dict):
    """The cached probe's five fold models, refit with its recorded (layer, C) on the same training folds; yields
    (held-out rows, layer, model)."""
    from physcal.eval import x2_analysis as XA

    y = np.array([r["correct"] for r in index], dtype=int)
    groups = ["|".join(map(str, r["key"])) for r in index]
    for (tr, te), f in zip(AU._folds(groups, cfg["scale"]["probe"]["folds"]), folds_info):
        yield te, f["layer"], XA.fit_probe(layers[f["layer"]][tr], y[tr], f["C"], cfg["scale"]["probe"]["max_iter"])


def analyze_controls(model: str, cfg: dict) -> dict:
    sc, rc, rt = cfg["scale"], cfg["r6"], cfg["r6"]["rich_text"]
    rows = kaggle_rows(model, cfg)
    items = probe_order_items(model, rows, cfg)
    corr = {_key_sample(it): it["correct"] for it in items}
    index, _, cache = A.probe_inputs(model, "hidden", corr, cfg)
    probe_cache = json.loads(cache.read_text())
    probe = {(tuple(k), s): v for k, s, v in probe_cache["conf"]}
    groups_k = by_problem(items)
    keys = list(groups_k)
    conf = {"probe": {it["id"]: probe[_key_sample(it)] for it in items}}
    res: dict = {"n": len(items), "n_problems": len(keys)}
    a5, tc = load_config(cfg["audit"]["text"]["a5_config"]), cfg["audit"]["text"]
    y = np.array([it["correct"] for it in items], dtype=int)
    groups = [_kid(it["key"]) for it in items]
    recs = [AU.text_fields(it["text"], it["completion"]) for it in items]
    value = {(tuple(r["key"]), r["sample"]): r["value"] for r in rows}
    H = np.array([hand_features(rec, it["completion"], value[_key_sample(it)], it["key"][1], it["n_tokens"], rt)
                  for rec, it in zip(recs, items)])
    def rich_fit():
        p, chosen = crossfit_rich(recs, H, y, groups, sc["probe"]["folds"], tc["inner_folds"], a5, rt)
        return {it["id"]: float(v) for it, v in zip(items, p)}, chosen
    conf["rich_text"], res["rich_text_folds"] = cached(cfg, f"rich_text_{model}", [
        f"rich_text_{model}", [it["id"] for it in items], recs, H, rt, tc, y.tolist()], rich_fit)
    text = AU.text_confidence(model, cfg, fit=False, rows=rows)
    conf["text"] = {it["id"]: text[_key_sample(it)] for it in items}
    ao_recs = [{k: r[k] for k in ("problem", "answer")} | {"reasoning": ""} for r in recs]

    def text_ao_fit():
        p, folds = AU.crossfit_text(ao_recs, y, groups, sc["probe"]["folds"], "problem_answer", tc["inner_folds"], a5)
        return {it["id"]: float(v) for it, v in zip(items, p)}, folds
    conf["text_answer_only"], res["text_answer_only_folds"] = cached(cfg, f"text_answer_only_{model}", [
        f"text_answer_only_{model}", [it["id"] for it in items], ao_recs, tc, y.tolist()], text_ao_fit)
    ao = load_set(f"answer_only_{model}", cfg)
    layers, ids, _ = load_states(f"answer_only_{model}", model, "p2", cfg)
    if ids != [it["id"] for it in ao] or [it["orig_id"] for it in ao] != [it["id"] for it in items]:
        raise SystemExit(f"answer_only_{model}: not in the probe's order")
    c, res["answer_only_probe"] = cached(cfg, f"answer_only_probe_{model}",
                                         _probe_inputs(f"answer_only_probe_{model}", layers, ao, cfg),
                                         lambda: crossfit_items(layers, ao, cfg))
    conf["probe_answer_only"] = {it["orig_id"]: c[it["id"]] for it in ao}
    # (d) the reasoning swap: the cached probe's fold models on the swapped states
    sw = load_set(f"swap_{model}", cfg)
    sw_layers, sw_ids, _ = load_states(f"swap_{model}", model, "p2", cfg)
    if sw_ids != [it["id"] for it in sw] or [it["orig_id"] for it in sw] != [it["id"] for it in items]:
        raise SystemExit(f"swap_{model}: not in the probe's order")
    orig_layers, _ = F.load_hidden(model, cfg)
    p_orig, p_swap = np.full(len(items), np.nan), np.full(len(items), np.nan)
    for te, layer, f in refit_fold_probes(orig_layers, index, probe_cache["info"]["folds"], cfg):
        p_orig[te], p_swap[te] = f(orig_layers[layer][te]), f(sw_layers[layer][te])
    cached_vals = np.array([conf["probe"][it["id"]] for it in items])
    gap = float(np.abs(p_orig - cached_vals).max())
    res["swap_refit_gap"] = gap
    if gap > rc["swap_refit_atol"]:
        raise SystemExit(f"{model}: refit fold probes differ from the cache by {gap} (refit under the cached settings)")
    conf["probe_swap"] = {it["id"]: float(v) for it, v in zip(items, p_swap)}
    sw_recs = [AU.text_fields(it["text"], it["completion"]) for it in sw]
    tf = json.loads((AU.audit_dir(cfg) / f"text_{model}.json").read_text())["info"]["folds"]
    from physcal.eval import nested_baselines as nb

    t_swap = np.full(len(items), np.nan)
    for (tr, te), fsel in zip(AU._folds(groups, sc["probe"]["folds"]), tf):
        cand = nb.Candidate(tc["feature_set"], fsel["kind"], "p", fsel["hp"], a5).fit([recs[i] for i in tr], y[tr])
        t_swap[te] = cand.raw([sw_recs[i] for i in te])
    conf["text_swap"] = {it["id"]: float(v) for it, v in zip(items, t_swap)}
    delta = p_swap - cached_vals
    res["swap_movement"] = {"median_abs": float(np.median(np.abs(delta))),
                            "mean_signed_correct": float(delta[y == 1].mean()),
                            "mean_signed_incorrect": float(delta[y == 0].mean())}
    rel = cfg["audit"]["length_match_rel"]
    lm = lambda a, b: AU.length_matched(a["n_tokens"], b["n_tokens"], rel)  # noqa: E731
    out = {}
    for variant, ok in (("all pairs", None), ("length-matched", lm)):
        ks = [k for k, n in zip(keys, n_pairs(groups_k, keys, ok)) if n > 0]
        pa = {c_: per_problem(groups_k, ks, v, ok) for c_, v in conf.items()}
        d = lambda a, b: A.paired(pa[a], pa[b], sc["bootstrap"], sc["seed"])  # noqa: E731
        out[variant] = {"n_problems": len(ks), "readouts": {c_: _summ(v, sc) for c_, v in pa.items()},
                        "probe - rich_text": d("probe", "rich_text"), "rich_text - text": d("rich_text", "text"),
                        "probe - probe_answer_only": d("probe", "probe_answer_only"),
                        "probe - probe_swap": d("probe", "probe_swap")}
    res["by_variant"] = out
    a = out["all pairs"]
    res["reading"] = {
        "(a) the probe reads more than a rich text reader": _side(a["probe - rich_text"]) > 0
        and _side(out["length-matched"]["probe - rich_text"]) > 0,
        "(c)": ("the reasoning carries the signal" if _side(a["probe - probe_answer_only"]) > 0 else
                "the answer alone carries it" if a["readouts"]["probe_answer_only"]["ci95"][0] > 0.5 else "neither"),
        "(d)": ("the probe tracks the answer" if a["readouts"]["probe_swap"]["ci95"][0] > 0.5 else
                "it tracks the reasoning" if a["readouts"]["probe_swap"]["ci95"][1] < 0.5 else "neither")}
    return res


# ------------------------------------------------------------------------------------------- the analysis
def analyze(cfg: dict | None = None) -> dict:
    """R6-3, R6-4, then R6-5 per model; results.json is rewritten after each, and every fit is cached (`cached`), so a
    re-run after a failure keeps what finished."""
    cfg = cfg or F._config()
    t0 = time.time()
    res: dict = {"decision": "D167 (exploratory, paper #2)", "threads": AU._threads()}
    path = r6_dir(cfg) / "results.json"

    def save():
        res["seconds"] = time.time() - t0
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1, default=float))
        tmp.rename(path)

    for name, run in (("R6-3", lambda: analyze_bank(cfg)), ("R6-4", lambda: analyze_factorial(cfg)),
                      ("R6-5 7b", lambda: analyze_controls("7b", cfg)), ("R6-5 1.5b", lambda: analyze_controls("1.5b", cfg))):
        print(f"{name}: start ({time.time() - t0:.0f}s)", flush=True)
        res[name] = run()
        save()
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["items", "estimate", "stages", "states", "yes", "analyze",
                                               "r6_4_pair_weighted"])
    ap.add_argument("--set", help="states / yes: the item set")
    ap.add_argument("--reader", choices=READERS, help="states / yes: the reader")
    ap.add_argument("--cap", type=float, help="states / yes: the VRAM cap in GiB")
    args = ap.parse_args()
    cfg = F._config()
    if args.action == "items":
        print(json.dumps(build_items(cfg), indent=1))
    elif args.action == "estimate":
        e = estimate(cfg)
        print(json.dumps({k: v for k, v in e.items() if k != "stages"}, indent=1))
    elif args.action == "stages":
        print("\n".join(" ".join(s) for s in GPU_STAGES))
    elif args.action == "states":
        print(json.dumps(run_states(args.set, args.reader, args.cap, cfg), indent=1))
    elif args.action == "yes":
        print(json.dumps(run_yes(args.set, args.reader, args.cap, cfg), indent=1))
    elif args.action == "r6_4_pair_weighted":
        analyze_factorial_pair_weighted(cfg)
    else:
        analyze(cfg)


if __name__ == "__main__":
    main()

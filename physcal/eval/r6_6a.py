"""D175 (EXPLORATORY, paper #2): R6-6a on Kaggle "GPU T4 x2": can what a probe reads be moved into a cheap verbal
confidence, beyond what direct labels and surface text give? Rules: docs/exploratory/DECISIONS_FOLLOWUP.md D175, D171.

Locally (CPU):
    uv run python -m physcal.eval.r6_6a pool --mode full|smoke      # build and freeze a pool, its splits and manifest
    uv run python -m physcal.eval.r6_6a ingest --mode M --zip Z     # D171 rule 8: the zip's checks, a re-grade, the CIs
On Kaggle (the notebook runs these with its venv's python, the repo on PYTHONPATH):
    python -m physcal.eval.r6_6a download --mode M --model-dir D
    python -m physcal.eval.r6_6a restore --mode M --out O --src /kaggle/input
    python -m physcal.eval.r6_6a run --mode M --out O --model-dir D --session-start T      # every stage, in order
    python -m physcal.eval.r6_6a pack --mode M --out O

`run` drives the stages S1-S6 as child processes, one per GPU (`worker` / `sample`), and stops itself at the deadline; a
later session with the output attached resumes. Mode smoke: the same code on 48 problems of its own, for rates only.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.eval import followup_audit as AU
from physcal.eval import followup_kaggle as K
from physcal.eval import followup_r6 as R6
from physcal.eval import followup_r6_controls as RC
from physcal.eval import followup_scale as F
from physcal.eval import followup_scale_analysis as A
from physcal.eval import kaggle_stages as S

S1, S2, S3, S4, S5 = "s1_sample", "s2_read", "s3_teachers", "s4_train", "s5_score"
EXIT_STOP = 4                  # a stop rule fired: a non-finite value
ONE_THREAD = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}   # one per job (D97)
FOLDS_PROBE, FOLDS_RICH = "probe_fold", "rich_fold"
FINALS = ("probe", "rich", "answer_only")
OK_NAMES = (r"results\.json|status\.json|scores\.jsonl\.gz|swap_scores\.jsonl\.gz|completions\.jsonl\.gz|environment\.json"
            r"|throughput\.json|plan\.json|config/[a-z0-9_]+\.yaml|logs/[A-Za-z0-9_.-]+\.log"
            r"|timing/[a-z0-9_]+\.jsonl|smoke/[a-z_]+\.json")


# ------------------------------------------------------------------------------------------- settings
def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out


def mode_config(mode: str = "full") -> dict:
    rc = load_config("r6_6a")
    rc = _merge(rc, load_config("r6_6a_smoke")) if mode == "smoke" else rc
    return rc | {"mode": mode}


def scale_config(rc: dict) -> dict:
    """followup.yaml overlaid so that followup_kaggle / followup_scale read this pool under the model key rc['name']."""
    c = copy.deepcopy(load_config("followup"))
    s, name = c["scale"], rc["name"]
    s["pool_config"] = "r6_6a" if rc["mode"] == "full" else "r6_6a_smoke"
    s["chunk_size"] = rc["pool"]["chunk_size"]
    s["dev_indices"] = [rc["pool"]["n"], rc["pool"]["n"]]
    s["models"][name] = s["models"][rc["model_from"]]
    s["kaggle_models"][name] = s["kaggle_models"][rc["model_from"]] | {"zip_name": rc["results"]["zip_name"]}
    s["kaggle"]["sampling_hours"] = rc["session"]["hours"]
    s["kaggle"]["rate_margin"] = rc["session"]["rate_margin"]
    return c


def _commit() -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def pins(rc: dict) -> dict:
    """What a run may not change between sessions: the commit, the configs and the pool (D171 rule 3)."""
    names = ["r6_6a", "followup"] + (["r6_6a_smoke"] if rc["mode"] == "smoke" else [])
    return {"decision": rc["decision"], "mode": rc["mode"], "commit": _commit(),
            "configs": {n: S.file_sha256(REPO_ROOT / "configs" / f"{n}.yaml") for n in names},
            "pool": S.file_sha256(REPO_ROOT / rc["candidates"]["file"])}


@dataclass
class Ctx:
    root: Path                       # the run's directory
    rc: dict
    scfg: dict
    run: S.Run
    model_dir: Path | None = None
    device: str = "cuda"
    load: object = None              # () -> the base model; default: fp16 from model_dir
    log: object = print
    _items: dict = field(default_factory=dict)


def make_ctx(mode: str, out: Path, session_start: float, model_dir: Path | None, worker: str = "main", device: str = "cuda",
             rc: dict | None = None) -> Ctx:
    rc = rc or mode_config(mode)
    root = Path(out) / rc["name"]
    run = S.Run(root, pins(rc), session_start, rc["session"]["hours"], rc["session"]["rate_margin"], worker)
    return Ctx(root, rc, scale_config(rc), run, model_dir, device)


# ------------------------------------------------------------------------------------------- the pool (CPU, local)
def question_kind(text: str) -> str:
    """The first word of the question sentence: the last paragraph of the native text."""
    return F.native_text(text).rsplit("\n", 1)[-1].split()[0]


def assign_splits(texts: list[str], pc: dict) -> dict[str, list[int]]:
    """D175's split by problem: every held-out question template; the rest permuted, first n_train and n_validation."""
    held = [i for i, t in enumerate(texts) if question_kind(t) == pc["held_out_question"]]
    rest = [i for i in range(len(texts)) if i not in set(held)]
    perm = [rest[j] for j in np.random.default_rng(pc["split_seed"]).permutation(len(rest))]
    a, b = pc["n_train"], pc["n_train"] + pc["n_validation"]
    return {"train": sorted(perm[:a]), "validation": sorted(perm[a:b]), "id_test": sorted(perm[b:]), "held_out": held}


def _frozen_problems() -> tuple[set, set]:
    seeds, texts = set(), set()
    for f in (REPO_ROOT / "data" / "eval_sets").rglob("*.jsonl"):
        if f.name.startswith("r6_6a"):
            continue
        for line in f.open():
            r = json.loads(line)
            seeds |= {r[k] for k in ("seed",) if k in r}
            if "text" in r:
                texts.add(r["text"])
    return seeds, texts


def manifest_path(rc: dict) -> Path:
    p = REPO_ROOT / rc["candidates"]["file"]
    return p.with_name(p.stem + "_manifest.json")


def build_pool(mode: str) -> dict:
    """The frozen pool: problem i is cell i mod 12, seed offset + i; disjoint from every frozen set (D175). A different
    freeze is refused."""
    from physcal.env.generator import generate_problem, generator_version

    rc = mode_config(mode)
    pc, env = rc["pool"], load_config("env")
    cells = [(f, d) for f in pc["families"] for d in pc["depths"]]
    probs = [generate_problem(*cells[i % len(cells)], pc["seed_offset"] + i, split="train", cfg=env).to_dict()
             for i in range(pc["n"])]
    seeds, texts = _frozen_problems()
    if {p["seed"] for p in probs} & seeds or {p["text"] for p in probs} & texts:
        raise SystemExit("the pool shares a seed or a text with a frozen set")
    path = REPO_ROOT / rc["candidates"]["file"]
    version = generator_version(env)
    if path.parent.name != version:
        raise SystemExit(f"{path} is not under the generator version's dir {version}")
    blob = "".join(json.dumps(p, sort_keys=True) + "\n" for p in probs)
    sp = assign_splits([p["text"] for p in probs], pc)
    manifest = {"decision": "D175", "generator_version": version, "n_problems": len(probs),
                "seeds": [pc["seed_offset"], pc["seed_offset"] + pc["n"] - 1], "sha256": hashlib.sha256(blob.encode()).hexdigest(),
                "split_sizes": {k: len(v) for k, v in sp.items()}, "splits": sp}
    mp = manifest_path(rc)
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise SystemExit(f"{mp}: a different freeze exists")
    path.write_text(blob)
    mp.write_text(json.dumps(manifest, indent=0))
    return {k: v for k, v in manifest.items() if k != "splits"}


def split_of(rc: dict) -> dict[int, str]:
    sp = json.loads(manifest_path(rc).read_text())["splits"]
    return {i: name for name, idx in sp.items() for i in idx}


# ------------------------------------------------------------------------------------------- answers
def samples_dir(ctx: Ctx) -> Path:
    return K.samples_dir(ctx.root, ctx.rc["name"])


def sampled_rows(ctx: Ctx) -> list[dict]:
    return F.sampled_prefix(ctx.rc["name"], ctx.scfg, samples_dir(ctx))


def sampling_done(ctx: Ctx) -> bool:
    d = samples_dir(ctx)
    return d.exists() and len(sampled_rows(ctx)) == ctx.rc["pool"]["n"] * ctx.scfg["scale"]["generation"]["k"]


def items_of(rows: list[dict], rc: dict, split: dict[int, str]) -> list[dict]:
    """Every answered sample as an item, in pool order (D175: a value from D154's grade, not truncated)."""
    out = []
    for r in rows:
        if r["value"] is None or r["truncated"]:
            continue
        it = R6.native_item(rc["name"], r | {"text": r["problem"]["text"]})
        out.append(it | {"split": split[r["index"]], "value": r["value"], "index": r["index"]})
    return out


def items(ctx: Ctx) -> list[dict]:
    if "all" not in ctx._items:
        ctx._items["all"] = items_of(sampled_rows(ctx), ctx.rc, split_of(ctx.rc))
    return ctx._items["all"]


def in_split(its: list[dict], *names: str) -> list[dict]:
    return [it for it in its if it["split"] in names]


def train_items(ctx: Ctx) -> list[dict]:
    return in_split(items(ctx), "train")[:ctx.rc["train"]["items"]]


def mixed_problems(its: list[dict]) -> list[tuple]:
    return [k for k, v in R6.by_problem(its).items() if {x["correct"] for x in v} == {True, False}]


def swap_sets(its: list[dict], rc: dict, tol: float) -> dict[str, list[dict]]:
    """D175's three swap sets on the first ID-test problems with both classes, in pool order. own_cut is each answer's
    completion cut after its box (D174): a swap's comparison shares its tail and brace token."""
    idt = in_split(its, "id_test")
    keep = set(mixed_problems(idt)[:rc["swap_problems"]])
    sel = [it for it in idt if tuple(it["key"]) in keep]
    value = {it["id"]: it["value"] for it in sel}
    diff = [t for t in RC.same_label_set(sel, value, tol) if t["kind"] == "diff_value"]
    cut = [it | {"id": it["id"] + "|cut", "completion": it["completion"][:it["p2"] + 1], "orig_id": it["id"]} for it in sel]
    return {"sw_opposite": R6.swap_set(rc["name"], sel), "sw_diff": diff, "sw_own_cut": cut}


# ------------------------------------------------------------------------------------------- reading (GPU)
@dataclass
class Reader:
    tok: object
    yes: list[int]
    no: list[int]
    pad_id: int
    marker: str
    W: object = None                 # the yes and no rows of the head, float32


def make_reader(tok, model, scfg: dict, device: str) -> Reader:
    from physcal.eval import x4_verify as X4

    rd = scfg["scale"]["reading"]
    r = Reader(tok, X4.token_ids(tok, rd["yes_tokens"]), X4.token_ids(tok, rd["no_tokens"]), tok.pad_token_id, rd["marker"])
    _, head = core_of(model)
    r.W = head.weight[r.yes + r.no].detach().float().to(device)
    return r


def core_of(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.get_output_embeddings()


def _pad(ids: list[list[int]], pad_id: int):
    import torch

    n = max(len(x) for x in ids)
    inp = torch.full((len(ids), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids), n), dtype=torch.long)
    for j, x in enumerate(ids):
        inp[j, :len(x)] = torch.tensor(x)
        mask[j, :len(x)] = 1
    return inp, mask


def _log_odds(h, r: Reader):
    """The yes-vs-no log-odds from final states: float32 throughout, only the head's yes and no rows (D89)."""
    import torch

    lg = h.float() @ r.W.T
    n = len(r.yes)
    return torch.logsumexp(lg[:, :n], 1) - torch.logsumexp(lg[:, n:], 1)


def forward_log_odds(model, ids: list[list[int]], r: Reader, device: str, pos: list[int] | None = None):
    """One forward over right-padded contexts: the log-odds at each last token and, with `pos`, every layer's state at
    those positions (n, layers, hidden). Gradients flow unless the caller is in inference mode."""
    import torch

    core, _ = core_of(model)
    inp, mask = _pad(ids, r.pad_id)
    out = core(input_ids=inp.to(device), attention_mask=mask.to(device), output_hidden_states=pos is not None, use_cache=False)
    rows = torch.arange(len(ids), device=device)
    lo = _log_odds(out.last_hidden_state[rows, mask.sum(1).to(device) - 1], r)
    if pos is None:
        return lo, None
    idx = torch.tensor(pos, device=device)
    return lo, torch.stack([h[rows, idx] for h in out.hidden_states], 1)


def batches(lengths: list[int], budget: int) -> list[list[int]]:
    """Indices, longest first, in batches of at most `budget` padded tokens (a longer item goes alone)."""
    out: list[list[int]] = []
    for i in sorted(range(len(lengths)), key=lambda j: (-lengths[j], j)):
        if out and (len(out[-1]) + 1) * lengths[out[-1][0]] <= budget:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def context(it: dict, r: Reader, scfg: dict, verify: str) -> tuple[list[int], int]:
    """D151's reading context: its token ids, and the index of the token holding the box's closing brace."""
    from physcal.eval import x2_hidden as XH

    tok = r.tok
    msgs = F.messages(it["text"], scfg)
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    conv = msgs + [{"role": "assistant", "content": it["completion"]}, {"role": "user", "content": verify}]
    s = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True) + r.marker
    if not s.startswith(prompt + it["completion"]):
        raise S.StageStop("the chat template changed the completion")
    enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], XH.token_at(enc["offset_mapping"], len(prompt) + it["p2"])


def _finite(x, what: str) -> None:
    if not np.isfinite(x).all():
        raise S.StageStop(f"non-finite {what}")


def read_log_odds(model, its: list[dict], r: Reader, scfg: dict, verify: str, budget: int, device: str,
                  states: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    """Log-odds per item and, with `states`, float16 states at the brace (n, layers, hidden), in item order."""
    import torch

    ctx = [context(it, r, scfg, verify) for it in its]
    ids = [c[0] for c in ctx]
    lo, st = np.zeros(len(its)), None
    for b in batches([len(x) for x in ids], budget):
        with torch.inference_mode():
            v, s = forward_log_odds(model, [ids[i] for i in b], r, device, [ctx[i][1] for i in b] if states else None)
        lo[b] = v.double().cpu().numpy()
        if states:
            s = s.float().cpu().numpy().astype(np.float16)
            st = np.zeros((len(its),) + s.shape[1:], np.float16) if st is None else st
            st[b] = s
    _finite(lo, "log-odds")
    if states:
        _finite(st, "state")
    return lo, st


def answer_only_states(model, its: list[dict], r: Reader, scfg: dict, budget: int, device: str) -> np.ndarray:
    """D174 (h): the state at the closing brace of the frame with the completion replaced by its box alone."""
    import torch

    from physcal.eval import x2_hidden as XH

    tok, ids, pos = r.tok, [], []
    for a in R6.answer_only_set("", its):
        p = tok.apply_chat_template(F.messages(a["text"], scfg), tokenize=False, add_generation_prompt=True)
        enc = tok(p + a["completion"], add_special_tokens=False, return_offsets_mapping=True)
        ids.append(enc["input_ids"])
        pos.append(XH.token_at(enc["offset_mapping"], len(p) + a["p2"]))
    st = None
    for b in batches([len(x) for x in ids], budget):
        with torch.inference_mode():
            _, s = forward_log_odds(model, [ids[i] for i in b], r, device, [pos[i] for i in b])
        s = s.float().cpu().numpy().astype(np.float16)
        st = np.zeros((len(its),) + s.shape[1:], np.float16) if st is None else st
        st[b] = s
    _finite(st, "answer-only state")
    return st


def default_load(ctx: Ctx):
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(str(ctx.model_dir), dtype=torch.float16, attn_implementation="sdpa").to(ctx.device)


def _tok(ctx: Ctx):
    from transformers import AutoTokenizer

    return ctx.load.tok if hasattr(ctx.load, "tok") else AutoTokenizer.from_pretrained(str(ctx.model_dir))


def _model(ctx: Ctx):
    return ctx.load() if ctx.load else default_load(ctx)


# ------------------------------------------------------------------------------------------- S2: read
def part_keys(n: int, size: int) -> list[str]:
    return [f"part_{i:04d}.npz" for i in range((n + size - 1) // size)]


def stage_read(ctx: Ctx, worker: int = 0, workers: int = 1) -> bool:
    """P(yes) and every layer's state at the brace in one pass, plus the short answer-only pass, in saved parts."""
    its, size = items(ctx), ctx.rc["readout"]["part_size"]
    keys = part_keys(len(its), size)
    mine = keys[worker::workers]
    if all(ctx.run.has(S2, k) for k in mine):
        return True
    model = _model(ctx).eval()
    tok = _tok(ctx)
    r = make_reader(tok, model, ctx.scfg, ctx.device)
    verify, budget = F.verify_text(ctx.scfg), ctx.rc["readout"]["token_budget"]

    def write(key: str, tmp: Path) -> None:
        i = int(key[5:9]) * size
        part = its[i:i + size]
        lo, st = read_log_odds(model, part, r, ctx.scfg, verify, budget, ctx.device, states=True)
        ao = answer_only_states(model, part, r, ctx.scfg, budget, ctx.device)
        with tmp.open("wb") as fh:
            np.savez(fh, ids=json.dumps([it["id"] for it in part]), log_odds=lo, states=st, answer_only=ao)
    return ctx.run.chunks(S2, mine, write, ctx.rc["estimates_s"]["read_part"], log=ctx.log)


def read_table(ctx: Ctx) -> dict[str, float]:
    """Item id -> the base model's log-odds (arm A), from S2's parts."""
    out: dict = {}
    for key in part_keys(len(items(ctx)), ctx.rc["readout"]["part_size"]):
        z = np.load(ctx.run.path(S2, key))
        out |= dict(zip(json.loads(str(z["ids"])), z["log_odds"].tolist()))
    return out


def load_layers(ctx: Ctx, ids: list[str], field_name: str = "states", layer: int | None = None) -> list[np.ndarray]:
    """Every layer's float16 states of `ids`, in that order; with `layer`, that layer alone (a list of one)."""
    want = {i: n for n, i in enumerate(ids)}
    arr = None
    for key in part_keys(len(items(ctx)), ctx.rc["readout"]["part_size"]):
        z = np.load(ctx.run.path(S2, key))
        pid = json.loads(str(z["ids"]))
        sel = [(j, want[i]) for j, i in enumerate(pid) if i in want]
        if not sel:
            continue
        st = z[field_name] if layer is None else z[field_name][:, layer:layer + 1]
        arr = np.zeros((len(ids),) + st.shape[1:], np.float16) if arr is None else arr
        arr[[b for _, b in sel]] = st[[a for a, _ in sel]]
    return [np.ascontiguousarray(arr[:, k, :]) for k in range(arr.shape[1])]


# ------------------------------------------------------------------------------------------- S3: teachers (CPU)
def teacher_keys(rc: dict, folds: int) -> list[str]:
    return ([f"{FOLDS_PROBE}_{i}.json" for i in range(folds)] + [f"{FOLDS_RICH}_{i}.json" for i in range(folds)]
            + [f"final_{n}.json" for n in FINALS])


def _probe_settings(ctx: Ctx) -> dict:
    pc = dict(ctx.scfg["scale"]["probe"])
    pc.pop("even_layers_only")
    pc["n_jobs"] = ctx.rc["teachers"]["n_jobs"] or len(os.sched_getaffinity(0))
    return pc


def _rich_inputs(its: list[dict], ctx: Ctx) -> tuple[list[dict], np.ndarray]:
    rt = ctx.scfg["r6"]["rich_text"]
    recs = [AU.text_fields(it["text"], it["completion"]) for it in its]
    return recs, np.array([R6.hand_features(rec, it["completion"], it["value"], it["key"][1], it["n_tokens"], rt)
                           for rec, it in zip(recs, its)])


def probe_fit(layers, y, groups, pc: dict):
    """x2_analysis's selection on the given answers, then a fit: (layer, scorer, info)."""
    from physcal.eval import x2_analysis as XA

    sel = XA.select_probe(layers, y, groups, pc)
    f = XA.fit_probe(layers[sel["layer"]], y, sel["C"], pc["max_iter"])
    return sel["layer"], f, {"layer": sel["layer"], "C": sel["C"], "inner_cv_auroc": sel["cv_auroc"]}


def stage_teachers(ctx: Ctx) -> bool:
    """D175's targets (out-of-fold on the train answers) and the references (refit on all of them, scored on validation
    and test), one chunk per fold and per final fit."""
    all_its = items(ctx)
    tr = in_split(all_its, "train")
    ev = in_split(all_its, "validation", "id_test", "held_out")
    y = np.array([it["correct"] for it in tr], dtype=int)
    groups = [R6._kid(it["key"]) for it in tr]
    pc = _probe_settings(ctx)
    folds = AU._folds(groups, pc["folds"])
    tc, a5 = ctx.scfg["audit"]["text"], load_config(ctx.scfg["audit"]["text"]["a5_config"])
    rt, cache = ctx.scfg["r6"]["rich_text"], {}

    def layers(name: str = "states"):
        if name not in cache:
            cache[name] = load_layers(ctx, [it["id"] for it in tr], name)
        return cache[name]

    def rich():
        if "rich" not in cache:
            cache["rich"] = _rich_inputs(tr, ctx)
        return cache["rich"]

    def write(key: str, tmp: Path) -> None:
        if key.startswith(FOLDS_PROBE):
            a, b = folds[int(key.rsplit("_", 1)[1].split(".")[0])]
            L = layers()
            k, f, info = probe_fit([x[a] for x in L], y[a], [groups[i] for i in a], pc)
            res = {"ids": [tr[i]["id"] for i in b], "p": f(L[k][b]).tolist(), "info": info}
        elif key.startswith(FOLDS_RICH):
            a, b = folds[int(key.rsplit("_", 1)[1].split(".")[0])]
            recs, H = rich()
            c, info = R6.rich_fit([recs[i] for i in a], H[a], y[a], [groups[i] for i in a], tc["inner_folds"], a5, rt)
            res = {"ids": [tr[i]["id"] for i in b], "p": c.raw([recs[i] for i in b], H[b]).tolist(), "info": info}
        elif key == "final_rich.json":
            recs, H = rich()
            c, info = R6.rich_fit(recs, H, y, groups, tc["inner_folds"], a5, rt)
            er, eH = _rich_inputs(ev, ctx)
            res = {"ids": [it["id"] for it in ev], "p": c.raw(er, eH).tolist(), "info": info}
        else:
            name = "states" if key == "final_probe.json" else "answer_only"
            k, f, info = probe_fit(layers(name), y, groups, pc)
            res = {"ids": [it["id"] for it in ev], "p": f(load_layers(ctx, [it["id"] for it in ev], name, k)[0]).tolist(),
                   "info": info}
        tmp.write_text(json.dumps(res))
    return ctx.run.chunks(S3, teacher_keys(ctx.rc, pc["folds"]), write, ctx.rc["estimates_s"]["teacher"], log=ctx.log)


def teacher_values(ctx: Ctx, prefix: str, folds: int) -> dict[str, float]:
    out: dict = {}
    for i in range(folds):
        c = json.loads(ctx.run.path(S3, f"{prefix}_{i}.json").read_text())
        out |= dict(zip(c["ids"], c["p"]))
    return out


def shuffled_within_problems(its: list[dict], t: np.ndarray, seed: int) -> np.ndarray:
    """Arm E: each problem's targets permuted among its answers."""
    rng, out = np.random.default_rng(seed), t.copy()
    by: dict = {}
    for i, it in enumerate(its):
        by.setdefault(tuple(it["key"]), []).append(i)
    for idx in by.values():
        out[idx] = t[rng.permutation(idx)]
    return out


def targets(ctx: Ctx, arm: str, its: list[dict]) -> np.ndarray:
    folds = ctx.scfg["scale"]["probe"]["folds"]
    if arm == "B":
        return np.array([float(it["correct"]) for it in its])
    v = teacher_values(ctx, FOLDS_RICH if arm == "D" else FOLDS_PROBE, folds)
    t = np.array([v[it["id"]] for it in its])
    return shuffled_within_problems(its, t, ctx.rc["shuffle_seed"]) if arm == "E" else t


# ------------------------------------------------------------------------------------------- S4: LoRA (GPU)
def run_names(rc: dict, arms: tuple[str, ...] | None = None) -> list[str]:
    return [f"{a}_{s}" for a, seeds in rc["arms"].items() if arms is None or a in arms for s in seeds]


def lora_config(rc: dict):
    from peft import LoraConfig

    t = rc["train"]
    return LoraConfig(r=t["rank"], lora_alpha=t["alpha"], lora_dropout=t["dropout"], target_modules=t["target_modules"],
                      bias="none", task_type="CAUSAL_LM")


def lr_scale(step: int, total: int, warmup: int) -> float:
    return (step + 1) / warmup if step < warmup else max(0.0, (total - step) / max(1, total - warmup))


def train_batches(n: int, seed: int, batch: int, epochs: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [e[i:i + batch] for e in (rng.permutation(n) for _ in range(epochs)) for i in range(0, n, batch)]


def train_one(ctx: Ctx, its: list[dict], t: np.ndarray, seed: int) -> dict:
    """D175's training: LoRA on the frozen fp16 base; BCE of the log-odds against the target. Returns the adapter and
    its log. A non-finite loss raises StageStop."""
    import torch
    from peft import get_peft_model
    from peft.utils import get_peft_model_state_dict

    tc, dev = ctx.rc["train"], ctx.device
    cuda = dev.startswith("cuda")
    base = _model(ctx)
    tok = _tok(ctx)
    torch.manual_seed(seed)
    model = get_peft_model(base, lora_config(ctx.rc))
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    r = make_reader(tok, base, ctx.scfg, dev)
    verify = F.verify_text(ctx.scfg)
    ids = [context(it, r, ctx.scfg, verify)[0] for it in its]
    steps = train_batches(len(its), seed, tc["batch_size"], tc["epochs"])
    warmup = math.ceil(tc["warmup_share"] * len(steps))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_scale(s, len(steps), warmup))
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    tt = torch.tensor(t, dtype=torch.float32, device=dev)
    log, t0, skipped = [], time.time(), 0
    for n, batch in enumerate(steps):
        run_loss = 0.0
        for mb in batches([len(ids[i]) for i in batch], tc["token_budget"]):
            sel = [int(batch[j]) for j in mb]
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                lo, _ = forward_log_odds(model, [ids[i] for i in sel], r, dev)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lo, tt[sel], reduction="sum") / len(batch)
            if not torch.isfinite(loss):
                raise S.StageStop(f"non-finite loss at step {n}")
            scaler.scale(loss).backward()
            run_loss += float(loss.detach())
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, tc["max_grad_norm"])
        before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skipped += int(scaler.get_scale() < before)
        sched.step()
        opt.zero_grad(set_to_none=True)
        if n % tc["log_every"] == 0 or n == len(steps) - 1:
            log.append({"step": n, "loss": run_loss, "s_per_step": (time.time() - t0) / (n + 1)})
            ctx.log(f"  step {n + 1}/{len(steps)}: loss {run_loss:.4f}, {(time.time() - t0) / (n + 1):.2f} s / step")
    sd = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}
    del model, base, opt
    if cuda:
        torch.cuda.empty_cache()
    return {"adapter": sd, "log": log, "steps": len(steps), "seconds": time.time() - t0, "skipped_steps": skipped,
            "n_items": len(its), "seed": seed}


def stage_train(ctx: Ctx, arms: tuple[str, ...], worker: int = 0, workers: int = 1) -> bool:
    import torch

    its = train_items(ctx)

    def write(key: str, tmp: Path) -> None:
        arm, seed = key[:-3].split("_")
        res = train_one(ctx, its, targets(ctx, arm, its), int(seed))
        with tmp.open("wb") as fh:
            torch.save(res, fh)
    keys = [f"{n}.pt" for n in run_names(ctx.rc, arms)][worker::workers]
    return ctx.run.chunks(S4, keys, write, ctx.rc["estimates_s"]["train_run"], log=ctx.log)


# ------------------------------------------------------------------------------------------- S5: test scoring (GPU)
def score_keys(rc: dict) -> dict[str, list[str]]:
    """model -> its sets. A's W0 readings on validation and test come from S2."""
    out = {m: ["validation", "id_test", "held_out"] for m in run_names(rc)}
    for m in rc["secondary_models"]:
        out.setdefault(m, []).extend(["id_w1", "sw_opposite", "sw_diff", "sw_own_cut"])
    return out


def all_score_chunks(rc: dict) -> list[str]:
    return [f"{m}__{s}.json" for m, ss in score_keys(rc).items() for s in ss]


def score_set(ctx: Ctx, name: str) -> list[dict]:
    its = items(ctx)
    if name.startswith("sw_"):
        return swap_sets(its, ctx.rc, ctx.scfg["scale"]["tolerance"])[name]
    return in_split(its, "id_test" if name == "id_w1" else name)


def stage_score(ctx: Ctx, worker: int = 0, workers: int = 1) -> bool:
    """Every model on its sets, the adapter loaded once per model."""
    import contextlib

    import torch
    from peft import get_peft_model
    from peft.utils import set_peft_model_state_dict

    plan = list(score_keys(ctx.rc).items())[worker::workers]
    if all(ctx.run.has(S5, f"{m}__{s}.json") for m, ss in plan for s in ss):
        return True
    base = _model(ctx)
    tok = _tok(ctx)
    model = get_peft_model(base, lora_config(ctx.rc)).eval()
    r = make_reader(tok, base, ctx.scfg, ctx.device)
    w0, w1, budget = F.verify_text(ctx.scfg), ctx.rc["verify_w1"], ctx.rc["readout"]["token_budget"]
    finished = True
    for m, sets in plan:
        keys = [f"{m}__{s}.json" for s in sets]
        if all(ctx.run.has(S5, k) for k in keys):
            continue
        if m != "A":
            with ctx.run.path(S4, f"{m}.pt").open("rb") as fh:
                set_peft_model_state_dict(model, torch.load(fh, weights_only=False)["adapter"])

        def write(key: str, tmp: Path, m=m) -> None:
            name = key[:-5].split("__")[1]
            its = score_set(ctx, name)
            with model.disable_adapter() if m == "A" else contextlib.nullcontext():
                lo, _ = read_log_odds(model, its, r, ctx.scfg, w1 if name == "id_w1" else w0, budget, ctx.device)
            tmp.write_text(json.dumps(dict(zip([it["id"] for it in its], lo.tolist()))))
        finished &= ctx.run.chunks(S5, keys, write, ctx.rc["estimates_s"]["score_set"], log=ctx.log)
    return finished


# ------------------------------------------------------------------------------------------- S6: the analysis (CPU)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """D175's selective risk: the mean over coverage of the error rate among the most confident answers (ties: in item
    order)."""
    err = (~correct[np.argsort(-conf, kind="stable")]).astype(float)
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def brier(p: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean((p - correct.astype(float)) ** 2))


def arm_columns(rc: dict) -> dict[str, list[str]]:
    return {"A": ["A"]} | {a: [f"{a}_{s}" for s in seeds] for a, seeds in rc["arms"].items()}


REFERENCES = ("probe", "rich", "answer_only", "self_consistency", "length")
CONTRASTS = (("B", "A"), ("C", "A"), ("C", "B"), ("C", "D"), ("E", "A"))
READINGS = {("B", "A"): ("direct supervision helps", "labels do not help", "labels hurt"),
            ("C", "A"): ("distillation helps", "distillation does not help", "distillation hurts"),
            ("C", "B"): ("the probe's target adds to labels", "labels are enough", "labels beat the probe's target"),
            ("C", "D"): ("it moves more than surface", "distillation is surface transfer", "the text target moves more"),
            ("E", "A"): ("the setup leaks: stop and diagnose", "as expected",
                         "as expected: shuffled targets cost the verdict's own signal")}


def reading(contrast: tuple[str, str], x: dict) -> str:
    side = R6._side(x)
    return READINGS[contrast][0 if side > 0 else 1 if side == 0 else 2]


def pairwise_block(rows: list[dict], columns: dict[str, list[str]], an: dict, readings: bool = True) -> dict:
    """Pairwise accuracy per arm (the mean of its seeds' per-problem values), per column, and D175's contrasts."""
    sc = {"bootstrap": an["bootstrap"], "seed": an["seed"]}
    groups = R6.by_problem(rows)
    keys = mixed_problems(rows)
    per = {c: R6.per_problem(groups, keys, {r["id"]: r["conf"].get(c) for r in rows})
           for cols in columns.values() for c in cols}
    pa = {a: np.mean([per[c] for c in cols], axis=0) for a, cols in columns.items()}
    res = {"n_problems": len(keys), "arms": {a: R6._summ(v, sc) for a, v in pa.items()},
           "columns": {c: float(v.mean()) for c, v in per.items()}, "contrasts": {}}
    for a, b in CONTRASTS:
        if a in pa and b in pa:
            x = A.paired(pa[a], pa[b], an["bootstrap"], an["seed"])
            res["contrasts"][f"{a} - {b}"] = x | ({"reading": reading((a, b), x)} if readings else {})
    return res


def item_metrics(rows: list[dict], columns: dict[str, list[str]], an: dict) -> dict:
    """Brier and AURC on every answered item: an arm is the mean of its seeds' values; bootstrap over problems for each
    arm and for its contrast with A. A column without a probability (length) has no Brier."""
    keys = list(R6.by_problem(rows))
    pos = {k: i for i, k in enumerate(keys)}
    prob = np.array([pos[tuple(r["key"])] for r in rows])
    order = [np.flatnonzero(prob == i) for i in range(len(keys))]
    y = np.array([r["correct"] for r in rows], dtype=bool)
    conf = {c: np.array([r["conf"][c] for r in rows], dtype=float) for cols in columns.values() for c in cols}
    rng = np.random.default_rng(an["seed"])
    p_of = {c: (sigmoid(v) if c.split("_")[0] in ("A", "B", "C", "D", "E") else v) for c, v in conf.items()}

    def stats(idx: np.ndarray) -> dict:
        out = {}
        for a, cols in columns.items():
            out[a] = {"aurc": float(np.mean([aurc(conf[c][idx], y[idx]) for c in cols]))}
            if cols[0] != "length":
                out[a]["brier"] = float(np.mean([brier(p_of[c][idx], y[idx]) for c in cols]))
        return out
    full = stats(np.arange(len(rows)))
    reps = []
    for _ in range(an["bootstrap"]):
        pick = rng.integers(0, len(keys), len(keys))
        reps.append(stats(np.concatenate([order[i] for i in pick])))
    res = {}
    for a in columns:
        res[a] = {}
        for m, v in full[a].items():
            d = np.array([x[a][m] for x in reps])
            res[a][m] = {"value": v, "ci95": np.percentile(d, [2.5, 97.5]).tolist()}
            if a != "A" and "A" in columns:
                dd = d - np.array([x["A"][m] for x in reps])
                res[a][f"{m} - A"] = {"value": v - full["A"][m], "ci95": np.percentile(dd, [2.5, 97.5]).tolist()}
    return res


def swap_block(rows: list[dict], models: list[str], an: dict) -> dict:
    """D175's swap test: pairwise accuracy on `sw_opposite` labelled by the box, and the box effect on `sw_diff`."""
    sc = {"bootstrap": an["bootstrap"], "seed": an["seed"]}
    opp = [r for r in rows if r["set"] == "sw_opposite"]
    diff = [r for r in rows if r["set"] == "sw_diff"]
    cut = {r["orig_id"]: r for r in rows if r["set"] == "sw_own_cut"}
    groups = R6.by_problem(opp)
    keys = mixed_problems(opp)
    res: dict = {"n_problems": len(keys), "models": {}}
    for m in models:
        pa = R6.per_problem(groups, keys, {r["id"]: r["conf"].get(m) for r in opp})
        s = R6._summ(pa, sc)
        lo, hi = s["ci95"]
        by: dict = {}
        for r in diff:
            d = float(sigmoid(r["conf"][m]) - sigmoid(cut[r["donor_id"]]["conf"][m]))
            by.setdefault(tuple(r["key"]), []).append(d)
        e = np.array([np.mean(v) for v in by.values()])
        res["models"][m] = {"opposite": s, "reading": "tracks the answer" if lo > 0.5 else "tracks the reasoning" if hi < 0.5
                            else "neither",
                            "diff_value_box_effect": A.paired(e, np.zeros_like(e), an["bootstrap"], an["seed"]) if len(e) else None}
    return res


def results_from_tables(rows: list[dict], swap_rows: list[dict], counts: dict, rc: dict) -> dict:
    """Every registered number from the per-answer tables (the ingest recomputes this and must get it exactly)."""
    an = rc["analysis"]
    arms = arm_columns(rc)
    refs = {c: [c] for c in REFERENCES}
    val = [r for r in rows if r["split"] == "validation"]
    bce = {}
    for c in [c for cols in arms.values() for c in cols]:
        lo = np.array([r["conf"][c] for r in val], dtype=float)
        y = np.array([r["correct"] for r in val], dtype=float)
        bce[c] = float(np.mean(np.logaddexp(0, lo) - y * lo)) if len(lo) else None
    diverged = sorted(c for c, v in bce.items() if v is not None and v > an["diverged_bce"])
    res: dict = {"decision": rc["decision"], "mode": rc["mode"], "counts": counts,
                 "validation_bce": bce, "diverged": diverged}
    for name, split in (("id_test", "id_test"), ("held_out", "held_out")):
        sub = [r for r in rows if r["split"] == split]
        res[name] = {"pairwise": pairwise_block(sub, arms | refs, an), "items": item_metrics(sub, arms | refs, an)}
        if diverged:
            kept = {a: [c for c in cols if c not in diverged] for a, cols in arms.items()}
            res[name]["pairwise_without_diverged"] = pairwise_block(sub, {a: c for a, c in kept.items() if c}, an)
        res[name]["seed_spread"] = {a: {"min": min(res[name]["pairwise"]["columns"][c] for c in cols),
                                        "max": max(res[name]["pairwise"]["columns"][c] for c in cols)}
                                    for a, cols in arms.items()}
    w1 = {m.split("_")[0]: [f"{m}@w1"] for m in rc["secondary_models"]}
    res["held_out_wording"] = pairwise_block([r for r in rows if r["split"] == "id_test"], w1, an, readings=False)
    res["swap"] = swap_block(swap_rows, rc["secondary_models"], an)
    return res


def build_tables(ctx: Ctx) -> tuple[list[dict], list[dict], dict]:
    """The per-answer tables from the stages' chunks."""
    rc, scfg = ctx.rc, ctx.scfg
    all_its, rows_all = items(ctx), sampled_rows(ctx)
    tol = scfg["scale"]["tolerance"]
    conf: dict[str, dict] = {"A": read_table(ctx)}
    for key in all_score_chunks(rc):
        m, s = key[:-5].split("__")
        c = json.loads(ctx.run.path(S5, key).read_text())
        if s == "id_w1":
            conf.setdefault(f"{m}@w1", {}).update(c)
        elif not s.startswith("sw_"):
            conf.setdefault(m, {}).update(c)
    for n in FINALS:
        c = json.loads(ctx.run.path(S3, f"final_{n}.json").read_text())
        conf[n] = dict(zip(c["ids"], c["p"]))
    sc = A.self_consistency(rows_all, tol)
    idk = {it["id"]: (tuple(it["key"]), it["sample"]) for it in all_its}
    conf["self_consistency"] = {i: sc.get(k) for i, k in idk.items()}
    conf["length"] = {it["id"]: -float(it["n_tokens"]) for it in all_its}
    rows = [{"id": it["id"], "key": it["key"], "sample": it["sample"], "split": it["split"], "correct": it["correct"],
             "n_tokens": it["n_tokens"], "conf": {c: v[it["id"]] for c, v in conf.items() if it["id"] in v}}
            for it in in_split(all_its, "validation", "id_test", "held_out")]
    for r in rows:
        if r["conf"].get("self_consistency") is None:    # a value of 0 has no self-consistency: the lowest
            r["conf"]["self_consistency"] = 0.0
    swap_rows = []
    for name, its in swap_sets(all_its, rc, tol).items():
        cs = {m: json.loads(ctx.run.path(S5, f"{m}__{name}.json").read_text()) for m in rc["secondary_models"]}
        swap_rows += [{"id": t["id"], "key": t["key"], "sample": t["sample"], "set": name, "correct": t["correct"],
                       "orig_id": t.get("orig_id"), "donor_id": t.get("donor_id"),
                       "conf": {m: c[t["id"]] for m, c in cs.items()}} for t in its]
    by = {}
    for r in rows_all:
        d = by.setdefault(split_of(rc)[r["index"]], {"samples": 0, "answered": 0, "truncated": 0})
        d["samples"] += 1
        d["truncated"] += bool(r["truncated"])
        d["answered"] += r["value"] is not None and not r["truncated"]
    return rows, swap_rows, by


def stage_analyze(ctx: Ctx) -> dict:
    rows, swap_rows, counts = build_tables(ctx)
    res = results_from_tables(rows, swap_rows, counts, ctx.rc)
    d = ctx.run.dir("s6_analysis")
    S.write_atomic(d / "results.json", json.dumps(res, indent=1, default=float))
    (d / "scores.jsonl.gz").write_bytes(_gz("".join(json.dumps(r) + "\n" for r in rows)))
    (d / "swap_scores.jsonl.gz").write_bytes(_gz("".join(json.dumps(r) + "\n" for r in swap_rows)))
    return res


def _gz(text: str) -> bytes:
    return gzip.compress(text.encode(), mtime=0)


# ------------------------------------------------------------------------------------------- smoke extras
def machine_report() -> dict:
    """What the session really has (D171: the sources disagree)."""
    def sh(cmd: str) -> str:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    mem = {k.strip(): v.strip() for k, v in (x.split(":", 1) for x in Path("/proc/meminfo").read_text().splitlines())}
    return {"cpu_count": os.cpu_count(), "cpu_affinity": len(os.sched_getaffinity(0)), "cpu_model": sh("lscpu | grep 'Model name'"),
            "mem_total": mem.get("MemTotal"), "mem_available": mem.get("MemAvailable"), "swap_total": mem.get("SwapTotal"),
            "cgroup_memory_max": sh("cat /sys/fs/cgroup/memory.max 2>/dev/null"), "cgroup_cpu_max": sh("cat /sys/fs/cgroup/cpu.max 2>/dev/null"),
            "df": sh("df -h /tmp /kaggle/working 2>/dev/null"), "gpus": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
            "kernel": sh("uname -r"), "python": sys.version.split()[0]}


def cpu_rate(rc: dict) -> dict:
    """Seconds per logistic fit at the real shape on synthetic data with a weak signal, all CPUs in parallel."""
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression

    n, d = rc["smoke"]["cpu_shape"]
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + 3 * rng.standard_normal(n) > 0).astype(int)
    jobs = len(os.sched_getaffinity(0))
    t0 = time.time()
    Parallel(n_jobs=jobs)(delayed(lambda: LogisticRegression(C=0.01, max_iter=2000).fit(X, y))()
                          for _ in range(rc["smoke"]["cpu_fits"]))
    dt = time.time() - t0
    return {"jobs": jobs, "fits": rc["smoke"]["cpu_fits"], "seconds": dt, "s_per_fit_in_parallel": dt / rc["smoke"]["cpu_fits"]}


def projection(ctx: Ctx) -> dict:
    """The full run's hours per stage from the rates this smoke session measured (D175: replaces the assumptions). A stage
    that did not get to run has no rate, and is listed as missing."""
    f, gpus = ctx.rc["smoke"]["full"], ctx.rc["session"]["gpus"]
    h, res = {}, {}
    tp_path = K.scale_dir(ctx.root, ctx.rc["name"]) / "throughput.json"
    if tp_path.exists():
        res["throughput"] = json.loads(tp_path.read_text())
        h["S1 sampling"] = res["throughput"]["s_per_problem"] * f["problems"] / 3600
    read = ctx.run.ledger(S2) if (ctx.root / S2).exists() else {}
    if read and sampling_done(ctx):
        n = sum(len(json.loads(str(np.load(ctx.run.path(S2, k))["ids"]))) for k in read)
        res["read_s_per_answer_per_gpu"] = sum(r["seconds"] for r in read.values()) / n
        h["S2 reading"] = res["read_s_per_answer_per_gpu"] * f["answers"] / gpus / 3600
    train = [_run_summary(ctx, n) for n in run_names(ctx.rc) if ctx.run.has(S4, f"{n}.pt")]
    if train:
        res |= {"train": train, "s_per_step_median": float(np.median([t["s_per_step"] for t in train]))}
        h["S4 training"] = res["s_per_step_median"] * f["train_steps"] * f["runs"] / gpus / 3600
    score = ctx.run.ledger(S5) if (ctx.root / S5).exists() else {}
    if score:
        n = sum(len(json.loads(ctx.run.path(S5, k).read_text())) for k in score)
        res["score_s_per_item_per_gpu"] = sum(r["seconds"] for r in score.values()) / max(n, 1)
        h["S5 scoring"] = res["score_s_per_item_per_gpu"] * f["score_items"] / gpus / 3600
    teach = ctx.run.ledger(S3) if (ctx.root / S3).exists() else {}
    res["teacher_chunks_s"] = {k: r["seconds"] for k, r in teach.items()}
    missing = sorted({"S1 sampling", "S2 reading", "S4 training", "S5 scoring"} - set(h))
    return res | {"hours": h, "total_gpu_hours": sum(h.values()), "sessions": sum(h.values()) / ctx.rc["full_session_hours"],
                  "missing": missing}


def _run_summary(ctx: Ctx, name: str) -> dict:
    import torch

    with ctx.run.path(S4, f"{name}.pt").open("rb") as fh:
        r = torch.load(fh, weights_only=False)
    return {"run": name, "steps": r["steps"], "seconds": r["seconds"], "skipped_steps": r["skipped_steps"],
            "s_per_step": r["seconds"] / r["steps"], "first_loss": r["log"][0]["loss"], "last_loss": r["log"][-1]["loss"],
            "adapter_mb": sum(v.numel() * v.element_size() for v in r["adapter"].values()) / 1e6}


def smoke_report(ctx: Ctx) -> None:
    d = ctx.run.dir("smoke")
    S.write_atomic(d / ("machine_second.json" if (d / "machine.json").exists() else "machine.json"),
                   json.dumps(machine_report(), indent=1))
    S.write_atomic(d / "cpu_rate.json", json.dumps(cpu_rate(ctx.rc), indent=1))
    S.write_atomic(d / "rates.json", json.dumps(projection(ctx), indent=1, default=float))
    f = ctx.run.dir("filecount")
    for i in range(ctx.rc["smoke"]["file_test_n"]):
        (f / f"f_{i:04d}").write_text("x")


def survival_report(root: Path, rc: dict, restored: dict) -> dict:
    """A second smoke session, with the first's output attached: what came through its deliberate crash."""
    run_files = [p for p in root.rglob("*") if p.is_file()]
    res = {"restored": restored, "files_now": len(run_files), "filecount_files": len(list((root / "filecount").glob("f_*"))),
           "filecount_expected": rc["smoke"]["file_test_n"], "at": time.time()}
    S.write_atomic(root / "smoke" / "survival.json", json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------------------------------- the zip and the ingest
def pack(ctx: Ctx, out: Path) -> dict:
    """D171's results zip: small tables, gzipped completions, the configs, the environment, logs and timings."""
    root, files = ctx.root, {}
    scale = K.scale_dir(root, ctx.rc["name"])
    d = root / "s6_analysis"
    for name in ("results.json", "scores.jsonl.gz", "swap_scores.jsonl.gz"):
        if (d / name).exists():
            files[name] = (d / name).read_bytes()
    if samples_dir(ctx).exists():
        rows = sampled_rows(ctx)
        files["completions.jsonl.gz"] = _gz("".join(json.dumps(r) + "\n" for r in rows))
    for name in ("throughput.json", "plan.json"):
        if (scale / name).exists():
            files[name] = (scale / name).read_bytes()
    for name in ("status.json", "environment.json"):
        if (root / name).exists():
            files[name] = (root / name).read_bytes()
    for n in ("r6_6a", "followup") + (("r6_6a_smoke",) if ctx.rc["mode"] == "smoke" else ()):
        files[f"config/{n}.yaml"] = (REPO_ROOT / "configs" / f"{n}.yaml").read_bytes()
    for f in sorted((root / "logs").glob("*.log")) if (root / "logs").exists() else []:
        files[f"logs/{f.name}"] = f.read_bytes()
    for stage in (S2, S3, S4, S5):
        led = ctx.run.ledger(stage) if (root / stage).exists() else {}
        if led:
            files[f"timing/{stage}.jsonl"] = "".join(json.dumps(r) + "\n" for r in led.values()).encode()
    for f in sorted((root / "smoke").glob("*.json")) if (root / "smoke").exists() else []:
        files[f"smoke/{f.name}"] = f.read_bytes()
    return S.write_zip(Path(out) / ctx.rc["results"]["zip_name"], files, ctx.rc["results"]["max_zip_mb"])


def _gz_rows(b: bytes) -> list[dict]:
    return [json.loads(x) for x in gzip.decompress(b).decode().splitlines()]


def ingest(zip_path: Path, mode: str, dest: Path | None = None) -> dict:
    """D171 rule 8: the zip's allow-list and manifest, a re-grade with the local parser, and every number recomputed from
    the per-answer tables, which must equal results.json."""
    rc = mode_config(mode)
    files = S.read_zip(zip_path, re.compile(OK_NAMES), rc["results"]["max_file_mb"])
    done: dict = {"files": len(files)}
    tol = load_config("followup")["scale"]["tolerance"]
    if "completions.jsonl.gz" in files:
        rows = _gz_rows(files["completions.jsonl.gz"])
        bad = [r["index"] for r in rows if F.grade(r["problem"], r["completion"], r["truncated"], tol)["correct"] != r["correct"]]
        if bad:
            raise SystemExit(f"{len(bad)} completions re-grade differently: {bad[:3]}")
        done["regraded"] = len(rows)
    if "results.json" in files:
        saved = json.loads(files["results.json"])
        again = json.loads(json.dumps(results_from_tables(_gz_rows(files["scores.jsonl.gz"]), _gz_rows(files["swap_scores.jsonl.gz"]),
                                                          saved["counts"], rc), default=float))
        if again != saved:
            raise SystemExit("results.json does not follow from its tables")
        done["results_recomputed"] = True
    out = Path(dest) if dest else REPO_ROOT / rc["results"]["ingest_dir"]
    for name, b in files.items():
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(b)
    (out / "ingest.json").write_text(json.dumps(done, indent=1))
    return done


# ------------------------------------------------------------------------------------------- the driver
def spawn(cmds: list[tuple[list[str], dict]], log_path: Path, stall_s: float, log=print) -> list[int]:
    """Each command as its own process group, lines streamed to the log; one silent for stall_s is killed (D157)."""
    codes: list = [None] * len(cmds)

    def one(i: int, cmd: list[str], env: dict) -> None:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**os.environ, **env},
                             start_new_session=True, bufsize=1)
        last = [time.time()]

        def pump() -> None:
            with log_path.open("a") as fh:
                for line in p.stdout:
                    last[0] = time.time()
                    log(f"[{i}] {line.rstrip()}")
                    fh.write(line)
                    fh.flush()
        t = threading.Thread(target=pump, daemon=True)
        t.start()
        while p.poll() is None:
            time.sleep(5)
            if time.time() - last[0] > stall_s:
                log(f"STALL: no output for {stall_s / 60:.0f} min; killing {cmd[3:6]}")
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        t.join(timeout=60)
        codes[i] = p.returncode
    ts = [threading.Thread(target=one, args=(i, c, e)) for i, (c, e) in enumerate(cmds)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return codes


def worker_cmd(stage: str, a: argparse.Namespace, w: int, n: int, extra: list[str] | None = None) -> list[str]:
    return [sys.executable, "-m", "physcal.eval.r6_6a", "worker", "--mode", a.mode, "--out", str(a.out), "--model-dir",
            str(a.model_dir), "--session-start", repr(a.session_start), "--stage", stage, "--worker", str(w),
            "--workers", str(n)] + (extra or [])


def drive(ctx: Ctx, launch, log=print) -> dict:
    """S1 to S6 in order; `launch(stage, arms=None) -> exit codes` runs a stage's workers. Each stage is finished only when
    every chunk exists. The first that cannot finish ends the session: the next one resumes there."""
    rc, run = ctx.rc, ctx.run
    folds = ctx.scfg["scale"]["probe"]["folds"]
    status: dict = {"stopped_at": None, "exit_codes": {}}

    def write_status() -> dict:
        status["done"] = [s for s in (S2, S3, S4, S5) if run.done(s)]
        S.write_atomic(ctx.root / "status.json", json.dumps(status, indent=1))
        return status

    def stage(name: str, keys_fn, arms=None) -> bool:
        if run.done(name):
            return True
        status["exit_codes"][name] = launch(name, arms)
        keys = keys_fn()
        if not all(run.has(name, k) for k in keys):
            return False
        run.finish(name, keys)
        return True

    if not sampling_done(ctx):
        status["exit_codes"][S1] = launch(S1, None)
        if not sampling_done(ctx):
            status["stopped_at"] = S1
            return write_status()
    log(f"S1: {len(sampled_rows(ctx))} samples, {len(items(ctx))} answered")
    if not stage(S2, lambda: part_keys(len(items(ctx)), rc["readout"]["part_size"])):
        status["stopped_at"] = S2
        return write_status()
    teach: dict = {}
    t3 = threading.Thread(target=lambda: teach.update(ok=stage(S3, lambda: teacher_keys(rc, folds))))
    t3.start()                                          # D175: the teachers run on the CPU during arm B
    if not run.done(S4) and not all(run.has(S4, f"{n}.pt") for n in run_names(rc, ("B",))):
        launch(S4, ("B",))
    t3.join()
    if not teach.get("ok"):
        status["stopped_at"] = S3
        return write_status()
    if not stage(S4, lambda: [f"{n}.pt" for n in run_names(rc)], ("C", "D", "E")):
        status["stopped_at"] = S4
        return write_status()
    if not stage(S5, lambda: all_score_chunks(rc)):
        status["stopped_at"] = S5
        return write_status()
    res = stage_analyze(ctx)
    log(f"S6: results.json written; {res['id_test']['pairwise']['n_problems']} ID-test problems with both classes")
    return write_status()


def _environment(ctx: Ctx) -> None:
    env = K._environment(ctx.model_dir)
    from importlib.metadata import PackageNotFoundError, version

    for n in ("peft", "scikit-learn", "scipy", "joblib", "accelerate", "safetensors"):
        try:
            env["packages"][n] = version(n)
        except PackageNotFoundError:
            env["packages"][n] = None
    S.write_atomic(ctx.root / "environment.json", json.dumps(env, indent=1))


def run_cli(a: argparse.Namespace) -> int:
    ctx = make_ctx(a.mode, a.out, a.session_start, a.model_dir)
    rc = ctx.rc
    (ctx.root / "logs").mkdir(exist_ok=True)
    log_path = ctx.root / "logs" / f"session_{int(a.session_start)}.log"
    n, stall = rc["session"]["gpus"], rc["session"]["stall_minutes"] * 60
    say = lambda m: print(m, flush=True)  # noqa: E731
    say(json.dumps(ctx.run.verify()))
    _environment(ctx)

    def launch(stage: str, arms) -> list[int]:
        if stage == S1:
            codes: list[int] = []
            base = [sys.executable, "-m", "physcal.eval.r6_6a", "sample", "--mode", a.mode, "--out", str(a.out), "--model-dir",
                    str(a.model_dir), "--session-start", repr(a.session_start)]
            for extra, env in [([], {}), (["--enforce-eager"], {"NCCL_P2P_DISABLE": "1"})][:rc["session"]["sampler_attempts"]]:
                codes += spawn([(base + extra, env)], log_path, stall, say)
                if codes[-1] in (0, K.EXIT_HEALTH):
                    break
            return codes
        if stage == S3:
            # a teacher chunk logs only when it ends, and a fold can take longer than a GPU chunk's silence
            return spawn([(worker_cmd(stage, a, 0, 1), {"CUDA_VISIBLE_DEVICES": ""} | ONE_THREAD)], log_path, 3 * stall, say)
        extra = ["--arms", ",".join(arms)] if arms else []
        return spawn([(worker_cmd(stage, a, g, n, extra), {"CUDA_VISIBLE_DEVICES": str(g)}) for g in range(n)], log_path, stall, say)
    st = drive(ctx, launch, say)
    if rc["mode"] == "smoke":                           # whatever drive reached: the rates are the point of the session
        smoke_report(ctx)
    say("STATUS: " + json.dumps({k: st.get(k) for k in ("stopped_at", "done")}))
    return 0


def sample_cli(a: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    ctx = make_ctx(a.mode, a.out, a.session_start, a.model_dir)
    tok = AutoTokenizer.from_pretrained(str(a.model_dir))
    env = K._environment(a.model_dir)
    settings = {**env, "enforce_eager": a.enforce_eager, "decision": ctx.rc["decision"]}
    return K.sample(lambda: K.VLLMEngine(a.model_dir, ctx.scfg, a.enforce_eager), tok, ctx.root, a.session_start, ctx.scfg,
                    log=lambda m: print(m, flush=True), settings=settings, model=ctx.rc["name"])


def worker_cli(a: argparse.Namespace) -> int:
    dev = "cuda" if a.stage != S3 else "cpu"
    ctx = make_ctx(a.mode, a.out, a.session_start, a.model_dir, f"w{a.worker}", dev)
    ctx.log = lambda m: print(m, flush=True)
    try:
        if a.stage == S2:
            stage_read(ctx, a.worker, a.workers)
        elif a.stage == S3:
            stage_teachers(ctx)
        elif a.stage == S4:
            stage_train(ctx, tuple(a.arms.split(",")), a.worker, a.workers)
        else:
            stage_score(ctx, a.worker, a.workers)
    except S.StageStop as exc:
        print(f"STOP RULE: {exc}", flush=True)
        return EXIT_STOP
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["pool", "download", "restore", "sample", "worker", "run", "pack", "ingest"])
    ap.add_argument("--mode", default="full", choices=["full", "smoke"])
    ap.add_argument("--out", type=Path, help="Kaggle: the output root (/kaggle/working)")
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--src", type=Path, help="restore: where a previous session's output is mounted")
    ap.add_argument("--session-start", type=float)
    ap.add_argument("--stage", choices=[S2, S3, S4, S5])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--arms", help="worker, S4: comma-separated arms")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--zip", type=Path)
    return ap


def main() -> None:
    a = parser().parse_args()
    if a.action == "pool":
        print(json.dumps(build_pool(a.mode), indent=1))
    elif a.action == "download":
        rc = mode_config(a.mode)
        print(json.dumps(K.download(a.model_dir, scale_config(rc), rc["name"]), indent=1))
    elif a.action == "restore":
        rc = mode_config(a.mode)
        res = S.restore(Path(a.out) / rc["name"], a.src)
        if res["restored"] and a.mode == "smoke":
            print(json.dumps(survival_report(Path(a.out) / rc["name"], rc, res), indent=1))
    elif a.action == "pack":
        ctx = make_ctx(a.mode, a.out, time.time(), None)
        print(json.dumps(pack(ctx, a.out), indent=1))
    elif a.action == "ingest":
        print(json.dumps(ingest(a.zip, a.mode), indent=1))
    elif a.action == "run":
        sys.exit(run_cli(a))
    elif a.action == "sample":
        sys.exit(sample_cli(a))
    else:
        sys.exit(worker_cli(a))


if __name__ == "__main__":
    main()

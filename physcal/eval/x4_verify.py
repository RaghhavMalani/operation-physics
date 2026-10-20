"""X4 (pre-registered in DECISIONS.md D89): self-verification on the X2 pairs.

    uv run python -m physcal.eval.x4_verify reference        # CPU, once, before any X4 data: X2's per-answer confidences
    uv run python -m physcal.eval.x4_verify run --arm sft|shuffled|random_init [--max-reserved-gib G] [--minutes M]  # GPU
    uv run python -m physcal.eval.x4_analysis analyze        # CPU: the D89 analysis

Every answer of X2's kept problems (the answers that form X2's pairs) gets one verification conversation: the v4
system prompt, the problem, the answer's own completion as the assistant turn, then the fixed verification prompt
`x4_v1` as a second user turn. The model recomputes greedily (up to 512 tokens); its text is cut before its last
"verdict" and the canonical line "Verdict:" is appended, so every answer is read at the same position. The confidence
is the yes-vs-no log-odds of the next token there, from float32 logits (lm_head in float32 on the final hidden state;
checked against the model's own logits on the first row of every run).

Arms: `sft` (the corrected SFT model of X2, 4-bit), `shuffled` (the same model; the user turn shows a donor problem's
text: the next kept problem of the same cell), `random_init` (X2's random-init model, loaded the same way). Rows are
appended after every batch to `<output_dir>/<arm>/rows.jsonl`; a rerun resumes where the last one stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

from physcal.config import REPO_ROOT, load_config
from physcal.eval import x2_pairs
from physcal.prompt import SYSTEM_PROMPTS, VERIFY_PROMPTS, build_messages


def _config() -> dict:
    return load_config("x4")


def _x2(cfg: dict) -> dict:
    """X2's config (a name, or the dict itself in tests)."""
    x = cfg["x2_config"]
    return x if isinstance(x, dict) else load_config(x)


def check_prompt(cfg: dict) -> str:
    """The verification prompt, after checking its SHA-256 against the config (D89 quotes it)."""
    text = VERIFY_PROMPTS[cfg["verify_prompt"]]
    if hashlib.sha256(text.encode()).hexdigest() != cfg["verify_prompt_sha256"]:
        raise ValueError(f"verification prompt {cfg['verify_prompt']} does not match its pre-registered SHA-256")
    return text


# ------------------------------------------------------------------------------------------- the answers
def items(cfg: dict | None = None) -> tuple[list[dict], list[tuple]]:
    """Every parsed answer of X2's kept problems (exactly the answers in X2's pairs), in the kept order then sample
    index, and the kept keys. Rows: key, sample, problem (dict), completion, correct."""
    from physcal.tape.tape import read_meta

    cfg = cfg or _config()
    x2 = _x2(cfg)
    prompt_ids = {read_meta(t)["prompt_id"] for _, t in x2_pairs.sampled_chunks(x2)}
    if prompt_ids != {cfg["prompt"]}:
        raise ValueError(f"the X2 answers were sampled under {prompt_ids}, the config says {cfg['prompt']}")
    groups, keys = x2_pairs.kept_groups(x2)
    recs, _ = x2_pairs.kept_records(x2)
    answered = {(tuple(k), a["sample"]): a for k in keys for a in groups[k]}
    order = {tuple(k): i for i, k in enumerate(keys)}
    out = []
    for r in recs:
        p = r["problem"]
        key, s = (p["family"], p["depth"], p["seed"]), r.get("sample", 0)
        a = answered.get((key, s))
        if a is not None:
            out.append({"key": list(key), "sample": s, "problem": p, "completion": r["completion"],
                        "correct": bool(a["correct"])})
    out.sort(key=lambda x: (order[tuple(x["key"])], x["sample"]))
    if len(out) != len(answered):
        raise ValueError(f"{len(out)} answers found, X2's groups have {len(answered)}")
    return out, [tuple(k) for k in keys]


def donors(keys: list) -> dict:
    """D89's derangement for the shuffled control: each kept problem's donor is the next kept problem of the same
    (family, depth) cell in the kept order, cyclically. Every problem's text is shown once, never with its own answers."""
    cells: dict = {}
    for k in keys:
        cells.setdefault((k[0], k[1]), []).append(tuple(k))
    out = {}
    for cell, ks in cells.items():
        if len(ks) < 2:
            raise ValueError(f"cell {cell} has {len(ks)} kept problem(s): no donor")
        for i, k in enumerate(ks):
            out[k] = ks[(i + 1) % len(ks)]
    return out


def conversation(problem_text: str, completion: str, prompt_id: str, verify_text: str) -> list[dict]:
    """The verification conversation: system prompt, problem, the answer's completion, the verification prompt."""
    return build_messages(problem_text, prompt_id) + [{"role": "assistant", "content": completion},
                                                       {"role": "user", "content": verify_text}]


# ------------------------------------------------------------------------------------------- the reading
def reading_text(generated: str, rc: dict) -> tuple[str, str | None]:
    """The generated text up to D89's canonical reading position, and the verdict word the model wrote itself (None
    if it wrote no verdict): cut before the LAST match of the cut pattern, trailing whitespace / markdown dropped,
    the marker appended (after a newline if any text remains)."""
    matches = list(re.finditer(rc["cut_pattern"], generated))
    kept, natural = generated, None
    if matches:
        m = matches[-1]
        kept = generated[:m.start()]
        w = re.search(r"[A-Za-z]+", generated[m.end():])
        natural = w.group(0).lower() if w else ""
    kept = kept.rstrip(rc["strip_chars"])
    return (kept + "\n" if kept else "") + rc["marker"], natural


def token_ids(tok, words: list[str]) -> list[int]:
    """The single-token ids of `words` (raises if any is not exactly one token, or two share an id)."""
    ids = []
    for w in words:
        e = tok.encode(w, add_special_tokens=False)
        if len(e) != 1:
            raise ValueError(f"{w!r} is not a single token: {e}")
        ids.append(e[0])
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate token ids for {words}: {ids}")
    return ids


def confidence(logits, yes: list[int], no: list[int]) -> dict:
    """One reading from next-token logits (a 1-D tensor): the yes-vs-no log-odds (the primary confidence), P(yes)
    renormalized over yes + no, and the raw P(yes), P(no) over the vocabulary. float64 throughout."""
    import torch

    x = logits.double()
    lse, ly, ln = torch.logsumexp(x, 0), torch.logsumexp(x[yes], 0), torch.logsumexp(x[no], 0)
    lo = float(ly - ln)
    return {"log_odds": lo, "p_yes": 1.0 / (1.0 + math.exp(-lo)) if lo > -700 else 0.0,
            "p_yes_raw": float(torch.exp(ly - lse)), "p_no_raw": float(torch.exp(ln - lse)),
            "top_token": int(x.argmax())}


def head_logits(model, h, chunk: int):
    """float32 next-token logits from final hidden states h (n, H): lm_head's weight in float32, chunked over the
    vocabulary (D89: the model's bf16 logits would create spurious ties). None if the head has no plain float weight."""
    import torch

    head = model.get_output_embeddings()
    W = getattr(head, "weight", None)
    if W is None or not W.is_floating_point() or W.dim() != 2 or W.shape[1] != h.shape[-1]:
        return None
    h32 = h.float()
    out = torch.cat([h32 @ W[i:i + chunk].float().T for i in range(0, W.shape[0], chunk)], dim=1)
    b = getattr(head, "bias", None)
    return out if b is None else out + b.float()


def final_states(model, ids: list[list[int]], pad_id: int, device: str):
    """The final (post-norm) hidden state at each sequence's last token: one forward, right padding (causal: a real
    token's state never sees the padding)."""
    import torch

    n = max(len(x) for x in ids)
    inp = torch.full((len(ids), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids), n), dtype=torch.long)
    for j, x in enumerate(ids):
        inp[j, :len(x)] = torch.tensor(x)
        mask[j, :len(x)] = 1
    with torch.inference_mode():
        out = model.model(input_ids=inp.to(device), attention_mask=mask.to(device), use_cache=False)
    hs = out.last_hidden_state if getattr(out, "last_hidden_state", None) is not None else out[0]
    last = torch.tensor([len(x) - 1 for x in ids], device=hs.device)
    return hs[torch.arange(len(ids), device=hs.device), last]


def own_logits(model, ids: list[int], device: str):
    """The model's own next-token logits after one unpadded sequence (the fallback path), float32."""
    import torch

    with torch.inference_mode():
        return model(input_ids=torch.tensor([ids], device=device)).logits[0, -1].float()


def logit_check(model, ids: list[int], pad_id: int, device: str, rc: dict) -> dict:
    """D89: the float32-head logits must reproduce the model's own at the reading position (max |diff| / max |logit|
    <= rtol, the same argmax). Raises if the model's own output is not vocabulary logits at all."""
    ref = own_logits(model, ids, device)
    head = model.get_output_embeddings()
    if getattr(head, "weight", None) is not None and ref.shape[-1] != head.weight.shape[0]:
        raise RuntimeError(f"the model returns {ref.shape[-1]} values per position, not vocabulary logits "
                           f"({head.weight.shape[0]})")
    import torch

    with torch.inference_mode():
        mine = head_logits(model, final_states(model, [ids], pad_id, device), rc["head_chunk"])
    if mine is None:
        return {"path": "model_logits", "reason": "lm_head has no plain float weight"}
    mine = mine[0]
    gap = float((mine - ref).abs().max() / ref.abs().max())
    ok = gap <= rc["logit_check_rtol"] and int(mine.argmax()) == int(ref.argmax())
    return {"path": "float32_head" if ok else "model_logits", "relative_gap": gap,
            "argmax_agree": int(mine.argmax()) == int(ref.argmax()), "vocab": int(ref.shape[-1])}


def _is_oom(exc: BaseException) -> bool:
    from physcal.eval.run_eval import _is_oom as is_oom

    return is_oom(exc)


def read_out(model, contexts: list[list[int]], pad_id: int, device: str, rc: dict, path: str, yes, no,
             bs: int) -> list[dict]:
    """Readings for tokenized reading contexts, `bs` per forward on the float32-head path (an OOM halves it), one at a
    time on the model-logits path."""
    import torch

    out: list[dict] = []
    i = 0
    while i < len(contexts):
        part = contexts[i:i + (bs if path == "float32_head" else 1)]
        try:
            if path == "float32_head":
                with torch.inference_mode():
                    lg = head_logits(model, final_states(model, part, pad_id, device), rc["head_chunk"])
                out += [confidence(lg[j], yes, no) for j in range(len(part))]
            else:
                out.append(confidence(own_logits(model, part[0], device), yes, no))
        except Exception as exc:
            if not _is_oom(exc) or len(part) == 1:
                raise
            torch.cuda.empty_cache()
            bs = max(1, len(part) // 2)
            print(f"  reading OOM on {len(part)} sequences; batch {bs}", flush=True)
            continue
        i += len(part)
    return out


# ------------------------------------------------------------------------------------------- the run
def cap_gib(cfg: dict, used_mib: int | None = None, total_mib: int | None = None) -> float:
    """D89's per-stage allocator cap: total VRAM - memory used now - margin, clipped (the D86/D87 WSL-spill rule).
    Reads nvidia-smi unless both values are given."""
    c = cfg["cap"]
    if used_mib is None or total_mib is None:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True).stdout.splitlines()[0].split(",")
        used_mib, total_mib = int(q[0]), int(q[1])
    return round(min(c["max_gib"], max(c["min_gib"], (total_mib - used_mib) / 1024 - c["margin_gib"])), 2)


def arm_dir(arm: str, cfg: dict) -> Path:
    return REPO_ROOT / cfg["output_dir"] / arm


def read_rows(arm: str, cfg: dict | None = None, repair: bool = False) -> list[dict] | None:
    """The arm's rows; a trailing line cut off by a kill mid-write is dropped (and, with `repair`, removed)."""
    p = arm_dir(arm, cfg or _config()) / "rows.jsonl"
    if not p.exists():
        return None
    text = p.read_text()
    if text and not text.endswith("\n"):
        text = text[:text.rfind("\n") + 1]
        if repair:
            p.write_text(text)
    return [json.loads(l) for l in text.splitlines()]


def make_backend(arm: str, cfg: dict, max_reserved_gib: float | None):
    """The arm's model through the eval loader (Unsloth 4-bit, no adapter), with greedy decoding and the cap."""
    from physcal.eval.run_eval import HFBackend

    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"  # GRPO's switch: forward would return states instead of logits
    m, g = cfg["model"], cfg["generation"]
    model_cfg = {"id": str(REPO_ROOT / cfg["arms"][arm]["model_dir"]), "loader": m["loader"], "lora": None,
                 "max_seq_length": m["max_seq_length"], "text_only": False, "dtype": "bfloat16",
                 "model_class": "causal_lm", "chat_template_kwargs": {},
                 "max_reserved_gib": max_reserved_gib or m["max_reserved_gib"]}
    gen = {"do_sample": False, "num_samples": 1, "max_new_tokens": g["max_new_tokens"], "batch_size": g["batch_size"],
           "repetition_penalty": g["repetition_penalty"], "seed": g["seed"], "temperature": None, "top_p": None,
           "top_k": None}
    return HFBackend(model_cfg["id"], model_cfg, gen)


def _model_hash(arm: str, cfg: dict) -> str | None:
    """The model dir's content hash; the random-init model must still be X2's (its recorded hash)."""
    from physcal.eval.fingerprint import _dir_content_hash

    d = REPO_ROOT / cfg["arms"][arm]["model_dir"]
    h = _dir_content_hash(d)
    rec = d / "physcal_random_init.json"
    if rec.exists() and json.loads(rec.read_text())["content_sha256"] != h:
        raise RuntimeError(f"{d} is not X2's random-init model (content hash {h})")
    return h


def run(arm: str, cfg: dict | None = None, backend=None, max_reserved_gib: float | None = None,
        minutes: float | None = None, limit: int | None = None) -> dict:
    """One arm over every X2 answer (or the first `limit`), resuming after the rows already written. Stops cleanly
    between batches once `minutes` have passed."""
    import torch

    from physcal.eval.run_eval import _git_commit

    cfg = cfg or _config()
    if arm not in cfg["arms"]:
        raise ValueError(f"unknown arm {arm!r}")
    t0 = time.time()
    verify_text = check_prompt(cfg)
    every, keys = items(cfg)
    its = every[:limit] if limit else every
    problems = {tuple(it["key"]): it["problem"] for it in every}
    don = donors(keys) if cfg["arms"][arm]["donor"] else {k: k for k in keys}
    out = arm_dir(arm, cfg)
    out.mkdir(parents=True, exist_ok=True)
    done = {(tuple(r["key"]), r["sample"]) for r in (read_rows(arm, cfg, repair=True) or [])}
    todo = [it for it in its if (tuple(it["key"]), it["sample"]) not in done]
    print(f"X4 {arm}: {len(every)} answers, {len(done)} already done, {len(todo)} to go"
          f"{f' (limit {limit})' if limit else ''}", flush=True)
    seg: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_done_before": len(done), "git_commit": _git_commit()}
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "decision": "D89 (X4)", "arm": arm, "model_dir": cfg["arms"][arm]["model_dir"],
        "model_content_sha256": _model_hash(arm, cfg) if backend is None else None, "prompt_id": cfg["prompt"],
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPTS[cfg["prompt"]].encode()).hexdigest(),
        "verify_prompt": cfg["verify_prompt"], "verify_prompt_sha256": cfg["verify_prompt_sha256"],
        "generation": cfg["generation"], "model": cfg["model"], "reading": cfg["reading"], "n_answers": len(every),
        "segments": []}
    if meta["arm"] != arm or meta["verify_prompt_sha256"] != cfg["verify_prompt_sha256"]:
        raise ValueError(f"{meta_path} belongs to another run")
    if not todo:
        return {"arm": arm, "n_rows": len(done), "complete": len(done) >= len(every)}
    if backend is None:
        backend = make_backend(arm, cfg, max_reserved_gib)
    tok, model, device = backend.tok, backend.model, backend.device
    rc = cfg["reading"]
    yes, no = token_ids(tok, rc["yes_tokens"]), token_ids(tok, rc["no_tokens"])
    kw = getattr(backend, "chat_template_kwargs", {}) or {}

    def prompt_of(it):
        shown = problems[don[tuple(it["key"])]]
        return tok.apply_chat_template(conversation(shown["text"], it["completion"], cfg["prompt"], verify_text),
                                       tokenize=False, add_generation_prompt=True, **kw)

    first = tok(prompt_of(todo[0]) + rc["marker"])["input_ids"]
    check = logit_check(model, first, backend.pad_id, device, rc)
    path = check["path"]
    print(f"X4 {arm}: reading path {path} ({check})", flush=True)
    cap = (max_reserved_gib or cfg["model"]["max_reserved_gib"]) if device == "cuda" else None
    seg.update(max_reserved_gib=cap, logit_check=check, path=path, yes_ids=yes, no_ids=no, device=device,
               loader=getattr(backend, "loader", None), loader_check=getattr(backend, "loader_check", None),
               provenance=getattr(backend, "provenance", None))
    meta["segments"].append(seg)
    meta_path.write_text(json.dumps(meta, indent=1, default=str))
    bs, n_new, stopped, b = cfg["generation"]["batch_size"], 0, False, 0
    while n_new < len(todo):
        if minutes is not None and time.time() - t0 > minutes * 60:
            stopped = True
            print(f"X4 {arm}: stopping after {minutes} min (soft limit), {len(done) + n_new} rows", flush=True)
            break
        batch = todo[n_new:n_new + bs]
        prompts = [prompt_of(it) for it in batch]
        before = getattr(backend, "oom_backoffs", 0)
        gens = [g[0] for g in backend._generate(prompts, 1)]
        if getattr(backend, "oom_backoffs", 0) > before and bs > 1:  # the backoff halved this call only: keep it
            bs //= 2
            print(f"  X4 {arm}: OOM backoff; batch size {bs} from now on", flush=True)
        cuts = [reading_text(g.text, rc) for g in gens]
        ctx = [tok(p + c[0])["input_ids"] for p, c in zip(prompts, cuts)]
        reads = read_out(model, ctx, backend.pad_id, device, rc, path, yes, no, rc["score_batch_size"])
        lines = [json.dumps({
            "arm": arm, "key": it["key"], "sample": it["sample"], "shown_key": list(don[tuple(it["key"])]),
            "correct": it["correct"], **r, "path": path, "natural_verdict": natural,
            "prompt_tokens": len(tok(p)["input_ids"]), "gen_tokens": g.n_tokens, "truncated": g.truncated,
            "context_tokens": len(c), "generation": g.text}) + "\n"
            for it, p, g, (_, natural), r, c in zip(batch, prompts, gens, cuts, reads, ctx)]
        with (out / "rows.jsonl").open("a") as f:
            f.write("".join(lines))  # one write per batch
        n_new += len(batch)
        b += 1
        if b % 10 == 1:
            print(f"  X4 {arm}: {len(done) + n_new}/{len(every)} ({time.time() - t0:.0f}s, OOM backoffs "
                  f"{getattr(backend, 'oom_backoffs', 0)})", flush=True)
    n_rows = len(done) + n_new
    seg.update(ended=time.strftime("%Y-%m-%dT%H:%M:%S%z"), seconds=time.time() - t0, n_new=n_new,
               stopped_by_time=stopped, oom_backoffs=getattr(backend, "oom_backoffs", 0), final_batch_size=bs,
               memory=backend.memory_stats() if hasattr(backend, "memory_stats") else None)
    meta.update(n_rows=n_rows, complete=n_rows >= len(every))
    meta_path.write_text(json.dumps(meta, indent=1, default=str))
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"arm": arm, "n_rows": n_rows, "complete": meta["complete"], "seconds": seg["seconds"], "path": path}


# ------------------------------------------------------------------------------------------- the X2 reference
def reference(cfg: dict | None = None) -> dict:
    """D89, before any X4 data: X2's per-answer confidences on the kept answers. Method 5 (the probe) is re-run with
    X2's own code on the saved X2 hidden states and must reproduce results/x2/results.json; method 1 (SFT stated σ)
    comes from the answers themselves. Writes <output_dir>/x2_reference.json."""
    from physcal.eval import x2_analysis as XA
    from physcal.eval import x2_hidden

    cfg = cfg or _config()
    x2 = _x2(cfg)
    groups, keys = x2_pairs.kept_groups(x2)
    fl, fi = x2_hidden.load("sft", "fit", x2)
    xl, xi = x2_hidden.load("sft", "x2", x2)
    probe, info = XA.probe_confidence(fl, fi, xl, xi, x2["analysis"]["probe"])
    sigma = {(tuple(k), a["sample"]): x2_pairs.stated_sigma_conf(k, a) for k in keys for a in groups[k]}
    mine = float(XA.per_problem(groups, keys, probe).mean())
    published = json.loads((REPO_ROOT / x2["output_dir"] / "results.json").read_text())["methods"]["probe"]
    gap = abs(mine - published["pairwise_accuracy"])
    if gap > cfg["analysis"]["probe_check_atol"]:
        raise RuntimeError(f"the re-run X2 probe gives {mine}, results/x2/results.json {published['pairwise_accuracy']}")
    res = {"decision": "D89 (X4): X2's per-answer confidences, computed before any X4 data",
           "probe_info": info, "check": {"probe_pairwise_accuracy": mine,
                                         "published": published["pairwise_accuracy"], "abs_gap": gap},
           "probe": [[list(k), s, v] for (k, s), v in probe.items()],
           "sft_sigma": [[list(k), s, v] for (k, s), v in sigma.items()]}
    out = REPO_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "x2_reference.json").write_text(json.dumps(res))
    return {k: v for k, v in res.items() if k not in ("probe", "sft_sigma")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["reference", "run", "cap"])
    ap.add_argument("--arm", choices=["sft", "shuffled", "random_init"], default="sft")
    ap.add_argument("--max-reserved-gib", type=float, default=None, help="run: allocator cap (default: the config's)")
    ap.add_argument("--minutes", type=float, default=None, help="run: stop cleanly between batches after this long")
    ap.add_argument("--limit", type=int, default=None, help="run: only the first N answers (checks)")
    args = ap.parse_args()
    if args.action == "reference":
        print(json.dumps(reference(), indent=1))
    elif args.action == "cap":
        print(cap_gib(_config()))
    else:
        print(json.dumps(run(args.arm, max_reserved_gib=args.max_reserved_gib, minutes=args.minutes,
                             limit=args.limit), indent=1))


if __name__ == "__main__":
    main()

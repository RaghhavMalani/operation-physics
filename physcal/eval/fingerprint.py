"""R2-1 (D72): the training-time fingerprint of a trained adapter, and the regression check a reloaded model must pass.

Recorded by the trainers right after the final adapter is saved, with the in-memory policy in inference mode
(`record`, written as <adapter>/physcal_fingerprint.json). For each of N fixed prompts (training-split seeds of their
own):
- teacher-forced log-probs of a fixed continuation, the generator-written solution in the output format, with the
  adapter on and off. "Off" measures the adapter's own effect, the scale of the tolerances;
- the next-token distribution at the first σ token of that continuation (top-k and the rest's mass), on and off;
- k sampled completions at the training decoding settings with a fixed torch seed, scored by the verifier: the
  stated relative σ distribution;
- the provenance of the loaded model (`provenance`): base, revision or content hash, quantization config, compute
  dtype, library versions, adapter hashes.

Checked by the eval's adapter loader on the reloaded model (`check`). It passes iff
- mean |Δ log p| (reloaded vs training) <= max_logprob_ratio × the adapter's own mean |Δ log p|;
- the TV distance at the σ token <= max_sigma_tv_ratio × the adapter's own TV distance;
- |median ln σ_rel, reloaded − training| <= max_abs_dlog_median_sigma, when both sides parse >= min_parsed σ;
- the `must_match` provenance fields are equal.
An adapter whose own log-prob effect is below `min_adapter_effect` cannot be judged by the log-prob criterion. D72
made such a check "inconclusive"; since D77 (`below_floor: sigma_token`, recorded with each new fingerprint) the
σ-token criterion decides it instead, and the log-prob criterion is reported only. Fingerprints recorded before D77
keep D72's rule. The bf16 merge of FINDINGS #15 erases > 90% of an RL update, so it gives a log-prob ratio near 1.
The same functions run on both sides, so only the loader differs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from physcal.config import load_config

VERSIONS = ("torch", "transformers", "peft", "trl", "accelerate", "bitsandbytes", "unsloth", "unsloth_zoo")


def _config() -> dict:
    return load_config("fingerprint")


# ---- provenance (R2-1: recorded in every eval tape) -------------------------------------------------------------

def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def adapter_hashes(lora_dir: str | Path) -> dict:
    """`file`: SHA-256 of the safetensors bytes (the tapes' lora_hash); `tensors`: SHA-256 over every tensor's name,
    dtype, shape and values, so two saves of the same weights (a checkpoint and the final adapter) hash alike."""
    from safetensors.numpy import load_file

    files = sorted(Path(lora_dir).glob("adapter_model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no adapter_model*.safetensors in {lora_dir}")
    h_file, h_t = hashlib.sha256(), hashlib.sha256()
    for f in files:
        h_file.update(f.read_bytes())
        tensors = load_file(str(f))
        for name in sorted(tensors):
            a = np.ascontiguousarray(tensors[name])
            h_t.update(f"{name}|{a.dtype}|{a.shape}|".encode())
            h_t.update(a.tobytes())
    return {"file": h_file.hexdigest()[:16], "tensors": h_t.hexdigest()[:16]}


def _dir_content_hash(d: Path) -> str | None:
    files = sorted(d.glob("*.safetensors"))
    if not files:
        return None
    h = hashlib.sha256()
    for f in files:
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in sorted(x.items())}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return str(x)


def provenance(model, model_id: str, lora_dir: str | Path | None = None) -> dict:
    """What was actually loaded: the base (a local dir's content hash, or the hub commit), the quantization config,
    the compute dtype, library versions and the adapter's hashes and LoRA settings."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    cfg = base.config
    qc = getattr(cfg, "quantization_config", None)
    qc = qc.to_dict() if hasattr(qc, "to_dict") else qc
    qc = _jsonable(qc) if qc else None
    local = Path(model_id).is_dir()
    out = {
        "base": {"model_id": str(model_id), "name_or_path": str(getattr(cfg, "_name_or_path", "")),
                 "commit": getattr(cfg, "_commit_hash", None),
                 "content_sha256": _dir_content_hash(Path(model_id)) if local else None},
        "quantization_config": qc,
        "compute_dtype": {"bnb_4bit_compute_dtype": (qc or {}).get("bnb_4bit_compute_dtype"),
                          "model_dtype": str(getattr(base, "dtype", None))},
        "versions": {p: _version(p) for p in VERSIONS},
        "adapter": None,
    }
    b = out["base"]  # a path-independent identity: training and eval name the same local base differently
    out["base_identity"] = (f"sha256:{b['content_sha256']}" if local else f"{b['name_or_path'] or model_id}@{b['commit']}")
    out["bitsandbytes"] = out["versions"]["bitsandbytes"]
    if lora_dir:
        lc = json.loads((Path(lora_dir) / "adapter_config.json").read_text())
        out["adapter"] = {"dir": str(lora_dir), **adapter_hashes(lora_dir),
                          "base_model_name_or_path": lc.get("base_model_name_or_path"),
                          "r": lc.get("r"), "lora_alpha": lc.get("lora_alpha"),
                          "target_modules": sorted(lc.get("target_modules") or [])}
    return out


# ---- the fixed probe set ----------------------------------------------------------------------------------------

def fixed_items(cfg: dict) -> list[dict]:
    """N fixed prompts with their generator-written continuations (the SFT target format) and σ spans."""
    from physcal.train.sft_data import build_examples

    p = cfg["prompts"]
    exs = build_examples(p["n"], p["seed_offset"], p["sigma_rel"], load_config(p["env_config"]))
    return [{"problem": e.problem, "continuation": e.completion, "sigma_span": e.sigma_span} for e in exs]


def _prompt(tok, problem_text: str, prompt_id: str, chat_kwargs: dict) -> str:
    from physcal.prompt import build_messages

    return tok.apply_chat_template(build_messages(problem_text, prompt_id), tokenize=False, add_generation_prompt=True,
                                   **chat_kwargs)


@contextmanager
def _logits_mode():
    """Unsloth's GRPO sets UNSLOTH_RETURN_HIDDEN_STATES=1 for its log-prob pass and does not always reset it; with it
    on, a forward returns hidden states as `.logits`. Force it off here."""
    old = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = old


def _device(model):
    return next(model.parameters()).device


def _forward(model, tok, items: list[dict], prompt_id: str, chat_kwargs: dict, top_k: int) -> list[dict]:
    """Per item: log p of every continuation token, and the next-token distribution at the first σ token (top-k ids
    and log-probs, plus the full log-probs for comparing against another side's top-k)."""
    import torch

    dev = _device(model)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    vocab = base.get_output_embeddings().weight.shape[0]
    out = []
    for it in items:
        prompt_ids = tok(_prompt(tok, it["problem"].text, prompt_id, chat_kwargs), add_special_tokens=False)["input_ids"]
        enc = tok(it["continuation"], add_special_tokens=False, return_offsets_mapping=True)
        lo, hi = it["sigma_span"]
        sigma_j = next(j for j, (s, e) in enumerate(enc["offset_mapping"]) if e > lo)
        ids = prompt_ids + enc["input_ids"]
        n_p = len(prompt_ids)
        with torch.inference_mode(), _logits_mode():
            logits = model(input_ids=torch.tensor([ids], device=dev)).logits[0]
        if logits.shape[-1] != vocab or logits.shape[0] != len(ids):
            raise RuntimeError(f"forward returned logits of shape {tuple(logits.shape)}, expected ({len(ids)}, {vocab})")
        logp = torch.log_softmax(logits[n_p - 1:-1].float(), -1)  # row t predicts continuation token t
        tgt = torch.tensor(enc["input_ids"], device=logp.device)
        tok_lp = logp.gather(1, tgt[:, None])[:, 0]
        dist = logp[sigma_j]
        top = torch.topk(dist, top_k)
        out.append({"token_logprobs": tok_lp.cpu().numpy().astype(np.float64), "sigma_j": sigma_j,
                    "top_ids": top.indices.cpu().tolist(), "top_logprobs": top.values.cpu().double().tolist(),
                    "dist": dist.cpu().double().numpy()})
    return out


def _sample(model, tok, items: list[dict], prompt_id: str, chat_kwargs: dict, s: dict, eos_ids: set[int]) -> list[str]:
    """k completions per prompt, prompt-major, batched left-padded with explicit decoding (as the eval backend)."""
    import torch

    dev = _device(model)
    prompts = [_prompt(tok, it["problem"].text, prompt_id, chat_kwargs) for it in items]
    old_side = tok.padding_side
    tok.padding_side = "left"
    pad = tok.pad_token_id if tok.pad_token_id is not None else next(iter(eos_ids))
    per_batch = max(1, s["batch_size"] // s["k"])
    texts: list[str] = []
    torch.manual_seed(s["seed"])
    try:
        for start in range(0, len(prompts), per_batch):
            enc = tok(prompts[start:start + per_batch], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            with torch.inference_mode(), _logits_mode():
                gen = model.generate(**enc, do_sample=True, temperature=s["temperature"], top_p=s["top_p"],
                                     top_k=s["top_k"], max_new_tokens=s["max_new_tokens"],
                                     num_return_sequences=s["k"], eos_token_id=sorted(eos_ids), pad_token_id=pad,
                                     repetition_penalty=1.0)
            new = gen[:, enc["input_ids"].shape[1]:]
            texts += tok.batch_decode(new, skip_special_tokens=True)
    finally:
        tok.padding_side = old_side
    return texts


def _sigma_rel(items: list[dict], texts: list[str], k: int, floor: float) -> list[float | None]:
    """The verifier's stated relative σ per completion (None if the answer does not parse or has no σ)."""
    from physcal.verify.reward import reward, reward_config

    rw = reward_config()
    out = []
    for n, t in enumerate(texts):
        _, b = reward(items[n // k]["problem"], t, rw)
        ok = b["reason"] == "ok" and b.get("sigma_si") is not None and b.get("mu_si")
        out.append(max(b["sigma_si"] / abs(b["mu_si"]), floor) if ok else None)
    return out


def _eos_ids(model, tok) -> set[int]:
    eos = getattr(model.generation_config, "eos_token_id", None)
    ids = {e for e in (eos if isinstance(eos, list) else [eos]) if e is not None}
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0 and im_end != tok.unk_token_id:
        ids.add(im_end)
    return ids


def measure(model, tok, chat_kwargs: dict | None = None, adapter_off: bool = False, cfg: dict | None = None) -> dict:
    """Everything a fingerprint holds for one model state (`adapter_off` adds the adapter-disabled forward)."""
    cfg = cfg or _config()
    chat_kwargs = chat_kwargs or {}
    items = fixed_items(cfg)
    pid, top_k, s = cfg["prompts"]["prompt_id"], cfg["logits"]["top_k_sigma"], cfg["sampling"]
    on = _forward(model, tok, items, pid, chat_kwargs, top_k)
    res: dict[str, Any] = {"on": on}
    if adapter_off:
        with model.disable_adapter():
            res["off"] = _forward(model, tok, items, pid, chat_kwargs, top_k)
    texts = _sample(model, tok, items, pid, chat_kwargs, s, _eos_ids(model, tok))
    res["samples"] = {"texts": texts, "sigma_rel": _sigma_rel(items, texts, s["k"], s["sigma_floor_rel"])}
    res["seeds"] = [it["problem"].seed for it in items]
    return res


def _tv(dist_full: np.ndarray, ref_ids: list[int], ref_lp: list[float]) -> float:
    """TV distance between a full next-token distribution and a stored top-k one (the rest lumped into one bin)."""
    p_ref = np.exp(np.asarray(ref_lp))
    p = np.exp(dist_full[ref_ids])
    rest_ref, rest = max(0.0, 1 - p_ref.sum()), max(0.0, 1 - p.sum())
    return float(0.5 * (np.abs(p - p_ref).sum() + abs(rest - rest_ref)))


def _strip(m: list[dict]) -> list[dict]:
    return [{"token_logprobs": x["token_logprobs"].tolist(), "sigma_j": x["sigma_j"], "top_ids": x["top_ids"],
             "top_logprobs": x["top_logprobs"]} for x in m]


def record(model, tok, model_id: str, lora_dir: str | Path, out_dirs: list[str | Path], chat_kwargs: dict | None = None,
           context: dict | None = None, cfg: dict | None = None) -> dict:
    """Training side: measure the in-memory policy (adapter on and off) and write the fingerprint into each dir.
    The caller puts an Unsloth model in inference mode first (as the eval loader does)."""
    cfg = cfg or _config()
    m = measure(model, tok, chat_kwargs, adapter_off=True, cfg=cfg)
    on, off = m["on"], m["off"]
    effect = float(np.mean(np.concatenate([np.abs(a["token_logprobs"] - b["token_logprobs"]) for a, b in zip(on, off)])))
    tv_effect = float(np.mean([_tv(b["dist"], a["top_ids"], a["top_logprobs"]) for a, b in zip(on, off)]))
    fp = {"decision": "D72 (R2-1)", "config": cfg, "context": context or {}, "seeds": m["seeds"],
          "provenance": provenance(model, model_id, lora_dir), "adapter_effect": {"mean_abs_dlogp": effect,
                                                                                "mean_sigma_tv": tv_effect},
          "on": _strip(on), "off": _strip(off), "samples": m["samples"]}
    for d in out_dirs:
        (Path(d) / cfg["file_name"]).write_text(json.dumps(fp))
    return fp


def _median_log(xs: list[float | None]) -> float | None:
    v = [math.log(x) for x in xs if x is not None]
    return float(np.median(v)) if v else None


def compare(reloaded: dict, fp: dict, prov: dict, cfg: dict | None = None) -> dict:
    """The regression verdict for a reloaded model's `measure` output against a stored fingerprint."""
    cfg = cfg or fp["config"]
    tol, s = cfg["tolerances"], cfg["sampling"]
    if reloaded["seeds"] != fp["seeds"]:
        raise ValueError("the reloaded model was measured on different prompts than the fingerprint")
    on = reloaded["on"]
    dlogp = float(np.mean(np.concatenate([np.abs(a["token_logprobs"] - np.asarray(b["token_logprobs"]))
                                          for a, b in zip(on, fp["on"])])))
    tv = float(np.mean([_tv(a["dist"], b["top_ids"], b["top_logprobs"]) for a, b in zip(on, fp["on"])]))
    eff, tv_eff = fp["adapter_effect"]["mean_abs_dlogp"], fp["adapter_effect"]["mean_sigma_tv"]
    crit: dict[str, Any] = {
        "logprob": {"mean_abs_dlogp": dlogp, "adapter_effect": eff, "ratio": dlogp / eff if eff > 0 else math.inf,
                    "max_ratio": tol["max_logprob_ratio"]},
        "sigma_token": {"mean_tv": tv, "adapter_effect": tv_eff, "ratio": tv / tv_eff if tv_eff > 0 else math.inf,
                        "max_ratio": tol["max_sigma_tv_ratio"]},
    }
    crit["logprob"]["pass"] = crit["logprob"]["ratio"] <= tol["max_logprob_ratio"]
    crit["sigma_token"]["pass"] = crit["sigma_token"]["ratio"] <= tol["max_sigma_tv_ratio"]
    a, b = reloaded["samples"], fp["samples"]
    n_a, n_b = sum(x is not None for x in a["sigma_rel"]), sum(x is not None for x in b["sigma_rel"])
    ma, mb = _median_log(a["sigma_rel"]), _median_log(b["sigma_rel"])
    evaluable = min(n_a, n_b) >= s["min_parsed"]
    crit["sigma_samples"] = {"n_parsed_reloaded": n_a, "n_parsed_training": n_b, "median_ln_sigma_reloaded": ma,
                             "median_ln_sigma_training": mb, "evaluable": evaluable,
                             "abs_dlog_median": abs(ma - mb) if evaluable else None,
                             "max_abs_dlog_median": tol["max_abs_dlog_median_sigma"],
                             "share_identical_texts": float(np.mean([x == y for x, y in zip(a["texts"], b["texts"])])),
                             "pass": (abs(ma - mb) <= tol["max_abs_dlog_median_sigma"]) if evaluable else None}
    fp_prov = fp["provenance"]
    mism = {k: {"reloaded": prov.get(k), "training": fp_prov.get(k)} for k in cfg["must_match"]
            if prov.get(k) != fp_prov.get(k)}
    crit["provenance"] = {"must_match": cfg["must_match"], "mismatches": mism, "pass": not mism,
                          "other_differences": sorted(k for k in ("versions",) if prov.get(k) != fp_prov.get(k))}
    # D77: below the log-prob floor, the σ-token criterion decides (`below_floor: sigma_token`); the log-prob
    # criterion is reported but not decisive. A fingerprint recorded before D77 has no `below_floor` key and keeps
    # D72's rule (any failure fails, otherwise "inconclusive"): the rule travels with the fingerprint, not retroactive.
    below = eff < tol["min_adapter_effect"]
    rule = cfg.get("below_floor", "inconclusive")
    if rule not in ("inconclusive", "sigma_token"):
        raise ValueError(f"unknown below_floor rule {rule!r}")
    by_sigma_token = below and rule == "sigma_token"
    crit["logprob"]["decisive"] = not by_sigma_token
    crit["sigma_token"]["decisive"] = not (by_sigma_token and tv_eff <= 0)  # no σ-token effect: nothing decides
    passes = [c["pass"] for c in crit.values() if c["pass"] is not None and c.get("decisive", True)]
    if not all(passes):
        status = "fail"
    elif below and not (by_sigma_token and crit["sigma_token"]["decisive"]):
        status = "inconclusive"
    else:
        status = "pass"
    return {"status": status, "criteria": crit, "below_floor_rule": rule if below else None}


def find(lora_dir: str | Path, cfg: dict | None = None) -> tuple[dict | None, str | None]:
    """The fingerprint recorded for these adapter weights: in the adapter dir itself, or in its run's final_adapter
    (a permanent checkpoint of the final step), matched by tensor hash. (None, reason) if there is none."""
    cfg = cfg or _config()
    lora_dir = Path(lora_dir)
    want = adapter_hashes(lora_dir)["tensors"]
    cands = [lora_dir] + [p / "final_adapter" for p in (lora_dir.parent, lora_dir.parent.parent)]
    for c in cands:
        f = c / cfg["file_name"]
        if f.exists():
            fp = json.loads(f.read_text())
            if fp["provenance"]["adapter"]["tensors"] == want:
                return fp, str(f)
    return None, "no fingerprint recorded for these adapter weights"


def check_adapter(model, tok, model_id: str, lora_dir: str | Path, chat_kwargs: dict | None = None,
                  cfg: dict | None = None) -> dict:
    """Eval side: provenance of the reloaded model, and the regression verdict if a fingerprint exists. With
    `enforce` (D72; on since D77) a "fail" raises and blocks the eval; "inconclusive" and "no_fingerprint" only warn."""
    import sys

    cfg = cfg or _config()
    prov = provenance(model, model_id, lora_dir)
    fp, where = find(lora_dir, cfg)
    if fp is None:
        res = {"status": "no_fingerprint", "reason": where, "provenance": prov}
    else:
        res = compare(measure(model, tok, chat_kwargs, cfg=fp["config"]), fp, prov, fp["config"])
        res.update(fingerprint=where, provenance=prov)
    if res["status"] == "fail" and cfg["enforce"]:
        raise RuntimeError(f"R2-1 regression check FAILED for {lora_dir}: {json.dumps(res['criteria'], default=str)}")
    if res["status"] in ("inconclusive", "no_fingerprint"):
        res["warning"] = (f"R2-1 check {res['status']} for {lora_dir}: the eval goes on, but this loader is not "
                          "certified for these weights (D77)")
        print(f"WARNING: {res['warning']}", file=sys.stderr, flush=True)
    return res


def record_after_training(model, tok, model_id: str, final_dir: str | Path, chat_kwargs: dict | None,
                          context: dict, events: Path | None = None, prepare=None, cfg: dict | None = None) -> dict:
    """The trainers' hook, called after the final adapter is saved (D72). `prepare` puts the model in the eval
    loader's mode (Unsloth: FastLanguageModel.for_inference). Never fails the run: an error is logged and returned.
    `cfg` (D137): another fingerprint config (stored in the fingerprint, so the check reuses it); default this file's."""
    import time
    import traceback

    t0 = time.time()
    try:
        if prepare is not None:
            prepare(model)
        fp = record(model, tok, model_id, final_dir, [final_dir], chat_kwargs, context, cfg)
        res = {"status": "recorded", "file": str(Path(final_dir) / _config()["file_name"]),
               "adapter_effect": fp["adapter_effect"], "seconds": time.time() - t0}
    except Exception as exc:  # noqa: BLE001 (the adapter is already saved; the check is reported, never fatal)
        res = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:1000],
               "traceback": traceback.format_exc()[-3000:], "seconds": time.time() - t0}
    if events is not None:
        with Path(events).open("a") as f:
            f.write(json.dumps({"event": "fingerprint", "time": time.time(), **res}) + "\n")
    return res

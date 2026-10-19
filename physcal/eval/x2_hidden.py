"""X2 methods 5 / 6 (D74): hidden states at the first token of "±" in `<answer>`, every layer.

    uv run python -m physcal.eval.x2_hidden make-random-init             # CPU, once: the random-init control's weights
    uv run python -m physcal.eval.x2_hidden extract --model sft|random_init --set fit|x2   # GPU

Sets: "fit" = D69's 4-bit SFT greedy calibration completions (training seeds 2e7..); "x2" = every answer of the kept
X2 problems. Both keep rows with a parsed answer (μ > 0) and a "±" inside `<answer>`. The position is the token that
contains the first character of that "±": its input holds the complete numeric answer.

Models: the corrected SFT model (the merged SFT dir, quantized to 4 bits at load by Unsloth, as the eval loader does)
and a random-init model of the same architecture (seed 0, saved once, then loaded the same way). Extraction is a
teacher-forced forward of prompt + completion with right padding. The Unsloth path is checked on the first batch:
lm_head of the last returned state (with or without the final norm) must reproduce the model's own logits at the
position; if it does not, or Unsloth returns no states, the model is loaded by transformers + bitsandbytes with the
same quantization config (D74's fallback), which is recorded.

Writes `<hidden.output_dir>/<model>/<set>/layer_XX.npy` (float16, rows in index order), `index.json` and `info.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config


def _config() -> dict:
    return load_config("x2")


def pm_char(completion: str) -> int | None:
    """Character offset of the first ± marker inside the (single) <answer> block, or None."""
    from physcal.train.e1 import PM_RE, truncate_after_pm

    if completion.count("<answer>") != 1 or truncate_after_pm(completion) is None:
        return None
    return PM_RE.search(completion, completion.index("<answer>")).start()


def token_at(offsets: list[tuple[int, int]], char: int) -> int:
    """Index of the token whose character span contains `char`."""
    for i, (a, b) in enumerate(offsets):
        if a <= char < b:
            return i
    raise ValueError(f"no token covers character {char}")


def row(r: dict, tolerance: float, prompt_id: str) -> dict | None:
    """An extraction row from a tape record: parsed answer (μ > 0) with a ± marker, else None."""
    from physcal.eval.nested_baselines import record

    b, p = r["breakdown"], r["problem"]
    rec = record(p, r["completion"], b, tolerance)
    at = pm_char(r["completion"])
    if not rec["ok"] or at is None:
        return None
    return {"key": [p["family"], p["depth"], p["seed"]], "sample": r.get("sample", 0), "template": rec["template"],
            "family": p["family"], "depth": p["depth"], "correct": rec["correct"], "abs_rel_error": b["abs_rel_error"],
            "text": p["text"], "completion": r["completion"], "pm_char": at, "prompt_id": prompt_id}


def records(which: str, cfg: dict | None = None) -> list[dict]:
    from physcal.tape.tape import read_meta, read_tape

    cfg = cfg or _config()
    tol = cfg["selection"]["tolerance"]
    if which == "fit":
        tapes = [REPO_ROOT / t for t in cfg["hidden"]["fit_tapes"]]
        src = [(r, read_meta(t)["prompt_id"]) for t in tapes for r in read_tape(t)]
    else:
        from physcal.eval.x2_pairs import kept_records, sampled_chunks

        prompt_id = read_meta(sampled_chunks(cfg)[0][1])["prompt_id"]
        src = [(r, prompt_id) for r in kept_records(cfg)[0]]
    return [x for x in (row(r, tol, pid) for r, pid in src) if x is not None]


def make_random_init(cfg: dict | None = None) -> dict:
    """The control's weights: the SFT model's architecture, random init (seed), bf16, with the same tokenizer."""
    import torch
    import transformers

    from physcal.eval.fingerprint import _dir_content_hash

    cfg = cfg or _config()
    h = cfg["hidden"]
    src, out = REPO_ROOT / h["model_dir"], REPO_ROOT / h["random_init_dir"]
    torch.manual_seed(h["random_init_seed"])
    config = transformers.AutoConfig.from_pretrained(str(src))
    model = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    model.save_pretrained(str(out))
    transformers.AutoTokenizer.from_pretrained(str(src)).save_pretrained(str(out))
    info = {"decision": "D74 (X2 random-init control)", "seed": h["random_init_seed"], "architecture_from": h["model_dir"],
            "content_sha256": _dir_content_hash(out)}
    (out / "physcal_random_init.json").write_text(json.dumps(info, indent=1))
    return info


def _batches(recs: list[dict], tok, bs: int):
    from physcal.prompt import build_messages

    for i in range(0, len(recs), bs):
        batch = recs[i:i + bs]
        prompts = [tok.apply_chat_template(build_messages(r["text"], r["prompt_id"]), tokenize=False,
                                           add_generation_prompt=True) for r in batch]
        enc = tok([p + r["completion"] for p, r in zip(prompts, batch)], return_tensors="pt", padding=True,
                  return_offsets_mapping=True)
        pos = [token_at(enc["offset_mapping"][j].tolist(), len(prompts[j]) + r["pm_char"]) for j, r in enumerate(batch)]
        yield enc, pos


def _states(model, enc, pos, device: str):
    """Every layer's state at the positions: list of (batch, hidden) float tensors, or None if not returned."""
    import torch

    rows = torch.arange(len(pos), device=device)
    idx = torch.tensor(pos, device=device)
    with torch.inference_mode():
        out = model.model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device),
                          output_hidden_states=True, use_cache=False)
    hs = getattr(out, "hidden_states", None)
    if hs is None or len(hs) != model.config.num_hidden_layers + 1:
        return None
    return [h[rows, idx].float() for h in hs]


def logit_check(model, enc, device: str) -> dict:
    """On one unpadded sequence: lm_head of the last returned state (as returned, and through the final norm) against
    the model's own logits at the last token, as max |diff| / max |logit|. Works whether the model returns logits for
    every position or only the last."""
    import torch

    n = enc["input_ids"].shape[1]
    states = _states(model, enc, [n - 1], device)
    if states is None:
        return {"returned_states": False}
    with torch.inference_mode():
        ref = model(input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device)).logits[:, -1].float()
        head = model.get_output_embeddings()
        last = states[-1].to(head.weight.dtype)
        cands = {"as_returned": head(last).float(), "normed": head(model.model.norm(last)).float()}
    scale = float(ref.abs().max())
    gaps = {k: float((v - ref).abs().max()) / scale for k, v in cands.items()}
    best = min(gaps, key=gaps.get)
    return {"returned_states": True, "n_states": len(states), "relative_gap": gaps, "best": best,
            "argmax_agree": bool(cands[best].argmax(-1).eq(ref.argmax(-1)).all())}


def _passes(check: dict, rtol: float) -> bool:
    return bool(check.get("returned_states")) and min(check["relative_gap"].values()) <= rtol


def load_unsloth(model_dir: Path, max_seq_length: int):
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    model, _ = FastLanguageModel.from_pretrained(model_name=str(model_dir), max_seq_length=max_seq_length,
                                                 load_in_4bit=True, fast_inference=False, dtype=None)
    FastLanguageModel.for_inference(model)
    return model


def load_transformers_bnb(model_dir: Path, quantization_config: dict):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    keep = {k: v for k, v in quantization_config.items()
            if k in BitsAndBytesConfig().to_dict() and not k.startswith("_") and k != "quant_method"}
    for k in ("bnb_4bit_compute_dtype", "bnb_4bit_quant_storage"):
        if isinstance(keep.get(k), str):
            keep[k] = getattr(torch, keep[k])
    return AutoModelForCausalLM.from_pretrained(str(model_dir), quantization_config=BitsAndBytesConfig(**keep),
                                                dtype=torch.bfloat16, device_map="cuda").eval()


def extract(model_name: str, which: str, cfg: dict | None = None) -> dict:
    import torch
    import transformers

    from physcal.eval.fingerprint import _dir_content_hash, _jsonable, _version

    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"  # GRPO's switch: forward would return states instead of logits
    cfg = cfg or _config()
    h = cfg["hidden"]
    model_dir = REPO_ROOT / (h["model_dir"] if model_name == "sft" else h["random_init_dir"])
    recs = records(which, cfg)
    tok = transformers.AutoTokenizer.from_pretrained(str(model_dir))
    tok.padding_side = "right"  # causal model: right padding leaves every real token's state unchanged
    t0 = time.time()
    model = load_unsloth(model_dir, h["max_seq_length"])
    qc = model.config.quantization_config
    qc = _jsonable(qc.to_dict() if hasattr(qc, "to_dict") else qc)
    path = "unsloth_4bit"
    first, _ = next(_batches(recs[:1], tok, 1))
    check = logit_check(model, first, "cuda")
    if not _passes(check, h["logit_check_rtol"]):
        print(f"Unsloth hidden-state check failed ({check}); falling back to transformers + bitsandbytes", flush=True)
        del model
        torch.cuda.empty_cache()
        model, path = load_transformers_bnb(model_dir, qc), "transformers_bnb"
        check = {"unsloth": check, "fallback": logit_check(model, first, "cuda")}
        if not _passes(check["fallback"], h["logit_check_rtol"]):
            raise RuntimeError(f"hidden states fail the logit check on both paths: {check}")
    layers: list[list[np.ndarray]] | None = None
    for n, (enc, pos) in enumerate(_batches(recs, tok, h["batch_size"])):
        states = _states(model, enc, pos, "cuda")
        if layers is None:
            layers = [[] for _ in states]
        for k, s in enumerate(states):
            layers[k].append(s.cpu().numpy().astype(np.float16))
        if n % 50 == 0:
            print(f"  {model_name}/{which}: {min((n + 1) * h['batch_size'], len(recs))}/{len(recs)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    out = REPO_ROOT / h["output_dir"] / model_name / which
    out.mkdir(parents=True, exist_ok=True)
    for k, arr in enumerate(layers):
        a = np.concatenate(arr)
        if not np.isfinite(a).all():
            raise ValueError(f"non-finite hidden states at layer {k}")
        np.save(out / f"layer_{k:02d}.npy", a)
    keep = ("key", "sample", "template", "family", "depth", "correct", "abs_rel_error")
    (out / "index.json").write_text(json.dumps([{k: r[k] for k in keep} for r in recs]))
    info = {"decision": "D74 (X2)", "model": model_name, "set": which, "n": len(recs), "n_layers": len(layers),
            "path": path, "check": check, "model_dir": str(model_dir.relative_to(REPO_ROOT)),
            "model_content_sha256": _dir_content_hash(model_dir), "quantization_config": qc,
            "versions": {p: _version(p) for p in ("torch", "transformers", "bitsandbytes", "unsloth")},
            "seconds": time.time() - t0, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
    (out / "info.json").write_text(json.dumps(info, indent=1, default=str))
    return info


def load(model_name: str, which: str, cfg: dict | None = None) -> tuple[list[np.ndarray], list[dict]]:
    """The saved states (float32, per layer) and their index."""
    cfg = cfg or _config()
    d = REPO_ROOT / cfg["hidden"]["output_dir"] / model_name / which
    index = json.loads((d / "index.json").read_text())
    return [np.load(p).astype(np.float32) for p in sorted(d.glob("layer_*.npy"))], index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["make-random-init", "extract"])
    ap.add_argument("--model", choices=["sft", "random_init"], default="sft")
    ap.add_argument("--set", choices=["fit", "x2"], default="fit")
    args = ap.parse_args()
    if args.action == "make-random-init":
        print(json.dumps(make_random_init(), indent=1))
    else:
        info = extract(args.model, args.set)
        print(json.dumps({k: v for k, v in info.items() if k != "quantization_config"}, indent=1, default=str))


if __name__ == "__main__":
    main()

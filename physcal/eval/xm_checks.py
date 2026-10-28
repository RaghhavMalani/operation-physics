"""D137 item 2: two GPU checks before the cross-model pilot, on development problems only.

    uv run python -m physcal.eval.xm_checks equivalence qwen     # and: equivalence llama
    uv run python -m physcal.eval.xm_checks batch-fit

`equivalence`: 30 development problems (A6c + A6b, 3 per depth per set) through run_eval's HFBackend twice, without and
with `last_logits_only` (logits_to_keep = 1 at prefill); the change is accepted only if every greedy completion is
byte-identical, for Qwen (stage R4's D-pos-v2 seed-0 adapter, stage R4's eval config) and for the Llama base. A forward
hook records the prefill logits' sequence length in each run, i.e. what the generate path really computes.
`batch-fit`: the Llama base on the longest development prompts, every completion forced to the full budget (the
worst-case KV cache), batch sizes from the largest down; the first that runs under the cap without OOM is the answer.
"""

from __future__ import annotations

import argparse
import copy
import json
import os

from physcal.config import REPO_ROOT, load_config


def _config() -> dict:
    return load_config("stage_xm")


def problems(cfg: dict) -> list[dict]:
    from physcal.eval import stage_a

    ck = cfg["checks"]
    allp = stage_a.load_sets(load_config("stage_a"))
    out = []
    for s in ck["sets"]:
        for d in sorted({p["depth"] for p in allp}):
            out += [p for p in allp if p["_set"] == s and p["depth"] == d][:ck["n_per_depth_per_set"]]
    return out


def llama_model_cfg(cfg: dict, cap: float) -> dict:
    """The Llama base as stage_xm's eval loads it, without an adapter."""
    from physcal.base import base_config, pinned_base_dir

    pin = base_config(cfg["pinned_base"])
    m = pin["model"]
    return {"id": pinned_base_dir(pin), "pinned_base": cfg["pinned_base"], "revision": None, "lora": None,
            "loader": "unsloth_4bit", "model_class": "causal_lm", "max_seq_length": m["max_seq_length"],
            "text_only": m.get("text_only", False), "chat_template_kwargs": m.get("chat_template_kwargs") or {},
            "prompt_add_special_tokens": m.get("prompt_add_special_tokens", True), "max_reserved_gib": cap}


def _backend(which: str, cfg: dict, cap: float):
    from physcal.eval.run_eval import HFBackend
    from physcal.train import stage_r4

    if which == "qwen":
        ev = stage_r4.eval_config(cfg["checks"]["qwen_arm"], seed=0, cap_gib=cap)
        gen = copy.deepcopy(ev["generation"])
        gen["batch_size"] = cfg["checks"]["qwen_batch_size"]
        return HFBackend(ev["model"]["id"], ev["model"], gen), gen, ev["prompt"]
    gen = copy.deepcopy(stage_r4.eval_config("dpos2", seed=0)["generation"])
    gen.update(batch_size=cfg["checks"]["llama_batch_size"], max_new_tokens=cfg["eval"]["max_new_tokens"])
    mc = llama_model_cfg(cfg, cap)
    return HFBackend(mc["id"], mc, gen), gen, cfg["eval"]["prompt"]


def _causal_lm(model):
    for m in model.modules():
        if type(m).__name__.endswith("ForCausalLM"):
            return m
    return model


def equivalence(which: str, cfg: dict | None = None) -> dict:
    from physcal.eval.r5_feasibility import launch_cap
    from physcal.prompt import build_messages

    cfg = cfg or _config()
    cap = launch_cap({"cap_max_gib": cfg["max_reserved_gib"], "cap_margin_gib": 0.5})
    backend, gen, prompt = _backend(which, cfg, cap)
    probs = problems(cfg)
    convs = [build_messages(p["text"], prompt) for p in probs]
    seen: list[int] = []

    def hook(_mod, _args, kwargs, out):
        ids = kwargs.get("input_ids")
        if ids is not None and ids.shape[1] > 1:  # a prefill call (decoding steps feed one token)
            seen.append(int(out.logits.shape[1]))

    h = _causal_lm(backend.model).register_forward_hook(hook, with_kwargs=True)
    runs = {}
    for flag in (False, True):
        seen.clear()
        gen["last_logits_only"] = flag
        backend.gen_cfg = gen
        outs = backend(convs)
        runs[flag] = {"texts": [o.text for o in outs], "n_tokens": [o.n_tokens for o in outs],
                      "prefill_logits_len": sorted(set(seen)), "oom_backoffs": backend.oom_backoffs}
    h.remove()
    a, b = runs[False], runs[True]
    same = [x == y for x, y in zip(a["texts"], b["texts"])]
    res = {"decision": "D137 item 2", "model": which, "n_problems": len(probs), "batch_size": gen["batch_size"],
           "max_new_tokens": gen["max_new_tokens"], "cap_gib": cap,
           "byte_identical": all(same) and a["n_tokens"] == b["n_tokens"], "n_identical": sum(same),
           "prefill_logits_len_without": a["prefill_logits_len"], "prefill_logits_len_with": b["prefill_logits_len"],
           "oom_backoffs": b["oom_backoffs"], "mismatches": [i for i, s in enumerate(same) if not s]}
    out = REPO_ROOT / cfg["checks"]["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"equivalence_{which}.json").write_text(json.dumps(res, indent=1))
    with open(out / f"equivalence_{which}_completions.jsonl", "w") as fh:
        for p, x, y in zip(probs, a["texts"], b["texts"]):
            fh.write(json.dumps({"set": p["_set"], "seed": p["seed"], "depth": p["depth"], "without": x, "with": y}) + "\n")
    return res


def batch_fit(cfg: dict | None = None) -> dict:
    """The largest batch of the longest development prompts that generates the full budget under the cap."""
    import torch

    from physcal.eval.r5_feasibility import launch_cap
    from physcal.prompt import build_messages

    cfg = cfg or _config()
    ck = cfg["checks"]
    cap = launch_cap({"cap_max_gib": cfg["max_reserved_gib"], "cap_margin_gib": 0.5})
    backend, gen, prompt = _backend("llama", cfg, cap)
    from physcal.eval import stage_a

    allp = [p for p in stage_a.load_sets(load_config("stage_a")) if p["_set"] in ck["sets"]]
    tok = backend.tok
    texts = [tok.apply_chat_template(build_messages(p["text"], prompt), tokenize=False, add_generation_prompt=True,
                                     **backend.chat_template_kwargs) for p in allp]
    texts.sort(key=lambda t: -len(tok(t, add_special_tokens=False)["input_ids"]))
    keep = {"logits_to_keep": 1} if cfg["eval"]["last_logits_only"] else {}
    tried = []
    for b in ck["batch_candidates"]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        enc = tok(texts[:b], return_tensors="pt", padding=True, add_special_tokens=backend.add_special_tokens).to("cuda")
        try:
            with torch.inference_mode():
                backend.model.generate(**enc, max_new_tokens=gen["max_new_tokens"], min_new_tokens=gen["max_new_tokens"],
                                       do_sample=False, temperature=None, top_p=None, top_k=None,
                                       eos_token_id=sorted(backend.eos_ids), pad_token_id=backend.pad_id, **keep)
            ok = True
        except torch.OutOfMemoryError:
            ok = False
        del enc
        tried.append({"batch": b, "fits": ok, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                      "prompt_tokens_max": len(tok(texts[0], add_special_tokens=False)["input_ids"])})
        print(tried[-1], flush=True)
        if ok:
            break
    res = {"decision": "D137 item 2", "cap_gib": cap, "max_new_tokens": gen["max_new_tokens"],
           "last_logits_only": cfg["eval"]["last_logits_only"], "tried": tried,
           "largest_fitting": next((t["batch"] for t in tried if t["fits"]), None)}
    out = REPO_ROOT / ck["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "batch_fit.json").write_text(json.dumps(res, indent=1))
    return res


def prefill_probe(which: str, cfg: dict | None = None) -> dict:
    """What the generate path projects through lm_head at prefill: the sequence length of the first lm_head call of
    one generate on 4 development prompts, with no logits_to_keep argument (the eval as it was), with 1, and with 0
    (every position). PEFT calls the inner forward directly, so the hook is on lm_head itself."""
    import torch

    from physcal.eval.r5_feasibility import launch_cap
    from physcal.prompt import build_messages

    cfg = cfg or _config()
    cap = launch_cap({"cap_max_gib": cfg["max_reserved_gib"], "cap_margin_gib": 0.5})
    backend, gen, prompt = _backend(which, cfg, cap)
    tok = backend.tok
    texts = [tok.apply_chat_template(build_messages(p["text"], prompt), tokenize=False, add_generation_prompt=True,
                                     **backend.chat_template_kwargs) for p in problems(cfg)[:4]]
    lm_head = _causal_lm(backend.model).lm_head
    shapes: list[tuple] = []
    h = lm_head.register_forward_hook(lambda _m, inp, _out: shapes.append(tuple(inp[0].shape)))
    res = {"decision": "D137 item 2", "model": which}
    for name, kw in (("no_argument", {}), ("logits_to_keep_1", {"logits_to_keep": 1}), ("logits_to_keep_0", {"logits_to_keep": 0})):
        shapes.clear()
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=backend.add_special_tokens).to("cuda")
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            backend.model.generate(**enc, max_new_tokens=4, do_sample=False, temperature=None, top_p=None, top_k=None,
                                   eos_token_id=sorted(backend.eos_ids), pad_token_id=backend.pad_id, **kw)
        res[name] = {"prompt_len": int(enc["input_ids"].shape[1]), "prefill_lm_head_input": list(shapes[0]) if shapes else None,
                     "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
    h.remove()
    out = REPO_ROOT / cfg["checks"]["output_dir"]
    (out / f"prefill_probe_{which}.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["equivalence", "batch-fit", "prefill-probe"])
    ap.add_argument("model", nargs="?", choices=["qwen", "llama"])
    a = ap.parse_args()
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    res = {"equivalence": lambda: equivalence(a.model), "batch-fit": batch_fit,
           "prefill-probe": lambda: prefill_probe(a.model)}[a.what]()
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

"""D81: the largest generate batch (sequences per call) at which the corrected SFT model samples X2, worst case.

    HF_HUB_OFFLINE=1 uv run python scripts/x2_batch_probe.py

The model and decoding are exactly X2's sampling settings (x2_pairs.sample_config, the 7.5 GiB allocator cap). For
each candidate batch size, largest first: the pool's longest prompts (k samples each), every sequence forced to the
full new-token budget (min_new_tokens), one generate call. The first size that completes without OOM is the night's
batch size. Writes results/x2/batch_probe.json (trials, the choice, the measured seconds per worst-case batch).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import torch

    from physcal.eval.run_eval import HFBackend, _is_oom
    from physcal.eval.x2_pairs import _config, sample_config
    from physcal.prompt import build_messages

    cfg = _config()
    bp, s = cfg["batch_probe"], cfg["sampling"]
    ev = sample_config(0, max(bp["candidates"]), 0)
    g = ev["generation"]
    backend = HFBackend(ev["model"]["id"], ev["model"], g)
    tok, model = backend.tok, backend.model
    texts = [json.loads(line)["text"] for line in (ROOT / cfg["candidates"]["file"]).open()]
    prompts = sorted((tok.apply_chat_template(build_messages(t, ev["prompt"]), tokenize=False, add_generation_prompt=True)
                      for t in texts), key=len, reverse=True)
    trials, chosen = [], None
    for b in sorted(bp["candidates"], reverse=True):
        n = b // s["k"]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(s["seed"])
        t0 = time.time()
        try:
            enc = tok(prompts[:n], return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=s["max_new_tokens"], min_new_tokens=bp["min_new_tokens"],
                                     num_return_sequences=s["k"], do_sample=True, temperature=s["temperature"],
                                     top_p=s["top_p"], top_k=s["top_k"], repetition_penalty=1.0,
                                     eos_token_id=sorted(backend.eos_ids), pad_token_id=backend.pad_id)
            new = out.shape[1] - enc["input_ids"].shape[1]
            del out, enc
            ok, err = True, None
        except Exception as exc:  # noqa: BLE001
            if not _is_oom(exc):
                raise
            ok, err, new = False, f"{type(exc).__name__}: {str(exc)[:200]}", None
        sec = time.time() - t0
        trials.append({"batch_size": b, "prompts": n, "ok": ok, "error": err, "new_tokens": new, "seconds": sec,
                       "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                       "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                       "prompt_tokens_max": len(tok(prompts[0])["input_ids"])})
        print(json.dumps(trials[-1]), flush=True)
        if ok:
            chosen = trials[-1]
            break
    if chosen is None:
        raise SystemExit("no candidate batch size fits")
    per_chunk = math.ceil(s["chunk_size"] * s["k"] / chosen["batch_size"]) * chosen["seconds"]
    rep = {"decision": "D81", "chosen_batch_size": chosen["batch_size"], "worst_case_seconds_per_batch": chosen["seconds"],
           "worst_case_seconds_per_chunk": per_chunk, "chunk_size": s["chunk_size"], "k": s["k"],
           "max_reserved_gib_cap": s["max_reserved_gib"], "trials": trials}
    out = ROOT / bp["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(f"CHOSEN {chosen['batch_size']}  worst-case chunk {per_chunk / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

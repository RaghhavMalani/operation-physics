"""Hidden states at the pre-answer token, every layer, for logged completions (Phase 9.5 step 2; PREREGISTRATION P).

    uv run python -m physcal.interp.extract            # GPU: SFT-only and random-init model, fit + test sets

For each tape record: rebuild the exact prompt (chat template + generation prompt, as in run_eval's HFBackend),
append the logged completion, run one teacher-forced forward pass and keep the hidden state of every layer
(embedding output + each block) at the last token that starts before the `<answer>` tag. Records whose completion
did not parse (no single `<answer>` block) are skipped. Writes, per model and set,
`<output_dir>/hidden/<model>/<set>/layer_XX.npy` (float16, rows in index order) and `index.json` (one entry per
row: problem key, split, labels and the stated σ).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.prompt import build_messages
from physcal.tape.tape import read_meta, read_tape

ANSWER_TAG = "<answer>"


def answer_char(completion: str) -> int | None:
    """Character offset of the (single) `<answer>` tag, or None if the completion has none or several."""
    if completion.count(ANSWER_TAG) != 1:
        return None
    return completion.index(ANSWER_TAG)


def pre_answer_position(offsets: list[tuple[int, int]], answer_at: int) -> int:
    """Index of the last token that starts before character `answer_at` (the token right before `<answer>`)."""
    starts = [i for i, (a, b) in enumerate(offsets) if a < answer_at and b > a]
    if not starts:
        raise ValueError("no token before the answer tag")
    return starts[-1]


def records(tape: Path, tolerance: float, log_floor: float) -> list[dict]:
    """Parsed records with labels: correct (|rel err| <= tolerance), log10 |rel err|, stated relative σ."""
    meta = read_meta(tape)
    out = []
    for r in read_tape(tape):
        b, p = r["breakdown"], r["problem"]
        if b["reason"] != "ok" or answer_char(r["completion"]) is None:
            continue
        out.append({
            "key": [p["family"], p["depth"], p["seed"]], "split": r.get("split", "all"),
            "family": p["family"], "depth": p["depth"], "text": p["text"], "completion": r["completion"],
            "prompt_id": meta["prompt_id"], "abs_rel_error": b["abs_rel_error"],
            "correct": bool(b["abs_rel_error"] <= tolerance),
            "log_error": math.log10(max(b["abs_rel_error"], log_floor)),
            "rel_sigma": b.get("rel_sigma"),
        })
    return out


def load_model(model_dir: Path, random_init: bool, seed: int, dtype: str):
    import torch
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(str(model_dir))
    tok.padding_side = "right"  # causal model: right padding leaves every real token's state unchanged
    torch_dtype = getattr(torch, dtype)
    if random_init:
        torch.manual_seed(seed)
        config = transformers.AutoConfig.from_pretrained(str(model_dir))
        model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch_dtype)
    return tok, model.to("cuda").eval()


def extract(recs: list[dict], tok, model, batch_size: int) -> list[np.ndarray]:
    """Per layer: (len(recs), hidden) float16 array of pre-answer hidden states."""
    import torch

    layers: list[list[np.ndarray]] | None = None
    for i in range(0, len(recs), batch_size):
        batch = recs[i:i + batch_size]
        prompts = [tok.apply_chat_template(build_messages(r["text"], r["prompt_id"]), tokenize=False,
                                           add_generation_prompt=True) for r in batch]
        texts = [p + r["completion"] for p, r in zip(prompts, batch)]
        enc = tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
        positions = [pre_answer_position(enc["offset_mapping"][j].tolist(), len(prompts[j]) + answer_char(r["completion"]))
                     for j, r in enumerate(batch)]
        backbone = model.get_decoder()  # skip the LM head: full-vocab logits would cost ~2 GB per batch
        with torch.no_grad():
            out = backbone(input_ids=enc["input_ids"].to("cuda"), attention_mask=enc["attention_mask"].to("cuda"),
                           output_hidden_states=True, use_cache=False)
        idx = torch.tensor(positions, device="cuda")
        rows = torch.arange(len(batch), device="cuda")
        states = [h[rows, idx].float().cpu().numpy().astype(np.float16) for h in out.hidden_states]
        if layers is None:
            layers = [[] for _ in states]
        for k, s in enumerate(states):
            layers[k].append(s)
        print(f"  extracted {min(i + batch_size, len(recs))}/{len(recs)}", flush=True)
    return [np.concatenate(l) for l in layers]


def save(out: Path, recs: list[dict], layers: list[np.ndarray]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for k, arr in enumerate(layers):
        np.save(out / f"layer_{k:02d}.npy", arr)
    keep = ("key", "split", "family", "depth", "abs_rel_error", "correct", "log_error", "rel_sigma")
    (out / "index.json").write_text(json.dumps([{k: r[k] for k in keep} for r in recs]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["sft", "random_init"])
    args = ap.parse_args()
    cfg = load_config("probe_sft")
    pc, ec = cfg["probes"], cfg["extract"]
    sets = {"fit": [r for t in cfg["fit_tapes"] for r in records(REPO_ROOT / t, pc["correct_tolerance"],
                                                                 pc["log_error_floor"])],
            "test": [r for t in cfg["test_tapes"] for r in records(REPO_ROOT / t, pc["correct_tolerance"],
                                                                  pc["log_error_floor"])]}
    model_dir = REPO_ROOT / "results" / "sft" / cfg["label"] / "merged"
    for name in args.models:
        tok, model = load_model(model_dir, name == "random_init", ec["random_init_seed"], ec["dtype"])
        for set_name, recs in sets.items():
            print(f"{name} / {set_name}: {len(recs)} records", flush=True)
            save(REPO_ROOT / cfg["output_dir"] / "hidden" / name / set_name, recs, extract(recs, tok, model,
                                                                                            ec["batch_size"]))
        del model
        import torch

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

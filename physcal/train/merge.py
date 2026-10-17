"""Merge a LoRA adapter into the bf16 base and save a standalone model directory (CPU by default).

    uv run python -m physcal.train.merge --base Qwen/Qwen3.5-2B --revision <sha> --adapter <dir> --out <dir>

For Qwen3.5 the base is loaded text-only (AutoModelForCausalLM), matching how the adapter was trained. RL adapters
are refused (D68, FINDINGS #15): this tool exists to build the SFT model, and a merge erases an RL update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from physcal.config import REPO_ROOT


def merge(base: str, revision: str | None, adapter: str, out: str, device: str = "cpu") -> dict:
    from physcal.adapters import check_merge_allowed

    check_merge_allowed(adapter, requested=True)  # running this tool is the explicit request; RL adapters still refused
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, revision=revision, dtype=torch.bfloat16, device_map=device)
    peft_model = PeftModel.from_pretrained(model, adapter)
    n_lora = sum(1 for n, _ in peft_model.named_modules() if n.endswith("lora_A"))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
    weights = sorted(Path(adapter).glob("adapter_model*.safetensors"))
    info = {"base": base, "revision": revision, "adapter": adapter, "lora_modules_attached": n_lora,
            "adapter_sha256_16": hashlib.sha256(b"".join(w.read_bytes() for w in weights)).hexdigest()[:16],
            "model_class": type(merged).__name__}
    (Path(out) / "merge_info.json").write_text(json.dumps(info, indent=2))
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    print(json.dumps(merge(a.base, a.revision, a.adapter, a.out, a.device), indent=2))


if __name__ == "__main__":
    main()

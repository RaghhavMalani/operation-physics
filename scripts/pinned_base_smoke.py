"""D78 GPU check (~2 min): the pinned 4-bit base loads from its snapshot dir through Unsloth, an adapter made on it
names that dir, and reloading that adapter loads the same base. A capability check, not a result.

    HF_HUB_OFFLINE=1 uv run python scripts/pinned_base_smoke.py

1. resolve_model({pinned_base: base_4bit}) -> the snapshot dir (weights hash checked); Unsloth loads it 4-bit;
   assert_loaded_base; 4-bit bnb linear modules present; a short greedy generation runs.
2. A fresh LoRA with the RL settings is saved; its adapter_config names the pinned dir; check_model_source passes.
3. The adapter dir is reloaded through Unsloth (the D70 path RL uses for the SFT adapter): assert_loaded_base passes,
   and R2-1 provenance gives base_identity sha256:<the pinned content hash>.
Writes results/pinned_base_smoke/report.json and prints PASS / FAIL.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "pinned_base_smoke"


def main() -> int:
    os.chdir(ROOT)  # adapter bases are repo-relative
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    import torch
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    from physcal.base import assert_loaded_base, base_config, check_model_source, resolve_model
    from physcal.config import load_config
    from physcal.eval.fingerprint import provenance

    OUT.mkdir(parents=True, exist_ok=True)
    bcfg, lora = base_config(), load_config("grpo_overnight2")["lora"]
    m = {"pinned_base": "base_4bit", **bcfg["model"]}
    rep: dict = {"decision": "D78 smoke", "checks": {}}
    t0 = time.time()
    name, revision, pin = resolve_model(m)
    rep["pin"] = pin
    model, tok = FastLanguageModel.from_pretrained(model_name=name, revision=revision, max_seq_length=m["max_seq_length"],
                                                   load_in_4bit=True, fast_inference=False, dtype=None,
                                                   text_only=m["text_only"])
    rep["checks"]["loaded_base_is_pinned"] = assert_loaded_base(model, bcfg) is not None
    rep["n_bnb_4bit_linear"] = sum(type(x).__module__.startswith("bitsandbytes") and "4bit" in type(x).__name__
                                   for x in model.modules())
    rep["checks"]["is_4bit"] = rep["n_bnb_4bit_linear"] > 0
    FastLanguageModel.for_inference(model)
    enc = tok(tok.apply_chat_template([{"role": "user", "content": "1 + 1 = ?"}], tokenize=False,
                                      add_generation_prompt=True), return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=8, do_sample=False)
    rep["generation"] = tok.decode(out[0, enc["input_ids"].shape[1]:])
    model = FastLanguageModel.get_peft_model(model, r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                             target_modules=lora["target_modules"],
                                             use_gradient_checkpointing="unsloth", random_state=0)
    ad = OUT / "adapter"
    model.save_pretrained(str(ad))
    tok.save_pretrained(str(ad))
    named = json.loads((ad / "adapter_config.json").read_text())["base_model_name_or_path"]
    rep["adapter_base_model_name_or_path"] = named
    try:
        check_model_source(str(ad.relative_to(ROOT)), bcfg)
        rep["checks"]["adapter_names_pinned_dir"] = True
    except Exception as exc:  # noqa: BLE001
        rep["checks"]["adapter_names_pinned_dir"], rep["adapter_error"] = False, str(exc)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model, tok = FastLanguageModel.from_pretrained(model_name=str(ad.relative_to(ROOT)),
                                                   max_seq_length=m["max_seq_length"], load_in_4bit=True,
                                                   fast_inference=False, dtype=None, text_only=m["text_only"])
    try:
        assert_loaded_base(model, bcfg)
        rep["checks"]["reloaded_adapter_base_is_pinned"] = True
    except Exception as exc:  # noqa: BLE001
        rep["checks"]["reloaded_adapter_base_is_pinned"], rep["reload_error"] = False, str(exc)
    prov = provenance(model, pin["dir"], ad)
    rep["base_identity"] = prov["base_identity"]
    rep["checks"]["provenance_base_identity"] = prov["base_identity"] == f"sha256:{bcfg['content_sha256']}"
    rep["seconds"] = time.time() - t0
    rep["status"] = "PASS" if all(rep["checks"].values()) else "FAIL"
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    print(rep["status"])
    return 0 if rep["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

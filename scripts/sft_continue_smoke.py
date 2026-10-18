"""D70 GPU smoke test: the SFT LoRA adapter, loaded on its 4-bit base, is trainable and RL continues training it.

    HF_HUB_OFFLINE=1 uv run python scripts/sft_continue_smoke.py

1. Load results/sft/<label>/adapter with FastLanguageModel.from_pretrained(<adapter dir>, load_in_4bit=True).
   Checks: a PEFT model; the trainable parameters are exactly the LoRA tensors; the loaded LoRA weights equal the
   saved adapter; get_peft_model with the RL LoRA config takes the "already has LoRA adapters" path (same object,
   nothing re-initialized).
2. Two GRPO steps from that adapter with the real trainer (physcal.train.grpo.train; grpo_overnight2's config with
   model.id = the SFT adapter dir, run id sft-continue-smoke).
   Checks: 2 steps completed with a finite loss; the final adapter has the same tensors as the SFT adapter, the same
   base, and different values (RL moved the SFT adapter; it did not start a new one).
Writes results/sft_continue_smoke/report.json and prints PASS / FAIL. Not a result; a capability check (D70).
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the 4-bit base is in the local cache; never resolve `main` online

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LABEL = "qwen2.5-math-1.5b-instruct"
ADAPTER = ROOT / "results" / "sft" / LABEL / "adapter"
RUN_ID = "sft-continue-smoke"
OUT = ROOT / "results" / "sft_continue_smoke"


def lora_tensors(path: Path) -> dict:
    from safetensors.torch import load_file

    return load_file(str(path / "adapter_model.safetensors"))


def main() -> int:
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    import torch
    import unsloth  # noqa: F401
    from peft import PeftModel
    from unsloth import FastLanguageModel

    from physcal.config import load_config

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(load_config("grpo_overnight2"))
    m, lora = cfg["model"], cfg["lora"]
    rep: dict = {"decision": "D70 smoke", "adapter": str(ADAPTER.relative_to(ROOT)),
                 "adapter_config": json.loads((ADAPTER / "adapter_config.json").read_text()), "checks": {}}
    t0 = time.time()

    # ---- 1. load and inspect -------------------------------------------------------------------------------
    model, tok = FastLanguageModel.from_pretrained(model_name=str(ADAPTER), max_seq_length=m["max_seq_length"],
                                                   load_in_4bit=True, fast_inference=False, dtype=None,
                                                   text_only=m.get("text_only", False))
    rep["checks"]["is_peft_model"] = isinstance(model, PeftModel)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    rep["n_trainable_tensors"] = len(trainable)
    rep["checks"]["trainable_are_exactly_lora"] = bool(trainable) and all(".lora_A." in n or ".lora_B." in n
                                                                          for n in trainable)
    saved = lora_tensors(ADAPTER)
    params = {n.replace(".default", ""): p for n, p in model.named_parameters() if ".lora_" in n}
    rep["n_saved_tensors"] = len(saved)
    rep["checks"]["same_tensor_names"] = set(saved) == set(params)
    rep["max_abs_diff_loaded_vs_saved"] = max(float((params[k].detach().float().cpu() - v.float()).abs().max())
                                              for k, v in saved.items() if k in params) if params else None
    rep["checks"]["loaded_equals_saved"] = rep["max_abs_diff_loaded_vs_saved"] == 0.0
    same = FastLanguageModel.get_peft_model(model, r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                            target_modules=lora["target_modules"], use_gradient_checkpointing="unsloth",
                                            random_state=cfg["seed"])
    after = {n.replace(".default", ""): p for n, p in same.named_parameters() if ".lora_" in n}
    rep["checks"]["get_peft_model_skips"] = same is model and all(
        torch.equal(after[k].detach().cpu(), params[k].detach().cpu()) for k in params)
    rep["load_seconds"] = time.time() - t0
    del model, same, params, after
    gc.collect()
    torch.cuda.empty_cache()

    # ---- 2. two GRPO steps from the SFT adapter -------------------------------------------------------------------
    from physcal.train.grpo import train

    cfg["model"]["id"] = str(ADAPTER.relative_to(ROOT))
    cfg["run_id"] = RUN_ID
    cfg["grpo"]["permanent_save_steps"] = []
    t1 = time.time()
    summary = train(cfg, RUN_ID, max_steps=2, deadline=time.time() + 1500)
    rep["train_seconds"] = time.time() - t1
    rep["train_summary"] = {k: summary.get(k) for k in ("steps_completed", "train_loss", "seconds_per_step",
                                                         "peak_vram", "fingerprint")}
    rep["checks"]["two_steps"] = summary.get("steps_completed") == 2
    loss = summary.get("train_loss")
    rep["checks"]["finite_loss"] = loss is not None and math.isfinite(loss)
    final = ROOT / cfg["output_dir"] / RUN_ID / "final_adapter"
    fin = lora_tensors(final)
    rep["checks"]["final_same_tensors"] = set(fin) == set(saved) and all(fin[k].shape == saved[k].shape for k in saved)
    diffs = {k: float((fin[k].float() - saved[k].float()).abs().max()) for k in saved if k in fin}
    rep["max_abs_change_final_vs_sft"] = max(diffs.values()) if diffs else None
    rep["share_tensors_changed"] = sum(d > 0 for d in diffs.values()) / max(1, len(diffs))
    rep["checks"]["rl_moved_the_sft_adapter"] = bool(diffs) and rep["max_abs_change_final_vs_sft"] > 0
    fc = json.loads((final / "adapter_config.json").read_text())
    rep["final_base_model_name_or_path"] = fc.get("base_model_name_or_path")
    rep["checks"]["same_base"] = fc.get("base_model_name_or_path") == rep["adapter_config"].get("base_model_name_or_path")
    rep["pass"] = all(rep["checks"].values())
    rep["seconds"] = time.time() - t0
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"D70 SMOKE {'PASS' if rep['pass'] else 'FAIL'}: {json.dumps(rep['checks'])}", flush=True)
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

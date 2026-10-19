"""Phase 3.6: time-capped LoRA SFT on generator-written solutions (σ masked from the loss).

    uv run python -m physcal.train.sft --model qwen3.5-2b [--minutes 30]

Writes results/sft/<model>/{adapter/, sft_metrics.json, data_manifest.json}. Merge with physcal.train.merge.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from physcal.config import REPO_ROOT, load_config
from physcal.env.generator import env_config
from physcal.train.grpo import make_stop_callback
from physcal.train.sft_data import build_examples, tokenize


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="sft")
    ap.add_argument("--model", required=True)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--n-examples", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    m, t, lora, d = cfg["models"][args.model], cfg["train"], cfg["lora"], cfg["data"]
    out = REPO_ROOT / cfg["output_dir"] / args.model
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))

    import torch
    import unsloth  # noqa: F401
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
    from unsloth import FastLanguageModel

    from physcal.eval.run_eval import GpuMemoryPoller

    from physcal.base import assert_loaded_base, base_config, resolve_model

    model_name, revision, pin = resolve_model(m)  # D78: Phase 12 loads the pinned snapshot dir, never a hub name
    torch.cuda.reset_peak_memory_stats()
    model, tok = FastLanguageModel.from_pretrained(
        model_name=model_name, revision=revision, max_seq_length=m["max_seq_length"], load_in_4bit=True,
        fast_inference=False, dtype=None, text_only=m["text_only"],
    )
    if pin:
        assert_loaded_base(model, base_config(pin["config"]))
    model = FastLanguageModel.get_peft_model(
        model, r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], use_gradient_checkpointing="unsloth", random_state=cfg["seed"],
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    env_cfg = env_config()
    n = args.n_examples or d["n_examples"]
    examples = build_examples(n, d["seed_offset"], d["sigma_rel"], env_cfg)
    rows = [tokenize(e, tok, cfg["prompt"], m["chat_template_kwargs"]) for e in examples]
    rows = [r for r in rows if len(r["input_ids"]) <= m["max_seq_length"]]
    random.Random(cfg["seed"]).shuffle(rows)
    ds = Dataset.from_list([{k: r[k] for k in ("input_ids", "attention_mask", "labels")} for r in rows])
    manifest = {
        "n_examples": len(rows), "seed_range": [d["seed_offset"], d["seed_offset"] + n - 1],
        "eval_seed_offset": env_cfg["splits"]["eval_seed_offset"], "split": "train",
        "cells": sorted({f"{e.problem.family}/{e.problem.depth}" for e in examples}),
        "generator_version": examples[0].problem.generator_version, "prompt": cfg["prompt"],
        "completion_tokens_mean": sum(sum(1 for x in r["labels"] if x != -100) for r in rows) / len(rows),
        "sequence_tokens_max": max(len(r["input_ids"]) for r in rows),
        "sigma_tokens_masked_mean": sum(r["n_masked_sigma"] for r in rows) / len(rows),
    }
    (out / "data_manifest.json").write_text(json.dumps(manifest, indent=2))

    args_tr = TrainingArguments(
        output_dir=str(out / "checkpoints"), per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"], learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"], warmup_steps=t["warmup_steps"], weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"], optim=t["optim"], logging_steps=t["logging_steps"],
        max_steps=t["max_steps"], num_train_epochs=1, bf16=True, save_strategy="no", report_to="none",
        seed=cfg["seed"], remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args_tr, train_dataset=ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    g = cfg["guards"]
    minutes = args.minutes or t["wall_clock_minutes"]
    trainer.add_callback(make_stop_callback(minutes * 60, None, g["max_gpu_temp_c"], g["pause_seconds"],
                                            g["check_every_steps"], out / "events.jsonl"))
    t0 = time.time()
    with GpuMemoryPoller(interval_s=2.0) as poller:
        trainer.train()
    seconds = time.time() - t0
    model.save_pretrained(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))
    gib = 1024**3
    steps = trainer.state.global_step
    summary = {
        "model": args.model, "id": model_name, "revision": revision, "pinned_base": pin, "steps": steps,
        "examples_seen": steps * t["per_device_train_batch_size"] * t["gradient_accumulation_steps"],
        "seconds": seconds, "seconds_per_step": seconds / max(1, steps),
        "peak_vram": {"torch_max_allocated_gib": torch.cuda.max_memory_allocated() / gib,
                      "nvidia_smi_peak_used_gib_device_wide": poller.peak_mib / 1024},
        "log_history": trainer.state.log_history,
    }
    (out / "sft_metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "log_history"}, indent=2, default=str))


if __name__ == "__main__":
    main()

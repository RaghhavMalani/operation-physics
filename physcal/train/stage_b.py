"""Phase 12 stage B (DECISIONS.md D104, pre-registered; R3-4 B): the supervised σ* ceiling, 3 seeds.

    uv run python -m physcal.train.stage_b train --seed S [--stop-at ISO]     # GPU: SFT on D80's supervised arm
    uv run python -m physcal.train.stage_b eval --seed S                      # GPU: greedy on A6 / A6b / A6c
    uv run python -m physcal.train.stage_b data                               # CPU: the tokenized rows' report
    uv run python -m physcal.eval.stage_b_analysis analyze                    # CPU: D104's tests

A fresh LoRA on the pinned 4-bit base (D70 / D78) is trained on D80's supervised arm (σ* written after "±" and
trained; the `<confidence>` block as built), prompt v5, 2 epochs with configs/sft.yaml's optimizer settings. The seed
sets the LoRA init, the shuffle and the trainer's seed. D103: the 7.1 GiB allocator cap, and an OOM check (one
forward + backward on the longest batch) before step 1. The eval is run_eval on the frozen A6 / A6b / A6c files with
the adapter unmerged on the pinned base.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config


def _config() -> dict:
    return load_config("stage_b")


def seed_dir(seed: int, cfg: dict | None = None) -> Path:
    return REPO_ROOT / (cfg or _config())["output_dir"] / f"seed{seed}"


# ------------------------------------------------------------------------------------------- data
def check_data(cfg: dict) -> tuple[Path, Path, dict]:
    """The arm's file and the problems, each checked against D80's manifest (SHA-256)."""
    pc = load_config(cfg["sft_phase12_config"])
    d = REPO_ROOT / pc["output_dir"]
    manifest = json.loads((d / "manifest.json").read_text())
    arm_file, prob_file = d / f"{cfg['arm']}.jsonl", d / "problems.jsonl"
    for f in (arm_file, prob_file):
        if hashlib.sha256(f.read_bytes()).hexdigest() != manifest["sha256"][f.name]:
            raise ValueError(f"{f} does not match its manifest; rebuild with physcal.train.sft_phase12 build")
    return arm_file, prob_file, manifest


def tokenized_rows(tok, cfg: dict, max_len: int) -> tuple[list[dict], dict]:
    """D80's arm rows tokenized by sft_phase12.tokenize (prompt v5, the arm's masks); rows longer than `max_len`
    are dropped and counted."""
    from physcal.train import sft_phase12

    pc = load_config(cfg["sft_phase12_config"])
    arm_file, prob_file, _ = check_data(cfg)
    texts = {json.loads(line)["seed"]: json.loads(line)["text"] for line in prob_file.open()}
    rows, dropped = [], 0
    for line in arm_file.open():
        r = json.loads(line)
        if r["arm"] != cfg["arm"]:
            raise ValueError(f"row {r['seed']} is arm {r['arm']}, expected {cfg['arm']}")
        t = sft_phase12.tokenize(r, texts[r["seed"]], tok, pc)
        if len(t["input_ids"]) > max_len:
            dropped += 1
            continue
        rows.append({**t, "seed": r["seed"], "noise": r["noise"],
                     "n_target": sum(1 for x in t["labels"] if x != -100)})
    lens = [len(r["input_ids"]) for r in rows]
    return rows, {"n_rows": len(rows), "dropped_too_long": dropped, "max_len": max_len, "sequence_tokens_max": max(lens),
                  "sequence_tokens_mean": float(np.mean(lens)),
                  "target_tokens_mean": float(np.mean([r["n_target"] for r in rows])),
                  "n_masked_mean": float(np.mean([r["n_masked"] for r in rows])), "prompt": pc["prompt"]}


# ------------------------------------------------------------------------------------------- training
def oom_check(model, tok, rows: list[dict], batch: int) -> dict:
    """D103: one forward + backward pass on the `batch` longest rows under the allocator cap; an OOM raises here,
    before step 1. Gradients are cleared afterwards."""
    import torch

    longest = sorted(rows, key=lambda r: -len(r["input_ids"]))[:batch]
    n = max(len(r["input_ids"]) for r in longest)
    pad = tok.pad_token_id
    ids = torch.tensor([r["input_ids"] + [pad] * (n - len(r["input_ids"])) for r in longest])
    att = torch.tensor([[1] * len(r["input_ids"]) + [0] * (n - len(r["input_ids"])) for r in longest])
    lab = torch.tensor([r["labels"] + [-100] * (n - len(r["labels"])) for r in longest])
    dev = next(model.parameters()).device
    model.train()
    t0 = time.time()
    loss = model(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev)).loss
    loss.backward()
    model.zero_grad(set_to_none=True)
    out = {"batch": len(longest), "tokens": n, "loss": float(loss.detach()), "seconds": time.time() - t0}
    if dev.type == "cuda":
        out["max_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
        torch.cuda.empty_cache()
    return out


def fit(model, tok, rows: list[dict], out: Path, cfg: dict, seed: int, max_steps: int | None = None,
        deadline: float | None = None, bf16: bool = True, optim: str | None = None) -> dict:
    """SFT of `model` (a PEFT model) on pre-tokenized rows, shuffled with `seed`; saves out/final_adapter.
    Device-agnostic (CPU tests pass a tiny model, bf16=False and a torch optimizer)."""
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    from physcal.train.grpo import make_stop_callback

    t, g = cfg["train"], cfg["guards"]
    data = list(rows)
    random.Random(seed).shuffle(data)
    ds = Dataset.from_list([{k: d[k] for k in ("input_ids", "attention_mask", "labels")} for d in data])
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"), per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"], learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"], warmup_steps=t["warmup_steps"], weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"], optim=optim or t["optim"], logging_steps=t["logging_steps"],
        num_train_epochs=t["epochs"], max_steps=max_steps if max_steps else -1, bf16=bf16, save_strategy="no",
        report_to="none", seed=seed, data_seed=seed, remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.add_callback(make_stop_callback(math.inf, deadline, g["max_gpu_temp_c"], g["pause_seconds"],
                                            g["check_every_steps"], out / "events.jsonl"))
    t0 = time.time()
    result = trainer.train()
    model.save_pretrained(str(out / "final_adapter"))
    tok.save_pretrained(str(out / "final_adapter"))
    steps = trainer.state.global_step
    per_epoch = math.ceil(len(ds) / (t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]))
    return {"seed": seed, "n_rows": len(ds), "steps": steps, "steps_per_epoch": per_epoch,
            "steps_planned": max_steps or per_epoch * t["epochs"], "epochs_completed": steps / per_epoch,
            "seconds": time.time() - t0, "seconds_per_step": (time.time() - t0) / max(1, steps),
            "train_loss": getattr(result, "training_loss", None), "log_history": trainer.state.log_history}


def train(seed: int, deadline: float | None = None, max_steps: int | None = None) -> dict:
    """GPU: the pinned base (checked), a fresh LoRA, the OOM check, SFT, the R2-1 fingerprint after the save."""
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    import torch
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    from physcal.base import assert_loaded_base, base_config, resolve_model
    from physcal.eval.fingerprint import record_after_training
    from physcal.eval.run_eval import GpuMemoryPoller

    cfg = _config()
    if seed not in cfg["seeds"]:
        raise ValueError(f"seed {seed} is not one of D104's {cfg['seeds']}")
    pin_cfg = base_config(cfg["pinned_base"])
    m, lora = pin_cfg["model"], cfg["lora"]
    model_name, _, pin = resolve_model({"pinned_base": cfg["pinned_base"]})
    out = seed_dir(seed, cfg)
    if (out / "final_adapter").exists():
        raise FileExistsError(f"{out}/final_adapter exists; stage B never overwrites a trained seed")
    out.mkdir(parents=True, exist_ok=True)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, cfg["max_reserved_gib"] * 1024**3 / total))  # D103
    torch.cuda.reset_peak_memory_stats()
    model, tok = FastLanguageModel.from_pretrained(model_name=model_name, max_seq_length=m["max_seq_length"],
                                                   load_in_4bit=True, fast_inference=False, dtype=None,
                                                   text_only=m.get("text_only", False))
    assert_loaded_base(model, pin_cfg)
    model = FastLanguageModel.get_peft_model(model, r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                             target_modules=lora["target_modules"],
                                             use_gradient_checkpointing="unsloth", random_state=seed)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows, data_info = tokenized_rows(tok, cfg, m["max_seq_length"])
    oom = oom_check(model, tok, rows, cfg["train"]["per_device_train_batch_size"])
    print(f"OOM check passed: {oom}", flush=True)
    with GpuMemoryPoller(interval_s=2.0) as poller:
        summary = fit(model, tok, rows, out, cfg, seed, max_steps, deadline)
    summary.update(decision="D104 (Phase 12 stage B)", arm=cfg["arm"], pinned_base=pin, model_dir=model_name,
                   lora=lora, data=data_info, oom_check=oom, max_reserved_gib_cap=cfg["max_reserved_gib"],
                   peak_vram={"torch_max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                              "torch_max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                              "nvidia_smi_peak_used_gib_device_wide": poller.peak_mib / 1024})
    summary["fingerprint"] = {k: v for k, v in record_after_training(
        model, tok, model_name, out / "final_adapter", m.get("chat_template_kwargs") or {},
        {"run_id": f"stage-b-seed{seed}", "step": summary["steps"]}, out / "events.jsonl",
        prepare=FastLanguageModel.for_inference).items() if k != "traceback"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


# ------------------------------------------------------------------------------------------- eval
def eval_config(seed: int, cfg: dict | None = None) -> dict:
    """run_eval's config: the frozen A6 / A6b sets plus A6c, prompt v5, greedy; the pinned base with the seed's
    adapter unmerged (D67 / D68 / D78), the cap (D103)."""
    from physcal.base import base_config, pinned_base_dir

    cfg = cfg or _config()
    ec = cfg["eval"]
    pin_cfg = base_config(cfg["pinned_base"])
    ev = copy.deepcopy(load_config(ec["frozen_sets_config"]))
    ev["splits"].update(copy.deepcopy(ec["extra_splits"]))
    out = seed_dir(seed, cfg)
    ev["run_id"] = f"stage-b-seed{seed}-a6-sets"
    ev["output_dir"] = str(out / "eval")
    ev["prompt"] = ec["prompt"]
    ev["generation"]["batch_size"] = ec["batch_size"]
    ev["model"].update(id=pinned_base_dir(pin_cfg), pinned_base=cfg["pinned_base"], revision=None, fallback_id=None,
                       lora=str(out / "final_adapter"), loader="unsloth_4bit", model_class="causal_lm",
                       max_seq_length=pin_cfg["model"]["max_seq_length"],
                       text_only=pin_cfg["model"].get("text_only", False),
                       chat_template_kwargs=pin_cfg["model"].get("chat_template_kwargs") or {},
                       max_reserved_gib=ec["max_reserved_gib"])
    return ev


def evaluate(seed: int) -> dict:
    from physcal.eval.run_eval import run

    ev = eval_config(seed)
    if not Path(ev["model"]["lora"]).exists():
        raise FileNotFoundError(f"no adapter for seed {seed}: {ev['model']['lora']}")
    return run(ev)


def main() -> None:
    from physcal.train.grpo import parse_deadline

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["train", "eval", "data"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop-at", default=None, help="train: stop cleanly at this time (ISO or 'HH:MM Asia/Kolkata')")
    ap.add_argument("--max-steps", type=int, default=None, help="train: override the 2-epoch budget (smoke tests)")
    args = ap.parse_args()
    if args.action == "train":
        s = train(args.seed, parse_deadline(args.stop_at), args.max_steps)
        print(json.dumps({k: v for k, v in s.items() if k != "log_history"}, indent=1, default=str))
    elif args.action == "eval":
        res = evaluate(args.seed)
        print(json.dumps({k: v for k, v in res.items() if k in ("n", "output_dir", "seconds")}, default=str))
    else:
        from transformers import AutoTokenizer

        from physcal.base import base_config, pinned_base_dir

        cfg = _config()
        pin_cfg = base_config(cfg["pinned_base"])
        tok = AutoTokenizer.from_pretrained(pinned_base_dir(pin_cfg))
        print(json.dumps(tokenized_rows(tok, cfg, pin_cfg["model"]["max_seq_length"])[1], indent=1))


if __name__ == "__main__":
    main()

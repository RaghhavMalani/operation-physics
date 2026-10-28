"""The cross-model pilot (DECISIONS.md D137, pre-registered; R5-5): D-pos-v2 and B-v2 SFT on Llama-3.2-3B-Instruct
(4-bit, pinned in configs/base_llama32_3b_4bit.yaml), seed 0, with stage R4's arms, R4-1 data, LoRA, optimizer and step
budget unchanged; evaluated greedily on the frozen confirmatory A6c + A6b.

    uv run python -m physcal.train.stage_xm train --arm dpos2 [--stop-at HH:MM] [--max-reserved-gib G]
    PHYSCAL_CONFIRMATORY=1 uv run python -m physcal.train.stage_xm eval --arm dpos2

What differs from stage R4, and only this: the base; its chat handling (the template's date fixed, no second BOS on
eval prompts, labels closed by its own end-of-turn token); the eval sets (A6 later), its batch size and the
last-logits flag (D137 item 2). `eval_config` builds on stage R4's own confirmatory config, and a test checks that no
other key differs.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.train import stage_b, stage_r4

ARMS = tuple(load_config("stage_r4")["arms"])


def _config() -> dict:
    return load_config("stage_xm")


def r4_config(cfg: dict | None = None) -> dict:
    """Stage R4's config with this stage's base and output dir: the arms, data, LoRA and train settings unchanged."""
    cfg = cfg or _config()
    r4 = copy.deepcopy(load_config(cfg["stage_r4_config"]))
    r4.update(pinned_base=cfg["pinned_base"], output_dir=cfg["output_dir"], seed=cfg["seed"],
              max_reserved_gib=cfg["max_reserved_gib"])
    return r4


def arm_dir(arm: str, cfg: dict | None = None) -> Path:
    cfg = cfg or _config()
    return stage_r4.arm_dir(arm, r4_config(cfg), cfg["seed"])


def tokenized_rows(tok, acfg: dict, pin_model: dict, max_len: int) -> tuple[list[dict], dict]:
    """stage_b.tokenized_rows with the base's chat-template kwargs and end-of-turn token."""
    from physcal.train import sft_phase12

    pc = load_config(acfg["sft_phase12_config"])
    arm_file, prob_file, _ = stage_b.check_data(acfg)
    texts = {json.loads(line)["seed"]: json.loads(line)["text"] for line in prob_file.open()}
    rows, dropped = [], 0
    for line in arm_file.open():
        r = json.loads(line)
        if r["arm"] != acfg["arm"]:
            raise ValueError(f"row {r['seed']} is arm {r['arm']}, expected {acfg['arm']}")
        t = sft_phase12.tokenize(r, texts[r["seed"]], tok, pc, pin_model.get("chat_template_kwargs") or {},
                                 pin_model["end_of_turn"])
        if len(t["input_ids"]) > max_len:
            dropped += 1
            continue
        rows.append({**t, "seed": r["seed"], "noise": r["noise"], "n_target": sum(1 for x in t["labels"] if x != -100)})
    lens = [len(r["input_ids"]) for r in rows]
    return rows, {"n_rows": len(rows), "dropped_too_long": dropped, "max_len": max_len, "sequence_tokens_max": max(lens),
                  "sequence_tokens_mean": float(np.mean(lens)),
                  "target_tokens_mean": float(np.mean([r["n_target"] for r in rows]))}


# ------------------------------------------------------------------------------------------- training
def train(arm: str, deadline: float | None = None, max_steps: int | None = None, cap_gib: float | None = None,
          out: Path | None = None) -> dict:
    """GPU: stage_r4.train's procedure on the Llama base (pinned and checked), the arm's R4-1 rows tokenized for it.
    `out` (smoke tests only): another directory than the arm's."""
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    import torch
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    from physcal.base import assert_loaded_base, base_config, resolve_model
    from physcal.eval.fingerprint import _config as fp_config, record_after_training
    from physcal.eval.run_eval import GpuMemoryPoller

    cfg = _config()
    r4 = r4_config(cfg)
    seed = cfg["seed"]
    acfg = stage_r4.arm_config(arm, r4, seed)
    cap = cfg["max_reserved_gib"] if cap_gib is None else cap_gib
    pin_cfg = base_config(cfg["pinned_base"])
    m, lora = pin_cfg["model"], r4["lora"]
    model_name, _, pin = resolve_model({"pinned_base": cfg["pinned_base"]})
    out = Path(out) if out else arm_dir(arm, cfg)
    if (out / "final_adapter").exists():
        raise FileExistsError(f"{out}/final_adapter exists; stage XM never overwrites a trained arm")
    out.mkdir(parents=True, exist_ok=True)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 1024**3 / total))  # D103
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
    rows, data_info = tokenized_rows(tok, acfg, m, m["max_seq_length"])
    oom = stage_b.oom_check(model, tok, rows, r4["train"]["per_device_train_batch_size"])
    print(f"OOM check passed: {oom}", flush=True)
    with GpuMemoryPoller(interval_s=2.0) as poller:
        summary = stage_b.fit(model, tok, rows, out, acfg, seed, max_steps, deadline)
    summary.update(decision="D137 (R5-5: the cross-model pilot)", stage_xm_arm=arm, pinned_base=pin,
                   model_dir=model_name, lora=lora, data=data_info, oom_check=oom, max_reserved_gib_cap=cap, seed=seed,
                   peak_vram={"torch_max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                              "torch_max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                              "nvidia_smi_peak_used_gib_device_wide": poller.peak_mib / 1024})
    gc.collect()  # D137: the trainer's memory, before R2-1's sampling (it OOM'd after training at batch 16)
    torch.cuda.empty_cache()
    fpc = copy.deepcopy(fp_config())
    fpc["sampling"]["batch_size"] = m.get("fingerprint_batch_size", fpc["sampling"]["batch_size"])
    summary["fingerprint"] = {k: v for k, v in record_after_training(
        model, tok, model_name, out / "final_adapter", m.get("chat_template_kwargs") or {},
        {"run_id": f"stage-xm-{arm}-seed{seed}", "step": summary["steps"]}, out / "events.jsonl",
        prepare=FastLanguageModel.for_inference, cfg=fpc).items() if k != "traceback"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


# ------------------------------------------------------------------------------------------- eval
def eval_config(arm: str, cfg: dict | None = None, cap_gib: float | None = None) -> dict:
    """Stage R4's confirmatory eval config (Qwen, seed 0) with exactly D137's changes: the Llama base and its chat
    handling, this arm's adapter, A6c + A6b only, the batch size and the last-logits flag, the run id and output dir."""
    from physcal.base import base_config, pinned_base_dir

    cfg = cfg or _config()
    ec = cfg["eval"]
    pin_cfg = base_config(cfg["pinned_base"])
    pm = pin_cfg["model"]
    ev = stage_r4.eval_config(arm, seed=cfg["seed"], confirmatory=True)
    if ev["generation"]["max_new_tokens"] != ec["max_new_tokens"]:
        raise ValueError(f"stage R4's budget is {ev['generation']['max_new_tokens']}, D137 fixes {ec['max_new_tokens']}")
    out = arm_dir(arm, cfg)
    ev["model"].update(id=pinned_base_dir(pin_cfg), pinned_base=cfg["pinned_base"], lora=str(out / "final_adapter"),
                       max_seq_length=pm["max_seq_length"], text_only=pm.get("text_only", False),
                       chat_template_kwargs=pm.get("chat_template_kwargs") or {},
                       prompt_add_special_tokens=pm.get("prompt_add_special_tokens", True),
                       max_reserved_gib=cfg["max_reserved_gib"] if cap_gib is None else cap_gib)
    ev["splits"] = copy.deepcopy(ec["splits"])
    ev["generation"].update(batch_size=ec["batch_size"], last_logits_only=ec["last_logits_only"])
    ev["prompt"] = ec["prompt"]
    ev["output_dir"] = str(out / ec["output_subdir"])
    ev["run_id"] = f"stage-xm-{arm}-seed{cfg['seed']}-confirmatory"
    return ev


def evaluate(arm: str, cap_gib: float | None = None) -> dict:
    from physcal.eval.run_eval import run

    ev = eval_config(arm, cap_gib=cap_gib)
    if not Path(ev["model"]["lora"]).exists():
        raise FileNotFoundError(f"no adapter for arm {arm}: {ev['model']['lora']}")
    return run(ev)


def smoke_eval(adapter: str, out: str, limit: int, cap_gib: float | None = None) -> dict:
    """A smoke eval of any adapter on the DEVELOPMENT A6c / A6b files (first `limit` problems each), through the same
    run_eval path as `evaluate` (reload check included); never the frozen set."""
    from physcal.eval.run_eval import run

    ev = eval_config("dpos2", cap_gib=cap_gib)
    ev["model"]["lora"] = adapter
    ev["splits"] = {s: {**v, "file": v["file"].replace("/confirmatory/", "/")} for s, v in ev["splits"].items()}
    ev["output_dir"], ev["run_id"] = out, "stage-xm-smoke-development"
    return run(ev, limit=limit)


def main() -> None:
    from physcal.train.grpo import parse_deadline

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["train", "eval", "smoke-train", "smoke-eval"])
    ap.add_argument("--arm", choices=ARMS, default="dpos2")
    ap.add_argument("--stop-at", default=None, help="train: stop cleanly at this time (ISO or 'HH:MM Asia/Kolkata')")
    ap.add_argument("--max-steps", type=int, default=None, help="smoke-train only: a few steps")
    ap.add_argument("--max-reserved-gib", type=float, default=None, help="the launch cap, if lower than the config's")
    ap.add_argument("--out", default=None, help="smoke-*: the output dir (never the arm's)")
    ap.add_argument("--adapter", default=None, help="smoke-eval: the adapter dir")
    ap.add_argument("--limit", type=int, default=10, help="smoke-eval: problems per development set")
    args = ap.parse_args()
    cap = args.max_reserved_gib
    if cap is not None and cap > _config()["max_reserved_gib"]:
        raise ValueError(f"--max-reserved-gib {cap} is above the config's {_config()['max_reserved_gib']} (never raised)")
    if args.action == "train":
        s = train(args.arm, parse_deadline(args.stop_at) if args.stop_at else None, None, cap)
        print(json.dumps({k: s.get(k) for k in ("steps", "train_loss", "seconds", "peak_vram")}, indent=1, default=str))
    elif args.action == "smoke-train":
        if not args.out or not args.max_steps:
            raise ValueError("smoke-train needs --out and --max-steps")
        s = train(args.arm, None, args.max_steps, cap, Path(args.out))
        print(json.dumps({k: s.get(k) for k in ("steps", "train_loss", "seconds_per_step", "peak_vram", "data")},
                         indent=1, default=str))
    elif args.action == "smoke-eval":
        m = smoke_eval(args.adapter, args.out, args.limit, cap)
        print(json.dumps({k: m.get(k) for k in ("run_id", "gen_seconds", "oom_backoffs")}, indent=1, default=str))
    else:
        m = evaluate(args.arm, cap)
        print(json.dumps({k: m.get(k) for k in ("run_id", "gen_seconds", "oom_backoffs")}, indent=1, default=str))


if __name__ == "__main__":
    main()

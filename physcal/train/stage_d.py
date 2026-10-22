"""Phase 12 stage D (DECISIONS.md D109, pre-registered; R3-4 D, R3-9): explicit propagation, 1 seed per arm.

    uv run python -m physcal.train.stage_d train --arm pos|fmt [--stop-at ISO]   # GPU: SFT on D80's explicit arm
    uv run python -m physcal.train.stage_d eval --arm pos|fmt                    # GPU: greedy on A6 / A6b / A6c
    uv run python -m physcal.train.stage_d data [--arm pos|fmt]                  # CPU: the tokenized rows' report
    uv run python -m physcal.eval.stage_d_analysis analyze                       # CPU: D109's tests + line diagnostics

Both arms train a fresh LoRA on the pinned 4-bit base with D80's explicit-arm completions (the propagation line
`sigma_rel = sqrt((a1*s1)^2 + ...) = X` at the end of <steps>; the answer's σ = X·|μ|), stage B's settings otherwise
(D104: 2 epochs, configs/sft.yaml's optimizer, seed 0, D103's cap and OOM check). The arms differ only in the loss
mask: `pos` trains the line's numbers and the answer's σ (the positive control); `fmt` keeps D84's masks (format
only). The data file is never rebuilt: `pos` replaces the rows' recorded D84 masks in memory. The eval is stage B's
(`stage_b.eval_config`) with the arm's adapter.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.train import stage_b

ARMS = ("pos", "fmt")


def _config() -> dict:
    return load_config("stage_d")


def arm_config(arm: str, cfg: dict | None = None) -> dict:
    """A stage-B-shaped config for one arm (stage B's functions take it): the explicit data, the arm's output dir."""
    cfg = cfg or _config()
    if arm not in cfg["arms"]:
        raise ValueError(f"unknown arm {arm!r}; D109's arms are {list(cfg['arms'])}")
    out = {k: copy.deepcopy(cfg[k]) for k in ("sft_phase12_config", "pinned_base", "lora", "train", "guards",
                                             "max_reserved_gib", "eval")}
    out.update(arm=cfg["data_arm"], seeds=[cfg["seed"]], output_dir=f"{cfg['output_dir']}/{arm}")
    return out


def arm_dir(arm: str, cfg: dict | None = None) -> Path:
    cfg = cfg or _config()
    return stage_b.seed_dir(cfg["seed"], arm_config(arm, cfg))


def phase12_config(arm: str, cfg: dict | None = None) -> dict:
    """D80's data config with the explicit arm's σ-channel masks set to the stage D arm's."""
    cfg = cfg or _config()
    pc = copy.deepcopy(load_config(cfg["sft_phase12_config"]))
    pc["arms"][cfg["data_arm"]].update(cfg["arms"][arm])
    return pc


# ------------------------------------------------------------------------------------------- data
def arm_rows(arm: str, cfg: dict | None = None) -> tuple[list[dict], dict, dict]:
    """The explicit-arm rows (file checked against the manifest) with the arm's masks, the problem texts, and a
    report. `fmt` must reproduce the rows' recorded D84 masks exactly; `pos` replaces them."""
    from physcal.train import sft_phase12

    cfg = cfg or _config()
    pc = phase12_config(arm, cfg)
    arm_file, prob_file, manifest = stage_b.check_data(arm_config(arm, cfg))
    texts = {json.loads(line)["seed"]: json.loads(line)["text"] for line in prob_file.open()}
    rows, replaced = [], 0
    for line in arm_file.open():
        r = json.loads(line)
        if r["arm"] != cfg["data_arm"]:
            raise ValueError(f"row {r['seed']} is arm {r['arm']}, expected {cfg['data_arm']}")
        masks = sft_phase12.masked_spans(r, pc)
        if arm == "fmt" and r.get("masked") != masks:
            raise ValueError(f"seed {r['seed']}: fmt must use D84's recorded masks exactly")
        replaced += int(r.get("masked") != masks)
        rows.append({**r, "masked": masks})
    report = {"arm": arm, "file": arm_file.name, "sha256": manifest["sha256"][arm_file.name],
              "built": manifest.get("built", {}).get(arm_file.name), "masks": cfg["arms"][arm],
              "rows_with_masks_replaced": replaced,
              "masked_spans_mean": float(np.mean([len(r["masked"]) for r in rows]))}
    return rows, texts, report


def tokenized_rows(tok, arm: str, max_len: int, cfg: dict | None = None) -> tuple[list[dict], dict]:
    """sft_phase12.tokenize with the arm's masks (prompt v5); rows longer than `max_len` are dropped and counted."""
    from physcal.train import sft_phase12

    cfg = cfg or _config()
    pc = phase12_config(arm, cfg)
    rows, texts, report = arm_rows(arm, cfg)
    out, dropped = [], 0
    for r in rows:
        t = sft_phase12.tokenize(r, texts[r["seed"]], tok, pc)
        if len(t["input_ids"]) > max_len:
            dropped += 1
            continue
        out.append({**t, "seed": r["seed"], "noise": r["noise"], "n_target": sum(1 for x in t["labels"] if x != -100)})
    lens = [len(r["input_ids"]) for r in out]
    return out, {**report, "n_rows": len(out), "dropped_too_long": dropped, "max_len": max_len,
                 "sequence_tokens_max": max(lens), "sequence_tokens_mean": float(np.mean(lens)),
                 "target_tokens_mean": float(np.mean([r["n_target"] for r in out])),
                 "n_masked_mean": float(np.mean([r["n_masked"] for r in out])), "prompt": pc["prompt"]}


# ------------------------------------------------------------------------------------------- training
def train(arm: str, deadline: float | None = None, max_steps: int | None = None) -> dict:
    """GPU: stage_b.train's procedure (pinned base checked, fresh LoRA, OOM check, SFT, fingerprint) on the arm's
    rows."""
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    import torch
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    from physcal.base import assert_loaded_base, base_config, resolve_model
    from physcal.eval.fingerprint import record_after_training
    from physcal.eval.run_eval import GpuMemoryPoller

    cfg = _config()
    acfg = arm_config(arm, cfg)
    seed = cfg["seed"]
    pin_cfg = base_config(cfg["pinned_base"])
    m, lora = pin_cfg["model"], cfg["lora"]
    model_name, _, pin = resolve_model({"pinned_base": cfg["pinned_base"]})
    out = arm_dir(arm, cfg)
    if (out / "final_adapter").exists():
        raise FileExistsError(f"{out}/final_adapter exists; stage D never overwrites a trained arm")
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
    rows, data_info = tokenized_rows(tok, arm, m["max_seq_length"], cfg)
    oom = stage_b.oom_check(model, tok, rows, cfg["train"]["per_device_train_batch_size"])
    print(f"OOM check passed: {oom}", flush=True)
    with GpuMemoryPoller(interval_s=2.0) as poller:
        summary = stage_b.fit(model, tok, rows, out, acfg, seed, max_steps, deadline)
    summary.update(decision="D109 (Phase 12 stage D)", stage_d_arm=arm, pinned_base=pin, model_dir=model_name,
                   lora=lora, data=data_info, oom_check=oom, max_reserved_gib_cap=cfg["max_reserved_gib"],
                   peak_vram={"torch_max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                              "torch_max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                              "nvidia_smi_peak_used_gib_device_wide": poller.peak_mib / 1024})
    summary["fingerprint"] = {k: v for k, v in record_after_training(
        model, tok, model_name, out / "final_adapter", m.get("chat_template_kwargs") or {},
        {"run_id": f"stage-d-{arm}-seed{seed}", "step": summary["steps"]}, out / "events.jsonl",
        prepare=FastLanguageModel.for_inference).items() if k != "traceback"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


# ------------------------------------------------------------------------------------------- eval
def eval_config(arm: str, cfg: dict | None = None) -> dict:
    """Stage B's eval config (frozen A6 / A6b / A6c, prompt v5, greedy 512, pinned base + adapter unmerged, the cap)
    with the arm's adapter and output dir."""
    cfg = cfg or _config()
    ev = stage_b.eval_config(cfg["seed"], arm_config(arm, cfg))
    ev["run_id"] = f"stage-d-{arm}-seed{cfg['seed']}-a6-sets"
    return ev


def evaluate(arm: str) -> dict:
    from physcal.eval.run_eval import run

    ev = eval_config(arm)
    if not Path(ev["model"]["lora"]).exists():
        raise FileNotFoundError(f"no adapter for arm {arm}: {ev['model']['lora']}")
    return run(ev)


def main() -> None:
    from physcal.train.grpo import parse_deadline

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["train", "eval", "data"])
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--stop-at", default=None, help="train: stop cleanly at this time (ISO or 'HH:MM Asia/Kolkata')")
    ap.add_argument("--max-steps", type=int, default=None, help="train: override the 2-epoch budget (smoke tests)")
    args = ap.parse_args()
    if args.action == "train":
        s = train(args.arm, parse_deadline(args.stop_at), args.max_steps)
        print(json.dumps({k: v for k, v in s.items() if k != "log_history"}, indent=1, default=str))
    elif args.action == "eval":
        res = evaluate(args.arm)
        print(json.dumps({k: v for k, v in res.items() if k in ("n", "output_dir", "seconds")}, default=str))
    else:
        from transformers import AutoTokenizer

        from physcal.base import base_config, pinned_base_dir

        cfg = _config()
        pin_cfg = base_config(cfg["pinned_base"])
        tok = AutoTokenizer.from_pretrained(pinned_base_dir(pin_cfg))
        print(json.dumps(tokenized_rows(tok, args.arm, pin_cfg["model"]["max_seq_length"], cfg)[1], indent=1))


if __name__ == "__main__":
    main()

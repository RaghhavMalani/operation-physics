"""R4-1 (DECISIONS.md D112; pre-registered in D113): D-pos-v2 and its matched implicit control B-v2, 1 seed each.

    uv run python -m physcal.train.stage_r4 data  --arm dpos2|b2                    # CPU: the tokenized rows' report
    uv run python -m physcal.train.stage_r4 train --arm dpos2|b2 [--stop-at ISO]    # GPU: SFT on the R4-1 data
    uv run python -m physcal.train.stage_r4 eval  --arm dpos2|b2                    # GPU: greedy on A6 / A6b / A6c
    uv run python -m physcal.eval.r4_analysis analyze                               # CPU: D113's criteria and tests
    uv run python -m physcal.train.stage_r4 train --arm dpos2|b2 --seed 1|2         # D118: the confirmatory seeds
    PHYSCAL_CONFIRMATORY=1 uv run python -m physcal.train.stage_r4 eval --arm dpos2|b2 --seed S --confirmatory

D-pos's procedure exactly (stage_d.train = stage_b's: the pinned 4-bit base checked before and after loading, a
fresh LoRA, D103's cap and OOM check, 2 epochs, the R2-1 fingerprint), on the R4-1 data (configs/sft_r4.yaml):
`dpos2` trains the explicit arm (sensitivity lines, the propagation line and σ, all trained), `b2` the supervised
arm (σ* trained, no line). The eval is stage B's (`stage_b.eval_config`) with the arm's adapter; D113's token rule
sets both arms' budget from the longest D-pos-v2 training completion. D118 (the confirmatory run): `--seed` trains
or evaluates seed 1 / 2 with the same recipe and data, and `--confirmatory` evaluates on the frozen R4-5 files instead
of the development sets (into `<arm>/seed<s>/confirmatory/`; run_eval refuses them without PHYSCAL_CONFIRMATORY=1).
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

ARMS = ("dpos2", "b2")


def _config() -> dict:
    return load_config("stage_r4")


def arm_config(arm: str, cfg: dict | None = None, seed: int | None = None) -> dict:
    """A stage-B-shaped config for one arm (stage B's functions take it): the arm's data file and output dir."""
    cfg = cfg or _config()
    seed = cfg["seed"] if seed is None else seed
    if arm not in cfg["arms"]:
        raise ValueError(f"unknown arm {arm!r}; D113's arms are {list(cfg['arms'])}")
    out = {k: copy.deepcopy(cfg[k]) for k in ("sft_phase12_config", "pinned_base", "lora", "train", "guards",
                                             "max_reserved_gib", "eval")}
    out.update(arm=cfg["arms"][arm]["data_arm"], seeds=[seed], output_dir=f"{cfg['output_dir']}/{arm}")
    return out


def arm_dir(arm: str, cfg: dict | None = None, seed: int | None = None) -> Path:
    cfg = cfg or _config()
    seed = cfg["seed"] if seed is None else seed
    return stage_b.seed_dir(seed, arm_config(arm, cfg, seed))


# ------------------------------------------------------------------------------------------- data
def completion_tokens(tok, arm: str, cfg: dict | None = None) -> list[int]:
    """Tokens per training completion of the arm's data file (checked against the manifest)."""
    arm_file, _, _ = stage_b.check_data(arm_config(arm, cfg))
    return [len(tok(json.loads(line)["completion"], add_special_tokens=False)["input_ids"]) for line in arm_file.open()]


def token_budget(cfg: dict | None = None) -> dict:
    """D113's rule, the same for both arms: 768 new tokens if the longest D-pos-v2 training completion exceeds 480,
    else 512. Computed once with the pinned base's tokenizer and stored, so both arms' evals read the same value."""
    cfg = cfg or _config()
    path = REPO_ROOT / cfg["output_dir"] / "token_budget.json"
    if path.exists():
        return json.loads(path.read_text())
    from transformers import AutoTokenizer

    from physcal.base import base_config, pinned_base_dir

    ec = cfg["eval"]
    tok = AutoTokenizer.from_pretrained(str(REPO_ROOT / pinned_base_dir(base_config(cfg["pinned_base"]))))
    n = completion_tokens(tok, "dpos2", cfg)
    longest = max(n)
    out = {"decision": "D113 token rule", "longest_dpos2_completion_tokens": longest,
           "dpos2_completion_tokens_p99": float(np.percentile(n, 99)), "limit": ec["long_completion_tokens"],
           "max_new_tokens": ec["long_max_new_tokens"] if longest > ec["long_completion_tokens"] else ec["max_new_tokens"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


def data_report(arm: str, cfg: dict | None = None) -> dict:
    from transformers import AutoTokenizer

    from physcal.base import base_config, pinned_base_dir

    cfg = cfg or _config()
    pin_cfg = base_config(cfg["pinned_base"])
    tok = AutoTokenizer.from_pretrained(str(REPO_ROOT / pinned_base_dir(pin_cfg)))
    rows, info = stage_b.tokenized_rows(tok, arm_config(arm, cfg), pin_cfg["model"]["max_seq_length"])
    n = completion_tokens(tok, arm, cfg)
    return {**info, "arm": arm, "completion_tokens_max": max(n), "completion_tokens_mean": float(np.mean(n)),
            "token_budget": token_budget(cfg)}


# ------------------------------------------------------------------------------------------- training
def train(arm: str, deadline: float | None = None, max_steps: int | None = None, seed: int | None = None,
          cap_gib: float | None = None) -> dict:
    """GPU: stage_d.train's procedure on the arm's rows (stage B's tokenization, OOM check and fit). `seed` (D118):
    the LoRA init, the shuffle and the trainer's seed; `cap_gib`: the queue's launch cap if lower than the config's."""
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    import torch
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    from physcal.base import assert_loaded_base, base_config, resolve_model
    from physcal.eval.fingerprint import record_after_training
    from physcal.eval.run_eval import GpuMemoryPoller

    cfg = _config()
    seed = cfg["seed"] if seed is None else seed
    acfg = arm_config(arm, cfg, seed)
    cap = cfg["max_reserved_gib"] if cap_gib is None else cap_gib
    pin_cfg = base_config(cfg["pinned_base"])
    m, lora = pin_cfg["model"], cfg["lora"]
    model_name, _, pin = resolve_model({"pinned_base": cfg["pinned_base"]})
    out = arm_dir(arm, cfg, seed)
    if (out / "final_adapter").exists():
        raise FileExistsError(f"{out}/final_adapter exists; stage R4 never overwrites a trained arm")
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
    rows, data_info = stage_b.tokenized_rows(tok, acfg, m["max_seq_length"])
    oom = stage_b.oom_check(model, tok, rows, cfg["train"]["per_device_train_batch_size"])
    print(f"OOM check passed: {oom}", flush=True)
    with GpuMemoryPoller(interval_s=2.0) as poller:
        summary = stage_b.fit(model, tok, rows, out, acfg, seed, max_steps, deadline)
    summary.update(decision="D113 (R4-1: D-pos-v2 / B-v2)", stage_r4_arm=arm, pinned_base=pin, model_dir=model_name,
                   lora=lora, data=data_info, oom_check=oom, max_reserved_gib_cap=cap, seed=seed,
                   peak_vram={"torch_max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                              "torch_max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                              "nvidia_smi_peak_used_gib_device_wide": poller.peak_mib / 1024})
    summary["fingerprint"] = {k: v for k, v in record_after_training(
        model, tok, model_name, out / "final_adapter", m.get("chat_template_kwargs") or {},
        {"run_id": f"stage-r4-{arm}-seed{seed}", "step": summary["steps"]}, out / "events.jsonl",
        prepare=FastLanguageModel.for_inference).items() if k != "traceback"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


# ------------------------------------------------------------------------------------------- eval
def eval_config(arm: str, cfg: dict | None = None, seed: int | None = None, confirmatory: bool = False,
                cap_gib: float | None = None) -> dict:
    """Stage B's eval config with the arm's adapter, output dir and D113's token budget. `confirmatory` (D118): the
    frozen R4-5 files replace the development sets, everything else unchanged."""
    cfg = cfg or _config()
    seed = cfg["seed"] if seed is None else seed
    ev = stage_b.eval_config(seed, arm_config(arm, cfg, seed))
    ev["run_id"] = f"stage-r4-{arm}-seed{seed}-a6-sets"
    ev["generation"]["max_new_tokens"] = token_budget(cfg)["max_new_tokens"]
    if confirmatory:
        cc = cfg["confirmatory"]
        if seed not in cc["seeds"]:
            raise ValueError(f"seed {seed} is not one of D118's confirmatory seeds {cc['seeds']}")
        ev["splits"] = copy.deepcopy(cc["splits"])
        ev["output_dir"] = str(Path(ev["output_dir"]).parent / cc["output_subdir"])
        ev["run_id"] = f"stage-r4-{arm}-seed{seed}-confirmatory"
    if cap_gib is not None:
        ev["model"]["max_reserved_gib"] = cap_gib
    return ev


def evaluate(arm: str, seed: int | None = None, confirmatory: bool = False, cap_gib: float | None = None) -> dict:
    from physcal.eval.run_eval import run

    ev = eval_config(arm, seed=seed, confirmatory=confirmatory, cap_gib=cap_gib)
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
    ap.add_argument("--seed", type=int, default=None, help="D118: the seed (default: the config's, 0)")
    ap.add_argument("--confirmatory", action="store_true", help="eval: the frozen R4-5 files (D118)")
    ap.add_argument("--max-reserved-gib", type=float, default=None, help="the launch cap, if lower than the config's")
    args = ap.parse_args()
    cap = args.max_reserved_gib
    if cap is not None and cap > _config()["max_reserved_gib"]:
        raise ValueError(f"--max-reserved-gib {cap} is above the config's {_config()['max_reserved_gib']} (D118: never raised)")
    if args.action == "data":
        print(json.dumps(data_report(args.arm), indent=1))
    elif args.action == "train":
        s = train(args.arm, parse_deadline(args.stop_at) if args.stop_at else None, args.max_steps, args.seed, cap)
        print(json.dumps({k: s.get(k) for k in ("steps", "train_loss", "seconds", "peak_vram")}, indent=1, default=str))
    else:
        m = evaluate(args.arm, args.seed, args.confirmatory, cap)
        print(json.dumps({k: m.get(k) for k in ("run_id", "gen_seconds", "oom_backoffs")}, indent=1, default=str))


if __name__ == "__main__":
    main()

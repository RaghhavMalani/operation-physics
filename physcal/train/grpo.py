"""Phase 4 GRPO entrypoint: Unsloth QLoRA + TRL GRPOTrainer, reward = pure verifier output.

    uv run python -m physcal.train.grpo --config grpo_phase4 [--max-steps 10] [--run-id smoke]

Every rollout is logged to a tape; per-step metrics (reward, zero-variance fraction, mean predicted σ,
coverage, ...) go to train_metrics.jsonl. Training stops at max_steps, at the wall-clock cap, or at the hard
deadline (`stop_at`), whichever comes first; checkpoints every `save_steps`, keeping the last two.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from physcal.config import REPO_ROOT, load_config
from physcal.env.generator import Problem, env_config, generate_problem, problem_from_dict, train_cells
from physcal.eval.signals import rule_configs, rule_outcomes
from physcal.prompt import build_messages
from physcal.tape.tape import TapeWriter
from physcal.verify.reward import reward, reward_config


# ------------------------------------------------------------------------------------------- data
def training_problems(n: int, seed_offset: int, env_cfg: dict) -> list[Problem]:
    """Uniform over the training cells, round-robin (the curriculum sampler is phase 5)."""
    cells = train_cells(env_cfg)
    return [generate_problem(*cells[i % len(cells)], seed_offset + i, split="train", cfg=env_cfg) for i in range(n)]


# ------------------------------------------------------------------------------------------- reward
class VerifierReward:
    """TRL reward function: verifier score under the configured rule, plus logging of every rollout."""

    __name__ = "verifier_reward"

    def __init__(self, reward_cfg: dict, tape: TapeWriter | None, metrics_path: Path | None, group_size: int,
                 max_completion_tokens: int, model_id: str):
        self.cfg = reward_cfg
        self.per_rule = rule_configs(reward_cfg)
        self.tape = tape
        self.metrics_path = metrics_path
        self.group_size = group_size
        self.max_completion_tokens = max_completion_tokens
        self.model_id = model_id
        self.state = None  # set to trainer.state once the trainer exists
        self.history: list[dict] = []

    @property
    def step(self) -> int:
        return (self.state.global_step + 1) if self.state is not None else 0

    def __call__(self, prompts, completions, problem_json, completion_ids=None, **_) -> list[float]:
        rewards, rows = [], []
        for i, (text, pj) in enumerate(zip(completions, problem_json)):
            problem = problem_from_dict(json.loads(pj))
            r, bd = reward(problem, text, self.cfg)
            n_tokens = len(completion_ids[i]) if completion_ids is not None else None
            truncated = n_tokens is not None and n_tokens >= self.max_completion_tokens
            by_rule = rule_outcomes(problem, text, self.per_rule)
            rewards.append(r)
            rows.append({"problem": problem, "bd": bd, "by_rule": by_rule, "truncated": truncated, "n_tokens": n_tokens})
            if self.tape is not None:
                self.tape.log(self.step, problem, text, r, bd, self.model_id, None,
                              {"split": "train", "n_tokens": n_tokens, "truncated": truncated, "by_rule": by_rule})
        stats = step_stats(rows, rewards, self.group_size)
        stats["step"] = self.step
        stats["time"] = time.time()
        self.history.append(stats)
        if self.metrics_path is not None:
            with self.metrics_path.open("a") as fh:
                fh.write(json.dumps(stats, allow_nan=False) + "\n")
        return rewards


def step_stats(rows: list[dict], rewards: list[float], group_size: int) -> dict[str, Any]:
    """Per-step training metrics. Groups = consecutive blocks of `group_size` completions of one prompt."""
    n = len(rows)
    groups = [rewards[i:i + group_size] for i in range(0, n, group_size)]
    binary = [r["by_rule"]["binary"]["score_final"] for r in rows]
    bin_groups = [binary[i:i + group_size] for i in range(0, n, group_size)]
    ok = [r["bd"] for r in rows if r["bd"]["reason"] == "ok"]
    with_sigma = [b for b in ok if b.get("rel_sigma") is not None]
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    return {
        "n_completions": n,
        "reward_mean": mean(rewards),
        "reward_std": (math.sqrt(mean([(x - mean(rewards)) ** 2 for x in rewards])) if rewards else None),
        "zero_variance_fraction": mean([max(g) - min(g) == 0.0 for g in groups]),
        "zero_variance_fraction_binary": mean([max(g) - min(g) == 0.0 for g in bin_groups]),
        "accuracy_binary": mean(binary),
        "parse_fail_rate": mean([r["bd"]["reason"] == "parse_fail" for r in rows]),
        "unit_mismatch_rate": mean([r["bd"]["reason"] == "unit_mismatch" for r in rows]),
        "mean_rel_sigma": mean([b["rel_sigma"] for b in with_sigma]),
        "median_rel_sigma": (sorted(b["rel_sigma"] for b in with_sigma)[len(with_sigma) // 2] if with_sigma else None),
        "coverage_1sigma": mean([bool(b["within_1sigma"]) for b in with_sigma]),
        "coverage_2sigma": mean([bool(b["within_2sigma"]) for b in with_sigma]),
        "mean_step_score": mean([r["bd"].get("score_steps", 0.0) for r in rows]),
        "truncation_rate": mean([r["truncated"] for r in rows]),
        "mean_completion_tokens": mean([r["n_tokens"] for r in rows if r["n_tokens"] is not None]),
    }


# ------------------------------------------------------------------------------------------- guards
def gpu_temperature() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except Exception:
        return None


def parse_deadline(stop_at: str | None) -> float | None:
    """'15:45 Asia/Kolkata' (today) or an ISO timestamp -> epoch seconds."""
    if not stop_at:
        return None
    try:
        return dt.datetime.fromisoformat(stop_at).timestamp()
    except ValueError:
        pass
    from zoneinfo import ZoneInfo

    hhmm, tz = stop_at.split()
    h, m = map(int, hhmm.split(":"))
    now = dt.datetime.now(ZoneInfo(tz))
    return now.replace(hour=h, minute=m, second=0, microsecond=0).timestamp()


def make_stop_callback(wall_limit_s: float, deadline: float | None, temp_limit: int, pause_s: float,
                       check_every: int, log_path: Path | None):
    from transformers import TrainerCallback

    class StopAndThermalGuard(TrainerCallback):
        def __init__(self):
            self.t0 = time.time()

        def _log(self, msg: str) -> None:
            print(msg, flush=True)
            if log_path is not None:
                with log_path.open("a") as fh:
                    fh.write(json.dumps({"time": time.time(), "event": msg}) + "\n")

        def on_step_end(self, args, state, control, **kw):
            if state.global_step % check_every == 0:
                temp = gpu_temperature()
                if temp is not None and temp > temp_limit:
                    self._log(f"GPU at {temp}C > {temp_limit}C after step {state.global_step}: pausing {pause_s:.0f}s")
                    time.sleep(pause_s)
            now = time.time()
            if now - self.t0 > wall_limit_s:
                self._log(f"wall-clock cap reached after step {state.global_step}; stopping")
                control.should_training_stop = True
                control.should_save = True
            elif deadline is not None and now > deadline:
                self._log(f"hard deadline reached after step {state.global_step}; stopping")
                control.should_training_stop = True
                control.should_save = True
            return control

    return StopAndThermalGuard()


# ------------------------------------------------------------------------------------------- main
def train(cfg: dict, run_id: str, max_steps: int | None = None, deadline: float | None = None) -> dict:
    out = REPO_ROOT / cfg["output_dir"] / run_id
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))

    import torch
    import unsloth  # noqa: F401  (must be imported before transformers / trl for its patches)
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel

    m, g, lora = cfg["model"], cfg["grpo"], cfg["lora"]
    env_cfg = env_config()
    rw_cfg = reward_config()
    rw_cfg["scoring"]["rule"] = cfg["reward_rule"]
    steps = max_steps or g["max_steps"]

    torch.cuda.reset_peak_memory_stats()
    model, tok = FastLanguageModel.from_pretrained(
        model_name=m["id"], revision=m["revision"], max_seq_length=m["max_seq_length"],
        load_in_4bit=True, fast_inference=False, dtype=None, text_only=m.get("text_only", False),
    )
    model = FastLanguageModel.get_peft_model(
        model, r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], use_gradient_checkpointing="unsloth", random_state=cfg["seed"],
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    n_prompts = steps * g["prompts_per_step"] + g["extra_prompts"]
    problems = training_problems(n_prompts, cfg["train_seed_offset"], env_cfg)
    prompts = [tok.apply_chat_template(build_messages(p.text, cfg["prompt"]), tokenize=False,
                                       add_generation_prompt=True, **m.get("chat_template_kwargs", {}))
               for p in problems]
    ds = Dataset.from_dict({"prompt": prompts, "problem_json": [json.dumps(p.to_dict()) for p in problems]})

    meta = {"env_config": env_cfg, "reward_config": rw_cfg, "train_config": cfg, "model_id": m["id"],
            "model_revision": m["revision"], "prompt_id": cfg["prompt"]}
    tape = TapeWriter(out / "rollouts.jsonl", run_id, meta)
    reward_fn = VerifierReward(rw_cfg, tape, out / "train_metrics.jsonl", g["num_generations"],
                               g["max_completion_length"], m["id"])

    wanted = dict(
        output_dir=str(out / "checkpoints"), run_name=run_id, seed=cfg["seed"], report_to="none",
        max_steps=steps, logging_steps=1, save_steps=g["save_steps"], save_total_limit=g["save_total_limit"],
        learning_rate=g["learning_rate"], lr_scheduler_type=g["lr_scheduler_type"], warmup_ratio=g["warmup_ratio"],
        optim=g["optim"], weight_decay=g["weight_decay"], max_grad_norm=g["max_grad_norm"], bf16=True,
        per_device_train_batch_size=g["per_device_train_batch_size"],
        gradient_accumulation_steps=g["gradient_accumulation_steps"],
        num_generations=g["num_generations"], max_completion_length=g["max_completion_length"],
        max_prompt_length=g.get("max_prompt_length"), beta=g["beta"], temperature=g["temperature"],
        top_p=g["top_p"], top_k=g["top_k"], repetition_penalty=g["repetition_penalty"],
    )
    fields = set(GRPOConfig.__dataclass_fields__)
    dropped = sorted(k for k in wanted if k not in fields)
    args = GRPOConfig(**{k: v for k, v in wanted.items() if k in fields and v is not None})
    trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=[reward_fn], args=args, train_dataset=ds)
    reward_fn.state = trainer.state
    trainer.add_callback(make_stop_callback(g["wall_clock_limit_minutes"] * 60, deadline, cfg["guards"]["max_gpu_temp_c"],
                                            cfg["guards"]["pause_seconds"], cfg["guards"]["check_every_steps"],
                                            out / "events.jsonl"))

    t0 = time.time()
    result = trainer.train()
    seconds = time.time() - t0
    trainer.save_model(str(out / "final_adapter"))
    tape.close()
    gib = 1024**3
    summary = {
        "run_id": run_id,
        "model_id": m["id"],
        "model_revision": m["revision"],
        "reward_rule": cfg["reward_rule"],
        "steps_completed": trainer.state.global_step,
        "train_seconds": seconds,
        "seconds_per_step": seconds / max(1, trainer.state.global_step),
        "peak_vram": {"torch_max_allocated_gib": torch.cuda.max_memory_allocated() / gib,
                      "torch_max_reserved_gib": torch.cuda.max_memory_reserved() / gib},
        "grpo_config_dropped_keys": dropped,
        "grpo_config_resolved": {k: getattr(args, k) for k in ("loss_type", "scale_rewards", "beta",
                                                               "per_device_train_batch_size",
                                                               "gradient_accumulation_steps", "num_generations",
                                                               "max_completion_length") if hasattr(args, k)},
        "train_loss": getattr(result, "training_loss", None),
        "final_adapter": str(out / "final_adapter"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="grpo_phase4")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--stop-at", default=None, help="hard deadline, e.g. '15:00 Asia/Kolkata' or an ISO time")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_id = args.run_id or cfg["run_id"]
    summary = train(cfg, run_id, args.max_steps, parse_deadline(args.stop_at))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

"""E1, the fixed-μ σ-only pilot (pre-registered in DECISIONS.md D52).

    uv run python -m physcal.train.e1 data                                    # CPU: prefix datasets
    uv run python -m physcal.train.e1 train --config grpo_e1a [--stop-at "05:00 Asia/Kolkata"]
    uv run python -m physcal.train.e1 eval --run results/train/<run_id> [--adapter final_adapter]
    uv run python -m physcal.train.e1 compare                                 # CPU: family E tests

Prefixes are existing SFT-only greedy completions truncated right after the "±" of their answer, so μ, the
reasoning and the steps are fixed and GRPO's advantage compares σ suffixes only. E1a scores the suffix with the
unchanged verifier; E1b maps the stated relative σ to the nearest of nine bins first (the "verifier mapping"
option). Tapes always log the unchanged verifier reward, so replay stays exact; the training reward is logged
next to it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.env.generator import env_config, problem_from_dict
from physcal.prompt import build_messages
from physcal.tape.tape import TapeWriter, read_meta, read_tape
from physcal.verify.reward import reward, reward_config, verifier_version
from physcal.verify.scoring import floor, relative_terms, score, sigma_floor

PM_RE = re.compile(r"±|\+/-|\+-|\\pm")


def _config() -> dict:
    return load_config("e1")


# ------------------------------------------------------------------------------------------- data
def truncate_after_pm(completion: str) -> str | None:
    """The completion up to and including the first ± marker inside its `<answer>` block (None if absent)."""
    a = completion.find("<answer>")
    if a < 0:
        return None
    m = PM_RE.search(completion, a + len("<answer>"))
    end = completion.find("</answer>", a)
    if m is None or (end >= 0 and m.start() > end):
        return None
    return completion[:m.end()]


def build_rows(tapes: list[Path], tokenizer, prompt_id: str, chat_template_kwargs: dict) -> tuple[list[dict], dict]:
    """One row per SFT completion with a parsed answer (reason ok) and a ± marker; plus exclusion counts."""
    rows, excluded = [], {"no_marker": 0, "not_ok": 0}
    for tape in tapes:
        for r in read_tape(tape):
            prefix = truncate_after_pm(r["completion"])
            if prefix is None:
                excluded["no_marker"] += 1
                continue
            b = r["breakdown"]
            if b["reason"] != "ok":
                excluded["not_ok"] += 1
                continue
            p = r["problem"]
            prompt = tokenizer.apply_chat_template(build_messages(p["text"], prompt_id), tokenize=False,
                                                   add_generation_prompt=True, **chat_template_kwargs)
            rows.append({"key": [p["family"], p["depth"], p["seed"]], "split": r.get("split", "all"),
                         "prompt": prompt + prefix, "prefix": prefix, "problem_json": json.dumps(p),
                         "orig_mu": b["mu_si"], "orig_sigma": b["sigma_si"], "orig_completion": r["completion"]})
    return rows, excluded


def build_data() -> dict:
    from transformers import AutoTokenizer

    cfg = _config()
    tok = AutoTokenizer.from_pretrained(str(REPO_ROOT / cfg["model_id"]))
    out = REPO_ROOT / cfg["data_dir"]
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in ("train", "eval"):
        tapes = [REPO_ROOT / t for t in cfg[f"{name}_tapes"]]
        rows, excluded = build_rows(tapes, tok, cfg["prompt"], cfg.get("chat_template_kwargs", {}))
        lens = [len(tok(r["prompt"])["input_ids"]) for r in rows]
        (out / f"{name}_prefixes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        report[name] = {"tapes": cfg[f"{name}_tapes"], "n_rows": len(rows), "excluded": excluded,
                        "prompt_tokens_max": max(lens), "prompt_tokens_median": float(np.median(lens)),
                        "verifier_versions": sorted({verifier_version(read_meta(t)["reward_config"]) for t in tapes}),
                        "generator_versions": sorted({read_meta(t)["generator_version"] for t in tapes})}
    (out / "data_report.json").write_text(json.dumps(report, indent=2))
    return report


def read_rows(name: str) -> list[dict]:
    return [json.loads(l) for l in (REPO_ROOT / _config()["data_dir"] / f"{name}_prefixes.jsonl").open()]


# ------------------------------------------------------------------------------------------- scoring
def nearest_bin(rel_sigma: float, bins: list[float]) -> float:
    """The bin closest to rel_sigma in log space; σ = 0 maps to the smallest bin."""
    if rel_sigma <= 0:
        return min(bins)
    return min(bins, key=lambda b: abs(math.log(b) - math.log(rel_sigma)))


def bin_entropy_bits(rel_sigmas: list[float], bins: list[float]) -> float:
    if not rel_sigmas:
        return float("nan")
    counts = np.array([sum(nearest_bin(s, bins) == b for s in rel_sigmas) for b in bins], dtype=float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def score_suffix(problem, prefix: str, suffix: str, orig_mu: float, rw_cfg: dict, mode: str, bins: list[float]) -> dict:
    """Verifier reward of prefix + suffix, and the E1 training reward (= verifier for "free"; bin-mapped σ for
    "bins"). `mu_changed`: the suffix's unit moved μ in SI away from the SFT completion's μ."""
    sc = rw_cfg["scoring"]
    full = prefix + suffix
    r, bd = reward(problem, full, rw_cfg)
    ok = bd["reason"] == "ok"
    mu_changed = ok and abs(bd["mu_si"] / orig_mu - 1.0) > load_config("e1")["mu_changed_rtol"]
    rel = bd["sigma_si"] / abs(bd["mu_si"]) if ok and bd["mu_si"] else None
    out = {"full": full, "verifier_reward": r, "breakdown": bd, "ok": ok, "mu_changed": bool(mu_changed),
           "rel_sigma": rel, "train_reward": r, "sigma_used_rel": rel, "score_final_used": bd["score_final"]}
    if mode == "bins" and ok:
        b = nearest_bin(rel, bins)
        e, s = relative_terms(bd["mu_si"], b * abs(bd["mu_si"]), bd["truth_si"], sigma_floor(sc["rule"], sc))
        sf = score(sc["rule"], e, s, sc)
        out.update(sigma_used_rel=b, score_final_used=sf,
                   train_reward=rw_cfg["weights"]["final"] * sf + rw_cfg["weights"]["steps"] * bd["score_steps"])
    return out


class E1Reward:
    """TRL reward function for E1: logs every rollout (full text, unchanged verifier reward) and per-step stats."""

    __name__ = "e1_reward"

    def __init__(self, rw_cfg: dict, mode: str, bins: list[float], tape: TapeWriter | None, metrics_path: Path | None,
                 group_size: int, max_completion_tokens: int, model_id: str):
        self.cfg, self.mode, self.bins = rw_cfg, mode, bins
        self.tape, self.metrics_path = tape, metrics_path
        self.group_size, self.max_completion_tokens, self.model_id = group_size, max_completion_tokens, model_id
        self.trainer = None

    @property
    def step(self) -> int:
        return (self.trainer.state.global_step + 1) if self.trainer is not None else 0

    def __call__(self, prompts, completions, problem_json, prefix, orig_mu, completion_ids=None, **_) -> list[float]:
        rewards, rels, oks, changed = [], [], [], []
        for i, (suffix, pj, pre, mu0) in enumerate(zip(completions, problem_json, prefix, orig_mu)):
            problem = problem_from_dict(json.loads(pj))
            s = score_suffix(problem, pre, suffix, mu0, self.cfg, self.mode, self.bins)
            n_tokens = len(completion_ids[i]) if completion_ids is not None else None
            truncated = n_tokens is not None and n_tokens >= self.max_completion_tokens
            rewards.append(s["train_reward"])
            oks.append(s["ok"])
            changed.append(s["mu_changed"])
            if s["rel_sigma"] is not None:
                rels.append(s["rel_sigma"])
            if self.tape is not None:
                self.tape.log(self.step, problem, s["full"], s["verifier_reward"], s["breakdown"], self.model_id, None,
                              {"split": "train", "n_tokens": n_tokens, "truncated": truncated, "e1_mode": self.mode,
                               "suffix": suffix, "train_reward": s["train_reward"],
                               "sigma_used_rel": s["sigma_used_rel"], "mu_changed": s["mu_changed"]})
        groups = [rewards[i:i + self.group_size] for i in range(0, len(rewards), self.group_size)]
        stats = {"step": self.step, "time": time.time(), "n": len(rewards),
                 "mean_train_reward": float(np.mean(rewards)), "parse_ok_rate": float(np.mean(oks)),
                 "mu_changed_rate": float(np.mean(changed)),
                 "zero_variance_groups": float(np.mean([max(g) == min(g) for g in groups])),
                 "median_rel_sigma": float(np.median(rels)) if rels else None,
                 "sigma_zero_share": float(np.mean([x == 0 for x in rels])) if rels else None,
                 "bin_entropy_bits": bin_entropy_bits(rels, self.bins)}
        if self.metrics_path is not None:
            with self.metrics_path.open("a") as fh:
                fh.write(json.dumps(stats) + "\n")
        return rewards


# ------------------------------------------------------------------------------------------- train
def train(cfg: dict, run_id: str, max_steps: int | None = None, deadline: float | None = None) -> dict:
    from physcal.train.grpo import make_keep_and_memory_callback, make_stop_callback

    out = REPO_ROOT / cfg["output_dir"] / run_id
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))

    import torch
    import unsloth  # noqa: F401  (must be imported before transformers / trl for its patches)
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel

    e1 = _config()
    m, g, lora = cfg["model"], cfg["grpo"], cfg["lora"]
    rw_cfg = reward_config()
    rw_cfg["scoring"]["rule"] = cfg["reward_rule"]
    steps = max_steps or g["max_steps"]
    max_reserved_gib = cfg["guards"].get("max_reserved_gib")
    if max_reserved_gib is not None:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, max_reserved_gib * 1024**3 / total))
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

    rows = read_rows("train")
    ds = Dataset.from_dict({"prompt": [r["prompt"] for r in rows], "problem_json": [r["problem_json"] for r in rows],
                            "prefix": [r["prefix"] for r in rows], "orig_mu": [r["orig_mu"] for r in rows]})
    meta = {"env_config": env_config(), "reward_config": rw_cfg, "train_config": cfg, "model_id": m["id"],
            "model_revision": m["revision"], "prompt_id": cfg["prompt"], "e1": e1}
    tape = TapeWriter(out / "rollouts.jsonl", run_id, meta)
    reward_fn = E1Reward(rw_cfg, cfg["e1_mode"], e1["bins"], tape, out / "train_metrics.jsonl", g["num_generations"],
                         g["max_completion_length"], m["id"])
    wanted = dict(
        output_dir=str(out / "checkpoints"), run_name=run_id, seed=cfg["seed"], report_to="none",
        max_steps=steps, logging_steps=1, save_steps=g["save_steps"], save_total_limit=g["save_total_limit"],
        learning_rate=g["learning_rate"], lr_scheduler_type=g["lr_scheduler_type"], warmup_steps=g["warmup_steps"],
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
    reward_fn.trainer = trainer
    trainer.add_callback(make_stop_callback(g["wall_clock_limit_minutes"] * 60, deadline, cfg["guards"]["max_gpu_temp_c"],
                                            cfg["guards"]["pause_seconds"], cfg["guards"]["check_every_steps"],
                                            out / "events.jsonl"))
    permanent_steps = list(g.get("permanent_save_steps", []))
    trainer.add_callback(make_keep_and_memory_callback(permanent_steps, out / "permanent_checkpoints",
                                                       out / "memory.jsonl", max_reserved_gib))
    t0 = time.time()
    result = trainer.train()
    seconds = time.time() - t0
    trainer.save_model(str(out / "final_adapter"))
    tape.close()
    gib = 1024**3
    summary = {"run_id": run_id, "e1_mode": cfg["e1_mode"], "steps_completed": trainer.state.global_step,
               "train_seconds": seconds, "seconds_per_step": seconds / max(1, trainer.state.global_step),
               "peak_vram": {"torch_max_reserved_gib": torch.cuda.max_memory_reserved() / gib,
                             "torch_max_allocated_gib": torch.cuda.max_memory_allocated() / gib,
                             "num_ooms": torch.cuda.memory_stats().get("num_ooms", 0)},
               "grpo_config_dropped_keys": dropped, "train_loss": getattr(result, "training_loss", None),
               "final_adapter": str(out / "final_adapter")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


# ------------------------------------------------------------------------------------------- eval
def evaluate(run_dir: Path, adapter: str, out_name: str, backend=None, samples: int = 0) -> dict:
    """Greedy σ suffixes from the E1 model on the frozen eval prefixes; a replayable tape plus per-row scores.
    `samples` = K > 0 instead samples K suffixes per prefix with the eval config's sampling settings (D62,
    exploratory); every sample is a row, tagged with its index."""
    e1 = _config()
    mode = read_meta(run_dir / "rollouts.jsonl")["train_config"]["e1_mode"]
    rows = read_rows("eval")
    gen_cfg = {**e1["eval"]["generation"], "num_samples": 1} if samples <= 0 else \
        {**e1["eval"]["generation"], "do_sample": True, "num_samples": samples}
    if backend is None:
        from physcal.eval.run_eval import HFBackend

        backend = HFBackend(str(REPO_ROOT / e1["model_id"]),
                            {"id": str(REPO_ROOT / e1["model_id"]), "dtype": "bfloat16", "model_class": "causal_lm",
                             "lora": str(run_dir / adapter), "chat_template_kwargs": e1.get("chat_template_kwargs", {})},
                            gen_cfg)
    rw_cfg = reward_config()
    rw_cfg["scoring"]["rule"] = e1["reward_rule"]
    out = run_dir / "e1_eval" / out_name
    out.mkdir(parents=True, exist_ok=True)
    gens = []
    bs = e1["eval"]["generation"]["batch_size"]
    if samples <= 0:
        for start in range(0, len(rows), bs):
            gens += [g[0] for g in backend._generate([r["prompt"] for r in rows[start:start + bs]], 1)]
    else:
        import torch

        torch.manual_seed(gen_cfg["seed"])
        per_batch = max(1, bs // samples)
        for start in range(0, len(rows), per_batch):
            gens += backend._generate([r["prompt"] for r in rows[start:start + per_batch]], samples)
    meta = {"env_config": env_config(), "reward_config": rw_cfg, "e1": e1, "e1_mode": mode,
            "adapter": str(run_dir / adapter), "eval_tapes": e1["eval_tapes"]}
    if samples > 0:
        meta["sampled"] = {"num_samples": samples, "generation": gen_cfg}
    (out / "rollouts.jsonl").unlink(missing_ok=True)
    scored = []
    with TapeWriter(out / "rollouts.jsonl", f"{run_dir.name}-e1-eval-{out_name}", meta) as tape:
        for r, g_or_group in zip(rows, gens):
            problem = problem_from_dict(json.loads(r["problem_json"]))
            group = [g_or_group] if samples <= 0 else g_or_group
            for j, g in enumerate(group):
                s = score_suffix(problem, r["prefix"], g.text, r["orig_mu"], rw_cfg, mode, e1["bins"])
                extra = {"split": r["split"], "n_tokens": g.n_tokens, "truncated": g.truncated, "e1_mode": mode,
                         "suffix": g.text, "sigma_used_rel": s["sigma_used_rel"], "mu_changed": s["mu_changed"]}
                row = e1_row(r, s, rw_cfg["scoring"])
                if samples > 0:
                    extra["sample"] = row["sample"] = j
                tape.log(0, problem, s["full"], s["verifier_reward"], s["breakdown"], str(run_dir / adapter),
                         getattr(backend, "lora_hash", None), extra)
                scored.append(row)
    (out / "rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in scored))
    return {"n": len(scored), "mode": mode, "out": str(out)}


def e1_row(r: dict, s: dict, sc: dict) -> dict:
    """Per-problem outcome of an E1 σ: failures (parse, units, μ changed) score the floor and are not covered."""
    bd = s["breakdown"]
    fail = (not s["ok"]) or s["mu_changed"]
    sig = None if fail else s["sigma_used_rel"] * abs(bd["mu_si"])
    err = None if fail else abs(bd["mu_si"] - bd["truth_si"])
    return {"key": r["key"], "split": r["split"], "fail": fail, "reason": bd["reason"], "mu_changed": s["mu_changed"],
            "rel_sigma_stated": s["rel_sigma"], "rel_sigma_used": None if fail else s["sigma_used_rel"],
            "crps_log": s["score_final_used"] if not fail else floor(sc["rule"], sc),
            "cov1": (not fail) and err <= sig, "cov2": (not fail) and err <= 2 * sig,
            "abs_log_error": None if fail or bd["mu_si"] / bd["truth_si"] <= 0 else abs(math.log(bd["mu_si"] / bd["truth_si"]))}


# ------------------------------------------------------------------------------------------- main
def main() -> None:
    from physcal.train.grpo import parse_deadline

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["data", "train", "eval", "compare"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--stop-at", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--adapter", default="final_adapter")
    ap.add_argument("--out-name", default="final")
    ap.add_argument("--samples", type=int, default=0, help="eval: K > 0 sampled σ suffixes per prefix (D62)")
    args = ap.parse_args()
    if args.action == "data":
        print(json.dumps(build_data(), indent=2))
    elif args.action == "train":
        cfg = load_config(args.config)
        print(json.dumps(train(cfg, args.run_id or cfg["run_id"], args.max_steps, parse_deadline(args.stop_at)),
                         indent=2, default=str))
    elif args.action == "eval":
        run_dir = Path(args.run) if Path(args.run).is_absolute() else REPO_ROOT / args.run
        print(json.dumps(evaluate(run_dir, args.adapter, args.out_name, samples=args.samples), indent=2))
    else:
        from physcal.train.e1_compare import compare

        print(compare())


if __name__ == "__main__":
    main()

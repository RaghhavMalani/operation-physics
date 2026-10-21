"""Phase 12 stage C pilot (DECISIONS.md D106, pre-registered): E1d-style σ-only RL on noisy problems, rewarded by
verifier 1.3.0's score_sigma (D105).

    uv run python -m physcal.train.stage_c data                       # CPU: training and eval prefixes
    uv run python -m physcal.train.stage_c train [--stop-at ISO]      # GPU: GRPO, E1d's settings
    uv run python -m physcal.train.stage_c eval --arm final|step150|sft_control   # GPU: greedy σ suffixes
    uv run python -m physcal.eval.stage_c_analysis analyze            # CPU: D106's tests

The answer is frozen: every prompt is a D80 implicit-arm completion (generator-written; the answer is f(x_meas))
truncated right after "±", so the GRPO advantage compares σ suffixes only (E1). E1d's model, LoRA and GRPO settings
(configs/grpo_e1d.yaml), with prompt v5, D103's 7.1 GiB cap and its OOM check, and the training reward = score_sigma
(the floor for a failed or μ-moving suffix). E1's own module is reused, not edited.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config


def _config() -> dict:
    return load_config("stage_c")


def train_config(cfg: dict | None = None) -> dict:
    """E1d's config with D106's changes: run id, prompt, the cap."""
    cfg = cfg or _config()
    t = copy.deepcopy(load_config(cfg["base_config"]))
    t.update(run_id=cfg["run_id"], output_dir=cfg["output_dir"], prompt=cfg["prompt"], stage_c=cfg)
    t["guards"]["max_reserved_gib"] = cfg["max_reserved_gib"]
    t["grpo"].update(cfg.get("grpo_overrides") or {})
    return t


def run_dir(cfg: dict | None = None) -> Path:
    return REPO_ROOT / (cfg or _config())["output_dir"] / "run"


def data_dir(cfg: dict | None = None) -> Path:
    return REPO_ROOT / (cfg or _config())["output_dir"] / "data"


# ------------------------------------------------------------------------------------------- data
def prefix_row(problem, completion: str, split: str, tok, prompt_id: str, rw_cfg: dict) -> dict | None:
    """An E1 row (`e1.rows_from_records`' fields) from one generator-written completion; None if it has no "±" or
    does not score "ok"."""
    from physcal.prompt import build_messages
    from physcal.train.e1 import truncate_after_pm
    from physcal.verify.reward import reward

    prefix = truncate_after_pm(completion)
    _, bd = reward(problem, completion, rw_cfg)
    if prefix is None or bd["reason"] != "ok":
        return None
    p = problem.to_dict()
    prompt = tok.apply_chat_template(build_messages(p["text"], prompt_id), tokenize=False, add_generation_prompt=True)
    return {"key": [p["family"], p["depth"], p["seed"]], "split": split, "prompt": prompt + prefix, "prefix": prefix,
            "problem_json": json.dumps(p), "orig_mu": bd["mu_si"], "orig_sigma": bd["sigma_si"],
            "orig_completion": completion, "source_sample": None}


def build_data(cfg: dict | None = None) -> dict:
    """Training prefixes (D80's implicit arm, noisy problems) and eval prefixes (the frozen A6 / A6b / A6c problems,
    each with its implicit-arm completion built the same way)."""
    from transformers import AutoTokenizer

    from physcal.env.generator import problem_from_dict
    from physcal.eval import stage_a
    from physcal.train import sft_phase12
    from physcal.verify.reward import reward_config

    cfg = cfg or _config()
    tc = train_config(cfg)
    pc = load_config(cfg["sft_phase12_config"])
    env_cfg = load_config(pc["env_config"])
    rw = reward_config()
    tok = AutoTokenizer.from_pretrained(str(REPO_ROOT / tc["model"]["id"]))
    src = REPO_ROOT / pc["output_dir"]
    manifest = json.loads((src / "manifest.json").read_text())
    for name in (f"{cfg['data_arm']}.jsonl", "problems.jsonl"):
        if hashlib.sha256((src / name).read_bytes()).hexdigest() != manifest["sha256"][name]:
            raise ValueError(f"{src / name} does not match its manifest")
    probs = {json.loads(line)["seed"]: json.loads(line) for line in (src / "problems.jsonl").open()}
    out, report = {"train": [], "eval": []}, {"decision": "D106", "excluded": {"train": 0, "eval": 0}}
    for line in (src / f"{cfg['data_arm']}.jsonl").open():
        r = json.loads(line)
        if cfg["noisy_only"] and not r["noise"] > 0:
            continue
        row = prefix_row(problem_from_dict(probs[r["seed"]]), r["completion"], "train", tok, cfg["prompt"], rw)
        out["train"].append(row) if row else report["excluded"].__setitem__("train", report["excluded"]["train"] + 1)
    bins = pc["confidence"]["bins"]
    for d in stage_a.load_sets(load_config(cfg["eval"]["sets_config"])):  # checksums checked
        problem = problem_from_dict({k: v for k, v in d.items() if k != "_set"})
        ex = sft_phase12.build_example(problem, cfg["data_arm"], pc, env_cfg,
                                       sft_phase12.confidence_draw(problem.seed, bins))
        row = prefix_row(problem, ex["completion"], d["_set"], tok, cfg["prompt"], rw)
        out["eval"].append(row) if row else report["excluded"].__setitem__("eval", report["excluded"]["eval"] + 1)
    d = data_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in out.items():
        lens = [len(tok(r["prompt"])["input_ids"]) for r in rows]
        blob = "".join(json.dumps(r) + "\n" for r in rows)
        (d / f"{name}_prefixes.jsonl").write_text(blob)
        report[name] = {"n_rows": len(rows), "prompt_tokens_max": max(lens), "prompt_tokens_median": float(np.median(lens)),
                        "sha256": hashlib.sha256(blob.encode()).hexdigest()}
    report["train"]["noise_levels"] = sorted({json.loads(r["problem_json"])["noise"]["level"] for r in out["train"]})
    report["eval"]["sets"] = {s: sum(r["split"] == s for r in out["eval"]) for s in sorted({r["split"] for r in out["eval"]})}
    if report["train"]["prompt_tokens_max"] > tc["grpo"]["max_prompt_length"]:
        raise ValueError(f"a training prompt exceeds max_prompt_length {tc['grpo']['max_prompt_length']}")
    (d / "data_report.json").write_text(json.dumps(report, indent=1))
    return report


def read_rows(name: str, cfg: dict | None = None) -> list[dict]:
    return [json.loads(line) for line in (data_dir(cfg) / f"{name}_prefixes.jsonl").open()]


# ------------------------------------------------------------------------------------------- reward
def score_suffix_v13(problem, prefix: str, suffix: str, orig_mu: float, rw13: dict) -> dict:
    """The 1.3.0 verifier on prefix + suffix and D106's training reward: score_sigma, or the floor for a suffix
    that fails or moves μ (E1's μ-changed rule)."""
    from physcal.verify import sigma_reward
    from physcal.verify.scoring import floor

    full = prefix + suffix
    r, bd = sigma_reward.reward(problem, full, rw13)
    ok = bd["reason"] == "ok"
    rtol = load_config("e1")["mu_changed_rtol"]
    mu_changed = ok and (bd["mu_si"] != 0.0 if orig_mu == 0 else abs(bd["mu_si"] / orig_mu - 1.0) > rtol)
    bad = not ok or mu_changed or bd.get("sigma_reason") != "ok"
    return {"full": full, "verifier_reward": r, "breakdown": bd, "ok": ok, "mu_changed": bool(mu_changed),
            "rel_sigma": bd.get("sigma_rel_scored"),
            "train_reward": floor(rw13["scoring"]["rule"], rw13["scoring"]) if bad else bd["score_sigma"]}


class StageCReward:
    """TRL reward function: logs every rollout (the 1.3.0 verifier's reward and breakdown, replayable with
    sigma_reward.replay) and per-step stats; returns D106's training reward."""

    __name__ = "stage_c_reward"

    def __init__(self, rw13: dict, tape, metrics_path: Path | None, group_size: int, max_completion_tokens: int,
                 model_id: str):
        self.cfg, self.tape, self.metrics_path = rw13, tape, metrics_path
        self.group_size, self.max_completion_tokens, self.model_id = group_size, max_completion_tokens, model_id
        self.trainer = None

    @property
    def step(self) -> int:
        return (self.trainer.state.global_step + 1) if self.trainer is not None else 0

    def __call__(self, prompts, completions, problem_json, prefix, orig_mu, completion_ids=None, **_) -> list[float]:
        from physcal.env.generator import problem_from_dict

        rewards, rels, oks, changed = [], [], [], []
        for i, (suffix, pj, pre, mu0) in enumerate(zip(completions, problem_json, prefix, orig_mu)):
            problem = problem_from_dict(json.loads(pj))
            s = score_suffix_v13(problem, pre, suffix, mu0, self.cfg)
            n_tokens = len(completion_ids[i]) if completion_ids is not None else None
            rewards.append(s["train_reward"])
            oks.append(s["ok"])
            changed.append(s["mu_changed"])
            if s["rel_sigma"] is not None:
                rels.append(s["rel_sigma"])
            if self.tape is not None:
                self.tape.log(self.step, problem, s["full"], s["verifier_reward"], s["breakdown"], self.model_id, None,
                              {"split": "train", "n_tokens": n_tokens,
                               "truncated": n_tokens is not None and n_tokens >= self.max_completion_tokens,
                               "suffix": suffix, "train_reward": s["train_reward"], "mu_changed": s["mu_changed"]})
        groups = [rewards[i:i + self.group_size] for i in range(0, len(rewards), self.group_size)]
        stats = {"step": self.step, "time": time.time(), "n": len(rewards),
                 "mean_train_reward": float(np.mean(rewards)), "parse_ok_rate": float(np.mean(oks)),
                 "mu_changed_rate": float(np.mean(changed)),
                 "zero_variance_groups": float(np.mean([max(g) == min(g) for g in groups])),
                 "median_rel_sigma": float(np.median(rels)) if rels else None,
                 "sigma_zero_share": float(np.mean([x == 0 for x in rels])) if rels else None}
        if self.metrics_path is not None:
            with self.metrics_path.open("a") as fh:
                fh.write(json.dumps(stats) + "\n")
        return rewards


# ------------------------------------------------------------------------------------------- train
def oom_check(model, tok, prompts: list[str], max_new: int, batch: int) -> dict:
    """D103: one forward + backward on the `batch` longest prompts plus `max_new` tokens, under the cap."""
    import torch

    enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    longest = sorted(enc, key=len, reverse=True)[:batch]
    ids = torch.tensor([x + [tok.pad_token_id] * (len(longest[0]) - len(x) + max_new) for x in longest])
    dev = next(model.parameters()).device
    model.train()
    t0 = time.time()
    loss = model(input_ids=ids.to(dev), labels=ids.to(dev)).loss
    loss.backward()
    model.zero_grad(set_to_none=True)
    out = {"batch": len(longest), "tokens": int(ids.shape[1]), "seconds": time.time() - t0}
    if dev.type == "cuda":
        out["max_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
        torch.cuda.empty_cache()
    return out


def train(deadline: float | None = None, max_steps: int | None = None) -> dict:
    """GPU: e1.train's procedure with stage C's prefixes, reward and cap (E1d's settings otherwise)."""
    from physcal.train.e1 import grpo_args
    from physcal.train.grpo import make_keep_and_memory_callback, make_stop_callback

    cfg = _config()
    tc = train_config(cfg)
    out = run_dir(cfg)
    if (out / "final_adapter").exists():
        raise FileExistsError(f"{out}/final_adapter exists; the pilot never overwrites a run")
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    import torch
    import unsloth  # noqa: F401  (before transformers / trl)
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel

    from physcal.tape.tape import TapeWriter
    from physcal.verify.reward import reward_config

    m, g, lora = tc["model"], tc["grpo"], tc["lora"]
    rw13 = reward_config(cfg["reward_config"])
    steps = max_steps or g["max_steps"]
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, cfg["max_reserved_gib"] * 1024**3 / total))  # D103
    torch.cuda.reset_peak_memory_stats()
    model, tok = FastLanguageModel.from_pretrained(model_name=str(REPO_ROOT / m["id"]), revision=m["revision"],
                                                   max_seq_length=m["max_seq_length"], load_in_4bit=True,
                                                   fast_inference=False, dtype=None, text_only=m.get("text_only", False))
    model = FastLanguageModel.get_peft_model(model, r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                             target_modules=lora["target_modules"],
                                             use_gradient_checkpointing="unsloth", random_state=tc["seed"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = read_rows("train", cfg)  # TRL's sampler shuffles them with the run's seed, as in E1
    oom = oom_check(model, tok, [r["prompt"] for r in rows], g["max_completion_length"], g["per_device_train_batch_size"])
    print(f"OOM check passed: {oom}", flush=True)
    ds = Dataset.from_dict({"prompt": [r["prompt"] for r in rows], "problem_json": [r["problem_json"] for r in rows],
                            "prefix": [r["prefix"] for r in rows], "orig_mu": [r["orig_mu"] for r in rows]})
    meta = {"env_config": load_config(load_config(cfg["sft_phase12_config"])["env_config"]),
            "reward_config": rw13, "train_config": tc, "model_id": m["id"], "model_revision": m["revision"],
            "prompt_id": cfg["prompt"], "replay_with": "physcal.verify.sigma_reward.replay (D105)"}
    tape = TapeWriter(out / "rollouts.jsonl", cfg["run_id"], meta)
    reward_fn = StageCReward(rw13, tape, out / "train_metrics.jsonl", g["num_generations"], g["max_completion_length"],
                             m["id"])
    wanted = grpo_args(tc, out, cfg["run_id"], steps)
    fields = set(GRPOConfig.__dataclass_fields__)
    args = GRPOConfig(**{k: v for k, v in wanted.items() if k in fields and v is not None})
    trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=[reward_fn], args=args, train_dataset=ds)
    reward_fn.trainer = trainer
    gd = tc["guards"]
    trainer.add_callback(make_stop_callback(g["wall_clock_limit_minutes"] * 60, deadline, gd["max_gpu_temp_c"],
                                            gd["pause_seconds"], gd["check_every_steps"], out / "events.jsonl"))
    trainer.add_callback(make_keep_and_memory_callback(list(g.get("permanent_save_steps", [])),
                                                       out / "permanent_checkpoints", out / "memory.jsonl",
                                                       cfg["max_reserved_gib"]))
    t0 = time.time()
    result = trainer.train()
    seconds = time.time() - t0
    trainer.save_model(str(out / "final_adapter"))
    tape.close()
    gib = 1024**3
    summary = {"decision": "D106 (stage C pilot)", "run_id": cfg["run_id"], "steps_completed": trainer.state.global_step,
               "train_seconds": seconds, "seconds_per_step": seconds / max(1, trainer.state.global_step),
               "oom_check": oom, "peak_vram": {"torch_max_reserved_gib": torch.cuda.max_memory_reserved() / gib,
                                               "torch_max_allocated_gib": torch.cuda.max_memory_allocated() / gib},
               "grpo_config_dropped_keys": sorted(k for k in wanted if k not in fields),
               "train_loss": getattr(result, "training_loss", None),
               "trainer_args": {k: getattr(trainer.args, k, None) for k in ("scale_rewards", "learning_rate",
                                                                            "loss_type", "beta")}}
    from physcal.eval.fingerprint import record_after_training

    fp = record_after_training(model, tok, str(REPO_ROOT / m["id"]), out / "final_adapter",
                               m.get("chat_template_kwargs", {}), {"run_id": cfg["run_id"],
                                                                   "step": trainer.state.global_step},
                               out / "events.jsonl", prepare=FastLanguageModel.for_inference)
    summary["fingerprint"] = {k: v for k, v in fp.items() if k != "traceback"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


# ------------------------------------------------------------------------------------------- eval
def evaluate(arm: str) -> dict:
    """Greedy σ suffixes on the A6 / A6b / A6c prefixes (E1's eval machinery, 4-bit, unmerged), one arm."""
    import unsloth  # noqa: F401  (before transformers; D87)

    from physcal.train import e1

    cfg = _config()
    tc = train_config(cfg)
    adapter = cfg["eval"]["arms"][arm]
    lora = str(run_dir(cfg) / adapter) if adapter else None
    if lora and not Path(lora).exists():
        raise FileNotFoundError(lora)
    out = REPO_ROOT / cfg["output_dir"] / arm / "seed0" / "eval"   # the layout stage_b_analysis reads
    rows = read_rows("eval", cfg)
    res = e1._evaluate(out, f"stage-c-{arm}", "free", lora, tc["model"], None, 0, "unsloth_4bit", rows=rows,
                       source_tapes=[str(data_dir(cfg) / "eval_prefixes.jsonl")], batch_size=cfg["eval"]["batch_size"],
                       max_reserved_gib=cfg["eval"]["max_reserved_gib"])
    summ = run_dir(cfg) / "summary.json"
    if summ.exists() and adapter:
        (out.parent / "summary.json").write_text(summ.read_text())
    return res


def main() -> None:
    from physcal.train.grpo import parse_deadline

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["data", "train", "eval"])
    ap.add_argument("--arm", default="final", choices=["final", "step150", "sft_control"])
    ap.add_argument("--stop-at", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    if args.action == "data":
        print(json.dumps(build_data(), indent=1))
    elif args.action == "train":
        print(json.dumps(train(parse_deadline(args.stop_at), args.max_steps), indent=1, default=str)[:4000])
    else:
        print(json.dumps(evaluate(args.arm), indent=1))


if __name__ == "__main__":
    main()

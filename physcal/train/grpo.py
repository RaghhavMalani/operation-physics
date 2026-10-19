"""Phase 4 GRPO entrypoint: Unsloth QLoRA + TRL GRPOTrainer, reward = pure verifier output.

    uv run python -m physcal.train.grpo --config grpo_phase4 [--max-steps 10] [--run-id smoke]
    uv run python -m physcal.train.grpo --config grpo_split_credit --resume latest   # after a crash (D66)

Every rollout is logged to a tape; per-step metrics (reward, zero-variance fraction, mean predicted σ,
coverage, ...) go to train_metrics.jsonl. Training stops at max_steps, at the wall-clock cap, or at the hard
deadline (`stop_at`), whichever comes first; checkpoints every `save_steps`, keeping `save_total_limit`, plus
permanent adapter checkpoints at `permanent_save_steps` (never rotated). Torch memory is logged every step to
memory.jsonl; `guards.max_reserved_gib` caps the caching allocator.
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
        self.trainer = None  # set once the trainer exists; its .state is replaced when train() starts
        self.history: list[dict] = []
        self.curriculum = None  # an env.sampler.Sampler, fed with each group's sigma-free binary results
        self.curriculum_path: Path | None = None
        self.sigma_zero_monitor = SigmaZeroMonitor(**load_config("monitor")["sigma_zero_alert"])
        self.events_path: Path | None = None
        self.split_credit = None  # a train.split_credit.SplitCredit when split credit is on (phase 9.5)

    @property
    def step(self) -> int:
        return (self.trainer.state.global_step + 1) if self.trainer is not None else 0

    def __call__(self, prompts, completions, problem_json, completion_ids=None, **_) -> list[float]:
        rewards, rows = [], []
        for i, (text, pj) in enumerate(zip(completions, problem_json)):
            problem = problem_from_dict(json.loads(pj))
            r, bd = reward(problem, text, self.cfg)
            if self.split_credit is not None:
                self.split_credit.add(r, bd, text, self.cfg)
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
        groups = [rows[i:i + self.group_size] for i in range(0, len(rows), self.group_size)]
        cells = [(g[0]["problem"].family, g[0]["problem"].depth) for g in groups]
        stats["cells"] = [f"{f}/{d}" for f, d in cells]
        if self.curriculum is not None:
            self.curriculum.observe(
                [(c, [r["by_rule"]["binary"]["score_final"] == 1.0 for r in g]) for c, g in zip(cells, groups)])
            if self.curriculum_path is not None:
                with self.curriculum_path.open("a") as fh:
                    fh.write(json.dumps({"step": self.step, "cells": stats["cells"], **self.curriculum.state()}) + "\n")
        alert = self.sigma_zero_monitor.update(stats["n_with_sigma"], stats["n_sigma_zero"])
        stats.update(sigma_zero_share_rolling=alert["rolling"], sigma_zero_alert=alert["active"])
        if alert["changed"]:  # alert only: logged, never acted on
            msg = (f"ALERT sigma=0 share {alert['rolling']:.2f} > {self.sigma_zero_monitor.threshold} over the last "
                   f"{self.sigma_zero_monitor.window} steps (step {self.step})" if alert["active"] else
                   f"sigma=0 alert cleared at step {self.step} (rolling share {alert['rolling']:.2f})")
            print(msg, flush=True)
            if self.events_path is not None:
                with self.events_path.open("a") as fh:
                    fh.write(json.dumps({"time": time.time(), "step": self.step, "event": msg}) + "\n")
        stats["time"] = time.time()
        self.history.append(stats)
        if self.metrics_path is not None:
            with self.metrics_path.open("a") as fh:
                fh.write(json.dumps(stats, allow_nan=False) + "\n")
        return rewards


class SigmaZeroMonitor:
    """Rolling share of answers stating σ = 0 over the last `window_steps` steps (pooled counts); `active` once it
    exceeds `threshold`. `changed` marks the steps where the alert switches on or off."""

    def __init__(self, threshold: float, window_steps: int):
        from collections import deque

        self.threshold, self.window = threshold, window_steps
        self.counts: "deque[tuple[int, int]]" = deque(maxlen=window_steps)
        self.active = False

    def checkpoint(self) -> dict:
        return {"counts": [list(c) for c in self.counts], "active": self.active}

    def restore(self, ckpt: dict) -> None:
        self.counts.clear()
        self.counts.extend(tuple(c) for c in ckpt["counts"])
        self.active = ckpt["active"]

    def update(self, n_with_sigma: int, n_sigma_zero: int) -> dict[str, Any]:
        self.counts.append((n_with_sigma, n_sigma_zero))
        n = sum(c[0] for c in self.counts)
        rolling = sum(c[1] for c in self.counts) / n if n else 0.0
        active = rolling > self.threshold
        changed, self.active = active != self.active, active
        return {"rolling": rolling, "active": active, "changed": changed}


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
        "n_with_sigma": len(with_sigma),
        "n_sigma_zero": sum(b["sigma_si"] == 0 for b in with_sigma),
        "sigma_zero_share": mean([b["sigma_si"] == 0 for b in with_sigma]),
        "coverage_1sigma": mean([bool(b["within_1sigma"]) for b in with_sigma]),
        "coverage_2sigma": mean([bool(b["within_2sigma"]) for b in with_sigma]),
        "mean_step_score": mean([r["bd"].get("score_steps", 0.0) for r in rows]),
        "truncation_rate": mean([r["truncated"] for r in rows]),
        "mean_completion_tokens": mean([r["n_tokens"] for r in rows if r["n_tokens"] is not None]),
    }


# ------------------------------------------------------------------------------------------- curriculum
class CurriculumRepeatSampler:
    """TRL RepeatSampler's iteration pattern, but each chunk's prompts are chosen by a curriculum sampler at the
    moment the chunk is requested: for chunk: for repeat_count: for prompt in chunk: yield it mini_repeat times.

    One chunk = one optimizer step's prompts. The trainer fetches a step's micro-batches up front and accelerate
    prefetches one more, so chunk k + 1 is drawn before step k's results are observed (checked against the logged
    curriculum of all three split-credit seeds, D66). A checkpoint after step k therefore holds chunks the run has
    drawn but not trained on; `restore` replays them before drawing new ones (D66)."""

    def __init__(self, cell_indices: dict, curriculum, mini_repeat_count: int, batch_size: int,
                 repeat_count: int, total_chunks: int):
        self.cell_indices = cell_indices
        self.curriculum = curriculum
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.total_chunks = total_chunks
        self.cursor = {c: 0 for c in cell_indices}
        self.chunks: list[list[int]] = []  # every chunk drawn, in order (dataset indices)
        self.pending: list[list[int]] = []  # restored chunks drawn before the checkpoint but not yet trained on
        self.resumed_at = 0                 # optimizer steps already trained when this sampler was restored

    def checkpoint(self) -> dict:
        return {"cursor": [[f, d, n] for (f, d), n in self.cursor.items()], "chunks": [list(c) for c in self.chunks]}

    def restore(self, ckpt: dict, trained_steps: int) -> None:
        """State saved after `trained_steps` optimizer steps: chunks past that point are replayed first."""
        cursor = {(f, d): n for f, d, n in ckpt["cursor"]}
        if set(cursor) != set(self.cursor) or len(ckpt["chunks"]) < trained_steps:
            raise ValueError("repeat-sampler checkpoint does not match this run")
        self.cursor = {c: cursor[c] for c in self.cursor}
        self.chunks = [list(c) for c in ckpt["chunks"]]
        self.pending = [list(c) for c in self.chunks[trained_steps:]]
        self.resumed_at = trained_steps

    def _draw(self) -> list[int]:
        chunk = []
        for cell in self.curriculum.next_batch(self.batch_size):
            idx = self.cell_indices[cell]
            chunk.append(idx[self.cursor[cell] % len(idx)])
            self.cursor[cell] += 1
        self.chunks.append(chunk)
        return chunk

    def __iter__(self):
        for _ in range(self.total_chunks - self.resumed_at):
            chunk = self.pending.pop(0) if self.pending else self._draw()
            for _ in range(self.repeat_count):
                for i in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield i

    def __len__(self) -> int:
        return self.total_chunks * self.batch_size * self.mini_repeat_count * self.repeat_count


def curriculum_pool(per_cell: int, seed_offset: int, env_cfg: dict) -> tuple[list[Problem], dict]:
    """`per_cell` training problems for every training cell, cell-major; returns (problems, cell -> indices)."""
    problems, cell_indices = [], {}
    for k, cell in enumerate(train_cells(env_cfg)):
        start = len(problems)
        problems += [generate_problem(*cell, seed_offset + k * per_cell + i, split="train", cfg=env_cfg)
                     for i in range(per_cell)]
        cell_indices[cell] = list(range(start, len(problems)))
    return problems, cell_indices


def make_curriculum_trainer(base, cell_indices: dict, curriculum, total_chunks: int, holder: dict):
    """A subclass of the GRPO trainer class `base` that draws prompts with a CurriculumRepeatSampler. The sampler is
    kept in `holder["sampler"]`; `holder["restore"]` (a saved sampler state and its step, D66) is applied to it."""

    class CurriculumGRPOTrainer(base):
        def _get_train_sampler(self, dataset=None):
            sampler = CurriculumRepeatSampler(
                cell_indices, curriculum, mini_repeat_count=self.num_generations,
                batch_size=self.args.generation_batch_size // self.num_generations,
                repeat_count=self.num_iterations * self.args.steps_per_generation, total_chunks=total_chunks)
            if holder.get("restore") is not None:
                sampler.restore(*holder.pop("restore"))
            holder["sampler"] = sampler
            return sampler

    return CurriculumGRPOTrainer


# ------------------------------------------------------------------------------------------- resume (D66)
RESUME_STATE = "physcal_state.json"  # written into every rotating checkpoint next to the trainer's own files
CHECKPOINT_FILES = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors",
                    RESUME_STATE)
STEP_LOGS = ("rollouts.jsonl", "train_metrics.jsonl", "curriculum.jsonl", "split_credit.jsonl", "memory.jsonl")


def make_resume_state_callback(holder: dict, reward_fn, max_steps: int, meta: dict):
    """After each checkpoint save: the exact curriculum, repeat-sampler and σ = 0 monitor state, written atomically
    into the checkpoint dir, so that `train(..., resume=...)` can continue the run as if uninterrupted."""
    from transformers import TrainerCallback

    digest = meta_digest(meta)

    class ResumeState(TrainerCallback):
        def on_save(self, args, state, control, **kw):
            target = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if not target.is_dir():
                raise RuntimeError(f"checkpoint dir {target} missing at save time")
            row = {"version": 1, "global_step": state.global_step, "max_steps": max_steps, "meta_digest": digest,
                   "curriculum": reward_fn.curriculum.checkpoint() if reward_fn.curriculum is not None else None,
                   "repeat_sampler": holder["sampler"].checkpoint() if holder.get("sampler") is not None else None,
                   "sigma_zero_monitor": reward_fn.sigma_zero_monitor.checkpoint()}
            tmp = target / (RESUME_STATE + ".tmp")
            tmp.write_text(json.dumps(row))
            os.replace(tmp, target / RESUME_STATE)
            return control

    return ResumeState()


def meta_digest(meta: dict) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()[:16]


def find_resume_checkpoint(out: Path, which: str) -> Path:
    """`which` = "latest" (the newest complete checkpoint under out/checkpoints) or a checkpoint dir."""
    complete = lambda d: d.is_dir() and all((d / f).exists() for f in CHECKPOINT_FILES)
    if which != "latest":
        ckpt = Path(which) if Path(which).is_absolute() else out / which
        if not complete(ckpt):
            raise ValueError(f"{ckpt} is not a complete resumable checkpoint (needs {', '.join(CHECKPOINT_FILES)})")
        return ckpt
    dirs = sorted((d for d in (out / "checkpoints").glob("checkpoint-*") if complete(d)),
                  key=lambda d: int(d.name.split("-")[1]))
    if not dirs:
        raise ValueError(f"no complete resumable checkpoint under {out / 'checkpoints'} "
                         f"(checkpoints written before D66 have no {RESUME_STATE})")
    return dirs[-1]


def load_resume(out: Path, ckpt: Path, max_steps: int, meta: dict) -> dict:
    """The saved state of `ckpt`, after checking that it belongs to this run (same configs, same step budget)."""
    state = json.loads((ckpt / RESUME_STATE).read_text())
    step = json.loads((ckpt / "trainer_state.json").read_text())["global_step"]
    if state["global_step"] != step:
        raise ValueError(f"{ckpt}: {RESUME_STATE} is for step {state['global_step']}, the trainer state for {step}")
    if state["max_steps"] != max_steps:
        raise ValueError(f"{ckpt}: saved with max_steps {state['max_steps']}, resuming with {max_steps}")
    if state["meta_digest"] != meta_digest(meta):
        raise ValueError(f"{ckpt}: the run's configs differ from the ones it was saved with")
    stored = json.loads((out / "rollouts.meta.json").read_text())
    stored.pop("run_id", None)
    if json.loads(json.dumps(meta, sort_keys=True, default=str)) != stored:
        raise ValueError(f"{out / 'rollouts.meta.json'} differs from this run's configs")
    return state


def truncate_step_logs(out: Path, step: int, rows_per_step: dict[str, int], names=STEP_LOGS) -> dict:
    """Cut every per-step log back to the records of steps <= `step` (the checkpoint's). The cut lines are kept in
    out/resume_discarded/. A partial last line (written at the crash) is dropped; any other unreadable line, a step
    going backwards, or a missing step <= `step` raises. `rows_per_step` = expected lines per step, by file name."""
    report = {}
    for name in names:
        path = out / name
        if not path.exists():
            continue
        lines = path.read_text().splitlines(keepends=True)
        keep, cut, last = [], [], 0
        for i, line in enumerate(lines):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    cut.append(line)
                    continue
                raise ValueError(f"{path}: unreadable line {i + 1}")
            if rec["step"] < last:
                raise ValueError(f"{path}: step goes backwards at line {i + 1}")
            last = rec["step"]
            (keep if rec["step"] <= step else cut).append(line)
        counts: dict[int, int] = {}
        for line in keep:
            k = json.loads(line)["step"]
            counts[k] = counts.get(k, 0) + 1
        want = rows_per_step.get(name)
        if want is not None and counts != {k: want for k in range(1, step + 1)}:
            raise ValueError(f"{path}: steps 1..{step} are not all present with {want} line(s) each")
        if cut:
            d = out / "resume_discarded"
            d.mkdir(exist_ok=True)
            (d / f"{path.stem}.after-step-{step}.{int(time.time())}.jsonl").write_text("".join(cut))
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("".join(keep))
        os.replace(tmp, path)
        report[name] = {"kept": len(keep), "cut": len(cut)}
    return report


# ------------------------------------------------------------------------------------------- guards
def gpu_temperature() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except Exception:
        return None


def parse_deadline(stop_at: str | None) -> float | None:
    """'06:30 Asia/Kolkata' (the next occurrence of that wall-clock time) or an ISO timestamp -> epoch seconds."""
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
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= now:  # e.g. a 06:30 stop set before midnight means tomorrow
        target += dt.timedelta(days=1)
    return target.timestamp()


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


def make_keep_and_memory_callback(permanent_steps: list[int], permanent_dir: Path, memory_path: Path | None,
                                  max_reserved_gib: float | None):
    """Saves the adapter at each step in `permanent_steps` to `permanent_dir/checkpoint-<step>` (outside the
    rotating checkpoints dir, so save_total_limit never deletes it), and logs torch memory after every step."""
    from transformers import TrainerCallback

    gib = 1024**3

    class KeepAndMemoryLog(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, processing_class=None, **kw):
            step = state.global_step
            if step in permanent_steps and model is not None:
                target = permanent_dir / f"checkpoint-{step}"
                model.save_pretrained(str(target))
                if processing_class is not None:
                    processing_class.save_pretrained(str(target))
                state.save_to_json(str(target / "trainer_state.json"))
                print(f"permanent checkpoint saved: {target}", flush=True)
            if memory_path is not None:
                import torch

                ms = torch.cuda.memory_stats()
                row = {"step": step, "time": time.time(),
                       "max_reserved_gib": torch.cuda.max_memory_reserved() / gib,
                       "max_allocated_gib": torch.cuda.max_memory_allocated() / gib,
                       "reserved_gib": torch.cuda.memory_reserved() / gib,
                       "num_alloc_retries": ms.get("num_alloc_retries", 0), "num_ooms": ms.get("num_ooms", 0)}
                if max_reserved_gib is not None and row["max_reserved_gib"] > max_reserved_gib:
                    row["over_cap"] = True
                with memory_path.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
            return control

    return KeepAndMemoryLog()


# ------------------------------------------------------------------------------------------- main
def setup_trainer(cfg: dict, run_id: str, out: Path, model, tok, steps: int, deadline: float | None,
                  resume: str | None, GRPOTrainer, GRPOConfig, max_reserved_gib: float | None = None,
                  extra_args: dict | None = None, memory_log: bool = True) -> dict:
    """Everything between the loaded model and `trainer.train()`: data, tape, reward, curriculum, split credit,
    callbacks and, with `resume`, the restored state (D66). `GRPOTrainer` / `GRPOConfig` are the classes to use
    (Unsloth-patched in `train`; plain TRL in the CPU resume test), `extra_args` overrides GRPOConfig fields."""
    from datasets import Dataset

    m, g = cfg["model"], cfg["grpo"]
    env_cfg = env_config()
    rw_cfg = reward_config()
    rw_cfg["scoring"]["rule"] = cfg["reward_rule"]
    cur_cfg = cfg.get("curriculum")
    if cur_cfg:
        problems, cell_indices = curriculum_pool(cur_cfg["pool_per_cell"], cfg["train_seed_offset"], env_cfg)
    else:
        n_prompts = steps * g["prompts_per_step"] + g["extra_prompts"]
        problems = training_problems(n_prompts, cfg["train_seed_offset"], env_cfg)
    prompts = [tok.apply_chat_template(build_messages(p.text, cfg["prompt"]), tokenize=False,
                                       add_generation_prompt=True, **m.get("chat_template_kwargs", {}))
               for p in problems]
    ds = Dataset.from_dict({"prompt": prompts, "problem_json": [json.dumps(p.to_dict()) for p in problems]})

    if m.get("id"):
        model_id = m["id"]
    else:  # D78: the pinned base itself
        from physcal.base import base_config, pinned_base_dir

        model_id = pinned_base_dir(base_config(m["pinned_base"]))
    meta = {"env_config": env_cfg, "reward_config": rw_cfg, "train_config": cfg, "model_id": model_id,
            "model_revision": m.get("revision"), "prompt_id": cfg["prompt"]}
    if m.get("pinned_base"):  # only when set: the meta digest of runs started before D78 must not change (D66)
        meta["pinned_base"] = m["pinned_base"]
    resume_ckpt = resume_state = truncated = None
    if resume:  # before anything opens a log: truncation replaces the files
        if not cur_cfg:
            raise NotImplementedError("resume is implemented for curriculum runs only (D66)")
        resume_ckpt = find_resume_checkpoint(out, resume)
        resume_state = load_resume(out, resume_ckpt, steps, meta)
        rows = {"rollouts.jsonl": g["per_device_train_batch_size"] * g["gradient_accumulation_steps"],
                "train_metrics.jsonl": 1, "curriculum.jsonl": 1, "memory.jsonl": 1}
        if cfg.get("split_credit"):
            rows["split_credit.jsonl"] = 1
        truncated = truncate_step_logs(out, resume_state["global_step"], rows)
        print(f"resuming from {resume_ckpt} (step {resume_state['global_step']}); logs: {truncated}", flush=True)
    tape = TapeWriter(out / "rollouts.jsonl", run_id, meta)
    reward_fn = VerifierReward(rw_cfg, tape, out / "train_metrics.jsonl", g["num_generations"],
                               g["max_completion_length"], model_id)
    reward_fn.events_path = out / "events.jsonl"

    wanted = dict(
        output_dir=str(out / "checkpoints"), run_name=run_id, seed=cfg["seed"], report_to="none",
        max_steps=steps, logging_steps=1, save_steps=g["save_steps"], save_total_limit=g["save_total_limit"],
        learning_rate=g["learning_rate"], lr_scheduler_type=g["lr_scheduler_type"], warmup_steps=g["warmup_steps"],
        optim=g["optim"], weight_decay=g["weight_decay"], max_grad_norm=g["max_grad_norm"], bf16=True,
        ignore_data_skip=True if resume else None,  # D66: the restored sampler continues the stream; no skipping
        per_device_train_batch_size=g["per_device_train_batch_size"],
        gradient_accumulation_steps=g["gradient_accumulation_steps"],
        num_generations=g["num_generations"], max_completion_length=g["max_completion_length"],
        max_prompt_length=g.get("max_prompt_length"), beta=g["beta"], temperature=g["temperature"],
        top_p=g["top_p"], top_k=g["top_k"], repetition_penalty=g["repetition_penalty"],
    )
    fields = set(GRPOConfig.__dataclass_fields__)
    dropped = sorted(k for k in wanted if k not in fields)
    wanted.update(extra_args or {})
    args = GRPOConfig(**{k: v for k, v in wanted.items() if k in fields and v is not None})
    trainer_cls = GRPOTrainer
    holder: dict = {}
    if cur_cfg:
        from physcal.env.sampler import EMASampler

        curriculum = EMASampler(list(cell_indices), cur_cfg, cfg["seed"])
        reward_fn.curriculum, reward_fn.curriculum_path = curriculum, out / "curriculum.jsonl"
        trainer_cls = make_curriculum_trainer(GRPOTrainer, cell_indices, curriculum, steps + 16, holder)
        if resume_state is not None:
            curriculum.restore(resume_state["curriculum"])
            holder["restore"] = (resume_state["repeat_sampler"], resume_state["global_step"])
            reward_fn.sigma_zero_monitor.restore(resume_state["sigma_zero_monitor"])
            reward_fn.history = [json.loads(x) for x in (out / "train_metrics.jsonl").read_text().splitlines()]
    sc_cfg = cfg.get("split_credit")
    if sc_cfg:  # phase 9.5: σ tokens get the advantage of r_sigma, all other tokens that of r_answer
        from physcal.train.split_credit import SplitCredit, make_split_credit_trainer

        reward_fn.split_credit = SplitCredit(sc_cfg["sigma_ref_rel"], out / "split_credit.jsonl")
        trainer_cls = make_split_credit_trainer(trainer_cls, reward_fn.split_credit)
    trainer = trainer_cls(model=model, processing_class=tok, reward_funcs=[reward_fn], args=args, train_dataset=ds)
    reward_fn.trainer = trainer
    trainer.add_callback(make_stop_callback(g["wall_clock_limit_minutes"] * 60, deadline, cfg["guards"]["max_gpu_temp_c"],
                                            cfg["guards"]["pause_seconds"], cfg["guards"]["check_every_steps"],
                                            out / "events.jsonl"))
    permanent_steps = list(g.get("permanent_save_steps", []))
    trainer.add_callback(make_keep_and_memory_callback(permanent_steps, out / "permanent_checkpoints",
                                                       out / "memory.jsonl" if memory_log else None, max_reserved_gib))
    if cur_cfg:  # every rotating checkpoint carries the curriculum state (D66)
        trainer.add_callback(make_resume_state_callback(holder, reward_fn, steps, meta))
    if resume_ckpt is not None:
        with (out / "events.jsonl").open("a") as fh:
            fh.write(json.dumps({"time": time.time(), "step": resume_state["global_step"],
                                 "event": f"resumed from {resume_ckpt}", "truncated_logs": truncated}) + "\n")
    return {"trainer": trainer, "tape": tape, "args": args, "dropped": dropped, "resume_ckpt": resume_ckpt,
            "resume_state": resume_state, "permanent_steps": permanent_steps}


def train(cfg: dict, run_id: str, max_steps: int | None = None, deadline: float | None = None,
          resume: str | None = None) -> dict:
    """`resume` ("latest" or a checkpoint dir) continues an interrupted curriculum run from a checkpoint written with
    its resume state (D66): optimizer, scheduler and RNG come back through the trainer, the curriculum, its
    repeat sampler and the σ = 0 monitor from the checkpoint, and every per-step log is cut back to that step."""
    out = REPO_ROOT / cfg["output_dir"] / run_id
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))

    import torch
    import unsloth  # noqa: F401  (must be imported before transformers / trl for its patches)
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel

    m, g, lora = cfg["model"], cfg["grpo"], cfg["lora"]
    steps = max_steps or g["max_steps"]

    max_reserved_gib = cfg["guards"].get("max_reserved_gib")
    if max_reserved_gib is not None:
        # Hard cap on the caching allocator: past it, torch frees cached blocks and retries before growing (WSL has
        # no expandable_segments, so fragmentation otherwise grows reserved memory into shared system memory).
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, max_reserved_gib * 1024**3 / total))
    from physcal.base import assert_loaded_base, base_config, resolve_model

    model_name, revision, pin = resolve_model(m)  # D78: Phase 12 loads the pinned snapshot dir, never a hub name
    torch.cuda.reset_peak_memory_stats()
    model, tok = FastLanguageModel.from_pretrained(
        model_name=model_name, revision=revision, max_seq_length=m["max_seq_length"],
        load_in_4bit=True, fast_inference=False, dtype=None, text_only=m.get("text_only", False),
    )
    if pin:
        assert_loaded_base(model, base_config(pin["config"]))
    model = FastLanguageModel.get_peft_model(
        model, r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], use_gradient_checkpointing="unsloth", random_state=cfg["seed"],
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    st = setup_trainer(cfg, run_id, out, model, tok, steps, deadline, resume, GRPOTrainer, GRPOConfig, max_reserved_gib)
    trainer, tape, args, dropped = st["trainer"], st["tape"], st["args"], st["dropped"]
    resume_ckpt, resume_state, permanent_steps = st["resume_ckpt"], st["resume_state"], st["permanent_steps"]

    from physcal.eval.run_eval import GpuMemoryPoller

    t0 = time.time()
    with GpuMemoryPoller(interval_s=2.0) as poller:
        result = trainer.train(resume_from_checkpoint=str(resume_ckpt)) if resume_ckpt else trainer.train()
    seconds = time.time() - t0
    trainer.save_model(str(out / "final_adapter"))
    tape.close()
    gib = 1024**3
    summary = {
        "run_id": run_id,
        "model_id": model_name,
        "model_revision": revision,
        "pinned_base": pin,
        "reward_rule": cfg["reward_rule"],
        "split_credit": cfg.get("split_credit"),
        "steps_completed": trainer.state.global_step,
        "train_seconds": seconds,  # after a resume: this segment only
        "resumed_from": str(resume_ckpt) if resume_ckpt else None,
        "resumed_at_step": resume_state["global_step"] if resume_state else None,
        "seconds_per_step": seconds / max(1, trainer.state.global_step - (resume_state or {}).get("global_step", 0)),
        "peak_vram": {"torch_max_allocated_gib": torch.cuda.max_memory_allocated() / gib,
                      "torch_max_reserved_gib": torch.cuda.max_memory_reserved() / gib,
                      "nvidia_smi_peak_used_gib_device_wide": poller.peak_mib / 1024,
                      "max_reserved_cap_gib": max_reserved_gib,
                      "num_alloc_retries": torch.cuda.memory_stats().get("num_alloc_retries", 0),
                      "num_ooms": torch.cuda.memory_stats().get("num_ooms", 0)},
        "permanent_checkpoints": [str(out / "permanent_checkpoints" / f"checkpoint-{s}") for s in permanent_steps
                                  if (out / "permanent_checkpoints" / f"checkpoint-{s}").exists()],
        "grpo_config_dropped_keys": dropped,
        "grpo_config_resolved": {k: getattr(args, k) for k in ("loss_type", "scale_rewards", "beta",
                                                               "per_device_train_batch_size",
                                                               "gradient_accumulation_steps", "num_generations",
                                                               "max_completion_length") if hasattr(args, k)},
        "train_loss": getattr(result, "training_loss", None),
        "final_adapter": str(out / "final_adapter"),
    }
    from physcal.eval.fingerprint import record_after_training  # R2-1 / D72: after the save and the memory stats

    # the BASE's identity: with D70 / D78, model.id is the SFT adapter dir and the base is the pinned snapshot
    fingerprint = record_after_training(model, tok, pin["dir"] if pin else m["id"], out / "final_adapter",
                                        m.get("chat_template_kwargs", {}),
                                        {"run_id": run_id, "step": trainer.state.global_step}, out / "events.jsonl",
                                        prepare=FastLanguageModel.for_inference)
    summary["fingerprint"] = {k: v for k, v in fingerprint.items() if k != "traceback"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="grpo_phase4")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--stop-at", default=None, help="hard deadline, e.g. '15:00 Asia/Kolkata' or an ISO time")
    ap.add_argument("--resume", default=None,
                    help="'latest' or a checkpoint dir: continue an interrupted curriculum run (D66)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_id = args.run_id or cfg["run_id"]
    summary = train(cfg, run_id, args.max_steps, parse_deadline(args.stop_at), args.resume)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

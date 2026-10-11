"""Evaluate a model on the eval splits. Writes a rollout tape, metrics.json, summary.md and plots.

Usage:
    uv run python -m physcal.eval.run_eval [--config configs/eval.yaml] [--limit N] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from physcal.config import REPO_ROOT, load_config
from physcal.env.formulas import FAMILY_NAMES
from physcal.env.generator import Problem, env_config, generate_problem, generator_version
from physcal.eval import calibration
from physcal.prompt import SYSTEM_PROMPT, build_messages
from physcal.tape.tape import TapeWriter
from physcal.verify.reward import reward, reward_config


@dataclass
class Generation:
    text: str
    n_tokens: int
    truncated: bool  # hit max_new_tokens without an end-of-sequence token


Backend = Callable[[list[list[dict]]], list[Generation]]


# ------------------------------------------------------------------------------------------- splits
def split_families(kind: str, env_cfg: dict) -> list[str]:
    heldout = env_cfg["splits"]["heldout_families"]
    if kind == "train":
        return [f for f in FAMILY_NAMES if f not in heldout]
    if kind == "heldout":
        return [f for f in FAMILY_NAMES if f in heldout]
    return []  # e.g. "simulators" before phase 6


def build_split(index: int, split_cfg: dict, eval_cfg: dict, env_cfg: dict, limit: int | None = None) -> list[Problem]:
    """Stratified over (family, depth) cells, round-robin; seeds from this split's own block."""
    cells = [(f, d) for f in split_families(split_cfg["families"], env_cfg) for d in split_cfg.get("depths", [])]
    if not cells:
        return []
    n = split_cfg["n"] if limit is None else min(limit, split_cfg["n"])
    base = env_cfg["splits"]["eval_seed_offset"] + index * eval_cfg["seed_block"]
    return [generate_problem(*cells[i % len(cells)], base + i, split="eval", cfg=env_cfg) for i in range(n)]


# ------------------------------------------------------------------------------------------- backend
class GpuMemoryPoller:
    """Polls nvidia-smi for device-wide memory use (includes other processes, e.g. the display)."""

    def __init__(self, interval_s: float):
        self.interval_s = interval_s
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                ).stdout
                self.peak_mib = max(self.peak_mib, int(out.split()[0]))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "GpuMemoryPoller":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


class HFBackend:
    """Plain transformers inference: bf16, left padding, batched greedy decoding."""

    def __init__(self, model_id: str, model_cfg: dict, gen_cfg: dict):
        os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if model_cfg.get("lora"):
            raise NotImplementedError("LoRA evaluation arrives with phase 4 (merge adapter before eval)")
        self.torch = torch
        self.gen_cfg = gen_cfg
        torch.cuda.reset_peak_memory_stats()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, model_cfg["dtype"]), device_map="cuda"
        ).eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])

    def __call__(self, conversations: list[list[dict]]) -> list[Generation]:
        torch = self.torch
        prompts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversations]
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))  # largest batch first: OOM early
        results: list[Generation | None] = [None] * len(prompts)
        bs = self.gen_cfg["batch_size"]
        t0 = time.time()
        for start in range(0, len(order), bs):
            idx = order[start:start + bs]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=self.gen_cfg["max_new_tokens"],
                    do_sample=self.gen_cfg["do_sample"],
                    temperature=None, top_p=None, top_k=None,
                    pad_token_id=self.tok.pad_token_id,
                )
            for row, i in zip(out[:, enc["input_ids"].shape[1]:].tolist(), idx):
                end = next((k for k, t in enumerate(row) if t in self.eos_ids), None)
                n = len(row) if end is None else end + 1
                results[i] = Generation(self.tok.decode(row[:n], skip_special_tokens=True), n, end is None)
            done = min(start + bs, len(order))
            print(f"  generated {done}/{len(order)}  ({time.time() - t0:.0f}s)", flush=True)
        return results  # type: ignore[return-value]

    def memory_stats(self) -> dict[str, float]:
        torch = self.torch
        gib = 1024**3
        return {
            "torch_max_allocated_gib": torch.cuda.max_memory_allocated() / gib,
            "torch_max_reserved_gib": torch.cuda.max_memory_reserved() / gib,
            "device_total_gib": torch.cuda.get_device_properties(0).total_memory / gib,
            "device_name": torch.cuda.get_device_name(0),
        }

    def close(self) -> None:
        del self.model
        self.torch.cuda.empty_cache()


def _is_oom(exc: BaseException) -> bool:
    return "OutOfMemoryError" in type(exc).__name__ or "CUDA out of memory" in str(exc)


# ------------------------------------------------------------------------------------------- run
def run(
    eval_cfg: dict,
    backend_factory: Callable[[str], Backend] | None = None,
    limit: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    env_cfg, rw_cfg = env_config(), reward_config()
    out = Path(output_dir or REPO_ROOT / eval_cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    factory = backend_factory or (lambda mid: HFBackend(mid, eval_cfg["model"], eval_cfg["generation"]))

    splits: dict[str, list[Problem]] = {}
    skipped: dict[str, str] = {}
    for index, (name, split_cfg) in enumerate(eval_cfg["splits"].items()):
        problems = build_split(index, split_cfg, eval_cfg, env_cfg, limit)
        if problems:
            splits[name] = problems
        else:
            skipped[name] = f"no implemented families for '{split_cfg['families']}'"
    items = [(name, p) for name, ps in splits.items() for p in ps]
    conversations = [build_messages(p.text) for _, p in items]
    print(f"eval: {len(items)} problems in splits {list(splits)}; skipped {list(skipped)}", flush=True)

    model_id, oom_fallback, oom_error = eval_cfg["model"]["id"], False, None
    t0 = time.time()
    backend = None
    with GpuMemoryPoller(interval_s=eval_cfg["memory_poll_seconds"]) as poller:
        try:
            backend = factory(model_id)
            generations = backend(conversations)
        except Exception as exc:
            if not _is_oom(exc):
                raise
            oom_fallback, oom_error = True, f"{type(exc).__name__}: {str(exc)[:300]}"
            print(f"CUDA OOM with {model_id}; falling back to {eval_cfg['model']['fallback_id']}", flush=True)
            if hasattr(backend, "close"):
                backend.close()
            model_id = eval_cfg["model"]["fallback_id"]
            backend = factory(model_id)
            generations = backend(conversations)
    gen_seconds = time.time() - t0
    memory = backend.memory_stats() if hasattr(backend, "memory_stats") else {}
    if poller.peak_mib:
        memory["nvidia_smi_peak_used_gib_device_wide"] = poller.peak_mib / 1024

    meta = {
        "model_id": model_id,
        "oom_fallback": oom_fallback,
        "oom_error": oom_error,
        "generation": eval_cfg["generation"],
        "generator_version": generator_version(env_cfg),
        "env_config": env_cfg,
        "reward_config": rw_cfg,
        "eval_config": eval_cfg,
        "system_prompt": SYSTEM_PROMPT,
        "git_commit": _git_commit(),
    }
    tape_path = out / "rollouts.jsonl"
    tape_path.unlink(missing_ok=True)
    records: dict[str, list[dict]] = {name: [] for name in splits}
    with TapeWriter(tape_path, eval_cfg["run_id"], meta) as tape:
        for (name, problem), gen in zip(items, generations):
            r, bd = reward(problem, gen.text, rw_cfg)
            extra = {"split": name, "n_tokens": gen.n_tokens, "truncated": gen.truncated}
            tape.log(0, problem, gen.text, r, bd, model_id, None, extra)
            records[name].append({"problem": problem.to_dict(), "breakdown": bd, **extra})

    cal = eval_cfg["calibration"]
    metrics: dict[str, Any] = {
        "run_id": eval_cfg["run_id"],
        "model_id": model_id,
        "oom_fallback": oom_fallback,
        "oom_error": oom_error,
        "generation_seconds": gen_seconds,
        "memory": memory,
        "generator_version": meta["generator_version"],
        "reward_rule": rw_cfg["scoring"]["rule"],
        "skipped_splits": skipped,
        "splits": {},
    }
    for name, recs in records.items():
        summary = calibration.summarize([r["breakdown"] for r in recs], rw_cfg["scoring"],
                                        cal["reliability_levels"], cal["pit_bins"])
        summary["truncation_rate"] = sum(r["truncated"] for r in recs) / len(recs)
        summary["mean_completion_tokens"] = sum(r["n_tokens"] for r in recs) / len(recs)
        summary["by_depth"] = calibration.breakdown_by(recs, "depth")
        summary["by_family"] = calibration.breakdown_by(recs, "family")
        metrics["splits"][name] = summary
        calibration.plot_split(name, summary, out / f"calibration_{name}.png")
    calibration.plot_reliability_all(metrics["splits"], out / "reliability_all_splits.png")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    (out / "summary.md").write_text(render_summary(metrics))
    print(f"eval done in {gen_seconds:.0f}s; results in {out}", flush=True)
    return metrics


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def _fmt(x: Any, pct: bool = False) -> str:
    if x is None:
        return "–"
    return f"{100 * x:.1f}%" if pct else f"{x:.3g}"


def render_summary(m: dict) -> str:
    lines = [
        f"# Eval summary: {m['run_id']}",
        "",
        f"- model: `{m['model_id']}`" + (" (**OOM fallback**)" if m["oom_fallback"] else ""),
        f"- generator: `{m['generator_version']}`, reward rule: `{m['reward_rule']}`",
        f"- generation time: {m['generation_seconds']:.0f} s",
    ]
    for k, v in m["memory"].items():
        lines.append(f"- {k}: {v:.2f}" if isinstance(v, float) else f"- {k}: {v}")
    for name, why in m["skipped_splits"].items():
        lines.append(f"- split `{name}` skipped: {why}")
    lines += [
        "",
        "| split | n | accuracy (≤1%) | median rel. err | parse fail | unit mismatch | missing ± | 1σ cov | 2σ cov "
        "| median rel σ | mean reward | truncated |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in m["splits"].items():
        lines.append(
            f"| {name} | {s['n']} | {_fmt(s['accuracy'], True)} | {_fmt(s['median_rel_error'])} "
            f"| {_fmt(s['parse_fail_rate'], True)} | {_fmt(s['unit_mismatch_rate'], True)} "
            f"| {_fmt(s['missing_sigma_rate'], True)} | {_fmt(s['coverage_1sigma'], True)} "
            f"| {_fmt(s['coverage_2sigma'], True)} | {_fmt(s['median_rel_sigma'])} | {_fmt(s['mean_reward'])} "
            f"| {_fmt(s['truncation_rate'], True)} |"
        )
    for name, s in m["splits"].items():
        lines += ["", f"## {name}: by depth", "", "| depth | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |",
                  "|---|---|---|---|---|---|---|"]
        for d, b in s["by_depth"].items():
            lines.append(f"| {d} | {b['n']} | {_fmt(b['accuracy'], True)} | {_fmt(b['median_rel_error'])} "
                         f"| {_fmt(b['parse_fail_rate'], True)} | {_fmt(b['unit_mismatch_rate'], True)} "
                         f"| {_fmt(b['coverage_1sigma'], True)} |")
        lines += ["", f"## {name}: by family", "", "| family | n | accuracy | median rel. err | parse fail | unit mismatch | 1σ cov |",
                  "|---|---|---|---|---|---|---|"]
        for f, b in s["by_family"].items():
            lines.append(f"| {f} | {b['n']} | {_fmt(b['accuracy'], True)} | {_fmt(b['median_rel_error'])} "
                         f"| {_fmt(b['parse_fail_rate'], True)} | {_fmt(b['unit_mismatch_rate'], True)} "
                         f"| {_fmt(b['coverage_1sigma'], True)} |")
        lines += ["", f"Mean score by rule (failures at the rule's floor): "
                  + ", ".join(f"{r} {_fmt(v)}" for r, v in s["mean_score_by_rule"].items()),
                  f"Mean step score: {_fmt(s['mean_step_score'])}; copied-given rate: {_fmt(s['copied_given_rate'], True)}; "
                  f"steps block present: {_fmt(s['steps_block_rate'], True)}; mean completion tokens: {s['mean_completion_tokens']:.0f}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="eval")
    ap.add_argument("--limit", type=int, default=None, help="max problems per split (smoke tests)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--replot", action="store_true", help="redraw plots and summary from metrics.json, no GPU")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.replot:
        replot(Path(args.output_dir or REPO_ROOT / cfg["output_dir"]))
    else:
        run(cfg, limit=args.limit, output_dir=args.output_dir)


def replot(out: Path) -> None:
    metrics = json.loads((out / "metrics.json").read_text())
    for name, summary in metrics["splits"].items():
        calibration.plot_split(name, summary, out / f"calibration_{name}.png")
    calibration.plot_reliability_all(metrics["splits"], out / "reliability_all_splits.png")
    (out / "summary.md").write_text(render_summary(metrics))


if __name__ == "__main__":
    main()

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
from physcal.eval import calibration, posthoc
from physcal.eval.signals import rule_configs, rule_outcomes
from physcal.prompt import SYSTEM_PROMPTS, build_messages
from physcal.tape.tape import TapeWriter
from physcal.verify.reward import reward, reward_config, verifier_version


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
    """Stratified over (family, depth) cells, round-robin; seeds from this split's own block.

    A split with `train_seed_offset` is built from *training* seeds instead (e.g. a post-hoc calibration set):
    the generator's train split (which refuses held-out cells), seeds train_seed_offset + i, all below the eval
    offset."""
    cells = [(f, d) for f in split_families(split_cfg["families"], env_cfg) for d in split_cfg.get("depths", [])]
    if not cells:
        return []
    n = split_cfg["n"] if limit is None else min(limit, split_cfg["n"])
    if "train_seed_offset" in split_cfg:
        base, gen_split = split_cfg["train_seed_offset"], "train"
        if base + n > env_cfg["splits"]["eval_seed_offset"]:
            raise ValueError(f"training-seed split reaches the eval seed range ({base} + {n})")
    else:
        base, gen_split = env_cfg["splits"]["eval_seed_offset"] + index * eval_cfg["seed_block"], "eval"
    return [generate_problem(*cells[i % len(cells)], base + i, split=gen_split, cfg=env_cfg) for i in range(n)]


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
    """Plain transformers inference: bf16, left padding, batched decoding with explicit parameters.

    Model config keys: id, revision (exact commit; pinned), dtype, chat_template_kwargs (e.g.
    {enable_thinking: false}), model_class ("causal_lm" or "image_text_to_text" for VL checkpoints).
    """

    def __init__(self, model_id: str, model_cfg: dict, gen_cfg: dict):
        os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
        import torch
        import transformers

        self.torch = torch
        self.gen_cfg = gen_cfg
        self.chat_template_kwargs = model_cfg.get("chat_template_kwargs") or {}
        self.oom_backoffs = 0
        revision = model_cfg.get("revision") if model_id == model_cfg.get("id") else None
        torch.cuda.reset_peak_memory_stats()
        self.tok = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.tok.padding_side = "left"
        model_class = {
            "causal_lm": transformers.AutoModelForCausalLM,
            "image_text_to_text": transformers.AutoModelForImageTextToText,
        }[model_cfg.get("model_class", "causal_lm")]
        self.model = model_class.from_pretrained(
            model_id, revision=revision, dtype=getattr(torch, model_cfg["dtype"]), device_map="cuda"
        ).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision
        if revision and self.revision != revision:
            raise RuntimeError(f"{model_id}: loaded revision {self.revision}, expected {revision}")
        self.lora_hash = None
        if model_cfg.get("lora"):  # trained adapter: merge into the bf16 base before eval
            import hashlib

            from peft import PeftModel

            lora_dir = Path(model_cfg["lora"])
            weights = sorted(lora_dir.glob("adapter_model*.safetensors"))
            self.lora_hash = hashlib.sha256(b"".join(w.read_bytes() for w in weights)).hexdigest()[:16]
            self.model = PeftModel.from_pretrained(self.model, str(lora_dir)).merge_and_unload().eval()
        # Stop on the model's EOS ids *and* the tokenizer's end-of-turn token. Qwen3.5-2B ships no
        # generation_config.json, and its config.json names <|endoftext|> while chat turns end with <|im_end|>.
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = {e for e in (eos if isinstance(eos, list) else [eos]) if e is not None}
        if self.tok.eos_token_id is not None:
            self.eos_ids.add(self.tok.eos_token_id)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else next(iter(self.eos_ids))

    def __call__(self, conversations: list[list[dict]]) -> list[Generation]:
        """`num_samples` completions per conversation, flattened conversation-major."""
        torch = self.torch
        g = self.gen_cfg
        k = g["num_samples"]
        torch.manual_seed(g["seed"])
        prompts = [
            self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, **self.chat_template_kwargs)
            for c in conversations
        ]
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))  # largest batch first: OOM early
        results: list[list[Generation]] = [[] for _ in prompts]
        per_batch = max(1, g["batch_size"] // k)
        t0 = time.time()
        for start in range(0, len(order), per_batch):
            idx = order[start:start + per_batch]
            for i, gens in zip(idx, self._generate([prompts[i] for i in idx], k)):
                results[i] = gens
            done = min(start + per_batch, len(order))
            print(f"  generated {done * k}/{len(order) * k}  ({time.time() - t0:.0f}s)", flush=True)
        return [gen for per_prompt in results for gen in per_prompt]

    def _generate(self, prompts: list[str], k: int) -> list[list[Generation]]:
        """k completions per prompt. On CUDA OOM, split the prompts (then the samples) in half and retry.
        A retried batch draws different random numbers, so sampled runs that back off are not bit-reproducible."""
        torch, g = self.torch, self.gen_cfg
        # Every decoding parameter is explicit: the model's generation_config.json would otherwise
        # silently add its own defaults (Qwen2.5 ships repetition_penalty=1.1).
        decoding = {"do_sample": True, "temperature": g["temperature"], "top_p": g["top_p"], "top_k": g["top_k"]} \
            if g["do_sample"] else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        try:
            enc = self.tok(prompts, return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=g["max_new_tokens"],
                    repetition_penalty=g["repetition_penalty"],
                    num_return_sequences=k,
                    eos_token_id=sorted(self.eos_ids),
                    pad_token_id=self.pad_id,
                    **decoding,
                )
        except Exception as exc:
            if not _is_oom(exc) or (len(prompts) == 1 and k == 1):
                raise
            self.oom_backoffs += 1
            print(f"  CUDA OOM on {len(prompts)} prompt(s) x {k} sample(s); splitting and retrying "
                  f"(backoff #{self.oom_backoffs})", flush=True)
            torch.cuda.empty_cache()
            if len(prompts) > 1:
                half = len(prompts) // 2
                return self._generate(prompts[:half], k) + self._generate(prompts[half:], k)
            first = self._generate(prompts, k // 2)[0]
            return [first + self._generate(prompts, k - k // 2)[0]]
        rows = out[:, enc["input_ids"].shape[1]:].tolist()
        del out, enc
        results: list[list[Generation]] = [[] for _ in prompts]
        for j, row in enumerate(rows):  # prompt-major: k consecutive rows per prompt
            end = next((p for p, t in enumerate(row) if t in self.eos_ids), None)
            n = len(row) if end is None else end + 1
            results[j // k].append(Generation(self.tok.decode(row[:n], skip_special_tokens=True), n, end is None))
        return results

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
    conversations = [build_messages(p.text, eval_cfg["prompt"]) for _, p in items]
    k = eval_cfg["generation"]["num_samples"]
    if k > 1 and not eval_cfg["generation"]["do_sample"]:
        raise ValueError("num_samples > 1 needs do_sample: true (greedy samples would be identical)")
    print(f"eval: {len(items)} problems x {k} samples in splits {list(splits)}; skipped {list(skipped)}", flush=True)

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
            if not eval_cfg["model"].get("fallback_id"):
                raise
            oom_fallback, oom_error = True, f"{type(exc).__name__}: {str(exc)[:300]}"
            print(f"CUDA OOM with {model_id}; falling back to {eval_cfg['model']['fallback_id']}", flush=True)
            if hasattr(backend, "close"):
                backend.close()
            model_id = eval_cfg["model"]["fallback_id"]
            backend = factory(model_id)
            generations = backend(conversations)
    gen_seconds = time.time() - t0
    if len(generations) != len(items) * k:
        raise RuntimeError(f"backend returned {len(generations)} completions, expected {len(items) * k}")
    memory = backend.memory_stats() if hasattr(backend, "memory_stats") else {}
    model_revision = getattr(backend, "revision", None)
    oom_backoffs = getattr(backend, "oom_backoffs", 0)
    lora_hash = getattr(backend, "lora_hash", None)
    if poller.peak_mib:
        memory["nvidia_smi_peak_used_gib_device_wide"] = poller.peak_mib / 1024

    meta = {
        "model_id": model_id,
        "model_revision": model_revision,
        "oom_fallback": oom_fallback,
        "oom_error": oom_error,
        "generation": eval_cfg["generation"],
        "generator_version": generator_version(env_cfg),
        "env_config": env_cfg,
        "reward_config": rw_cfg,
        "eval_config": eval_cfg,
        "prompt_id": eval_cfg["prompt"],
        "system_prompt": SYSTEM_PROMPTS[eval_cfg["prompt"]],
        "git_commit": _git_commit(),
    }
    tape_path = out / "rollouts.jsonl"
    tape_path.unlink(missing_ok=True)
    records: dict[str, list[dict]] = {name: [] for name in splits}
    per_rule_cfgs = rule_configs(rw_cfg)
    with TapeWriter(tape_path, eval_cfg["run_id"], meta) as tape:
        for n, gen in enumerate(generations):
            name, problem = items[n // k]
            r, bd = reward(problem, gen.text, rw_cfg)
            extra = {"split": name, "sample": n % k, "n_tokens": gen.n_tokens, "truncated": gen.truncated,
                     "by_rule": rule_outcomes(problem, gen.text, per_rule_cfgs)}
            tape.log(0, problem, gen.text, r, bd, model_id, lora_hash, extra)
            records[name].append({"problem": problem.to_dict(), "completion": gen.text, "breakdown": bd, **extra})

    cal = eval_cfg["calibration"]
    metrics: dict[str, Any] = {
        "run_id": eval_cfg["run_id"],
        "model_id": model_id,
        "model_revision": model_revision,
        "lora": eval_cfg["model"].get("lora"),
        "lora_hash": lora_hash,
        "oom_backoffs": oom_backoffs,
        "oom_fallback": oom_fallback,
        "oom_error": oom_error,
        "generation_seconds": gen_seconds,
        "memory": memory,
        "generator_version": meta["generator_version"],
        "reward_rule": rw_cfg["scoring"]["rule"],
        "prompt_id": eval_cfg["prompt"],
        "skipped_splits": skipped,
        "splits": {},
    }
    write_metrics(metrics, records, rw_cfg, cal, k, out)
    print(f"eval done in {gen_seconds:.0f}s; results in {out}", flush=True)
    return metrics


def write_metrics(metrics: dict, records: dict[str, list[dict]], rw_cfg: dict, cal: dict, k: int, out: Path) -> None:
    """Per-split summaries into `metrics["splits"]`, plus plots, metrics.json and summary.md in `out`."""
    for name, recs in records.items():
        summary = calibration.summarize([r["breakdown"] for r in recs], rw_cfg["scoring"],
                                        cal["reliability_levels"], cal["pit_bins"])
        summary["truncation_rate"] = sum(r["truncated"] for r in recs) / len(recs)
        summary["mean_completion_tokens"] = sum(r["n_tokens"] for r in recs) / len(recs)
        summary["by_depth"] = calibration.breakdown_by(recs, "depth")
        summary["by_family"] = calibration.breakdown_by(recs, "family")
        summary["accuracy_at_tolerance"] = calibration.accuracy_at_tolerances(recs, cal["accuracy_tolerances"])
        if k > 1:
            summary["groups"] = calibration.group_stats(recs)
        metrics["splits"][name] = summary
        calibration.plot_split(name, summary, out / f"calibration_{name}.png")
    calibration.plot_reliability_all(metrics["splits"], out / "reliability_all_splits.png")
    metrics["posthoc_baseline"] = posthoc.baseline_report(records, rw_cfg["scoring"], metrics["generator_version"],
                                                          verifier_version(rw_cfg))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    (out / "summary.md").write_text(render_summary(metrics))


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
        f"- model: `{m['model_id']}` @ `{m.get('model_revision')}`" + (" (**OOM fallback**)" if m["oom_fallback"] else ""),
        f"- generator: `{m['generator_version']}`, reward rule: `{m['reward_rule']}`, prompt: `{m.get('prompt_id', 'v1')}`",
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
    pb = m.get("posthoc_baseline") or {}
    if pb.get("available"):
        lines += ["", f"## Standing calibration baseline: σ × {pb['scale']:.3g} ({pb['scale_name']}, `{pb['fit_file']}`)",
                  "", "SFT-only rows are on the same frozen split" + ("" if pb["baseline_comparable"] else
                  " (**not comparable here**: different generator or verifier version, so omitted)") + ".", "",
                  "| split | rows | σ = 0 | 1σ cov | 2σ cov | mean crps_log |", "|---|---|---|---|---|---|"]
        for name, e in pb["splits"].items():
            sz = m["splits"][name].get("sigma_zero_rate")
            rows = [("this model", e["this_model_unscaled"]), (f"this model, σ × {pb['scale']:.3g}", e["this_model_scaled"])]
            rows += [(label, e[k]) for k, label in (("sft_only_unscaled", "SFT-only"),
                                                    ("sft_only_scaled", f"SFT-only, σ × {pb['scale']:.3g} (baseline)"))
                     if k in e]
            for label, r in rows:
                lines.append(f"| {name} | {label} | {_fmt(sz, True) if label == 'this model' else ''} "
                             f"| {_fmt(r['coverage_1sigma'], True)} | {_fmt(r['coverage_2sigma'], True)} "
                             f"| {_fmt(r['mean_crps_log'])} |")
    cond = pb.get("conditional") or {}
    if cond.get("splits"):
        lines += ["", "## Conditional post-hoc baselines (D49)", "",
                  "(a) text-feature scale on σ_eff = max(σ, crps_log floor·|μ|), fitted on the calibration split: the "
                  "control RL calibration claims are judged against. (a') the same text model with σ' = c(x)·|μ| "
                  "(ignores the stated σ): sensitivity. (b) oracle per-(family, depth) scale on σ_eff, fitted in-sample "
                  "on the evaluated completions: an upper bound only."
                  + ("" if cond["baseline_comparable"] else " SFT-only rows omitted (not comparable here)."), "",
                  "| split | rows | 1σ cov | 2σ cov | mean crps_log |", "|---|---|---|---|---|"]
        labels = (("this_model_text_scale", "this model × (a) text scale"),
                  ("this_model_text_direct", "this model × (a') text direct"),
                  ("this_model_oracle_cell", "this model × (b) oracle per cell (upper bound)"),
                  ("sft_only_text_scale", "SFT-only × (a) text scale (**control**)"),
                  ("sft_only_text_direct", "SFT-only × (a') text direct"),
                  ("sft_only_oracle_cell", "SFT-only × (b) oracle per cell (upper bound)"))
        for name, e in cond["splits"].items():
            for k, label in labels:
                if k in e:
                    lines.append(f"| {name} | {label} | {_fmt(e[k]['coverage_1sigma'], True)} "
                                 f"| {_fmt(e[k]['coverage_2sigma'], True)} | {_fmt(e[k]['mean_crps_log'])} |")
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

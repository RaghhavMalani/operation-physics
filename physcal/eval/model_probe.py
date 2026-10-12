"""Phase 3.5 model probe: greedy eval + k=8 group probe per candidate, GRPO footprint estimate, selection.

    uv run python -m physcal.eval.model_probe                     # every candidate, then the summary
    uv run python -m physcal.eval.model_probe --only qwen3.5-2b   # one candidate
    uv run python -m physcal.eval.model_probe --summarize         # just (re)build the summary

Results: results/model_probe/<label>/{greedy,group}/ (normal run_eval output, with tapes) and
results/model_probe/summary.{json,md}. Inference only: nothing is trained.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

from physcal.config import REPO_ROOT, load_config
from physcal.env.generator import env_config
from physcal.eval import footprint
from physcal.eval.run_eval import build_split, run
from physcal.prompt import build_messages


def candidate_eval_config(base: dict, cand: dict, part: str, out_root: Path) -> dict:
    cfg = copy.deepcopy(base)
    cfg["run_id"] = f"model-probe-{part}-{cand['label']}"
    cfg["output_dir"] = str(out_root / cand["label"] / part)
    cfg["model"].update(id=cand["id"], revision=cand["revision"], fallback_id=None,
                        model_class=cand.get("model_class", "causal_lm"),
                        chat_template_kwargs=cand.get("chat_template_kwargs", {}))
    cfg["generation"]["batch_size"] = cand.get("batch_size", cfg["generation"]["batch_size"])
    return cfg


def prompt_token_lengths(cand: dict, eval_cfg: dict) -> list[int]:
    """Prompt lengths (system + user, chat template applied) in this model's tokens, over the probe problems."""
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cand["id"], revision=cand["revision"])
    env_cfg = env_config()
    lengths = []
    for index, (_, split_cfg) in enumerate(eval_cfg["splits"].items()):
        for p in build_split(index, split_cfg, eval_cfg, env_cfg):
            text = tok.apply_chat_template(build_messages(p.text, eval_cfg["prompt"]), tokenize=False,
                                           add_generation_prompt=True, **cand.get("chat_template_kwargs", {}))
            lengths.append(len(tok(text)["input_ids"]))
    return lengths


def _pct(xs: list[int], q: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summarize(probe_cfg: dict) -> dict[str, Any]:
    out_root = REPO_ROOT / probe_cfg["output_dir"]
    fp_cfg = load_config(probe_cfg["footprint_config"])
    greedy_base = load_config(probe_cfg["greedy_config"])
    rows = []
    for cand in probe_cfg["candidates"]:
        d = out_root / cand["label"]
        if not (d / "greedy" / "metrics.json").exists() or not (d / "group" / "metrics.json").exists():
            rows.append({"label": cand["label"], "id": cand["id"], "revision": cand["revision"], "missing": True})
            continue
        greedy = json.loads((d / "greedy" / "metrics.json").read_text())
        group = json.loads((d / "group" / "metrics.json").read_text())
        lengths = prompt_token_lengths(cand, greedy_base)
        shape = footprint.model_shape(cand["id"], cand["revision"])
        prompt_tokens = max(lengths)
        g = group["splits"]["id"]
        rows.append({
            "label": cand["label"],
            "id": cand["id"],
            "revision": greedy["model_revision"],
            "params_text": shape.params_text,
            "greedy": {
                split: {
                    "accuracy_sigma_free": s["accuracy_at_tolerance"]["sigma_free"],
                    "accuracy_strict": s["accuracy_at_tolerance"]["strict"],
                    "truncation_rate": s["truncation_rate"],
                    "parse_fail_rate": s["parse_fail_rate"],
                    "missing_sigma_rate": s["missing_sigma_rate"],
                    "unit_mismatch_rate": s["unit_mismatch_rate"],
                    "coverage_1sigma": s["coverage_1sigma"],
                    "coverage_2sigma": s["coverage_2sigma"],
                    "median_rel_sigma": s["median_rel_sigma"],
                    "mean_completion_tokens": s["mean_completion_tokens"],
                }
                for split, s in greedy["splits"].items()
            },
            "group": {
                "k": g["groups"]["k"],
                "n_groups": g["groups"]["n_groups"],
                "pass_at_k": g["groups"]["pass_at_k"],
                "mean_pass_rate": g["groups"]["mean_pass_rate"],
                "zero_variance": g["groups"]["zero_variance_fraction"],
                "zero_variance_with_steps": g["groups"]["zero_variance_fraction_with_steps"],
                "zero_variance_by_depth": {dep: b["zero_variance_fraction"] for dep, b in g["groups"]["by_depth"].items()},
                "truncation_rate": g["truncation_rate"],
                "accuracy_sigma_free": g["accuracy_at_tolerance"]["sigma_free"],
            },
            "peak_vram": {"greedy": greedy["memory"], "group": group["memory"]},
            "generation_seconds": {"greedy": greedy["generation_seconds"], "group": group["generation_seconds"]},
            "prompt_tokens": {"max": prompt_tokens, "p50": _pct(lengths, 0.5)},
            "footprint": footprint.estimate_grpo(shape, prompt_tokens, fp_cfg),
            "footprint_pessimistic": footprint.estimate_grpo(shape, prompt_tokens, fp_cfg, pessimistic=True),
            "shape": footprint.describe(shape),
        })
    result = {"candidates": rows, "selection": select(rows, probe_cfg, fp_cfg)}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(result, indent=2))
    (out_root / "summary.md").write_text(render(result, fp_cfg))
    return result


def select(rows: list[dict], probe_cfg: dict, fp_cfg: dict) -> dict[str, Any]:
    """Your rule: lowest binary zero-variance fraction among models whose GRPO footprint fits the budget;
    if every model is above the threshold, train nothing and plan an SFT warm-up (Phase 3.6)."""
    done = [r for r in rows if not r.get("missing")]
    budget, threshold = fp_cfg["budget_gb"], probe_cfg["selection"]["max_binary_zero_variance"]
    if len(done) < len(rows):
        return {"decision": "incomplete", "missing": [r["label"] for r in rows if r.get("missing")]}
    fits = [r for r in done if r["footprint"]["peak_device_gb"] <= budget]
    zv = lambda r: r["group"]["zero_variance"]["binary"]
    acc = lambda r: r["greedy"]["id"]["accuracy_sigma_free"]["0.01"]
    ranked = sorted(fits, key=lambda r: (zv(r), -acc(r)))
    if all(zv(r) > threshold for r in done):
        return {"decision": "no_training_sft_warmup", "threshold": threshold,
                "best_label": ranked[0]["label"] if ranked else None,
                "reason": f"every model's binary zero-variance fraction is above {threshold:.0%}"}
    if not ranked:
        return {"decision": "nothing_fits", "budget_gb": budget}
    return {"decision": "train", "selected": ranked[0]["label"], "threshold": threshold, "budget_gb": budget,
            "ranking": [(r["label"], zv(r)) for r in ranked]}


def _p(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def render(result: dict, fp_cfg: dict) -> str:
    rows = [r for r in result["candidates"] if not r.get("missing")]
    lines = ["# Model probe (phase 3.5)", "",
             "Inference only. Greedy: 200 ID + 200 depth-5 problems. Group: 48 ID problems × k = 8 at T = 1.0.",
             "Prompt v4 (worked example, no σ hint), max_new_tokens 1024, generator 1.1.0, reward rule crps_log.", "",
             "## Greedy accuracy (σ-free, failures count as wrong)", "",
             "| model | split | ≤1% | ≤2% | ≤5% | ≤10% | strict ≤1% | truncated | parse fail | missing ± | 1σ cov | mean tokens |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for split, s in r["greedy"].items():
            a = s["accuracy_sigma_free"]
            lines.append(f"| {r['label']} | {split} | {_p(a['0.01'])} | {_p(a['0.02'])} | {_p(a['0.05'])} | {_p(a['0.1'])} "
                         f"| {_p(s['accuracy_strict']['0.01'])} | {_p(s['truncation_rate'])} | {_p(s['parse_fail_rate'])} "
                         f"| {_p(s['missing_sigma_rate'])} | {_p(s['coverage_1sigma'])} | {s['mean_completion_tokens']:.0f} |")
    lines += ["", "## Group probe (what a GRPO group of 8 sees)", "",
              "| model | zero-var binary | zero-var binary+steps | zero-var crps_log | zero-var crps | pass@8 | pass rate | truncated |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        g = r["group"]
        lines.append(f"| {r['label']} | **{_p(g['zero_variance']['binary'])}** | {_p(g['zero_variance_with_steps']['binary'])} "
                     f"| {_p(g['zero_variance']['crps_log'])} | {_p(g['zero_variance']['crps'])} | {_p(g['pass_at_k'])} "
                     f"| {_p(g['mean_pass_rate'])} | {_p(g['truncation_rate'])} |")
    lines += ["", "Binary zero-variance by depth: " + "; ".join(
        f"{r['label']}: " + ", ".join(f"d{d} {_p(v['binary'])}" for d, v in r["group"]["zero_variance_by_depth"].items())
        for r in rows)]
    lines += ["", "## Memory", "",
              f"Estimated GRPO footprint: 4-bit + LoRA r={fp_cfg['lora']['rank']}, group {fp_cfg['grpo']['group_size']}, "
              f"{fp_cfg['grpo']['completion_tokens']} completion tokens, micro-batch {fp_cfg['grpo']['micro_batch']}, "
              f"Unsloth-style chunked log-probs and offloaded checkpoints (configs/footprint.yaml). Budget {fp_cfg['budget_gb']} GB.", "",
              "| model | text params | prompt tokens (max) | est. GRPO peak | phase at peak | pessimistic (plain TRL) | measured inference peak, greedy (torch / device) | measured, group |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f, fpess, mem = r["footprint"], r["footprint_pessimistic"], r["peak_vram"]
        lines.append(f"| {r['label']} | {r['params_text'] / 1e9:.2f} B | {r['prompt_tokens']['max']} | **{f['peak_device_gb']:.2f} GB** "
                     f"| {f['phase_at_peak']} | {fpess['peak_device_gb']:.1f} GB "
                     f"| {mem['greedy']['torch_max_allocated_gib']:.2f} / {mem['greedy'].get('nvidia_smi_peak_used_gib_device_wide', float('nan')):.2f} GiB "
                     f"| {mem['group']['torch_max_allocated_gib']:.2f} / {mem['group'].get('nvidia_smi_peak_used_gib_device_wide', float('nan')):.2f} GiB |")
    lines += ["", "## Selection", "", "```", json.dumps(result["selection"], indent=2), "```", "",
              "Revisions: " + "; ".join(f"`{r['id']}` @ `{r['revision']}`" for r in rows)]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="model_probe")
    ap.add_argument("--only", default=None, help="run a single candidate label")
    ap.add_argument("--part", choices=["greedy", "group", "both"], default="both")
    ap.add_argument("--summarize", action="store_true", help="only rebuild the summary from existing results")
    ap.add_argument("--limit", type=int, default=None, help="max problems per split (smoke tests)")
    ap.add_argument("--output-dir", default=None, help="override output_dir (e.g. for smoke tests)")
    args = ap.parse_args()
    probe_cfg = load_config(args.config)
    if args.output_dir:
        probe_cfg["output_dir"] = args.output_dir
    out_root = REPO_ROOT / probe_cfg["output_dir"]
    if not args.summarize:
        parts = ["greedy", "group"] if args.part == "both" else [args.part]
        for cand in probe_cfg["candidates"]:
            if args.only and cand["label"] != args.only:
                continue
            for part in parts:
                base = load_config(probe_cfg[f"{part}_config"])
                print(f"=== {cand['label']} / {part}", flush=True)
                run(candidate_eval_config(base, cand, part, out_root), limit=args.limit)
    if args.limit is None:
        print(json.dumps(summarize(probe_cfg)["selection"], indent=2))


if __name__ == "__main__":
    main()

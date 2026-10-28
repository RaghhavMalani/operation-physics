"""R5-5 feasibility (DECISIONS.md D133): for each candidate family, does 4-bit + LoRA fit the 8 GB GPU at our sequence
lengths, what is the step time on stage R4's own training path, and how does the untrained base do on our format and
arithmetic (50 development A6c problems)? GPU; one process per model and step. Not a pre-registration.

    uv run python -m physcal.eval.r5_feasibility baseline llama32_3b
    uv run python -m physcal.eval.r5_feasibility train llama32_3b
    uv run python -m physcal.eval.r5_feasibility report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time

import numpy as np

from physcal.config import REPO_ROOT, load_config


def _config() -> dict:
    return load_config("r5_feasibility")


# ------------------------------------------------------------------------------------------- per-model chat
def messages(text: str, cand: dict, prompt_id: str) -> list[dict]:
    """The R4 eval messages; a model whose template has no system role gets the system prompt atop the user turn."""
    from physcal.prompt import build_messages

    msgs = build_messages(text, prompt_id)
    if cand["system_in_user"]:
        return [{"role": "user", "content": f"{msgs[0]['content']}\n\n{msgs[1]['content']}"}]
    return msgs


def tokenize_row(row: dict, text: str, tok, pc: dict, cand: dict) -> dict:
    """sft_phase12.tokenize with the candidate's chat handling and end-of-turn token (that one hard-codes Qwen's)."""
    from physcal.train.sft_phase12 import masked_spans

    prompt = tok.apply_chat_template(messages(text, cand, pc["prompt"]), tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    enc = tok(row["completion"], add_special_tokens=False, return_offsets_mapping=True)
    spans = masked_spans(row, pc)
    hit = lambda s, e: any(s < hi and e > lo for lo, hi in spans)
    labels = [-100 if hit(s, e) else t for t, (s, e) in zip(enc["input_ids"], enc["offset_mapping"])]
    eot = tok.convert_tokens_to_ids(cand["end_of_turn"])
    ids = prompt_ids + enc["input_ids"] + [eot]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(prompt_ids) + labels + [eot]}


# ------------------------------------------------------------------------------------------- GPU helpers
def gpu_temp() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return float(out.split()[0]) if out.strip() else 0.0


def cool(cfg: dict) -> int:
    """The overnight rule inside a stage: above the limit, pause; returns the number of pauses."""
    n = 0
    while gpu_temp() > cfg["max_gpu_temp_c"]:
        print(f"GPU above {cfg['max_gpu_temp_c']} C: pausing {cfg['pause_seconds']} s", flush=True)
        time.sleep(cfg["pause_seconds"])
        n += 1
    return n


def launch_cap(cfg: dict) -> float:
    """min(cap_max, total - host - margin) GiB from nvidia-smi, read before the model loads (D118 §1's rule)."""
    q = lambda f: float(subprocess.run(["nvidia-smi", f"--query-gpu={f}", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True).stdout.split()[0])
    total, used = q("memory.total"), q("memory.used")
    return min(cfg["cap_max_gib"], math.floor(((total - used) / 1024 - cfg["cap_margin_gib"]) * 10) / 10)


def model_path(name: str, cfg: dict) -> tuple[str, dict]:
    """A local snapshot of the candidate (downloaded once into the project's HF cache) and its provenance."""
    cand = cfg["candidates"][name]
    if "pinned_base" in cand:
        from physcal.base import resolve_model

        path, _, pin = resolve_model({"pinned_base": cand["pinned_base"]})
        return path, {"pinned_base": cand["pinned_base"], "pin": pin}
    from huggingface_hub import snapshot_download

    path = snapshot_download(cand["repo"])
    return path, {"repo": cand["repo"], "revision": os.path.basename(path.rstrip("/"))}


def _load(path: str, cap: float, max_seq: int):
    import torch
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 1024**3 / total))
    torch.cuda.reset_peak_memory_stats()
    model, tok = FastLanguageModel.from_pretrained(model_name=path, max_seq_length=max_seq, load_in_4bit=True,
                                                   fast_inference=False, dtype=None)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok, FastLanguageModel


# ------------------------------------------------------------------------------------------- baseline
def baseline_problems(cfg: dict) -> list[dict]:
    from physcal.eval import stage_a

    bc = cfg["baseline"]
    probs = [p for p in stage_a.load_sets(load_config("stage_a")) if p["_set"] == bc["set"]
             and p["noise"]["pair"]["member"] == bc["member"]]
    return [p for d in sorted({p["depth"] for p in probs}) for p in [q for q in probs if q["depth"] == d][:bc["n_per_depth"]]]


BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def lenient_number(completion: str) -> float | None:
    """For an untrained base that ignores our format: the first number of the last \\boxed{...}, else of the last
    <answer>, else the last number in the completion."""
    boxed = BOXED_RE.findall(completion)
    if boxed:
        nums = NUM_RE.findall(boxed[-1])
        return float(nums[0]) if nums else None
    if "<answer>" in completion:
        nums = NUM_RE.findall(completion.rsplit("<answer>", 1)[1])
        return float(nums[0]) if nums else None
    nums = NUM_RE.findall(completion)
    return float(nums[-1]) if nums else None


def score(problems: list[dict], completions: list[str], truncated: list[bool], tau: float) -> dict:
    """Format (the verifier parses; units right), the stated σ, arithmetic (μ against f(x_meas) and the truth), and a
    lenient arithmetic read that ignores the format (the final number, in the requested unit or in SI)."""
    from physcal.verify.reward import reward

    rows = []
    for p, c, t in zip(problems, completions, truncated):
        _, b = reward({k: v for k, v in p.items() if k != "_set"}, c)
        mu = b.get("mu_si")
        fx = p["noise"]["point_estimate_si"]
        ln_err = abs(math.log(mu / fx)) if b["reason"] == "ok" and mu and mu > 0 else None
        x, unit = lenient_number(c), p["target_value"] / p["target_value_si"]
        close = lambda v, tol: v is not None and v > 0 and abs(math.log(v / fx)) <= tol
        rows.append({"depth": p["depth"], "reason": b["reason"], "truncated": t, "sigma": b.get("sigma_si") is not None,
                     "steps_block": bool(b.get("steps_block_found")), "ln_err": ln_err,
                     "rel_err_truth": b.get("rel_error"),
                     "lenient_unit_1pct": close(None if x is None else x / unit, math.log(1.01)),
                     "lenient_any_1pct": close(None if x is None else x / unit, math.log(1.01)) or close(x, math.log(1.01))})
    share = lambda f: float(np.mean([f(r) for r in rows]))
    return {"n": len(rows), "reasons": {k: sum(r["reason"] == k for r in rows) for k in {r["reason"] for r in rows}},
            "format_parsed": share(lambda r: r["reason"] != "parse_fail"), "units_ok": share(lambda r: r["reason"] == "ok"),
            "sigma_stated": share(lambda r: r["reason"] == "ok" and r["sigma"]),
            "steps_block": share(lambda r: r["steps_block"]), "truncated": share(lambda r: r["truncated"]),
            "computation_correct": share(lambda r: r["ln_err"] is not None and r["ln_err"] <= tau),
            "within_5pct_of_f_xmeas": share(lambda r: r["ln_err"] is not None and r["ln_err"] <= math.log(1.05)),
            "within_1pct_of_truth": share(lambda r: r["rel_err_truth"] is not None and abs(r["rel_err_truth"]) <= 0.01),
            "lenient_within_1pct_requested_unit": share(lambda r: r["lenient_unit_1pct"]),
            "lenient_within_1pct_unit_or_si": share(lambda r: r["lenient_any_1pct"]),
            "lenient_by_depth": {str(d): float(np.mean([r["lenient_any_1pct"] for r in rows if r["depth"] == d]))
                                 for d in sorted({r["depth"] for r in rows})},
            "by_depth_computation_correct": {str(d): float(np.mean([r["ln_err"] is not None and r["ln_err"] <= tau
                                                                    for r in rows if r["depth"] == d]))
                                             for d in sorted({r["depth"] for r in rows})},
            "rows": rows}


def baseline(name: str, cfg: dict | None = None) -> dict:
    import torch

    from physcal.eval.run_eval import GpuMemoryPoller

    cfg = cfg or _config()
    bc, cand = cfg["baseline"], cfg["candidates"][name]
    path, prov = model_path(name, cfg)
    cap = launch_cap(cfg)
    model, tok, FLM = _load(path, cap, cfg["train"]["max_seq_length"])
    FLM.for_inference(model)
    tok.padding_side = "left"
    eot = tok.convert_tokens_to_ids(cand["end_of_turn"])
    probs = baseline_problems(cfg)
    texts = [tok.apply_chat_template(messages(p["text"], cand, bc["prompt"]), tokenize=False, add_generation_prompt=True)
             for p in probs]
    comps, trunc, ntok, pauses = [], [], [], 0
    batch, backoffs, i = bc["batch_size"], 0, 0
    t0 = time.time()
    with GpuMemoryPoller(interval_s=2.0) as poller:
        while i < len(texts):
            pauses += cool(cfg)
            enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            try:
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=bc["max_new_tokens"], do_sample=False,
                                         eos_token_id=list({tok.eos_token_id, eot}), pad_token_id=tok.pad_token_id)
            except torch.OutOfMemoryError:  # run_eval's rule: halve the batch and retry the same prompts
                if batch == 1:
                    raise
                del enc
                torch.cuda.empty_cache()
                batch, backoffs = max(1, batch // 2), backoffs + 1
                print(f"  OOM: batch -> {batch} (backoff #{backoffs})", flush=True)
                continue
            i += enc["input_ids"].shape[0]
            for row in out[:, enc["input_ids"].shape[1]:]:
                ids = row.tolist()
                n = next((j for j, t in enumerate(ids) if t in (tok.eos_token_id, eot)), len(ids))
                comps.append(tok.decode(ids[:n], skip_special_tokens=True))
                trunc.append(n >= bc["max_new_tokens"])
                ntok.append(n)
            print(f"  generated {len(comps)}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
    tau = load_config("pairs")["analysis"]["tau"]
    res = {"kind": "R5-5 feasibility: untrained base on 50 development A6c problems (D133)", "model": name, **prov,
           "cap_gib": cap, "seconds": time.time() - t0, "temperature_pauses": pauses, "oom_backoffs": backoffs,
           "batch_size_final": batch,
           "completion_tokens_median": float(np.median(ntok)), "prompt_tokens_max": max(len(tok(t)["input_ids"]) for t in texts),
           "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
           "nvidia_smi_peak_gib": poller.peak_mib / 1024, **score(probs, comps, trunc, tau)}
    res["examples"] = comps[:3]
    out = REPO_ROOT / cfg["output_dir"] / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline.json").write_text(json.dumps(res, indent=1, default=str))
    with open(out / "baseline_completions.jsonl", "w") as fh:
        for p, c in zip(probs, comps):
            fh.write(json.dumps({"seed": p["seed"], "family": p["family"], "depth": p["depth"], "completion": c}) + "\n")
    return res


def rescore(name: str, cfg: dict | None = None) -> dict:
    """CPU: re-score a saved baseline (the same problems, the saved completions and truncation flags)."""
    cfg = cfg or _config()
    out = REPO_ROOT / cfg["output_dir"] / name
    res = json.loads((out / "baseline.json").read_text())
    comps = [json.loads(line)["completion"] for line in open(out / "baseline_completions.jsonl")]
    res.update(score(baseline_problems(cfg), comps, [r["truncated"] for r in res["rows"]],
                     load_config("pairs")["analysis"]["tau"]))
    (out / "baseline.json").write_text(json.dumps(res, indent=1, default=str))
    return res


# ------------------------------------------------------------------------------------------- train
def train(name: str, cfg: dict | None = None) -> dict:
    import torch

    from physcal.eval.run_eval import GpuMemoryPoller
    from physcal.train import stage_b, stage_r4

    cfg = cfg or _config()
    tc, cand = cfg["train"], cfg["candidates"][name]
    r4 = load_config("stage_r4")
    acfg = stage_r4.arm_config(tc["arm"], r4, r4["seed"])
    pc = load_config(acfg["sft_phase12_config"])
    path, prov = model_path(name, cfg)
    cap = launch_cap(cfg)
    model, tok, FLM = _load(path, cap, tc["max_seq_length"])
    lora = r4["lora"]
    model = FLM.get_peft_model(model, r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                               target_modules=lora["target_modules"], use_gradient_checkpointing="unsloth",
                               random_state=r4["seed"])
    arm_file, prob_file, _ = stage_b.check_data(acfg)
    texts = {json.loads(line)["seed"]: json.loads(line)["text"] for line in prob_file.open()}
    rows, lens, dropped = [], [], 0
    for line in arm_file.open():
        r = json.loads(line)
        t = tokenize_row(r, texts[r["seed"]], tok, pc, cand)
        lens.append(len(t["input_ids"]))
        if len(t["input_ids"]) > tc["max_seq_length"]:
            dropped += 1
            continue
        rows.append(t)
    pauses = cool(cfg)
    oom = stage_b.oom_check(model, tok, rows, r4["train"]["per_device_train_batch_size"])
    scratch = REPO_ROOT / cfg["output_dir"] / name / "scratch_train"
    with GpuMemoryPoller(interval_s=2.0) as poller:
        fit = stage_b.fit(model, tok, rows, scratch, acfg, r4["seed"], tc["max_steps"])
    shutil.rmtree(scratch, ignore_errors=True)  # a few steps' adapter: not kept
    res = {"kind": "R5-5 feasibility: stage R4's SFT path for a few steps (D133)", "model": name, **prov, "cap_gib": cap,
           "sequence_tokens": {"n_rows": len(lens), "max": max(lens), "mean": float(np.mean(lens)),
                               "p99": float(np.percentile(lens, 99)), "dropped_over_max": dropped},
           "oom_check": oom, "steps": fit["steps"], "seconds_per_step": fit["seconds_per_step"],
           "train_loss": fit["train_loss"], "temperature_pauses": pauses,
           "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
           "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
           "nvidia_smi_peak_gib": poller.peak_mib / 1024}
    out = REPO_ROOT / cfg["output_dir"] / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.json").write_text(json.dumps(res, indent=1, default=str))
    return res


# ------------------------------------------------------------------------------------------- report
def report(cfg: dict | None = None) -> str:
    cfg = cfg or _config()
    root = REPO_ROOT / cfg["output_dir"]
    load = lambda n, f: json.loads((root / n / f).read_text()) if (root / n / f).exists() else {}
    pct = lambda v: "–" if v is None else f"{100 * v:.0f}%"
    L = ["# R5-5 feasibility: 4-bit + LoRA on the 8 GB GPU, and the untrained bases on our task (D133)", "",
         "Not a pre-registration. **Training:** stage R4's own path (LoRA r 16 on all 7 projections, batch 2 × 4, "
         "adamw_8bit, gradient checkpointing) on D-pos-v2's 3,000 R4-1 rows, tokenized with each model's template, "
         f"{cfg['train']['max_steps']} steps under the launch cap. **Baseline:** 50 development A6c problems (10 per "
         "depth), prompt v5, greedy, 1,280 new tokens, the default verifier; OOM halves the batch (run_eval's rule). "
         "**Lenient arithmetic** ignores the format: the final number (boxed, else in <answer>, else the last) within "
         "1% of f(x_meas), in the requested unit or SI.", "",
         "## Training", "", "| model | seq tokens mean / p99 / max | rows > 2,048 | OOM check: longest batch | "
         "peak reserved (GiB) | device peak (GiB) | s / step | × reference |", "|---|---|---|---|---|---|---|---|"]
    ref = load("qwen25_math_1p5b", "train.json").get("seconds_per_step")
    for n in cfg["candidates"]:
        t = load(n, "train.json")
        if not t:
            L.append(f"| {n} | – | – | – | – | – | – | – |")
            continue
        st, o = t["sequence_tokens"], t["oom_check"]
        L.append(f"| {n} | {st['mean']:.0f} / {st['p99']:.0f} / {st['max']} | {st['dropped_over_max']} | "
                 f"pass ({o['tokens']} tokens, {o['max_reserved_gib']:.2f} GiB) | {t['peak_reserved_gib']:.2f} | "
                 f"{t['nvidia_smi_peak_gib']:.2f} | {t['seconds_per_step']:.1f} | "
                 f"{t['seconds_per_step'] / ref:.2f} |" if ref else " – |")
    L += ["", "## Untrained base on 50 development problems", "",
          "| model | parsed | units ok | σ stated | truncated | median tokens | strict computation-correct | lenient "
          "arithmetic (by depth 1–5) | eval batch that fits | peak reserved (GiB) | seconds |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in cfg["candidates"]:
        b = load(n, "baseline.json")
        if not b:
            continue
        bd = " / ".join(pct(v) for v in b["lenient_by_depth"].values())
        L.append(f"| {n} | {pct(b['format_parsed'])} | {pct(b['units_ok'])} | {pct(b['sigma_stated'])} | "
                 f"{pct(b['truncated'])} | {b['completion_tokens_median']:.0f} | {pct(b['computation_correct'])} | "
                 f"{pct(b['lenient_within_1pct_unit_or_si'])} ({bd}) | {b.get('batch_size_final', cfg['baseline']['batch_size'])} "
                 f"| {b['peak_reserved_gib']:.2f} | {b['seconds']:.0f} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["baseline", "train", "rescore", "report"])
    ap.add_argument("model", nargs="?")
    a = ap.parse_args()
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    if a.what == "report":
        md = report()
        (REPO_ROOT / _config()["output_dir"] / "summary.md").write_text(md)
        print(md)
    else:
        res = {"baseline": baseline, "train": train, "rescore": rescore}[a.what](a.model)
        print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "examples")}, indent=1, default=str))


if __name__ == "__main__":
    main()

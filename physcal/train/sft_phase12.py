"""Phase 12 SFT data (D80): PREP ONLY. Phase 12 is pre-registered after X2 reports; choices marked [REVIEW] in
configs/sft_phase12.yaml stay open until then.

    uv run python -m physcal.train.sft_phase12 build      # CPU: problems + one completion file per arm + manifest

Two-channel format (round-2 synthesis): `<answer>μ ± σ unit</answer>` then `<confidence>c%</confidence>`, with c one
of the configured bins. Problems are generator-2.0.0 training-split problems over the 12 training cells × the noise
levels; the same problems and the same confidence draw go into every arm. The answer is the computation from the
displayed values, f(x_meas) (the generator's point estimate), never the noise-free truth. The arms differ only in
the σ channel:
- implicit: σ* written after "±" and masked from the loss (RL learns σ);
- explicit: one propagation line in <steps>, `sigma_rel = sqrt((a1*s1)^2 + ...) = X`, with a_i = |d ln f / d ln x_i|
  (the chain's log-sensitivities at the displayed values) and s_i the stated noise levels; the answer's σ is X·|μ|.
  The line's numbers are trained or masked per config;
- supervised: σ* written and trained (the supervised σ* baseline; no RL).
Completions are stored untokenized with character spans (σ, the line's numbers, c); `tokenize` applies an arm's
masking, so a masking change needs no rebuild.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
from collections import Counter

import numpy as np

from physcal.config import REPO_ROOT, load_config

CONFIDENCE_RE = re.compile(r"<confidence>\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*</confidence>", re.I)
LINE_NUMBER_RE = re.compile(r"(?<![\^0-9.])[0-9]+(?:\.[0-9]*)?(?:[eE][-+]?[0-9]+)?")  # not the ^2 exponents
CONFIDENCE_STREAM = "phase12-confidence"  # the draw's own random stream, keyed by the problem seed


def _config() -> dict:
    return load_config("sft_phase12")


def parse_confidence(completion: str, bins) -> int | None:
    """c from the single <confidence> block after </answer>; None unless there is exactly one and it is a bin."""
    end = completion.find("</answer>")
    if end < 0:
        return None
    found = CONFIDENCE_RE.findall(completion[end:])
    if len(found) != 1:
        return None
    v = float(found[0])
    return int(v) if v.is_integer() and int(v) in bins else None


def confidence_draw(seed: int, bins) -> int:
    h = int.from_bytes(hashlib.sha256(f"{CONFIDENCE_STREAM}|{seed}".encode()).digest()[:8], "little")
    return int(np.random.default_rng(h).choice(list(bins)))


def point_estimate(problem) -> tuple[float, float]:
    """f(x_meas) in SI and in the requested unit (the computation-correct answer for the displayed values)."""
    pe = problem.noise["point_estimate_si"]
    return pe, pe * problem.target_value / problem.target_value_si


def propagation_line(problem, env_cfg: dict, fmt: str) -> tuple[str, float, float]:
    """The explicit arm's line, its value X (computed from the displayed, rounded terms, so the line is
    self-consistent) and the exact first-order σ_log it rounds."""
    from physcal.env import noise as noise_mode
    from physcal.env.formulas import get_family
    from physcal.env.generator import chain_of

    noisy = [g for g in problem.givens if g.get("noise_rel")]
    if not noisy:
        return "sigma_rel = 0", 0.0, 0.0
    fam, chain = get_family(problem.family), chain_of(problem)
    sens = noise_mode.log_sensitivities(fam, chain, problem.given_values_si, env_cfg["generator"]["condition_rel_step"])
    levels = {g["symbol"]: g["noise_rel"] for g in noisy}
    exact = noise_mode.delta_sigma_log(sens, levels)
    shown = [(float(format(abs(sens[g["symbol"]]), fmt)), g["noise_rel"]) for g in noisy]
    x = float(format(float(np.sqrt(sum((a * s) ** 2 for a, s in shown))), fmt))
    terms = " + ".join(f"({format(a, fmt)}*{s:g})^2" for a, s in shown)
    return f"sigma_rel = sqrt({terms}) = {format(x, fmt)}", x, exact


def build_example(problem, arm: str, cfg: dict, env_cfg: dict, c: int) -> dict:
    """One arm's completion for one problem, with character spans and the σ_log its answer encodes."""
    from physcal.train.sft_data import solution

    acfg = cfg["arms"][arm]
    pe_si, mu = point_estimate(problem)
    shown = dataclasses.replace(problem, target_value_si=pe_si, target_value=mu)
    line = None
    if acfg["explicit_line"]:
        line, sigma_log, exact = propagation_line(problem, env_cfg, cfg["sigma_format"])
    else:
        sigma_log = problem.noise["sigma_star_log"]
    ex = solution(shown, sigma_log)
    completion, (lo, hi) = ex.completion, ex.sigma_span
    spans: dict = {"line_numbers": []}
    if line is not None:
        at = completion.rindex("</steps>\n<answer>")
        completion = completion[:at] + line + "\n" + completion[at:]
        lo, hi = lo + len(line) + 1, hi + len(line) + 1
        eq = line.index("=") + 1
        spans["line_numbers"] = [[at + eq + m.start(), at + eq + m.end()] for m in LINE_NUMBER_RE.finditer(line[eq:])]
    spans["sigma"] = [lo, hi]
    tail = f"\n<confidence>{c}%</confidence>"
    spans["confidence"] = [len(completion) + tail.index(">") + 1, len(completion) + tail.index("%")]
    completion += tail
    out = {"seed": problem.seed, "family": problem.family, "depth": problem.depth, "noise": problem.noise["level"],
           "arm": arm, "completion": completion, "spans": spans, "sigma_log_answer": sigma_log,
           "sigma_star_log": problem.noise["sigma_star_log"], "confidence": c}
    if line is not None:
        out["line"], out["sigma_log_delta_exact"] = line, exact
    return out


def masked_spans(row: dict, cfg: dict) -> list[list[int]]:
    """The character spans an arm masks from the loss."""
    acfg, spans = cfg["arms"][row["arm"]], row["spans"]
    masked = []
    if not acfg["train_sigma"]:
        masked.append(spans["sigma"])
    if acfg["explicit_line"] and not acfg.get("train_line_numbers", True):
        masked += spans["line_numbers"] + [spans["sigma"]]
    if not cfg["confidence"]["train"]:
        masked.append(spans["confidence"])
    return masked


def tokenize(row: dict, problem_text: str, tok, cfg: dict, chat_template_kwargs: dict | None = None) -> dict:
    """input_ids / labels as physcal.train.sft_data.tokenize, with the arm's spans masked (-100)."""
    from physcal.prompt import build_messages

    prompt = tok.apply_chat_template(build_messages(problem_text, cfg["prompt"]), tokenize=False,
                                     add_generation_prompt=True, **(chat_template_kwargs or {}))
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    enc = tok(row["completion"], add_special_tokens=False, return_offsets_mapping=True)
    spans = masked_spans(row, cfg)
    hit = lambda s, e: any(s < hi and e > lo for lo, hi in spans)
    labels = [-100 if hit(s, e) else t for t, (s, e) in zip(enc["input_ids"], enc["offset_mapping"])]
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    ids = prompt_ids + enc["input_ids"] + [eos]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(prompt_ids) + labels + [eos],
            "n_masked": sum(1 for x in labels if x == -100)}


def problems(cfg: dict) -> list:
    from physcal.env.generator import generate_problem, train_cells

    env_cfg = load_config(cfg["env_config"])
    if cfg["seed_offset"] + cfg["n_examples"] > env_cfg["splits"]["eval_seed_offset"]:
        raise ValueError("Phase 12 SFT seeds reach the eval range")
    combos = [(f, d, s) for f, d in train_cells(env_cfg) for s in cfg["noise_levels"]]
    return [generate_problem(*combos[i % len(combos)][:2], cfg["seed_offset"] + i, split="train", cfg=env_cfg,
                             noise=combos[i % len(combos)][2]) for i in range(cfg["n_examples"])]


def build() -> dict:
    from physcal.env.generator import generator_version

    cfg = _config()
    env_cfg = load_config(cfg["env_config"])
    out = REPO_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    probs = problems(cfg)
    bins = cfg["confidence"]["bins"]
    files = {"problems.jsonl": "".join(json.dumps(p.to_dict(), sort_keys=True) + "\n" for p in probs)}
    rows = {}
    for arm in cfg["arms"]:
        rows[arm] = [build_example(p, arm, cfg, env_cfg, confidence_draw(p.seed, bins)) for p in probs]
        files[f"{arm}.jsonl"] = "".join(json.dumps(r) + "\n" for r in rows[arm])
    for name, blob in files.items():
        (out / name).write_text(blob)
    expl = rows.get("explicit", [])
    manifest = {
        "decision": "D80 (Phase 12 SFT data, prep; not pre-registered)", "config": cfg,
        "generator_version": generator_version(env_cfg), "n_examples": len(probs),
        "seeds": [cfg["seed_offset"], cfg["seed_offset"] + len(probs) - 1],
        "cells": dict(sorted(Counter(f"{p.family}/{p.depth}/noise={p.noise['level']}" for p in probs).items())),
        "confidence_counts": dict(sorted(Counter(r["confidence"] for r in rows[next(iter(rows))]).items())),
        "sigma_star_log_median_by_noise": {str(s): float(np.median([p.noise["sigma_star_log"] for p in probs
                                                                    if p.noise["level"] == s]))
                                           for s in cfg["noise_levels"]},
        "explicit_line_vs_sigma_star": None if not expl else {
            "median_abs_log_ratio_noisy": float(np.median([abs(np.log(r["sigma_log_answer"] / r["sigma_star_log"]))
                                                           for r in expl if r["noise"] > 0]))},
        "sha256": {name: hashlib.sha256(blob.encode()).hexdigest() for name, blob in files.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    sample = ["# Phase 12 SFT data: examples (D80, prep)", ""]
    for i in range(0, len(probs), max(1, len(probs) // 8))[:8]:
        sample += [f"## seed {probs[i].seed}: {probs[i].family} / depth {probs[i].depth} / noise "
                   f"{probs[i].noise['level']}", "", "Problem:", "", "```", probs[i].text, "```", ""]
        for arm in cfg["arms"]:
            sample += [f"**{arm}** (masked: {masked_spans(rows[arm][i], cfg)})", "", "```", rows[arm][i]["completion"],
                       "```", ""]
    (out / "samples.md").write_text("\n".join(sample))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["build"])
    ap.parse_args()
    m = build()
    print(json.dumps({k: v for k, v in m.items() if k not in ("config", "sha256")}, indent=1))


if __name__ == "__main__":
    main()

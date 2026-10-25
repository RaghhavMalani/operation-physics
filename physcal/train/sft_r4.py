"""R4-1 (DECISIONS.md D112; pre-registered in D113): the D-pos-v2 / B-v2 SFT data, CPU only.

    OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.train.sft_r4 build   # results/phase12_sft_data_r4/

D80's 3,000 training seeds and cells with mixed per-given noise levels (each given 0 / 1 / 2 / 5%, independently;
generator 2.0.0's `mixed_levels` option). Two arms on the same problems and the same `<confidence>` draw:
- explicit (D-pos-v2): D80's solution, then one line per noisy given before `<steps>`,
  `S_x = <expression> = <the expression with the solution's numbers> = <value>`, where the expression is the given's
  total log-sensitivity d ln f / d ln x by the chain rule over the steps, in the problem's own symbols (a power-law
  term is a constant and is stated as its exponent); then D80's line at the end of `<steps>`,
  `sigma_rel = sqrt((S_1*s_1)^2 + ...) = X`, with the S values as written and each given's own level; the answer's
  σ is X·|μ| (D80's copy);
- supervised (B-v2): D80's supervised arm (σ* written after "±" and trained; no line).
Rows have sft_phase12's shape (character spans, recorded masks), so `sft_phase12.tokenize` applies.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from collections import Counter
from functools import lru_cache

import numpy as np
import sympy as sp

from physcal.config import REPO_ROOT, load_config

LINE_PREFIX = "S_"  # sensitivity names: no family quantity starts with it


def _config() -> dict:
    return load_config("sft_r4")


# ------------------------------------------------------------------------------------------- sensitivities
def _local(fam, name: str, out: str) -> dict:
    """d ln out / d ln a for each non-constant input a of one solved step: the shorter of the implicit form
    a·(−∂F/∂a)/(∂F/∂out)/out (F = lhs − rhs) and the explicit form a·∂h/∂a / h; a constant is kept exact."""
    f = fam.formula(name)
    solved = f.solved[out]
    y, F = fam.symbols[out], f.eq.lhs - f.eq.rhs
    res = {}
    for a in solved.args:
        if a in fam.si_constants:
            continue
        x = fam.symbols[a]
        explicit = sp.simplify(x * sp.diff(solved.expr, x) / solved.expr)
        implicit = sp.cancel(-x * sp.diff(F, x) / (sp.diff(F, y) * y))
        res[a] = explicit if explicit.is_number else min((implicit, explicit),
                                                         key=lambda e: (sp.count_ops(e), len(sp.sstr(e))))
    return res


@lru_cache(maxsize=None)
def sensitivity_expressions(family: str, steps: tuple, givens: tuple) -> dict:
    """Each given's total d ln(target) / d ln(given) as a sympy expression in the chain's symbols (chain rule over
    the steps, paths summed, then factor(cancel(·)))."""
    from physcal.env.formulas import get_family

    fam = get_family(family)
    per_output: dict = {}
    for name, out in steps:
        total: dict = {}
        for a, e in _local(fam, name, out).items():
            if a in givens:
                total[a] = total.get(a, 0) + e
            else:
                for g, s in per_output[a].items():
                    total[g] = total.get(g, 0) + e * s
        per_output[out] = total
    return {g: sp.factor(sp.cancel(v)) for g, v in per_output[steps[-1][1]].items()}


def solution_values(problem) -> dict:
    """The SI numbers the solution writes: givens, constants, intermediates and the target f(x_meas)."""
    from physcal.env.formulas import get_family

    fam = get_family(problem.family)
    vals = {g["symbol"]: g["value_si"] for g in problem.givens}
    vals.update({k: v for k, (v, _) in fam.constants.items()})
    vals.update(problem.intermediates)
    vals[problem.target_symbol] = problem.noise["point_estimate_si"] if problem.noise else problem.target_value_si
    return vals


def sensitivity_lines(problem, fmt: str) -> list[dict]:
    """One line per noisy given (problem order): its text, the shown value, and whether it is a power-law term."""
    from physcal.env.formulas import get_family
    from physcal.env.generator import chain_of
    from physcal.train.sft_data import _num

    fam = get_family(problem.family)
    chain = chain_of(problem)
    exprs = sensitivity_expressions(problem.family, tuple(chain.steps), tuple(chain.givens))
    vals = solution_values(problem)
    out = []
    for g in problem.givens:
        if not g.get("noise_rel"):
            continue
        sym, expr = g["symbol"], exprs[g["symbol"]]
        if expr.is_number:
            shown = format(float(expr), fmt)
            text = f"{LINE_PREFIX}{sym} = {shown}"
        else:
            names = sorted(str(s) for s in expr.free_symbols)
            numbers = {fam.symbols[a]: sp.Symbol(_num(vals[a])) for a in names}
            value = float(expr.subs({fam.symbols[a]: float(_num(vals[a])) for a in names}))
            shown = format(value, fmt)
            text = (f"{LINE_PREFIX}{sym} = {sp.sstr(expr)} = {sp.sstr(expr.subs(numbers, simultaneous=True))} = "
                    f"{shown}")
        out.append({"symbol": sym, "line": text, "value": float(shown), "shown": shown,
                    "power_law": bool(expr.is_number), "level": g["noise_rel"]})
    return out


def propagation_line(sens: list[dict], fmt: str) -> tuple[str, float]:
    """`sigma_rel = sqrt((S*s)^2 + ...) = X` from the shown S values and levels (X at `fmt`), or `sigma_rel = 0`."""
    if not sens:
        return "sigma_rel = 0", 0.0
    x = float(format(math.sqrt(sum((t["value"] * t["level"]) ** 2 for t in sens)), fmt))
    terms = " + ".join(f"({t['shown']}*{t['level']:g})^2" for t in sens)
    return f"sigma_rel = sqrt({terms}) = {format(x, fmt)}", x


# ------------------------------------------------------------------------------------------- rows
def build_example(problem, arm: str, cfg: dict, c: int) -> dict:
    """One arm's completion (sft_phase12's row shape: spans for σ, the line's numbers, the S lines' numbers, c)."""
    from physcal.train import sft_phase12
    from physcal.train.sft_data import solution

    if not cfg["arms"][arm]["explicit_line"]:
        return sft_phase12.build_example(problem, arm, cfg, None, c)
    pe_si, mu = sft_phase12.point_estimate(problem)
    shown = dataclasses.replace(problem, target_value_si=pe_si, target_value=mu)
    sens = sensitivity_lines(problem, cfg["sensitivity_format"])
    line, x = propagation_line(sens, cfg["sigma_format"])
    ex = solution(shown, x)
    completion, (lo, hi) = ex.completion, ex.sigma_span
    spans: dict = {"line_numbers": [], "sensitivity_numbers": []}
    if sens:
        at = completion.index("\n<steps>\n")
        block = "".join("\n" + t["line"] for t in sens)
        for t, start in zip(sens, np.cumsum([0] + [len(t["line"]) + 1 for t in sens[:-1]])):
            base = at + 1 + int(start)
            eq = t["line"].index("=") + 1
            spans["sensitivity_numbers"] += [[base + eq + m.start(), base + eq + m.end()]
                                             for m in sft_phase12.LINE_NUMBER_RE.finditer(t["line"][eq:])]
        completion = completion[:at] + block + completion[at:]
        lo, hi = lo + len(block), hi + len(block)
    at = completion.rindex("</steps>\n<answer>")
    completion = completion[:at] + line + "\n" + completion[at:]
    lo, hi = lo + len(line) + 1, hi + len(line) + 1
    eq = line.index("=") + 1
    spans["line_numbers"] = [[at + eq + m.start(), at + eq + m.end()]
                             for m in sft_phase12.LINE_NUMBER_RE.finditer(line[eq:])]
    spans["sigma"] = [lo, hi]
    tail = f"\n<confidence>{c}%</confidence>"
    spans["confidence"] = [len(completion) + tail.index(">") + 1, len(completion) + tail.index("%")]
    completion += tail
    return {"seed": problem.seed, "family": problem.family, "depth": problem.depth, "noise": problem.noise["level"],
            "arm": arm, "completion": completion, "spans": spans, "sigma_log_answer": x,
            "sigma_star_log": problem.noise["sigma_star_log"], "confidence": c, "line": line,
            "sensitivities": [{k: t[k] for k in ("symbol", "value", "power_law", "level")} for t in sens]}


def problems(cfg: dict) -> list:
    """D80's seeds and cells (sft_phase12.problems' round-robin), each with mixed per-given levels."""
    from physcal.env.generator import generate_problem, train_cells

    p12 = load_config(cfg["seeds_from"])
    env_cfg = load_config(cfg["env_config"])
    combos = [(f, d) for f, d in train_cells(env_cfg) for _ in p12["noise_levels"]]
    return [generate_problem(*combos[i % len(combos)], p12["seed_offset"] + i, split="train", cfg=env_cfg,
                             mixed_levels=cfg["mixed_levels"]) for i in range(p12["n_examples"])]


def build() -> dict:
    from physcal.env.generator import generator_version
    from physcal.train import sft_phase12

    cfg = _config()
    env_cfg = load_config(cfg["env_config"])
    out = REPO_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    probs = problems(cfg)
    files = {"problems.jsonl": "".join(json.dumps(p.to_dict(), sort_keys=True) + "\n" for p in probs)}
    rows = {}
    for arm in cfg["arms"]:
        rows[arm] = [build_example(p, arm, cfg, sft_phase12.confidence_draw(p.seed, cfg["confidence"]["bins"]))
                     for p in probs]
        for r in rows[arm]:
            r["masked"] = sft_phase12.masked_spans(r, cfg)
        files[f"{arm}.jsonl"] = "".join(json.dumps(r) + "\n" for r in rows[arm])
    for name, blob in files.items():
        (out / name).write_text(blob)
    levels = Counter(g["noise_rel"] if g.get("noise_rel") else 0.0 for p in probs for g in p.givens)
    terms = [t for r in rows["explicit"] for t in r["sensitivities"]]
    noisy = [(r, p) for r, p in zip(rows["explicit"], probs) if r["sigma_star_log"] > 0]
    manifest = {
        "decision": "D113 (R4-1: the D-pos-v2 / B-v2 data, pre-registered)", "config": cfg,
        "generator_version": generator_version(env_cfg), "n_examples": len(probs),
        "seeds": [probs[0].seed, probs[-1].seed],
        "cells": dict(sorted(Counter(f"{p.family}/{p.depth}" for p in probs).items())),
        "given_levels": {f"{k:g}": v for k, v in sorted(levels.items())},
        "problems_all_exact": sum(1 for p in probs if p.noise["level"] == 0.0),
        "problems_mixing_noisy_levels": sum(1 for p in probs if len({g["noise_rel"] for g in p.givens
                                                                     if g.get("noise_rel")}) > 1),
        "sensitivity_terms": len(terms), "power_law_terms": sum(t["power_law"] for t in terms),
        "explicit_x_vs_sigma_star_median_abs_log": float(np.median(
            [abs(math.log(r["sigma_log_answer"] / r["sigma_star_log"])) for r, _ in noisy if r["sigma_log_answer"] > 0])),
        "confidence_counts": dict(sorted(Counter(r["confidence"] for r in rows["explicit"]).items())),
        "sha256": {name: hashlib.sha256(blob.encode()).hexdigest() for name, blob in files.items()},
        "masked_spans_per_example": {arm: float(np.mean([len(r["masked"]) for r in rows[arm]])) for arm in rows},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    sample = ["# R4-1 SFT data (D113): examples", ""]
    for i in range(0, len(probs), max(1, len(probs) // 8))[:8]:
        sample += [f"## seed {probs[i].seed}: {probs[i].family} / depth {probs[i].depth}", "", "Problem:", "", "```",
                   probs[i].text, "```", ""]
        for arm in cfg["arms"]:
            sample += [f"**{arm}** (masked: {rows[arm][i]['masked']})", "", "```", rows[arm][i]["completion"], "```", ""]
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

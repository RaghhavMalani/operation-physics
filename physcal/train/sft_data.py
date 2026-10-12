"""Phase 3.6 SFT data: short generator-written solutions in the exact output format.

Every example comes from `generate_problem(..., split="train")` with a seed below the eval offset, so no eval
seed and no held-out family or depth can appear. The σ number in `<answer>` is masked out of the loss (D35):
σ is learned only from the proper scoring rule during RL.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from physcal.env.formulas import get_family
from physcal.env.generator import Problem, env_config, generate_problem, train_cells
from physcal.prompt import build_messages

SIG = 4  # significant figures written in solutions (rounding error <= 0.05%, far inside the 1% band)


def _num(x: float) -> str:
    return f"{x:.{SIG}g}"


def _with_unit(value: str, unit: str) -> str:
    return f"{value} {unit}".strip()


@dataclass(frozen=True)
class Example:
    problem: Problem
    completion: str
    sigma_span: tuple[int, int]  # character span of the σ number inside `completion`


def solution(problem: Problem, sigma_rel: float) -> Example:
    """SI conversion, one line per chain step with substituted numbers, then <steps> and <answer>."""
    fam = get_family(problem.family)
    lines = []
    conversions = []
    for g in problem.givens:
        si_unit = fam.quantities[g["symbol"]].si_unit
        shown = _with_unit(g["display"], g["unit"])
        if g["unit"] == si_unit:
            conversions.append(f"{g['symbol']} = {shown}")
        else:
            conversions.append(f"{g['symbol']} = {shown} = {_with_unit(_num(g['value_si']), si_unit)}")
    lines.append("SI values: " + "; ".join(conversions) + ".")
    if problem.distractors:
        lines.append("Not needed: " + ", ".join(d["symbol"] for d in problem.distractors) + ".")

    values = {g["symbol"]: g["value_si"] for g in problem.givens}
    values.update({k: v for k, (v, _) in fam.constants.items()})
    for name, out in problem.chain:
        solved = fam.formula(name).solved[out]
        result = problem.intermediates.get(out, problem.target_value_si)
        numbers = {fam.symbols[a]: sp.Symbol(_num(values[a])) for a in solved.args}
        substituted = sp.sstr(solved.expr.subs(numbers, simultaneous=True))
        unit = fam.quantities[out].si_unit
        lines.append(f"{out} = {sp.sstr(solved.expr)} = {substituted} = {_with_unit(_num(result), unit)}")
        values[out] = result
    if problem.target_unit != problem.target_si_unit:
        lines.append(f"In {problem.target_unit}: {problem.target_symbol} = "
                     f"{_with_unit(_num(problem.target_value), problem.target_unit)}")

    steps = "\n".join(f"{s} = {_with_unit(_num(v), problem.intermediate_si_units[s])}"
                      for s, v in problem.intermediates.items())
    steps_block = f"<steps>\n{steps}\n</steps>" if steps else "<steps>\n</steps>"
    head = "\n".join(lines) + f"\n{steps_block}\n<answer>{_num(problem.target_value)} ± "
    sigma = f"{abs(problem.target_value) * sigma_rel:.2g}"
    tail = f" {problem.target_unit}".rstrip() + "</answer>"
    completion = head + sigma + tail
    return Example(problem, completion, (len(head), len(head) + len(sigma)))


def build_examples(n: int, seed_offset: int, sigma_rel: float, cfg: dict | None = None) -> list[Example]:
    """Round-robin over the training cells; refuses any seed in the eval range."""
    cfg = cfg or env_config()
    offset = cfg["splits"]["eval_seed_offset"]
    if seed_offset + n > offset:
        raise ValueError(f"SFT seeds {seed_offset}..{seed_offset + n} reach the eval range (>= {offset})")
    cells = train_cells(cfg)
    return [solution(generate_problem(*cells[i % len(cells)], seed_offset + i, split="train", cfg=cfg), sigma_rel)
            for i in range(n)]


def tokenize(example: Example, tokenizer, prompt_id: str, chat_template_kwargs: dict) -> dict:
    """input_ids / labels: prompt tokens and the σ number masked (-100); the end-of-turn token is trained."""
    prompt = tokenizer.apply_chat_template(build_messages(example.problem.text, prompt_id), tokenize=False,
                                           add_generation_prompt=True, **chat_template_kwargs)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    enc = tokenizer(example.completion, add_special_tokens=False, return_offsets_mapping=True)
    lo, hi = example.sigma_span
    labels = [-100 if (start < hi and end > lo) else tid for tid, (start, end) in zip(enc["input_ids"], enc["offset_mapping"])]
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    input_ids = prompt_ids + enc["input_ids"] + [eos]
    labels = [-100] * len(prompt_ids) + labels + [eos]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels,
            "n_masked_sigma": sum(1 for (s, e) in enc["offset_mapping"] if s < hi and e > lo)}

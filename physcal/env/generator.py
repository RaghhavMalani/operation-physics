"""Problem generation: world -> chain -> displayed givens -> ground truth -> text.

`generate_problem(family, depth, seed)` is a pure function of (generator_version, family, depth, seed).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from physcal.config import load_config, stable_hash
from physcal.env import render
from physcal.env.formulas import FAMILY_NAMES, get_family
from physcal.env.graph import Chain, log_uniform
from physcal.env.units import from_si


class HeldOutError(ValueError):
    """Raised when a training request touches a held-out family or depth."""


@dataclass
class Problem:
    text: str
    target_symbol: str
    target_desc: str
    target_value_si: float
    target_si_unit: str
    target_unit: str
    target_value: float  # in target_unit
    intermediates: dict[str, float]  # symbol -> SI value, in chain order
    intermediate_si_units: dict[str, str]
    givens: list[dict]
    distractors: list[dict]
    chain: list[list[str]]  # [formula_name, output] in evaluation order
    family: str
    depth: int
    seed: int
    generator_version: str
    condition_number: float
    constants: dict[str, list] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def given_values_si(self) -> dict[str, float]:
        return {g["symbol"]: g["value_si"] for g in self.givens}


def env_config() -> dict:
    return load_config("env")


def generator_version(cfg: dict | None = None) -> str:
    gen = (cfg or env_config())["generator"]
    return f"{gen['version']}+{stable_hash(gen)}"


def problem_rng(version: str, family: str, depth: int, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{version}|{family}|{depth}|{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "little"))


def train_cells(cfg: dict | None = None, families: tuple[str, ...] = FAMILY_NAMES) -> list[tuple[str, int]]:
    s = (cfg or env_config())["splits"]
    return [
        (f, d)
        for f in families
        if f not in s["heldout_families"]
        for d in s["depths"]
        if d not in s["heldout_depths"]
    ]


def check_split(family: str, depth: int, split: str, cfg: dict) -> None:
    s = cfg["splits"]
    if depth not in s["depths"]:
        raise ValueError(f"depth {depth} outside configured depths {s['depths']}")
    if split == "train":
        if family in s["heldout_families"] or depth in s["heldout_depths"]:
            raise HeldOutError(f"({family}, depth {depth}) is held out from training")
    elif split != "eval":
        raise ValueError(f"unknown split {split!r}")


def generate_problem(family: str, depth: int, seed: int, split: str = "train", cfg: dict | None = None) -> Problem:
    cfg = cfg or env_config()
    check_split(family, depth, split, cfg)
    gen = cfg["generator"]
    version = generator_version(cfg)
    no_steps_text = render.no_steps_template(gen["version"])
    fam = get_family(family)
    rng = problem_rng(version, family, depth, seed)

    for attempt in range(gen["max_attempts"]):
        if attempt % gen["worlds_per_chain"] == 0:
            chain = fam.sample_chain(depth, rng)
        world = fam.sample_world(rng, gen["max_world_tries"])
        shown = [render.show_value(fam.quantities[g], world[g], rng, gen["display"]) for g in chain.givens]
        givens_si = {s.symbol: s.value_si for s in shown}
        outputs = fam.evaluate_chain_with_units(chain, givens_si)
        if not all(math.isfinite(v) and v > 0 for v in outputs.values()):
            continue
        kappa = fam.condition_number(chain, givens_si, gen["condition_rel_step"])
        if kappa <= gen["max_condition_number"]:
            break
    else:
        raise RuntimeError(f"no acceptable {family} depth-{depth} problem for seed {seed}")

    distractors = _sample_distractors(fam, rng, gen)
    target_q = fam.quantities[chain.target]
    target_si = outputs[chain.target]
    target_unit = render.choose_unit(target_q, target_si, rng, tuple(gen["display"]["magnitude_range"]))
    text = render.render_text(
        fam, target_q, target_unit, shown + distractors, chain.intermediates, rng, gen["display"], no_steps_text
    )
    return Problem(
        text=text,
        target_symbol=chain.target,
        target_desc=target_q.desc,
        target_value_si=target_si,
        target_si_unit=target_q.si_unit,
        target_unit=target_unit,
        target_value=from_si(target_si, target_q.si_unit, target_unit),
        intermediates={x: outputs[x] for x in chain.intermediates},
        intermediate_si_units={x: fam.quantities[x].si_unit for x in chain.intermediates},
        givens=[s.to_dict() for s in shown],
        distractors=[s.to_dict() for s in distractors],
        chain=[list(step) for step in chain.steps],
        family=family,
        depth=depth,
        seed=seed,
        generator_version=version,
        condition_number=kappa,
        constants={k: [v, u] for k, (v, u) in fam.constants.items()},
    )


def _sample_distractors(fam, rng: np.random.Generator, gen: dict) -> list[render.ShownValue]:
    lo, hi = gen["distractors"]
    pool = sorted(fam.distractors)
    n = min(int(rng.integers(lo, hi + 1)), len(pool))
    picked = [pool[i] for i in sorted(rng.choice(len(pool), size=n, replace=False))]
    return [
        render.show_value(fam.distractors[p], log_uniform(rng, *fam.distractors[p].range_si), rng, gen["display"])
        for p in picked
    ]


def chain_of(problem: Problem | dict) -> Chain:
    """Rebuild the Chain object from a problem (or its dict)."""
    p = problem if isinstance(problem, dict) else problem.to_dict()
    return Chain(steps=tuple((f, o) for f, o in p["chain"]), givens=tuple(sorted(g["symbol"] for g in p["givens"])))


def problem_from_dict(d: dict) -> Problem:
    return Problem(**d)


def worked_solution(problem: Problem) -> str:
    """Step-by-step ground truth in SI, for humans checking the physics."""
    fam = get_family(problem.family)
    env = {**problem.given_values_si, **fam.si_constants}
    units = {q.symbol: q.si_unit for q in fam.quantities.values()}
    units.update({k: u for k, (_, u) in fam.constants.items()})
    fmt = lambda sym: f"{env[sym]:.6g} {units[sym]}".strip()
    lines = ["Givens in SI: " + "; ".join(f"{g} = {fmt(g)}" for g in problem.given_values_si)]
    if fam.constants:
        lines.append("Constants: " + "; ".join(f"{k} = {fmt(k)}" for k in fam.constants))
    for i, (name, out) in enumerate(problem.chain, start=1):
        f = fam.formula(name)
        solved = f.solved[out]
        env[out] = problem.intermediates.get(out, problem.target_value_si)
        args = ", ".join(f"{a} = {fmt(a)}" for a in solved.args)
        lines.append(f"Step {i} [{name}: {f.relation}]  {out} = {solved.expr}  with {args}  =>  {out} = {fmt(out)}")
    unit = problem.target_unit or "(dimensionless)"
    lines.append(
        f"Answer: {problem.target_symbol} = {problem.target_value_si:.6g} {problem.target_si_unit} (SI)"
        f" = {problem.target_value:.6g} {unit} (requested unit)"
    )
    return "\n".join(lines)

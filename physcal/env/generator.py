"""Problem generation: world -> chain -> displayed givens -> ground truth -> text.

`generate_problem(family, depth, seed)` is a pure function of (generator_version, family, depth, seed), plus the
noise level in generator 2.0.0's measurement-noise mode (D51). The noise level is not part of the main random
stream, so problems that differ only in their noise level share world, chain, units and phrasing.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from physcal.config import load_config, stable_hash
from physcal.env import noise as noise_mode
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
    # Generator >= 2.0.0 only (D51): noise level, true values of noisy givens, f(x_meas) and σ*. None keeps the
    # generator-1.x dict shape, so frozen 1.x eval sets and tapes are unchanged.
    noise: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.noise is None:
            del d["noise"]
        for key in ("givens", "distractors"):  # ShownValue.to_dict drops noise_rel for exact values
            d[key] = [{k: v for k, v in x.items() if k != "noise_rel" or v} for x in d[key]]
        return d

    @property
    def given_values_si(self) -> dict[str, float]:
        return {g["symbol"]: g["value_si"] for g in self.givens}


def env_config() -> dict:
    return load_config("env")


def generator_version(cfg: dict | None = None) -> str:
    gen = (cfg or env_config())["generator"]
    return f"{gen['version']}+{stable_hash(gen)}"


def problem_rng(version: str, family: str, depth: int, seed: int, stream: str = "") -> np.random.Generator:
    """The main stream (stream = "") or a named side stream ("noise", "mc"; generator >= 2.0.0)."""
    key = f"{version}|{family}|{depth}|{seed}" + (f"|{stream}" if stream else "")
    digest = hashlib.sha256(key.encode()).digest()
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


def generate_problem(family: str, depth: int, seed: int, split: str = "train", cfg: dict | None = None,
                     noise: float = 0.0) -> Problem:
    """`noise` is the relative standard deviation of every given's measurement error (generator 2.0.0's noise
    mode; one of `generator.noise.levels`). Noise 0 behaves exactly as before: exact givens at `display.sig_figs`
    and truth from the displayed values (D4)."""
    cfg = cfg or env_config()
    check_split(family, depth, split, cfg)
    gen = cfg["generator"]
    noise_cfg = gen.get("noise")
    if noise_cfg is None and noise != 0.0:
        raise ValueError(f"generator {gen['version']} has no noise mode")
    if noise_cfg is not None and noise not in noise_cfg["levels"]:
        raise ValueError(f"noise level {noise} not in {noise_cfg['levels']}")
    noisy = noise != 0.0
    version = generator_version(cfg)
    no_steps_text = render.no_steps_template(gen["version"])
    fam = get_family(family)
    rng = problem_rng(version, family, depth, seed)
    noise_rng = problem_rng(version, family, depth, seed, "noise") if noise_cfg is not None else None

    for attempt in range(gen["max_attempts"]):
        if attempt % gen["worlds_per_chain"] == 0:
            chain = fam.sample_chain(depth, rng)
        world = fam.sample_world(rng, gen["max_world_tries"])
        if noise_rng is not None:
            # drawn every attempt at every level (also 0), so the streams stay aligned across noise levels
            e = noise * noise_rng.standard_normal(len(chain.givens))
        if noisy:
            shown = [render.show_measured(fam.quantities[g], world[g], float(e[i]), noise, rng, gen["display"],
                                          noise_cfg["sig_figs"]) for i, g in enumerate(chain.givens)]
            truth_givens = {g: world[g] for g in chain.givens}
        else:
            shown = [render.show_value(fam.quantities[g], world[g], rng, gen["display"]) for g in chain.givens]
            truth_givens = {s.symbol: s.value_si for s in shown}
        outputs = fam.evaluate_chain_with_units(chain, truth_givens)
        if not all(math.isfinite(v) and v > 0 for v in outputs.values()):
            continue
        kappa = fam.condition_number(chain, truth_givens, gen["condition_rel_step"])
        if kappa > gen["max_condition_number"]:
            continue
        if not noisy:
            break
        measured = {s.symbol: s.value_si for s in shown}
        outputs_meas = fam.evaluate_chain(chain, measured)
        if not all(math.isfinite(v) and v > 0 for v in outputs_meas.values()):
            continue
        z = problem_rng(version, family, depth, seed, f"mc|{attempt}").standard_normal(
            (len(chain.givens), noise_cfg["n_mc"]))
        log_draws = noise_mode.predictive(fam, chain, measured, {g: noise for g in chain.givens}, z)
        if log_draws is not None:
            break
    else:
        raise RuntimeError(f"no acceptable {family} depth-{depth} problem for seed {seed} (noise {noise})")

    distractors = _sample_distractors(fam, rng, gen, noise, noise_rng)
    target_q = fam.quantities[chain.target]
    target_si = outputs[chain.target]
    target_unit = render.choose_unit(target_q, target_si, rng, tuple(gen["display"]["magnitude_range"]))
    text = render.render_text(
        fam, target_q, target_unit, shown + distractors, chain.intermediates, rng, gen["display"], no_steps_text,
        render.noise_note(gen["version"]) if noisy else None,
    )
    # Step credit uses what a correct solver computes from the displayed values (D51): for noisy problems the
    # intermediates come from the measured givens; the clean ones are kept in `noise`.
    step_values = outputs_meas if noisy else outputs
    noise_info = None
    if noise_cfg is not None:
        noise_info = {"level": noise, "sig_figs": noise_cfg["sig_figs"] if noisy else gen["display"]["sig_figs"],
                      "point_estimate_si": step_values[chain.target]}
        if noisy:
            noise_info.update(
                givens={g: {"rel": noise, "true_si": world[g], "e": float(e[i])} for i, g in enumerate(chain.givens)},
                intermediates_true_si={x: outputs[x] for x in chain.intermediates},
                sigma_star_log_delta=noise_mode.delta_sigma_log(
                    noise_mode.log_sensitivities(fam, chain, measured, gen["condition_rel_step"]),
                    {g: noise for g in chain.givens}),
                **noise_mode.summarize(log_draws, step_values[chain.target], target_si))
        else:
            noise_info.update(sigma_star_log=0.0, sigma_star_log_sd=0.0, sigma_star_log_delta=0.0,
                              sigma_star_si_sd=0.0, pit_truth=None, mc_draws=0)
    return Problem(
        text=text,
        target_symbol=chain.target,
        target_desc=target_q.desc,
        target_value_si=target_si,
        target_si_unit=target_q.si_unit,
        target_unit=target_unit,
        target_value=from_si(target_si, target_q.si_unit, target_unit),
        intermediates={x: step_values[x] for x in chain.intermediates},
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
        noise=noise_info,
    )


def regenerate(problem: dict, cfg: dict, split: str = "eval") -> Problem:
    """The problem a logged dict came from (tape replay), including its noise level."""
    level = (problem.get("noise") or {}).get("level", 0.0)
    return generate_problem(problem["family"], problem["depth"], problem["seed"], split=split, cfg=cfg, noise=level)


def _sample_distractors(fam, rng: np.random.Generator, gen: dict, noise: float = 0.0,
                        noise_rng: np.random.Generator | None = None) -> list[render.ShownValue]:
    """Irrelevant quantities; in noise mode they are shown as measurements too (same level), so the "± p%"
    annotation does not tell relevant from irrelevant values."""
    lo, hi = gen["distractors"]
    pool = sorted(fam.distractors)
    n = min(int(rng.integers(lo, hi + 1)), len(pool))
    picked = [pool[i] for i in sorted(rng.choice(len(pool), size=n, replace=False))]
    out = []
    for p in picked:
        value = log_uniform(rng, *fam.distractors[p].range_si)
        if noise:
            e = noise * float(noise_rng.standard_normal())
            out.append(render.show_measured(fam.distractors[p], value, e, noise, rng, gen["display"],
                                            gen["noise"]["sig_figs"]))
        else:
            out.append(render.show_value(fam.distractors[p], value, rng, gen["display"]))
    return out


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

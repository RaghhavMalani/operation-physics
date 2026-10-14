"""D50/A6: counterfactual sensitivity pairs (propagation vs lookup), eval only.

A pair is two problems that share family, depth, chain (template), phrasing, display units, distractors, noise
level and standardized measurement errors z, built on two different worlds whose propagated σ* differ by at least
`min_ratio` (delta method; the stored σ* is then the MC value, D50/A2). A model that propagates uncertainty states
a larger σ on the high-σ* member; a lookup from surface features (template, depth, noise level) cannot tell the
two apart.

    nice uv run python -m physcal.env.pairs export        # frozen set: data/eval_sets/<version>/a6_pairs.jsonl

Pair members get problem seed 2·pair_seed + member_index (0 = lo, 1 = hi); `noise.pair` stores the pair seed, the
member and the construction config, so `generator.regenerate` replays them exactly.
"""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from physcal.config import REPO_ROOT, load_config, stable_hash
from physcal.env import noise as noise_mode
from physcal.env import render
from physcal.env.formulas import get_family
from physcal.env.generator import (Problem, _sample_distractors, check_split, generator_version, noise_record,
                                   problem_rng)
from physcal.env.units import from_si

MEMBERS = ("lo", "hi")


def pair_version(construction: dict) -> str:
    return f"{construction['version']}+{stable_hash(construction)}"


def generate_pair(family: str, depth: int, pair_seed: int, noise: float, env_cfg: dict,
                  construction: dict) -> tuple[Problem, Problem] | None:
    """The (lo, hi) pair for this seed, or None if no chain within `max_chains` qualifies."""
    check_split(family, depth, "eval", env_cfg)
    gen = env_cfg["generator"]
    ncfg = gen.get("noise")
    if ncfg is None or noise == 0.0 or noise not in ncfg["levels"]:
        raise ValueError(f"A6 pairs need a noisy level of generator >= 2.0.0, got {noise}")
    version = generator_version(env_cfg)
    pv = pair_version(construction)
    stream = lambda name: problem_rng(version, family, depth, pair_seed, f"pair|{pv}|{name}")
    fam = get_family(family)
    rng_chain, rng_world = stream("chain"), stream("world")
    for t in range(construction["max_chains"]):
        chain = fam.sample_chain(depth, rng_chain)
        levels = {g: noise for g in chain.givens}
        cands = []
        for _ in range(construction["candidate_worlds"]):
            world = fam.sample_world(rng_world, gen["max_world_tries"])
            true = {g: world[g] for g in chain.givens}
            out = fam.evaluate_chain(chain, true)
            if not all(math.isfinite(v) and v > 0 for v in out.values()):
                continue
            kappa = fam.condition_number(chain, true, gen["condition_rel_step"])
            if kappa > gen["max_condition_number"]:
                continue
            d = noise_mode.delta_sigma_log(noise_mode.log_sensitivities(fam, chain, true, gen["condition_rel_step"]),
                                           levels)
            cands.append((d, world, kappa))
        if len(cands) < 2:
            continue
        lo, hi = min(cands, key=lambda c: c[0]), max(cands, key=lambda c: c[0])
        if hi[0] < construction["min_ratio"] * lo[0]:
            continue
        z = stream(f"z|{t}").standard_normal(len(chain.givens))
        zmc = stream(f"mc|{t}").standard_normal((len(chain.givens), ncfg["n_mc"]))
        rng_units = stream(f"units|{t}")
        units = {g: render.choose_unit(fam.quantities[g], lo[1][g], rng_units, tuple(gen["display"]["magnitude_range"]))
                 for g in chain.givens}
        meta = {"pair_seed": pair_seed, "pair_version": pv, "construction": construction, "chain_try": t,
                "delta_ratio": hi[0] / lo[0]}
        members: list[Problem] = []
        for idx, (d, world, kappa) in enumerate((lo, hi)):
            target_unit = members[0].target_unit if members else None  # hi reuses lo's target unit
            p = _member(fam, chain, world, kappa, noise, z, zmc, units, target_unit, env_cfg, version, family, depth,
                        pair_seed, idx, meta, stream, t)
            if p is None:
                break
            members.append(p)
        if len(members) == 2:
            return members[0], members[1]
    return None


def _member(fam, chain, world, kappa, noise, z, zmc, units, target_unit, env_cfg, version, family, depth, pair_seed,
            idx, meta, stream, t) -> Problem | None:
    gen = env_cfg["generator"]
    e = noise * z
    shown = [render.measured_in_unit(fam.quantities[g], world[g], float(e[i]), noise, units[g], gen["display"],
                                     gen["noise"]["sig_figs"]) for i, g in enumerate(chain.givens)]
    truth_givens = {g: world[g] for g in chain.givens}
    outputs = fam.evaluate_chain_with_units(chain, truth_givens)
    measured = {s.symbol: s.value_si for s in shown}
    outputs_meas = fam.evaluate_chain(chain, measured)
    if not all(math.isfinite(v) and v > 0 for v in list(outputs.values()) + list(outputs_meas.values())):
        return None
    log_draws = noise_mode.predictive(fam, chain, measured, {g: noise for g in chain.givens}, zmc)
    if log_draws is None:
        return None
    # identical distractors and phrasing for both members: fresh streams with the same seeds
    distractors = _sample_distractors(fam, stream(f"distractors|{t}"), gen, noise, stream(f"distractor-noise|{t}"))
    target_q = fam.quantities[chain.target]
    target_si = outputs[chain.target]
    if target_unit is None:
        target_unit = render.choose_unit(target_q, target_si, stream(f"target-unit|{t}"),
                                         tuple(gen["display"]["magnitude_range"]))
    text = render.render_text(fam, target_q, target_unit, shown + distractors, chain.intermediates,
                              stream(f"render|{t}"), gen["display"], render.no_steps_template(gen["version"]),
                              render.noise_note(gen["version"]))
    info = noise_record(fam, chain, gen, noise, world, e, measured, outputs, outputs_meas, log_draws)
    info["pair"] = {**meta, "member": MEMBERS[idx], "member_index": idx}
    return Problem(
        text=text, target_symbol=chain.target, target_desc=target_q.desc, target_value_si=target_si,
        target_si_unit=target_q.si_unit, target_unit=target_unit,
        target_value=from_si(target_si, target_q.si_unit, target_unit),
        intermediates={x: outputs_meas[x] for x in chain.intermediates},
        intermediate_si_units={x: fam.quantities[x].si_unit for x in chain.intermediates},
        givens=[s.to_dict() for s in shown], distractors=[s.to_dict() for s in distractors],
        chain=[list(step) for step in chain.steps], family=family, depth=depth, seed=2 * pair_seed + idx,
        generator_version=version, condition_number=kappa,
        constants={k: [v, u] for k, (v, u) in fam.constants.items()}, noise=info)


# ------------------------------------------------------------------------------------------- export
def _attempt(args: tuple) -> tuple:
    cell, i, family, depth, level, pair_seed = args
    pcfg = load_config("pairs")
    pair = generate_pair(family, depth, pair_seed, level, load_config(pcfg["env_config"]), pcfg["construction"])
    return cell, i, None if pair is None else [p.to_dict() for p in pair]


def export() -> dict:
    pcfg = load_config("pairs")
    ex, env = pcfg["export"], load_config(pcfg["env_config"])
    cells = [(d, lv) for d in ex["depths"] for lv in ex["levels"]]
    jobs = [(c, i, ex["families"][i % len(ex["families"])], d, lv,
             ex["seed_offset"] + c * ex["cell_seed_stride"] + i)
            for c, (d, lv) in enumerate(cells) for i in range(ex["max_seeds_per_cell"])]
    with ProcessPoolExecutor(ex["n_jobs"]) as pool:
        results = sorted(pool.map(_attempt, jobs, chunksize=8), key=lambda r: (r[0], r[1]))
    pairs, report = [], {}
    for c, (d, lv) in enumerate(cells):
        got = [r for r in results if r[0] == c and r[2] is not None][: ex["pairs_per_cell"]]
        tried = max((r[1] for r in got), default=-1) + 1 if len(got) == ex["pairs_per_cell"] else ex["max_seeds_per_cell"]
        failed = sum(1 for r in results if r[0] == c and r[1] < tried and r[2] is None)
        pairs += [r[2] for r in got]
        ratios = [p[1]["noise"]["sigma_star_log"] / p[0]["noise"]["sigma_star_log"] for _, _, p in got]
        report[f"d{d}/s{lv:g}"] = {"pairs": len(got), "seeds_tried": tried, "seeds_without_pair": failed,
                                   "families": {f: sum(p[0]["family"] == f for _, _, p in got) for f in ex["families"]},
                                   "mc_ratio_median": float(np.median(ratios)) if ratios else None,
                                   "mc_ratio_min": float(np.min(ratios)) if ratios else None}
    version = generator_version(env)
    out = REPO_ROOT / "data" / "eval_sets" / version
    out.mkdir(parents=True, exist_ok=True)
    blob = "".join(json.dumps(p, sort_keys=True) + "\n" for pair in pairs for p in pair)
    (out / "a6_pairs.jsonl").write_text(blob)
    manifest = {"generator_version": version, "pair_version": pair_version(pcfg["construction"]),
                "construction": pcfg["construction"], "export": ex, "n_pairs": len(pairs), "n_problems": 2 * len(pairs),
                "sha256": hashlib.sha256(blob.encode()).hexdigest(), "cells": report}
    (out / "a6_pairs_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    import sys

    if sys.argv[1:] != ["export"]:
        raise SystemExit("usage: python -m physcal.env.pairs export")
    from physcal.env import pairs as _pairs  # through the package, so worker processes import the same module

    print(json.dumps(_pairs.export(), indent=2))

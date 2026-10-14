"""D50/A6 counterfactual sensitivity pairs and D59/A6b invariance pairs (propagation vs lookup), eval only.

A pair is two problems that share family, depth, chain (template), phrasing, display units, distractors, noise
level and standardized measurement errors z, built on two different worlds whose propagated σ* differ by at least
`min_ratio` (delta method; the stored σ* is then the MC value, D50/A2). A model that propagates uncertainty states
a larger σ on the high-σ* member; a lookup from surface features (template, depth, noise level) cannot tell the
two apart.

A6b (`kind: invariance` in the construction) uses power-law templates only, where the relative σ* cannot depend
on the values. Its two worlds differ in the target's magnitude (≥ `min_target_ratio`), and each pair is checked by
MC to have the same σ*.

    nice uv run python -m physcal.env.pairs export [a6|a6b]  # data/eval_sets/<version>/a6_pairs.jsonl / a6b_pairs.jsonl

Pair members get problem seed 2·pair_seed + member_index (0 = lo, 1 = hi); `noise.pair` stores the pair seed, the
member and the construction config, so `generator.regenerate` replays them exactly.
"""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

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
    if construction.get("kind") == "invariance":
        return _invariance_pair(family, depth, pair_seed, noise, env_cfg, construction)
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


@lru_cache(maxsize=None)
def _power_law_chains(family: str, depth: int, tol: float, n_ref: int, ref_seed: int, rel_step: float,
                      max_world_tries: int) -> tuple:
    """Chains of this cell whose log-sensitivities are the same at `n_ref` reference worlds (power laws)."""
    fam = get_family(family)
    rng = np.random.default_rng(ref_seed)
    worlds = [fam.sample_world(rng, max_world_tries) for _ in range(n_ref)]
    out = []
    for chain in (c for cs in fam.chains(depth).values() for c in cs):
        sens = [noise_mode.log_sensitivities(fam, chain, {g: w[g] for g in chain.givens}, rel_step) for w in worlds]
        if all(np.isfinite(list(x.values())).all() for x in sens) and all(
                abs(x[g] - sens[0][g]) <= tol for x in sens[1:] for g in chain.givens):
            out.append(chain)
    return tuple(out)


def power_law_chains(family: str, depth: int, construction: dict, gen: dict) -> tuple:
    return _power_law_chains(family, depth, construction["power_law_tol"], construction["power_law_reference_worlds"],
                             construction["power_law_reference_seed"], gen["condition_rel_step"], gen["max_world_tries"])


def _invariance_pair(family: str, depth: int, pair_seed: int, noise: float, env_cfg: dict,
                     construction: dict) -> tuple[Problem, Problem] | None:
    """A6b: a power-law template on two worlds whose targets differ by >= min_target_ratio; σ* checked equal by MC."""
    check_split(family, depth, "eval", env_cfg)
    gen = env_cfg["generator"]
    ncfg = gen.get("noise")
    if ncfg is None or noise == 0.0 or noise not in ncfg["levels"]:
        raise ValueError(f"A6b pairs need a noisy level of generator >= 2.0.0, got {noise}")
    eligible = power_law_chains(family, depth, construction, gen)
    if not eligible:
        return None
    version = generator_version(env_cfg)
    pv = pair_version(construction)
    stream = lambda name: problem_rng(version, family, depth, pair_seed, f"pair|{pv}|{name}")
    fam = get_family(family)
    rng_chain, rng_world = stream("chain"), stream("world")
    mc_rejected = 0
    for t in range(construction["max_chains"]):
        chain = eligible[int(rng_chain.integers(len(eligible)))]
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
            cands.append((out[chain.target], world, kappa))
        if len(cands) < 2:
            continue
        lo, hi = min(cands, key=lambda c: c[0]), max(cands, key=lambda c: c[0])
        if hi[0] < construction["min_target_ratio"] * lo[0]:
            continue
        z = stream(f"z|{t}").standard_normal(len(chain.givens))
        zmc = stream(f"mc|{t}").standard_normal((len(chain.givens), ncfg["n_mc"]))
        rng_units = stream(f"units|{t}")
        units = {g: render.choose_unit(fam.quantities[g], lo[1][g], rng_units, tuple(gen["display"]["magnitude_range"]))
                 for g in chain.givens}
        meta = {"pair_seed": pair_seed, "pair_version": pv, "construction": construction, "chain_try": t,
                "kind": "invariance", "target_ratio": hi[0] / lo[0], "mc_rejected_before": mc_rejected}
        members: list[Problem] = []
        for idx, (_, world, kappa) in enumerate((lo, hi)):
            target_unit = members[0].target_unit if members else None
            p = _member(fam, chain, world, kappa, noise, z, zmc, units, target_unit, env_cfg, version, family, depth,
                        pair_seed, idx, meta, stream, t)
            if p is None:
                break
            members.append(p)
        if len(members) < 2:
            continue
        log_ratio = math.log(members[1].noise["sigma_star_log"] / members[0].noise["sigma_star_log"])
        if abs(log_ratio) > construction["max_log_sigma_ratio"]:
            mc_rejected += 1
            continue
        for m in members:
            m.noise["pair"]["log_sigma_ratio_mc"] = log_ratio
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
SETS = {"a6": ("construction", "export", "a6_pairs"),
        "a6b": ("invariance_construction", "invariance_export", "a6b_pairs")}


def _attempt(args: tuple) -> tuple:
    cell, i, family, depth, level, pair_seed, which = args
    pcfg = load_config("pairs")
    pair = generate_pair(family, depth, pair_seed, level, load_config(pcfg["env_config"]), pcfg[SETS[which][0]])
    return cell, i, None if pair is None else [p.to_dict() for p in pair]


def _families(which: str, depth: int, pcfg: dict, env: dict) -> list[str]:
    """A6: every configured family. A6b: only families with power-law templates at this depth."""
    fams = pcfg[SETS[which][1]]["families"]
    if which == "a6":
        return fams
    return [f for f in fams if power_law_chains(f, depth, pcfg[SETS[which][0]], env["generator"])]


def export(which: str = "a6") -> dict:
    pcfg = load_config("pairs")
    con_key, ex_key, name = SETS[which]
    ex, env = pcfg[ex_key], load_config(pcfg["env_config"])
    cells = [(d, lv) for d in ex["depths"] for lv in ex["levels"]]
    fams = {d: _families(which, d, pcfg, env) for d in ex["depths"]}
    jobs = [(c, i, fams[d][i % len(fams[d])], d, lv, ex["seed_offset"] + c * ex["cell_seed_stride"] + i, which)
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
        if which == "a6b":
            tr = [p[0]["noise"]["pair"]["target_ratio"] for _, _, p in got]
            lr = [abs(p[0]["noise"]["pair"]["log_sigma_ratio_mc"]) for _, _, p in got]
            report[f"d{d}/s{lv:g}"].update(
                target_ratio_median=float(np.median(tr)) if tr else None,
                max_abs_log_sigma_ratio_mc=float(np.max(lr)) if lr else None,
                mc_rejected=sum(p[0]["noise"]["pair"]["mc_rejected_before"] for _, _, p in got),
                eligible_families=fams[d])
    version = generator_version(env)
    out = REPO_ROOT / "data" / "eval_sets" / version
    out.mkdir(parents=True, exist_ok=True)
    blob = "".join(json.dumps(p, sort_keys=True) + "\n" for pair in pairs for p in pair)
    (out / f"{name}.jsonl").write_text(blob)
    manifest = {"generator_version": version, "pair_version": pair_version(pcfg[con_key]),
                "construction": pcfg[con_key], "export": ex, "n_pairs": len(pairs), "n_problems": 2 * len(pairs),
                "sha256": hashlib.sha256(blob.encode()).hexdigest(), "cells": report}
    (out / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    import sys

    if not sys.argv[1:] or sys.argv[1] != "export" or (sys.argv[2:] and sys.argv[2] not in SETS):
        raise SystemExit("usage: python -m physcal.env.pairs export [a6|a6b]")
    from physcal.env import pairs as _pairs  # through the package, so worker processes import the same module

    print(json.dumps(_pairs.export(sys.argv[2] if sys.argv[2:] else "a6"), indent=2))

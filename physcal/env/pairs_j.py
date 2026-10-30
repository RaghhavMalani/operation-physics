"""R5-6 part J (DECISIONS.md D147): sensitivity and invariance pairs whose Jacobian lies outside the training range.
Eval only; same formulas, families and generator (2.0.0) as A6c / A6b.

A J world (D147 §2): the chain occurs in the R4-1 training problems; at least one given's log-sensitivity is outside
its training range for that chain by `margin`; every displayed given and every value the chain computes lies inside
that quantity's training range; finite, positive, κ ≤ max. The support is `jacobian_support`'s file, named with its
SHA-256 in the construction, so replay uses the same one.

- `j_allocation` (D147 §3): A6c's construction (D95) on a J world. Both members share the displayed values, so the
  J test is applied once, at the displayed values.
- `j_invariance` (D147 §4): a J world and a copy scaled along a dimensional direction that leaves every constant of
  the family unchanged (each given × λ^(its exponents)). Dimensional homogeneity keeps every log-sensitivity, so the
  relative σ*, identical, while the target changes by ≥ `min_target_ratio`. The same z, units, distractors and
  phrasing for both members; σ* checked equal by MC, as A6b.

Each member stores `noise.pair.j`: D147 §10's distances (jacobian_support.distances) and, per member, the clamped
ln(σ_δ / σ_δ clamped) at its own levels.
"""

from __future__ import annotations

import math

import numpy as np

from physcal.env import jacobian_support as JS
from physcal.env import noise as noise_mode
from physcal.env import render
from physcal.env.formulas import get_family
from physcal.env.generator import Problem, check_split, generator_version, problem_rng
from physcal.env.pairs import _member, pair_version
from physcal.env.units import dimensionality

DIMS = ("[length]", "[mass]", "[time]")  # the order of a scaling direction's exponents in the construction


def dim_vector(unit: str) -> np.ndarray:
    d = dict(dimensionality(unit))
    extra = set(d) - set(DIMS)
    if extra:
        raise ValueError(f"{unit}: dimensions {sorted(extra)} are not in {DIMS}")
    return np.array([d.get(k, 0) for k in DIMS], dtype=float)


def generate(family: str, depth: int, pair_seed: int, noise: float, env_cfg: dict,
             construction: dict) -> tuple[Problem, Problem] | None:
    if construction["kind"] == "j_allocation":
        return _allocation(family, depth, pair_seed, noise, env_cfg, construction)
    if construction["kind"] == "j_invariance":
        return _invariance(family, depth, pair_seed, noise, env_cfg, construction)
    raise ValueError(f"not a J construction: {construction.get('kind')}")


class _Ctx:
    """What both constructions share: the family, the support, the streams and the generator's validity checks."""

    def __init__(self, family: str, depth: int, pair_seed: int, env_cfg: dict, construction: dict):
        check_split(family, depth, "eval", env_cfg)
        self.family, self.gen = family, env_cfg["generator"]
        self.fam = get_family(family)
        self.sup = JS.load(construction["jacobian"])
        self.margin = construction["jacobian"]["margin"]
        self.step, self.max_kappa = self.gen["condition_rel_step"], self.gen["max_condition_number"]
        version = generator_version(env_cfg)
        pv = pair_version(construction)
        self.version, self.pv = version, pv
        self.stream = lambda name: problem_rng(version, family, depth, pair_seed, f"pair|{pv}|{name}")

    def outputs(self, chain, givens: dict) -> dict | None:
        out = self.fam.evaluate_chain(chain, givens)
        return out if all(math.isfinite(v) and v > 0 for v in out.values()) else None

    def valid(self, chain, givens: dict) -> float | None:
        """κ if the chain is finite, positive, conditioned and inside the training magnitudes at these givens."""
        out = self.outputs(chain, givens)
        if out is None or not JS.in_magnitudes(self.sup, self.family, {**givens, **out}):
            return None
        kappa = self.fam.condition_number(chain, givens, self.step)
        return kappa if kappa <= self.max_kappa else None

    def sens(self, chain, givens: dict) -> dict:
        return noise_mode.log_sensitivities(self.fam, chain, givens, self.step)

    def j_world(self, chain, rng_world, n: int) -> tuple[dict, float] | None:
        """The first of `n` world draws that is a J world (D147 §2) at its true givens."""
        for _ in range(n):
            world = self.fam.sample_world(rng_world, self.gen["max_world_tries"])
            true = {g: world[g] for g in chain.givens}
            kappa = self.valid(chain, true)
            if kappa is not None and JS.exceedance(self.sup, self.family, chain, self.sens(chain, true))[0] >= self.margin:
                return world, kappa
        return None

    def shown_ok(self, chain, members: list[Problem]) -> dict | None:
        """D147 §2 at the displayed values of every member: the J test and the magnitudes. Returns the first member's
        distances (D147 §10), or None."""
        dist = None
        for m in members:
            shown = {g["symbol"]: g["value_si"] for g in m.givens}
            out = self.outputs(chain, shown)
            if out is None or not JS.in_magnitudes(self.sup, self.family, {**shown, **out}):
                return None
            s = self.sens(chain, shown)
            if JS.exceedance(self.sup, self.family, chain, s)[0] < self.margin:
                return None
            dist = dist or JS.distances(self.sup, self.family, chain, s, self.margin)
        return dist


def _store(members: list[Problem], dist: dict, levels: list[dict], extra: dict | None = None) -> None:
    for m, lv in zip(members, levels):
        m.noise["pair"]["j"] = {**dist, "clamped_log_ratio": JS.clamped_sigma_log_ratio(dist, lv), **(extra or {})}


def _allocation(family, depth, pair_seed, noise, env_cfg, construction):
    """D147 §3: A6c's allocation pair (pairs._allocation_pair) on a J world."""
    cx = _Ctx(family, depth, pair_seed, env_cfg, construction)
    ncfg = cx.gen.get("noise")
    lo_level, hi_level = construction["levels"]
    if ncfg is None or not 0.0 < lo_level < hi_level or noise != hi_level or \
            not {lo_level, hi_level} <= set(ncfg["levels"]):
        raise ValueError(f"J allocation pairs need noisy levels l < h = noise, got {construction['levels']} / {noise}")
    fam, stream = cx.fam, cx.stream
    rng_chain, rng_world = stream("chain"), stream("world")
    for t in range(construction["max_chains"]):
        chain = fam.sample_chain(depth, rng_chain)
        if len(chain.givens) < 2 or not JS.trained(cx.sup, family, chain):
            continue
        found = cx.j_world(chain, rng_world, construction["candidate_worlds"])
        if found is None:
            continue
        world, kappa = found
        sens = cx.sens(chain, {g: world[g] for g in chain.givens})
        alloc = lambda j: {g: (hi_level if g == j else lo_level) for g in chain.givens}
        ds = [(noise_mode.delta_sigma_log(sens, alloc(g)), i, g) for i, g in enumerate(chain.givens)]
        lo = min(ds)  # ties: the earlier given (as A6c)
        hi = max(ds, key=lambda x: (x[0], -x[1]))
        if hi[0] < construction["min_ratio"] * lo[0]:
            continue
        z = stream(f"z|{t}").standard_normal(len(chain.givens))
        zmc = stream(f"mc|{t}").standard_normal((len(chain.givens), ncfg["n_mc"]))
        rng_units = stream(f"units|{t}")
        units = {g: render.choose_unit(fam.quantities[g], world[g], rng_units, tuple(cx.gen["display"]["magnitude_range"]))
                 for g in chain.givens}
        lv_lo, lv_hi = alloc(lo[2]), alloc(hi[2])
        measured = {g: world[g] * (1.0 + lv_lo[g] * float(z[i])) for i, g in enumerate(chain.givens)}
        world_hi = {**world, **{g: measured[g] / (1.0 + lv_hi[g] * float(z[i])) for i, g in enumerate(chain.givens)}}
        kappa_hi = cx.valid(chain, {g: world_hi[g] for g in chain.givens})
        if kappa_hi is None:
            continue
        meta = {"pair_seed": pair_seed, "pair_version": cx.pv, "construction": construction, "chain_try": t,
                "kind": "j_allocation", "delta_ratio": hi[0] / lo[0]}
        members: list[Problem] = []
        for idx, (w, k, lv, g_hi) in enumerate(((world, kappa, lv_lo, lo[2]), (world_hi, kappa_hi, lv_hi, hi[2]))):
            target_unit = members[0].target_unit if members else None
            p = _member(fam, chain, w, k, noise, z, zmc, units, target_unit, env_cfg, cx.version, family, depth,
                        pair_seed, idx, {**meta, "high_given": g_hi}, stream, t, levels=lv, distractor_level=lo_level)
            if p is None:
                break
            members.append(p)
        if len(members) < 2:
            continue
        shown = lambda m: [(g["display"], g["unit"]) for g in m.givens]
        if shown(members[0]) != shown(members[1]):  # identical displayed values (D95)
            continue
        dist = cx.shown_ok(chain, members)
        if dist is None:
            continue
        _store(members, dist, [lv_lo, lv_hi])
        return members[0], members[1]
    return None


def _invariance(family, depth, pair_seed, noise, env_cfg, construction):
    """D147 §4: a J world and its dimensionally scaled copy; lo / hi by target value, as A6b."""
    cx = _Ctx(family, depth, pair_seed, env_cfg, construction)
    ncfg = cx.gen.get("noise")
    if ncfg is None or noise == 0.0 or noise not in ncfg["levels"]:
        raise ValueError(f"J invariance pairs need a noisy level of generator >= 2.0.0, got {noise}")
    fam, stream = cx.fam, cx.stream
    directions = [np.array(d, dtype=float) for d in construction["directions"][family]]
    tdim_of = lambda q: dim_vector(fam.quantities[q].si_unit)
    rng_chain, rng_world = stream("chain"), stream("world")
    mc_rejected = 0
    for t in range(construction["max_chains"]):
        chain = fam.sample_chain(depth, rng_chain)
        if not JS.trained(cx.sup, family, chain):
            continue
        found = cx.j_world(chain, rng_world, construction["candidate_worlds"])
        if found is None:
            continue
        world, kappa = found
        scaled = _scaled(cx, chain, world, directions, tdim_of, construction)
        if scaled is None:
            continue
        world2, kappa2, scale = scaled
        target = lambda w: cx.outputs(chain, {g: w[g] for g in chain.givens})[chain.target]
        cands = sorted([(target(world), 0, world, kappa), (target(world2), 1, world2, kappa2)])
        (t_lo, _, w_lo, k_lo), (t_hi, _, w_hi, k_hi) = cands
        z = stream(f"z|{t}").standard_normal(len(chain.givens))
        zmc = stream(f"mc|{t}").standard_normal((len(chain.givens), ncfg["n_mc"]))
        rng_units = stream(f"units|{t}")
        units = {g: render.choose_unit(fam.quantities[g], w_lo[g], rng_units, tuple(cx.gen["display"]["magnitude_range"]))
                 for g in chain.givens}
        meta = {"pair_seed": pair_seed, "pair_version": cx.pv, "construction": construction, "chain_try": t,
                "kind": "j_invariance", "target_ratio": t_hi / t_lo, "mc_rejected_before": mc_rejected, "scale": scale}
        members: list[Problem] = []
        for idx, (w, k) in enumerate(((w_lo, k_lo), (w_hi, k_hi))):
            target_unit = members[0].target_unit if members else None
            p = _member(fam, chain, w, k, noise, z, zmc, units, target_unit, env_cfg, cx.version, family, depth,
                        pair_seed, idx, meta, stream, t)
            if p is None:
                break
            members.append(p)
        if len(members) < 2:
            continue
        dist = cx.shown_ok(chain, members)
        if dist is None:
            continue
        log_ratio = math.log(members[1].noise["sigma_star_log"] / members[0].noise["sigma_star_log"])
        if abs(log_ratio) > construction["max_log_sigma_ratio"]:
            mc_rejected += 1
            continue
        lv = {g: noise for g in chain.givens}
        s_hi = cx.sens(chain, {g["symbol"]: g["value_si"] for g in members[1].givens})
        _store(members, dist, [lv, lv], {"member_sensitivity_max_abs_diff":
                                         max(abs(s_hi[g] - dist["sensitivities"][g]) for g in chain.givens)})
        for m in members:
            m.noise["pair"]["log_sigma_ratio_mc"] = log_ratio
        return members[0], members[1]
    return None


def _scaled(cx: _Ctx, chain, world: dict, directions: list[np.ndarray], tdim_of, construction: dict):
    """The first (direction, factor), in the configured order, that changes the target by >= min_target_ratio and
    keeps the scaled chain valid and inside the training magnitudes."""
    k_t = [float(tdim_of(chain.target) @ d) for d in directions]
    need = math.log(construction["min_target_ratio"])
    for d, kt in zip(directions, k_t):
        if abs(kt) < 1e-12:
            continue
        for f in construction["scale_factors"]:
            if abs(kt * math.log(f)) < need - 1e-12:
                continue
            sc = {g: world[g] * f ** float(tdim_of(g) @ d) for g in chain.givens}
            kappa = cx.valid(chain, sc)
            if kappa is not None:
                return {**world, **sc}, kappa, {"direction": d.tolist(), "factor": f}
    return None

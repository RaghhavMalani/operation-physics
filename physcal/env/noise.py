"""Measurement-noise mode (generator 2.0.0, Phase 10 step 3; σ* as amended by D50/A2).

A noisy given is displayed as x_meas = x_true·(1 + e), e ~ N(0, s²). The predictive distribution of the clean
truth given the display draws x = x_meas / (1 + e'), e' ~ N(0, s²), for every noisy given (exact givens and
constants fixed) and pushes the draws through the chain. With a log-uniform prior on x_true this is the exact
posterior; the generator's own priors are only approximately log-uniform, which the Phase 10 gate checks.

σ* (primary, A2) is the CRPS-optimal Gaussian scale in log space centred at ln f(x_meas). Also kept: the SD of the
log draws (the original spec's σ*) and the first-order (delta-method) value.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

from physcal.env.graph import Chain, Family

_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
SCALE_SEARCH_DECADES = 2.0  # the CRPS-optimal scale is searched within 10^±2 of the draws' SD
SCALE_SEARCH_XATOL = 1e-9   # in ln(scale)


def chain_target(fam: Family, chain: Chain, givens: dict[str, np.ndarray | float]) -> np.ndarray:
    """The chain's target, vectorized over arrays of given values (plain SI floats, no unit checks)."""
    env = {**givens, **fam.si_constants}
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for name, out in chain.steps:
            env[out] = np.asarray(fam.formula(name).solved[out](env), dtype=float)
    return np.asarray(env[chain.target], dtype=float)


def log_sensitivities(fam: Family, chain: Chain, givens_si: dict[str, float], rel_step: float) -> dict[str, float]:
    """d ln(target) / d ln(given) for every given, by central differences."""
    out = {}
    for g in chain.givens:
        up, down = dict(givens_si), dict(givens_si)
        up[g] *= 1.0 + rel_step
        down[g] *= 1.0 - rel_step
        out[g] = float((np.log(chain_target(fam, chain, up)) - np.log(chain_target(fam, chain, down)))
                       / (math.log1p(rel_step) - math.log1p(-rel_step)))
    return out


def delta_sigma_log(sensitivities: dict[str, float], levels: dict[str, float]) -> float:
    """First-order σ of ln(target): sqrt(Σ (d ln f / d ln x_i)² s_i²)."""
    return math.sqrt(sum((sensitivities[g] * levels.get(g, 0.0)) ** 2 for g in sensitivities))


def mean_crps(scale: float, draws: np.ndarray, center: float) -> float:
    z = (draws - center) / scale
    return float(np.mean(scale * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * _INV_SQRT_2PI * np.exp(-0.5 * z * z) - _INV_SQRT_PI)))


def crps_optimal_scale(draws: np.ndarray, center: float) -> float:
    """argmin_s mean CRPS(N(center, s²), draw): convex in s, searched in ln s around the draws' RMS spread."""
    ref = math.sqrt(float(np.mean((draws - center) ** 2)))
    if ref == 0.0:
        return 0.0
    half = SCALE_SEARCH_DECADES * math.log(10.0)
    res = minimize_scalar(lambda ls: mean_crps(math.exp(ls), draws, center),
                          bounds=(math.log(ref) - half, math.log(ref) + half), method="bounded",
                          options={"xatol": SCALE_SEARCH_XATOL})
    return float(math.exp(res.x))


def predictive(fam: Family, chain: Chain, givens_meas: dict[str, float], levels: dict[str, float],
               z: np.ndarray) -> np.ndarray | None:
    """ln(target) draws of the predictive distribution, or None if any draw leaves the valid domain.

    `z` holds standard normals, one row per chain given (in chain.givens order) and one column per draw, so e' =
    s·z and the same z can be reused across noise levels (paired problems)."""
    draws = {}
    for i, g in enumerate(chain.givens):
        s = levels.get(g, 0.0)
        if s == 0.0:
            draws[g] = givens_meas[g]
            continue
        denom = 1.0 + s * z[i]
        if np.any(denom <= 0.0):
            return None
        draws[g] = givens_meas[g] / denom
    y = chain_target(fam, chain, draws)
    if not np.all(np.isfinite(y) & (y > 0)):
        return None
    return np.log(y)


def summarize(log_draws: np.ndarray, point_estimate_si: float, truth_si: float) -> dict:
    """σ* (CRPS-optimal, primary), its SD and SI versions, and the PIT of the truth under the draws."""
    center = math.log(point_estimate_si)
    y = np.exp(log_draws)
    return {"sigma_star_log": crps_optimal_scale(log_draws, center),
            "sigma_star_log_sd": float(np.std(log_draws)),
            "sigma_star_si_sd": float(np.std(y)),
            "pit_truth": float(np.mean(log_draws <= math.log(truth_si))),
            "mc_draws": int(log_draws.size)}

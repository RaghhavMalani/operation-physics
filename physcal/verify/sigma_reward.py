"""Verifier 1.3.0 (DECISIONS.md D105; your approval 2026-10-21): the σ reward against the generator's predictive
distribution, centred at f(x_meas).

    from physcal.verify.sigma_reward import reward      # the 1.3.0 entry point; older configs pass through
    reward(problem, completion, reward_config("reward_v13"))

score_sigma is the expected crps_log of the stated relative spread σ_rel = σ / |μ|, with the forecast centred at
f(x_meas) (the generator's point estimate from the displayed values) instead of the stated μ: the mean, over fixed
draws y_j of the problem's predictive distribution of ln(target) (`noise.predictive`'s model, its own seeded stream),
of the verifier's crps_log of N(ln f(x_meas), σ_rel²) at y_j. It depends on σ_rel and the problem only, so it credits
σ without the noise of one realized truth (FINDINGS #24: 5–23× lower variance) and without μ's error (R3-6). It is an
average of a proper score, hence maximized at the CRPS-optimal scale of the predictive distribution (A2's σ*). On
exact givens the predictive is the truth itself, and σ_rel = 0 is optimal.

Everything 1.2.0 computes is unchanged: `reward()` (physcal/verify/reward.py, not edited) scores the answer, the steps
and the confidence block, and this module adds score_sigma with weights.sigma. Configs older than 1.3.0 pass through
untouched, so every existing tape scores exactly as before. Draws that leave the valid domain are dropped (counted);
if fewer than `min_valid_share` remain, the draws are N(ln f(x_meas), σ*²) on the same stream (flagged).
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from physcal.verify.reward import (
    OK,
    _as_dict,
    _version_tuple,
    computation_reference_si,
    reward as reward_v12,
    validate_reward_config,
    verifier_version,
)
from physcal.verify.scoring import ceiling, floor

SIGMA_VERSION = (1, 3, 0)  # the first verifier with the predictive σ reward (D105)
SIGMA_UNSCORED, NO_RELATIVE_SIGMA = "answer_unscored", "no_relative_sigma"
_DRAWS: dict[tuple, tuple[np.ndarray, dict]] = {}


def has_sigma_reward(cfg: dict) -> bool:
    return _version_tuple(verifier_version(cfg)) >= SIGMA_VERSION


def validate(cfg: dict) -> None:
    """1.2.0's checks, plus: the sigma_reward section and weights.sigma exist iff version >= 1.3.0; a σ weight that
    cannot outweigh the final answer together with the steps and the confidence."""
    validate_reward_config(cfg)
    has = has_sigma_reward(cfg)
    if has != ("sigma_reward" in cfg) or has != ("sigma" in cfg["weights"]):
        raise ValueError(f"verifier {verifier_version(cfg)}: the sigma_reward section and weights.sigma must be set "
                         f"iff version >= {'.'.join(map(str, SIGMA_VERSION))}")
    if not has:
        return
    sr, sc, w = cfg["sigma_reward"], cfg["scoring"], cfg["weights"]
    if sc["rule"] != "crps_log":
        raise ValueError("the predictive σ reward is defined for the crps_log rule only")
    if sr["n_draws"] < 100 or not 0.0 < sr["min_valid_share"] <= 1.0 or not sr["seed_tag"]:
        raise ValueError("sigma_reward needs n_draws >= 100, 0 < min_valid_share <= 1 and a seed_tag")
    if w["sigma"] < 0:
        raise ValueError("weights.sigma must be >= 0")
    span = ceiling(sc["rule"], sc) - floor(sc["rule"], sc)
    side = w["steps"] + w["confidence"] * 1.0 + w["sigma"] * span
    if not side < w["final"] * span:
        raise ValueError(f"step + confidence + sigma weights {side} can outweigh the final answer ({w['final'] * span})")


# ------------------------------------------------------------------------------------------- the predictive draws
def _stream(problem: dict, tag: str) -> np.random.Generator:
    key = f"{tag}|{problem['family']}|{problem['depth']}|{problem['seed']}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:16], "little"))


def predictive_masked(fam, chain, measured: dict, levels: dict, z: np.ndarray) -> tuple[np.ndarray, int]:
    """`noise.predictive` with the invalid draws dropped instead of failing the whole set: (valid ln(target) draws,
    the number dropped). Identical to `noise.predictive` when every draw is valid (tested)."""
    from physcal.env.noise import chain_target

    n = z.shape[1]
    good = np.ones(n, bool)
    draws = {}
    for i, g in enumerate(chain.givens):
        s = levels.get(g, 0.0)
        if s == 0.0:
            draws[g] = measured[g]
            continue
        denom = 1.0 + s * z[i]
        good &= denom > 0.0
        draws[g] = measured[g] / np.where(denom > 0.0, denom, 1.0)
    with np.errstate(all="ignore"):
        y = np.asarray(chain_target(fam, chain, draws), dtype=float) * np.ones(n)
    good &= np.isfinite(y) & (y > 0)
    return np.log(y[good]), int(n - good.sum())


def predictive_draws(problem: dict, sr: dict) -> tuple[np.ndarray, dict]:
    """Fixed ln(target) draws of the problem's predictive distribution (cached per problem and settings), and how
    they were made. Exact givens: the truth itself (one draw)."""
    from physcal.env.formulas import get_family
    from physcal.env.generator import chain_of

    key = (sr["seed_tag"], sr["n_draws"], sr["min_valid_share"], problem["family"], problem["depth"], problem["seed"])
    if key in _DRAWS:
        return _DRAWS[key]
    levels = {g["symbol"]: g.get("noise_rel") or 0.0 for g in problem["givens"]}
    noise = problem.get("noise") or {}
    if not noise or not any(levels.values()):
        out = (np.array([math.log(problem["target_value_si"])]), {"sigma_draws": "exact", "sigma_draws_dropped": 0})
    else:
        fam, chain = get_family(problem["family"]), chain_of(problem)
        measured = {g["symbol"]: g["value_si"] for g in problem["givens"]}
        z = _stream(problem, sr["seed_tag"]).standard_normal((len(chain.givens), sr["n_draws"]))
        logs, dropped = predictive_masked(fam, chain, measured, levels, z)
        if len(logs) >= sr["min_valid_share"] * sr["n_draws"]:
            out = (logs, {"sigma_draws": "predictive", "sigma_draws_dropped": dropped})
        else:
            centre = math.log(computation_reference_si(problem))
            out = (centre + noise["sigma_star_log"] * z[0], {"sigma_draws": "gaussian_fallback",
                                                              "sigma_draws_dropped": dropped})
    _DRAWS[key] = out
    return out


# ------------------------------------------------------------------------------------------- the score
def expected_score(log_draws: np.ndarray, centre_si: float, sigma_rel: float, sc: dict) -> float:
    """Mean over the draws of the verifier's crps_log of N(ln centre, σ_rel²) at the draw (oracle_centred's
    vectorized form of `scoring.score`, floors and clip included)."""
    from physcal.eval.oracle_centred import crps_log_vec

    return float(np.mean(crps_log_vec(math.log(centre_si) - log_draws, np.full(log_draws.shape, sigma_rel), sc)))


def sigma_terms(problem: dict, bd: dict, cfg: dict) -> dict:
    sc, sr = cfg["scoring"], cfg["sigma_reward"]
    centre = computation_reference_si(problem)
    out: dict[str, Any] = {"sigma_centre_si": centre, "sigma_reason": OK, "sigma_rel_scored": None}
    if bd["reason"] != OK:
        return out | {"score_sigma": floor(sc["rule"], sc), "sigma_reason": SIGMA_UNSCORED}
    if not bd.get("mu_si") or bd.get("sigma_si") is None:
        return out | {"score_sigma": floor(sc["rule"], sc), "sigma_reason": NO_RELATIVE_SIGMA}
    rel = bd["sigma_si"] / abs(bd["mu_si"])
    draws, how = predictive_draws(problem, sr)
    return out | how | {"sigma_rel_scored": rel, "sigma_n_draws": int(len(draws)),
                        "score_sigma": expected_score(draws, centre, rel, sc)}


def reward(problem: Any, completion: str, cfg: dict | None = None) -> tuple[float, dict]:
    """The verifier with the 1.3.0 σ reward. For a config older than 1.3.0 this is `reward.reward`, unchanged."""
    from physcal.verify.reward import reward_config

    cfg = cfg or reward_config()
    validate(cfg)
    r, bd = reward_v12(problem, completion, cfg)
    if not has_sigma_reward(cfg):
        return r, bd
    p = _as_dict(problem)
    bd.update(sigma_terms(p, bd, cfg), w_sigma=cfg["weights"]["sigma"])
    bd["reward"] = r + cfg["weights"]["sigma"] * bd["score_sigma"]
    return bd["reward"], bd


def replay(path) -> dict:
    """`tape.replay` through this entry point: every logged completion re-scored with the tape's own reward config
    (any version); rewards and breakdowns must match exactly. 1.3.0 tapes need this until `tape.replay` dispatches
    on the version (D105)."""
    from physcal.env.generator import regenerate
    from physcal.tape.tape import read_meta, read_tape

    meta = read_meta(path)
    n, bad_r, bad_b = 0, [], []
    for i, rec in enumerate(read_tape(path)):
        n += 1
        r, bd = reward(regenerate(rec["problem"], meta["env_config"]), rec["completion"], meta["reward_config"])
        if r != rec["reward"]:
            bad_r.append(i)
        if bd != rec["breakdown"]:
            bad_b.append(i)
    return {"n": n, "reward_mismatches": bad_r, "breakdown_mismatches": bad_b}

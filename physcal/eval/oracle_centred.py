"""The oracle-centred σ diagnostic (DECISIONS.md D100; R3-6's primary for σ propagation). CPU only.

A source of stated relative σ is scored on noisy problems with its centre replaced by μ = f(x_meas), the chain on the
displayed givens (`noise.point_estimate_si`), so no problem is selected on the correctness of an answer:
- realized score: the verifier's crps_log of (μ, σ_rel·μ) at the noise-free truth (`scoring.score`, clip included);
- expected score: the same per-draw score averaged over fresh draws of the MC predictive distribution
  (`noise.predictive`, a per-problem seed of its own; an MC estimate, not the generator's own stream);
- ln(σ / σ*), the vol beta (OLS slope of ln σ on ln σ*) and the coverage of ln truth by ln μ ± k·σ_rel.
Uncertainty is a bootstrap over clusters (a pair on the pair sets, R3-7; a problem otherwise).
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy.special import ndtr

from physcal.env import noise as noise_mode
from physcal.env.formulas import get_family
from physcal.env.generator import chain_of
from physcal.verify.scoring import relative_terms, score, sigma_floor

_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
NOMINAL = {1: 0.6827, 2: 0.9545}


def verifier_crps_log(mu: float, sigma_rel: float, truth: float, sc: dict) -> float:
    """The verifier's final-answer score for the answer μ ± σ_rel·|μ| (rule crps_log, `sc` = the scoring block)."""
    e, s = relative_terms(mu, sigma_rel * abs(mu), truth, sigma_floor("crps_log", sc))
    return score("crps_log", e, s, sc)


def crps_log_vec(ln_ratio: np.ndarray, sigma_rel: np.ndarray, sc: dict) -> np.ndarray:
    """`verifier_crps_log` vectorized, from ln(μ / truth): the same floors (relative to the truth, then to μ) and clip."""
    fl = sigma_floor("crps_log", sc)
    ratio = np.exp(ln_ratio)
    s_log = np.maximum(np.maximum(sigma_rel * ratio, fl) / ratio, fl)
    z = ln_ratio / s_log
    crps = s_log * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * _INV_SQRT_2PI * np.exp(-0.5 * z * z) - _INV_SQRT_PI)
    return np.maximum(-crps, -sc["crps_clip"])


def mc_draws(problem: dict, n: int, tag: str) -> np.ndarray | None:
    """Fresh ln(target) draws of the problem's predictive distribution (per-given levels), or None if a draw leaves
    the valid domain. The stream is seeded from (tag, family, depth, seed), not the generator's own."""
    fam, chain = get_family(problem["family"]), chain_of(problem)
    measured = {g["symbol"]: g["value_si"] for g in problem["givens"]}
    levels = {g["symbol"]: g.get("noise_rel") or 0.0 for g in problem["givens"]}
    key = f"{tag}|{problem['family']}|{problem['depth']}|{problem['seed']}".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:16], "little"))
    return noise_mode.predictive(fam, chain, measured, levels, rng.standard_normal((len(chain.givens), n)))


def expected_crps_log(log_draws: np.ndarray, ln_mu: float, sigma_rel: float, sc: dict) -> float:
    """Mean over the draws of the verifier's score with the draw as the truth (the expected score under the
    predictive distribution)."""
    return float(np.mean(crps_log_vec(ln_mu - log_draws, np.full(log_draws.shape, sigma_rel), sc)))


def problem_terms(problem: dict, n_draws: int, tag: str) -> dict:
    """What the diagnostic needs from one noisy problem (fresh MC draws included)."""
    n = problem["noise"]
    return {"mu": n["point_estimate_si"], "truth": problem["target_value_si"],
            "ln_ratio": math.log(n["point_estimate_si"]) - math.log(problem["target_value_si"]),
            "sigma_star": n["sigma_star_log"], "draws": mc_draws(problem, n_draws, tag)}


def score_source(terms: list[dict], sigma_rel: np.ndarray, sc: dict, mu_scale: float = 1.0) -> dict:
    """Per-problem realized and expected scores of one source (μ = f(x_meas)·mu_scale; NaN where no draws)."""
    realized = np.array([verifier_crps_log(t["mu"] * mu_scale, float(s), t["truth"], sc) for t, s in zip(terms, sigma_rel)])
    expected = np.array([np.nan if t["draws"] is None else
                         expected_crps_log(t["draws"], math.log(t["mu"] * mu_scale), float(s), sc)
                         for t, s in zip(terms, sigma_rel)])
    return {"realized": realized, "expected": expected}


class ClusterBootstrap:
    """Bootstrap over clusters: resampled cluster indices, fixed once so every statistic uses the same draws."""

    def __init__(self, clusters: list, n_boot: int, seed: int):
        keys = {c: i for i, c in enumerate(dict.fromkeys(clusters))}
        self.ids = np.array([keys[c] for c in clusters])
        self.k = len(keys)
        self.counts = np.random.default_rng(seed).multinomial(self.k, np.full(self.k, 1.0 / self.k), size=n_boot)

    def sums(self, v: np.ndarray) -> np.ndarray:
        return np.bincount(self.ids, weights=v, minlength=self.k)

    def mean(self, v: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, list[float]]:
        m = np.ones(len(v), bool) if mask is None else mask
        num, den = self.sums(np.where(m, v, 0.0)), self.sums(m.astype(float))
        boot = (self.counts @ num) / (self.counts @ den)
        return float(num.sum() / den.sum()), np.percentile(boot, [2.5, 97.5]).tolist()

    def slope(self, x: np.ndarray, y: np.ndarray) -> tuple[float, list[float]]:
        """OLS slope of y on x (with an intercept) from per-cluster sums."""
        s = [self.sums(a) for a in (np.ones_like(x), x, y, x * y, x * x)]

        def b(n, sx, sy, sxy, sxx):
            return (n * sxy - sx * sy) / (n * sxx - sx * sx)
        boot = b(*(self.counts @ a for a in s))
        return float(b(*(a.sum() for a in s))), np.percentile(boot, [2.5, 97.5]).tolist()


def summarize(terms: list[dict], sigma_rel: np.ndarray, scores: dict, boot: ClusterBootstrap, sc: dict,
              ks: list[int]) -> dict:
    """The diagnostic of one source: scores, ln(σ / σ*), vol beta, coverage, and the verifier's clip / floor shares."""
    star = np.array([t["sigma_star"] for t in terms])
    dev = np.abs(np.array([t["ln_ratio"] for t in terms]))
    fl = sigma_floor("crps_log", sc)
    pos = sigma_rel > 0
    lr = np.log(np.maximum(sigma_rel, fl) / star)
    has_draws = np.isfinite(scores["expected"])
    out = {"n": len(terms), "n_with_draws": int(has_draws.sum()),
           "realized": boot.mean(scores["realized"]), "expected": boot.mean(np.nan_to_num(scores["expected"]), has_draws),
           "median_log_ratio": float(np.median(lr)), "median_abs_log_ratio": float(np.median(np.abs(lr))),
           "vol_beta": boot.slope(np.log(star), np.log(np.maximum(sigma_rel, fl))),
           "share_sigma_zero": float(np.mean(~pos)), "share_at_clip": float(np.mean(scores["realized"] <= -sc["crps_clip"])),
           "share_floor_binds": float(np.mean(sigma_rel < fl))}
    for k in ks:
        out[f"coverage_{k}"] = boot.mean((dev <= k * sigma_rel).astype(float))
        out[f"nominal_{k}"] = NOMINAL.get(k)
    return out

"""Scoring rules on relative errors.

Notation (all SI): truth y, prediction mu ± sigma.
    e = (mu - y) / |y|          relative error
    s = max(sigma / |y|, floor) relative sigma
    z = e / s

Only affine transforms of these scores are allowed downstream: a non-linear map such as exp(-CRPS)
breaks propriety, and then honest sigma is no longer optimal.

Caveat (see DECISIONS.md): dividing by the *truth* |y| weights outcomes by 1/|y|, so `crps` and `log_score`
are only approximately proper. The bias is negligible for relative spreads of a few percent and grows to
about -11% in the optimal mu at 30% spread. `crps_log` scores ln(y) against N(ln mu, (sigma/mu)^2) instead.
That is strictly proper for beliefs that are Gaussian in log space, and it is still scale-free.
"""

from __future__ import annotations

import math

RULES = ("crps", "binary", "log_score", "abs_rel_error", "crps_log")
RULES_NEEDING_SIGMA = ("crps", "log_score", "crps_log")

_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_pdf(z: float) -> float:
    return _INV_SQRT_2PI * math.exp(-0.5 * z * z)


def crps_standard(z: float) -> float:
    """CRPS of N(0, 1) at outcome z: z(2Φ(z) − 1) + 2φ(z) − 1/√π."""
    return z * (2.0 * norm_cdf(z) - 1.0) + 2.0 * norm_pdf(z) - _INV_SQRT_PI


def crps_gaussian(mu: float, sigma: float, y: float) -> float:
    """CRPS of N(mu, sigma^2) at outcome y."""
    return sigma * crps_standard((y - mu) / sigma)


def relative_terms(mu: float, sigma: float | None, y: float, sigma_floor: float) -> tuple[float, float | None]:
    e = (mu - y) / abs(y)
    s = None if sigma is None else max(sigma / abs(y), sigma_floor)
    return e, s


def score(rule: str, e: float, s: float | None, cfg: dict) -> float:
    """Final-answer score under `rule` (higher is better), clipped below at `floor(rule)`."""
    if rule == "crps":
        return max(-s * crps_standard(e / s), -cfg["crps_clip"])
    if rule == "log_score":
        nll = 0.5 * math.log(2.0 * math.pi * s * s) + 0.5 * (e / s) ** 2
        return max(-nll, -cfg["log_score_clip"])
    if rule == "abs_rel_error":
        return max(-abs(e), -cfg["abs_rel_error_clip"])
    if rule == "binary":
        return 1.0 if abs(e) <= cfg["binary_tolerance"] else 0.0
    if rule == "crps_log":
        ratio = 1.0 + e  # mu / y, truths are positive
        if ratio <= 0:
            return -cfg["crps_clip"]
        s_log = max(s / ratio, cfg["sigma_floor"])  # sigma / mu
        return max(-s_log * crps_standard(math.log(ratio) / s_log), -cfg["crps_clip"])
    raise ValueError(f"unknown scoring rule {rule!r}")


def floor(rule: str, cfg: dict) -> float:
    """Worst possible score; also given to parse failures and unit mismatches."""
    return {
        "crps": -cfg["crps_clip"],
        "crps_log": -cfg["crps_clip"],
        "log_score": -cfg["log_score_clip"],
        "abs_rel_error": -cfg["abs_rel_error_clip"],
        "binary": 0.0,
    }[rule]


def ceiling(rule: str, cfg: dict) -> float:
    """Best possible score (supremum)."""
    return {
        "crps": 0.0,
        "crps_log": 0.0,
        "log_score": -0.5 * math.log(2.0 * math.pi * cfg["sigma_floor"] ** 2),
        "abs_rel_error": 0.0,
        "binary": 1.0,
    }[rule]

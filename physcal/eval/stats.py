"""Small statistical helpers shared by the Phase 10+ analyses (coverage tests, multiple comparisons)."""

from __future__ import annotations

import math

from scipy.stats import binom, chi2


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * math.log(y)


def kupiec(n_covered: int, n: int, nominal: float) -> dict:
    """Kupiec's unconditional-coverage likelihood-ratio test (two-sided, χ²(1)): is the coverage rate `nominal`?"""
    miss, p = n - n_covered, 1.0 - nominal
    obs = miss / n
    lr = -2.0 * (_xlogy(n - miss, 1.0 - p) + _xlogy(miss, p) - _xlogy(n - miss, 1.0 - obs) - _xlogy(miss, obs))
    lr = max(lr, 0.0)
    return {"n": n, "covered": n_covered, "coverage": n_covered / n, "nominal": nominal, "lr": lr,
            "p": float(chi2.sf(lr, 1))}


def holm(pvalues: dict[str, float], alpha: float) -> dict[str, dict]:
    """Holm step-down: per hypothesis, its threshold and whether it is rejected."""
    out, still = {}, True
    for i, name in enumerate(sorted(pvalues, key=pvalues.get)):
        thr = alpha / (len(pvalues) - i)
        still = still and pvalues[name] <= thr
        out[name] = {"p": pvalues[name], "threshold": thr, "rejected": still}
    return out


def binomial_interval(n: int, nominal: float, level: float) -> tuple[float, float]:
    """Central `level` interval of the coverage rate of n Bernoulli(nominal) trials."""
    tail = (1.0 - level) / 2.0
    return float(binom.ppf(tail, n, nominal) / n), float(binom.ppf(1.0 - tail, n, nominal) / n)

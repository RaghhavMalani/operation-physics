"""Coverage-test helpers used by the Phase 10 gate (synthetic checks only)."""

import numpy as np
import pytest

from physcal.eval.stats import binomial_interval, holm, kupiec


def test_kupiec_accepts_nominal_and_rejects_off_nominal():
    assert kupiec(683, 1000, 0.6827)["p"] > 0.9
    assert kupiec(600, 1000, 0.6827)["p"] < 1e-6
    assert kupiec(1000, 1000, 0.95)["p"] < 1e-10  # zero misses is handled (0·ln 0 = 0)


def test_kupiec_size_is_close_to_alpha():
    rng = np.random.default_rng(0)
    p = [kupiec(int(rng.binomial(500, 0.6827)), 500, 0.6827)["p"] for _ in range(4000)]
    assert np.mean(np.array(p) < 0.05) == pytest.approx(0.05, abs=0.015)


def test_holm():
    h = holm({"a": 0.001, "b": 0.03, "c": 0.04}, 0.05)
    assert h["a"]["rejected"] and not h["b"]["rejected"] and not h["c"]["rejected"]  # 0.03 > 0.05/2, then stop
    assert holm({"a": 0.01, "b": 0.02}, 0.05)["b"]["rejected"]


def test_binomial_interval_contains_99_percent():
    lo, hi = binomial_interval(500, 0.6827, 0.99)
    rng = np.random.default_rng(1)
    cov = rng.binomial(500, 0.6827, 20000) / 500
    assert 0.985 <= np.mean((cov >= lo) & (cov <= hi)) <= 1.0

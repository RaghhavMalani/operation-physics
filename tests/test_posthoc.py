"""Post-hoc σ-scale control: both fits recover a known mis-scaling on synthetic data."""

import numpy as np
import pytest

from physcal.config import load_config
from physcal.eval.posthoc import coverage_scale, fit_crps_scale, summary
from physcal.verify.reward import reward_config

SC = reward_config()["scoring"]
FIT = load_config("posthoc")["fit"]


def _records(true_scale: float, n: int = 3000, seed: int = 0) -> list[dict]:
    """Stated relative σ = 1%, actual log-error spread = true_scale × 1%; plus a few failures."""
    rng = np.random.default_rng(seed)
    recs = [{"split": "x", "key": ("k", 1, i), "truth": 1.0, "mu": float(np.exp(true_scale * 0.01 * z)),
             "sigma": 0.01 * float(np.exp(true_scale * 0.01 * z))} for i, z in enumerate(rng.normal(size=n))]
    recs += [{"split": "x", "key": ("k", 1, n + i), "truth": 1.0, "mu": None, "sigma": None} for i in range(20)]
    return recs


@pytest.mark.parametrize("true_scale", [0.5, 3.0, 40.0])
def test_fits_recover_the_true_scale(true_scale):
    recs = _records(true_scale)
    assert fit_crps_scale(recs, SC, FIT) == pytest.approx(true_scale, rel=0.08)
    assert coverage_scale(recs, FIT["coverage_target"]) == pytest.approx(true_scale, rel=0.08)


def test_summary_counts_failures_as_floor_and_excludes_them_from_coverage():
    recs = _records(1.0)
    s = summary(recs, 1.0, SC)
    assert s["n"] == 3020 and s["n_with_sigma"] == 3000
    assert s["coverage_1sigma"] == pytest.approx(0.6827, abs=0.03)
    assert s["mean_crps_log"] < -5.0 * 20 / 3020


def test_coverage_scale_counts_zero_sigma_answers_in_the_denominator():
    """A stated σ = 0 can't be covered by any scale; the target is over all answers with a σ, as reported."""
    recs = _records(3.0, n=1000)
    for r in recs[:200]:
        r["sigma"] = 0.0
    c = coverage_scale(recs, 0.6)
    assert summary(recs, c, SC)["coverage_1sigma"] == pytest.approx(0.6, abs=0.002)
    with pytest.raises(ValueError):
        coverage_scale(recs, 0.9)

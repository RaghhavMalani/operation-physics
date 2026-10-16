"""D62 exploratory report: summary and within-problem statistics on synthetic rows."""

import math
import sys

import pytest

from physcal.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import sigma_greedy_vs_sampled as gs  # noqa: E402


def row(key, sigma, err=0.01, fail=False, cov1=False):
    return {"key": key, "fail": fail, "rel_sigma": None if fail else sigma, "err": None if fail else err,
            "cov1": cov1, "cov2": cov1, "crps_log": gs.FAIL_CRPS_LOG if fail else -0.1}


def test_within_problem_ranks_the_greedy_sigma_among_its_samples():
    greedy = [row("a", 0.001), row("b", 0.02), row("c", 0.0)]
    sampled = [row("a", s) for s in (0.01, 0.02, 0.03, 0.04)] + [row("b", s) for s in (0.01, 0.02, 0.03, 0.04)] + \
              [row("c", 0.0), row("c", 0.0, fail=True)]
    w = gs.within_problem(greedy, sampled)
    assert w["problems_with_2plus_samples"] == 2  # c has one answered sample
    assert w["greedy_below_all_samples"] == pytest.approx(1 / 3)
    assert w["greedy_mean_midrank"] == pytest.approx((0 + 1.5 / 4 + 0.5) / 3)
    assert w["median_sd_ln_sigma"] == pytest.approx(math.sqrt(sum((math.log(s) - sum(math.log(x) for x in
                                                    (0.01, 0.02, 0.03, 0.04)) / 4) ** 2 for s in (0.01, 0.02, 0.03, 0.04)) / 4))


def test_describe_counts_failures_as_not_covered_at_the_floor():
    d = gs.describe([row("a", 0.0, cov1=True), row("b", 0.02), row("c", 0.0, fail=True), row("d", 0.04, err=0.05)])
    assert d["fail_rate"] == 0.25 and d["cov1"] == 0.25 and d["sigma_zero"] == pytest.approx(1 / 3)
    assert d["mean_crps_log"] == pytest.approx((3 * -0.1 + gs.FAIL_CRPS_LOG) / 4)
    assert d["median_rel_sigma"] == 0.02

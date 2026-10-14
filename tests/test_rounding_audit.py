"""Phase 10 step 2 rounding audit helpers."""

import json

import numpy as np
import pytest

from physcal.config import REPO_ROOT
from physcal.env.generator import generator_version
from physcal.eval.rounding_audit import audit_problem, half_ulp_rel, round_sig


def test_half_ulp_and_round_sig():
    assert half_ulp_rel("3.40", 3) == pytest.approx(0.005 / 3.40)
    assert half_ulp_rel("0.0479", 3) == pytest.approx(0.00005 / 0.0479)
    assert half_ulp_rel("5080", 3) == pytest.approx(5 / 5080)
    assert np.allclose(round_sig(np.array([12345.6, 0.0012349, 9.996]), 3), [12300.0, 0.00123, 10.0])


def test_truth_is_reproduced_from_displayed_givens_and_rounding_errors_are_small():
    path = REPO_ROOT / "data" / "eval_sets" / generator_version() / "id.jsonl"
    rng = np.random.default_rng(0)
    for line in list(path.open())[:24]:
        r = audit_problem(json.loads(line), 3, 500, rng, [4, 6])
        assert r["truth_rel_diff"] < 1e-12 and r["given_si_rel_diff"] < 1e-12
        assert 0 < r["sigma_star_rel"] < 0.05
        assert r["solver_rounding_rel_error"]["6"] < r["solver_rounding_rel_error"]["4"] + 1e-12 < 0.01

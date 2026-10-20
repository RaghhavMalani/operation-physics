"""D95: the noise heuristics (max / mean / RSS of the visible levels) and the A6c analysis. Every heuristic is
exactly constant within an A6c pair; an honest propagator gives slope ≈ 1 on A6c, a lookup 0."""

import hashlib
import json
import math

import numpy as np
import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.eval import a6_pairs, noise_heuristics as H

PC = load_config("pairs")
A6C = REPO_ROOT / "data" / "eval_sets" / PC["noise_heuristics"]["generator_version"] / "a6c_pairs.jsonl"


def test_heuristics_on_a_toy_problem():
    p = {"givens": [{"noise_rel": 0.01}, {"noise_rel": 0.05}, {}], "distractors": [{"noise_rel": 0.02}]}
    assert H.levels(p, "givens") == [0.01, 0.05] and H.levels(p, "shown") == [0.01, 0.02, 0.05]  # exact values skipped
    assert H.sigma(p, "max", "givens") == 0.05 and H.sigma(p, "mean", "givens") == pytest.approx(0.03)
    assert H.sigma(p, "rss", "shown") == pytest.approx(math.sqrt(0.01**2 + 0.02**2 + 0.05**2))
    assert H.sigma({"givens": [{}]}, "max", "givens") is None


@pytest.mark.skipif(not A6C.exists(), reason="A6c not exported")
def test_the_frozen_a6c_set_and_heuristics_are_constant_within_its_pairs():
    blob = A6C.read_bytes()
    man = json.loads(A6C.with_name("a6c_pairs_manifest.json").read_text())
    assert hashlib.sha256(blob).hexdigest() == man["sha256"] and man["n_problems"] == 1020 == len(blob.splitlines())
    probs = [json.loads(l) for l in blob.splitlines()]
    by = {}
    for p in probs:
        by.setdefault(p["noise"]["pair"]["pair_seed"], []).append(p)
    assert len(by) == 510 and all(len(v) == 2 for v in by.values())
    for lo, hi in by.values():
        assert lo["noise"]["sigma_star_log"] < hi["noise"]["sigma_star_log"]
        for h in H.HEURISTICS:
            for scope in H.SCOPES:
                assert H.sigma(lo, h, scope) == H.sigma(hi, h, scope)  # exactly: y = 0 on every pair


def _rows(stated):
    """Synthetic A6c rows: x = Δ ln σ* in [ln 1.5, ln 5]; `stated(ln σ*)` gives the stated ln σ."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        lo = math.log(0.01) + rng.normal(scale=0.3)
        for m, s in (("lo", lo), ("hi", lo + rng.uniform(math.log(1.5), math.log(5)))):
            rows.append({"pair": ("energy", 2, 0.05, i), "member": m, "kind": "allocation", "sigma_star_log": math.exp(s),
                         "truth": 1.0, "ok": True, "computation_correct": True, "stated_log": stated(s, rng),
                         "sigma_zero": False})
    return rows


def test_allocation_analysis_separates_propagation_from_lookup():
    cfg = PC["analysis"] | {"bootstrap": 500}
    honest = a6_pairs.analyze_rows(_rows(lambda s, rng: s + rng.normal(scale=0.05)), cfg)
    lookup = a6_pairs.analyze_rows(_rows(lambda s, rng: math.log(0.05)), cfg)
    assert honest["kind"] == "allocation" and honest["supported"] and honest["label"] == "full propagation"
    assert honest["primary_all_answered"]["slope"] == pytest.approx(1.0, abs=0.1)
    assert honest["secondary_mean"]["through_origin_slope"] == pytest.approx(1.0, abs=0.05)
    assert not lookup["supported"] and lookup["primary_all_answered"]["slope"] == 0.0
    assert lookup["secondary_mean"]["mean_dy"] == 0.0 and lookup["label"] == "no propagation"

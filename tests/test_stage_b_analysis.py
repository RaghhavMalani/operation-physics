"""D104's analysis on synthetic eval tapes over the real frozen A6 / A6b / A6c sets: a model that states σ = σ* at
μ = f(x_meas) must read "propagates" and beat the surface readers; one that states the RSS heuristic must not."""

import copy
import json

import numpy as np
import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.env.generator import problem_from_dict
from physcal.eval import noise_heuristics, stage_a
from physcal.eval import stage_b_analysis as A
from physcal.verify.reward import reward, reward_config

pytestmark = pytest.mark.skipif(not (REPO_ROOT / "results/x1_text_sigma/a6c/predictions.jsonl").exists(),
                                reason="X1 predictions not present")


def _tape(path, problems, sigma_of):
    rw = reward_config()
    lines = []
    for p in problems:
        mu = p["noise"]["point_estimate_si"]
        text = f"<steps>\n</steps>\n<answer>{mu!r} ± {sigma_of(p) * abs(mu)!r} {p['target_si_unit']}</answer>\n" \
               "<confidence>50%</confidence>"
        r, bd = reward(problem_from_dict({k: v for k, v in p.items() if k != "_set"}), text, rw)
        lines.append(json.dumps({"split": p["_set"], "problem": {k: v for k, v in p.items() if k != "_set"},
                                 "completion": text, "reward": r, "breakdown": bd}))
    path.mkdir(parents=True)
    (path / "rollouts.jsonl").write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def tapes(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage_b")
    problems = stage_a.load_sets(load_config("stage_a"))
    _tape(root / "seed0" / "eval", problems, lambda p: p["noise"]["sigma_star_log"])
    _tape(root / "seed1" / "eval", problems, lambda p: noise_heuristics.sigma(p, "rss", "givens"))
    return root


def _cfg(seeds):
    cfg = copy.deepcopy(A._config())
    cfg["seeds"] = seeds
    cfg["analysis"]["bootstrap"] = 200
    return cfg


def test_sigma_star_model_propagates_and_beats_the_surface_readers(tapes):
    res = A.analyze(_cfg([0, 2]), tapes)  # seed 2 has no eval: left out of the mean
    t = res["tests"]
    assert not res["seeds"][2]["ran"] and t["B1"]["seeds"] == [0]
    assert t["B1"]["value"] == pytest.approx(1.0, abs=1e-9) and t["B2"]["value"] == pytest.approx(1.0, abs=1e-9)
    assert all(t[k]["holm"]["rejected"] for k in A.TESTS)
    assert res["seed_mean"]["a6_label"] == "full propagation" and res["seed_mean"]["a6b_equivalent_to_zero"]
    assert res["reading"][0].startswith("the ceiling propagates") and "beats the surface readers" in res["reading"][1]
    s0 = res["seeds"][0]
    assert s0["tape"]["parse_fail_rate"] == 0 and s0["tape"]["a1_computation_correct_rate"] == 1.0
    assert s0["tape"]["confidence_parse_rate"] == 1.0
    # the per-seed pair statistic is a6_pairs' own (D59 code) and agrees with the joint bootstrap's point estimate
    assert s0["a6"]["primary_computation_correct"]["slope"] == pytest.approx(t["B1"]["value"], abs=1e-9)
    assert s0["oracle_centred"]["median_abs_log_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert (tapes / "summary.md").exists()


def test_rss_model_fails_b1_b2_b3(tapes):
    res = A.analyze(_cfg([1]), tapes)
    t = res["tests"]
    assert t["B1"]["value"] == pytest.approx(0.0, abs=1e-9) and t["B2"]["value"] == pytest.approx(0.0, abs=1e-9)
    assert t["B3"]["value"] == pytest.approx(0.0, abs=1e-12)
    assert not any(t[k]["holm"]["rejected"] for k in ("B1", "B2", "B3"))
    assert any(r.startswith("B2 not supported") for r in res["reading"])
    rss = res["comparators"]["rss"]["oracle_centred"]
    assert res["seeds"][1]["oracle_centred"]["expected"][0] == pytest.approx(rss["expected"][0], abs=1e-12)


def test_no_seed_counts_as_p_one(tapes):
    res = A.analyze(_cfg([2]), tapes)
    assert all(res["tests"][k].get("missing") and res["tests"][k]["holm"]["p"] == 1.0 for k in A.TESTS)

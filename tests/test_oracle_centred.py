"""The oracle-centred diagnostic (D100): verifier identity, the expected score's optimum, the cluster bootstrap."""

import math

import numpy as np
import pytest

from physcal.config import load_config
from physcal.env.generator import generate_problem
from physcal.eval import oracle_centred as oc
from physcal.verify.reward import reward, reward_config


@pytest.fixture(scope="module")
def sc():
    return reward_config()["scoring"]


@pytest.fixture(scope="module")
def noisy():
    cfg = load_config("env_v2")
    return [generate_problem("kinematics", d, 7_000_000 + d, split="eval", cfg=cfg, noise=0.02).to_dict()
            for d in (1, 2, 3)]


def test_vectorized_score_equals_the_verifier(sc):
    rng = np.random.default_rng(0)
    ln_ratio = rng.normal(0, 0.05, 400)
    ln_ratio[:5] = [0.0, 3.0, -3.0, 1e-9, -0.4]
    sig = np.exp(rng.uniform(math.log(1e-8), math.log(0.5), 400))
    sig[:3] = [0.0, 1e-7, 2.0]
    vec = oc.crps_log_vec(ln_ratio, sig, sc)
    ref = [oc.verifier_crps_log(math.exp(lr), s, 1.0, sc) for lr, s in zip(ln_ratio, sig)]
    assert np.max(np.abs(vec - np.array(ref))) < 1e-12
    assert vec.min() >= -sc["crps_clip"]


def test_realized_score_equals_reward_score_final(noisy, sc):
    rw = reward_config()
    for p in noisy:
        mu = p["noise"]["point_estimate_si"]
        for rel in (p["noise"]["sigma_star_log"], 0.01, 0.0):
            text = f"<steps>\n</steps>\n<answer>{mu!r} ± {rel * mu!r} {p['target_si_unit']}</answer>"
            _, bd = reward(p, text, rw)
            assert bd["reason"] == "ok"
            assert bd["score_final"] == oc.verifier_crps_log(mu, rel, p["target_value_si"], sc)


def test_expected_score_peaks_near_sigma_star(noisy, sc):
    for p in noisy:
        t = oc.problem_terms(p, 4000, "test")
        assert t["draws"] is not None and t["draws"].shape == (4000,)
        cs = np.exp(np.linspace(math.log(0.5), math.log(2.0), 41))
        e = [oc.expected_crps_log(t["draws"], math.log(t["mu"]), c * t["sigma_star"], sc) for c in cs]
        assert 0.85 < cs[int(np.argmax(e))] < 1.15  # fresh draws: an MC estimate around the stored σ*


def test_fresh_draws_are_reproducible_and_seeded_by_tag(noisy):
    a, b, c = (oc.mc_draws(noisy[0], 100, t) for t in ("x", "x", "y"))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_cluster_bootstrap_mean_and_slope():
    rng = np.random.default_rng(1)
    x = rng.normal(size=60)
    y = 0.7 * x + rng.normal(scale=0.1, size=60)
    clusters = [i // 2 for i in range(60)]
    boot = oc.ClusterBootstrap(clusters, 500, 0)
    slope, ci = boot.slope(x, y)
    assert math.isclose(slope, np.polyfit(x, y, 1)[0], rel_tol=1e-12) and ci[0] < slope < ci[1]
    m, mci = boot.mean(y)
    assert math.isclose(m, y.mean(), rel_tol=1e-12) and mci[0] < m < mci[1]
    mask = x > 0
    assert math.isclose(boot.mean(y, mask)[0], y[mask].mean(), rel_tol=1e-12)


def test_summarize_coverage_and_shares(sc):
    terms = [{"ln_ratio": d, "sigma_star": 0.01, "mu": math.exp(d), "truth": 1.0, "draws": None}
             for d in (0.005, -0.015, 0.025, 0.001)]
    sig = np.array([0.01, 0.01, 0.01, 0.0])
    scores = oc.score_source(terms, sig, sc)
    s = oc.summarize(terms, sig, scores, oc.ClusterBootstrap([0, 1, 2, 3], 50, 0), sc, [1, 2])
    assert s["coverage_1"][0] == 0.25 and s["coverage_2"][0] == 0.5  # σ = 0 covers no nonzero deviation
    assert s["share_sigma_zero"] == 0.25 and s["share_floor_binds"] == 0.25 and s["n_with_draws"] == 0

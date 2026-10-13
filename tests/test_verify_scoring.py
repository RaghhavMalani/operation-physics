"""Scoring rules: closed form vs integral definition, propriety by Monte Carlo, clipping."""

import math

import numpy as np
import pytest
from scipy import integrate, stats

from physcal.config import load_config
from physcal.verify.scoring import RULES, ceiling, crps_gaussian, crps_standard, floor, relative_terms, score

SC = load_config("reward")["scoring"]


def crps_by_integration(mu, sigma, y):
    """CRPS(F, y) = ∫ (F(x) - 1[x >= y])^2 dx."""
    F = lambda x: stats.norm.cdf(x, mu, sigma)
    left, _ = integrate.quad(lambda x: F(x) ** 2, -np.inf, y)
    right, _ = integrate.quad(lambda x: (F(x) - 1.0) ** 2, y, np.inf)
    return left + right


@pytest.mark.parametrize("mu, sigma, y", [(0, 1, 0), (0, 1, 1.3), (5, 0.2, 4.1), (-2, 3, 7), (100, 5, 101)])
def test_crps_closed_form_matches_integral(mu, sigma, y):
    assert math.isclose(crps_gaussian(mu, sigma, y), crps_by_integration(mu, sigma, y), rel_tol=1e-6, abs_tol=1e-9)


def test_crps_standard_values():
    assert math.isclose(crps_standard(0.0), 2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi), rel_tol=1e-12)
    assert crps_standard(1.0) == pytest.approx(crps_standard(-1.0))
    assert crps_standard(10.0) == pytest.approx(10.0 - 1 / math.sqrt(math.pi), rel=1e-9)


def test_relative_crps_is_absolute_crps_over_abs_truth():
    mu, sigma, y = 12.0, 0.7, 12.5
    e, s = relative_terms(mu, sigma, y, SC["sigma_floor"])
    assert math.isclose(-score("crps", e, s, SC), crps_gaussian(mu, sigma, y) / abs(y), rel_tol=1e-12)


N_MC = 400_000


def test_expected_crps_minimized_at_true_sigma_monte_carlo():
    """If the truth is distributed N(m, s0^2), reporting sigma = s0 minimizes the expected CRPS."""
    rng = np.random.default_rng(0)
    s0 = 1.0
    y = rng.normal(0.0, s0, N_MC)
    grid = np.linspace(0.5, 2.0, 31)
    z = lambda sig: y / sig
    phi, Phi = stats.norm.pdf, stats.norm.cdf
    expected = [np.mean(sig * (z(sig) * (2 * Phi(z(sig)) - 1) + 2 * phi(z(sig)) - 1 / np.sqrt(np.pi))) for sig in grid]
    best = grid[int(np.argmin(expected))]
    assert abs(best - s0) <= grid[1] - grid[0]
    # under- and over-confidence are both strictly worse
    assert expected[0] > min(expected) and expected[-1] > min(expected)


def test_expected_crps_minimized_at_true_mean_monte_carlo():
    rng = np.random.default_rng(1)
    y = rng.normal(3.0, 1.0, N_MC)
    grid = np.linspace(2.0, 4.0, 41)
    zs = [(y - m) for m in grid]
    expected = [np.mean(z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)) for z in zs]
    assert abs(grid[int(np.argmin(expected))] - 3.0) <= grid[1] - grid[0]


def test_relative_score_pipeline_rewards_honest_sigma():
    """Full pipeline (relative units, floor, clip): with a 2% relative spread the best reported
    sigma is the true one, within grid resolution."""
    rng = np.random.default_rng(2)
    y0, s0 = 100.0, 2.0
    ys = rng.normal(y0, s0, 100_000)
    grid = np.linspace(1.0, 3.0, 21)

    def mean_score(sig):
        e = (y0 - ys) / np.abs(ys)
        s = np.maximum(sig / np.abs(ys), SC["sigma_floor"])
        z = e / s
        crps = s * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
        return np.mean(np.maximum(-crps, -SC["crps_clip"]))

    scores = [mean_score(sig) for sig in grid]
    assert abs(grid[int(np.argmax(scores))] - s0) <= grid[1] - grid[0]


def test_scores_are_clipped_and_bounded():
    for rule in RULES:
        for e in (0.0, 0.005, 0.3, 50.0, -1e6):
            for s in (1e-3, 0.05, 10.0):
                v = score(rule, e, s, SC)
                assert floor(rule, SC) <= v <= ceiling(rule, SC) + 1e-12, (rule, e, s, v)
    assert score("crps", 1e6, 1e-3, SC) == -SC["crps_clip"]
    assert score("binary", 0.0099, None, SC) == 1.0 and score("binary", 0.0101, None, SC) == 0.0
    assert score("abs_rel_error", -0.25, None, SC) == -0.25


def test_crps_prefers_correct_sharp_answers():
    sharp_right = score("crps", 0.0, 0.01, SC)
    vague_right = score("crps", 0.0, 0.2, SC)
    sharp_wrong = score("crps", 0.3, 0.01, SC)
    honest_wrong = score("crps", 0.3, 0.3, SC)
    assert sharp_right > vague_right > honest_wrong > sharp_wrong


def test_log_score_ceiling_matches_sigma_floor():
    assert math.isclose(score("log_score", 0.0, SC["sigma_floor"], SC), ceiling("log_score", SC), rel_tol=1e-12)


def _mean_score(rule, ys, mu, sigma):
    return np.mean([score(rule, (mu - y) / abs(y), max(sigma / abs(y), SC["sigma_floor"]), SC) for y in ys])


def test_relative_crps_bias_for_wide_beliefs_is_documented():
    """Known property of the spec's normalization by |y| (DECISIONS.md): with a 30% belief spread,
    shading mu downwards beats reporting the belief's mean. If this ever fails, update DECISIONS.md."""
    rng = np.random.default_rng(3)
    ys = rng.normal(1.0, 0.3, 20_000)
    ys = ys[ys > 0]
    assert _mean_score("crps", ys, 0.9, 0.3) > _mean_score("crps", ys, 1.0, 0.3)


def test_crps_log_is_proper_for_lognormal_beliefs():
    """Belief ln(y) ~ N(0, s0^2): the best report is mu = median = 1 and sigma = s0 * mu, even at 30% spread."""
    rng = np.random.default_rng(4)
    s0 = 0.3
    ys = np.exp(rng.normal(0.0, s0, 20_000))
    sig_grid = np.linspace(0.2, 0.4, 21)
    best_sig = sig_grid[int(np.argmax([_mean_score("crps_log", ys, 1.0, c) for c in sig_grid]))]
    assert abs(best_sig - s0) <= sig_grid[1] - sig_grid[0]
    mu_grid = np.linspace(0.9, 1.1, 21)
    best_mu = mu_grid[int(np.argmax([_mean_score("crps_log", ys, m, s0 * m) for m in mu_grid]))]
    assert abs(best_mu - 1.0) <= mu_grid[1] - mu_grid[0]


def test_crps_log_is_proper_below_the_old_floor():
    """Belief ln(y) ~ N(0, s0^2) with s0 = 0.03%, below the old 0.1% floor: the best reported sigma is s0. Under
    verifier 1.0.0 every sigma below 0.1% scored the same, so nothing pulled sigma toward s0 (D37, D38)."""
    from physcal.verify.scoring import sigma_floor

    rng = np.random.default_rng(5)
    s0 = 3e-4
    ys = np.exp(rng.normal(0.0, s0, 20_000))
    grid = np.linspace(1e-4, 6e-4, 26)

    def mean_score(cfg, sig):
        return np.mean([score("crps_log", (1.0 - y) / y, max(sig / y, sigma_floor("crps_log", cfg)), cfg) for y in ys])

    new = [mean_score(SC, c) for c in grid]
    assert abs(grid[int(np.argmax(new))] - s0) <= grid[1] - grid[0]
    assert new[0] < max(new) and new[-1] < max(new)
    legacy = {k: v for k, v in SC.items() if k != "crps_log_sigma_floor"}
    assert len({mean_score(legacy, c) for c in grid}) == 1

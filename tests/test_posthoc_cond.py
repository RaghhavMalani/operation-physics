"""Conditional post-hoc baselines (D49): the text-feature scale and the oracle per-cell scale recover known
mis-scalings on synthetic data, the gradient is right, and the global scale is the intercept-only special case."""

import numpy as np
import pytest
import scipy.sparse as sp

from physcal.config import load_config
from physcal.eval import posthoc
from physcal.eval.posthoc_cond import (TextScale, _log_terms, anchored, fit_text_scale, loss_and_grad,
                                       oracle_cell_scales, oracle_scales, with_sigma_eff)
from physcal.verify.reward import reward_config
from physcal.verify.scoring import sigma_floor

SC = reward_config()["scoring"]
CFG = load_config("posthoc")
TXT, FIT = CFG["text_feature"], CFG["fit"]
F = sigma_floor("crps_log", SC)


def _records(groups: dict[str, float], n: int = 600, seed: int = 0, stated: float = 0.01) -> list[dict]:
    """Per group g: stated relative σ = `stated`, actual log-error spread = groups[g] × 1%. The group is visible in the
    completion text (token `route_<g>`) and is also the cell; filler words are shared by every group."""
    rng = np.random.default_rng(seed)
    out = []
    for gi, (g, k) in enumerate(groups.items()):
        for i, z in enumerate(rng.normal(size=n)):
            mu = float(np.exp(k * 0.01 * z))
            out.append({"split": "x", "key": (g, 1, gi * n + i), "cell": (g, 1),
                        "problem_text": f"a cart on a track problem {i % 7}",
                        "completion": f"compute the value route_{g} then answer {i % 5}",
                        "mu": mu, "sigma": stated * mu, "truth": 1.0})
    return out


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    n, d = 50, 6
    X = sp.csr_matrix((rng.random((n, d)) < 0.4).astype(float))
    L = rng.normal(scale=0.05, size=n)
    a = np.exp(rng.normal(np.log(0.01), 1.0, size=n))
    live = np.ones(n, bool)
    theta = rng.normal(scale=0.5, size=d + 1)
    _, g = loss_and_grad(theta, X, L, a, live, 1e-3, F, SC["crps_clip"])
    eps = 1e-6
    num = [(loss_and_grad(theta + eps * e, X, L, a, live, 1e-3, F, SC["crps_clip"])[0]
            - loss_and_grad(theta - eps * e, X, L, a, live, 1e-3, F, SC["crps_clip"])[0]) / (2 * eps)
           for e in np.eye(d + 1)]
    assert np.allclose(g, num, rtol=1e-4, atol=1e-10)


def test_vectorized_loss_matches_the_verifier():
    recs = _records({"a": 3.0}, n=200)
    L, a, live = _log_terms(recs)
    X0 = sp.csr_matrix((len(recs), 0))
    for c in (0.5, 1.0, 7.0):
        vec = loss_and_grad(np.array([np.log(c)]), X0, L, a, live, 0.0, F, SC["crps_clip"])[0]
        assert vec == pytest.approx(-posthoc.summary(recs, c, SC)["mean_crps_log"], rel=1e-9)


def test_text_scale_recovers_per_group_scales_and_beats_the_global_scale():
    groups = {"a": 1.0, "b": 20.0}
    fit_recs, test_recs = _records(groups, seed=0), _records(groups, seed=1)
    m = fit_text_scale(fit_recs, SC, TXT, FIT, lam=1e-6)
    sc_test = m.scales(test_recs)
    for g, k in groups.items():
        got = np.median([c for c, r in zip(sc_test, test_recs) if r["cell"][0] == g])
        assert got == pytest.approx(k, rel=0.15)
    glob = posthoc.fit_crps_scale(fit_recs, SC, FIT)
    assert posthoc.summary(test_recs, sc_test, SC)["mean_crps_log"] > posthoc.summary(test_recs, glob, SC)["mean_crps_log"]
    cov = posthoc.summary([r for r in test_recs if r["cell"][0] == "b"], sc_test[len(test_recs) // 2:], SC)
    assert cov["coverage_1sigma"] == pytest.approx(0.6827, abs=0.06)


def test_heavy_penalty_gives_the_global_scale():
    recs = _records({"a": 1.0, "b": 20.0}, n=300)
    m = fit_text_scale(recs, SC, TXT, FIT, lam=1e6)
    assert np.allclose(m.scales(recs), m.scales(recs)[0])
    assert m.scales(recs)[0] == pytest.approx(posthoc.fit_crps_scale(recs, SC, FIT), rel=0.01)


def test_zero_sigma_answers_get_a_scale_through_sigma_eff():
    """A group that states σ = 0 is rescaled from the crps_log floor when its text identifies it."""
    recs = _records({"a": 1.0, "z": 5.0}, n=500)
    for r in recs:
        if r["cell"][0] == "z":
            r["sigma"] = 0.0
    eff = with_sigma_eff(recs, F)
    assert all(r["sigma"] > 0 for r in eff)
    m = fit_text_scale(eff, SC, TXT, FIT, lam=1e-8)
    z = [(r, c) for r, c in zip(eff, m.scales(eff)) if r["cell"][0] == "z"]
    s = posthoc.summary([r for r, _ in z], np.array([c for _, c in z]), SC)
    assert s["coverage_1sigma"] == pytest.approx(0.6827, abs=0.08)
    assert posthoc.summary(recs, 1e6, SC)["coverage_1sigma"] < 0.55  # without σ_eff no scale helps the σ = 0 group


def test_json_round_trip():
    recs = _records({"a": 1.0, "b": 4.0}, n=100)
    m = fit_text_scale(recs, SC, TXT, FIT, lam=1e-4)
    m2 = TextScale.from_json(m.to_json())
    assert np.allclose(m.scales(recs), m2.scales(recs), rtol=1e-12)


def test_oracle_per_cell_recovers_cells_and_upper_bounds_the_global_scale():
    groups = {"a": 0.5, "b": 3.0, "c": 40.0}
    recs = _records(groups, n=1500)
    cells = oracle_cell_scales(recs, SC, FIT)
    for g, k in groups.items():
        assert cells[(g, 1)] == pytest.approx(k, rel=0.1)
    best_global = posthoc.summary(recs, posthoc.fit_crps_scale(recs, SC, FIT), SC)["mean_crps_log"]
    assert posthoc.summary(recs, oracle_scales(recs, SC, FIT), SC)["mean_crps_log"] >= best_global


def test_per_record_scales_equal_a_scalar_scale():
    recs = _records({"a": 2.0}, n=100)
    assert posthoc.summary(recs, np.full(len(recs), 3.0), SC) == posthoc.summary(recs, 3.0, SC)


def test_direct_form_ignores_the_stated_sigma():
    """(a'): σ' = c(x)·|μ|, so a stated σ that is pure noise does not hurt it, while it does hurt the scale form."""
    rng = np.random.default_rng(3)
    fit_recs, test_recs = _records({"a": 1.0, "b": 20.0}, seed=0), _records({"a": 1.0, "b": 20.0}, seed=1)
    for r in fit_recs + test_recs:
        r["sigma"] = r["mu"] * 0.01 * float(np.exp(rng.normal(0, 2)))  # stated σ uninformative, spread ×e^±2
    assert all(r["sigma"] == abs(r["mu"]) for r in anchored(test_recs, "answer", F))
    with pytest.raises(ValueError):
        anchored(test_recs, "nonsense", F)
    scores = {}
    for anchor in ("stated_sigma", "answer"):
        m = fit_text_scale(anchored(fit_recs, anchor, F), SC, TXT, FIT, lam=1e-6)
        t = anchored(test_recs, anchor, F)
        scores[anchor] = posthoc.summary(t, m.scales(t), SC)["mean_crps_log"]
    assert scores["answer"] > scores["stated_sigma"]

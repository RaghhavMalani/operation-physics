"""Phase 12 stage A (D100): the optimum search, the paired resolution and the ranking, on synthetic data."""

import math

import numpy as np

from physcal.config import load_config
from physcal.eval import stage_a
from physcal.verify.reward import reward_config


def _cfg():
    cfg = load_config("stage_a")
    cfg["bootstrap"] = 200
    return cfg


def test_optimum_finds_the_generating_scale():
    rng = np.random.default_rng(0)
    rows = [{"depth": 1 + i % 2, "level": 0.02, "exact": False, "sigma": s, "dev": rng.normal(0, s)}
            for i, s in enumerate(np.exp(rng.uniform(math.log(0.005), math.log(0.05), 4000)))]
    rows.append({"depth": 1, "level": 0.0, "exact": True, "sigma": 0.0, "dev": 0.0})  # noise 0: left out
    out = stage_a.optimum(rows, _cfg(), reward_config()["scoring"])
    assert out["n_problems"] == 4000 and out["vectorized_vs_scalar_max_abs"] < 1e-12
    lo, hi = out["pooled"]["argmax_c_ci95"]
    assert lo <= 1.0 <= hi and 0.9 < out["pooled"]["argmax_c"] < 1.1
    assert out["pooled"]["loss_at_half_vs_1"] > 0 and out["pooled"]["loss_at_2_vs_1"] > 0
    assert set(out["cells"]) == {"1|0.02", "2|0.02"}


def test_resolution_and_ranking():
    sets = np.array(["a", "a", "b", "b"])
    star = {"realized": np.array([-1.0, -1.0, -2.0, -2.0]), "expected": np.array([-1.0, -1.0, -2.0, np.nan])}
    alt = {"realized": np.array([-1.5, -0.5, -3.0, -2.5]), "expected": np.array([-1.2, -1.1, -2.2, -2.1])}
    res = stage_a.resolution({stage_a.STAR: star, "alt": alt}, sets)["alt"]
    assert math.isclose(res["pooled"]["realized"]["mean_diff"], 0.375)
    assert res["pooled"]["realized"]["share_star_better"] == 0.75
    assert res["pooled"]["expected"]["n"] == 3 and res["b"]["expected"]["n"] == 1  # the NaN expected score is left out
    summ = {"s": {"realized": (-1.0, None), "median_abs_log_ratio": 0.0},
            "x": {"realized": (-2.0, None), "median_abs_log_ratio": 0.5},
            "y": {"realized": (-3.0, None), "median_abs_log_ratio": 0.9}}
    rk = stage_a.ranking(summ)
    assert rk["by_mean_realized_score"] == ["s", "x", "y"] == rk["by_median_abs_log_ratio"]
    assert math.isclose(rk["kendall_tau"], 1.0)

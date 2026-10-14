"""D50/A5 nested baselines on synthetic completions: the feature set that carries the signal wins, metadata
cannot see text, and the fitted σ scale makes coverage sensible."""

import math

import numpy as np

from physcal.config import load_config
from physcal.eval.nested_baselines import Candidate, log_error, select, sigma_summary
from physcal.verify.reward import reward_config

SC = reward_config()["scoring"]
FIT = load_config("posthoc")["fit"]


def _cfg():
    cfg = load_config("nested_baselines")
    cfg["grids"] = {"ridge_alpha": [1.0], "logistic_C": [1.0],
                    "gbt": [{"max_iter": 50, "max_depth": 2, "learning_rate": 0.1, "min_samples_leaf": 10}]}
    return cfg


def _records(n=400, seed=0):
    """Error size is set by a word in the reasoning ("carefully" = small, "roughly" = large); the problem text
    and the cell carry no signal. 40 templates, so grouped CV has groups to hold out."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        careful = i % 2 == 0
        err = rng.normal(0, 0.001 if careful else 0.2)
        out.append({"split": "x", "key": ("kinematics", 1 + (i // 2) % 4, i), "family": "kinematics", "depth": 1 + (i // 2) % 4,
                    "template": f"t{i % 40}", "problem": f"a cart problem number {i % 13}",
                    "answer": f"{math.exp(err):.4f} ± 0 m", "reasoning": ("carefully" if careful else "roughly")
                    + f" computed step {i % 7}", "ok": True, "mu": math.exp(err), "truth": 1.0,
                    "stated_sigma": 0.0, "correct": abs(math.exp(err) - 1) <= 0.01})
    return out


def test_the_feature_set_with_the_signal_is_strongest():
    cfg = _cfg()
    table = select(_records(), SC, cfg, FIT)
    cv = {k: v["cv"]["mean_crps_log"] for k, v in table["sigma"].items()}
    assert table["strongest"]["sigma"].startswith("problem_answer")  # the answer or the reasoning carries it
    assert cv["problem_answer_reasoning/linear"] > cv["metadata/linear"] + 0.01
    assert table["p"]["problem_answer_reasoning/linear"]["cv"]["brier"] < table["p"]["metadata/linear"]["cv"]["brier"]


def test_refit_sigma_is_calibrated_out_of_sample():
    cfg = _cfg()
    fit, test = _records(seed=0), _records(seed=1)
    table = select(fit, SC, cfg, FIT)
    entry = table["sigma"]["problem_answer_reasoning/linear"]
    cand = Candidate("problem_answer_reasoning", "linear", "sigma", entry["hp"], cfg).fit(fit, log_error(fit, 1e-6))
    cand.c = entry["c"]
    s = sigma_summary(test, cand.sigma_log(test), SC)
    assert s["spearman_sigma_error"] > 0.5 and 0.4 < s["coverage_1sigma"] < 0.95

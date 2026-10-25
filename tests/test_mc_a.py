"""R4-4 (DECISIONS.md D113 §7): MC-A's perturbations, step names, estimators and AUROC. CPU only."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.eval import mc_a

ENV = load_config("env_v2")
A6C = REPO_ROOT / "data" / "eval_sets" / "2.0.0+bdbb2279" / "a6c_pairs.jsonl"


def _problems(n=6):
    return [json.loads(line) for line in list(A6C.open())[:n]]


def test_perturb_is_deterministic_replaces_each_noisy_number_and_matches_the_chain():
    from physcal.env import noise as noise_mode
    from physcal.env.formulas import get_family
    from physcal.env.generator import chain_of
    from physcal.env.units import to_si

    cfg = mc_a._config()
    for p in _problems():
        a, b = mc_a.perturb(p, "a6c_pairs", 3, cfg, ENV), mc_a.perturb(p, "a6c_pairs", 3, cfg, ENV)
        assert a == b and a != mc_a.perturb(p, "a6c_pairs", 4, cfg, ENV)
        si = {g["symbol"]: g["value_si"] for g in p["givens"]}
        for g in p["givens"]:
            if g.get("noise_rel"):
                new = mc_a._shown_text(a["displays"][g["symbol"]], g["unit"], g["noise_rel"])
                assert new in a["text"] and mc_a._shown_text(g["display"], g["unit"], g["noise_rel"]) not in a["text"]
                si[g["symbol"]] = to_si(float(a["displays"][g["symbol"]]), g["unit"])
        y = noise_mode.chain_target(get_family(p["family"]), chain_of(p), si)
        assert a["ln_oracle"] == pytest.approx(math.log(float(y)))
        assert a["text"].replace(" ", "") != p["text"].replace(" ", "")


def test_perturbations_follow_the_predictive_draw():
    """x = x_meas / (1 + s z): over many runs, ln(x_meas / x) has SD ≈ s."""
    cfg = mc_a._config()
    p = _problems(1)[0]
    g = next(g for g in p["givens"] if g.get("noise_rel"))
    from physcal.env.units import to_si
    lr = [math.log(g["value_si"] / to_si(float(mc_a.perturb(p, "a6c_pairs", j, cfg, ENV)["displays"][g["symbol"]]),
                                         g["unit"])) for j in range(1, 401)]
    assert np.std(lr) == pytest.approx(g["noise_rel"], rel=0.15)


def test_step_names():
    comp = ("SI values: t1 = 11.5 s; t3 = 0.337 min = 20.22 s.\nNot needed: len_cart.\n"
            "t2 = -t1 - t3 + t_tot = -11.5 - 20.22 + 59.16 = 27.44 s\nIn min: t2 = 0.4573 min\n"
            "v = a1*t1 = 3 m/s\n<steps>\nx = 1 m\n</steps>")
    assert mc_a.step_names(comp) == ("t2", "v")


def test_estimates_rules():
    cfg = {"min_runs": 2, "mad_scale": 1.4826}
    r0 = {"completion": "a = 1\nb = 2\n<steps>", "breakdown": {"reason": "ok", "mu_si": 1.0}}
    same, other = "a = 1\nb = 2\n<steps>", "a = 1\nc = 2\n<steps>"
    runs = [{"j": 1, "reason": "ok", "mu_si": 1.0, "ln_oracle": 0.0, "completion": same},
            {"j": 2, "reason": "ok", "mu_si": math.e, "ln_oracle": 0.1, "completion": same},
            {"j": 3, "reason": "ok", "mu_si": math.e ** 5, "ln_oracle": 0.2, "completion": other},
            {"j": 4, "reason": "parse_fail", "mu_si": None, "ln_oracle": 0.3, "completion": same}]
    e = mc_a.estimates(r0, runs, 4, cfg)
    assert e["raw"] == pytest.approx(np.std([0, 1, 5], ddof=1))
    assert e["robust"] == pytest.approx(1.4826 * 0.5)
    assert e["oracle"] == pytest.approx(np.std([0, 0.1, 0.2, 0.3], ddof=1))
    assert (e["n_invalid"], e["n_filtered"]) == (1, 1)
    assert mc_a.estimates(r0, runs, 2, cfg)["robust"] == pytest.approx(1.4826 * 0.5)
    bad0 = {**r0, "breakdown": {"reason": "parse_fail"}}
    assert mc_a.estimates(bad0, runs, 4, cfg)["robust"] is None
    assert mc_a.estimates(r0, runs[:1], 1, cfg)["raw"] is None


def test_auroc():
    s = np.array([0.1, 0.2, 0.3, 0.4])
    assert mc_a.auroc(s, np.array([False, False, True, True])) == 1.0
    assert mc_a.auroc(s, np.array([True, True, False, False])) == 0.0
    assert mc_a.auroc(np.ones(4), np.array([True, False, True, False])) == 0.5
    assert mc_a.auroc(s, np.zeros(4, bool)) is None

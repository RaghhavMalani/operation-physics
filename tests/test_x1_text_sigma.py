"""X1 (D73): the text parser, the features and the pipeline references (CPU, small)."""

import math

import numpy as np
import pytest

from physcal.config import load_config
from physcal.env.generator import generate_problem
from physcal.eval import x1_text_sigma as X

V2 = load_config("env_v2")


def test_parser_reads_every_shown_measurement_and_no_exact_value():
    for i, (f, d, lv) in enumerate([("kinematics", 2, 0.02), ("energy", 3, 0.05), ("dynamics", 4, 0.01)]):
        p = generate_problem(f, d, 53000000 + i, split="train", cfg=V2, noise=lv).to_dict()
        meas = {m["symbol"]: m for m in X.parse_measurements(p["text"])}
        shown = p["givens"] + (p.get("distractors") or [])
        assert set(meas) == {g["symbol"] for g in shown}
        for g in shown:
            assert meas[g["symbol"]]["si"] == pytest.approx(g["value_si"], rel=1e-12)
            assert meas[g["symbol"]]["p"] == pytest.approx(lv)
        assert "g" not in meas  # "Use g = 9.81 m/s^2" is exact, not a measurement


def test_parser_formats():
    t = "v = 30.506 km/h ± 2%. W_f, equals 6414.5 mJ ± 5%. mu = 0.25 ± 1%. p = 3e2 kg m/s ± 10%. Use g = 9.81 m/s^2."
    m = {x["symbol"]: x for x in X.parse_measurements(t)}
    assert set(m) == {"v", "W_f", "mu", "p"}
    assert m["v"]["si"] == pytest.approx(30.506 / 3.6) and m["W_f"]["si"] == pytest.approx(6.4145)
    assert m["mu"]["si"] == pytest.approx(0.25) and m["p"]["si"] == pytest.approx(300) and m["p"]["p"] == 0.10


def test_feature_sets_nest():
    meas = [{"symbol": "m", "si": 2.0, "p": 0.02}, {"symbol": "v", "si": 3.0, "p": 0.02}]
    noise = X.numeric_features(meas, ["m", "v", "x"], False)
    full = X.numeric_features(meas, ["m", "v", "x"], True)
    assert noise == full[:4] and noise[0] == 2 and noise[3] == pytest.approx(math.sqrt(2) * 0.02)
    assert full[-3:-1] == [math.log(2.0), math.log(3.0)] and math.isnan(full[-1])


def _pairs(n):
    from physcal.config import REPO_ROOT

    cfg = X._config()
    a6 = X.load_pairs(cfg["pairs"]["a6"])[:n]
    a6b = X.load_pairs(cfg["pairs"]["a6b"])[:n]
    assert REPO_ROOT and len(a6) == n
    return a6, a6b


def test_references_exact_propagation_and_constant_sigma():
    a6, a6b = _pairs(40)
    star = lambda ps: np.array([math.log(d["noise"]["sigma_star_log"]) for d in ps])
    ex = X.pair_analysis(a6, a6b, star(a6), star(a6b))
    assert ex["A6"]["primary_computation_correct"]["slope"] == pytest.approx(1.0)
    assert ex["A6"]["n_pairs_total"] == 20 and ex["joint"]["labelled_verdict"] == "full propagation"
    assert ex["A6b"]["equivalent_to_zero"]
    const = X.pair_analysis(a6, a6b, np.full(40, -4.0), np.full(40, -4.0))
    assert const["A6"]["primary_computation_correct"]["slope"] == pytest.approx(0.0, abs=1e-12)
    assert const["joint"]["A6_label"] == "no propagation"


def test_round_robin_set_is_reproducible_and_disjoint():
    cfg = X._config()
    part = {**cfg["train"], "n": 36, "seed_offset": 54000000}
    a, b = X.build_set(part, cfg["env_config"], 2), X.build_set(part, cfg["env_config"], 1)
    assert a == b
    cells = {(d["family"], d["depth"], d["noise"]["level"]) for d in a}
    assert len(cells) == 36 and {d["depth"] for d in a} == {1, 2, 3, 4}

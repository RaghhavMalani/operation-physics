"""D151 (exploratory follow-up track): the answer parser, the adaptive-sampling simulator and the line calculator."""

import math

import numpy as np
import pytest

from physcal.eval import followup_arith as FA
from physcal.eval import followup_scale as FS
from physcal.eval import followup_ttc as FT


# ------------------------------------------------------------------------------------------- item 3: the answer
@pytest.mark.parametrize("content, unit, value, reason", [
    ("54.4", "cm/s^2", 54.4, "ok_no_unit"),
    (r"5.44 \times 10^{-1} \, \text{m/s}^2", "cm/s^2", 54.4, "ok_converted"),
    (r"b = 1,234.5 \mathrm{m}", "km", 1.2345, "ok_converted"),
    (r"\frac{1}{4}", "m", 0.25, "ok_no_unit"),
    (r"12 \text{ kg}", "m/s", None, "unit_mismatch"),
    (r"2.5 \text{ meters per second}", "m/s", 2.5, "ok_unit_unparsed"),
    ("no number here", "m", None, "no_number"),
])
def test_parse_boxed(content, unit, value, reason):
    p = FS.parse_boxed(content, unit)
    assert p["reason"] == reason
    assert (p["value"] is None) if value is None else math.isclose(p["value"], value, rel_tol=1e-9)


def test_last_boxed_takes_the_last_balanced_box():
    s = r"so \boxed{\frac{1}{2}} and \boxed{54.7 \text{ cm/s}^{2}} end"
    content, brace = FS.last_boxed(s)
    assert content == r"54.7 \text{ cm/s}^{2}" and s[brace] == "}" and s[brace + 1:] == " end"
    assert FS.last_boxed(r"\boxed{3.2 \text{m") is None and FS.last_boxed("none") is None


def test_grade_counts_truncation_and_missing_box_as_no_answer():
    p = {"target_value": 10.0, "target_unit": "m"}
    assert FS.grade(p, r"\boxed{10.05}", False, 0.01)["correct"] is True
    assert FS.grade(p, r"\boxed{10.2}", False, 0.01)["correct"] is False
    assert FS.grade(p, r"\boxed{10}", True, 0.01)["value"] is None
    assert FS.grade(p, "10 m", False, 0.01)["reason"] == "no_box"


def test_grade_reads_a_unit_right_after_the_box():
    p = {"target_value": 9574.4, "target_unit": "ms"}
    g = FS.grade(p, r"Thus \(\boxed{9.55}\) seconds.", False, 0.01)
    assert g["reason"] == "ok_unit_after_box" and math.isclose(g["value"], 9550.0) and g["correct"] is True
    assert FS.grade(p, "\\boxed{9.55} \\]\nThe answer is in seconds.", False, 0.01)["reason"] == "ok_no_unit"
    assert FS.grade(p, r"\boxed{9570} and so on", False, 0.01)["reason"] == "ok_no_unit"
    km = {"target_value": 0.2295, "target_unit": "km"}
    assert math.isclose(FS.grade(km, r"\boxed{237.0} \, \text{m} \]", False, 0.01)["value"], 0.237)


def test_native_text_drops_every_steps_sentence():
    for t in ("What is v? Give your answer in m/s. Your <steps> must include the intermediate values t1, t3.",
              "What is v? Give your answer in m/s. Show your calculation first; no intermediate values need to be "
              "listed in <steps>."):
        assert FS.native_text(t) == "What is v? Give your answer in m/s."


# ------------------------------------------------------------------------------------------- item 1: the simulator
def _problem(mu, correct, tokens=None, sigma=None):
    k = len(mu)
    return {"index": 0, "mu": np.array(mu, float), "correct": np.array(correct),
            "sigma": np.array(sigma if sigma is not None else [np.nan] * k, float),
            "tokens": np.array(tokens if tokens is not None else [10.0] * k)}


def test_majority_with_ties_to_the_earliest_and_unparsed_answers():
    p = _problem([1.0, 2.0, 2.0, np.nan], [True, False, False, False])
    st = FT.prefix_states(p, np.array([[0, 1, 2, 3]]), 0.01)
    assert st["ok"][0].tolist() == [True, True, False, False]   # m = 2: a 1-1 tie goes to the earlier draw
    assert st["top"][0].tolist() == [1, 1, 2, 2] and st["second"][0].tolist() == [0, 1, 1, 1]
    assert st["tokens"][0].tolist() == [10, 20, 30, 40]


def test_stop_rules():
    p = _problem([1.0, 1.001, 5.0, 5.0, 5.0, 7.0, 8.0, 9.0], [True] * 2 + [False] * 6, sigma=[1e-4] + [1.0] * 7)
    st = FT.prefix_states(p, np.arange(8)[None, :], 0.01)
    two = FT.stop_index(st, {"kind": "two_then_all", "first": 2}, 8, {})
    b3 = FT.stop_index(st, {"kind": "sequential", "first": 2, "votes": 3}, 8, {})
    gate = FT.stop_index(st, {"name": "D", "kind": "sigma_gate", "then_votes": 2}, 8, {"D": 1e-3})
    assert two.tolist() == [2] and b3.tolist() == [5] and gate.tolist() == [1]
    assert FT.stop_index(st, {"kind": "no_loss"}, 8, {}).tolist() == [8]


def test_matched_saving_interpolates_the_monotone_curve():
    acc, tok = np.array([0.3, 0.4, 0.38, 0.5]), np.array([100.0, 200.0, 300.0, 400.0])
    (t1, f1), (t2, f2) = FT.t_fix(acc, tok, 0.35), FT.t_fix(acc, tok, 0.45)
    assert math.isclose(t1, 150.0) and f1 == "ok"
    assert math.isclose(t2, 350.0) and f2 == "ok"                 # the dip at k = 3 is flattened
    assert FT.t_fix(acc, tok, 0.6)[1] == "above_curve" and FT.t_fix(acc, tok, 0.2)[1] == "below_k1"
    assert math.isclose(FT.saving(0.35, 75.0, acc, tok)[0], 0.5)


# ------------------------------------------------------------------------------------------- item 2: the calculator
COMPLETION = """SI values: s1 = 231.15 m; t2 = 0.11032 min = 6.619 s; u = 338.46 cm/s = 3.385 m/s; v = 30.593 m/s.
t1 = 2*s1/(u + v) = 2*231.1/(3.385 + 30.59) = 13.09 s
S_s1 = t1/s1 = 13.09/231.15 = 0.0566
S_v = 2 = 2
<steps>
sigma_rel = sqrt((0.06*0.01)^2 + (2*0.02)^2) = 0.04
</steps>
<answer>1 ± 0.04 m</answer>"""


def test_model_values_and_reevaluation():
    vals, der = FA.model_values(COMPLETION)
    assert vals == {"s1": 231.15, "t2": 6.619, "u": 3.385, "v": 30.593, "t1": 13.09}
    assert der == [("t1", "2*s1/(u + v)")]
    e = FA.reevaluate(COMPLETION, [(0.06, 0.01), (2.0, 0.02)], vals, 12)
    assert e["n_evaluated"] == 2 and math.isclose(e["terms"][0][0], 13.09 / 231.15) and e["terms"][1][0] == 2.0
    ch = FA.chain_values(COMPLETION, 12)
    assert math.isclose(ch["t1"], 2 * 231.15 / (3.385 + 30.593))
    assert FA.reevaluate(COMPLETION, [(0.06, 0.01)], vals, 12) is None   # S lines and σ terms differ in number

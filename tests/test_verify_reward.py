"""reward(): failure handling, units, step credit, weighting, config validation, real problems."""

import copy
import json
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from physcal.env.formulas import FAMILY_NAMES
from physcal.env.generator import generate_problem
from physcal.env.units import from_si
from physcal.verify.reward import reward, reward_config, validate_reward_config
from physcal.verify.scoring import ceiling, floor

CFG = reward_config()
SC = CFG["scoring"]
W = CFG["weights"]

# Synthetic problem: target v = 12.4 m/s; intermediates a1 = 2.45 m/s^2, t1 = 4.1 s.
PROBLEM = {
    "target_symbol": "v",
    "target_value_si": 12.4,
    "target_si_unit": "m/s",
    "target_unit": "km/h",
    "intermediates": {"a1": 2.45, "t1": 4.1},
    "intermediate_si_units": {"a1": "m/s^2", "t1": "s"},
    "givens": [
        {"symbol": "u", "desc": "", "display": "2.36", "unit": "m/s", "value_si": 2.36},
        {"symbol": "s1", "desc": "", "display": "30.0", "unit": "m", "value_si": 30.0},
    ],
    "distractors": [{"symbol": "m_cart", "desc": "", "display": "50.0", "unit": "kg", "value_si": 50.0}],
}
GOOD_STEPS = "a1 = 2.45 m/s^2\nt1 = 4.1 s"


def completion(answer: str, steps: str = GOOD_STEPS) -> str:
    return f"Reasoning...\n<steps>\n{steps}\n</steps>\n<answer>{answer}</answer>"


def test_perfect_answer_scores_near_ceiling():
    r, bd = reward(PROBLEM, completion("12.4 ± 0.0124 m/s"))
    assert bd["reason"] == "ok" and bd["correct"] and bd["score_steps"] == 1.0
    assert bd["score_final"] > -1e-3
    assert math.isclose(r, W["final"] * bd["score_final"] + W["steps"] * 1.0)


def test_other_unit_same_dimension_is_equivalent():
    r_m, _ = reward(PROBLEM, completion("12.4 ± 0.5 m/s"))
    r_kmh, bd = reward(PROBLEM, completion("44.64 ± 1.8 km/h"))
    assert bd["reason"] == "ok"
    assert math.isclose(r_m, r_kmh, rel_tol=1e-9)


@pytest.mark.parametrize("answer", ["12.4 ± 0.5 m", "12.4 ± 0.5 s", "12.4 ± 0.5", "12.4 ± 0.5 blarg"])
def test_unit_mismatch_gets_floor_but_keeps_step_credit(answer):
    r, bd = reward(PROBLEM, completion(answer))
    assert bd["reason"] == "unit_mismatch"
    assert bd["score_final"] == floor(SC["rule"], SC)
    assert bd["score_steps"] == 1.0
    assert math.isclose(r, W["final"] * floor(SC["rule"], SC) + W["steps"])


@pytest.mark.parametrize(
    "text",
    [
        "no tags at all, v = 12.4 m/s",
        completion("12.4 m/s"),  # missing ± under CRPS
        completion("twelve ± 1 m/s"),
        "<answer>12.4 ± 0.8 m/s",
        completion("1 ± 1 m/s") + "<answer>12.4 ± 1 m/s</answer>",
        completion("1e999 ± 1 m/s"),
    ],
)
def test_parse_fail_gets_the_minimum_reward(text):
    r, bd = reward(PROBLEM, text)
    assert bd["reason"] == "parse_fail"
    assert r == W["final"] * floor(SC["rule"], SC)
    # a malformed completion never beats even a wildly wrong but well-formed one
    worst_valid, _ = reward(PROBLEM, completion("1e6 ± 1 m/s", steps=""))
    assert r <= worst_valid


def test_missing_sigma_is_fine_for_rules_without_sigma():
    for rule in ("binary", "abs_rel_error"):
        cfg = copy.deepcopy(CFG)
        cfg["scoring"]["rule"] = rule
        r, bd = reward(PROBLEM, completion("12.45 m/s"), cfg)
        assert bd["reason"] == "ok"
    cfg["scoring"]["rule"] = "binary"
    assert reward(PROBLEM, completion("12.45 m/s"), cfg)[1]["score_final"] == 1.0
    assert reward(PROBLEM, completion("12.6 m/s"), cfg)[1]["score_final"] == 0.0


def test_negative_value_is_scored_not_crashed():
    r, bd = reward(PROBLEM, completion("-12.4 ± 0.5 m/s"))
    assert bd["reason"] == "ok" and not bd["correct"]
    assert math.isclose(bd["rel_error"], -2.0)


def test_scientific_notation_and_percent_sigma():
    _, a = reward(PROBLEM, completion("1.24e1 ± 6.2e-1 m/s"))
    _, b = reward(PROBLEM, completion("12.4 m/s ± 5%"))
    assert a["reason"] == b["reason"] == "ok"
    assert math.isclose(a["sigma_si"], 0.62) and math.isclose(b["sigma_si"], 0.62)
    assert math.isclose(a["score_final"], b["score_final"])


def test_zero_sigma_uses_floor():
    _, bd = reward(PROBLEM, completion("12.5 ± 0 m/s"))
    assert bd["rel_sigma_floored"] == SC["sigma_floor"] and bd["pit"] == 0.0


def test_step_credit():
    steps = "a1 = 2.49 m/s^2\nt_1 = 4.3 s"  # a1 within 2% (1.6%), t1 off by 4.9%
    _, bd = reward(PROBLEM, completion("12.4 ± 0.5 m/s", steps))
    assert bd["score_steps"] == 0.5 and bd["n_steps_correct"] == 1
    _, bd = reward(PROBLEM, completion("12.4 ± 0.5 m/s", "a1 = 245 cm/s^2\nt1 = 4100 ms"))
    assert bd["score_steps"] == 1.0
    _, bd = reward(PROBLEM, completion("12.4 ± 0.5 m/s", "a1 = 2.45 m/s\nt1 = 4.1 s"))  # wrong unit on a1
    assert bd["score_steps"] == 0.5 and bd["steps"]["a1"]["error"] == "unit_mismatch"
    _, bd = reward(PROBLEM, completion("12.4 ± 0.5 m/s", "t1 = 4.1 s\nextra = 3 m"))
    assert bd["score_steps"] == 0.5 and bd["steps"]["a1"]["error"] == "missing"
    _, bd = reward(PROBLEM, "<answer>12.4 ± 0.5 m/s</answer>")
    assert bd["score_steps"] == 0.0 and not bd["steps_block_found"]


def test_no_requested_steps_uses_empty_score():
    p = dict(PROBLEM, intermediates={}, intermediate_si_units={})
    _, bd = reward(p, completion("12.4 ± 0.5 m/s", ""))
    assert bd["score_steps"] == CFG["steps"]["empty_score"]


def test_steps_never_outweigh_the_final_answer():
    rule = SC["rule"]
    perfect_no_steps, _ = reward(PROBLEM, completion("12.4 ± 0.0124 m/s", ""))
    failed_all_steps, _ = reward(PROBLEM, completion("12.4 ± 0.5 kg"))
    assert perfect_no_steps > failed_all_steps
    bad = copy.deepcopy(CFG)
    bad["weights"]["steps"] = (ceiling(rule, SC) - floor(rule, SC)) * W["final"] + 0.1
    with pytest.raises(ValueError):
        validate_reward_config(bad)


def test_calibration_fields():
    _, bd = reward(PROBLEM, completion("13.0 ± 0.5 m/s"))
    assert not bd["within_1sigma"] and bd["within_2sigma"]
    assert 0.0 < bd["pit"] < 0.5
    assert math.isclose(bd["z"], (0.6 / 12.4) / (0.5 / 12.4))


def test_copied_given_detection():
    _, bd = reward(PROBLEM, completion("2.36 ± 0.1 m/s"))
    assert bd["copied_given"]
    _, bd = reward(PROBLEM, completion("30.0 ± 1 km/h"))  # the number of a given (s1 = 30.0 m)
    assert bd["copied_given"]
    _, bd = reward(PROBLEM, completion("12.4 ± 0.5 m/s"))
    assert not bd["copied_given"]


def test_breakdown_is_json_serializable():
    for text in [completion("12.4 ± 0.5 m/s"), completion("12.4 ± 0.5 kg"), "junk", completion("1e300 ± 1e300 m/s")]:
        _, bd = reward(PROBLEM, text)
        json.dumps(bd, allow_nan=False)


def test_every_rule_reports_all_scores():
    _, bd = reward(PROBLEM, completion("12.0 ± 0.5 m/s"))
    assert set(bd["scores"]) == {"crps", "binary", "log_score", "abs_rel_error", "crps_log"}


@settings(max_examples=60, deadline=None)
@given(family=st.sampled_from(FAMILY_NAMES), depth=st.integers(1, 5), seed=st.integers(0, 10**9))
def test_truthful_completion_on_generated_problems(family, depth, seed):
    """An oracle completion written in the requested units earns full step credit and a correct answer."""
    p = generate_problem(family, depth, seed, split="eval")
    lines = [f"{s} = {v:.6g} {p.intermediate_si_units[s]}" for s, v in p.intermediates.items()]
    sigma = 0.01 * p.target_value
    text = completion(f"{p.target_value:.6g} ± {sigma:.3g} {p.target_unit}", "\n".join(lines))
    r, bd = reward(p, text)
    assert bd["reason"] == "ok" and bd["correct"] and bd["within_1sigma"]
    assert bd["score_steps"] == (1.0 if p.intermediates else CFG["steps"]["empty_score"])
    # and a dimensionally wrong answer is caught
    wrong = "kg" if p.target_si_unit != "kg" else "s"
    _, bd = reward(p, completion(f"{p.target_value:.6g} ± {sigma:.3g} {wrong}"))
    assert bd["reason"] == "unit_mismatch"

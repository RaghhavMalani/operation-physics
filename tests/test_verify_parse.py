"""Parser and units gate: hand-written edge cases."""

import math

import pytest

from physcal.prompt import SYSTEM_PROMPT
from physcal.verify.parse import ParseError, canonical_symbol, normalize, parse_completion, parse_value
from physcal.verify.units import UnitMismatch, to_si_checked


def wrap(answer: str, steps: str = "") -> str:
    return f"Some reasoning.\n<steps>\n{steps}\n</steps>\n<answer>{answer}</answer>"


@pytest.mark.parametrize(
    "text, value, sigma, unit",
    [
        ("12.4 ± 0.8 m/s", 12.4, 0.8, "m/s"),
        ("12.4 +/- 0.8 m/s", 12.4, 0.8, "m/s"),
        ("12.4 +- 0.8 m/s", 12.4, 0.8, "m/s"),
        ("12.4 \\pm 0.8 m/s", 12.4, 0.8, "m/s"),
        ("$12.4 \\pm 0.8\\ \\text{m/s}$", 12.4, 0.8, "m/s"),
        ("(12.4 ± 0.8) m/s", 12.4, 0.8, "m/s"),
        ("v = 12.4 ± 0.8 m/s", 12.4, 0.8, "m/s"),
        ("1.24e1 ± 8e-1 m/s", 12.4, 0.8, "m/s"),
        ("1.24E+01 ± 8.0E-01 m/s", 12.4, 0.8, "m/s"),
        ("1.24 × 10^1 ± 0.8 m/s", 12.4, 0.8, "m/s"),
        ("1.24 x 10^1 ± 8 × 10^-1 m/s", 12.4, 0.8, "m/s"),
        ("1.24×10¹ ± 8×10⁻¹ m/s", 12.4, 0.8, "m/s"),
        ("1.24 \\times 10^{1} \\pm 0.8 m/s", 12.4, 0.8, "m/s"),
        ("(1.24 ± 0.08) × 10^1 m/s", 12.4, 0.8, "m/s"),
        ("(1.24 ± 0.08)×10^-3 m", 1.24e-3, 0.08e-3, "m"),
        ("−12.4 ± 0.8 m/s", -12.4, 0.8, "m/s"),
        ("-1.5e-3 ± 2e-4 kg", -1.5e-3, 2e-4, "kg"),
        ("12.4 ± -0.8 m/s", 12.4, 0.8, "m/s"),
        ("12.4 ± 0.8 m/s.", 12.4, 0.8, "m/s"),
        ("1,234.5 ± 10 m", 1234.5, 10.0, "m"),
        ("9.81 ± 0.05 m/s²", 9.81, 0.05, "m/s^2"),
        ("0.32 ± 0.02", 0.32, 0.02, ""),
        ("0.32 ± 0.02 (dimensionless)", 0.32, 0.02, ""),
        (".5 ± .1 s", 0.5, 0.1, "s"),
        ("12.4 \\pm 0.8 \\, \\mathrm{m\\,s^{-1}}", 12.4, 0.8, "m s^(-1)"),
        ("\\boxed{12.4 \\pm 0.8 \\text{ m/s}}", 12.4, 0.8, "m/s"),
        ("**12.4 ± 0.8 m/s**", 12.4, 0.8, "m/s"),
        ("12.4 ± 0.8 [m/s]", 12.4, 0.8, "m/s"),
    ],
)
def test_answer_forms(text, value, sigma, unit):
    r = parse_value(text)
    assert math.isclose(r.value, value, rel_tol=1e-12)
    assert math.isclose(r.sigma, sigma, rel_tol=1e-12)
    assert r.unit == unit and r.sigma_unit == unit
    assert not r.sigma_is_percent


def test_separate_units_for_value_and_sigma():
    r = parse_value("12.4 km/h ± 800 m/h")
    assert (r.value, r.unit, r.sigma, r.sigma_unit) == (12.4, "km/h", 800.0, "m/h")


def test_unit_only_before_plus_minus():
    r = parse_value("12.4 m/s ± 0.8")
    assert (r.unit, r.sigma_unit) == ("m/s", "m/s")


@pytest.mark.parametrize("text", ["12.4 m/s ± 5%", "12.4 ± 5% m/s"])
def test_percent_sigma(text):
    r = parse_value(text)
    assert r.sigma_is_percent and r.sigma == 5.0 and r.unit == "m/s"


def test_missing_sigma_parses_without_sigma():
    r = parse_value("12.4 m/s")
    assert r.sigma is None and r.unit == "m/s"


@pytest.mark.parametrize("text", ["", "about twelve m/s", "± 0.8 m/s", "12 ± 1 ± 2 m", "m/s"])
def test_malformed_values(text):
    with pytest.raises(ParseError):
        parse_value(text)


@pytest.mark.parametrize(
    "completion",
    [
        "The answer is 12.4 m/s.",  # no tags
        "<answer>12.4 ± 0.8 m/s",  # truncated, no closing tag
        "<answer>1 ± 0.1 m</answer> or maybe <answer>2 ± 0.1 m</answer>",  # hedging
        "<answer></answer>",
    ],
)
def test_malformed_completions(completion):
    with pytest.raises(ParseError):
        parse_completion(completion)


def test_steps_parsing():
    steps = """
    - a1 = 2.45 m/s^2
    2. t_1 = 4.1 s
    v = u + a1*t1 = 12.4 m/s
    s_{tot} ≈ 120 m
    garbage line without equals
    F_net = 3 * m   (unparseable unit is kept; the units gate rejects it later)
    d_tot = not a number
    """
    parsed = parse_completion(wrap("1 ± 0.1 m", steps))
    assert parsed.steps_block_found
    got = {k: (v.value, v.unit) for k, v in parsed.steps.items()}
    assert got["a1"] == (2.45, "m/s^2")
    assert got["t1"] == (4.1, "s")
    assert got["v"] == (12.4, "m/s")
    assert got["stot"] == (120.0, "m")
    assert "dtot" not in got


def test_steps_later_line_wins_and_missing_block_ok():
    parsed = parse_completion(wrap("1 ± 0.1 m", "v = 1 m/s\nv = 2 m/s"))
    assert parsed.steps["v"].value == 2.0
    parsed = parse_completion("<answer>1 ± 0.1 m</answer>")
    assert not parsed.steps_block_found and parsed.steps == {}


def test_canonical_symbol():
    assert canonical_symbol("t_1") == canonical_symbol("t1") == canonical_symbol("t_{1}") == "t1"
    assert canonical_symbol("v_{avg}") == canonical_symbol("v_avg")
    assert canonical_symbol("F") != canonical_symbol("f")


def test_normalize_superscripts_and_minus():
    assert normalize("10⁻³ m s⁻¹") == "10^-3 m s^-1"
    assert normalize("−5") == "-5"


def test_system_prompt_example_parses():
    example = SYSTEM_PROMPT[SYSTEM_PROMPT.index("<steps>\na = "):]
    parsed = parse_completion(example)
    assert parsed.answer.value == 12.4 and parsed.answer.sigma == 0.8 and parsed.answer.unit == "m/s"
    assert parsed.steps["a"].value == 2.45 and parsed.steps["t"].unit == "s"


# ------------------------------------------------------------------ units gate
@pytest.mark.parametrize(
    "value, unit, si_unit, expected",
    [
        (12.4, "m/s", "m/s", 12.4),
        (36.0, "km/h", "m/s", 10.0),
        (1.5, "km", "m", 1500.0),
        (250.0, "g", "kg", 0.25),
        (2.0, "min", "s", 120.0),
        (3.0, "kN", "N", 3000.0),
        (5.0, "N s", "kg m/s", 5.0),
        (5.0, "kg·m/s", "kg m/s", 5.0),
        (2.0, "m/s^2", "m/s^2", 2.0),
        (2.0, "m/s²", "m/s^2", 2.0),
        (4.0, "N/cm", "N/m", 400.0),
        (0.3, "", "", 0.3),
        (12.0, "meters per second", "m/s", 12.0),
    ],
)
def test_units_gate_converts(value, unit, si_unit, expected):
    assert math.isclose(to_si_checked(value, unit, si_unit), expected, rel_tol=1e-12)


@pytest.mark.parametrize(
    "unit, si_unit",
    [("m", "m/s"), ("s", "m"), ("", "m"), ("J", "N"), ("blarg", "m"), ("kg", "N"), ("m/s", "")],
)
def test_units_gate_rejects(unit, si_unit):
    with pytest.raises(UnitMismatch):
        to_si_checked(1.0, unit, si_unit)


def test_units_gate_rejects_scaling_factor_in_unit():
    with pytest.raises(UnitMismatch):
        to_si_checked(1.0, "1000 m", "m")

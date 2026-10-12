"""Problem generator: ground truth vs sympy, determinism, splits, rendering properties."""

import json
import math
import re

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from physcal.env.formulas import FAMILY_NAMES, get_family
from physcal.env.generator import (
    HeldOutError,
    chain_of,
    env_config,
    generate_problem,
    generator_version,
    problem_from_dict,
    train_cells,
)
from physcal.env.render import format_number
from physcal.env.units import from_si, to_si

SEEDS_PER_FAMILY = 1000
REL_TOL = 1e-9
CFG = env_config()


def _sympy_truth(problem):
    """Independent ground truth: compose the chain symbolically and evaluate with sympy at 30 digits."""
    fam = get_family(problem.family)
    chain = chain_of(problem)
    env = {**problem.given_values_si, **fam.si_constants}
    subs = {fam.symbols[k]: v for k, v in env.items()}
    out = {}
    for i in range(1, chain.depth + 1):
        partial = type(chain)(steps=chain.steps[:i], givens=chain.givens)
        out[partial.target] = float(fam.chain_expression(partial).evalf(30, subs=subs))
    return out


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_ground_truth_matches_sympy_1000_seeds(family):
    for seed in range(SEEDS_PER_FAMILY):
        depth = seed % 5 + 1
        p = generate_problem(family, depth, seed, split="eval")
        truth = _sympy_truth(p)
        assert math.isclose(p.target_value_si, truth[p.target_symbol], rel_tol=REL_TOL), (family, seed)
        for sym, val in p.intermediates.items():
            assert math.isclose(val, truth[sym], rel_tol=REL_TOL), (family, seed, sym)
        assert math.isclose(p.target_value, from_si(p.target_value_si, p.target_si_unit, p.target_unit), rel_tol=REL_TOL)


def test_determinism():
    a = generate_problem("energy", 3, 123, split="eval").to_dict()
    b = generate_problem("energy", 3, 123, split="eval").to_dict()
    assert a == b
    c = generate_problem("energy", 3, 124, split="eval").to_dict()
    assert c["text"] != a["text"]


def test_train_split_refuses_heldout_cells():
    with pytest.raises(HeldOutError):
        generate_problem("kinematics", 5, 0, split="train")
    with pytest.raises(HeldOutError):
        generate_problem("optics", 1, 0, split="train")
    generate_problem("kinematics", 4, 0, split="train")


def test_train_cells_exclude_heldout():
    cells = train_cells(CFG)
    assert cells
    assert all(d not in CFG["splits"]["heldout_depths"] for _, d in cells)
    assert all(f not in CFG["splits"]["heldout_families"] for f, _ in cells)


def test_unknown_split_and_depth_rejected():
    with pytest.raises(ValueError):
        generate_problem("kinematics", 1, 0, split="test")
    with pytest.raises(ValueError):
        generate_problem("kinematics", 6, 0, split="eval")


def test_generator_version_tracks_config():
    cfg = env_config()
    v1 = generator_version(cfg)
    cfg["generator"]["display"]["sig_figs"] = 4
    assert generator_version(cfg) != v1
    assert v1.startswith(CFG["generator"]["version"] + "+")


@pytest.mark.parametrize("family", FAMILY_NAMES)
@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_every_cell_generates_varied_problems(family, depth):
    problems = [generate_problem(family, depth, s, split="eval") for s in range(30)]
    assert len({p.target_symbol for p in problems}) >= 2
    assert len({p.text for p in problems}) == len(problems)


@pytest.mark.parametrize(
    "x, expected",
    [
        (12.345, "12.3"),
        (0.012345, "0.0123"),
        (1234.5, "1230"),
        (99999.0, "1.00e5"),
        (0.00012345, "1.23e-4"),
        (9.9951, "10.0"),
        (100.0, "100"),
        (0.5, "0.500"),
    ],
)
def test_format_number(x, expected):
    assert format_number(x, 3, (-2, 4)) == expected


def _check_sig_figs(display: str, sig: int) -> None:
    mantissa = display.split("e")[0].lstrip("-")
    if "." in mantissa:
        assert len(mantissa.replace(".", "").lstrip("0")) == sig, display
    else:  # integers like "5560": trailing zeros are ambiguous, so only bound the count
        assert len(mantissa) >= sig and len(mantissa.rstrip("0")) <= sig, display


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    family=st.sampled_from(FAMILY_NAMES),
    depth=st.integers(1, 5),
    seed=st.integers(0, 2**40),
)
def test_problem_properties(family, depth, seed):
    gen = CFG["generator"]
    p = generate_problem(family, depth, seed, split="eval")
    fam = get_family(family)

    # structure
    assert p.depth == depth and len(p.chain) == depth
    assert list(p.intermediates) == [out for _, out in p.chain[:-1]]
    assert p.chain[-1][1] == p.target_symbol
    assert math.isfinite(p.target_value_si) and p.target_value_si > 0
    assert all(math.isfinite(v) and v > 0 for v in p.intermediates.values())
    assert p.condition_number <= gen["max_condition_number"]

    # honest depth: minimal derivation depth from the givens equals the chain length
    levels = fam.derivation_levels(p.given_values_si)
    assert levels[p.target_symbol] == depth
    for i, sym in enumerate(p.intermediates, start=1):
        assert levels[sym] == i

    # displayed values are exactly the values used for the ground truth
    for g in p.givens + p.distractors:
        assert to_si(float(g["display"]), g["unit"]) == g["value_si"]
        assert f"{g['display']} {g['unit']}".strip() in p.text
        assert re.search(rf"\b{re.escape(g['symbol'])}\b", p.text)
        _check_sig_figs(g["display"], gen["display"]["sig_figs"])

    # distractors are irrelevant and within the configured count
    lo, hi = gen["distractors"]
    assert lo <= len(p.distractors) <= hi
    assert not {d["symbol"] for d in p.distractors} & set(fam.quantities)

    # the question names the target, its unit, and the requested intermediates
    assert p.target_symbol in p.text
    if p.target_unit:
        assert f"in {p.target_unit}" in p.text
    for sym in p.intermediates:
        assert re.search(rf"\b{re.escape(sym)}\b", p.text.split("\n\n")[-1])

    # JSON round trip
    assert problem_from_dict(json.loads(json.dumps(p.to_dict()))) == p


@settings(max_examples=200, deadline=None)
@given(value=st.floats(1e-6, 1e6), family=st.sampled_from(FAMILY_NAMES), data=st.data())
def test_unit_round_trip(value, family, data):
    q = data.draw(st.sampled_from(sorted(get_family(family).quantities.values(), key=lambda q: q.symbol)))
    unit = data.draw(st.sampled_from(q.units))
    back = to_si(from_si(value, q.si_unit, unit), unit)
    assert math.isclose(back, value, rel_tol=1e-12)


def test_zero_intermediate_after_rounding_is_rejected_not_crashed():
    """Regression: for this eval seed a subtraction step (t3 = t_tot - t1 - t2) came out exactly 0 after
    rounding, and the next step divided by it (ZeroDivisionError inside pint). Pinned to generator 1.0.0,
    where the seed triggers the bug (every version draws different problems)."""
    cfg = env_config()
    cfg["generator"]["version"] = "1.0.0"
    p = generate_problem("kinematics", 5, 1001000087, split="eval", cfg=cfg)
    assert p.target_value_si > 0 and all(v > 0 for v in p.intermediates.values())


def test_chain_evaluation_stops_at_zero_intermediate():
    fam = get_family("kinematics")
    chain = next(c for cs in fam.chains(2).values() for c in cs if c.steps[0] == ("K11", "t3"))
    # t3 = t_tot - t1 - t2 = 0 exactly; the next step divides by t3.
    givens = {g: 10.0 for g in chain.givens}
    givens.update(t_tot=30.0, t1=10.0, t2=20.0)
    for evaluate in (fam.evaluate_chain, fam.evaluate_chain_with_units):
        out = evaluate(chain, givens)
        assert all(math.isnan(v) for v in out.values())


def test_condition_number_is_infinite_outside_valid_domain():
    fam = get_family("kinematics")
    chain = next(c for cs in fam.chains(2).values() for c in cs if c.steps[0] == ("K11", "t3"))
    givens = {g: 10.0 for g in chain.givens}
    givens.update(t_tot=30.0, t1=10.0, t2=20.0)
    assert fam.condition_number(chain, givens, 1e-6) == math.inf


def _cfg_for(version: str) -> dict:
    cfg = env_config()
    cfg["generator"]["version"] = version
    return cfg


def test_depth1_wording_is_keyed_by_generator_version():
    from physcal.env.render import NO_STEPS_TEMPLATES

    old = generate_problem("energy", 1, 5, split="eval", cfg=_cfg_for("1.0.0"))
    new = generate_problem("energy", 1, 5, split="eval")
    assert NO_STEPS_TEMPLATES["1.0.0"] in old.text and NO_STEPS_TEMPLATES["1.1.0"] not in old.text
    assert NO_STEPS_TEMPLATES["1.1.0"] in new.text and NO_STEPS_TEMPLATES["1.0.0"] not in new.text
    assert old.generator_version == "1.0.0+afcc2019"  # the version every phase-3 tape was logged with
    assert new.generator_version.startswith("1.1.0+")
    with pytest.raises(ValueError):
        generate_problem("energy", 1, 5, split="eval", cfg=_cfg_for("9.9.9"))

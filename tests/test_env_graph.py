"""Formula graph: physics consistency, dimensions, derivation closure, chain validation."""

import math

import numpy as np
import pytest

from physcal.env.formulas import FAMILY_NAMES, get_family
from physcal.env.graph import Chain
from physcal.env.units import quantity

N_WORLDS = 200
MAX_WORLD_TRIES = 1000
REL_TOL = 1e-9


@pytest.fixture(params=FAMILY_NAMES)
def family(request):
    return get_family(request.param)


def test_every_formula_has_unique_solutions(family):
    for f in family.formulas:
        assert set(f.solved) == set(f.solve_for)
        assert set(f.solve_for) <= set(f.variables)


def test_formulas_are_dimensionally_consistent(family):
    units = {q.symbol: q.si_unit for q in family.quantities.values()}
    units.update({k: u for k, (_, u) in family.constants.items()})
    for f in family.formulas:
        values = {s: quantity(1.7, units[s]) for s in f.variables + f.constants}
        lhs, rhs = f.sides(values)
        dim = lambda x: x.dimensionality if hasattr(x, "dimensionality") else quantity(1.0, "").dimensionality
        assert dim(lhs) == dim(rhs), f.name


def test_solve_directions_have_declared_dimensions(family):
    rng = np.random.default_rng(0)
    world = family.sample_world(rng, MAX_WORLD_TRIES)
    for f in family.formulas:
        for out, solved in f.solved.items():
            chain = Chain(steps=((f.name, out),), givens=f.inputs(out))
            got = family.evaluate_chain_with_units(chain, {g: world[g] for g in chain.givens})
            assert math.isclose(got[out], world[out], rel_tol=REL_TOL), (f.name, out)


def test_worlds_satisfy_every_formula(family):
    """The world is overdetermined; every relation must hold, not just those used to build it."""
    rng = np.random.default_rng(1)
    for _ in range(N_WORLDS):
        world = family.sample_world(rng, MAX_WORLD_TRIES)
        env = {**world, **family.si_constants}
        for f in family.formulas:
            lhs, rhs = f.sides(env)
            assert math.isclose(float(lhs), float(rhs), rel_tol=REL_TOL), f.name


def test_every_solve_direction_recovers_world_value(family):
    rng = np.random.default_rng(2)
    for _ in range(N_WORLDS):
        world = family.sample_world(rng, MAX_WORLD_TRIES)
        env = {**world, **family.si_constants}
        for f in family.formulas:
            for out, solved in f.solved.items():
                assert math.isclose(float(solved(env)), world[out], rel_tol=1e-8), (f.name, out)


def test_worlds_are_positive_and_within_bounds(family):
    rng = np.random.default_rng(3)
    for _ in range(N_WORLDS):
        world = family.sample_world(rng, MAX_WORLD_TRIES)
        assert set(world) == set(family.quantities)
        assert all(v > 0 and math.isfinite(v) for v in world.values())
        assert family.within_bounds(world)


def test_world_plan_covers_every_quantity(family):
    planned = set(family.basis) | {out for _, out in family.world_plan}
    assert planned == set(family.quantities)


def test_distractors_do_not_appear_in_formulas(family):
    used = {v for f in family.formulas for v in f.variables}
    assert not used & set(family.distractors)
    for q in family.distractors.values():
        assert q.range_si is not None


def test_derivation_levels_kinematics():
    fam = get_family("kinematics")
    levels = fam.derivation_levels(["u", "a1", "t1", "t2", "b"])
    assert levels["v"] == 1 and levels["s1"] == 1
    assert levels["s2"] == 2 and levels["t3"] == 2 and levels["s3"] == 2
    assert levels["d_tot"] == 3 and levels["t_tot"] == 3
    assert levels["v_avg"] == 4


def test_chain_with_shortcut_is_rejected():
    fam = get_family("kinematics")
    # s1 from (u, v, t1) after v from (u, a1, t1): but s1 = u t1 + a1 t1^2 / 2 is reachable in one step.
    chain = Chain(steps=(("K1", "v"), ("K2", "s1")), givens=("a1", "t1", "u"))
    assert not fam.is_valid_chain(chain)


def test_overdetermined_givens_are_rejected():
    fam = get_family("kinematics")
    # v is derivable from (u, a1, t1), so giving all four over-determines the problem.
    chain = Chain(steps=(("K6", "s2"),), givens=("a1", "t1", "t2", "u", "v"))
    assert not fam.is_valid_chain(chain)


def test_overdetermined_by_simultaneous_equations_is_rejected():
    """Six givens in a five-degree-of-freedom world: no single given follows from one formula,
    but together they over-determine the cart (found while hand-checking a generated problem)."""
    fam = get_family("kinematics")
    givens = frozenset({"d_tot", "v_avg", "a1", "s1", "t2", "t3"})
    assert all(g not in fam.derivation_levels(givens - {g}) for g in givens)  # the closure misses it
    assert not fam.independent(givens)
    chain = Chain(steps=(("K12", "t_tot"), ("K11", "t1"), ("K5", "v"), ("K6", "s2")), givens=tuple(sorted(givens)))
    assert not fam.is_valid_chain(chain)
    assert fam.independent(frozenset({"d_tot", "s3", "t_tot", "v", "t1"}))


def test_givens_never_exceed_degrees_of_freedom(family):
    for depth in range(1, 6):
        for chains in family.chains(depth).values():
            for c in chains:
                assert len(c.givens) <= len(family.basis)


def test_valid_chain_is_accepted():
    fam = get_family("dynamics")
    chain = Chain(steps=(("D1", "N"), ("D2", "f_k"), ("D3", "F_net")), givens=("F", "m", "mu"))
    assert fam.is_valid_chain(chain)
    assert fam.derivation_levels(chain.givens)["F_net"] == 3


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_sampled_chains_are_valid_and_honest(family, depth):
    rng = np.random.default_rng(depth)
    for _ in range(20):
        chain = family.sample_chain(depth, rng)
        assert chain.depth == depth
        assert family.is_valid_chain(chain)
        levels = family.derivation_levels(chain.givens)
        assert levels[chain.target] == depth
        assert not set(chain.givens) & {out for _, out in chain.steps}
        # every given is consumed by some step of the chain
        consumed = {i for name, out in chain.steps for i in family.formula(name).inputs(out)}
        assert set(chain.givens) <= consumed


def test_chain_expression_matches_evaluation(family):
    rng = np.random.default_rng(5)
    for depth in range(1, 6):
        chain = family.sample_chain(depth, rng)
        world = family.sample_world(rng, MAX_WORLD_TRIES)
        givens = {g: world[g] for g in chain.givens}
        expr = family.chain_expression(chain)
        subs = {family.symbols[k]: v for k, v in {**givens, **family.si_constants}.items()}
        exact = float(expr.evalf(30, subs=subs))
        assert math.isclose(exact, world[chain.target], rel_tol=1e-8)


def test_enumeration_is_exhaustive_and_valid(family):
    """Every enumerated chain is valid; every family has chains at every depth 1..5."""
    for depth in range(1, 6):
        by_target = family.chains(depth)
        assert by_target, (family.name, depth)
        for target, chains in by_target.items():
            for c in chains:
                assert c.target == target and c.depth == depth and family.is_valid_chain(c)


def test_enumeration_contains_hand_built_chain():
    fam = get_family("dynamics")
    chain = Chain(steps=(("D1", "N"), ("D2", "f_k"), ("D3", "F_net")), givens=("F", "m", "mu"))
    assert chain in fam.chains(3)["F_net"]

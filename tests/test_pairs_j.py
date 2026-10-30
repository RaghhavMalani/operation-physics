"""R5-6 part J (D147): the training support, the J test, J sensitivity pairs (A6c on J worlds) and J invariance pairs
(dimensional scaling), their exact replay, and that the existing pair sets are unchanged. CPU only."""

import json
import math
import re

import numpy as np
import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.env import jacobian_support as JS
from physcal.env import pairs_j
from physcal.env.formulas import get_family
from physcal.env.generator import generator_version, regenerate
from physcal.env.noise import log_sensitivities
from physcal.env.pairs import cell_construction, generate_pair

PC = load_config("pairs")
ENV = load_config(PC["env_config"])
STEP = ENV["generator"]["condition_rel_step"]
SUP = JS.load(PC["j_allocation_construction"]["jacobian"])
MARGIN = PC["j_jacobian"]["margin"]
PCT = lambda s: re.sub(r"± \d+%", "± #", s)
MASK = lambda s: re.sub(r"[-+]?\d[\d.]*(e-?\d+)?", "#", s)


def _pair(which, family, depth, level, base=77_000_000):
    con, noise = cell_construction(which, level, PC)
    return next(p for p in (generate_pair(family, depth, base + i, noise, ENV, con) for i in range(40)) if p)


def _shown(p):
    return {g["symbol"]: g["value_si"] for g in p.givens}


def _chain(p):
    fam = get_family(p.family)
    steps = tuple(tuple(s) for s in p.chain)
    return fam, next(c for cs in fam.chains(p.depth).values() for c in cs if c.steps == steps)


# ------------------------------------------------------------------------------------------- the support
def test_support_is_the_training_problems_and_is_checksummed():
    assert SUP["n_problems"] == 3000 and SUP["training_problems"] == PC["r5_6"]["support"]["training_problems"]
    assert sum(c["n"] for c in SUP["chains"].values()) == 3000
    path = REPO_ROOT / SUP["training_problems"]
    if path.exists():  # the R4-1 data is local only (gitignored); the support file is committed
        import hashlib

        assert hashlib.sha256(path.read_bytes()).hexdigest() == SUP["training_problems_sha256"]
        assert JS.build() == SUP  # deterministic
    with pytest.raises(ValueError):
        JS.load({**PC["j_jacobian"], "support_sha256": "0" * 64})


def test_a_training_problem_is_inside_its_own_support():
    """The J test rejects in-support worlds: every training problem's own Jacobian has exceedance 0."""
    path = REPO_ROOT / SUP["training_problems"]
    if not path.exists():
        pytest.skip("the R4-1 data is local only")
    for line in path.read_text().splitlines()[::150]:
        r = json.loads(line)
        fam = get_family(r["family"])
        steps = tuple(tuple(s) for s in r["chain"])
        chain = next(c for cs in fam.chains(r["depth"]).values() for c in cs if c.steps == steps)
        sens = log_sensitivities(fam, chain, {g["symbol"]: g["value_si"] for g in r["givens"]}, STEP)
        assert JS.trained(SUP, r["family"], chain) and JS.exceedance(SUP, r["family"], chain, sens) == (0.0, None)
        assert JS.in_magnitudes(SUP, r["family"], {g["symbol"]: g["value_si"] for g in r["givens"]})


def test_power_law_chains_never_qualify():
    """A chain whose sensitivities are constant has the same Jacobian in every world, so never an exceedance."""
    from physcal.env.pairs import power_law_chains

    fam = get_family("dynamics")
    rng = np.random.default_rng(5)
    for chain in power_law_chains("dynamics", 1, PC["invariance_construction"], ENV["generator"]):
        if not JS.trained(SUP, "dynamics", chain):
            continue
        for _ in range(20):
            w = fam.sample_world(rng, 100)
            sens = log_sensitivities(fam, chain, {g: w[g] for g in chain.givens}, STEP)
            assert JS.exceedance(SUP, "dynamics", chain, sens)[0] < 1e-6


def test_scaling_directions_leave_every_constant_unchanged():
    for family, dirs in PC["j_invariance_construction"]["directions"].items():
        fam = get_family(family)
        for d in dirs:
            for _, unit in fam.constants.values():
                assert float(pairs_j.dim_vector(unit) @ np.array(d, float)) == 0.0
        # kinematics' mass direction scales nothing (no mass in the family), so it is never used; the others do
        assert sum(any(float(pairs_j.dim_vector(q.si_unit) @ np.array(d, float)) != 0 for q in fam.quantities.values())
                   for d in dirs) >= 2


# ------------------------------------------------------------------------------------------- J sensitivity pairs
@pytest.mark.parametrize("family,depth,levels", [("kinematics", 2, [0.01, 0.05]), ("energy", 3, [0.02, 0.05]),
                                                 ("dynamics", 4, [0.01, 0.02]), ("energy", 1, [0.01, 0.05])])
def test_j_allocation_pair_is_an_a6c_pair_on_a_j_world(family, depth, levels):
    lo, hi = _pair("j_a6c", family, depth, levels)
    assert lo.chain == hi.chain and PCT(lo.text) == PCT(hi.text)
    assert [(g["display"], g["unit"]) for g in lo.givens] == [(g["display"], g["unit"]) for g in hi.givens]
    rel = lambda p: {g: v["rel"] for g, v in p.noise["givens"].items()}
    assert sorted(rel(lo).values()) == sorted(rel(hi).values())
    assert lo.noise["pair"]["high_given"] != hi.noise["pair"]["high_given"]
    assert hi.noise["pair"]["delta_ratio"] >= PC["j_allocation_construction"]["min_ratio"]
    assert hi.noise["sigma_star_log"] > lo.noise["sigma_star_log"]
    fam, chain = _chain(lo)
    assert JS.trained(SUP, family, chain) and len(chain.givens) >= 2
    sens = log_sensitivities(fam, chain, _shown(lo), STEP)  # the J test at the displayed values (D147 §2)
    exc, g = JS.exceedance(SUP, family, chain, sens)
    j = lo.noise["pair"]["j"]
    assert exc >= MARGIN and j["exceedance"] == pytest.approx(exc) and j["exceedance_given"] == g
    assert j["nearest_training_linf"] >= exc - 1e-12 and g in j["out_of_range_givens"]
    for p in (lo, hi):  # magnitudes in the training range: shown, true and computed
        vals = {**_shown(p), **p.intermediates, p.target_symbol: p.target_value_si,
                **{k: v["true_si"] for k, v in p.noise["givens"].items()}}
        assert JS.in_magnitudes(SUP, family, vals)
        assert regenerate(p.to_dict(), ENV).to_dict() == p.to_dict()
    assert hi.noise["pair"]["j"]["clamped_log_ratio"] == pytest.approx(
        JS.clamped_sigma_log_ratio(j, rel(hi)))


# ------------------------------------------------------------------------------------------- J invariance pairs
@pytest.mark.parametrize("family,depth,level", [("kinematics", 3, 0.01), ("dynamics", 2, 0.05), ("energy", 4, 0.02)])
def test_j_invariance_pair_scales_the_world_and_keeps_the_jacobian(family, depth, level):
    lo, hi = _pair("j_a6b", family, depth, level)
    assert lo.chain == hi.chain and MASK(lo.text) == MASK(hi.text) and lo.distractors == hi.distractors
    assert [g["unit"] for g in lo.givens] == [g["unit"] for g in hi.givens] and lo.target_unit == hi.target_unit
    assert hi.target_value_si / lo.target_value_si >= PC["j_invariance_construction"]["min_target_ratio"] - 1e-9
    z = lambda p: {g: v["e"] / level for g, v in p.noise["givens"].items()}
    assert z(lo) == pytest.approx(z(hi))
    assert abs(math.log(hi.noise["sigma_star_log"] / lo.noise["sigma_star_log"])) <= 1e-3
    fam, chain = _chain(lo)
    s_lo = log_sensitivities(fam, chain, _shown(lo), STEP)
    s_hi = log_sensitivities(fam, chain, _shown(hi), STEP)
    assert max(abs(s_lo[g] - s_hi[g]) for g in chain.givens) < 1e-3  # display rounding only
    assert JS.exceedance(SUP, family, chain, s_lo)[0] >= MARGIN and JS.exceedance(SUP, family, chain, s_hi)[0] >= MARGIN
    sc = lo.noise["pair"]["scale"]
    ratio = lambda g: hi.noise["givens"][g]["true_si"] / lo.noise["givens"][g]["true_si"]
    d = np.array(sc["direction"], float)
    for g in chain.givens:  # every given scaled by λ^(its exponents) (or its inverse: lo / hi follow the target)
        e = float(pairs_j.dim_vector(fam.quantities[g].si_unit) @ d)
        assert ratio(g) == pytest.approx(sc["factor"] ** e, rel=1e-9) or ratio(g) == pytest.approx(sc["factor"] ** -e, rel=1e-9)
    for p in (lo, hi):
        assert regenerate(p.to_dict(), ENV).to_dict() == p.to_dict()


def test_j_constructions_check_their_inputs():
    con, _ = cell_construction("j_a6c", [0.01, 0.05], PC)
    with pytest.raises(ValueError):
        generate_pair("energy", 2, 1, 0.02, ENV, con)  # noise must be h
    with pytest.raises(ValueError):
        pairs_j.generate("energy", 2, 1, 0.05, ENV, {**con, "kind": "other"})
    with pytest.raises(ValueError):
        generate_pair("energy", 2, 1, 0.0, ENV, cell_construction("j_a6b", 0.02, PC)[0])


def test_existing_pair_sets_are_unchanged():
    """A6 / A6b / A6c (development and confirmatory) regenerate byte-identically: the J kinds are new paths only."""
    root = REPO_ROOT / "data" / "eval_sets" / generator_version(ENV)
    for sub in ("", "confirmatory/"):
        for name in ("a6_pairs", "a6b_pairs", "a6c_pairs"):
            lines = (root / f"{sub}{name}.jsonl").read_text().splitlines()
            for line in lines[::97]:
                assert json.dumps(regenerate(json.loads(line), ENV).to_dict(), sort_keys=True) == line


def test_the_frozen_j_set_regenerates_and_is_checksummed():
    root = REPO_ROOT / "data" / "eval_sets" / generator_version(ENV) / PC["r5_6"]["subdir"]
    if not (root / "a6c_pairs.jsonl").exists():
        pytest.skip("the J set is not built yet")
    import hashlib

    for name, kind in (("a6c_pairs", "j_allocation"), ("a6b_pairs", "j_invariance")):
        blob = (root / f"{name}.jsonl").read_bytes()
        man = json.loads((root / f"{name}_manifest.json").read_text())
        assert hashlib.sha256(blob).hexdigest() == man["sha256"] and man["n_problems"] == len(blob.splitlines())
        lines = blob.decode().splitlines()
        assert all(json.loads(x)["noise"]["pair"]["kind"] == kind for x in lines)
        assert {json.loads(x)["depth"] for x in lines} <= {1, 2, 3, 4}
        for line in lines[::53]:
            assert json.dumps(regenerate(json.loads(line), ENV).to_dict(), sort_keys=True) == line

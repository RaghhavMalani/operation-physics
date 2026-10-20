"""D50/A6 sensitivity pairs and D59/A6b invariance pairs: construction, exact replay, and the analyses on synthetic
answers (A6: an honest propagator gives slope ≈ 1, a lookup ≈ 0; A6b: relative σ ≈ 0, a fixed absolute σ ≈ −1).
D95/A6c noise-allocation pairs: the same displayed values and level multiset, the high level moved."""

import json
import math
import re

import numpy as np
import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.env.generator import generator_version, regenerate
from physcal.env.formulas import get_family
from physcal.env.noise import log_sensitivities
from physcal.env.pairs import generate_pair
from physcal.eval.a6_pairs import analyze_rows, joint, member_row, slope_label
from physcal.tape.tape import TapeWriter, replay
from physcal.verify.reward import reward, reward_config

PC = load_config("pairs")
ENV = load_config(PC["env_config"])
MASK = lambda s: re.sub(r"[-+]?\d[\d.]*(e-?\d+)?", "#", s)


@pytest.mark.parametrize("family,depth,level", [("kinematics", 2, 0.01), ("energy", 4, 0.05), ("dynamics", 3, 0.02)])
def test_pair_members_differ_only_in_the_world(family, depth, level):
    lo, hi = generate_pair(family, depth, 11, level, ENV, PC["construction"])
    assert lo.chain == hi.chain and lo.target_unit == hi.target_unit and MASK(lo.text) == MASK(hi.text)
    assert [g["unit"] for g in lo.givens] == [g["unit"] for g in hi.givens]
    assert lo.distractors == hi.distractors and lo.noise["level"] == hi.noise["level"] == level
    z = lambda p: {g: v["e"] / level for g, v in p.noise["givens"].items()}
    assert z(lo) == pytest.approx(z(hi))  # the same standardized measurement errors
    assert hi.noise["pair"]["delta_ratio"] >= PC["construction"]["min_ratio"]
    assert hi.noise["sigma_star_log"] > lo.noise["sigma_star_log"]
    assert (lo.seed, hi.seed) == (22, 23)
    assert lo.target_value_si != hi.target_value_si


def test_pairs_replay_exactly(tmp_path):
    rw = reward_config()
    path = tmp_path / "pairs.jsonl"
    with TapeWriter(path, "a6-test", {"env_config": ENV, "reward_config": rw}) as tape:
        for fam, d in (("kinematics", 1), ("energy", 5)):
            for p in generate_pair(fam, d, 3, 0.02, ENV, PC["construction"]):
                assert regenerate(p.to_dict(), ENV).to_dict() == p.to_dict()
                text = f"<answer>{p.noise['point_estimate_si']:.6g} ± {0.03 * p.target_value_si:.3g} {p.target_si_unit}</answer>"
                r, bd = reward(p, text, rw)
                tape.log(step=0, problem=p, completion=text, reward_value=r, breakdown=bd, model_id="fake")
    assert replay(path).ok


@pytest.mark.parametrize("name", ["a6_pairs", "a6b_pairs", "a6c_pairs"])
def test_frozen_pair_set_regenerates(name):
    root = REPO_ROOT / "data" / "eval_sets" / generator_version(ENV)
    if not (root / f"{name}.jsonl").exists():
        pytest.skip(f"{name} not exported yet")
    lines = (root / f"{name}.jsonl").read_text().splitlines()
    for line in lines[:6] + lines[-4:]:
        d = json.loads(line)
        assert json.dumps(regenerate(d, ENV).to_dict(), sort_keys=True) == line


@pytest.mark.parametrize("family,depth,level", [("kinematics", 2, 0.01), ("dynamics", 4, 0.05), ("energy", 5, 0.02)])
def test_invariance_pairs_share_sigma_star(family, depth, level):
    con = PC["invariance_construction"]
    lo, hi = generate_pair(family, depth, 9, level, ENV, con)
    assert lo.chain == hi.chain and lo.target_unit == hi.target_unit and MASK(lo.text) == MASK(hi.text)
    assert lo.distractors == hi.distractors and hi.noise["pair"]["kind"] == "invariance"
    assert hi.target_value_si >= con["min_target_ratio"] * lo.target_value_si
    assert abs(math.log(hi.noise["sigma_star_log"] / lo.noise["sigma_star_log"])) <= con["max_log_sigma_ratio"]
    fam = get_family(family)
    from physcal.env.generator import chain_of
    s_lo = log_sensitivities(fam, chain_of(lo), {g: v["true_si"] for g, v in lo.noise["givens"].items()}, 1e-6)
    s_hi = log_sensitivities(fam, chain_of(hi), {g: v["true_si"] for g, v in hi.noise["givens"].items()}, 1e-6)
    assert all(abs(s_lo[g] - s_hi[g]) < 1e-5 for g in s_lo)  # a power law: same sensitivities everywhere
    assert regenerate(lo.to_dict(), ENV).to_dict() == lo.to_dict()
    assert regenerate(hi.to_dict(), ENV).to_dict() == hi.to_dict()


def _rows(stated, n_pairs=120, seed=0, correct=True):
    """Synthetic pair answers: σ*_hi / σ*_lo spread over 2–6×; stated(σ*, rng) gives the model's relative σ."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_pairs):
        s_lo = float(np.exp(rng.uniform(np.log(0.01), np.log(0.05))))
        s_hi = s_lo * float(rng.uniform(2, 6))
        for member, s in (("lo", s_lo), ("hi", s_hi)):
            mu = 1.0
            problem = {"family": "energy", "depth": 3, "target_value_si": mu,
                       "noise": {"level": 0.02, "sigma_star_log": s, "point_estimate_si": mu if correct else 1.1,
                                 "pair": {"pair_seed": k, "member": member}}}
            b = {"reason": "ok", "mu_si": mu, "sigma_si": stated(s, rng) * mu}
            rows.append(member_row(problem, b, PC["analysis"]["tau"], PC["analysis"]["sigma_floor_rel"]))
    return rows


def test_analysis_separates_propagation_from_lookup():
    cfg = {**PC["analysis"], "bootstrap": 2000}
    honest = analyze_rows(_rows(lambda s, r: s * float(np.exp(r.normal(0, 0.2)))), cfg)
    assert honest["primary_computation_correct"]["slope"] == pytest.approx(1.0, abs=0.15) and honest["supported"]
    lookup = analyze_rows(_rows(lambda s, r: 0.02 * float(np.exp(r.normal(0, 0.2)))), cfg)
    assert abs(lookup["primary_computation_correct"]["slope"]) < 0.2 and not lookup["supported"]
    wrong = analyze_rows(_rows(lambda s, r: s, correct=False), cfg)  # not computation-correct: no primary pairs
    assert wrong["primary_computation_correct"]["n_pairs"] == 0 and not wrong["supported"]
    assert wrong["secondary_all_answered"]["slope"] == pytest.approx(1.0, abs=1e-9)


def _inv_rows(stated_abs: bool, n_pairs=120, seed=0):
    """Invariance pairs: σ* equal within a pair, targets differ 2–200×. The model states either a relative σ
    (≈ σ*, propagation) or a fixed absolute σ (its relative σ then falls as 1/target)."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_pairs):
        s = float(np.exp(rng.uniform(np.log(0.01), np.log(0.05))))
        t_lo = float(np.exp(rng.uniform(0, 3)))
        t_hi = t_lo * float(np.exp(rng.uniform(np.log(2), np.log(200))))
        for member, t in (("lo", t_lo), ("hi", t_hi)):
            sigma = 0.05 * float(np.exp(rng.normal(0, 0.1))) if stated_abs else s * t * float(np.exp(rng.normal(0, 0.1)))
            problem = {"family": "dynamics", "depth": 2, "target_value_si": t,
                       "noise": {"level": 0.02, "sigma_star_log": s, "point_estimate_si": t,
                                 "pair": {"pair_seed": k, "member": member, "kind": "invariance"}}}
            b = {"reason": "ok", "mu_si": t, "sigma_si": sigma}
            rows.append(member_row(problem, b, PC["analysis"]["tau"], PC["analysis"]["sigma_floor_rel"]))
    return rows


def test_invariance_analysis_and_joint_verdict():
    cfg = {**PC["analysis"], "bootstrap": 2000}
    rel = analyze_rows(_inv_rows(stated_abs=False), cfg, PC["invariance_analysis"])
    assert abs(rel["primary_computation_correct"]["slope"]) < 0.05 and rel["equivalent_to_zero"]
    absolute = analyze_rows(_inv_rows(stated_abs=True), cfg, PC["invariance_analysis"])
    assert absolute["primary_computation_correct"]["slope"] == pytest.approx(-1.0, abs=0.05)
    assert not absolute["equivalent_to_zero"]
    honest = analyze_rows(_rows(lambda s, r: s * float(np.exp(r.normal(0, 0.2)))), cfg)
    lookup = analyze_rows(_rows(lambda s, r: 0.02), cfg)
    assert joint(honest, rel)["verdict"] == "propagates"
    assert joint(honest, absolute)["verdict"] == "magnitude-driven"
    assert joint(lookup, rel)["verdict"] == "no propagation (lookup-like)"
    with pytest.raises(ValueError):
        analyze_rows(_inv_rows(False)[:4] + _rows(lambda s, r: s)[:4], cfg, PC["invariance_analysis"])


@pytest.mark.parametrize("ci,label", [
    ([0.2, 1.3], "full propagation"), ([0.1, 0.7], "partial propagation"), ([1.2, 1.6], "over-propagation"),
    ([-0.1, 0.5], "no propagation"), ([-0.2, 1.4], "no propagation"), ([0.0, 0.4], "no propagation"),
    ([-1.3, -0.4], "anti-propagation"), ([0.3, 1.0], "full propagation"), (None, None)])
def test_slope_labels(ci, label):
    """D61 / D64: labels from the slope's CI; a CI entirely below 0 is "anti-propagation" (D64)."""
    assert slope_label(ci) == label


def test_a6_labels_and_labelled_joint_verdict():
    cfg = {**PC["analysis"], "bootstrap": 2000}
    honest = analyze_rows(_rows(lambda s, r: s * float(np.exp(r.normal(0, 0.2)))), cfg)
    half = analyze_rows(_rows(lambda s, r: s ** 0.5 * float(np.exp(r.normal(0, 0.1)))), cfg)
    lookup = analyze_rows(_rows(lambda s, r: 0.02 * float(np.exp(r.normal(0, 0.2)))), cfg)
    anti = analyze_rows(_rows(lambda s, r: 1e-4 / s * float(np.exp(r.normal(0, 0.1)))), cfg)
    assert honest["primary_computation_correct"]["ci_level"] == PC["analysis"]["label_ci_level"] == 0.90
    assert [x["label"] for x in (honest, half, lookup, anti)] == \
        ["full propagation", "partial propagation", "no propagation", "anti-propagation"]
    rel = analyze_rows(_inv_rows(stated_abs=False), cfg, PC["invariance_analysis"])
    absolute = analyze_rows(_inv_rows(stated_abs=True), cfg, PC["invariance_analysis"])
    assert joint(honest, rel)["labelled_verdict"] == "full propagation"
    assert joint(half, rel)["labelled_verdict"] == "partial propagation"
    assert joint(honest, absolute)["labelled_verdict"] == "magnitude-driven"
    assert joint(lookup, rel)["labelled_verdict"] == "no propagation"
    assert joint(anti, absolute)["labelled_verdict"] == "anti-propagation"  # A6 fails first, as in D59
    j = joint(anti, rel)  # the D59 verdict is still reported alongside
    assert j["verdict"] == "no propagation (lookup-like)" and j["A6_label"] == "anti-propagation"
    assert j["table"][0]["label_ci"] == anti["primary_computation_correct"]["ci"]


# ------------------------------------------------------------------------------------------- D95: A6c
from physcal.env.pairs import cell_construction  # noqa: E402

PCT = lambda s: re.sub(r"± \d+(\.\d+)?%", "± p%", s)


@pytest.mark.parametrize("family,depth,levels", [("energy", 2, [0.01, 0.05]), ("dynamics", 4, [0.01, 0.02]),
                                                 ("kinematics", 1, [0.02, 0.05]), ("kinematics", 5, [0.01, 0.05])])
def test_allocation_pair_moves_only_the_high_level(family, depth, levels):
    con, noise = cell_construction("a6c", levels, PC)
    pair = next(p for p in (generate_pair(family, depth, 2_000_000_000 + i, noise, ENV, con) for i in range(10)) if p)
    lo, hi = pair
    assert lo.chain == hi.chain and lo.text != hi.text and PCT(lo.text) == PCT(hi.text)  # only the "± p%" differ
    assert [(g["display"], g["unit"]) for g in lo.givens] == [(g["display"], g["unit"]) for g in hi.givens]
    assert lo.distractors == hi.distractors and {d["noise_rel"] for d in lo.distractors} <= {levels[0]}
    rel = lambda p: {g: v["rel"] for g, v in p.noise["givens"].items()}
    assert sorted(rel(lo).values()) == sorted(rel(hi).values())  # the same multiset of visible levels
    assert [g for g in rel(lo) if rel(lo)[g] == levels[1]] == [lo.noise["pair"]["high_given"]]
    assert [g for g in rel(hi) if rel(hi)[g] == levels[1]] == [hi.noise["pair"]["high_given"]]
    assert lo.noise["pair"]["high_given"] != hi.noise["pair"]["high_given"]
    assert lo.noise["level"] == hi.noise["level"] == levels[1] and lo.noise["pair"]["construction"]["levels"] == levels
    assert lo.noise["point_estimate_si"] == hi.noise["point_estimate_si"]  # the same measured values, the same f(x)
    assert hi.noise["pair"]["delta_ratio"] >= con["min_ratio"]
    assert hi.noise["sigma_star_log_delta"] / lo.noise["sigma_star_log_delta"] == pytest.approx(
        hi.noise["pair"]["delta_ratio"], rel=0.05)  # at the measured vs the true givens
    assert hi.noise["sigma_star_log"] > lo.noise["sigma_star_log"]  # MC
    z = lambda p: {g: v["e"] / rel(p)[g] for g, v in p.noise["givens"].items()}
    assert z(lo) == pytest.approx(z(hi))  # the same standardized errors
    for m in pair:
        assert regenerate(m.to_dict(), ENV).to_dict() == m.to_dict()


def test_allocation_pair_checks_its_levels():
    con, _ = cell_construction("a6c", [0.01, 0.05], PC)
    with pytest.raises(ValueError):
        generate_pair("energy", 2, 2_000_000_000, 0.02, ENV, con)  # noise must be h
    with pytest.raises(ValueError):
        generate_pair("energy", 2, 2_000_000_000, 0.05, ENV, {**con, "levels": [0.05, 0.01]})
    assert cell_construction("a6", 0.02, PC) == (PC["construction"], 0.02)  # A6 / A6b cells are unchanged


def test_a6_and_a6b_members_are_unchanged_by_d95():
    """The per-given path (levels=None) leaves A6 / A6b exactly as frozen: every 25th frozen problem regenerates."""
    root = REPO_ROOT / "data" / "eval_sets" / generator_version(ENV)
    for name in ("a6_pairs", "a6b_pairs"):
        lines = (root / f"{name}.jsonl").read_text().splitlines()
        for line in lines[::25]:
            assert json.dumps(regenerate(json.loads(line), ENV).to_dict(), sort_keys=True) == line

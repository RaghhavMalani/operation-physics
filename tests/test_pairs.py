"""D50/A6 counterfactual sensitivity pairs: construction, exact replay, and the slope analysis on synthetic answers
(an honest propagator gives slope ≈ 1, a surface lookup ≈ 0)."""

import json
import math
import re

import numpy as np
import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.env.generator import generator_version, regenerate
from physcal.env.pairs import generate_pair
from physcal.eval.a6_pairs import analyze_rows, member_row
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


def test_frozen_pair_set_regenerates():
    root = REPO_ROOT / "data" / "eval_sets" / generator_version(ENV)
    if not (root / "a6_pairs.jsonl").exists():
        pytest.skip("pair set not exported yet")
    lines = (root / "a6_pairs.jsonl").read_text().splitlines()
    for line in lines[:6] + lines[-4:]:
        d = json.loads(line)
        assert json.dumps(regenerate(d, ENV).to_dict(), sort_keys=True) == line


def _rows(stated, n_pairs=120, seed=0, correct=True):
    """Synthetic pair answers: σ*_hi / σ*_lo spread over 2–6×; stated(σ*, rng) gives the model's relative σ."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_pairs):
        s_lo = float(np.exp(rng.uniform(np.log(0.01), np.log(0.05))))
        s_hi = s_lo * float(rng.uniform(2, 6))
        for member, s in (("lo", s_lo), ("hi", s_hi)):
            mu = 1.0
            problem = {"family": "energy", "depth": 3, "noise": {"level": 0.02, "sigma_star_log": s,
                       "point_estimate_si": mu if correct else 1.1, "pair": {"pair_seed": k, "member": member}}}
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

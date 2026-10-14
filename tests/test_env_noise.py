"""Generator 2.0.0 measurement-noise mode (Phase 10 step 3, D51; σ* per D50/A2).

Not the Phase 10 self-consistency gate (that is physcal/eval/sigma_star_gate.py, run once under A3)."""

import math
import re

import numpy as np
import pytest

from physcal.config import load_config
from physcal.env.formulas import get_family
from physcal.env.generator import chain_of, env_config, generate_problem
from physcal.env.noise import chain_target, crps_optimal_scale, predictive
from physcal.env.render import format_number
from physcal.tape.tape import TapeWriter, replay
from physcal.verify.reward import reward, reward_config

V2 = load_config("env_v2")
FAMILIES = ("kinematics", "dynamics", "energy")


def _shown_at(display: str, sig_figs: int) -> bool:
    """The display string is exactly the value formatted at `sig_figs` significant figures."""
    return format_number(float(display), sig_figs, tuple(V2["generator"]["display"]["fixed_exponent_range"])) == display


@pytest.mark.parametrize("family", FAMILIES)
def test_noise_zero_is_exact(family):
    for depth in (1, 3, 5):
        p = generate_problem(family, depth, 11, split="eval", cfg=V2, noise=0.0)
        assert "±" not in p.text and "measurements" not in p.text
        assert p.noise["sigma_star_log"] == 0.0 and p.noise["point_estimate_si"] == p.target_value_si
        assert all(_shown_at(g["display"], V2["generator"]["display"]["sig_figs"]) for g in p.givens)
        assert all("noise_rel" not in g for g in p.givens)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("level", [0.01, 0.1])
def test_noisy_problem_structure(family, level):
    p = generate_problem(family, 4, 5, split="eval", cfg=V2, noise=level)
    fam, chain = get_family(family), chain_of(p)
    pct = f"± {100 * level:g}%"
    assert "Values marked ± p% are measurements" in p.text
    for g in p.givens + p.distractors:
        assert g["noise_rel"] == level and f"{g['display']} {g['unit']}".strip() + f" {pct}" in p.text
        assert _shown_at(g["display"], V2["generator"]["noise"]["sig_figs"])
    true = {g: v["true_si"] for g, v in p.noise["givens"].items()}
    shown = {g["symbol"]: g["value_si"] for g in p.givens}
    # truth = noise-free chain on the true values; point estimate and step values = chain on the displayed values
    assert float(chain_target(fam, chain, true)) == pytest.approx(p.target_value_si, rel=1e-12)
    assert float(chain_target(fam, chain, shown)) == pytest.approx(p.noise["point_estimate_si"], rel=1e-12)
    for x, v in p.intermediates.items():
        assert v != p.noise["intermediates_true_si"][x]
    for g, v in p.noise["givens"].items():
        assert shown[g] == pytest.approx(v["true_si"] * (1 + v["e"]), rel=1e-4)  # 5 s.f. rounding
    assert p.noise["mc_draws"] == V2["generator"]["noise"]["n_mc"] and 0.0 <= p.noise["pit_truth"] <= 1.0


def test_problems_are_paired_across_noise_levels():
    """Same seed: same chain, units, phrasing and clean truth; the measurement errors scale with the level."""
    ps = {lv: generate_problem("energy", 3, 21, split="eval", cfg=V2, noise=lv) for lv in (0.01, 0.02, 0.05, 0.1)}
    base = ps[0.01]
    for lv, p in ps.items():
        assert p.chain == base.chain and p.target_unit == base.target_unit
        assert [g["unit"] for g in p.givens] == [g["unit"] for g in base.givens]
        assert p.target_value_si == base.target_value_si
        for g in p.noise["givens"]:
            assert p.noise["givens"][g]["e"] == pytest.approx(base.noise["givens"][g]["e"] * lv / 0.01, rel=1e-12)
    clean = generate_problem("energy", 3, 21, split="eval", cfg=V2, noise=0.0)
    assert clean.chain == base.chain and clean.target_value_si == pytest.approx(base.target_value_si, rel=0.02)


def test_noise_levels_are_validated():
    with pytest.raises(ValueError):
        generate_problem("energy", 2, 1, split="eval", cfg=env_config(), noise=0.01)  # 1.1.0 has no noise mode
    with pytest.raises(ValueError):
        generate_problem("energy", 2, 1, split="eval", cfg=V2, noise=0.03)


@pytest.mark.parametrize("level", [0.01, 0.02])
def test_mc_sigma_star_matches_the_delta_method_at_small_noise(level):
    """Phase 10 step 3 cross-check: for s <= 2%, MC σ* (SD and CRPS-optimal) agrees with first-order propagation."""
    ratios_sd, ratios_crps = [], []
    for family in FAMILIES:
        for depth in range(1, 6):
            for seed in range(3):
                n = generate_problem(family, depth, 300 + seed, split="eval", cfg=V2, noise=level).noise
                ratios_sd.append(n["sigma_star_log_sd"] / n["sigma_star_log_delta"])
                ratios_crps.append(n["sigma_star_log"] / n["sigma_star_log_sd"])
    assert np.max(np.abs(np.array(ratios_sd) - 1)) < 0.06  # MC SE of an SD with 4,000 draws is ~1.1%
    assert np.max(np.abs(np.array(ratios_crps) - 1)) < 0.06


def test_crps_optimal_scale_of_gaussian_draws_is_their_sd():
    x = np.random.default_rng(0).normal(0.3, 0.05, 200_000)
    assert crps_optimal_scale(x, 0.3) == pytest.approx(0.05, rel=0.01)
    assert crps_optimal_scale(x, 0.3) < crps_optimal_scale(x, 0.35)  # off-centre forecasts need a wider scale


def test_predictive_rejects_draws_outside_the_domain():
    p = generate_problem("kinematics", 2, 3, split="eval", cfg=V2, noise=0.1)
    fam, chain = get_family("kinematics"), chain_of(p)
    shown = {g["symbol"]: g["value_si"] for g in p.givens}
    z = np.zeros((len(chain.givens), 5))
    z[0, 2] = -20.0  # 1 + 0.1·(−20) < 0
    assert predictive(fam, chain, shown, {g: 0.1 for g in chain.givens}, z) is None
    ok = predictive(fam, chain, shown, {g: 0.1 for g in chain.givens}, np.zeros((len(chain.givens), 5)))
    assert np.allclose(ok, math.log(p.noise["point_estimate_si"]))


def test_noisy_tape_replays_exactly(tmp_path):
    rw = reward_config()
    path = tmp_path / "noisy.jsonl"
    with TapeWriter(path, "noise-test", {"env_config": V2, "reward_config": rw}) as tape:
        for i, lv in enumerate((0.0, 0.01, 0.05, 0.1) * 3):
            p = generate_problem(FAMILIES[i % 3], i % 5 + 1, 900 + i, split="eval", cfg=V2, noise=lv)
            text = (f"<answer>{p.noise['point_estimate_si']:.5g} ± {0.05 * p.target_value_si:.3g} "
                    f"{p.target_si_unit}</answer>")
            r, bd = reward(p, text, rw)
            tape.log(step=0, problem=p, completion=text, reward_value=r, breakdown=bd, model_id="fake")
    report = replay(path)
    assert report.ok and report.n == 12
    assert re.search(r"2\.0\.0\+[0-9a-f]{8}", open(path).readline())

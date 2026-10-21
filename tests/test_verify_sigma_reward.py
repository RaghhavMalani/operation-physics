"""Verifier 1.3.0 (D105): the σ reward against the generator's predictive distribution, centred at f(x_meas); every
older config and tape scores exactly as before (the 1.3.0 entry point passes them through to reward.reward)."""

import copy
import json
import math

import numpy as np
import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.env import noise as noise_mode
from physcal.env.formulas import get_family
from physcal.env.generator import chain_of, generate_problem
from physcal.eval import oracle_centred as oc
from physcal.verify import sigma_reward as S
from physcal.verify.reward import reward as reward_v12, reward_config
from physcal.verify.scoring import floor

V11, V12, V13 = reward_config(), reward_config("reward_v12"), reward_config("reward_v13")


@pytest.fixture(scope="module")
def noisy():
    return generate_problem("kinematics", 3, 4242, split="train", cfg=load_config("env_v2"), noise=0.05).to_dict()


@pytest.fixture(scope="module")
def exact():
    return generate_problem("energy", 2, 777, split="train", cfg=load_config("env_v2"), noise=0.0).to_dict()


def answer(p, mu_si, rel_sigma, unit=None):
    mu = mu_si * p["target_value"] / p["target_value_si"]
    return f"<steps>\n</steps>\n<answer>{mu!r} ± {abs(mu) * rel_sigma!r} {unit or p['target_unit']}</answer>\n" \
           "<confidence>50%</confidence>"


def test_versions_and_config():
    assert V13["version"] == "1.3.0" and V13["weights"]["sigma"] == 0.0
    assert {k: v for k, v in V13.items() if k not in ("version", "sigma_reward", "weights")} == \
        {k: v for k, v in V12.items() if k not in ("version", "weights")}  # 1.3.0 = 1.2.0 + the sigma reward
    assert {k: v for k, v in V13["weights"].items() if k != "sigma"} == V12["weights"]
    for cfg in (V11, V12, V13):
        S.validate(cfg)
    assert reward_config()["version"] == "1.1.0"  # the default is unchanged


@pytest.mark.parametrize("bad", ["no_section", "no_weight", "negative", "too_heavy", "section_in_v12", "rule"])
def test_invalid_configs_are_refused(bad):
    cfg = copy.deepcopy(V13 if bad != "section_in_v12" else V12)
    if bad == "no_section":
        cfg.pop("sigma_reward")
    elif bad == "no_weight":
        cfg["weights"].pop("sigma")
    elif bad == "negative":
        cfg["weights"]["sigma"] = -0.1
    elif bad == "too_heavy":
        cfg["weights"]["sigma"] = 1.0
    elif bad == "section_in_v12":
        cfg["sigma_reward"] = copy.deepcopy(V13["sigma_reward"])
        cfg["weights"]["sigma"] = 0.0
    else:
        cfg["scoring"]["rule"] = "crps"
    with pytest.raises(ValueError):
        S.validate(cfg)


def test_older_configs_pass_through_unchanged(noisy):
    pe = noisy["noise"]["point_estimate_si"]
    for cfg in (V11, V12):
        for text in (answer(noisy, pe, 0.03), answer(noisy, pe * 1.05, 0.0), "no answer here",
                     answer(noisy, pe, 0.03, unit="kg")):
            assert S.reward(noisy, text, cfg) == reward_v12(noisy, text, cfg)


@pytest.mark.parametrize("tape", [
    "results/posthoc_calibration_4bit/sft_calibration_set/rollouts.jsonl",
    "results/train/overnight2-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512/compare/sft_4bit/rollouts.jsonl",
])
def test_old_tapes_replay_exactly_through_the_new_entry_point(tape):
    path = REPO_ROOT / tape
    if not path.exists():
        pytest.skip("tape not present")
    rep = S.replay(path)
    assert rep["n"] >= 400 and rep["reward_mismatches"] == [] and rep["breakdown_mismatches"] == []


def test_v13_total_equals_v12_and_adds_score_sigma(noisy):
    pe = noisy["noise"]["point_estimate_si"]
    text = answer(noisy, pe * 1.02, 0.04)
    r12, b12 = reward_v12(noisy, text, V12)
    r13, b13 = S.reward(noisy, text, V13)
    assert r13 == r12 and {k: v for k, v in b13.items() if k in b12 and k != "verifier_version"} == \
        {k: v for k, v in b12.items() if k != "verifier_version"}
    assert b13["verifier_version"] == "1.3.0" and b13["sigma_reason"] == "ok" and b13["w_sigma"] == 0.0
    assert b13["sigma_rel_scored"] == pytest.approx(0.04, rel=1e-12) and b13["sigma_draws"] == "predictive"
    heavy = copy.deepcopy(V13)
    heavy["weights"]["sigma"] = 0.5
    r, b = S.reward(noisy, text, heavy)
    assert r == pytest.approx(r12 + 0.5 * b13["score_sigma"], abs=1e-12)


def test_score_sigma_is_the_expected_crps_log_centred_at_f_xmeas(noisy):
    pe = noisy["noise"]["point_estimate_si"]
    sr = V13["sigma_reward"]
    _, b = S.reward(noisy, answer(noisy, pe * 1.07, 0.03), V13)  # μ 7% off: the σ reward does not see it
    _, b0 = S.reward(noisy, answer(noisy, pe, 0.03), V13)
    assert b["score_sigma"] == b0["score_sigma"] and b["sigma_centre_si"] == pe
    draws, how = S.predictive_draws(noisy, sr)
    assert how["sigma_draws_dropped"] == 0 and len(draws) == sr["n_draws"]
    z = S._stream(noisy, sr["seed_tag"]).standard_normal((len(chain_of(noisy).givens), sr["n_draws"]))
    ref = noise_mode.predictive(get_family(noisy["family"]), chain_of(noisy),
                                {g["symbol"]: g["value_si"] for g in noisy["givens"]},
                                {g["symbol"]: g.get("noise_rel") or 0.0 for g in noisy["givens"]}, z)
    assert np.array_equal(draws, ref)  # the masked predictive equals noise.predictive when every draw is valid
    assert b["score_sigma"] == pytest.approx(oc.expected_crps_log(ref, math.log(pe), 0.03, V13["scoring"]), abs=1e-15)


def test_score_sigma_peaks_at_the_crps_optimal_scale(noisy):
    pe = noisy["noise"]["point_estimate_si"]
    draws, _ = S.predictive_draws(noisy, V13["sigma_reward"])
    opt = noise_mode.crps_optimal_scale(draws, math.log(pe))
    assert opt == pytest.approx(noisy["noise"]["sigma_star_log"], rel=0.1)  # A2's σ*, up to MC
    s = {c: S.reward(noisy, answer(noisy, pe, c * opt), V13)[1]["score_sigma"] for c in (0.5, 0.8, 1.0, 1.25, 2.0)}
    assert s[1.0] == max(s.values()) and s[0.5] < s[0.8] < s[1.0] > s[1.25] > s[2.0]


def test_exact_givens_reward_zero_spread(exact):
    t = exact["target_value_si"]
    s = [S.reward(exact, answer(exact, t, r), V13)[1] for r in (0.0, 0.001, 0.01, 0.1)]
    assert s[0]["sigma_draws"] == "exact" and s[0]["sigma_n_draws"] == 1
    assert s[0]["score_sigma"] > s[1]["score_sigma"] > s[2]["score_sigma"] > s[3]["score_sigma"]
    assert s[0]["score_sigma"] == pytest.approx(0.0, abs=1e-5)


def test_failures_get_the_floor(noisy):
    pe = noisy["noise"]["point_estimate_si"]
    worst = floor("crps_log", V13["scoring"])
    _, b = S.reward(noisy, "no answer", V13)
    assert b["score_sigma"] == worst and b["sigma_reason"] == "answer_unscored"
    _, b = S.reward(noisy, answer(noisy, pe, 0.03, unit="kg"), V13)
    assert b["reason"] == "unit_mismatch" and b["score_sigma"] == worst


def test_invalid_draws_are_dropped_or_fall_back(noisy):
    sr = copy.deepcopy(V13["sigma_reward"])
    fam, chain = get_family(noisy["family"]), chain_of(noisy)
    measured = {g["symbol"]: g["value_si"] for g in noisy["givens"]}
    levels = {g["symbol"]: 0.6 for g in noisy["givens"]}  # 60% noise: some 1 + s·z <= 0
    z = np.random.default_rng(0).standard_normal((len(chain.givens), 2000))
    logs, dropped = S.predictive_masked(fam, chain, measured, levels, z)
    assert dropped > 0 and len(logs) == 2000 - dropped and np.all(np.isfinite(logs))
    assert noise_mode.predictive(fam, chain, measured, levels, z) is None
    sr["min_valid_share"] = 1.0
    p = copy.deepcopy(noisy)
    for g in p["givens"]:
        if g.get("noise_rel"):
            g["noise_rel"] = 0.6
    S._DRAWS.clear()
    draws, how = S.predictive_draws(p, sr)
    assert how["sigma_draws"] == "gaussian_fallback" and how["sigma_draws_dropped"] > 0 and len(draws) == sr["n_draws"]


def test_a_v13_tape_replays_exactly(tmp_path, noisy):
    from physcal.env.generator import env_config
    from physcal.tape.tape import TapeWriter

    pe = noisy["noise"]["point_estimate_si"]
    prob = generate_problem("kinematics", 3, 4242, split="train", cfg=load_config("env_v2"), noise=0.05)
    path = tmp_path / "rollouts.jsonl"
    with TapeWriter(path, "v13-test", {"env_config": load_config("env_v2"), "reward_config": V13}) as tape:
        for mu, s in ((pe, 0.03), (pe * 1.01, 0.0), (pe, 0.2)):
            text = answer(noisy, mu, s)
            r, bd = S.reward(prob, text, V13)
            tape.log(0, prob, text, r, bd, "test", None, {})
    S._DRAWS.clear()  # replay recomputes the draws from scratch
    rep = S.replay(path)
    assert rep == {"n": 3, "reward_mismatches": [], "breakdown_mismatches": []}

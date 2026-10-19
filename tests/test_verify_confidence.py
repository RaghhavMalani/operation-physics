"""Verifier 1.2.0 (D83): the <confidence> channel, Brier against the computation-correct event (D50/A1), with 1.0.0 /
1.1.0 configs and tapes unchanged."""

import copy
import json
import math

import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.env.generator import generate_problem, regenerate
from physcal.verify.parse import parse_confidence
from physcal.verify.reward import (BRIER_FLOOR, CONFIDENCE_MISSING, computation_reference_si, reward, reward_config,
                                   validate_reward_config)
from physcal.verify.scoring import floor

V11, V12 = reward_config(), reward_config("reward_v12")


@pytest.fixture(scope="module")
def noisy():
    p = generate_problem("kinematics", 2, 123, split="train", cfg=load_config("env_v2"), noise=0.05)
    assert abs(math.log(p.target_value_si / p.noise["point_estimate_si"])) > V12["confidence"]["tau_log"]
    return p


@pytest.fixture(scope="module")
def exact():
    return generate_problem("energy", 1, 777, split="train")  # generator 1.1.0: no noise field


def answer(problem, mu_si, rel_sigma=0.02, c=None, unit=None):
    mu = mu_si * problem.target_value / problem.target_value_si
    s = f"<steps>\n</steps>\n<answer>{mu:.7g} ± {abs(mu) * rel_sigma:.2g} {unit or problem.target_unit}</answer>"
    return s if c is None else s + f"\n<confidence>{c}%</confidence>"


def test_versions_and_default():
    assert V11["version"] == "1.1.0" and "confidence" not in V11  # the default is unchanged
    assert V12["version"] == "1.2.0" and V12["weights"]["confidence"] == 0.0
    assert V12["confidence"]["bins"] == load_config("sft_phase12")["confidence"]["bins"]  # the data's bins
    assert {k: v for k, v in V12.items() if k not in ("version", "confidence")} | {"weights": V11["weights"]} \
        == {k: v for k, v in V11.items() if k != "version"}  # 1.2.0 = 1.1.0 + the channel


def test_v11_ignores_confidence_blocks(noisy):
    pe = noisy.noise["point_estimate_si"]
    r0, b0 = reward(noisy, answer(noisy, pe), V11)
    r1, b1 = reward(noisy, answer(noisy, pe, c=90), V11)
    assert r0 == r1 and b0 == b1 and "score_confidence" not in b1


@pytest.mark.parametrize("tape", [
    "results/posthoc_calibration_4bit/sft_calibration_set/rollouts.jsonl",
    "results/train/overnight2-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512/compare/sft_4bit/rollouts.jsonl",
])
def test_v11_tapes_replay_exactly_with_identical_breakdowns(tape):
    from physcal.tape.tape import read_meta, read_tape

    path = REPO_ROOT / tape
    if not path.exists():
        pytest.skip("tape not available")
    meta = read_meta(path)
    assert meta["reward_config"]["version"] == "1.1.0"
    n = 0
    for rec in read_tape(path):
        r, bd = reward(regenerate(rec["problem"], meta["env_config"]), rec["completion"], meta["reward_config"])
        assert r == rec["reward"] and json.loads(json.dumps(bd)) == rec["breakdown"]
        n += 1
    assert n == 400


def test_brier_against_the_computation_correct_event(noisy):
    pe, truth = noisy.noise["point_estimate_si"], noisy.target_value_si
    _, right = reward(noisy, answer(noisy, pe, c=90), V12)
    assert right["computation_correct"] and right["score_confidence"] == pytest.approx(-0.01)
    assert right["computation_reference_si"] == pe and right["confidence"] == 90
    _, clean = reward(noisy, answer(noisy, truth, c=90), V12)  # the noise-free truth is NOT f(x_meas)
    assert not clean["computation_correct"] and clean["score_confidence"] == pytest.approx(-0.81)
    _, low = reward(noisy, answer(noisy, truth, c=10), V12)
    assert low["score_confidence"] == pytest.approx(-0.01)
    # the answer score is unchanged by the channel: scored against the truth, as in 1.1.0
    assert right["score_final"] == reward(noisy, answer(noisy, pe), V11)[1]["score_final"]


def test_noise_free_reference_is_the_truth(exact):
    assert computation_reference_si(exact.to_dict()) == exact.target_value_si
    _, bd = reward(exact, answer(exact, exact.target_value_si * (1 + 5e-4), c=99), V12)
    assert bd["computation_correct"] and bd["score_confidence"] == pytest.approx(-(0.01 ** 2))
    _, bd = reward(exact, answer(exact, exact.target_value_si * (1 + 5e-3), c=99), V12)  # correct within 1%, not 0.1%
    assert bd["correct"] and not bd["computation_correct"]


@pytest.mark.parametrize("tail", ["", "\n<confidence>55%</confidence>", "\n<confidence>0.9</confidence>",
                                  "\n<confidence>90%</confidence>\n<confidence>10%</confidence>",
                                  "\n<confidence>90</confidence>", "\n<confidence>90%"])
def test_missing_or_malformed_confidence_gets_the_brier_floor(noisy, tail):
    pe = noisy.noise["point_estimate_si"]
    _, good = reward(noisy, answer(noisy, pe, c=90), V12)
    r, bd = reward(noisy, answer(noisy, pe) + tail, V12)
    assert bd["reason"] == "ok" and bd["confidence"] is None and bd["confidence_reason"] == CONFIDENCE_MISSING
    assert bd["score_confidence"] == BRIER_FLOOR and bd["score_final"] == good["score_final"]


def test_a_confidence_before_the_answer_does_not_count(noisy):
    pe = noisy.noise["point_estimate_si"]
    text = "<confidence>90%</confidence>\n" + answer(noisy, pe)
    assert parse_confidence(text, V12["confidence"]["bins"]) is None


def test_unparsed_answer_gets_both_floors(noisy):
    cfg = copy.deepcopy(V12)
    cfg["weights"]["confidence"] = 0.5
    r, bd = reward(noisy, "<answer>no number here</answer>\n<confidence>10%</confidence>", cfg)
    assert bd["reason"] == "parse_fail" and bd["score_confidence"] == BRIER_FLOOR and bd["confidence"] == 10
    assert r == pytest.approx(floor("crps_log", cfg["scoring"]) + 0.5 * BRIER_FLOOR)


def test_unit_mismatch_is_not_computation_correct(noisy):
    pe = noisy.noise["point_estimate_si"]
    _, bd = reward(noisy, answer(noisy, pe, c=10, unit="kg"), V12)
    assert bd["reason"] == "unit_mismatch" and not bd["computation_correct"]
    assert bd["score_confidence"] == pytest.approx(-0.01)


def test_the_weight_enters_the_reward_and_zero_keeps_1_1_0s_value(noisy):
    pe = noisy.noise["point_estimate_si"]
    text = answer(noisy, noisy.target_value_si, c=70)
    r11, _ = reward(noisy, text, V11)
    r12, bd = reward(noisy, text, V12)
    assert r12 == r11  # weight 0
    cfg = copy.deepcopy(V12)
    cfg["weights"]["confidence"] = 0.5
    r, bd = reward(noisy, text, cfg)
    assert r == pytest.approx(r11 + 0.5 * bd["score_confidence"]) and bd["w_confidence"] == 0.5


def test_brier_over_the_bins_is_proper():
    """Expected score of stating c when the event has probability q: −[(c − q)² + q(1 − q)], maximized by the bin
    nearest q. No bin other than the nearest can do better."""
    bins = V12["confidence"]["bins"]
    for i in range(101):
        q = i / 100
        best = max(bins, key=lambda c: -((c / 100 - q) ** 2 + q * (1 - q)))
        assert abs(best / 100 - q) <= min(abs(b / 100 - q) for b in bins) + 1e-12  # ties: either neighbour


def test_validation():
    for broken in ("no_section", "section_in_v11", "negative", "too_heavy", "bad_bins"):
        cfg = copy.deepcopy(V12 if broken != "section_in_v11" else V11)
        if broken == "no_section":
            cfg.pop("confidence")
        elif broken == "section_in_v11":
            cfg["confidence"] = copy.deepcopy(V12["confidence"])
            cfg["weights"]["confidence"] = 0.0
        elif broken == "negative":
            cfg["weights"]["confidence"] = -0.1
        elif broken == "too_heavy":
            cfg["weights"]["confidence"] = 5.0  # steps 0.2 + 5 x 1 >= final span 5
        else:
            cfg["confidence"]["bins"] = [0, 50, 100]
        with pytest.raises(ValueError):
            validate_reward_config(cfg)


def test_phase12_completions_score_under_1_2_0():
    from physcal.train import sft_phase12 as p12

    cfg = p12._config()
    env = load_config(cfg["env_config"])
    for i, (f, d, s) in enumerate([("dynamics", 3, 0.02), ("energy", 2, 0.0), ("kinematics", 4, 0.05)]):
        prob = generate_problem(f, d, cfg["seed_offset"] + 95_000 + i, split="train", cfg=env, noise=s)
        c = p12.confidence_draw(prob.seed, cfg["confidence"]["bins"])
        for arm in cfg["arms"]:
            row = p12.build_example(prob, arm, cfg, env, c)
            _, bd = reward(prob, row["completion"], V12)
            assert bd["reason"] == "ok" and bd["confidence"] == c and bd["computation_correct"]  # 4 s.f. answers
            assert bd["score_confidence"] == pytest.approx(-(c / 100 - 1) ** 2)

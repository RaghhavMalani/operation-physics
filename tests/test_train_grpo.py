"""GRPO entrypoint pieces that run without a GPU: data, reward logging, per-step stats, deadlines."""

import json

import pytest

from physcal.config import load_config
from physcal.env.generator import env_config
from physcal.tape.tape import TapeWriter, read_tape, replay
from physcal.train.grpo import VerifierReward, parse_deadline, step_stats, training_problems
from physcal.verify.reward import reward_config

CFG = load_config("grpo_phase4")


def test_training_problems_are_train_split_and_uniform():
    probs = training_problems(24, 0, env_config())
    assert all(p.depth <= 4 for p in probs) and all(p.seed < env_config()["splits"]["eval_seed_offset"] for p in probs)
    assert len({(p.family, p.depth) for p in probs}) == 12


def _oracle(p):
    steps = "\n".join(f"{s} = {v:.6g} {p.intermediate_si_units[s]}" for s, v in p.intermediates.items())
    return f"<steps>\n{steps}\n</steps>\n<answer>{p.target_value:.6g} ± {0.01 * p.target_value:.3g} {p.target_unit}</answer>"


def test_verifier_reward_logs_every_rollout_and_step_metrics(tmp_path):
    probs = training_problems(2, 100, env_config())
    rw = reward_config()
    rw["scoring"]["rule"] = "binary"
    meta = {"env_config": env_config(), "reward_config": rw}
    tape = TapeWriter(tmp_path / "rollouts.jsonl", "t", meta)
    fn = VerifierReward(rw, tape, tmp_path / "m.jsonl", group_size=4, max_completion_tokens=10, model_id="fake")
    # group 1: 4 correct; group 2: 1 correct + 3 garbage
    completions = [_oracle(probs[0])] * 4 + [_oracle(probs[1])] + ["junk"] * 3
    pj = [json.dumps(probs[0].to_dict())] * 4 + [json.dumps(probs[1].to_dict())] * 4
    ids = [[1] * 5] * 7 + [[1] * 10]
    rewards = fn(prompts=["p"] * 8, completions=completions, problem_json=pj, completion_ids=ids)
    tape.close()
    assert rewards[:5] == [r for r in rewards[:5]] and min(rewards[:5]) >= 1.0 and max(rewards[5:]) == 0.0
    stats = json.loads((tmp_path / "m.jsonl").read_text().splitlines()[0])
    assert stats["zero_variance_fraction"] == 0.5 and stats["accuracy_binary"] == pytest.approx(5 / 8)
    assert stats["truncation_rate"] == pytest.approx(1 / 8) and stats["parse_fail_rate"] == pytest.approx(3 / 8)
    assert stats["coverage_1sigma"] == 1.0 and stats["mean_rel_sigma"] == pytest.approx(0.01, rel=0.05)
    assert len(list(read_tape(tmp_path / "rollouts.jsonl"))) == 8
    assert replay(tmp_path / "rollouts.jsonl").ok


def test_step_stats_handles_all_failures():
    rows = [{"bd": {"reason": "parse_fail"}, "by_rule": {"binary": {"score_final": 0.0}}, "truncated": True,
             "n_tokens": 1024}] * 8
    s = step_stats(rows, [-5.0] * 8, 8)
    assert s["zero_variance_fraction"] == 1.0 and s["mean_rel_sigma"] is None and s["coverage_1sigma"] is None


def test_parse_deadline():
    assert parse_deadline(None) is None
    import datetime as dt
    from zoneinfo import ZoneInfo

    iso = dt.datetime(2026, 10, 12, 10, 15, tzinfo=dt.timezone.utc)
    assert parse_deadline(iso.isoformat()) == iso.timestamp()
    local = parse_deadline("15:45 Asia/Kolkata")
    when = dt.datetime.fromtimestamp(local, ZoneInfo("Asia/Kolkata"))
    assert (when.hour, when.minute) == (15, 45) and when.date() == dt.datetime.now(ZoneInfo("Asia/Kolkata")).date()


def test_config_caps_match_the_brief():
    g = CFG["grpo"]
    assert g["max_steps"] == 300 and g["wall_clock_limit_minutes"] == 120
    assert g["save_steps"] == 50 and g["save_total_limit"] == 2
    assert g["per_device_train_batch_size"] * g["gradient_accumulation_steps"] == g["num_generations"] == 8
    assert g["max_completion_length"] == 1024 and CFG["guards"]["max_gpu_temp_c"] == 87

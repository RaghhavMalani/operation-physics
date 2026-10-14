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
    import time

    for hhmm in ("06:30", "15:45", "23:59"):
        local = parse_deadline(f"{hhmm} Asia/Kolkata")
        when = dt.datetime.fromtimestamp(local, ZoneInfo("Asia/Kolkata"))
        assert f"{when.hour:02d}:{when.minute:02d}" == hhmm
        assert 0 < local - time.time() <= 24 * 3600  # always the next occurrence, never in the past


def test_config_caps_match_the_brief():
    g = CFG["grpo"]
    assert g["max_steps"] == 300 and g["wall_clock_limit_minutes"] == 120
    assert g["save_steps"] == 50 and g["save_total_limit"] == 2
    assert g["per_device_train_batch_size"] * g["gradient_accumulation_steps"] == g["num_generations"] == 8
    assert g["max_completion_length"] == 1024 and CFG["guards"]["max_gpu_temp_c"] == 87


def test_curriculum_sampler_matches_trl_repeat_pattern_and_feedback():
    from physcal.env.sampler import EMASampler
    from physcal.train.grpo import CurriculumRepeatSampler, curriculum_pool

    probs, cell_indices = curriculum_pool(3, 0, env_config())
    assert len(probs) == 36 and all(p.seed < env_config()["splits"]["eval_seed_offset"] for p in probs)
    cur = EMASampler(list(cell_indices), {"alpha": 0.2, "weight_floor": 0.02, "prior_pass_rate": 0.5,
                                         "start_depths": [1, 2], "unlock_pass_rate": 0.5, "min_observations": 2},
                     seed=0)
    s = CurriculumRepeatSampler(cell_indices, cur, mini_repeat_count=8, batch_size=4, repeat_count=8, total_chunks=3)
    out = list(s)
    assert len(out) == len(s) == 3 * 4 * 8 * 8
    chunk = out[:256]
    assert chunk[:32] == chunk[32:64]  # whole chunk repeated repeat_count times
    assert all(len(set(chunk[i:i + 8])) == 1 for i in range(0, 32, 8))  # each prompt mini-repeated 8 times
    assert all(probs[i].depth in (1, 2) for i in out)


def test_overnight2_config_is_identical_to_the_control_except_allowed_keys():
    """Phase 3.7 brief: identical to last night's run except the verifier, the memory-only micro-batch split (same
    effective batch), the permanent step-287 checkpoint, the memory cap and the run id (D39)."""
    old, new = load_config("grpo_overnight"), load_config("grpo_overnight2")
    allowed = {("run_id",), ("grpo", "per_device_train_batch_size"), ("grpo", "gradient_accumulation_steps"),
               ("grpo", "permanent_save_steps"), ("guards", "max_reserved_gib")}

    def flat(d, prefix=()):
        for k, v in d.items():
            yield from (flat(v, prefix + (k,)) if isinstance(v, dict) else [(prefix + (k,), v)])

    a, b = dict(flat(old)), dict(flat(new))
    assert {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} <= allowed
    eff = lambda c: c["grpo"]["per_device_train_batch_size"] * c["grpo"]["gradient_accumulation_steps"]
    assert eff(old) == eff(new) == 4 * 8 == new["grpo"]["prompts_per_step"] * new["grpo"]["num_generations"]
    assert new["grpo"]["permanent_save_steps"] == [287] and new["grpo"]["save_total_limit"] == 3
    assert new["guards"]["max_reserved_gib"] == 7.5


def test_control_run_used_the_same_reward_config_except_the_floor():
    """Last night's logged reward config differs from verifier 1.1.0 only in the version and crps_log floor."""
    from physcal.config import REPO_ROOT
    from physcal.tape.tape import read_meta

    run = REPO_ROOT / "results" / "train" / "overnight-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-512"
    logged = read_meta(run / "rollouts.jsonl")["reward_config"]
    now = reward_config()
    assert now.pop("version") == "1.1.0" and "version" not in logged
    assert now["scoring"].pop("crps_log_sigma_floor") == 1e-6
    assert logged == now


def test_permanent_checkpoint_callback_saves_only_at_listed_steps(tmp_path):
    from types import SimpleNamespace

    from physcal.train.grpo import make_keep_and_memory_callback

    saved = []

    class FakeModel:
        def save_pretrained(self, path):
            saved.append(path)
            __import__("pathlib").Path(path).mkdir(parents=True)

    class FakeState(SimpleNamespace):
        def save_to_json(self, path):
            __import__("pathlib").Path(path).write_text("{}")

    cb = make_keep_and_memory_callback([287], tmp_path / "permanent", None, None)
    for step in (286, 287, 288):
        cb.on_step_end(None, FakeState(global_step=step), SimpleNamespace(), model=FakeModel())
    assert saved == [str(tmp_path / "permanent" / "checkpoint-287")]
    assert (tmp_path / "permanent" / "checkpoint-287" / "trainer_state.json").exists()


def test_sigma_zero_monitor_alerts_on_rolling_share():
    from physcal.train.grpo import SigmaZeroMonitor

    mon = SigmaZeroMonitor(threshold=0.5, window_steps=3)
    out = [mon.update(n, z) for n, z in [(10, 2), (10, 9), (10, 9), (10, 9), (10, 0), (10, 0), (10, 0)]]
    assert [o["active"] for o in out] == [False, True, True, True, True, False, False]
    assert [o["changed"] for o in out] == [False, True, False, False, False, True, False]
    assert out[1]["rolling"] == pytest.approx(11 / 20)


def test_verifier_reward_logs_sigma_zero_share_and_alert(tmp_path):
    probs = training_problems(1, 100, env_config())
    p = probs[0]
    fn = VerifierReward(reward_config(), None, tmp_path / "m.jsonl", group_size=4, max_completion_tokens=999,
                        model_id="fake")
    fn.events_path = tmp_path / "events.jsonl"
    zero = f"<answer>{p.target_value:.6g} ± 0 {p.target_unit}</answer>"
    fn(prompts=["p"] * 4, completions=[zero] * 3 + [_oracle(p)], problem_json=[json.dumps(p.to_dict())] * 4)
    stats = json.loads((tmp_path / "m.jsonl").read_text().splitlines()[0])
    assert stats["sigma_zero_share"] == pytest.approx(0.75) and stats["sigma_zero_alert"] is True
    assert "ALERT" in (tmp_path / "events.jsonl").read_text()

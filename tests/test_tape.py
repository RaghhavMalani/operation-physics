"""Rollout tape: logging and exact replay."""

import json

from physcal.env.generator import env_config, generate_problem
from physcal.tape.tape import TAPE_FIELDS, TapeWriter, read_tape, replay
from physcal.verify.reward import reward, reward_config


def _completion(p, kind: int) -> str:
    if kind == 0:  # oracle
        steps = "\n".join(f"{s} = {v:.6g} {p.intermediate_si_units[s]}" for s, v in p.intermediates.items())
        return f"<steps>\n{steps}\n</steps>\n<answer>{p.target_value:.5g} ± {0.02 * p.target_value:.3g} {p.target_unit}</answer>"
    if kind == 1:  # wrong value, odd notation
        return f"<answer>({1.3 * p.target_value_si:.3e} ± 0.1) {p.target_si_unit}</answer>"
    if kind == 2:  # wrong dimensions
        return "<answer>3.0 ± 0.5 kg*m^3</answer>"
    return "I am not sure."  # parse failure


def _write_tape(path, n=24):
    env_cfg, rw_cfg = env_config(), reward_config()
    meta = {"env_config": env_cfg, "reward_config": rw_cfg}
    with TapeWriter(path, "test-run", meta) as tape:
        for i in range(n):
            fam = ("kinematics", "dynamics", "energy")[i % 3]
            p = generate_problem(fam, i % 5 + 1, 7000 + i, split="eval")
            text = _completion(p, i % 4)
            r, bd = reward(p, text, rw_cfg)
            tape.log(step=i // 4, problem=p, completion=text, reward_value=r, breakdown=bd, model_id="fake")


def test_tape_lines_have_spec_fields(tmp_path):
    path = tmp_path / "run.jsonl"
    _write_tape(path)
    recs = list(read_tape(path))
    assert len(recs) == 24
    for rec in recs:
        assert set(TAPE_FIELDS) <= set(rec)
    assert {r["breakdown"]["reason"] for r in recs} == {"ok", "unit_mismatch", "parse_fail"}


def test_replay_reproduces_rewards_exactly(tmp_path):
    path = tmp_path / "run.jsonl"
    _write_tape(path)
    report = replay(path)
    assert report.ok and report.n == 24


def test_replay_detects_tampering(tmp_path):
    path = tmp_path / "run.jsonl"
    _write_tape(path)
    recs = list(read_tape(path))
    recs[3]["reward"] += 1e-9
    recs[5]["problem"]["target_value_si"] *= 1.01
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    report = replay(path)
    assert not report.ok
    assert report.reward_mismatches == [3]
    assert report.problem_mismatches == [5]

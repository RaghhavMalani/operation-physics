"""E1 (D52) on CPU: prefixes, the σ-only reward, bin mapping, tape replay, and the eval + compare path with a
fake backend that writes back the SFT model's own σ suffix (so E1 must equal SFT exactly)."""

import json
import math
from pathlib import Path

import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.env.generator import env_config, problem_from_dict
from physcal.eval.run_eval import Generation
from physcal.tape.tape import TapeWriter, read_tape, replay
from physcal.train import e1
from physcal.verify.reward import reward_config

E1 = load_config("e1")
EVAL_TAPE = REPO_ROOT / E1["eval_tapes"][0]
RW = reward_config()


class FakeTok:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **_):
        return "<prompt>" + json.dumps(messages) + "<assistant>"


def test_truncate_after_pm():
    assert e1.truncate_after_pm("x\n<answer>12.4 ± 0.8 m/s</answer>") == "x\n<answer>12.4 ±"
    assert e1.truncate_after_pm("<answer>12.4 +/- 0.8 m/s</answer>") == "<answer>12.4 +/-"
    assert e1.truncate_after_pm("a ± b <answer>12.4 m/s</answer>") is None  # ± outside the answer only
    assert e1.truncate_after_pm("no answer ± here") is None


def test_original_suffix_reproduces_the_sft_reward_exactly():
    rows, excluded = e1.build_rows([EVAL_TAPE], FakeTok(), E1["prompt"], {})
    logged = {tuple((r["problem"]["family"], r["problem"]["depth"], r["problem"]["seed"])): r for r in read_tape(EVAL_TAPE)}
    assert len(rows) + sum(excluded.values()) == len(logged)
    for r in rows:
        orig = logged[tuple(r["key"])]
        suffix = orig["completion"][len(r["prefix"]):]
        assert r["prefix"] + suffix == orig["completion"]
        s = e1.score_suffix(problem_from_dict(json.loads(r["problem_json"])), r["prefix"], suffix, r["orig_mu"], RW,
                            "free", E1["bins"])
        assert s["verifier_reward"] == orig["reward"] and s["train_reward"] == orig["reward"]
        assert not s["mu_changed"]


def test_bins():
    bins = E1["bins"]
    assert e1.nearest_bin(0.0, bins) == 0.001 and e1.nearest_bin(0.0004, bins) == 0.001
    assert e1.nearest_bin(0.015, bins) == 0.02 and e1.nearest_bin(0.4, bins) == 0.3 and e1.nearest_bin(7.0, bins) == 1.0
    assert e1.bin_entropy_bits([0.01] * 8, bins) == 0.0
    assert e1.bin_entropy_bits([0.001, 0.01], bins) == pytest.approx(1.0)


def _one_row():
    rows, _ = e1.build_rows([EVAL_TAPE], FakeTok(), E1["prompt"], {})
    return rows[0], problem_from_dict(json.loads(rows[0]["problem_json"]))


def test_bin_mapping_changes_only_the_sigma_used():
    r, p = _one_row()
    mu = r["orig_mu"]
    unit = p.target_unit
    on_bin = e1.score_suffix(p, r["prefix"], f" {0.02 * mu / _si_per_unit(p):.6g} {unit}</answer>", mu, RW, "bins", E1["bins"])
    free = e1.score_suffix(p, r["prefix"], f" {0.02 * mu / _si_per_unit(p):.6g} {unit}</answer>", mu, RW, "free", E1["bins"])
    assert on_bin["sigma_used_rel"] == 0.02 and on_bin["verifier_reward"] == free["verifier_reward"]
    assert on_bin["train_reward"] == pytest.approx(free["train_reward"], abs=1e-6)
    off = e1.score_suffix(p, r["prefix"], f" {0.013 * mu / _si_per_unit(p):.6g} {unit}</answer>", mu, RW, "bins", E1["bins"])
    assert off["sigma_used_rel"] == 0.01


def _si_per_unit(p):
    return p.target_value_si / p.target_value


def test_unit_change_is_flagged_as_mu_changed():
    r, p = _one_row()
    s = e1.score_suffix(p, r["prefix"], " 1 mm</answer>" if p.target_si_unit == "m" else " 1 km/h</answer>",
                        r["orig_mu"], RW, "free", E1["bins"])
    assert (s["ok"] and s["mu_changed"]) or not s["ok"]


def test_e1_reward_logs_a_replayable_tape(tmp_path):
    rows, _ = e1.build_rows([EVAL_TAPE], FakeTok(), E1["prompt"], {})
    rows = rows[:4]
    logged = {tuple((x["problem"]["family"], x["problem"]["depth"], x["problem"]["seed"])): x for x in read_tape(EVAL_TAPE)}
    suffixes = [logged[tuple(r["key"])]["completion"][len(r["prefix"]):] for r in rows]
    suffixes[1] = " banana</answer>"  # a parse failure
    tape = TapeWriter(tmp_path / "rollouts.jsonl", "e1-test", {"env_config": env_config(), "reward_config": RW})
    fn = e1.E1Reward(RW, "bins", E1["bins"], tape, tmp_path / "m.jsonl", group_size=2, max_completion_tokens=24,
                     model_id="fake")
    out = fn(None, suffixes, [r["problem_json"] for r in rows], [r["prefix"] for r in rows], [r["orig_mu"] for r in rows])
    tape.close()
    assert len(out) == 4 and out[1] == RW["weights"]["final"] * -RW["scoring"]["crps_clip"]
    assert replay(tmp_path / "rollouts.jsonl").ok
    m = json.loads((tmp_path / "m.jsonl").read_text())
    assert m["parse_ok_rate"] == 0.75 and m["n"] == 4


class EchoBackend:
    """Writes back the SFT model's own suffix for every prefix."""

    lora_hash = None

    def __init__(self, rows, logged):
        self.map = {r["prompt"]: logged[tuple(r["key"])]["completion"][len(r["prefix"]):] for r in rows}

    def _generate(self, prompts, k):
        return [[Generation(self.map[p], 10, False)] for p in prompts]


def test_eval_and_compare_with_an_echo_model_equal_sft(tmp_path, monkeypatch):
    rows = e1.read_rows("eval")
    logged = {tuple((x["problem"]["family"], x["problem"]["depth"], x["problem"]["seed"])): x for x in read_tape(EVAL_TAPE)}
    run = tmp_path / "e1a-run"
    run.mkdir()
    TapeWriter(run / "rollouts.jsonl", "fake", {"env_config": env_config(), "reward_config": RW,
                                                 "train_config": {"e1_mode": "free"}}).close()
    e1.evaluate(run, "final_adapter", "final", backend=EchoBackend(rows, logged))
    assert replay(run / "e1_eval" / "final" / "rollouts.jsonl").ok
    from physcal.train import e1_compare

    cfg = load_config("e1")
    cfg["compare"]["runs"] = {"E1a": str(run), "E1b": str(tmp_path / "missing")}
    cfg["compare"]["output_dir"] = str(tmp_path / "out")
    cfg["compare"]["bootstrap"] = 200
    monkeypatch.setattr(e1_compare, "load_config", lambda name: cfg)
    if not (REPO_ROOT / cfg["compare"]["a5_sigma_tape"]).exists():
        pytest.skip("A5 baselines not fitted yet")
    e1_compare.compare()
    res = json.loads((tmp_path / "out" / "family_E.json").read_text())
    assert res["tests"]["E1a-P"]["mean_diff"] == 0.0 and res["tests"]["E1b"] == "not run"
    assert res["describe"]["E1a"]["mean_crps_log"] == pytest.approx(res["describe"]["SFT stated σ"]["mean_crps_log"])
    assert not any(v["rejected"] for v in res["holm"].values())


class EchoKBackend(EchoBackend):
    """The SFT suffix, k times per prefix (the sampled path's shape)."""

    def _generate(self, prompts, k):
        return [[Generation(self.map[p], 10, False) for _ in range(k)] for p in prompts]


def test_sampled_eval_logs_k_rows_per_prefix_and_replays(tmp_path, monkeypatch):
    """D62: --samples K writes K tagged rows per prefix; with identical samples they equal the greedy rows."""
    rows = e1.read_rows("eval")[:6]
    logged = {tuple((x["problem"]["family"], x["problem"]["depth"], x["problem"]["seed"])): x for x in read_tape(EVAL_TAPE)}
    run = tmp_path / "e1a-run"
    run.mkdir()
    TapeWriter(run / "rollouts.jsonl", "fake", {"env_config": env_config(), "reward_config": RW,
                                                 "train_config": {"e1_mode": "free"}}).close()
    monkeypatch.setattr(e1, "read_rows", lambda which: rows)
    e1.evaluate(run, "final_adapter", "greedy", backend=EchoBackend(rows, logged))
    e1.evaluate(run, "final_adapter", "sampled", backend=EchoKBackend(rows, logged), samples=3)
    greedy = [json.loads(x) for x in (run / "e1_eval" / "greedy" / "rows.jsonl").read_text().splitlines()]
    sampled = [json.loads(x) for x in (run / "e1_eval" / "sampled" / "rows.jsonl").read_text().splitlines()]
    assert "sample" not in greedy[0] and len(sampled) == 3 * len(greedy)
    assert [r["sample"] for r in sampled[:3]] == [0, 1, 2]
    for i, g in enumerate(greedy):
        assert all({k: v for k, v in s.items() if k != "sample"} == g for s in sampled[3 * i:3 * i + 3])
    assert replay(run / "e1_eval" / "sampled" / "rollouts.jsonl").ok


def test_a_zero_answer_is_scored_as_a_failed_suffix_not_a_crash():
    """D88: an X2 sample answered μ = 0; the μ-changed check divided by it. Now: no crash, the suffix fails (no
    relative σ exists for μ = 0), and a moved μ is still caught."""
    from physcal.env.generator import generate_problem
    from physcal.train.e1 import e1_row, score_suffix
    from physcal.verify.reward import reward_config

    p = generate_problem("kinematics", 1, 10_000_321, split="train")
    prefix = f"x\n<steps>\n</steps>\n<answer>0 ±"
    cfg = reward_config()
    s = score_suffix(p, prefix, f" 0.1 {p.target_unit}</answer>", 0.0, cfg, "free", [0.01])
    assert s["ok"] and not s["mu_changed"] and s["sigma_used_rel"] is None
    row = e1_row({"key": ["kinematics", 1, 10_000_321], "split": "x2"}, s, cfg["scoring"])
    assert row["fail"] and row["rel_sigma_used"] is None and row["abs_log_error"] is None
    moved = score_suffix(p, "x\n<steps>\n</steps>\n<answer>2 ±", f" 0.1 {p.target_unit}</answer>", 0.0, cfg, "free",
                         [0.01])
    assert moved["mu_changed"]

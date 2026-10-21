"""D106, the stage C pilot: E1d's settings with the stated changes; the prefixes (μ = f(x_meas), noisy only, the
frozen eval sets); the training reward (score_sigma, the floor for a failed or μ-moving suffix) and its replayable
tape; the OOM check; and the analysis (C1-C6) on synthetic eval tapes."""

import copy
import json
import math
import shutil

import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.env.generator import problem_from_dict
from physcal.train import stage_c
from physcal.verify import sigma_reward
from physcal.verify.reward import reward_config

DATA = REPO_ROOT / "results" / "stage_c" / "data"
need_data = pytest.mark.skipif(not (DATA / "eval_prefixes.jsonl").exists(), reason="stage C data not built")


def test_train_config_is_e1d_with_the_stated_changes():
    e1d, tc = load_config("grpo_e1d"), stage_c.train_config()
    assert tc["model"] == e1d["model"] and tc["lora"] == e1d["lora"] and tc["seed"] == e1d["seed"]
    changed = {k for k in e1d["grpo"] if tc["grpo"][k] != e1d["grpo"][k]}
    assert changed == {"max_prompt_length"} and tc["grpo"]["learning_rate"] == 5e-5
    assert tc["grpo"]["max_prompt_length"] + tc["grpo"]["max_completion_length"] <= tc["model"]["max_seq_length"]
    assert tc["guards"]["max_reserved_gib"] == 7.1 and tc["prompt"] == "v5_two_channel"
    assert {k: v for k, v in tc["guards"].items() if k != "max_reserved_gib"} == \
        {k: v for k, v in e1d["guards"].items() if k != "max_reserved_gib"}
    from physcal.train.e1 import grpo_args

    a = grpo_args(tc, REPO_ROOT, "x", 300)
    assert a["learning_rate"] == 5e-5 and a["scale_rewards"] is None and a["beta"] == 0.0


@need_data
def test_prefixes():
    rows = stage_c.read_rows("train")
    rep = json.loads((DATA / "data_report.json").read_text())
    assert len(rows) == 2250 and rep["train"]["noise_levels"] == [0.01, 0.02, 0.05]
    assert rep["train"]["prompt_tokens_max"] <= stage_c.train_config()["grpo"]["max_prompt_length"]
    for r in rows[:50]:
        p = json.loads(r["problem_json"])
        assert r["prefix"].endswith("±") and r["prompt"].endswith(r["prefix"])
        assert abs(r["orig_mu"] / p["noise"]["point_estimate_si"] - 1) < 5e-4  # μ = f(x_meas) at 4 s.f.
        assert "<confidence>" in r["prompt"]  # prompt v5's format block
        assert 6e7 <= p["seed"] < 6e7 + 3000
    ev = stage_c.read_rows("eval")
    assert len(ev) == 3060 and {r["split"] for r in ev} == {"a6_pairs", "a6b_pairs", "a6c_pairs"}


@need_data
def test_training_reward_and_replayable_tape(tmp_path):
    from physcal.tape.tape import TapeWriter

    rw13 = reward_config("reward_v13")
    row = stage_c.read_rows("train")[0]
    prob = problem_from_dict(json.loads(row["problem_json"]))
    star = prob.noise["sigma_star_log"]
    unit = row["orig_completion"].split("</answer>")[0].rsplit(" ", 1)[1]
    mu = float(row["prefix"].split("<answer>")[1].replace("±", "").strip())
    sfx = lambda s, u=unit: f" {mu * s:.2g} {u}</answer>"
    good = stage_c.score_suffix_v13(prob, row["prefix"], sfx(star), row["orig_mu"], rw13)
    wide = stage_c.score_suffix_v13(prob, row["prefix"], sfx(4 * star), row["orig_mu"], rw13)
    assert good["train_reward"] == good["breakdown"]["score_sigma"] > wide["train_reward"]
    junk = stage_c.score_suffix_v13(prob, row["prefix"], " oops", row["orig_mu"], rw13)
    assert junk["train_reward"] == -5.0
    path = tmp_path / "rollouts.jsonl"
    meta = {"env_config": load_config("env_v2"), "reward_config": rw13}
    with TapeWriter(path, "stage-c-test", meta) as tape:
        fn = stage_c.StageCReward(rw13, tape, tmp_path / "m.jsonl", 2, 24, "test")
        r = fn(None, [sfx(star), sfx(2 * star)], [row["problem_json"]] * 2, [row["prefix"]] * 2, [row["orig_mu"]] * 2)
    assert r[0] == good["train_reward"] and r[0] > r[1]
    assert json.loads((tmp_path / "m.jsonl").read_text())["zero_variance_groups"] == 0.0
    sigma_reward._DRAWS.clear()
    assert sigma_reward.replay(path) == {"n": 2, "reward_mismatches": [], "breakdown_mismatches": []}


def test_mu_moving_suffix_gets_the_floor():
    rw13 = reward_config("reward_v13")
    from physcal.env.generator import generate_problem

    p = generate_problem("kinematics", 1, 60000001, split="train", cfg=load_config("env_v2"), noise=0.02)
    from physcal.train.sft_phase12 import build_example

    pc = load_config("sft_phase12")
    ex = build_example(p, "implicit", pc, load_config("env_v2"), 50)
    from physcal.train.e1 import truncate_after_pm

    prefix = truncate_after_pm(ex["completion"])
    tail = ex["completion"][len(prefix):].split("</answer>")[0].strip()   # "σ unit"
    sig, unit = tail.split(" ", 1)
    other = {"m": "km", "km": "m", "s": "min", "min": "s"}.get(unit)
    if other is None:
        pytest.skip(f"no same-dimension swap for unit {unit}")
    _, bd = sigma_reward.reward(p, ex["completion"], rw13)
    s = stage_c.score_suffix_v13(p, prefix, f" {sig} {other}</answer>", bd["mu_si"], rw13)
    assert s["mu_changed"] and s["train_reward"] == -5.0


def test_oom_check_on_a_tiny_cpu_model():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    d = REPO_ROOT / "results/sft/qwen2.5-math-1.5b-instruct/merged"
    if not (d / "tokenizer.json").exists():
        pytest.skip("tokenizer not present")
    tok = AutoTokenizer.from_pretrained(str(d))
    torch.manual_seed(0)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                        tie_word_embeddings=True, pad_token_id=tok.pad_token_id))
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=4, target_modules=["q_proj"]))
    out = stage_c.oom_check(model, tok, ["short", "a much longer prompt than the other one", "mid one"], 24, 2)
    assert out["batch"] == 2 and out["tokens"] == len(tok("a much longer prompt than the other one")["input_ids"]) + 24
    assert all(p.grad is None for p in model.parameters())


# ------------------------------------------------------------------------------------------- the analysis (C1-C6)
def _tape(path, problems, sigma_of):
    from physcal.verify.reward import reward

    rw = reward_config()
    lines = []
    for p in problems:
        mu = p["noise"]["point_estimate_si"]
        text = f"<steps>\n</steps>\n<answer>{mu!r} ± {sigma_of(p) * abs(mu)!r} {p['target_si_unit']}</answer>"
        r, bd = reward(problem_from_dict({k: v for k, v in p.items() if k != "_set"}), text, rw)
        lines.append(json.dumps({"split": p["_set"], "problem": {k: v for k, v in p.items() if k != "_set"},
                                 "completion": text, "reward": r, "breakdown": bd}))
    path.mkdir(parents=True)
    (path / "rollouts.jsonl").write_text("\n".join(lines) + "\n")


@pytest.mark.skipif(not (REPO_ROOT / "results/x1_text_sigma/a6c/predictions.jsonl").exists(), reason="no X1")
def test_analysis_c1_to_c6(tmp_path):
    from physcal.eval import noise_heuristics, stage_a
    from physcal.eval import stage_c_analysis as A

    problems = stage_a.load_sets(load_config("stage_a"))
    _tape(tmp_path / "final" / "seed0" / "eval", problems, lambda p: p["noise"]["sigma_star_log"])
    _tape(tmp_path / "sft_control" / "seed0" / "eval", problems, lambda p: noise_heuristics.sigma(p, "rss", "givens"))
    cfg = copy.deepcopy(A._config())
    cfg["analysis"]["bootstrap"] = 200
    res = A.analyze(cfg, tmp_path)
    t = res["tests"]
    assert all(t[k]["holm"]["rejected"] for k in A.TESTS)
    assert t["C6"]["value"] > 0 and res["arms"]["step150"] is None
    assert res["reading"][0].startswith("σ-RL propagates") and "RL moved σ" in res["reading"][1]
    assert (tmp_path / "summary.md").exists() and (tmp_path / "final" / "results.json").exists()

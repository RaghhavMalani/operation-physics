"""X3 (D79): targets, tokenization, the CPU training path and the analysis on synthetic student rows."""

import copy
import json
import math

import numpy as np
import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.train import x3

TOKENIZER_DIR = REPO_ROOT / "results" / "sft" / "qwen2.5-math-1.5b-instruct" / "merged"


def _need_tokenizer():
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        pytest.skip("local tokenizer not available")


def test_oracle_sigma_minimizes_the_single_answer_crps_log():
    from physcal.verify.scoring import crps_standard

    for mu, truth in ((1.03, 1.0), (0.5, 2.0), (1.0 + 1e-5, 1.0)):
        e = math.log(mu / truth)
        crps = lambda s: s * crps_standard(e / s)
        best = x3.oracle_sigma_log(mu, truth, 1e-9)
        grid = best * np.exp(np.linspace(-1, 1, 201))
        assert min(grid, key=crps) == pytest.approx(best, rel=0.011)
    assert x3.oracle_sigma_log(1.0, 1.0, 1e-6) == pytest.approx(1e-6 / x3.ORACLE_Z)


def _solution_completion(sigma_rel=0.02):
    from physcal.env.generator import generate_problem
    from physcal.train.sft_data import solution

    ex = solution(generate_problem("kinematics", 2, 10_000_123, split="train"), sigma_rel)
    return ex.problem, ex.completion


@pytest.mark.parametrize("target", [0.0123, 0.4, 3e-5])
def test_replaced_sigma_reads_back_through_the_verifier(target):
    from physcal.train.e1 import truncate_after_pm
    from physcal.verify.reward import reward, reward_config

    problem, completion = _solution_completion()
    prefix = truncate_after_pm(completion)
    new = x3.replace_sigma(prefix, completion[len(prefix):], target, ".2g")
    rw = reward_config()
    _, before = reward(problem, completion, rw)
    _, after = reward(problem, prefix + new, rw)
    assert after["reason"] == "ok" and after["mu_si"] == before["mu_si"]
    assert after["sigma_si"] / abs(after["mu_si"]) == pytest.approx(target, rel=0.05)
    assert new.endswith("</answer>") and completion.endswith(new[new.index(" ", 1):])  # unit and tag unchanged


def test_percent_sigma_stays_a_percentage():
    prefix = "x\n<answer>12.4 m/s ±"
    assert x3.replace_sigma(prefix, " 2% </answer>", 0.05, ".2g") == " 5% </answer>"
    assert x3.replace_sigma(prefix, " m/s</answer>", 0.05, ".2g") is None  # no σ number to replace


def test_tokenized_rows_train_only_the_suffix():
    _need_tokenizer()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    problem, completion = _solution_completion()
    from physcal.train.e1 import truncate_after_pm

    prefix = truncate_after_pm(completion)
    row = {"problem_json": json.dumps(problem.to_dict()), "prefix": prefix,
           "suffix_a": x3.replace_sigma(prefix, completion[len(prefix):], 0.07, ".2g")}
    t = x3.tokenize_row(row, "a", tok, "v4_worked_example", {})
    n_ctx = sum(1 for x in t["labels"] if x == -100)
    target = [x for x in t["labels"] if x != -100]
    assert t["input_ids"][n_ctx:] == target and len(target) == t["n_target"]
    assert tok.decode(target) == row["suffix_a"] + "<|im_end|>"
    from physcal.prompt import build_messages

    prompt = tok.apply_chat_template(build_messages(problem.text, "v4_worked_example"), tokenize=False,
                                     add_generation_prompt=True)
    assert t["input_ids"][:n_ctx] == tok(prompt + prefix, add_special_tokens=False)["input_ids"]  # as the eval feeds it


def test_committed_data_reproduces_a5_and_reads_back():
    rep_path = REPO_ROOT / "results" / "x3" / "data" / "data_report.json"
    if not rep_path.exists():
        pytest.skip("X3 data not built")
    rep = json.loads(rep_path.read_text())
    assert rep["teacher_check"]["max_abs_diff"] == 0.0
    assert rep["n_rows"] + sum(rep["excluded"].values()) == rep["n_answered"]
    assert max(rep["readback_max_rel_diff"].values()) <= x3._config()["target_check_rtol"]
    rows = x3.read_data()
    assert all(2e7 <= r["key"][2] < 2e7 + 2000 for r in rows)  # calibration seeds only, never eval
    b = [r["sigma_log_b"] for r in rows]
    assert np.allclose(b, [x3.oracle_sigma_log(r["mu_si"], r["truth_si"], 1e-6) for r in rows])


def test_fit_runs_on_cpu_with_a_tiny_model(tmp_path):
    _need_tokenizer()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    torch.manual_seed(0)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                        tie_word_embeddings=True, eos_token_id=tok.eos_token_id,
                                        pad_token_id=tok.pad_token_id))
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=4, lora_dropout=0.0,
                                            target_modules=["q_proj", "v_proj"]))
    from physcal.train.e1 import truncate_after_pm

    rows = []
    for seed in range(6):
        from physcal.env.generator import generate_problem
        from physcal.train.sft_data import solution

        ex = solution(generate_problem("dynamics", 1, 10_000_500 + seed, split="train"), 0.02)
        prefix = truncate_after_pm(ex.completion)
        rows.append({"problem_json": json.dumps(ex.problem.to_dict()), "prefix": prefix,
                     "suffix_b": x3.replace_sigma(prefix, ex.completion[len(prefix):], 0.01, ".2g")})
    cfg = copy.deepcopy(x3._config())
    cfg["train"].update(per_device_train_batch_size=2, gradient_accumulation_steps=1, logging_steps=1)
    s = x3.fit(model, tok, rows, "b", tmp_path, cfg, max_steps=2, bf16=False, optim="adamw_torch")
    assert s["steps"] == 2 and math.isfinite(s["train_loss"])
    assert (tmp_path / "final_adapter" / "adapter_model.safetensors").exists()
    assert s["steps_per_epoch"] == 3


def _fake_rows(path, keys, sigma_of):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for k, err in keys:
            s = sigma_of(err)
            f.write(json.dumps({"key": list(k), "split": "id", "fail": False, "reason": "ok", "mu_changed": False,
                                "rel_sigma_stated": s, "rel_sigma_used": s, "crps_log": -0.1, "cov1": True,
                                "cov2": True, "abs_log_error": err, "source_sample": None}) + "\n")


def test_analysis_on_synthetic_students(tmp_path, monkeypatch):
    from physcal.train.e1_compare import tape_arm

    cfg = copy.deepcopy(x3._config())
    for p in (cfg["eval"]["a5_sigma_tape"], cfg["eval"]["eval_tape"]):
        if not (REPO_ROOT / p).exists():
            pytest.skip("4-bit eval tapes not available")
    cfg["output_dir"] = str(tmp_path)
    cfg["analysis"]["bootstrap"] = 200
    monkeypatch.setattr(x3, "_config", lambda: cfg)
    teacher = tape_arm(REPO_ROOT / cfg["eval"]["a5_sigma_tape"])
    keys = [(k, v["abs_log_error"]) for k, v in teacher.items() if not v["fail"] and v["abs_log_error"] is not None]
    tsig = {k: teacher[k]["rel_sigma"] for k, _ in keys}
    (tmp_path / "a" / "eval_eval").mkdir(parents=True)
    with (tmp_path / "a" / "eval_eval" / "rows.jsonl").open("w") as f:  # student (a) = the teacher exactly
        for k, err in keys:
            f.write(json.dumps({"key": list(k), "split": "id", "fail": False, "reason": "ok", "mu_changed": False,
                                "rel_sigma_stated": tsig[k], "rel_sigma_used": tsig[k], "crps_log": -0.1,
                                "cov1": True, "cov2": True, "abs_log_error": err, "source_sample": None}) + "\n")
    rng = np.random.default_rng(0)
    _fake_rows(tmp_path / "b" / "eval_eval" / "rows.jsonl", keys, lambda e: float(rng.uniform(0.001, 0.1)))
    res = x3.analyze()
    a, b = res["arms"]["a"], res["arms"]["b"]
    assert a["fidelity_spearman_vs_teacher"] == pytest.approx(1.0)
    assert a["rho"]["value"] == pytest.approx(res["reference"]["A5 teacher σ (4-bit)"]["spearman_sigma_error"])
    assert a["rho_minus_teacher"]["value"] == pytest.approx(0.0, abs=1e-12)
    assert res["label_a"] == x3.label(a["rho"], 0.5)
    assert abs(b["rho"]["value"]) < 0.15 and res["rho_b_minus_a"]["value"] < 0
    assert (tmp_path / "pilot.md").exists()


def test_label_rule():
    assert x3.label({"value": 0.6, "ci95": [0.52, 0.68]}, 0.5) == "capacity shown"
    assert x3.label({"value": 0.55, "ci95": [0.45, 0.62]}, 0.5) == "capacity likely"
    assert x3.label({"value": 0.3, "ci95": [0.2, 0.4]}, 0.5) == "not shown at the pilot budget"

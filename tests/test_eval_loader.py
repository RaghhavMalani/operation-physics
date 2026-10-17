"""D65 / D67 / D68: adapters are evaluated unmerged on their 4-bit training base (`unsloth_4bit`, the default with an
adapter), and RL adapters are never merged. The loader itself needs CUDA; these CPU tests check the defaults, that the
4-bit loader takes its settings from the run's training config, and that the merge guard fires before anything is
loaded."""

import json

import pytest

from physcal.adapters import MergeRefused, check_merge_allowed, rl_adapter_evidence
from physcal.config import REPO_ROOT
from physcal.env.generator import env_config
from physcal.eval.run_eval import HFBackend, default_loader
from physcal.tape.tape import TapeWriter, read_meta, replay
from physcal.train import e1
from physcal.train.compare import eval_config
from physcal.verify.reward import reward_config

LABEL = "qwen2.5-math-1.5b-instruct"
SPLIT0 = REPO_ROOT / "results/train/split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed0"
REAL_RL = [SPLIT0 / "permanent_checkpoints/checkpoint-287", REPO_ROOT / "results/train/e1a-sigma-only-free-seed0/final_adapter"]
REAL_SFT = REPO_ROOT / "results/sft/qwen2.5-math-1.5b-instruct/adapter"


def fake_run(tmp_path, train_config, name="run"):
    run = tmp_path / name
    run.mkdir()
    TapeWriter(run / "rollouts.jsonl", "fake", {"env_config": env_config(), "reward_config": reward_config(),
                                                 "train_config": train_config}).close()
    return run


def fake_adapter(d, card=""):
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "x"}))
    (d / "README.md").write_text(card)
    return d


def test_compare_defaults_to_the_4bit_training_loader_for_sft_and_rl(tmp_path):
    run = fake_run(tmp_path, {"model": {"max_seq_length": 1234, "text_only": False}})
    for part in ("rl", "sft"):
        default = eval_config(part, LABEL, 512, run, tmp_path / "sft", "a", "x")
        assert default == eval_config(part, LABEL, 512, run, tmp_path / "sft", "a", "x", loader="unsloth_4bit")
        assert default["model"]["loader"] == "unsloth_4bit"
        assert default["model"]["max_seq_length"] == 1234 and default["model"]["text_only"] is False
    rl, sft = (eval_config(p, LABEL, 512, run, tmp_path / "sft", "a", "x") for p in ("rl", "sft"))
    assert rl["model"]["lora"] == str(run / "a") and sft["model"]["lora"] is None
    base = eval_config("base", LABEL, 512, run, tmp_path / "sft", "a", "x")
    assert base["model"]["loader"] == "transformers" and "max_seq_length" not in base["model"]
    bf16 = eval_config("rl", LABEL, 512, run, tmp_path / "sft", "a", "x", loader="transformers")
    assert bf16["generation"] == rl["generation"] and bf16["splits"] == rl["splits"]


def test_default_loader_depends_on_the_adapter():
    assert default_loader({"lora": "some/adapter"}) == "unsloth_4bit"
    assert default_loader({"lora": None}) == default_loader({}) == "transformers"


@pytest.mark.parametrize("model_cfg,match", [
    ({"loader": "vllm"}, "unknown loader"),
    ({"loader": "unsloth_4bit", "device": "cpu"}, "needs CUDA"),
    ({"lora": "some/adapter", "device": "cpu"}, "needs CUDA"),  # an adapter defaults to the 4-bit loader
])
def test_backend_refuses_bad_loader_settings_before_loading(model_cfg, match):
    with pytest.raises(ValueError, match=match):
        HFBackend("does-not-exist", {"id": "does-not-exist", "dtype": "bfloat16", **model_cfg}, {})


def test_rl_adapter_evidence_from_the_run_meta_and_the_model_card(tmp_path):
    rl_run = fake_run(tmp_path, {"grpo": {"max_steps": 3}, "model": {}}, "rl-run")
    sft_run = fake_run(tmp_path, {"model": {}}, "sft-run")
    assert rl_adapter_evidence(fake_adapter(rl_run / "permanent_checkpoints/checkpoint-3"))  # two levels up
    assert rl_adapter_evidence(fake_adapter(rl_run / "final_adapter"))
    assert rl_adapter_evidence(fake_adapter(tmp_path / "copied", card="Trained with GRPO, a method ..."))
    assert rl_adapter_evidence(fake_adapter(sft_run / "adapter", card="Trained with SFT.")) == []
    assert rl_adapter_evidence(fake_adapter(tmp_path / "bare")) == []


def test_merge_guard_refuses_rl_adapters_always_and_others_unless_requested(tmp_path):
    rl = fake_adapter(fake_run(tmp_path, {"grpo": {}, "model": {}}, "rl-run") / "final_adapter")
    other = fake_adapter(tmp_path / "other")
    with pytest.raises(MergeRefused, match="RL adapter"):
        check_merge_allowed(rl, requested=True)
    with pytest.raises(MergeRefused, match="not requested"):
        check_merge_allowed(other, requested=False)
    check_merge_allowed(other, requested=True)
    for cfg in ({"lora": str(rl), "loader": "transformers", "merge_adapter": True},
                {"lora": str(other), "loader": "transformers"}):
        with pytest.raises(MergeRefused):  # before any model is loaded (the model id does not exist)
            HFBackend("does-not-exist", {"id": "does-not-exist", "dtype": "bfloat16", **cfg}, {})


def test_merge_tool_refuses_rl_adapters(tmp_path):
    from physcal.train.merge import merge

    rl = fake_adapter(fake_run(tmp_path, {"grpo": {}, "model": {}}, "rl-run") / "final_adapter")
    with pytest.raises(MergeRefused):
        merge("does-not-exist", None, str(rl), str(tmp_path / "out"))


def test_the_real_adapters_are_classified():
    for d in REAL_RL:
        if d.exists():
            assert rl_adapter_evidence(d), d
    if REAL_SFT.exists():
        assert rl_adapter_evidence(REAL_SFT) == []


class FixedBackend:
    """Answers every E1 prefix with the same σ suffix; carries a loader check like the 4-bit backend."""
    loader = "unsloth_4bit"
    loader_check = {"loader": "unsloth_4bit", "fake": True}

    def _generate(self, prompts, k):
        from physcal.eval.run_eval import Generation

        return [[Generation(" 0.01 m</answer>", 5, False)] * k for _ in prompts]


def test_e1_eval_records_the_loader_and_replays(tmp_path, monkeypatch):
    rows = e1.read_rows("eval")[:3]
    monkeypatch.setattr(e1, "read_rows", lambda which: rows)
    run = fake_run(tmp_path, {"e1_mode": "free", "model": {"max_seq_length": 2048}})
    e1.evaluate(run, "final_adapter", "q", backend=FixedBackend())
    meta = read_meta(run / "e1_eval" / "q" / "rollouts.jsonl")
    assert meta["loader"] == "unsloth_4bit" and meta["loader_check"] == FixedBackend.loader_check
    assert meta["adapter"] == str(run / "final_adapter")
    assert replay(run / "e1_eval" / "q" / "rollouts.jsonl").ok


def test_e1_sft_eval_has_no_adapter_and_scores_like_e1a(tmp_path, monkeypatch):
    """D67: family E's SFT control: the SFT model's own σ suffix on the same prefixes, unbinned."""
    rows = e1.read_rows("eval")[:3]
    monkeypatch.setattr(e1, "read_rows", lambda which: rows)
    run = fake_run(tmp_path, {"e1_mode": "free", "model": {"max_seq_length": 2048}})
    e1.evaluate(run, "final_adapter", "q", backend=FixedBackend())
    e1.evaluate_sft(tmp_path / "sft", backend=FixedBackend())
    meta = read_meta(tmp_path / "sft" / "rollouts.jsonl")
    assert meta["adapter"] is None and meta["e1_mode"] == "free" and meta["loader"] == "unsloth_4bit"
    assert (tmp_path / "sft" / "rows.jsonl").read_text() == (run / "e1_eval" / "q" / "rows.jsonl").read_text()
    assert replay(tmp_path / "sft" / "rollouts.jsonl").ok
    cfg = e1._config()["sft_eval"]
    from physcal.config import load_config

    assert load_config(cfg["loader_settings_from"])["model"]["id"] == e1._config()["model_id"]

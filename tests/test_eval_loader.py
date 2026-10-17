"""D65: the `unsloth_4bit` eval loader option (the RL training loader at eval). The loader itself needs CUDA; these CPU
tests check the plumbing: the default path is unchanged, the option takes its settings from the run's training config,
and bad settings are refused before anything is loaded."""

import pytest

from physcal.env.generator import env_config
from physcal.eval.run_eval import HFBackend
from physcal.tape.tape import TapeWriter, read_meta, replay
from physcal.train import e1
from physcal.train.compare import eval_config
from physcal.verify.reward import reward_config

LABEL = "qwen2.5-math-1.5b-instruct"


def fake_run(tmp_path, train_config):
    run = tmp_path / "run"
    run.mkdir()
    TapeWriter(run / "rollouts.jsonl", "fake", {"env_config": env_config(), "reward_config": reward_config(),
                                                 "train_config": train_config}).close()
    return run


def test_compare_default_model_block_is_unchanged_and_4bit_reads_the_training_config(tmp_path):
    run = fake_run(tmp_path, {"model": {"max_seq_length": 1234, "text_only": False}})
    default = eval_config("rl", LABEL, 512, run, tmp_path / "sft", "a", "x")
    explicit = eval_config("rl", LABEL, 512, run, tmp_path / "sft", "a", "x", loader="transformers")
    assert default == explicit and "loader" not in default["model"] and "max_seq_length" not in default["model"]
    q = eval_config("rl", LABEL, 512, run, tmp_path / "sft", "a", "x", loader="unsloth_4bit")
    assert q["model"] == default["model"] | {"loader": "unsloth_4bit", "max_seq_length": 1234, "text_only": False}
    assert q["generation"] == default["generation"] and q["splits"] == default["splits"]


@pytest.mark.parametrize("model_cfg,match", [
    ({"loader": "vllm"}, "unknown loader"),
    ({"loader": "unsloth_4bit", "device": "cpu"}, "needs CUDA"),
])
def test_backend_refuses_bad_loader_settings_before_loading(model_cfg, match):
    with pytest.raises(ValueError, match=match):
        HFBackend("does-not-exist", {"id": "does-not-exist", "dtype": "bfloat16", **model_cfg}, {})


class FixedBackend:
    """Answers every E1 prefix with the same σ suffix; carries a loader check like the 4-bit backend."""
    loader_check = {"loader": "unsloth_4bit", "fake": True}

    def _generate(self, prompts, k):
        from physcal.eval.run_eval import Generation

        return [[Generation(" 0.01 m</answer>", 5, False)] * k for _ in prompts]


def test_e1_eval_records_the_loader_and_replays(tmp_path, monkeypatch):
    rows = e1.read_rows("eval")[:3]
    monkeypatch.setattr(e1, "read_rows", lambda which: rows)
    run = fake_run(tmp_path, {"e1_mode": "free", "model": {"max_seq_length": 2048}})
    e1.evaluate(run, "final_adapter", "bf16", backend=FixedBackend())
    e1.evaluate(run, "final_adapter", "q", backend=FixedBackend(), loader="unsloth_4bit")
    assert "loader" not in read_meta(run / "e1_eval" / "bf16" / "rollouts.jsonl")
    meta = read_meta(run / "e1_eval" / "q" / "rollouts.jsonl")
    assert meta["loader"] == "unsloth_4bit" and meta["loader_check"] == FixedBackend.loader_check
    assert (run / "e1_eval" / "q" / "rows.jsonl").read_text() == (run / "e1_eval" / "bf16" / "rows.jsonl").read_text()
    assert replay(run / "e1_eval" / "q" / "rollouts.jsonl").ok

"""D104, Phase 12 stage B: the data (D80's supervised arm, checked against its manifest), the trainer on a tiny CPU
model (fit + the D103 OOM check), and the eval config (pinned base, adapter unmerged, A6 / A6b / A6c, prompt v5)."""

import copy
import math

import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.train import stage_b

DATA = REPO_ROOT / "results" / "phase12_sft_data" / "supervised.jsonl"


def _tok():
    from physcal.base import base_config, pinned_base_dir

    try:
        d = pinned_base_dir(base_config("base_4bit"))
    except Exception:
        pytest.skip("the pinned base snapshot is not cached")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(d)


def test_config_matches_d104():
    cfg = stage_b._config()
    assert cfg["arm"] == "supervised" and cfg["seeds"] == [0, 1, 2] and cfg["pinned_base"] == "base_4bit"
    sft = load_config("sft")
    for k in ("per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type",
              "warmup_steps", "weight_decay", "max_grad_norm", "optim"):
        assert cfg["train"][k] == sft["train"][k], k
    assert cfg["train"]["epochs"] == 2 and cfg["max_reserved_gib"] == 7.1
    assert cfg["lora"] == sft["lora"]


@pytest.mark.skipif(not DATA.exists(), reason="Phase 12 SFT data not built")
def test_rows_are_the_supervised_arm_with_sigma_trained():
    tok = _tok()
    cfg = stage_b._config()
    rows, info = stage_b.tokenized_rows(tok, cfg, 2048)
    assert info["n_rows"] == 3000 and info["dropped_too_long"] == 0 and info["prompt"] == "v5_two_channel"
    assert info["n_masked_mean"] == 0.0  # the supervised arm trains sigma (and c)
    r = next(r for r in rows if r["noise"] > 0)
    text = tok.decode([t for t in r["labels"] if t != -100])
    assert "±" in text and "<confidence>" in text and " ± 0 " not in text


def test_data_check_refuses_a_changed_file(tmp_path, monkeypatch):
    cfg = copy.deepcopy(stage_b._config())
    pc = copy.deepcopy(load_config(cfg["sft_phase12_config"]))
    src = REPO_ROOT / pc["output_dir"]
    if not (src / "supervised.jsonl").exists():
        pytest.skip("Phase 12 SFT data not built")
    for name in ("manifest.json", "problems.jsonl"):
        (tmp_path / name).write_bytes((src / name).read_bytes())
    (tmp_path / "supervised.jsonl").write_bytes((src / "supervised.jsonl").read_bytes() + b"\n")
    pc["output_dir"] = str(tmp_path)
    monkeypatch.setattr(stage_b, "load_config", lambda name: pc if name == cfg["sft_phase12_config"] else load_config(name))
    with pytest.raises(ValueError, match="manifest"):
        stage_b.check_data(cfg)


def test_fit_and_oom_check_on_a_tiny_cpu_model(tmp_path):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    if not DATA.exists():
        pytest.skip("Phase 12 SFT data not built")
    tok = _tok()
    cfg = copy.deepcopy(stage_b._config())
    rows, _ = stage_b.tokenized_rows(tok, cfg, 2048)
    rows = rows[:6]
    torch.manual_seed(0)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                        tie_word_embeddings=True, eos_token_id=tok.eos_token_id,
                                        pad_token_id=tok.pad_token_id))
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=4, lora_dropout=0.0, target_modules=["q_proj", "v_proj"]))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    oom = stage_b.oom_check(model, tok, rows, 2)
    assert oom["batch"] == 2 and oom["tokens"] == max(len(r["input_ids"]) for r in rows) and math.isfinite(oom["loss"])
    assert all(p.grad is None for p in model.parameters())
    cfg["train"].update(per_device_train_batch_size=2, gradient_accumulation_steps=1, logging_steps=1)
    s = stage_b.fit(model, tok, rows, tmp_path, cfg, seed=1, max_steps=2, bf16=False, optim="adamw_torch")
    assert s["steps"] == 2 and s["steps_per_epoch"] == 3 and math.isfinite(s["train_loss"])
    assert (tmp_path / "final_adapter" / "adapter_model.safetensors").exists()


def test_eval_config_is_the_pinned_base_with_the_adapter_unmerged():
    try:
        ev = stage_b.eval_config(1)
    except Exception as exc:
        pytest.skip(f"pinned base not cached: {exc}")
    from physcal.base import base_config, pinned_base_dir

    m = ev["model"]
    assert m["id"] == pinned_base_dir(base_config("base_4bit")) and m["pinned_base"] == "base_4bit"
    assert m["loader"] == "unsloth_4bit" and m["lora"].endswith("results/stage_b/seed1/final_adapter")
    assert m["revision"] is None and m["max_reserved_gib"] == 7.1
    assert set(ev["splits"]) == {"a6_pairs", "a6b_pairs", "a6c_pairs"}
    assert all(s["verify_regenerate"] == -1 for s in ev["splits"].values())
    assert ev["prompt"] == "v5_two_channel" and ev["env_config"] == "env_v2"
    g = ev["generation"]
    assert not g["do_sample"] and g["max_new_tokens"] == 512 and g["num_samples"] == 1 and g["batch_size"] == 32
    assert ev["output_dir"].endswith("results/stage_b/seed1/eval")
    assert "a6c_pairs" not in load_config("eval_frozen_sets")["splits"]  # the shared config is not modified

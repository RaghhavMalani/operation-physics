"""Model probe: footprint estimator and the selection rule (no GPU, no downloads)."""

from pathlib import Path

import pytest

from physcal.config import load_config
from physcal.eval import footprint
from physcal.eval.model_probe import candidate_eval_config, select

FP = load_config("footprint")
PROBE = load_config("model_probe")

# A toy 2-layer hybrid model: layer 0 linear attention, layer 1 full attention, plus a vision tower.
CFG = {"text_config": {
    "num_hidden_layers": 2, "hidden_size": 8, "intermediate_size": 16, "vocab_size": 100,
    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
    "layer_types": ["linear_attention", "full_attention"],
    "linear_num_value_heads": 2, "linear_key_head_dim": 4, "linear_value_head_dim": 4,
}}
TENSORS = {
    "model.language_model.embed_tokens.weight": [100, 8],
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": [24, 8],
    "model.language_model.layers.0.linear_attn.conv1d.weight": [24, 1, 4],
    "model.language_model.layers.0.linear_attn.A_log": [2],
    "model.language_model.layers.1.self_attn.q_proj.weight": [8, 8],
    "model.language_model.layers.1.mlp.down_proj.weight": [8, 16],
    "model.language_model.layers.1.input_layernorm.weight": [8],
    "model.visual.blocks.0.attn.qkv.weight": [24, 8],
    "mtp.fc.weight": [8, 16],
}


def test_shape_accounting():
    s = footprint.shape_from(CFG, TENSORS)
    assert s.params_embedding == 800
    assert s.params_quantized == 24 * 8 + 8 * 8 + 8 * 16
    assert s.lora_params_per_rank == (24 + 8) + (8 + 8) + (8 + 16)
    assert s.params_other == 24 * 4 + 2 + 8
    assert s.params_excluded == 24 * 8 + 8 * 16  # vision tower + MTP head
    assert s.kv_bytes_per_token == 2 * 1 * 1 * 4 * 2  # one full-attention layer
    assert s.recurrent_state_bytes == 1 * 2 * 4 * 4 * 4  # one linear-attention layer


def test_estimate_is_monotone_and_pessimistic_is_larger():
    s = footprint.shape_from(CFG, TENSORS)
    base = footprint.estimate_grpo(s, 100, FP)["peak_device_gib"]
    longer = dict(FP, grpo=dict(FP["grpo"], completion_tokens=FP["grpo"]["completion_tokens"] * 2))
    assert footprint.estimate_grpo(s, 100, longer)["peak_device_gib"] > base
    assert footprint.estimate_grpo(s, 200, FP)["peak_device_gib"] > base
    assert footprint.estimate_grpo(s, 100, FP, pessimistic=True)["peak_device_gib"] >= base


def _row(label, zv, fits=True, acc=0.0):
    return {"label": label, "group": {"zero_variance": {"binary": zv}},
            "greedy": {"id": {"accuracy_sigma_free": {"0.01": acc}}},
            "footprint": {"peak_device_gb": 4.0 if fits else 9.0}}


def test_selection_rule():
    rows = [_row("a", 0.70), _row("b", 0.40, fits=False), _row("c", 0.55)]
    sel = select(rows, PROBE, FP)
    assert sel["decision"] == "train" and sel["selected"] == "c"  # b is lower but does not fit
    rows = [_row("a", 0.90), _row("b", 0.85)]
    sel = select(rows, PROBE, FP)
    assert sel["decision"] == "no_training_sft_warmup" and sel["best_label"] == "b"
    assert select([_row("a", 0.5), {"label": "x", "missing": True}], PROBE, FP)["decision"] == "incomplete"
    tie = select([_row("a", 0.5, acc=0.01), _row("b", 0.5, acc=0.03)], PROBE, FP)
    assert tie["selected"] == "b"


def test_candidate_config_overrides():
    base = load_config(PROBE["group_config"])
    cand = PROBE["candidates"][-1]
    cfg = candidate_eval_config(base, cand, "group", Path("/tmp/x"))
    assert cfg["model"]["id"] == cand["id"] and cfg["model"]["revision"] == cand["revision"]
    assert cfg["model"]["fallback_id"] is None
    assert cfg["generation"]["num_samples"] == 8 and cfg["generation"]["max_new_tokens"] == 1024
    assert cfg["prompt"] == "v4_worked_example"
    assert base["model"]["revision"] is None  # base config untouched


def test_every_candidate_is_pinned_to_a_full_revision():
    for cand in PROBE["candidates"]:
        assert len(cand["revision"]) == 40 and int(cand["revision"], 16) >= 0


def test_footprint_reads_local_model_directories(tmp_path):
    import json as _json

    import torch
    from safetensors.torch import save_file

    (tmp_path / "config.json").write_text(_json.dumps(CFG))
    save_file({k: torch.zeros(v) for k, v in TENSORS.items()}, str(tmp_path / "model.safetensors"))
    assert footprint.model_shape(str(tmp_path), None) == footprint.shape_from(CFG, TENSORS)


def test_compare_configs_differ_only_in_the_model(tmp_path):
    from physcal.env.generator import env_config
    from physcal.tape.tape import TapeWriter
    from physcal.train.compare import eval_config

    run_dir, sft = tmp_path / "run", tmp_path / "sft"
    TapeWriter(run_dir / "rollouts.jsonl", "fake", {"env_config": env_config(),
                                                     "train_config": {"model": {"max_seq_length": 2048}}}).close()
    cfgs = {p: eval_config(p, "qwen3.5-2b", 512, run_dir, sft) for p in ("base", "sft", "rl")}
    assert [cfgs[p]["model"]["loader"] for p in ("base", "sft", "rl")] == ["transformers", "unsloth_4bit", "unsloth_4bit"]
    for c in cfgs.values():
        assert c["generation"]["max_new_tokens"] == 512 and c["prompt"] == "v4_worked_example"
        assert c["model"]["model_class"] == "causal_lm" and c["generation"]["do_sample"] is False
    assert cfgs["base"]["model"]["revision"] == "15852e8c16360a2fea060d615a32b45270f8a8fc"
    assert cfgs["sft"]["model"]["id"] == cfgs["rl"]["model"]["id"] == str(sft)
    assert cfgs["rl"]["model"]["lora"] == str(run_dir / "final_adapter") and cfgs["sft"]["model"]["lora"] is None
    assert cfgs["base"]["splits"] == cfgs["sft"]["splits"] == cfgs["rl"]["splits"]

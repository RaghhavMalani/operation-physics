"""D160 (exploratory follow-up track): the probes' random-weights control (transformers' Qwen2 init, tensor by tensor),
the cached probe values, and the paired comparisons. CPU only; a tiny Qwen2 stands in for the readout copies."""

import copy
import json

import numpy as np
import pytest

from physcal.eval import followup_scale as F
from physcal.eval import followup_scale_analysis as A

TINY = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 96,
        "max_position_embeddings": 64, "initializer_range": 0.02, "rms_norm_eps": 1e-6, "pad_token_id": 3,
        "torch_dtype": "bfloat16", "quantization_config": {"load_in_4bit": True, "quant_method": "bitsandbytes"}}


@pytest.fixture()
def tiny(tmp_path):
    def make(tied: bool, shard_gb: float = 2.0):
        src = tmp_path / f"copy_{tied}"
        src.mkdir(exist_ok=True)
        (src / "config.json").write_text(json.dumps(TINY | {"tie_word_embeddings": tied}))
        (src / "tokenizer_config.json").write_text('{"chat_template": "x"}')
        (src / "model.safetensors").write_text("the quantized weights, never copied")
        cfg = copy.deepcopy(F._config())
        cfg["output_dir"] = str(tmp_path / f"out_{tied}_{shard_gb}")
        cfg["scale"]["models"]["tiny"] = {"dir": str(src), "repo_id": "test/tiny", "revision": "0"}
        cfg["scale"]["random_init"]["shard_gb"] = shard_gb
        return cfg
    return make


def _weights(d):
    from safetensors.torch import load_file

    out = {}
    for f in sorted(d.glob("model-*.safetensors")):
        out |= load_file(str(f))
    return out


@pytest.mark.parametrize("tied", [False, True])
def test_random_init_follows_transformers_qwen2_init_and_loads(tiny, tied):
    import torch
    import transformers

    cfg = tiny(tied, shard_gb=1e-5)                       # every tensor its own shard: the index is exercised
    info = F.make_random_init("tiny", cfg)
    d = F.random_init_dir("tiny", cfg)
    assert info["seed"] == 0 and info["n_shards"] > 1
    assert "quantization_config" not in json.loads((d / "config.json").read_text())
    assert (d / "tokenizer_config.json").read_text() == '{"chat_template": "x"}'
    assert (d / "model.safetensors.index.json").exists() and not (d / "model.safetensors").exists()
    w = _weights(d)
    assert all(t.dtype == torch.bfloat16 for t in w.values())
    assert ("lm_head.weight" in w) is (not tied)
    normal = torch.cat([t.float().flatten() for n, t in w.items() if n.endswith("proj.weight") or "lm_head" in n])
    assert float(normal.mean()) == pytest.approx(0.0, abs=2e-3) and float(normal.std()) == pytest.approx(0.02, rel=0.05)
    emb = w["model.embed_tokens.weight"].float()
    assert torch.all(emb[3] == 0) and float(emb.std()) == pytest.approx(0.02, rel=0.1)
    assert all(torch.all(t == 0) for n, t in w.items() if n.endswith(".bias"))
    assert all(torch.all(t == 1) for n, t in w.items() if n.endswith("norm.weight"))
    model = transformers.AutoModelForCausalLM.from_pretrained(str(d), dtype=torch.bfloat16)
    sd = model.state_dict()
    assert all(torch.equal(sd[n], t) for n, t in w.items())
    assert F.make_random_init("tiny", cfg) == info          # idempotent: kept, not regenerated


def test_random_init_is_reproducible_and_seeded(tiny):
    a, b = tiny(False), tiny(False, shard_gb=1.0)
    ia, ib = F.make_random_init("tiny", a), F.make_random_init("tiny", b)
    wa, wb = _weights(F.random_init_dir("tiny", a)), _weights(F.random_init_dir("tiny", b))
    assert wa.keys() == wb.keys() and all((wa[n] == wb[n]).all() for n in wa)
    c = tiny(False, shard_gb=0.5)
    c["scale"]["random_init"]["seed"] = 1
    wc = _weights(F.random_init_dir("tiny", c)) if F.make_random_init("tiny", c) else None
    assert not (wa["model.embed_tokens.weight"] == wc["model.embed_tokens.weight"]).all()


def _fake_parts(cfg, model, kind, n_problems=6, k=4, dim=5, seed=0):
    rng = np.random.default_rng(seed)
    d = F.out_dir(model, cfg) / kind
    d.mkdir(parents=True, exist_ok=True)
    idx = [["kinematics", 1, 100 + p, s] for p in range(n_problems) for s in range(k)]
    layers = [rng.normal(size=(len(idx), dim)).astype(np.float16) for _ in range(3)]
    np.savez(d / "part_0000.npz", index=json.dumps(idx), **{f"layer_{i:02d}": a for i, a in enumerate(layers)})
    return {(("kinematics", 1, 100 + p), s): s % 2 == 0 for p in range(n_problems) for s in range(k)}


def test_probe_values_are_cached_by_their_inputs(tiny, monkeypatch):
    from physcal.eval import x2_analysis as XA

    cfg = tiny(False)
    corr = _fake_parts(cfg, "tiny", "hidden")
    calls = []

    def fake(layers, index, pc):
        calls.append(len(index))
        return {(tuple(r["key"]), r["sample"]): float(i) for i, r in enumerate(index)}, {"folds": []}

    monkeypatch.setattr(XA, "crossfit_confidence", fake)
    first = A.probe_confidence("tiny", "hidden", corr, cfg)
    assert A.probe_confidence("tiny", "hidden", corr, cfg)[0] == first[0] and calls == [24]
    flipped = {k: not v for k, v in corr.items()}              # other labels: other inputs, a refit
    A.probe_confidence("tiny", "hidden", flipped, cfg)
    assert calls == [24, 24]
    _fake_parts(cfg, "tiny", "hidden", seed=1)                 # other states: a refit
    A.probe_confidence("tiny", "hidden", flipped, cfg)
    assert calls == [24, 24, 24]
    assert A.probe_confidence("tiny", "hidden_random", corr, cfg) is None


def test_paired_difference_and_its_bootstrap():
    a, b = np.array([0.9, 0.8, 0.7, 0.6]), np.array([0.5, 0.5, 0.5, 0.5])
    r = A.paired(a, b, 2000, 0)
    assert r["n"] == 4 and r["diff"] == pytest.approx(0.25) and r["ci95"][0] <= 0.25 <= r["ci95"][1]
    assert A.paired(a, a, 2000, 0) == {"n": 4, "diff": 0.0, "ci95": [0.0, 0.0]}

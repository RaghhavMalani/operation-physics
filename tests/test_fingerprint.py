"""R2-1 (D72): the training-time fingerprint and the reloaded-model regression check, CPU only, tiny random Qwen2.

The check must pass for a faithful reload (the adapter attached unmerged to the same base) and fail for the two
loader bugs it exists to catch: an adapter that is silently dropped, and FINDINGS #15's bf16 merge, which rounds a
small update away."""

import copy
import json

import numpy as np
import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.eval import fingerprint as fpm

TOKENIZER_DIR = REPO_ROOT / "results" / "sft" / "qwen2.5-math-1.5b-instruct" / "merged"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def small_cfg(**tol):
    cfg = copy.deepcopy(load_config("fingerprint"))
    cfg["prompts"]["n"] = 4
    cfg["logits"]["top_k_sigma"] = 16
    cfg["sampling"].update(k=2, max_new_tokens=12, batch_size=4, min_parsed=1)
    cfg["tolerances"].update(tol)
    return cfg


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory):
    """A tiny random Qwen2 whose fp32 weights are exactly bf16-representable (a fixed low-precision base, as the
    4-bit base is in training), saved in fp32."""
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        pytest.skip("local tokenizer not available")
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    d = tmp_path_factory.mktemp("tiny_qwen_fp")
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      tie_word_embeddings=True, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(p.to(torch.bfloat16).float())
    m.save_pretrained(str(d))
    tok.save_pretrained(str(d))
    return d


def _trained(base_dir, b_std: float, seed: int = 1):
    """The 'training-time' policy: base + LoRA with B ~ N(0, b_std) (a fresh LoRA has B = 0)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(base_dir))
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.float32),
                           LoraConfig(r=4, lora_alpha=4, target_modules=TARGETS, lora_dropout=0.0))
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * b_std)
    return model.eval(), tok


def _reload_unmerged(base_dir, adapter_dir):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    return PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.float32),
                                     str(adapter_dir)).eval()


def _reload_bf16_merged(base_dir, adapter_dir):
    """FINDINGS #15's bug: merge into the bf16 base (PEFT's merge, W <- bf16(W + bf16(delta))); the forward then runs
    in fp32 so that only the merge's rounding differs."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16)
    return PeftModel.from_pretrained(m, str(adapter_dir)).merge_and_unload().float().eval()


def _run_case(base_dir, tmp_path, b_std, cfg):
    model, tok = _trained(base_dir, b_std)
    ad = tmp_path / "final_adapter"
    model.save_pretrained(str(ad))
    fp = fpm.record(model, tok, str(base_dir), ad, [ad], cfg=cfg)
    return fp, tok, ad


def _verdict(model, tok, base_dir, ad, cfg):
    fp, where = fpm.find(ad, cfg)
    assert fp is not None, where
    return fpm.compare(fpm.measure(model, tok, cfg=fp["config"]), fp, fpm.provenance(model, str(base_dir), ad), cfg)


def test_faithful_reload_passes_and_a_dropped_adapter_fails(base_dir, tmp_path):
    cfg = small_cfg()
    fp, tok, ad = _run_case(base_dir, tmp_path, 0.05, cfg)
    assert fp["adapter_effect"]["mean_abs_dlogp"] > cfg["tolerances"]["min_adapter_effect"]
    good = _verdict(_reload_unmerged(base_dir, ad), tok, base_dir, ad, cfg)
    assert good["status"] == "pass", good
    assert good["criteria"]["logprob"]["ratio"] < 1e-3
    assert good["criteria"]["sigma_samples"]["share_identical_texts"] == 1.0  # same weights, same seed, CPU

    import torch
    from transformers import AutoModelForCausalLM

    dropped = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.float32).eval()  # adapter ignored
    bad = fpm.compare(fpm.measure(dropped, tok, cfg=cfg), fp, fpm.provenance(dropped, str(base_dir), ad), cfg)
    assert bad["status"] == "fail"
    assert bad["criteria"]["logprob"]["ratio"] == pytest.approx(1.0, abs=1e-6)
    assert not bad["criteria"]["logprob"]["pass"] and not bad["criteria"]["sigma_token"]["pass"]


def test_bf16_merge_of_a_small_update_fails(base_dir, tmp_path):
    """An update below half a bf16 step of the weights (as the RL updates in FINDINGS #15): the unmerged reload passes,
    the bf16 merge erases it and fails. The update is tiny on this model, so the effect floor is lowered here; with
    the configured floor the verdict would be "inconclusive", which is tested too."""
    cfg = small_cfg(min_adapter_effect=0.0)
    fp, tok, ad = _run_case(base_dir, tmp_path, 2e-5, cfg)
    good = _verdict(_reload_unmerged(base_dir, ad), tok, base_dir, ad, cfg)
    assert good["status"] == "pass", good["criteria"]["logprob"]
    merged = _reload_bf16_merged(base_dir, ad)
    bad = _verdict(merged, tok, base_dir, ad, cfg)
    assert bad["status"] == "fail"
    assert bad["criteria"]["logprob"]["ratio"] > 0.8  # most of the update rounded away
    floor = small_cfg()
    del floor["below_floor"]  # D72's rule, as stored in every fingerprint recorded before D77
    assert fp["adapter_effect"]["mean_abs_dlogp"] < floor["tolerances"]["min_adapter_effect"]
    assert fpm.compare(fpm.measure(_reload_unmerged(base_dir, ad), tok, cfg=floor), fp,
                       fpm.provenance(merged, str(base_dir), ad), floor)["status"] == "inconclusive"


def test_below_the_floor_the_sigma_token_criterion_decides(base_dir, tmp_path):
    """D77: an adapter whose log-prob effect is under the floor is judged by the σ-token criterion. A faithful reload
    passes, the bf16 merge fails, and the log-prob criterion is reported but not decisive. A pre-D77 fingerprint
    (no `below_floor` key) keeps "inconclusive"."""
    fp, tok, ad = _run_case(base_dir, tmp_path, 2e-5, small_cfg(min_adapter_effect=0.0))
    new, old = small_cfg(), small_cfg()
    del old["below_floor"]
    assert new["below_floor"] == "sigma_token"
    assert fp["adapter_effect"]["mean_abs_dlogp"] < new["tolerances"]["min_adapter_effect"]
    assert fp["adapter_effect"]["mean_sigma_tv"] > 0
    faithful, merged = _reload_unmerged(base_dir, ad), _reload_bf16_merged(base_dir, ad)

    def verdict(model, cfg):
        return fpm.compare(fpm.measure(model, tok, cfg=cfg), fp, fpm.provenance(model, str(base_dir), ad), cfg)

    good, bad = verdict(faithful, new), verdict(merged, new)
    assert good["status"] == "pass" and good["below_floor_rule"] == "sigma_token"
    assert good["criteria"]["logprob"]["decisive"] is False and good["criteria"]["sigma_token"]["decisive"] is True
    assert bad["status"] == "fail" and bad["criteria"]["sigma_token"]["pass"] is False
    assert verdict(faithful, old)["status"] == "inconclusive"  # not retroactive
    assert verdict(merged, old)["status"] == "fail"


def test_below_the_floor_rule_on_synthetic_inputs():
    """D77 on synthetic measurements: below the floor a log-prob 'failure' (ratio 1) with an identical σ-token
    distribution passes under the new rule and fails under D72's; with no σ-token effect either, nothing can decide."""
    fp = _fp([0.01] * 10, PROV)
    fp["adapter_effect"] = {"mean_abs_dlogp": 1e-3, "mean_sigma_tv": 0.1}
    re = _fake([0.01] * 10)
    for x in re["on"]:
        x["token_logprobs"] = np.full(3, 1e-3)  # |Δ log p| = the adapter's own effect: ratio 1
    cfg = fp["config"]
    old = copy.deepcopy(cfg)
    del old["below_floor"]
    r = fpm.compare(re, fp, PROV, cfg)
    assert r["status"] == "pass" and r["criteria"]["logprob"]["pass"] is False
    assert fpm.compare(re, fp, PROV, old)["status"] == "fail"
    fp["adapter_effect"]["mean_sigma_tv"] = 0.0
    assert fpm.compare(re, fp, PROV, cfg)["status"] == "inconclusive"
    bad_prov = dict(PROV, bitsandbytes="0.49")  # provenance still fails below the floor
    fp["adapter_effect"]["mean_sigma_tv"] = 0.1
    assert fpm.compare(re, fp, bad_prov, cfg)["status"] == "fail"
    with pytest.raises(ValueError):
        fpm.compare(re, fp, PROV, dict(cfg, below_floor="majority"))


def test_find_matches_by_tensor_hash(base_dir, tmp_path):
    cfg = small_cfg()
    model, tok = _trained(base_dir, 0.05)
    run = tmp_path / "run"
    final, perm, other = run / "final_adapter", run / "permanent_checkpoints" / "checkpoint-5", run / "checkpoint-3"
    for d in (final, perm):
        model.save_pretrained(str(d))
    (final / cfg["file_name"]).write_text(json.dumps({"provenance": fpm.provenance(model, str(base_dir), final)}))
    fp, where = fpm.find(perm, cfg)
    assert fp is not None and where.endswith("final_adapter/" + cfg["file_name"])
    model2, _ = _trained(base_dir, 0.05, seed=2)
    model2.save_pretrained(str(other))
    assert fpm.find(other, cfg)[0] is None
    assert fpm.adapter_hashes(final)["tensors"] == fpm.adapter_hashes(perm)["tensors"]


def test_tensor_hash_ignores_file_metadata(tmp_path):
    from safetensors.numpy import save_file

    t = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.ones(4, dtype=np.float32)}
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    save_file(t, str(tmp_path / "x" / "adapter_model.safetensors"), metadata={"format": "pt"})
    save_file(t, str(tmp_path / "y" / "adapter_model.safetensors"), metadata={"format": "pt", "extra": "1"})
    hx, hy = fpm.adapter_hashes(tmp_path / "x"), fpm.adapter_hashes(tmp_path / "y")
    assert hx["file"] != hy["file"] and hx["tensors"] == hy["tensors"]


def _fake(sigmas, seeds=(1, 2)):
    on = [{"token_logprobs": np.zeros(3), "dist": np.log(np.full(4, 0.25)), "top_ids": [0, 1, 2, 3],
           "top_logprobs": [float(np.log(0.25))] * 4} for _ in seeds]
    return {"on": on, "seeds": list(seeds), "samples": {"texts": ["t"] * len(sigmas), "sigma_rel": sigmas}}


def _fp(sigmas, prov):
    m = _fake(sigmas)
    return {"config": small_cfg(), "seeds": m["seeds"], "provenance": prov, "samples": m["samples"],
            "adapter_effect": {"mean_abs_dlogp": 0.1, "mean_sigma_tv": 0.1},
            "on": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in x.items() if k != "dist"}
                   for x in m["on"]]}


PROV = {"base_identity": "sha256:x", "quantization_config": {"load_in_4bit": True},
        "compute_dtype": {"bnb_4bit_compute_dtype": "bfloat16"}, "bitsandbytes": "0.50.2", "versions": {"torch": "1"}}


def test_sigma_distribution_and_provenance_criteria():
    ref = [0.01] * 10 + [None] * 2
    fp = _fp(ref, PROV)
    cfg = fp["config"]
    same = fpm.compare(_fake([0.012] * 10), fp, PROV, cfg)
    assert same["status"] == "pass" and same["criteria"]["sigma_samples"]["pass"]
    wide = fpm.compare(_fake([0.05] * 10), fp, PROV, cfg)  # 5x the training σ: the merged-model symptom
    assert wide["status"] == "fail" and wide["criteria"]["sigma_samples"]["pass"] is False
    few = small_cfg()
    few["sampling"]["min_parsed"] = 50
    assert fpm.compare(_fake([0.05] * 10), fp, PROV, few)["criteria"]["sigma_samples"]["pass"] is None
    other = dict(PROV, quantization_config={"load_in_4bit": True, "bnb_4bit_quant_type": "fp4"})
    q = fpm.compare(_fake([0.01] * 10), fp, other, cfg)
    assert q["status"] == "fail" and "quantization_config" in q["criteria"]["provenance"]["mismatches"]
    ver = dict(PROV, versions={"torch": "2"})  # library versions are reported, not failed
    v = fpm.compare(_fake([0.01] * 10), fp, ver, cfg)
    assert v["status"] == "pass" and v["criteria"]["provenance"]["other_differences"] == ["versions"]
    with pytest.raises(ValueError):
        fpm.compare(_fake([0.01] * 10, seeds=(1, 3)), fp, PROV, cfg)


def test_provenance_of_a_local_base(base_dir, tmp_path):
    model, _ = _trained(base_dir, 0.05)
    ad = tmp_path / "ad"
    model.save_pretrained(str(ad))
    p = fpm.provenance(model, str(base_dir), ad)
    assert p["base_identity"].startswith("sha256:") and p["base"]["content_sha256"]
    assert p["adapter"]["r"] == 4 and p["adapter"]["target_modules"] == sorted(TARGETS)
    assert set(fpm.VERSIONS) <= set(p["versions"]) and p["quantization_config"] is None
    # the same base named by another path has the same identity
    assert fpm.provenance(model, str(base_dir.resolve()), ad)["base_identity"] == p["base_identity"]


def test_check_adapter_without_fingerprint_reports_provenance(base_dir, tmp_path):
    model, tok = _trained(base_dir, 0.05)
    ad = tmp_path / "ad"
    model.save_pretrained(str(ad))
    res = fpm.check_adapter(model, tok, str(base_dir), ad, cfg=small_cfg())  # enforce: true (D77) only warns here
    assert res["status"] == "no_fingerprint" and res["provenance"]["adapter"]["tensors"]
    assert "not certified" in res["warning"]


def test_enforce_raises_on_fail(base_dir, tmp_path):
    cfg = small_cfg()
    fp, tok, ad = _run_case(base_dir, tmp_path, 0.05, cfg)
    import torch
    from transformers import AutoModelForCausalLM

    dropped = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.float32).eval()
    assert fpm.check_adapter(dropped, tok, str(base_dir), ad, cfg=dict(cfg, enforce=False))["status"] == "fail"
    assert cfg["enforce"] is True  # D77: enforced from now on
    with pytest.raises(RuntimeError, match="R2-1 regression check FAILED"):
        fpm.check_adapter(dropped, tok, str(base_dir), ad, cfg=cfg)

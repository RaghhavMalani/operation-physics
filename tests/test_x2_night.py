"""X2 night code (D81): chunked sampling must form a prefix of the frozen pool; the D74 analysis end to end on
synthetic chunks, suffixes and hidden states; the reading rules; the hidden-state helpers on a tiny CPU model."""

import copy
import json

import numpy as np
import pytest

from physcal.config import REPO_ROOT
from physcal.eval import x2_analysis as XA
from physcal.eval import x2_hidden as XH
from physcal.eval import x2_pairs as X

TOKENIZER_DIR = REPO_ROOT / "results" / "sft" / "qwen2.5-math-1.5b-instruct" / "merged"
K = 4


def _pool(n):
    from physcal.env.generator import problem_from_dict

    lines = (REPO_ROOT / X._config()["candidates"]["file"]).read_text().splitlines()[:n]
    return [problem_from_dict(json.loads(l)) for l in lines]


def _write_chunk(root, start, problems, mixed):
    """k answers per problem: on 'mixed' problems two correct (small σ) and two wrong (large σ), else all correct."""
    from physcal.tape.tape import TapeWriter

    d = root / f"chunk_{start:05d}"
    d.mkdir(parents=True)
    with TapeWriter(d / "rollouts.jsonl", "fake", {"generation": {"num_samples": K}, "prompt_id": "v4_worked_example"}) as t:
        for p in problems:
            for j in range(K):
                wrong = mixed(p) and j >= 2
                err, rel = (0.2, 0.1) if wrong else (0.001, 0.0005)
                mu = p.target_value * (1 + err)
                comp = f"work {j}\n<steps>\n</steps>\n<answer>{mu:.4g} ± {rel * mu:.2g} {p.target_unit}</answer>"
                bd = {"reason": "ok", "mu_si": p.target_value_si * (1 + err), "sigma_si": rel * p.target_value_si,
                      "abs_rel_error": err, "truth_si": p.target_value_si}
                t.log(0, p, comp, 0.0, bd, "fake", None, {"split": "x2", "sample": j})


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(X._config())
    c["sampling"]["output_dir"] = str(tmp_path / "sft_k8")
    c["output_dir"] = str(tmp_path)
    c["suffix"]["output_dir"] = str(tmp_path / "suffix")
    c["hidden"]["output_dir"] = str(tmp_path / "hidden")
    c["analysis"].update(bootstrap=300)
    c["analysis"]["probe"].update(n_jobs=1, C=[0.1, 1.0])
    return c


def test_chunks_must_form_a_prefix_of_the_pool(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(X, "_config", lambda: cfg)
    probs = _pool(24)
    root = tmp_path / "sft_k8"
    _write_chunk(root, 0, probs[:12], lambda p: True)
    _write_chunk(root, 12, probs[12:24], lambda p: True)
    assert [s for s, _ in X.sampled_chunks(cfg)] == [0, 12]
    recs, n = X.sampled_records(cfg)
    assert n == 24 and len(recs) == 24 * K
    _write_chunk(root, 30, probs[:6], lambda p: True)  # a gap: 24..29 missing
    with pytest.raises(ValueError, match="not a prefix"):
        X.sampled_chunks(cfg)


def test_sample_config_per_chunk():
    ev = X.sample_config(start=400, batch_size=64, chunk=2)
    assert ev["splits"]["x2"]["start"] == 400 and ev["generation"]["batch_size"] == 64
    assert ev["generation"]["seed"] == X._config()["sampling"]["seed"] + 2
    assert ev["model"]["max_reserved_gib"] == X._config()["sampling"]["max_reserved_gib"]
    assert ev["output_dir"].endswith("chunk_00400")
    with pytest.raises(ValueError):
        X.sample_config(batch_size=20)  # not a multiple of k


def _states(rng, labels, informative):
    """3 layers x 6 dims; layer 1, feature 0 carries the label if `informative`."""
    n = len(labels)
    layers = [rng.normal(size=(n, 6)).astype(np.float16) for _ in range(3)]
    if informative:
        layers[1][:, 0] = (np.array(labels, dtype=float) * 3 + rng.normal(scale=0.5, size=n)).astype(np.float16)
    return layers


def _save(d, layers, index):
    d.mkdir(parents=True, exist_ok=True)
    for k, a in enumerate(layers):
        np.save(d / f"layer_{k:02d}.npy", a)
    (d / "index.json").write_text(json.dumps(index))


def test_analysis_end_to_end_on_synthetic_inputs(cfg, tmp_path, monkeypatch):
    if not (REPO_ROOT / "results" / "a5_baselines_4bit" / "models.joblib").exists():
        pytest.skip("4-bit A5 models not available")
    monkeypatch.setattr(X, "_config", lambda: cfg)
    probs = _pool(24)
    mixed = lambda p: p.seed % 2 == 0
    _write_chunk(tmp_path / "sft_k8", 0, probs, mixed)
    groups, keys = X.kept_groups(cfg)
    assert len(keys) == sum(mixed(p) for p in probs) == 12
    # method 2: split credit's suffix as informative as the SFT σ, except one failed suffix (a tie) per problem
    (tmp_path / "suffix").mkdir()
    with (tmp_path / "suffix" / "rows.jsonl").open("w") as f:
        for k in keys:
            for a in groups[k]:
                fail = a["sample"] == 3
                f.write(json.dumps({"key": list(k), "source_sample": a["sample"], "fail": fail,
                                    "rel_sigma_used": None if fail else a["rel_sigma"]}) + "\n")
    rng = np.random.default_rng(0)
    fit_index = [{"key": ["energy", 1, 20_000_000 + i], "sample": 0, "template": f"t{i % 12}", "correct": bool(i % 3 == 0),
                  "family": "energy", "depth": 1, "abs_rel_error": 0.0} for i in range(240)]
    x2_index = [{"key": list(k), "sample": a["sample"], "template": "x", "correct": a["correct"], "family": k[0],
                 "depth": k[1], "abs_rel_error": 0.0} for k in keys for a in groups[k]]
    for model, informative in (("sft", True), ("random_init", False)):
        _save(tmp_path / "hidden" / model / "fit", _states(rng, [r["correct"] for r in fit_index], informative), fit_index)
        _save(tmp_path / "hidden" / model / "x2", _states(rng, [r["correct"] for r in x2_index], informative), x2_index)
    res = XA.analyze(cfg)
    m = res["methods"]
    assert res["problem_only_check"] == {"pairwise_accuracy": 0.5, "pass": True}
    assert m["sft_sigma"]["pairwise_accuracy"] == 1.0
    assert 0.5 < m["split_sigma"]["pairwise_accuracy"] < 1.0 and m["split_sigma"]["share_pairs_tied_or_missing"] > 0
    assert m["probe"]["pairwise_accuracy"] > 0.9 and res["info"]["probe"]["layer"] == 1
    assert m["probe_crossfit"]["pairwise_accuracy"] > 0.8
    assert abs(m["random_init_probe"]["pairwise_accuracy"] - 0.5) < 0.3
    assert set(res["holm"]) == set(XA.PRIMARY) and res["n_pairs"] == 12 * 2 * 2
    assert res["correct_count_distribution"] == {"2": 12}
    assert (tmp_path / "results.json").exists() and "Reading (D74)" in (tmp_path / "summary.md").read_text()


def test_missing_inputs_count_as_p_one(cfg, tmp_path, monkeypatch):
    if not (REPO_ROOT / "results" / "a5_baselines_4bit" / "models.joblib").exists():
        pytest.skip("4-bit A5 models not available")
    monkeypatch.setattr(X, "_config", lambda: cfg)
    _write_chunk(tmp_path / "sft_k8", 0, _pool(12), lambda p: True)
    res = XA.analyze(cfg)
    for name in ("split_sigma", "probe", "random_init_probe"):
        assert res["methods"][name] == {"missing": True} and res["holm"][name]["p"] == 1.0
        assert not res["holm"][name]["rejected"]


def test_reading_rules():
    def holm(**sup):
        return {m: {"rejected": sup.get(m, False)} for m in XA.PRIMARY}
    ci = lambda lo: {"probe - random_init_probe": {"ci95": [lo, lo + 0.1]}}
    assert "information exists" in XA.reading(holm(probe=True), ci(0.02))["outcome"]
    assert "signal is in the text" in XA.reading(holm(a5_p=True), ci(0.02))["outcome"]
    assert "signal is in the text" in XA.reading(holm(a5_p=True, probe=True), ci(-0.01))["outcome"]  # probe ≯ random
    assert "every primary method fails" in XA.reading(holm(), ci(0.0))["outcome"]
    assert "observed" in XA.reading(holm(sft_sigma=True, probe=True), ci(0.1))["outcome"]


def test_hidden_state_helpers_on_a_tiny_cpu_model():
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        pytest.skip("local tokenizer not available")
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    tok.padding_side = "right"
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                         tie_word_embeddings=True)).eval()
    p = _pool(2)
    recs = []
    for q in p:
        comp = f"x\n<steps>\n</steps>\n<answer>{q.target_value:.4g} ± 0.1 {q.target_unit}</answer>"
        recs.append({"text": q.text, "completion": comp, "pm_char": XH.pm_char(comp), "prompt_id": "v4_worked_example"})
    enc, pos = next(XH._batches(recs, tok, 2))
    for j, ps in enumerate(pos):
        assert "±" in tok.decode([enc["input_ids"][j][ps]])
    states = XH._states(model, enc, pos, "cpu")
    assert len(states) == 3 and states[0].shape == (2, 32)
    one, _ = next(XH._batches(recs[:1], tok, 1))
    chk = XH.logit_check(model, one, "cpu")
    assert XH._passes(chk, 0.02) and chk["best"] == "as_returned"  # HF returns the last state after the norm
    assert XH.pm_char("<answer>1 m</answer>") is None and XH.pm_char("a ± b <answer>1 ± 2 m</answer>") == 16

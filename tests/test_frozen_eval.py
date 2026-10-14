"""The eval path on frozen problem files (A6, A6b, generator-2.0.0 noise sets), CPU only, with a tiny random model.

The pipeline runs in a subprocess with CUDA_VISIBLE_DEVICES="" so that no GPU can be touched."""

import hashlib
import json
import os
import subprocess
import sys

import pytest
import yaml

from physcal.config import REPO_ROOT, load_config
from physcal.env.generator import env_config, generate_problem, generator_version
from physcal.eval.a6_pairs import analyze
from physcal.eval.run_eval import load_frozen_split
from physcal.tape.tape import read_tape, replay

V2 = load_config("env_v2")
SETS = REPO_ROOT / "data" / "eval_sets" / generator_version(V2)
TOKENIZER_DIR = REPO_ROOT / "results" / "sft" / "qwen2.5-math-1.5b-instruct" / "merged"


def _noise_file(path):
    probs = [generate_problem(f, d, 50 + i, split="eval", cfg=V2, noise=lv)
             for i, (f, d, lv) in enumerate([("kinematics", 2, 0.01), ("dynamics", 4, 0.05), ("energy", 5, 0.02),
                                             ("energy", 1, 0.0)])]
    path.write_text("".join(json.dumps(p.to_dict(), sort_keys=True) + "\n" for p in probs))
    return path


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        pytest.skip("local tokenizer not available")
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    d = tmp_path_factory.mktemp("tiny_qwen")
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      tie_word_embeddings=True, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    Qwen2ForCausalLM(cfg).save_pretrained(str(d))
    tok.save_pretrained(str(d))
    return d


def test_frozen_sets_run_end_to_end_on_cpu(tiny_model, tmp_path):
    cfg = load_config("eval_frozen_sets")
    cfg["model"].update(id=str(tiny_model), dtype="float32", device="cpu")
    cfg["generation"].update(max_new_tokens=8, batch_size=4)
    cfg["output_dir"] = str(tmp_path / "out")
    for s in cfg["splits"].values():
        s["verify_regenerate"] = 2
    cfg["splits"]["noise"] = {"file": str(_noise_file(tmp_path / "noise.jsonl")), "verify_regenerate": -1}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    subprocess.run([sys.executable, "-m", "physcal.eval.run_eval", "--config", str(tmp_path / "cfg.yaml"),
                    "--limit", "4"], cwd=REPO_ROOT, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                   capture_output=True)
    out = tmp_path / "out"
    m = json.loads((out / "metrics.json").read_text())
    assert set(m["splits"]) == {"a6_pairs", "a6b_pairs", "noise"} and m["memory"] == {"device": "cpu"}
    assert m["generator_version"] == generator_version(V2)
    assert set(m["splits"]["noise"]["by_noise"]) == {"0", "0.01", "0.02", "0.05"}
    assert replay(out / "rollouts.jsonl").ok
    recs = list(read_tape(out / "rollouts.jsonl"))
    assert len(recs) == 12 and recs[0]["problem"]["noise"]["pair"]["member"] == "lo"
    assert analyze(out / "rollouts.jsonl", "a6_pairs")["kind"] == "sensitivity"
    assert analyze(out / "rollouts.jsonl", "a6b_pairs")["kind"] == "invariance"
    assert "by noise level" in (out / "summary.md").read_text()


def test_frozen_split_refuses_a_version_mismatch():
    with pytest.raises(ValueError, match="generator versions"):
        load_frozen_split({"file": str(SETS / "a6_pairs.jsonl")}, env_config(), limit=4)


def test_frozen_split_checks_the_checksum_and_regeneration(tmp_path):
    lines = (SETS / "a6_pairs.jsonl").read_text().splitlines()[:4]
    good = tmp_path / "pairs.jsonl"
    good.write_text("\n".join(lines) + "\n")
    sha = hashlib.sha256(good.read_bytes()).hexdigest()
    assert len(load_frozen_split({"file": str(good), "sha256": sha, "verify_regenerate": -1}, V2)) == 4
    with pytest.raises(ValueError, match="checksum"):
        load_frozen_split({"file": str(good), "sha256": "0" * 64}, V2)
    d = json.loads(lines[1])
    d["target_value_si"] *= 1.01
    bad = tmp_path / "tampered.jsonl"
    bad.write_text("\n".join([lines[0], json.dumps(d, sort_keys=True)] + lines[2:]) + "\n")
    with pytest.raises(ValueError, match="regenerate"):
        load_frozen_split({"file": str(bad), "verify_regenerate": -1}, V2)

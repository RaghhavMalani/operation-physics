"""Phase 3.6 SFT data: provenance (training seeds only), correctness under our own verifier, σ masking."""

import os

import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.env.generator import env_config
from physcal.train.sft_data import build_examples, tokenize
from physcal.verify.parse import parse_completion
from physcal.verify.reward import reward

CFG = load_config("sft")


def test_examples_come_only_from_training_seeds_and_cells():
    env = env_config()
    ex = build_examples(120, CFG["data"]["seed_offset"], CFG["data"]["sigma_rel"])
    assert all(e.problem.seed < env["splits"]["eval_seed_offset"] for e in ex)
    assert all(e.problem.depth not in env["splits"]["heldout_depths"] for e in ex)
    assert all(e.problem.family not in env["splits"]["heldout_families"] for e in ex)
    assert len({(e.problem.family, e.problem.depth) for e in ex}) == 12
    with pytest.raises(ValueError):
        build_examples(10, env["splits"]["eval_seed_offset"] - 5, 0.02)


def test_every_target_is_correct_under_the_verifier():
    for e in build_examples(240, 20_000_000, CFG["data"]["sigma_rel"]):
        r, bd = reward(e.problem, e.completion)
        assert bd["reason"] == "ok" and bd["correct"], (e.problem.family, e.problem.depth, e.completion)
        assert bd["score_steps"] == (1.0 if e.problem.intermediates else 0.0)
        assert parse_completion(e.completion).answer.sigma is not None


def test_sigma_span_points_at_the_sigma_number():
    for e in build_examples(24, 30_000_000, 0.02):
        lo, hi = e.sigma_span
        assert e.completion[lo - 2:lo] == "± "
        assert float(e.completion[lo:hi]) == pytest.approx(0.02 * abs(e.problem.target_value), rel=0.05)


@pytest.mark.skipif(not (REPO_ROOT / ".cache" / "huggingface").exists(), reason="no local HF cache")
def test_tokenize_masks_prompt_and_sigma_and_trains_eos():
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    from transformers import AutoTokenizer

    m = CFG["models"]["qwen2.5-math-1.5b-instruct"]
    tok = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"])
    e = build_examples(5, 40_000_000, 0.02)[4]
    row = tokenize(e, tok, CFG["prompt"], m["chat_template_kwargs"])
    ids, labels = row["input_ids"], row["labels"]
    trained = [i for i, l in zip(ids, labels) if l != -100]
    masked_tail = [i for i, l in zip(ids, labels) if l == -100][-row["n_masked_sigma"]:]
    sigma = e.completion[e.sigma_span[0]:e.sigma_span[1]]
    assert row["n_masked_sigma"] >= 1 and sigma in tok.decode(masked_tail)
    assert labels[-1] == tok.convert_tokens_to_ids("<|im_end|>")
    text = tok.decode(trained)
    assert "<answer>" in text and "</answer>" in text and sigma not in text.split("±")[-1].split()[0:1]

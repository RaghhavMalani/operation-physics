"""Phase 9.5 probe plumbing: pre-answer token position, records, probe statistics (CPU only)."""

import numpy as np
import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.interp.extract import answer_char, pre_answer_position, records

MODEL = REPO_ROOT / "results" / "sft" / load_config("probe_sft")["label"] / "merged"


def test_answer_char_requires_exactly_one_tag():
    assert answer_char("x <answer>1</answer>") == 2
    assert answer_char("no tag") is None
    assert answer_char("<answer>1</answer><answer>2</answer>") is None


@pytest.mark.skipif(not MODEL.exists(), reason="local SFT model not present")
def test_pre_answer_position_is_the_last_token_before_the_tag():
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(str(MODEL))
    prompt = "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n"
    for completion in ("v = 3.1 m/s\n</steps>\n<answer>3.1 ± 0.1 m/s</answer>", "x<answer>2 ± 1 m</answer>"):
        enc = tok(prompt + completion, return_offsets_mapping=True)
        at = len(prompt) + answer_char(completion)
        offsets = enc["offset_mapping"]
        pos = pre_answer_position(offsets, at)
        assert offsets[pos][0] < at <= offsets[pos + 1][1]  # starts before the tag; the next token reaches into it
        assert all(a >= at or b <= at or i == pos for i, (a, b) in enumerate(offsets))  # nothing later straddles


def test_records_label_correctness_and_skip_failures():
    tape = REPO_ROOT / "results/posthoc_calibration/sft_calibration_set/rollouts.jsonl"
    recs = records(tape, 0.01, 1e-6)
    assert 390 <= len(recs) <= 400  # 2 failures in this tape
    for r in recs:
        assert r["correct"] == (r["abs_rel_error"] <= 0.01)
        assert r["log_error"] >= -6.0


def _pc():
    return {**load_config("probe_sft")["probes"], "n_jobs": 1, "bootstrap": 300}


def test_probe_selects_the_informative_layer_and_beats_noise():
    from physcal.interp.probes import paired_bootstrap, probe

    rng = np.random.default_rng(0)
    n, d = 300, 8
    y = rng.random(2 * n) < 0.4
    layers = [rng.normal(size=(2 * n, d)) for _ in range(4)]
    layers[2][:, 0] += 1.5 * y  # only layer 2 carries the label
    fit, test = [l[:n] for l in layers], [l[n:] for l in layers]
    r = probe(fit, test, y[:n], y[n:], "correct", "linear", 0, _pc())
    assert r["selected_layer"] == 2 and r["test"] > 0.75
    assert max(p["test"] for p in r["per_layer"] if p["layer"] != 2) < 0.65
    bt = paired_bootstrap("correct", y[n:], r["test_scores"], rng.random(n), 300, 0)
    assert bt["p_one_sided"] < 0.01 and bt["diff_ci"][0] > 0


def test_holm_is_step_down():
    from physcal.interp.probes import holm

    h = holm({"a": 0.01, "b": 0.03, "c": 0.04}, 0.05)
    assert h["a"]["reject_null"] and not h["b"]["reject_null"] and not h["c"]["reject_null"]  # 0.03 > 0.05/2
    h = holm({"a": 0.01, "b": 0.02, "c": 0.04}, 0.05)
    assert all(v["reject_null"] for v in h.values())

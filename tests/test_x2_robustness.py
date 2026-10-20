"""X2 robustness checks (D92, post-hoc): the separated-pairs filter, the self-consistency and length baselines, the
pooled cluster bootstrap, and the recomputed X2 confidences reproducing X2's published numbers."""

import json

import numpy as np
import pytest

from physcal.config import REPO_ROOT
from physcal.eval import x2_robustness as R

K = ("energy", 2, 7)


def _rec(sample, mu, err, n_tokens=100, ok=True):
    return {"problem": {"family": K[0], "depth": K[1], "seed": K[2]}, "sample": sample, "n_tokens": n_tokens,
            "breakdown": {"reason": "ok" if ok else "parse_fail", "mu_si": mu if ok else None,
                          "abs_rel_error": err if ok else None}}


def test_self_consistency_counts_agreeing_other_samples():
    recs = [_rec(0, 10.0, 0.0), _rec(1, 10.05, 0.005), _rec(2, 10.2, 0.02), _rec(3, 13.0, 0.3), _rec(4, 0, 0, ok=False)]
    c = R.self_consistency_conf(recs, 0.01)
    assert c[(K, 0)] == 1 / 4  # 10.05 agrees; 10.2, 13 and the unparsed sample do not; denominator: the other 4
    assert c[(K, 1)] == 1 / 4 and c[(K, 2)] == 0.0 and c[(K, 3)] == 0.0 and (K, 4) not in c
    assert R.length_conf([_rec(0, 1.0, 0.0, n_tokens=120)]) == {(K, 0): -120.0}  # shorter = more confident


def test_separated_pairs_keep_only_clear_cases():
    groups = {K: [{"sample": 0, "correct": True}, {"sample": 1, "correct": True},
                  {"sample": 2, "correct": False}, {"sample": 3, "correct": False}]}
    err = {(K, 0): 0.001, (K, 1): 0.008, (K, 2): 0.05, (K, 3): 0.5}
    conf = {(K, 0): 1.0, (K, 1): -1.0, (K, 2): 0.0, (K, 3): 0.0}
    r = R.separated(groups, [K], conf, err, 0.0025, 0.10, 200, 0)
    assert r["n_pairs"] == 1 and r["n_problems"] == 1 and r["pairwise_accuracy"] == 1.0  # only (0, 3)
    assert R.separated(groups, [K], conf, err, 0.0001, 0.10, 200, 0)["n_problems"] == 0


def test_pooled_weighs_pairs_and_resamples_problems():
    k2 = ("energy", 3, 9)
    groups = {K: [{"sample": 0, "correct": True}, {"sample": 1, "correct": False}],
              k2: [{"sample": s, "correct": s < 3} for s in range(6)]}  # 1 pair vs 9 pairs
    conf = {(K, 0): 1.0, (K, 1): 0.0} | {(k2, s): 0.0 for s in range(6)}  # 1.0 vs nine ties
    r = R.pooled(groups, [K, k2], conf, 2000, 0)
    assert r["pooled_pairwise_accuracy"] == pytest.approx((1 + 9 * 0.5) / 10) and r["n_pairs"] == 10
    assert r["ci95"][0] == pytest.approx(0.5) and r["ci95"][1] == pytest.approx(1.0)  # only two problems to resample


def test_recomputed_x2_confidences_reproduce_x2():
    groups, keys, recs = R.answer_records(R._cfg_of("x2"))
    published = json.loads((REPO_ROOT / "results" / "x2" / "results.json").read_text())["methods"]
    from physcal.eval import x2_pairs

    conf = {"sft_sigma": {(k, a["sample"]): x2_pairs.stated_sigma_conf(k, a) for k in keys for a in groups[k]}}
    assert R.check_reproduces(groups, keys, conf, published, 1e-9) == {"sft_sigma": 0.0}
    bad = {"sft_sigma": {k: 0.0 for k in conf["sft_sigma"]}}
    with pytest.raises(RuntimeError):
        R.check_reproduces(groups, keys, bad, published, 1e-9)
    err = R.errors(recs)
    assert all(err[(k, a["sample"])] <= 0.01 for k in keys for a in groups[k] if a["correct"])  # X2's 1% label
    sc = R.self_consistency_conf(recs, 0.01)
    answers = [(k, a) for k in keys for a in groups[k]]
    missing = [(k, a) for k, a in answers if (k, a["sample"]) not in sc]
    assert len(answers) == 4116 and [a["mu_si"] for _, a in missing] == [0.0]  # μ = 0: agreement undefined, a tie
    assert all(0.0 <= v <= 1.0 for v in sc.values())
    assert {sum(R._key(r)[0] == k for r in recs) for k in keys} == {8}  # 8 samples per problem: denominator 7

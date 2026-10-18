"""X2 (D74): the selection rule and the pairwise-accuracy metric (CPU, synthetic), and the frozen candidate pool."""

import hashlib
import json

import numpy as np
import pytest

from physcal.config import REPO_ROOT
from physcal.env.generator import env_config, regenerate
from physcal.eval import x2_pairs as X


def _rec(key, err, sigma, mu=2.0, reason="ok", sample=0):
    f, d, s = key
    return {"problem": {"family": f, "depth": d, "seed": s}, "sample": sample,
            "breakdown": {"reason": reason, "mu_si": mu if reason == "ok" else None, "sigma_si": sigma,
                          "abs_rel_error": err}}


def test_selection_keeps_mixed_problems_only():
    a, b, c = ("energy", 1, 1), ("energy", 2, 2), ("dynamics", 3, 3)
    recs = [_rec(a, 0.001, 0.02), _rec(a, 0.5, 0.02),                         # mixed
            _rec(b, 0.001, 0.02), _rec(b, 0.002, 0.02),                       # all correct
            _rec(c, 0.5, 0.02), _rec(c, 0.0, None, reason="parse_fail")]      # a parse failure is not an answer
    g = X.group(recs)
    assert X.select(g) == [a] and len(g[c]) == 1


def test_pairwise_accuracy_ties_and_direction():
    k = ("energy", 1, 1)
    g = {k: [{"correct": True, "rel_sigma": 0.01}, {"correct": False, "rel_sigma": 0.1},
             {"correct": False, "rel_sigma": 0.005}]}
    pa = X.pairwise_accuracy(g, [k], X.stated_sigma_conf)
    assert pa[0] == pytest.approx(0.5)  # wins one pair (0.01 < 0.1), loses one (0.01 > 0.005)
    const = X.pairwise_accuracy(g, [k], lambda key, a: 1.0)  # a problem-text-only method: identical within a problem
    assert const[0] == 0.5
    oracle = X.pairwise_accuracy(g, [k], lambda key, a: float(a["correct"]))
    assert oracle[0] == 1.0


def test_summary_bootstrap():
    s = X.summarize(np.array([1.0] * 30 + [0.5] * 10), 2000, 0)
    assert s["pairwise_accuracy"] == pytest.approx(0.875) and s["p_one_sided_gt_half"] == 0.0
    flat = X.summarize(np.full(40, 0.5), 500, 0)
    assert flat["p_one_sided_gt_half"] == 1.0


def test_frozen_candidates_match_the_manifest_and_regenerate():
    cfg = X._config()
    path = REPO_ROOT / cfg["candidates"]["file"]
    if not path.exists():
        pytest.skip("candidate pool not built")
    man = json.loads(path.with_name(path.stem + "_manifest.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == man["sha256"] and man["n_problems"] == 1200
    lines = path.read_text().splitlines()
    assert set(man["cells"].values()) == {100}
    env = env_config()
    for line in lines[:: 97]:
        d = json.loads(line)
        assert regenerate(d, env).to_dict() == d and 30000000 <= d["seed"] < 30001200

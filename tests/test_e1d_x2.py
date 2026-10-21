"""D102: E1d on the X2 pairs. The tests, Holm and reading rules on synthetic numbers; the whole analysis on X2's real
rows with X3 student (a)'s suffix rows standing in for E1d's (so T3's difference is exactly 0)."""

import copy
import shutil

import numpy as np
import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.eval import e1d_x2_analysis as A


def test_tests_holm_and_reading():
    rng = np.random.default_rng(0)
    pa = {"e1d": np.clip(rng.normal(0.6, 0.2, 300), 0, 1), "sft_sigma": np.full(300, 0.5),
          "x3_a": np.clip(rng.normal(0.5, 0.2, 300), 0, 1)}
    t = A.tests(pa, 500, 0, 0.05)
    assert t["T1"]["holm"]["rejected"] and t["T2"]["holm"]["rejected"] and t["T3"]["holm"]["rejected"]
    assert A.reading(t).startswith("T1 and T3 supported")
    # the paired difference against a constant 0.5 equals the accuracy's distance from 0.5
    assert t["T2"]["mean_diff"] == pytest.approx(t["T1"]["pairwise_accuracy"] - 0.5)
    flat = {"e1d": np.full(300, 0.5), "sft_sigma": np.full(300, 0.5), "x3_a": np.full(300, 0.5)}
    t = A.tests(flat, 500, 0, 0.05)
    assert not any(r["holm"]["rejected"] for r in t.values())
    assert A.reading(t).startswith("T1 not supported")


def test_missing_method_counts_as_p_one():
    pa = {"e1d": np.full(50, 0.9), "sft_sigma": np.full(50, 0.5)}
    t = A.tests(pa, 200, 0, 0.05)
    assert t["T3"]["missing"] and t["T3"]["holm"]["p"] == 1.0 and not t["T3"]["holm"]["rejected"]
    assert t["T1"]["holm"]["rejected"]


@pytest.mark.skipif(not (REPO_ROOT / "results/x2/sft_k8/chunk_01000/rollouts.jsonl").exists()
                    or not (REPO_ROOT / "results/x3/a/eval_x2/rows.jsonl").exists(), reason="X2 / X3 rows not present")
def test_analysis_on_real_rows_with_x3_standing_in(tmp_path):
    cfg = copy.deepcopy(load_config("e1d_x2"))
    cfg["bootstrap"] = 200
    cfg["output_dir"] = str(tmp_path)
    (tmp_path / "e1d_final").mkdir()
    shutil.copy(REPO_ROOT / cfg["x3_rows"], tmp_path / "e1d_final" / "rows.jsonl")
    res = A.analyze(cfg)
    assert res["n_kept"] == 516
    assert max(res["reproduction_gaps"].values()) <= 1e-9
    assert res["tests"]["T3"]["mean_diff"] == 0.0
    m = res["methods"]
    assert m["e1d"]["pairwise_accuracy"]["pairwise_accuracy"] == m["x3_a"]["pairwise_accuracy"]["pairwise_accuracy"]
    assert m["e1d"]["across_problems"]["n_answers"] > 4000
    assert res["suffix_evals"]["e1d"]["n_rows"] == 4116
    assert "sft_suffix" not in res["methods"] and (tmp_path / "summary.md").exists()

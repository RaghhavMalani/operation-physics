"""X1 on A6c (D99): the reproduction comparison and the within-pair identity count."""

import math

import numpy as np

from physcal.eval import x1_a6c


def test_diffs_compares_every_published_leaf():
    old = {"a": 1.0, "b": [0.5, 0.25], "c": {"d": "sigma_star", "e": None, "f": True}}
    assert max(d for _, d in x1_a6c._diffs(old, old)) == 0.0
    new = {"a": 1.0 + 1e-12, "b": [0.5, 0.26], "c": {"d": "sigma_star", "e": None, "f": True}, "extra": 9.0}
    d = dict(x1_a6c._diffs(new, old))
    assert d["/a"] < 1e-11 and math.isclose(d["/b[1]"], 0.01)
    assert "/extra" not in d  # only the published leaves are compared


def test_diffs_flags_missing_or_changed_leaves():
    old = {"a": 1.0, "b": [1.0, 2.0], "s": "x"}
    d = dict(x1_a6c._diffs({"b": [1.0], "s": "y"}, old))
    assert d["/a"] == math.inf and d["/b[0]"] == math.inf and d["/s"] == math.inf


class _Feat:
    def transform(self, recs):
        return np.array([[r["text"].count("5%."), len(r["meas"])] for r in recs], dtype=float)


class _Model:
    def predict(self, X):
        return X[:, 0]


def _member(seed, member, text):
    return {"text": text, "family": "kinematics", "depth": 1, "seed": seed, "target_symbol": "v", "givens": [],
            "noise": {"level": 0.05, "sigma_star_log": 0.02, "pair": {"pair_seed": seed // 2, "member": member}}}


def test_identical_share_counts_pairs_with_equal_features():
    problems = [_member(0, "lo", "a = 1 m ± 5%. b = 2 m ± 1%,"), _member(1, "hi", "a = 1 m ± 1%. b = 2 m ± 5%,"),
                _member(2, "lo", "a = 1 m ± 5%, b = 2 m ± 1%."), _member(3, "hi", "a = 1 m ± 5%, b = 2 m ± 1%.")]
    out = x1_a6c.identical_share((_Feat(), _Model()), problems)
    assert out["n_pairs"] == 2
    assert out["share_identical_features"] == 0.5 and out["share_identical_sigma_hat"] == 0.5

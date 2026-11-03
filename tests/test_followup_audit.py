"""D164 (exploratory follow-up track): the confound audit of D162. CPU only; synthetic answers stand in for the
Kaggle samples (the real run is checked against D162's results.json by `analyze` itself)."""

import json

import numpy as np
import pytest

from physcal.eval import followup_audit as AU
from physcal.eval import x2_pairs

SC = {"bootstrap": 200, "seed": 0, "generation": {"k": 8}}
READ = ("self_consistency", "yes", "probe", "probe_random", "length", "text")


def _model(seed: int, n: int = 40, drift: float = 0.0, with_text: bool = True) -> dict:
    """n kept problems of 8 answers each, 1..7 correct; each readout is the label plus noise."""
    rng = np.random.default_rng(seed)
    groups, conf, ntok = {}, {r: {} for r in READ if with_text or r != "text"}, {}
    for i in range(n):
        k = ("kinematics", 1 + i % 4, 1000 + i)
        c = 1 + i % 7
        groups[k] = [{"correct": s < c, "sample": s} for s in range(8)]
        for a in groups[k]:
            for r in conf:
                conf[r][(k, a["sample"])] = float(a["correct"]) * (1 + drift) + rng.normal()
            ntok[(k, a["sample"])] = int(rng.integers(200, 1200))
    keys = list(groups)
    return AU._data(groups, keys, conf, ntok, {keys[0], keys[3]})


def _data() -> dict:
    d7, d1 = _model(0, drift=1.0), _model(1)
    d1["keys"] = d1["keys"][5:] + [("kinematics", 2, 5000)]
    d1["groups"][("kinematics", 2, 5000)] = [{"correct": s < 2, "sample": s} for s in range(8)]
    for r in d1["conf"]:
        d1["conf"][r] |= {(("kinematics", 2, 5000), s): 0.0 for s in range(8)}
    d1["ntok"] |= {(("kinematics", 2, 5000), s): 500 for s in range(8)}
    d1["pass"][("kinematics", 2, 5000)] = 2
    sft = _model(2, with_text=False)
    for r in ("probe_random",):
        sft["conf"].pop(r)
    return {"7b": d7, "1.5b": d1, "sft": sft}


def test_group_rows_is_the_kept_rule():
    rows = [{"key": ["f", 1, 1], "sample": 0, "value": 1.0, "correct": True},
            {"key": ["f", 1, 1], "sample": 1, "value": None, "correct": None},     # truncated: no answer
            {"key": ["f", 1, 1], "sample": 2, "value": 2.0, "correct": False},
            {"key": ["f", 1, 2], "sample": 0, "value": 1.0, "correct": True},
            {"key": ["f", 1, 2], "sample": 1, "value": None, "correct": None}]
    groups, keys = AU.group_rows(rows)
    assert keys == [("f", 1, 1)]
    assert [a["sample"] for a in groups[("f", 1, 1)]] == [0, 2] and groups[("f", 1, 2)] == [groups[("f", 1, 2)][0]]


def test_length_matched_is_relative_to_the_longer():
    assert AU.length_matched(100, 110, 0.1) and AU.length_matched(110, 100, 0.1)
    assert not AU.length_matched(100, 112, 0.1)
    assert AU.length_matched(500, 500, 0.0)


def test_per_problem_is_x2s_metric_and_filters_pairs():
    d = _model(3)
    conf = d["conf"]["probe"]
    want = x2_pairs.pairwise_accuracy(d["groups"], d["keys"], lambda k, a: conf.get((tuple(k), a["sample"])))
    assert np.array_equal(AU.per_problem(d, d["keys"], conf), want)
    assert np.allclose(AU.per_problem(d, d["keys"], conf, rel=1.0), want)     # every pair matches at 100%
    k = d["keys"][1]                                                          # 2 correct, 6 incorrect
    for s in range(8):
        d["ntok"][(k, s)] = {0: 100, 1: 5000, 2: 100}.get(s, 1000)           # only (0, 2) is matched
    conf[(k, 0)], conf[(k, 2)] = 1.0, 0.0
    assert AU.per_problem(d, [k], conf, rel=0.1)[0] == 1.0
    conf[(k, 2)] = 1.0                                                         # a tie
    assert AU.per_problem(d, [k], conf, rel=0.1)[0] == 0.5
    assert len(AU._pairs(d, k, 0.1)) == 1 and len(AU._pairs(d, k, None)) == 12


def test_compare_has_d162s_structure_and_the_new_controls():
    data = _data()
    res = AU.compare(data, SC)
    assert res["n_common"] == 35
    m7 = res["models"]["7b"]
    assert m7["n_kept"] == 40 and set(m7["readouts"]) == set(READ)
    assert "text" not in res["models"]["sft"]["readouts"]
    pa = AU.per_problem(data["7b"], data["7b"]["keys"], data["7b"]["conf"]["probe"])
    assert m7["readouts"]["probe"]["pairwise_accuracy"] == pytest.approx(pa.mean())
    d = res["differences"]["probe"]
    assert set(d) == {"7b - native 1.5b (unpaired)", "7b - native 1.5b (paired, common kept problems)",
                      "7b - SFT 1.5b X2 (same prefix, unpaired)", "native 1.5b - SFT 1.5b X2 (same prefix, unpaired)"}
    assert "7b - SFT 1.5b X2 (same prefix, unpaired)" not in res["differences"]["probe_random"]
    assert {"probe - text (paired)", "text - probe_random (paired)", "probe - probe_random (paired)"} <= set(
        res["controls"]["7b"])
    assert d["7b - native 1.5b (unpaired)"]["diff"] > 0                       # the 7B's readouts were made sharper


def test_compare_length_matched_drops_problems_without_a_matched_pair():
    data = _data()
    k = data["7b"]["keys"][0]
    for s in range(8):
        data["7b"]["ntok"][(k, s)] = 100 if data["7b"]["groups"][k][s]["correct"] else 1000
    res = AU.compare(data, SC, rel=0.1)
    full = AU.compare(data, SC)
    assert res["models"]["7b"]["n_kept"] < full["models"]["7b"]["n_kept"]
    assert res["models"]["7b"]["n_pairs"] < full["models"]["7b"]["n_pairs"]


def test_restrict_keeps_order_and_filters_per_model():
    data = _data()
    out = AU.restrict(data, lambda m, k: k not in data[m]["truncated"])
    assert out["7b"]["keys"] == [k for k in data["7b"]["keys"] if k not in data["7b"]["truncated"]]
    assert len(out["7b"]["keys"]) == 38 and data["7b"]["keys"] is not out["7b"]["keys"]


def test_standardization_removes_a_pass_count_mix_difference():
    """Both models have the same per-stratum accuracy, but the 7B has more easy strata: raw means differ, the
    standardized difference is 0."""
    keys = [("f", 1, i) for i in range(70)]
    data = {}
    for m, cmap in (("7b", lambda i: 7 if i < 50 else 1), ("1.5b", lambda i: 7 if i < 10 else 1)):
        groups, conf, ntok = {}, {}, {}
        for i, k in enumerate(keys):
            c = cmap(i)
            groups[k] = [{"correct": s < c, "sample": s} for s in range(8)]
            for a in groups[k]:                         # stratum 7: perfect ranking; stratum 1: all ties (0.5)
                conf[(k, a["sample"])] = (float(a["correct"]) if c == 7 else 0.0)
                ntok[(k, a["sample"])] = 100
        data[m] = AU._data(groups, keys, {"probe": conf}, ntok, set())
    out = AU.by_pass_count(data, SC)
    st = out["standardized"]["probe"]
    assert st["diff"] == pytest.approx(0.0) and st["ci95"] == pytest.approx([0.0, 0.0])
    assert out["n"]["7b"]["7"] == 50 and out["n"]["1.5b"]["7"] == 10
    assert out["weights"][0] == pytest.approx(80 / 140) and out["weights"][6] == pytest.approx(60 / 140)
    raw = {m: AU.per_problem(data[m], keys, data[m]["conf"]["probe"]).mean() for m in data}
    assert raw["7b"] - raw["1.5b"] == pytest.approx(40 / 70 * 0.5)
    assert out["strata"]["probe"]["7b"]["7"]["pairwise_accuracy"] == 1.0


def test_by_pass_count_needs_the_same_problems():
    data = _data()
    with pytest.raises(ValueError):
        AU.by_pass_count({m: data[m] for m in ("7b", "1.5b")}, SC)


def test_text_fields_stop_at_the_last_box():
    text = "A ball is thrown. What is its speed in m/s?"
    comp = "First \\boxed{3} then more work, so the answer is \\boxed{12.4 \\text{ m/s}}. Done, I re-checked."
    f = AU.text_fields(text, comp)
    assert f["answer"] == "12.4 \\text{ m/s}"
    assert f["reasoning"] == "First \\boxed{3} then more work, so the answer is "
    assert "re-checked" not in json.dumps(f)
    with pytest.raises(ValueError):
        AU.text_fields(text, "no box at all")


A5_SMALL = {"text": {"token_pattern": "\\S+", "ngram_range": [1, 1], "min_df": 3, "sublinear_tf": True,
                     "gbt_max_features": 50},
            "grids": {"logistic_C": [0.1, 10.0], "gbt": [{"max_iter": 20, "max_depth": 2, "learning_rate": 0.1,
                                                            "min_samples_leaf": 5}]},
            "seed": 0}


def test_crossfit_text_is_grouped_by_problem_and_reads_the_text():
    rng = np.random.default_rng(0)
    recs, y, groups = [], [], []
    for p in range(30):
        for s in range(6):
            ok = s % 2 == 0
            words = " ".join(rng.choice(["alpha", "beta", "gamma", "delta"], 8))
            recs.append({"problem": f"problem {p % 5} text", "answer": f"{s} m",
                         "reasoning": words + (" verified" if ok else " hmm wait")})
            y.append(int(ok))
            groups.append(f"p{p}")
    y = np.array(y)
    folds = AU._folds(groups, 5)
    for tr, te in folds:
        assert not {groups[i] for i in tr} & {groups[i] for i in te}
    p, chosen = AU.crossfit_text(recs, y, groups, 5, "problem_answer_reasoning", 3, A5_SMALL)
    assert not np.isnan(p).any() and len(chosen) == 5
    assert all(c["kind"] in ("linear", "gbt") and len(c["table"]) == 3 for c in chosen)
    assert np.mean(p[y == 1]) > np.mean(p[y == 0]) + 0.3


def test_text_grid_is_a5s_p_grid_in_order():
    g = AU.text_grid(A5_SMALL)
    assert g == [("linear", 0.1), ("linear", 10.0), ("gbt", A5_SMALL["grids"]["gbt"][0])]


def test_cached_probe_refuses_a_stale_cache(tmp_path, monkeypatch):
    cache = tmp_path / "probe_hidden.json"
    cache.write_text(json.dumps({"inputs_sha256": "old", "conf": [[["f", 1, 1], 0, 0.5]]}))
    monkeypatch.setattr(AU.A, "probe_inputs", lambda *a: ([], "new", cache))
    with pytest.raises(SystemExit):
        AU.cached_probe("7b", "hidden", {}, {})
    monkeypatch.setattr(AU.A, "probe_inputs", lambda *a: ([], "old", cache))
    assert AU.cached_probe("7b", "hidden", {}, {}) == {(("f", 1, 1), 0): 0.5}


def test_sensitivity_flags_a_changed_sign_or_ci_side():
    def v(diff, lo, hi):
        return {"diff": diff, "ci95": [lo, hi]}
    base = {"differences": {"yes": {"7b - native 1.5b (unpaired)": v(0.1, 0.05, 0.15)},
                            "probe": {"7b - native 1.5b (unpaired)": v(-0.01, -0.03, 0.02)}},
            "controls": {"7b vs native 1.5b": {"(probe - probe_random): 7b - native 1.5b (unpaired)": v(0, -1, 1)},
                         "7b": {"probe - probe_random (paired)": v(0.09, 0.07, 0.11)},
                         "1.5b": {"probe - probe_random (paired)": v(0.08, 0.06, 0.1)}}}
    other = json.loads(json.dumps(base))
    other["differences"]["yes"]["7b - native 1.5b (unpaired)"] = v(0.04, -0.01, 0.09)
    out = AU.sensitivity(base, other)
    assert out["yes: 7b - native 1.5b (unpaired)"]["changed"]
    assert sum(x["changed"] for x in out.values()) == 1


def test_with_others_drops_lone_samples():
    rows = [{"key": ["f", 1, 1], "sample": 0}, {"key": ["f", 1, 1], "sample": 1}, {"key": ["f", 1, 2], "sample": 0}]
    assert AU.with_others(rows, lambda r: tuple(r["key"])) == rows[:2]


def test_pair_weighting_weighs_pairs_not_problems():
    """Two problems: one with 1 pair (score 1), one with 12 pairs (score 0): per problem 0.5, per pair 1/13."""
    def model(seed):
        groups = {("f", 1, 1): [{"correct": s < 1, "sample": s} for s in range(2)],
                  ("f", 1, 2): [{"correct": s < 2, "sample": s} for s in range(8)]}
        conf = {}
        for k, g in groups.items():
            for a in g:
                good = k == ("f", 1, 1)
                conf[(k, a["sample"])] = float(a["correct"] == good)
        ntok = {(k, a["sample"]): 100 for k, g in groups.items() for a in g}
        c = {r: conf for r in ("yes", "probe", "probe_random")}
        return AU._data(groups, list(groups), c, ntok, set())
    data = {"7b": model(0), "1.5b": model(1)}
    out = AU.pair_weighted(data, SC, paired=True)
    m = out["models"]["7b"]["probe"]
    assert m["per_problem_weighted"] == pytest.approx(0.5) and m["pairwise_accuracy"] == pytest.approx(1 / 13)
    assert out["differences"]["probe"]["diff"] == pytest.approx(0.0)
    assert out["differences"]["probe"]["ci95"] == pytest.approx([0.0, 0.0])      # paired: the same resamples
    assert out["net_scale_effect"]["diff"] == pytest.approx(0.0)


def test_pair_weighting_paired_needs_the_same_problems():
    data = _data()
    with pytest.raises(ValueError):
        AU.pair_weighted(data, SC, paired=True)
    assert set(AU.pair_weighted(data, SC, paired=False)["differences"]) >= {"yes", "probe"}

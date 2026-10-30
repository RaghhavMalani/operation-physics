"""D147: R5-6 part J's one-time analysis (physcal/eval/r5_6.py), checked as D123 checked R4-3's: on the DEVELOPMENT
A6b / A6c files with seed 0's development tapes copied into three seeds (the frozen J files are not read). The J
metadata the development pairs lack is injected (a fixed pattern per pair), so every descriptive path runs. CPU only."""

import copy
import hashlib
import json
import shutil

import numpy as np
import pytest

from physcal.config import REPO_ROOT, load_config

DEV_EVAL = {arm: REPO_ROOT / "results" / "stage_r4" / arm / "seed0" / "eval" for arm in ("dpos2", "b2")}


def test_the_analysis_refuses_without_the_guard_and_before_all_six_evals(monkeypatch, tmp_path):
    from physcal.eval import r5_6
    from physcal.eval.run_eval import CONFIRMATORY_ENV

    monkeypatch.delenv(CONFIRMATORY_ENV, raising=False)
    with pytest.raises(PermissionError):
        r5_6.analyze(root=tmp_path)
    monkeypatch.setenv(CONFIRMATORY_ENV, "1")
    with pytest.raises(RuntimeError, match="all six evals"):
        r5_6.analyze(root=tmp_path)


def test_stage_a_config_points_at_the_frozen_j_files():
    from physcal.eval import r5_6

    sa = r5_6.stage_a_config()
    assert sa["generator_version"] == "2.0.0+bdbb2279/confirmatory/r5_6"
    assert sa["pair_sets"] == ["a6b_pairs", "a6c_pairs"]
    assert sa["alternatives"]["x1_predictions"] == "results/x1_text_sigma/r5_6/predictions.jsonl"
    assert load_config("stage_a")["pair_sets"] == ["a6_pairs", "a6b_pairs", "a6c_pairs"]  # untouched


def test_the_reading_is_d147s():
    from physcal.eval.r5_6 import reading

    assert reading({"J1": True, "C2": True, "C3": True}).startswith("P_J confirmed")
    assert "below the practical threshold" in reading({"J1": True, "C2": False, "C3": True})
    assert "not scale-invariant" in reading({"J1": True, "C2": True, "C3": False})
    assert reading({"J1": False, "C2": True, "C3": True}).startswith("P_J not confirmed")


def _inject_j(problems: list[dict]) -> list[dict]:
    """A fixed, pair-level J record for development problems (both members share it, as in the J set)."""
    out = []
    for p in problems:
        h = int(hashlib.sha256(f"{p['_set']}|{p['noise']['pair']['pair_seed']}".encode()).hexdigest()[:8], 16)
        gs = sorted(p["noise"]["givens"])
        j = {"exceedance": 0.1 + (h % 100) / 100, "exceedance_given": gs[0], "nearest_training_linf": 0.2,
             "n_training": 1 + h % 6, "out_of_range_givens": gs[:1], "strict": h % 3 == 0, "rank_flip": h % 4 == 0,
             "clamped": {g: 0.5 for g in gs}, "clamped_log_ratio": 0.0}
        q = copy.deepcopy(p)
        q["noise"]["pair"]["j"] = j
        out.append(q)
    return out


@pytest.mark.skipif(not all((d / "rollouts.jsonl").exists() or (d / "rollouts.jsonl.gz").exists()
                            for d in DEV_EVAL.values()), reason="seed 0's development evals not on disk")
def test_three_identical_seeds_on_the_development_files(monkeypatch, tmp_path):
    """J1 equals D113's R1 (D-pos-v2 (a)'s development A6c slope), B-v2's slope equals R4, J2 = R1 − R4, J3-J5 are
    K6-K8 restricted to A6b + A6c; the seed SD is 0; P_J follows D147 §8; every descriptive path runs."""
    from physcal.eval import r5_6, stage_a
    from physcal.eval.run_eval import CONFIRMATORY_ENV

    monkeypatch.setenv(CONFIRMATORY_ENV, "1")
    cfg = copy.deepcopy(load_config("r5_6"))
    cfg.update(set_subdir=".", x1_predictions="results/x1_text_sigma/a6c/predictions.jsonl",
               x1_reproduction="results/x1_text_sigma/a6c/reproduction.json", output_dir=str(tmp_path / "out"))
    r4 = copy.deepcopy(load_config("stage_r4"))
    r4["analysis"]["bootstrap"] = 200
    real_load = load_config
    monkeypatch.setattr(r5_6, "load_config", lambda name: r4 if name == "stage_r4" else real_load(name))
    load_sets = stage_a.load_sets
    monkeypatch.setattr(r5_6.stage_a, "load_sets", lambda sa: _inject_j(load_sets(sa)))
    for arm, d in DEV_EVAL.items():
        for s in r4["r5_6"]["seeds"]:
            dst = tmp_path / "root" / arm / f"seed{s}" / r4["r5_6"]["output_subdir"]
            dst.mkdir(parents=True)
            src = d / "rollouts.jsonl"
            shutil.copy(src if src.exists() else d / "rollouts.jsonl.gz", dst)
    res = r5_6.analyze(cfg, root=tmp_path / "root")
    dev = json.loads((REPO_ROOT / "results" / "stage_r4" / "results.json").read_text())["tests"]
    t = res["main"]["tests"]
    assert res["n_problems"] == 2040 and res["complete"] and res["seeds"] == {"dpos2": [0, 1, 2], "b2": [0, 1, 2]}
    assert t["J1"]["value"] == pytest.approx(dev["R1"]["value"], abs=1e-12)
    assert t["J1"]["seed_sd"] == pytest.approx(0.0, abs=1e-12)
    assert t["J2"]["explicit"] == pytest.approx(dev["R1"]["value"], abs=1e-12)
    assert t["J2"]["implicit"] == pytest.approx(dev["R4"]["value"], abs=1e-12)
    assert t["J2"]["value"] == pytest.approx(dev["R1"]["value"] - dev["R4"]["value"], abs=1e-12)
    j1_dist = res["main"]["criteria"]["dpos2_a"]["a6c"]
    assert t["J1"]["ci90"] == j1_dist["ci90"]  # J1 is the criteria's A6c slope, one machinery
    for k in ("J3", "J4", "J5", "J6"):
        assert not t[k].get("missing") and t[k]["n_problems"] > 0
    assert all("holm" in t[k] for k in r5_6.TESTS)
    pr = res["primary"]
    assert pr["criteria"] == ["J1", "C2", "C3"]
    assert pr["confirmed"] == (t["J1"]["holm"]["rejected"] and dev["R1"]["value"] >= 0.5
                               and res["main"]["criteria"]["dpos2_a"]["C3"])
    assert set(res["comparators"]) == {"sigma_star", "rss", "x1", "clamped"}
    sp = res["splits"]
    assert {"depth_1", "depth_5", "exceedance_above_median", "strict", "rank_flip_sensitivity",
            "training_count_3_plus"} <= set(sp)
    assert sp["depth_1"]["n_pairs"] + sum(sp[f"depth_{d}"]["n_pairs"] for d in (2, 3, 4, 5)) == res["main"]["n_pairs"]
    assert set(res["s_lines_by_range"]) == {0, 1, 2}
    assert (tmp_path / "out" / "summary.md").read_text().startswith("# R5-6 part J")
    rows = [json.loads(x) for x in (tmp_path / "out" / "pair_distances.jsonl").read_text().splitlines()]
    assert len(rows) == res["main"]["n_pairs"] == 1020
    assert np.isfinite([r["d_ln_sigma_star"] for r in rows]).all()

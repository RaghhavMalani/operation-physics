"""D71 (R2-4 / R2-6): the 3-vs-3 robustness analysis, on synthetic seed values and end to end on the real corrected
evals with a stand-in uniform arm (the seed-0 control twice; tonight's seeds 1 and 2 do not exist yet)."""

import sys

import numpy as np
import pytest

from physcal.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import robustness_3v3 as R  # noqa: E402


def test_exact_permutation_test():
    t = R.permutation([3.0, 4.0, 5.0], [0.0, 1.0, 2.0], "greater")
    assert t["n_relabellings"] == 20 and t["delta"] == 3.0
    assert t["p_one_sided"] == pytest.approx(0.05) and t["p_two_sided"] == pytest.approx(0.10)
    assert t["min_p_one_sided"] == pytest.approx(0.05) and t["min_p_two_sided"] == pytest.approx(0.10)
    assert t["share_cross_positive"] == 1.0 and t["hedges_g"] > 0
    t2 = R.permutation([1.0, 2.0, 3.0], [1.0, 2.0], None)
    assert t2["n_relabellings"] == 10 and "p_one_sided" not in t2 and t2["p_two_sided"] > 0.5


def test_seed_of():
    assert [R.seed_of(r.name) for r in R.UNIFORM] == [0, 1, 2] and [R.seed_of(r.name) for r in R.SPLIT] == [0, 1, 2]


@pytest.mark.skipif(not (R.UNIFORM[0] / R.EVAL).exists(), reason="corrected evals not available")
def test_end_to_end_with_a_stand_in_uniform_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "UNIFORM", [R.UNIFORM[0], R.UNIFORM[0]])
    monkeypatch.setattr(R, "OUT", tmp_path)
    monkeypatch.setattr(R, "N_BOOT", 200)
    monkeypatch.setattr(R, "STEP", 50)
    out = R.main()
    a = out["A_problem_level_seed_mean"]
    # with a duplicated control the uniform seed mean equals the single control: S1b/S2 statistics match D62's
    from phase95_seed_mean import seed_mean_tests  # noqa: E402
    from phase95_compare import load  # noqa: E402

    split = [load(r / R.EVAL) for r in R.SPLIT]
    ref = seed_mean_tests(split, load(R.SFT), load(R.UNIFORM[0] / R.EVAL), sorted(split[0]), n_boot=200)
    for k in ("S1a", "S1b", "S2", "S3"):
        assert a[k]["diff"] == pytest.approx(ref[k]["diff"])
    assert (tmp_path / "trajectories.png").stat().st_size > 10_000 and "(C)" in (tmp_path / "report.md").read_text()
    traj = out["C_trajectories"]["split"]
    assert all(len(v) == 2 and v[0]["steps"] == [1, 25] for v in traj.values())
    assert np.isfinite(out["B_seed_level"]["tests"]["cov1"]["delta"])

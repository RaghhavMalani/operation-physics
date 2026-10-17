"""D62: the seed-mean family S tests (scripts/phase95_seed_mean.py) on synthetic eval outcomes."""

import sys

import numpy as np
import pytest

from physcal.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import phase95_compare  # noqa: E402
import phase95_seed_mean as sm  # noqa: E402

N = 60
KEYS = [("id" if i % 2 else "depth_extrapolation", "dynamics", 1 + i % 5, 1_000_000_000 + i) for i in range(N)]


def arm(seed: int, p_cov: float, p_correct: float) -> dict:
    rng = np.random.default_rng(seed)
    return {k: {"cov1": bool(rng.random() < p_cov), "cov2": True, "correct": bool(rng.random() < p_correct),
                "rel_sigma": float(rng.lognormal(-5, 1)), "abs_err": float(rng.lognormal(-4, 1)), "crps_log": -0.3}
            for k in KEYS}


@pytest.fixture
def small_boot(monkeypatch):
    monkeypatch.setattr(phase95_compare, "N_BOOT", 300)
    return 300


def test_three_identical_seeds_reduce_to_the_per_seed_bootstrap(small_boot):
    split, sft, uni = arm(1, 0.4, 0.4), arm(2, 0.2, 0.4), arm(3, 0.2, 0.35)
    t = sm.seed_mean_tests([split, split, split], sft, uni, KEYS, n_boot=small_boot)
    one = sm.per_seed_tests(split, sft, uni, KEYS)
    assert t["S2"]["p_one_sided"] == one["S2"]["p_one_sided"]
    assert t["S3"]["p_one_sided"] == one["S3"]["p_one_sided"]
    assert t["S1a"]["diff"] == pytest.approx((one["S1a"]["after"] - one["S1a"]["before"]) / N)


def test_seed_mean_coverage_is_the_per_problem_average():
    seeds = [arm(s, 0.5, 0.5) for s in (4, 5, 6)]
    expected = np.mean([np.mean([d[k]["cov1"] for d in seeds]) for k in KEYS])
    t = sm.seed_mean_tests(seeds, arm(7, 0.0, 0.5), arm(8, 0.0, 0.5), KEYS, n_boot=50)
    assert t["S1a"]["seed_mean"] == pytest.approx(expected) and t["S1a"]["control"] == 0.0


def test_a_dominating_arm_is_supported_and_a_null_arm_is_not():
    ctrl = arm(9, 0.0, 0.5)
    better = [{k: {**v, "cov1": True} for k, v in arm(s, 0.0, 0.5).items()} for s in (10, 11, 12)]
    t = sm.seed_mean_tests(better, ctrl, ctrl, KEYS, n_boot=200)
    assert t["S1a"]["p_one_sided"] == 0.0 and t["holm"]["S1a"]["supported"]
    same = sm.seed_mean_tests([ctrl, ctrl, ctrl], ctrl, ctrl, KEYS, n_boot=200)
    assert same["S1a"]["p_one_sided"] == 1.0 and not same["holm"]["S1a"]["supported"]
    assert same["S3"]["p_one_sided"] == 0.0  # identical accuracy is non-inferior at a 5-point margin


def test_corrected_mode_changes_only_the_input_and_output_paths():
    """D67: --corrected reads the 4-bit unmerged evals and their controls on the same base, writes results/corrected."""
    import phase95_compare as pc
    import phase95_seed_mean as sm

    run = pc.ROOT / "results/train/some-run"
    old, new = pc.paths(run, False), pc.paths(run, True)
    assert set(old) == set(new)
    assert old["split"] == run / "compare/rl287/rollouts.jsonl" and old["sft"] == pc.OLD / "sft/rollouts.jsonl"
    assert old["out"] == run / "compare/phase95_family_S" and old["family_a"] is True
    assert new["split"] == run / "compare/rl287_4bit/rollouts.jsonl"
    assert new["sft"] == pc.UNIFORM / "sft_4bit/rollouts.jsonl" and new["uniform"] == pc.UNIFORM / "rl287_4bit/rollouts.jsonl"
    assert all(str(new[k]).startswith(str(pc.CORRECTED)) for k in ("posthoc", "sft_a5", "a5_out", "out"))
    assert sm.CORRECTED_INPUTS["eval_tape"] == "compare/rl287_4bit/rollouts.jsonl"
    assert sm.CORRECTED_INPUTS["controls"] == (new["sft"], new["uniform"])


def test_per_seed_output_names_keep_the_dotted_run_name():
    """Run names contain dots (v1.1.0): the per-seed outputs must not collapse into one file (bug found 2026-10-18)."""
    import phase95_compare as pc

    outs = {f"{pc.paths(pc.ROOT / f'results/train/x-v1.1.0-512-seed{s}', True)['out']}.json" for s in (0, 1, 2)}
    assert len(outs) == 3 and all(o.endswith(f"seed{s}.json") for s, o in zip((0, 1, 2), sorted(outs)))

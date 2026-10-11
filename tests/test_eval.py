"""Eval harness: calibration metrics on synthetic data, and end-to-end runs with fake backends."""

import json
import math

import numpy as np
import pytest

from physcal.config import load_config
from physcal.eval import calibration
from physcal.eval.rescore import rescore
from physcal.eval.run_eval import Generation, build_split, replot, run
from physcal.env.generator import env_config
from physcal.tape.tape import replay
from physcal.verify.reward import reward_config

SC = reward_config()["scoring"]
CAL = load_config("eval")["calibration"]


def _synthetic_breakdowns(n, sigma_scale, seed=0):
    """Truth y = 100; predictions mu = y + N(0, s^2), stated sigma = sigma_scale * s."""
    rng = np.random.default_rng(seed)
    s = 2.0
    out = []
    for mu in 100.0 + rng.normal(0.0, s, n):
        sigma = sigma_scale * s
        out.append({
            "reason": "ok", "mu_si": mu, "sigma_si": sigma, "truth_si": 100.0,
            "pit": float(calibration.stats.norm.cdf((100.0 - mu) / sigma)),
            "within_1sigma": abs(mu - 100) <= sigma, "within_2sigma": abs(mu - 100) <= 2 * sigma,
            "correct": abs(mu - 100) <= 1.0, "abs_rel_error": abs(mu - 100) / 100, "rel_sigma": sigma / 100,
            "reward": 0.0, "scores": {}, "score_steps": 0.0, "n_steps_requested": 0,
        })
    return out


def test_calibrated_predictions_have_nominal_coverage():
    s = calibration.summarize(_synthetic_breakdowns(20_000, 1.0), SC, CAL["reliability_levels"], CAL["pit_bins"])
    assert s["coverage_1sigma"] == pytest.approx(0.6827, abs=0.01)
    assert s["coverage_2sigma"] == pytest.approx(0.9545, abs=0.01)
    for r in s["reliability"]:
        assert r["observed"] == pytest.approx(r["nominal"], abs=0.015)
    counts = np.array(s["pit_histogram"]["counts"])
    assert counts.min() > 0.9 * counts.mean() and counts.max() < 1.1 * counts.mean()


def test_overconfident_predictions_undercover():
    s = calibration.summarize(_synthetic_breakdowns(20_000, 0.5), SC, CAL["reliability_levels"], CAL["pit_bins"])
    assert s["coverage_1sigma"] < 0.45
    counts = s["pit_histogram"]["counts"]
    assert counts[0] > 2 * counts[len(counts) // 2]  # U-shaped PIT


def test_failures_count_against_accuracy():
    bds = _synthetic_breakdowns(10, 1.0)
    bds += [{"reason": "parse_fail", "reward": -5.0, "scores": None, "score_steps": 0.0, "n_steps_requested": 2,
             "detail": "no <answer>...</answer> block"}] * 10
    s = calibration.summarize(bds, SC, CAL["reliability_levels"], CAL["pit_bins"])
    assert s["parse_fail_rate"] == 0.5
    assert s["accuracy"] == pytest.approx(s["accuracy_among_answered"] / 2)
    assert s["mean_score_by_rule"]["binary"] <= 0.5


def test_build_split_is_stratified_and_seeded():
    env_cfg, eval_cfg = env_config(), load_config("eval")
    probs = build_split(0, eval_cfg["splits"]["id"], eval_cfg, env_cfg, limit=24)
    cells = {(p.family, p.depth) for p in probs}
    assert len(cells) == 12 and all(p.depth <= 4 for p in probs)
    assert all(p.seed >= env_cfg["splits"]["eval_seed_offset"] for p in probs)
    extrap = build_split(1, eval_cfg["splits"]["depth_extrapolation"], eval_cfg, env_cfg, limit=6)
    assert {p.depth for p in extrap} == {5}
    assert not {p.seed for p in probs} & {p.seed for p in extrap}
    assert build_split(2, eval_cfg["splits"]["heldout_family"], eval_cfg, env_cfg) == []


def _oracle_backend(_model_id):
    def generate(conversations):
        out = []
        for conv in conversations:
            text = conv[-1]["content"]
            # The oracle looks the problem up by text (the harness only passes messages).
            p = _ORACLE[text]
            steps = "\n".join(f"{s} = {v:.6g} {p.intermediate_si_units[s]}" for s, v in p.intermediates.items())
            ans = f"{p.target_value:.6g} ± {0.005 * p.target_value:.3g} {p.target_unit}"
            out.append(Generation(f"Thinking.\n<steps>\n{steps}\n</steps>\n<answer>{ans}</answer>", 50, False))
        return out
    return generate


_ORACLE: dict = {}


def test_run_end_to_end_with_oracle(tmp_path):
    env_cfg, eval_cfg = env_config(), load_config("eval")
    for i, (name, scfg) in enumerate(eval_cfg["splits"].items()):
        for p in build_split(i, scfg, eval_cfg, env_cfg, limit=12):
            _ORACLE[p.text] = p
    m = run(eval_cfg, backend_factory=_oracle_backend, limit=12, output_dir=tmp_path)
    assert set(m["splits"]) == {"id", "depth_extrapolation"}
    assert set(m["skipped_splits"]) == {"heldout_family", "simulators"}
    for s in m["splits"].values():
        assert s["accuracy"] == 1.0 and s["parse_fail_rate"] == 0.0 and s["mean_step_score"] == 1.0
        assert s["coverage_2sigma"] == 1.0
    for f in ("metrics.json", "summary.md", "rollouts.jsonl", "rollouts.meta.json",
              "calibration_id.png", "calibration_depth_extrapolation.png", "reliability_all_splits.png"):
        assert (tmp_path / f).exists(), f
    json.loads((tmp_path / "metrics.json").read_text())
    assert replay(tmp_path / "rollouts.jsonl").ok
    for rule in ("binary", "crps_log"):
        rescored = rescore(tmp_path / "rollouts.jsonl", rule)
        assert all(s["accuracy"] == 1.0 for s in rescored.values())
    (tmp_path / "calibration_id.png").unlink()
    replot(tmp_path)
    assert (tmp_path / "calibration_id.png").exists()


def test_run_with_garbage_backend(tmp_path):
    def factory(_):
        return lambda convs: [Generation("no idea", 3, False) for _ in convs]
    m = run(load_config("eval"), backend_factory=factory, limit=4, output_dir=tmp_path)
    for s in m["splits"].values():
        assert s["parse_fail_rate"] == 1.0 and s["accuracy"] == 0.0 and s["coverage_1sigma"] is None
    assert (tmp_path / "calibration_id.png").exists()


def test_run_falls_back_on_oom(tmp_path):
    class OutOfMemoryError(RuntimeError):
        pass

    calls = []

    def factory(model_id):
        calls.append(model_id)
        if len(calls) == 1:
            def boom(convs):
                raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return boom
        return lambda convs: [Generation("<answer>1 ± 1 m</answer>", 5, False) for _ in convs]

    cfg = load_config("eval")
    m = run(cfg, backend_factory=factory, limit=2, output_dir=tmp_path)
    assert m["oom_fallback"] and m["model_id"] == cfg["model"]["fallback_id"]
    assert calls == [cfg["model"]["id"], cfg["model"]["fallback_id"]]


def test_multi_sample_groups(tmp_path):
    """k samples per problem: the first sample is right, the rest unparseable -> every group has variance."""
    import copy

    cfg = copy.deepcopy(load_config("eval"))
    cfg["generation"].update(do_sample=True, num_samples=3)
    env_cfg = env_config()
    for i, (name, scfg) in enumerate(cfg["splits"].items()):
        for p in build_split(i, scfg, cfg, env_cfg, limit=4):
            _ORACLE[p.text] = p

    def factory(_):
        oracle = _oracle_backend(None)
        def generate(convs):
            out = []
            for g in oracle(convs):
                out += [g, Generation("garbage", 1, False), Generation("garbage", 1, False)]
            return out
        return generate

    m = run(cfg, backend_factory=factory, limit=4, output_dir=tmp_path)
    groups = m["splits"]["id"]["groups"]
    assert groups["k"] == 3 and groups["n_groups"] == 4
    assert groups["pass_at_k"] == 1.0 and groups["mean_pass_rate"] == pytest.approx(1 / 3)
    assert groups["zero_variance_fraction_binary"] == 0.0 and groups["zero_variance_fraction_main_reward"] == 0.0
    assert replay(tmp_path / "rollouts.jsonl").ok


def test_greedy_with_several_samples_is_rejected(tmp_path):
    import copy

    cfg = copy.deepcopy(load_config("eval"))
    cfg["generation"].update(do_sample=False, num_samples=4)
    with pytest.raises(ValueError):
        run(cfg, backend_factory=lambda _: (lambda convs: []), limit=1, output_dir=tmp_path)

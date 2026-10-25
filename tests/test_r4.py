"""R4-1 / R4-5 (DECISIONS.md D113): mixed per-given noise levels, the D-pos-v2 / B-v2 data, stage R4's config, and
the frozen confirmatory set's guard. CPU only."""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.env import noise as noise_mode
from physcal.env.formulas import get_family
from physcal.env.generator import chain_of, generate_problem, problem_from_dict, regenerate, train_cells

ENV = load_config("env_v2")
LEVELS = [0.0, 0.01, 0.02, 0.05]
D80 = REPO_ROOT / "results" / "phase12_sft_data" / "problems.jsonl"
R4 = REPO_ROOT / "results" / "phase12_sft_data_r4"
CONF = REPO_ROOT / "data" / "eval_sets" / "2.0.0+bdbb2279" / "confirmatory"


# ------------------------------------------------------------------------------------------- the generator
@pytest.mark.skipif(not D80.exists(), reason="D80's data not built")
def test_default_path_unchanged_against_d80_problems():
    """D80's problems (built before the mixed option existed) regenerate byte-identically: noise 0 and noisy."""
    for line in list(D80.open())[:48]:
        d = json.loads(line)
        assert regenerate(d, ENV, split="train").to_dict() == d


def test_mixed_levels_are_per_given_recorded_and_replayable():
    seen = set()
    for i, (fam, depth) in enumerate(train_cells(ENV)):
        p = generate_problem(fam, depth, 70_000 + i, split="train", cfg=ENV, mixed_levels=LEVELS)
        again = generate_problem(fam, depth, 70_000 + i, split="train", cfg=ENV, mixed_levels=LEVELS)
        assert p.to_dict() == again.to_dict()
        assert regenerate(p.to_dict(), ENV, split="train").to_dict() == p.to_dict()
        n = p.noise
        assert n["mixed"] == LEVELS
        rel = {g["symbol"]: g.get("noise_rel", 0.0) for g in p.givens}
        assert {g: v["rel"] for g, v in n["givens"].items()} == rel
        assert n["level"] == max(rel.values())
        assert all(v in LEVELS for v in rel.values())
        seen |= set(rel.values())
        assert (n["sigma_star_log"] > 0) == any(rel.values())
        assert ("± " in p.text) == any(v.get("noise_rel") for v in p.givens + p.distractors)
        assert n["sigma_star_log_delta"] == pytest.approx(noise_mode.delta_sigma_log(
            noise_mode.log_sensitivities(get_family(fam), chain_of(p), p.given_values_si,
                                         ENV["generator"]["condition_rel_step"]), rel))
    assert seen == set(LEVELS)


def test_mixed_levels_need_noise_zero_and_known_levels():
    with pytest.raises(ValueError):
        generate_problem("kinematics", 1, 1, split="train", cfg=ENV, noise=0.01, mixed_levels=LEVELS)
    with pytest.raises(ValueError):
        generate_problem("kinematics", 1, 1, split="train", cfg=ENV, mixed_levels=[0.0, 0.03])


# ------------------------------------------------------------------------------------------- the data
def _rows(arm: str) -> list[dict]:
    return [json.loads(line) for line in (R4 / f"{arm}.jsonl").open()]


@pytest.mark.skipif(not (R4 / "manifest.json").exists(), reason="R4-1 data not built")
def test_r4_data_matches_manifest_and_d80_seeds():
    m = json.loads((R4 / "manifest.json").read_text())
    for name, sha in m["sha256"].items():
        assert hashlib.sha256((R4 / name).read_bytes()).hexdigest() == sha
    d80 = [json.loads(line) for line in D80.open()] if D80.exists() else None
    probs = [json.loads(line) for line in (R4 / "problems.jsonl").open()]
    assert [p["seed"] for p in probs] == list(range(60_000_000, 60_003_000))
    if d80:
        assert [(p["family"], p["depth"]) for p in probs] == [(p["family"], p["depth"]) for p in d80]
        same_chain = sum(a["chain"] == b["chain"] for a, b in zip(probs, d80)) / len(probs)
        assert same_chain > 0.9  # the main stream is untouched; only rejections differ
    assert m["problems_mixing_noisy_levels"] > 0.5 * len(probs)


@pytest.mark.skipif(not (R4 / "manifest.json").exists(), reason="R4-1 data not built")
def test_r4_explicit_rows_lines_sensitivities_copy():
    from physcal.eval.stage_d_analysis import copy_ok, mu_rel_slack, parse_line, valid_2sf
    from physcal.train import sft_r4

    cfg = sft_r4._config()
    tol = cfg["sensitivity_check"]
    probs = {json.loads(line)["seed"]: json.loads(line) for line in (R4 / "problems.jsonl").open()}
    rows = _rows("explicit")
    assert all(r["masked"] == [] for r in rows)
    for r in rows[::10]:
        p = problem_from_dict(probs[r["seed"]])
        ln = parse_line(r["completion"])
        assert ln["parsed"]
        noisy = [g for g in p.givens if g.get("noise_rel")]
        assert len(ln["terms"]) == len(noisy) == len(r["sensitivities"])
        assert [s for _, s in ln["terms"]] == [g["noise_rel"] for g in noisy]
        own = math.sqrt(sum((a * s) ** 2 for a, s in ln["terms"]))
        assert valid_2sf(ln["x"], own)
        assert copy_ok(r["completion"], ln["x"], mu_rel_slack(cfg))
        fd = noise_mode.log_sensitivities(get_family(p.family), chain_of(p), p.given_values_si,
                                          load_config(cfg["env_config"])["generator"]["condition_rel_step"])
        for t, (a, _) in zip(r["sensitivities"], ln["terms"]):
            assert a == t["value"]
            assert abs(t["value"] - fd[t["symbol"]]) <= max(tol["abs_tol"], tol["rel_tol"] * abs(fd[t["symbol"]]))
            assert f"S_{t['symbol']} = " in r["completion"]


@pytest.mark.skipif(not (R4 / "manifest.json").exists(), reason="R4-1 data not built")
def test_r4_supervised_rows_have_no_line_and_sigma_star():
    probs = {json.loads(line)["seed"]: json.loads(line) for line in (R4 / "problems.jsonl").open()}
    for r in _rows("supervised")[::25]:
        assert "sigma_rel" not in r["completion"] and "S_" not in r["completion"].split("<steps>")[0][-200:]
        assert r["sigma_log_answer"] == probs[r["seed"]]["noise"]["sigma_star_log"]
        assert r["masked"] == []


# ------------------------------------------------------------------------------------------- stage R4
def test_stage_r4_is_stage_d_except_data_and_budget():
    from physcal.train import stage_r4

    r4, d = load_config("stage_r4"), load_config("stage_d")
    for k in ("pinned_base", "lora", "train", "guards", "seed"):
        assert r4[k] == d[k], k
    ev_r4, ev_d = dict(r4["eval"]), dict(d["eval"])
    for k in ("max_new_tokens", "long_completion_tokens", "long_max_new_tokens", "max_reserved_gib"):  # D114 / D115
        ev_r4.pop(k)
    ev_d.pop("max_reserved_gib")
    assert ev_r4 == ev_d
    a = stage_r4.arm_config("dpos2")
    assert a["arm"] == "explicit" and a["sft_phase12_config"] == "sft_r4"
    assert stage_r4.arm_config("b2")["arm"] == "supervised"
    with pytest.raises(ValueError):
        stage_r4.arm_config("pos")


# ------------------------------------------------------------------------------------------- R4-5
@pytest.mark.skipif(not CONF.exists(), reason="confirmatory set not built")
def test_confirmatory_set_is_frozen_and_guarded(monkeypatch):
    from physcal.eval.run_eval import CONFIRMATORY_ENV, load_frozen_split

    pc = load_config("pairs")
    for s, name in (("a6", "a6_pairs"), ("a6b", "a6b_pairs"), ("a6c", "a6c_pairs")):
        m = json.loads((CONF / f"{name}_manifest.json").read_text())
        assert hashlib.sha256((CONF / f"{name}.jsonl").read_bytes()).hexdigest() == m["sha256"]
        assert m["export"]["seed_offset"] == pc["confirmatory"]["seed_offsets"][s] and m["n_pairs"] == 510
        assert "frozen" in m
    split = {"file": str((CONF / "a6_pairs.jsonl").relative_to(REPO_ROOT))}
    monkeypatch.delenv(CONFIRMATORY_ENV, raising=False)
    with pytest.raises(PermissionError):
        load_frozen_split(split, ENV, limit=2)
    monkeypatch.setenv(CONFIRMATORY_ENV, "1")
    assert len(load_frozen_split(split, ENV, limit=2)) == 2
    dev = load_config("eval_frozen_sets")["splits"]
    assert not any("confirmatory" in str(v.get("file", "")) for v in dev.values())


def test_confirmatory_seeds_disjoint_from_development_sets():
    pc = load_config("pairs")
    span = lambda ex: (ex["seed_offset"], ex["seed_offset"] + len(ex["depths"]) * len(ex["levels"]) *
                       ex["cell_seed_stride"])
    dev = [span(pc[k]) for k in ("export", "invariance_export", "allocation_export")]
    for s, key in (("a6", "export"), ("a6b", "invariance_export"), ("a6c", "allocation_export")):
        lo, hi = span({**pc[key], "seed_offset": pc["confirmatory"]["seed_offsets"][s]})
        assert all(hi <= a or lo >= b for a, b in dev)

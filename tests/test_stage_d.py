"""D109, Phase 12 stage D: the config (stage B's except the data), the two arms' masks (pos trains the line's numbers
and the answer's σ; fmt is D84's data exactly), the line parser and reference line, copy fidelity and the line
diagnostics, and the analysis (D1-D7, F1-F6) on synthetic eval tapes."""

import copy
import json
import math

import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.eval import stage_d_analysis as A
from physcal.train import stage_b, stage_d

DATA = REPO_ROOT / "results" / "phase12_sft_data" / "explicit.jsonl"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="Phase 12 SFT data not built")


def _tok():
    from physcal.base import base_config, pinned_base_dir

    try:
        d = pinned_base_dir(base_config("base_4bit"))
    except Exception:
        pytest.skip("the pinned base snapshot is not cached")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(d)


def _rows():
    return [json.loads(line) for line in DATA.open()]


# ------------------------------------------------------------------------------------------- config and data
def test_config_is_stage_b_except_the_data():
    cfg, b = stage_d._config(), stage_b._config()
    for k in ("sft_phase12_config", "pinned_base", "lora", "train", "guards", "max_reserved_gib", "eval"):
        assert cfg[k] == b[k], k
    assert cfg["data_arm"] == "explicit" and cfg["seed"] == 0 and cfg["order"] == ["pos", "fmt"]
    assert cfg["arms"]["pos"] == {"train_line_numbers": True, "train_sigma": True}
    d84 = load_config("sft_phase12")["arms"]["explicit"]
    assert cfg["arms"]["fmt"] == {k: v for k, v in d84.items() if k != "explicit_line"}
    a = stage_d.arm_config("pos")
    assert a["arm"] == "explicit" and a["seeds"] == [0] and a["output_dir"] == "results/stage_d/pos"
    assert str(stage_d.arm_dir("fmt")).endswith("results/stage_d/fmt/seed0")
    assert load_config("sft_phase12")["arms"]["explicit"]["train_line_numbers"] is False  # the shared config unchanged


@needs_data
def test_arms_share_the_text_and_differ_only_in_masks():
    pos, _, rp = stage_d.arm_rows("pos")
    fmt, _, rf = stage_d.arm_rows("fmt")
    assert [r["completion"] for r in pos] == [r["completion"] for r in fmt] == [r["completion"] for r in _rows()]
    assert rp["rows_with_masks_replaced"] == 3000 and rp["masked_spans_mean"] == 0.0
    assert rf["rows_with_masks_replaced"] == 0  # fmt is D84's recorded masks exactly
    for r in fmt:
        assert sorted(map(tuple, r["masked"])) == sorted({tuple(s) for s in r["spans"]["line_numbers"] + [r["spans"]["sigma"]]})


@needs_data
def test_pos_trains_the_numbers_fmt_masks_them_with_the_real_tokenizer():
    tok = _tok()
    pos, info_p = stage_d.tokenized_rows(tok, "pos", 2048)
    fmt, info_f = stage_d.tokenized_rows(tok, "fmt", 2048)
    assert info_p["n_rows"] == info_f["n_rows"] == 3000 and info_p["dropped_too_long"] == 0
    assert info_p["n_masked_mean"] == 0.0 and info_f["n_masked_mean"] > 0
    src = next(r for r in _rows() if r["noise"] > 0 and r["line"].count("^2") >= 2)
    i = [r["seed"] for r in pos].index(src["seed"])
    lab_p = tok.decode([t for t in pos[i]["labels"] if t != -100])
    lab_f = tok.decode([t for t in fmt[i]["labels"] if t != -100])
    assert src["line"] in lab_p and "sigma_rel = sqrt((" in lab_f and src["line"] not in lab_f
    sigma = src["completion"][slice(*src["spans"]["sigma"])]
    assert f"± {sigma} " in lab_p and f"± {sigma} " not in lab_f


def test_eval_config_is_stage_bs_with_the_arms_adapter():
    try:
        ev, eb = stage_d.eval_config("pos"), stage_b.eval_config(0)
    except Exception as exc:
        pytest.skip(f"pinned base not cached: {exc}")
    assert ev["model"]["lora"].endswith("results/stage_d/pos/seed0/final_adapter")
    assert ev["output_dir"].endswith("results/stage_d/pos/seed0/eval") and ev["run_id"] == "stage-d-pos-seed0-a6-sets"
    for k in ("splits", "generation", "prompt", "env_config"):
        assert ev[k] == eb[k], k
    assert {k: v for k, v in ev["model"].items() if k != "lora"} == {k: v for k, v in eb["model"].items() if k != "lora"}


# ------------------------------------------------------------------------------------------- the line
def test_parse_line_cases():
    p = A.parse_line("x\nsigma_rel = sqrt((1.1*0.05)^2 + (2*0.01)^2) = 0.059\n</steps>\n<answer>1 ± 0.06 m</answer>")
    assert p["parsed"] and p["terms"] == [(1.1, 0.05), (2.0, 0.01)] and p["x"] == 0.059
    assert A.parse_line("sigma_rel = 0\n<answer>1 ± 0 m</answer>") == {"present": True, "parsed": True, "zero": True,
                                                                     "terms": [], "x": 0.0}
    assert A.parse_line("sigma_rel = sqrt((3e-05*0.02)^2) = 6e-07\n<answer>")["terms"] == [(3e-05, 0.02)]
    assert A.parse_line("sigma_rel = sqrt((1.1*0.05)^2 + oops) = 0.06\n<answer>") == {"present": True, "parsed": False}
    assert A.parse_line("sigma_rel = about 5%\n<answer>") == {"present": True, "parsed": False}
    assert A.parse_line("<steps>\n</steps>\n<answer>1 ± 0.1 m</answer>") == {"present": False, "parsed": False}
    two = "sigma_rel = sqrt((1*0.05)^2) = 0.05\nsigma_rel = sqrt((2*0.05)^2) = 0.1\n<answer>x</answer>"
    assert A.parse_line(two)["x"] == 0.1  # the last line before <answer>
    assert not A.parse_line("<answer>x</answer>\nsigma_rel = 0")["present"]  # after the answer: ignored


def test_valid_2sf_and_mu_slack():
    assert A.valid_2sf(0.069, 0.0695) and A.valid_2sf(0.07, 0.0695) and not A.valid_2sf(0.068, 0.0695)
    assert A.valid_2sf(10.0, 9.96) and A.valid_2sf(0.0, 0.0) and not A.valid_2sf(0.1, 0.0)
    assert not A.valid_2sf(37.0, 36.4994) and A.valid_2sf(37.0, 36.4994, A.mu_rel_slack(load_config("sft_phase12")))
    assert A.mu_rel_slack({"mu_sig_figs": 4}) == pytest.approx(5e-4)


@needs_data
def test_training_lines_parse_reproduce_and_copy():
    from physcal.env.generator import problem_from_dict
    from physcal.train.sft_phase12 import propagation_line

    rows, slack, env = _rows(), A.mu_rel_slack(load_config("sft_phase12")), load_config("env_v2")
    probs = {json.loads(line)["seed"]: json.loads(line) for line in (DATA.parent / "problems.jsonl").open()}
    for i, r in enumerate(rows):
        ln = A.parse_line(r["completion"])
        assert ln["parsed"] and len(ln["terms"]) == r["line"].count("^2") and ln["x"] == r["sigma_log_answer"]
        assert A.copy_ok(r["completion"], ln["x"], slack) is True
        if i < 200:
            assert propagation_line(problem_from_dict(probs[r["seed"]]), env, ".2g")[0] == r["line"]
    r = next(r for r in rows if r["noise"] > 0)
    lo, hi = r["spans"]["sigma"]
    doubled = r["completion"][:lo] + format(2 * float(r["completion"][lo:hi]), ".2g") + r["completion"][hi:]
    assert A.copy_ok(doubled, A.parse_line(r["completion"])["x"], slack) is False


# ------------------------------------------------------------------------------------------- synthetic eval tapes
def _completion(p, ref, sens=None, sigma_rel=None):
    """A minimal completion: the reference line (or its sensitivities replaced), μ = f(x_meas) in SI, σ = X·|μ|
    (or sigma_rel·|μ|)."""
    mu = p["noise"]["point_estimate_si"]
    if ref["zero"]:
        line, x = "sigma_rel = 0", 0.0
    else:
        t = [(a if sens is None else sens, s) for a, s in ref["terms"]]
        x = float(format(math.sqrt(sum((a * s) ** 2 for a, s in t)), ".2g"))
        line = "sigma_rel = sqrt(" + " + ".join(f"({format(a, '.2g')}*{s:g})^2" for a, s in t) + f") = {format(x, '.2g')}"
    sig = x if sigma_rel is None else sigma_rel
    return f"<steps>\n{line}\n</steps>\n<answer>{mu!r} ± {format(sig * abs(mu), '.2g')} {p['target_si_unit']}</answer>"


def _tape(path, problems, refs, **kw):
    from physcal.env.generator import problem_from_dict
    from physcal.verify.reward import reward, reward_config

    rw = reward_config()
    lines = []
    for p, ref in zip(problems, refs):
        q = {k: v for k, v in p.items() if k != "_set"}
        f = {k: (v(p) if callable(v) else v) for k, v in kw.items()}
        text = _completion(p, ref, **f)
        r, bd = reward(problem_from_dict(q), text, rw)
        lines.append(json.dumps({"split": p["_set"], "problem": q, "completion": text, "reward": r, "breakdown": bd}))
    path.mkdir(parents=True)
    (path / "rollouts.jsonl").write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def eval_sets():
    from physcal.eval import stage_a

    problems = stage_a.load_sets(load_config("stage_a"))
    p12 = load_config("sft_phase12")
    return problems, A.reference_lines(problems, load_config(p12["env_config"]), p12["sigma_format"])


def test_line_diagnostics_perfect_and_wrong_sensitivities(eval_sets, tmp_path):
    from physcal.eval import stage_b_analysis as B

    problems, refs = eval_sets
    slack = A.mu_rel_slack(load_config("sft_phase12"))
    _tape(tmp_path / "ref", problems, refs)
    d, xs = A.line_diagnostics(problems, B.read_records(tmp_path / "ref"), refs, 0.1, slack)
    s = d["summary"]
    assert s["line_parsed"] == 1.0 and s["sensitivity_all_exact"] == 1.0 and s["levels_ok"] == 1.0
    assert s["copy_ok"] == 1.0 and s["arithmetic_ok"] == 1.0 and s["median_abs_log_x_ref"] == 0.0
    assert list(xs) == [r["x"] for r in refs]
    _tape(tmp_path / "ones", problems, refs, sens=1.0)
    s1 = A.line_diagnostics(problems, B.read_records(tmp_path / "ones"), refs, 0.1, slack)[0]["summary"]
    assert s1["sensitivity_all_exact"] < 0.5 and s1["copy_ok"] == 1.0 and s1["arithmetic_ok"] == 1.0
    _tape(tmp_path / "nocopy", problems, refs, sigma_rel=lambda p: 3 * p["noise"]["sigma_star_log"])
    s2 = A.line_diagnostics(problems, B.read_records(tmp_path / "nocopy"), refs, 0.1, slack)[0]["summary"]
    assert s2["sensitivity_all_exact"] == 1.0 and s2["copy_ok"] < 0.1


@pytest.mark.skipif(not (REPO_ROOT / "results/x1_text_sigma/a6c/predictions.jsonl").exists(), reason="no X1")
def test_analysis_d_and_f_families(eval_sets, tmp_path):
    from physcal.eval import noise_heuristics

    problems, refs = eval_sets
    rss = lambda p: noise_heuristics.sigma(p, "rss", "givens") or 0.0
    _tape(tmp_path / "pos" / "seed0" / "eval", problems, refs)  # the reference line, copied: first-order σ
    _tape(tmp_path / "fmt" / "seed0" / "eval", problems, refs, sens=1.0, sigma_rel=rss)  # a surface reader
    for s in (0, 1, 2):
        _tape(tmp_path / "stage_b" / f"seed{s}" / "eval", problems, refs, sigma_rel=rss)
    cfg = copy.deepcopy(A._config())
    cfg["analysis"].update(bootstrap=200, stage_b_root=str(tmp_path / "stage_b"))
    res = A.analyze(cfg, tmp_path)
    td, tf = res["tests_d"], res["tests_f"]
    assert all(td[k]["holm"]["rejected"] for k in A.TESTS_D), {k: td[k].get("p_one_sided") for k in A.TESTS_D}
    assert not tf["F2"]["holm"]["rejected"] and not tf["F6"]["holm"]["rejected"]
    assert res["reading"][0].startswith("trained explicit propagation propagates")
    assert "3 seeds of D-pos are warranted" in res["reading"][-1]
    assert res["arms"]["pos"]["lines"]["summary"]["copy_ok"] == 1.0
    assert res["arms"]["fmt"]["lines"]["summary"]["sensitivity_all_exact"] < 0.5
    assert (tmp_path / "summary.md").exists() and (tmp_path / "pos" / "results.json").exists()


@pytest.mark.skipif(not (REPO_ROOT / "results/x1_text_sigma/a6c/predictions.jsonl").exists(), reason="no X1")
def test_analysis_with_a_missing_arm(eval_sets, tmp_path):
    problems, refs = eval_sets
    _tape(tmp_path / "pos" / "seed0" / "eval", problems, refs)
    cfg = copy.deepcopy(A._config())
    cfg["analysis"].update(bootstrap=200, stage_b_root=str(tmp_path / "no_stage_b"))
    res = A.analyze(cfg, tmp_path)
    assert res["arms"]["fmt"] is None and res["stage_b"] is None
    assert res["tests_d"]["D6"]["missing"] and res["tests_d"]["D7"]["missing"]
    assert all(res["tests_f"][k]["missing"] and not res["tests_f"][k]["holm"]["rejected"] for k in A.TESTS_F)
    assert "not run" in (tmp_path / "summary.md").read_text()

"""Phase 12 SFT data (D80, prep): the three arms on the same problems, the answer = f(x_meas), σ read back by the
verifier, the propagation line, the confidence channel, and each arm's loss masking."""

import copy
import hashlib
import json

import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.train import sft_phase12 as p12

TOKENIZER_DIR = REPO_ROOT / "results" / "sft" / "qwen2.5-math-1.5b-instruct" / "merged"
CASES = [("kinematics", 2, 0.02), ("energy", 3, 0.05), ("dynamics", 4, 0.01), ("dynamics", 1, 0.0),
         ("energy", 2, 0.0), ("kinematics", 4, 0.05)]


@pytest.fixture(scope="module")
def made():
    from physcal.env.generator import generate_problem

    cfg = p12._config()
    env = load_config(cfg["env_config"])
    out = []
    for i, (f, d, s) in enumerate(CASES):
        prob = generate_problem(f, d, cfg["seed_offset"] + 90_000 + i, split="train", cfg=env, noise=s)
        c = p12.confidence_draw(prob.seed, cfg["confidence"]["bins"])
        out.append((prob, {arm: p12.build_example(prob, arm, cfg, env, c) for arm in cfg["arms"]}))
    return cfg, env, out


def test_existing_prompts_are_unchanged():
    from physcal.prompt import SYSTEM_PROMPTS

    h = {k: hashlib.sha256(v.encode()).hexdigest()[:16] for k, v in SYSTEM_PROMPTS.items()}
    assert h["v4_worked_example"] == "c73d5a99d2c7cf47" and h["v1"] == "3834502c46e5629c"
    v5 = SYSTEM_PROMPTS["v5_two_channel"]
    assert "<confidence>c%</confidence>" in v5 and "99%" in v5 and "nothing after it" in v5


def test_answers_are_the_point_estimate_and_sigma_reads_back(made):
    from physcal.verify.reward import reward, reward_config

    cfg, _, out = made
    rw = reward_config()
    for prob, rows in out:
        pe_si, _ = p12.point_estimate(prob)
        for arm, r in rows.items():
            _, bd = reward(prob, r["completion"], rw)
            assert bd["reason"] == "ok", (arm, bd)
            assert bd["mu_si"] == pytest.approx(pe_si, rel=1e-3)  # 4 s.f., the computation from the displayed values
            if prob.noise["level"] > 0:
                assert pe_si != prob.target_value_si  # never the noise-free truth
            want = r["sigma_log_answer"]
            got = bd["sigma_si"] / abs(bd["mu_si"])
            assert (got == 0.0) if want == 0.0 else got == pytest.approx(want, rel=0.05)


def test_arms_differ_only_in_the_sigma_channel(made):
    cfg, _, out = made
    for prob, rows in out:
        imp, exp_, sup = rows["implicit"], rows["explicit"], rows["supervised"]
        assert imp["completion"] == sup["completion"]  # same text, different masks
        assert exp_["completion"].replace(exp_["line"] + "\n", "", 1).replace(
            exp_["completion"][slice(*exp_["spans"]["sigma"])], imp["completion"][slice(*imp["spans"]["sigma"])], 1) \
            == imp["completion"]
        assert len({r["confidence"] for r in rows.values()}) == 1
        for r in rows.values():
            c = r["completion"]
            assert c[slice(*r["spans"]["confidence"])] == str(r["confidence"])
            assert float(c[slice(*r["spans"]["sigma"])]) == pytest.approx(r["sigma_log_answer"] * abs(
                p12.point_estimate(prob)[1]), rel=0.05, abs=0)
            assert p12.parse_confidence(c, cfg["confidence"]["bins"]) == r["confidence"]


def test_propagation_line(made):
    from physcal.verify.parse import parse_completion

    _, _, out = made
    for prob, rows in out:
        r = rows["explicit"]
        steps = parse_completion(r["completion"]).steps
        assert steps["sigmarel"].value == r["sigma_log_answer"]
        if prob.noise["level"] == 0:
            assert r["line"] == "sigma_rel = 0" and r["sigma_log_answer"] == 0
            continue
        assert r["sigma_log_delta_exact"] == pytest.approx(prob.noise["sigma_star_log_delta"], rel=1e-9)
        assert r["sigma_log_answer"] == pytest.approx(r["sigma_log_delta_exact"], rel=0.1)  # .2g terms and result
        nums = [r["completion"][a:b] for a, b in r["spans"]["line_numbers"]]
        assert len(nums) == 2 * len(prob.givens) + 1 and nums[-1] == format(r["sigma_log_answer"], ".2g")


def test_confidence_parser_refuses_malformed_blocks():
    bins = [10, 50, 99]
    ok = "<answer>1 ± 0 m</answer>\n<confidence>50%</confidence>"
    assert p12.parse_confidence(ok, bins) == 50
    for bad in ("<answer>1 ± 0 m</answer>\n<confidence>55%</confidence>",
                "<answer>1 ± 0 m</answer>\n<confidence>50%</confidence><confidence>10%</confidence>",
                "<confidence>50%</confidence><answer>1 ± 0 m</answer>", "<answer>1 ± 0 m</answer>",
                "<answer>1 ± 0 m</answer>\n<confidence>0.5</confidence>"):
        assert p12.parse_confidence(bad, bins) is None
    draws = [p12.confidence_draw(s, bins) for s in range(300)]
    assert set(draws) == set(bins) and p12.confidence_draw(7, bins) == p12.confidence_draw(7, bins)


def test_masking_per_arm(made):
    """implicit masks σ; supervised masks nothing; explicit (D84) masks the line's numbers and the answer's σ, keeping
    the line's structure (sqrt, the ^2 exponents) trained. Rows built since D84 carry their masks; a config that
    disagrees with them is refused."""
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        pytest.skip("local tokenizer not available")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    cfg, _, out = made
    prob, rows = out[1]  # energy, depth 3, 5% noise

    def masked_text(row, c):
        t = p12.tokenize(row, prob.text, tok, c)
        n_prompt = len(t["input_ids"]) - len(tok(row["completion"], add_special_tokens=False)["input_ids"]) - 1
        comp = t["input_ids"][n_prompt:-1]
        return [tok.decode([i]) for i, l in zip(comp, t["labels"][n_prompt:-1]) if l == -100]

    sigma = rows["implicit"]["completion"][slice(*rows["implicit"]["spans"]["sigma"])]
    assert "".join(masked_text(rows["implicit"], cfg)).strip() == sigma
    assert masked_text(rows["supervised"], cfg) == []
    ex = rows["explicit"]
    got = "".join(masked_text(ex, cfg))
    nums = [ex["completion"][a:b] for a, b in ex["spans"]["line_numbers"]]
    assert all(n in got for n in nums) and ex["completion"][slice(*ex["spans"]["sigma"])] in got
    assert "sqrt" not in got and "^" not in got and ")" not in got  # the format stays trained
    assert p12.masked_spans(ex, cfg) == sorted(ex["spans"]["line_numbers"] + [ex["spans"]["sigma"]])
    trained = copy.deepcopy(cfg)
    trained["arms"]["explicit"].update(train_sigma=True, train_line_numbers=True)  # D80's original default
    assert masked_text(ex, trained) == []
    no_c = copy.deepcopy(cfg)
    no_c["confidence"]["train"] = False
    assert p12.masked_spans(rows["supervised"], no_c) == [rows["supervised"]["spans"]["confidence"]]
    stale = dict(ex, masked=p12.masked_spans(ex, trained))  # data built under another config
    with pytest.raises(ValueError, match="rebuild"):
        p12.tokenize(stale, prob.text, tok, cfg)


def test_built_data_manifest():
    out = REPO_ROOT / p12._config()["output_dir"]
    if not (out / "manifest.json").exists():
        pytest.skip("Phase 12 SFT data not built")
    m = json.loads((out / "manifest.json").read_text())
    cfg = p12._config()
    assert m["seeds"] == [cfg["seed_offset"], cfg["seed_offset"] + cfg["n_examples"] - 1]
    assert m["generator_version"].startswith("2.0.0+")
    assert all("/5/" not in k and not k.startswith("optics") for k in m["cells"])  # held-out depth / family
    assert {float(k.split("noise=")[1]) for k in m["cells"]} == set(cfg["noise_levels"])
    for name, sha in m["sha256"].items():
        if (out / name).exists():
            assert hashlib.sha256((out / name).read_bytes()).hexdigest() == sha
    assert m["built"]["explicit.jsonl"].startswith("D84")  # rebuilt with the format-only masks (your decision)
    assert m["config"]["arms"]["explicit"] == cfg["arms"]["explicit"]
    if (out / "explicit.jsonl").exists():
        with (out / "explicit.jsonl").open() as f:
            row = json.loads(f.readline())
        assert row["masked"] == p12.masked_spans(row, cfg) and len(row["masked"]) >= 2

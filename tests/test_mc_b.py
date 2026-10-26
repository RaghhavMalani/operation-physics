"""MC-B (R4-7, D119): the reference programs, the prompt, the sandbox's static check and limits, the failure
accounting, and the pipeline end to end with a fake model (generate -> execute -> analyze). CPU only."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.eval import mc_b, sandbox

CFG = load_config("mc_b")
MC_A = REPO_ROOT / "results" / "mc_a" / "a6c_pairs"
PROGRAM = "```python\nimport math\n\ndef solve({args}):\n{body}\n```"


def _first(set_name="a6c_pairs", n=1):
    return mc_b.problems_of(set_name)[:n]


# ------------------------------------------------------------------------------------------- the reference programs
@pytest.mark.parametrize("set_name,stride", [("a6c_pairs", 1), ("a6_pairs", 4), ("a6b_pairs", 4)])
def test_reference_programs_reproduce_mc_a_oracle(set_name, stride):
    r = mc_b.reference_check(set_name, stride)
    assert r["n"] > 0 and not r["failures"] and r["excluded"] == 0
    assert r["max_abs_sigma_diff"] < 1e-9 and r["max_abs_ln_diff"] < 1e-9


def test_examples_follow_the_selection_and_the_prompt_has_no_answer_format():
    ex = mc_b.examples(CFG)
    sel = CFG["prompt"]["examples"]
    assert [[p["family"], p["depth"]] for p in ex] == sel["cells"]
    for p in ex:
        assert sel["seed_offset"] <= p["seed"] < sel["seed_offset"] + sel["max_tries"]
        assert p["distractors"] and any(g.get("noise_rel") for g in p["givens"])
        assert p["noise"]["mixed"] == sel["mixed_levels"]
    msgs = mc_b.messages(_first()[0]["text"], CFG)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert not any("<steps>" in m["content"] or "<answer>" in m["content"] for m in msgs[1:])
    for m in msgs[2:5:2]:
        tree, why = sandbox.static_check(sandbox.extract(m["content"]))
        assert tree is not None, why
    for s in CFG["sets"]:  # every eval text loses exactly its format sentence, nothing else
        for p in mc_b.problems_of(s)[::97]:
            t = mc_b.prompt_text(p["text"], CFG)
            assert "<steps>" not in t and t.split("\n\n")[:-1] == p["text"].split("\n\n")[:-1]


def test_prompt_fits_the_budget():
    from physcal.base import base_config, pinned_base_dir

    if not (REPO_ROOT / pinned_base_dir(base_config("base_4bit"))).exists():
        pytest.skip("pinned base not on disk")
    r = mc_b.prompt_report(CFG)
    assert r["fits"] and r["longest_prompt_tokens"] + r["max_new_tokens"] <= r["max_seq_length"]


# ------------------------------------------------------------------------------------------- the static check
@pytest.mark.parametrize("code,why", [
    ("import os\ndef solve(x):\n    return x", "import"),
    ("from os import path\ndef solve(x):\n    return x", "import"),
    ("def solve(x):\n    return open('/etc/passwd').read()", "forbidden"),
    ("def solve(x):\n    return ().__class__", "underscore"),
    ("def solve(x):\n    return getattr(x, 'real')", "forbidden"),
    ("def solve(x):\n    return eval('x')", "forbidden"),
    ("def solve(x):\n    m = __import__('os')\n    return x", "underscore"),
    ("class A:\n    pass\ndef solve(x):\n    return x", "ClassDef"),
    ("def solve(x):\n    global y\n    return x", "Global"),
    ("def solve(x):\n    with x:\n        return x", "With"),
    ("f = lambda x: x\ndef solve(x):\n    return f(x)", "lambda"),
    ("def solve(x)\n    return x", "syntax"),
])
def test_static_check_rejects(code, why):
    tree, reason = sandbox.static_check(code)
    assert tree is None and why in reason


def test_static_check_allows_math_print_loops_and_inner_lambdas():
    code = ("import math\nfrom math import sqrt\nG = 9.81\n\ndef solve(m, h):\n    f = lambda a: a * 2\n"
            "    s = 0.0\n    for i in range(3):\n        s += i\n    print(s)\n    return f(math.sqrt(m) * sqrt(h))\n"
            "print(solve(1.0, 4.0))\n")
    tree, why = sandbox.static_check(code)
    assert tree is not None, why


def test_signature_rules():
    tree, _ = sandbox.static_check("def solve(m, v, g=9.81):\n    return m")
    assert sandbox.signature(tree, {"m", "v", "len_cart"}) == (["m", "v"], None)  # g has a default: not passed
    tree, _ = sandbox.static_check("def solve(m, q):\n    return m")
    assert sandbox.signature(tree, {"m", "v"})[0] is None
    tree, _ = sandbox.static_check("def solve(*a):\n    return 1")
    assert sandbox.signature(tree, {"m"})[0] is None
    tree, _ = sandbox.static_check("def solve(m, *, v):\n    return 1")
    assert sandbox.signature(tree, {"m", "v"})[0] is None


# ------------------------------------------------------------------------------------------- execution and failures
def _exec(body: str, args: str, problem=None):
    p = problem or _first()[0]
    calls = mc_b.run_inputs(p, "a6c_pairs", load_config("mc_a"), load_config("env_v2"))[0]
    return mc_b.execute_one(p, PROGRAM.format(args=args, body=body), calls, CFG)


def test_failure_types():
    p = _first()[0]
    g = p["givens"][0]["symbol"]
    assert mc_b.execute_one(p, "no code here", [{}], CFG)["failure"] == "no_program"
    assert mc_b.execute_one(p, "```python\nprint(1)\n```", [{}], CFG)["failure"] == "no_program"
    assert _exec("    return 1.0", "zz")["failure"] == "signature"
    assert _exec("    import os\n    return 1.0", g)["failure"] == "rejected"
    assert _exec(f"    return {g} / 0", g)["failure"] == "exception"
    assert _exec("    while True:\n        pass", g)["failure"] == "timeout"
    assert _exec("    return -1.0", g)["failure"] == "bad_output"
    assert _exec("    return (-1.0) ** 0.5", g)["failure"] == "bad_output"  # complex
    assert _exec("    return float('nan')", g)["failure"] == "bad_output"
    bomb = _exec("    x = [0.0] * (10 ** 9)\n    return 1.0", g)
    assert bomb["failure"] in ("exception", "timeout")
    ex = _exec(f"    print('hi')\n    return {g} * 2", g)
    assert ex["failure"] is None and len(ex["y_si"]) == 9


def test_a_failure_on_one_perturbed_run_fails_the_program():
    p = _first()[0]
    g = p["givens"][0]["symbol"]
    calls = mc_b.run_inputs(p, "a6c_pairs", load_config("mc_a"), load_config("env_v2"))[0]
    cut = sorted(c[g] for c in calls)[4]  # fails on the runs whose value is below the median
    ex = mc_b.execute_one(p, PROGRAM.format(args=g, body=f"    if {g} < {cut!r}:\n        raise ValueError('x')\n"
                                                          f"    return {g}"), calls, CFG)
    assert ex["failure"] == "exception" and any(v is None for v in ex["y_si"]) and any(v is not None for v in ex["y_si"])
    est = mc_b.estimates({**ex, "ln_oracle": [0.0] * 9}, 8, CFG)
    assert est["failed"] and est["mc_b"] is None and est["mu0_si"] is None


def test_estimates_from_an_exact_program_equal_mc_a_oracle():
    from physcal.eval import mc_a

    p = _first()[0]
    calls, ln_oracle = mc_b.run_inputs(p, "a6c_pairs", load_config("mc_a"), load_config("env_v2"))
    ex = {**mc_b.execute_one(p, f"```python\n{mc_b.reference_program(p)}```", calls, CFG), "ln_oracle": ln_oracle}
    for k in (2, 4, 8):
        assert abs(mc_b.estimates(ex, k, CFG)["mc_b"] - mc_a.sd(ln_oracle[1:k + 1])) < 1e-12
    assert mc_b.exact_program(ex, 1e-3)
    assert abs(mc_b.estimates(ex, 8, CFG)["mu0_si"] - p["noise"]["point_estimate_si"]) < 1e-9 * p["noise"]["point_estimate_si"]


# ------------------------------------------------------------------------------------------- end to end (fake model)
@dataclass
class _Gen:
    text: str
    n_tokens: int = 50
    truncated: bool = False


class FakeModel:
    """Returns, per problem, the reference program, a 10%-off program, or prose (by seed), as a model would."""

    def __init__(self, problems, kind):
        self.by_text = {mc_b.prompt_text(p["text"], CFG): p for p in problems}
        self.kind = kind
        self.calls = 0

    def __call__(self, conversations):
        self.calls += 1
        out = []
        for conv in conversations:
            assert conv[0]["role"] == "system" and len(conv) == 6
            p = self.by_text[conv[-1]["content"]]
            kind = self.kind(p)
            prog = mc_b.reference_program(p)
            if kind == "exact":
                out.append(_Gen(f"```python\n{prog}```"))
            elif kind == "off":
                out.append(_Gen("```python\n" + prog.replace("    return ", "    return 1.1 * ") + "```"))
            else:
                out.append(_Gen("I would compute this step by step.", 736, True))
        return out


@pytest.mark.skipif(not (MC_A / "meta.json").exists(), reason="MC-A's runs not on disk")
def test_end_to_end_with_a_fake_model(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["output_dir"] = str(tmp_path / "mc_b")
    cfg["chunk_problems"] = 16
    cfg["analysis"]["bootstrap"] = 200
    probs = mc_b.problems_of("a6c_pairs")[:40]  # 20 whole pairs (lo / hi adjacent)
    kind = lambda p: "exact" if p["seed"] % 5 else ("off" if p["seed"] % 10 else "prose")
    fake = FakeModel(probs, kind)
    meta = mc_b.generate("a6c_pairs", limit=40, backend=fake, cfg=cfg)
    assert meta["n_new"] == 40 and fake.calls == 3
    assert mc_b.generate("a6c_pairs", limit=40, backend=fake, cfg=cfg)["n_new"] == 0  # resumable: nothing redone
    r = mc_b.execute("a6c_pairs", limit=40, cfg=cfg)
    assert r["n_executed"] == 40 and r["n_excluded"] == 0
    res = mc_b.analyze("a6c_pairs", cfg)
    kinds = [kind(p) for p in probs]
    assert res["n_used"] == 40 and res["n_generated"] == 40
    assert res["programs"]["failure_rate"] == pytest.approx(kinds.count("prose") / 40)
    assert res["programs"]["failure_by_type"]["no_program"] == pytest.approx(kinds.count("prose") / 40)
    assert res["programs"]["exact_program"] == pytest.approx(kinds.count("exact") / 40)
    assert res["programs"]["run0_computation_correct"] == pytest.approx(kinds.count("exact") / 40)
    for t in mc_b.TESTS:
        assert "holm" in res["tests"][t] or res["tests"][t].get("missing")
    assert set(res["comparators"]) >= {"mc_a_oracle", "mc_a_robust", "mc_a_raw", "sigma_star", "rss", "x1", "dpos2_a"}
    assert (Path(cfg["output_dir"]) / "a6c_pairs" / "summary.md").read_text().startswith("# MC-B on a6c_pairs")

    # every program exact: MC-B's σ is MC-A's oracle σ problem by problem, so the scores coincide
    cfg2 = copy.deepcopy(cfg)
    cfg2["output_dir"] = str(tmp_path / "mc_b_exact")
    mc_b.generate("a6c_pairs", limit=40, backend=FakeModel(probs, lambda p: "exact"), cfg=cfg2)
    mc_b.execute("a6c_pairs", limit=40, cfg=cfg2)
    r2 = mc_b.analyze("a6c_pairs", cfg2)
    a, o = r2["variants"]["mc_b_k8"]["oracle_centred"], r2["comparators"]["mc_a_oracle"]["oracle_centred"]
    assert r2["programs"]["failure_rate"] == 0 and r2["programs"]["exact_program"] == 1
    assert a["expected"][0] == pytest.approx(o["expected"][0], abs=1e-12)
    assert r2["mc_b_minus_mc_a_oracle"]["value"] == pytest.approx(0.0, abs=1e-12)

"""Round-5 diagnostics (DECISIONS.md D133-): the descriptive helpers, on synthetic pairs. CPU only."""

import math

import numpy as np

from physcal.eval import a6_pairs
from physcal.eval import r5_diag as D

AC = {"bootstrap": 200, "seed": 0, "criterion_ci_level": 0.90}
DC = {"gross_factor": 10, "influence_top": 2, "theil_sen_ci": 0.90}


def _pair(i: int, level: float, x: float, y: float, ok: bool = True) -> dict:
    key = ("kinematics", 5, level, 1000 + i)
    lo = {"pair": key, "member": "lo", "sigma_star_log": 0.01, "stated_log": math.log(0.01), "ok": ok, "truth": 1.0,
          "computation_correct": ok}
    hi = {**lo, "member": "hi", "sigma_star_log": 0.01 * math.exp(x), "stated_log": math.log(0.01) + y}
    return {"lo": lo, "hi": hi}


def test_influence_finds_the_reversed_high_leverage_pair():
    x = np.linspace(0.4, 1.0, 30)
    y = x.copy()
    x = np.append(x, 1.6)
    y = np.append(y, -1.4)  # one pair far out in x, with the stated direction reversed
    inf = D.influence(x, y)
    assert int(np.argmax(np.abs(inf))) == 30 and inf[30] < 0  # it pulls the slope down
    keep = np.arange(31) != 30
    assert math.isclose(a6_pairs.ols_slope(x[keep], y[keep]), 1.0, rel_tol=1e-9)


def test_describe_cell_counts_reversals_levels_and_removes_the_influential_pairs():
    pairs = [_pair(i, 0.05 if i % 2 else 0.02, 0.4 + 0.02 * i, 0.4 + 0.02 * i) for i in range(30)]
    pairs += [_pair(30, 0.05, 1.6, -1.4), _pair(31, 0.05, 1.5, -1.2)]
    types = {p["lo"]["pair"]: "value_dependent" for p in pairs}
    c = D.describe_cell(pairs, pairs, types, AC, DC)
    assert c["n_used"] == 32 and c["n_reversed"] == 2
    assert c["slope"] < 0.5 and math.isclose(c["slope_without_top2"], 1.0, rel_tol=1e-9)
    assert {t["pair"][3] for t in c["top_influence"]} == {1030, 1031}
    assert c["by_level"]["0.05"]["n_reversed"] == 2 and c["by_level"]["0.02"]["n_reversed"] == 0
    assert c["n_gross_residual"] == 2  # |y - x| = 3.0 and 2.7 > ln 10
    assert c["by_type"]["value_dependent"]["n_used"] == 32 and c["by_type"]["power_law"]["slope"] is None


def test_safe_eval_is_arithmetic_only():
    assert D.safe_eval("-(12.66 + 2*130.9)/(2*12.66)", {}, 12) == -(12.66 + 2 * 130.9) / (2 * 12.66)
    assert D.safe_eval("s3/s2", {"s3": 6.0, "s2": 3.0}, 12) == 2.0
    assert D.safe_eval("sqrt(x)^2", {"x": 4.0}, 12) == 4.0
    for bad in ("__import__('os')", "x.real", "open('f')", "2**100", "(-8)**(1/3)", "1/0", "y + 1", "[1][0]"):
        assert D.safe_eval(bad, {"x": 1.0}, 12) is None


def _problem(sig_star: float = 0.05) -> dict:
    return {"depth": 3, "_set": "a6c_pairs", "noise": {"point_estimate_si": 100.0, "sigma_star_log": sig_star}}


def _ref() -> dict:  # two noisy givens: u (value-dependent, S = 2.0) and t (power law, S = 1)
    sens = {"u": {"value": 2.0, "power_law": False}, "t": {"value": 1.0, "power_law": True}}
    return {"terms": [(2.0, 0.02), (1.0, 0.01)], "sens": sens}


def _rec(line: str, s_lines: str, sigma_si: float) -> dict:
    return {"completion": f"{s_lines}\n<steps>\n{line}\n</steps>\n<answer>100 ± {sigma_si} m</answer>",
            "breakdown": {"reason": "ok", "mu_si": 100.0, "sigma_si": sigma_si}}


def test_decompose_factors_and_causes():
    exact = math.sqrt((2.0 * 0.02) ** 2 + (1.0 * 0.01) ** 2)
    vals = {"v": 6.0, "s": 3.0}
    # the formula for S_u is wrong (v/s/2 = 1, not 2): the sensitivity factor is the cause; X and the copy are right
    rec = _rec("sigma_rel = sqrt((1*0.02)^2 + (1*0.01)^2) = 0.022", "S_u = v/s/2 = 6/3/2 = 1\nS_t = 1", 2.2)
    ref = _ref()
    ref["sens"]["u"]["value"] = 2.0
    d = D.decompose(_problem(), ref, rec, vals, 0.10, 1e-3, 12)
    assert d["decomposable"] and d["cause"] == "sensitivity: value-dependent: formula"
    f = d["log_factors"]
    assert math.isclose(math.exp(sum(f.values())), (2.2 / 100) / exact, rel_tol=1e-9)  # the factors are exact
    assert math.isclose(d["sigma"]["S"], 0.022 / math.sqrt(0.02 ** 2 + 0.01 ** 2) * exact, rel_tol=1e-9)
    assert d["sigma"]["exact"] == exact and math.isclose(d["sigma"]["a"], 0.022, rel_tol=1e-12)
    # the right formula, its own wrong intermediates: substitution (the answer is not computation-correct)
    rec2 = _rec("sigma_rel = sqrt((1*0.02)^2 + (1*0.01)^2) = 0.022", "S_u = v/s = 3/3 = 1\nS_t = 1", 2.2)
    rec2["breakdown"]["mu_si"] = 50.0
    rec2["breakdown"]["sigma_si"] = 1.1
    d2 = D.decompose(_problem(), ref, rec2, vals, 0.10, 1e-3, 12)
    assert d2["cause"] == "sensitivity: value-dependent: substitution (own wrong intermediates)"
    # every term right, σ written ten times too small: a decimal shift in the copy
    rec3 = _rec(f"sigma_rel = sqrt((2*0.02)^2 + (1*0.01)^2) = {exact:.2g}", "S_u = v/s = 6/3 = 2\nS_t = 1",
                round(exact, 3) * 10)
    assert D.decompose(_problem(), ref, rec3, vals, 0.10, 1e-3, 12)["cause"] == "copy: decimal shift (×10^-1)"
    # a line with a missing term: chain selection; no line: format
    rec4 = _rec("sigma_rel = sqrt((2*0.02)^2) = 0.04", "S_u = 2", 4.0)
    assert D.decompose(_problem(), ref, rec4, vals, 0.10, 1e-3, 12)["cause"] == "chain selection: wrong number of terms"
    assert D.decompose(_problem(), ref, None, vals, 0.10, 1e-3, 12)["cause"] == "format: no parsable σ line"

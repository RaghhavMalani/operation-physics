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

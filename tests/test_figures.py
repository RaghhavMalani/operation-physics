"""The paper's headline figure (scripts/figures/headline.py) plots committed results only: its values are the
published FINDINGS numbers, and it renders. CPU only."""

import importlib.util

import pytest

from physcal.config import REPO_ROOT


def _headline():
    spec = importlib.util.spec_from_file_location("headline", REPO_ROOT / "scripts" / "figures" / "headline.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_headline_figure_plots_the_published_numbers():
    d = _headline().data()
    a = d["A"]  # FINDINGS #18
    assert a["split"] == pytest.approx([0.4175, 0.4125, 0.425])  # 41.8 / 41.3 / 42.5% (half up)
    assert a["uniform"] == pytest.approx([0.105, 0.115, 0.105])
    assert a["diff"] == pytest.approx(0.310, abs=5e-4) and a["sft_only"] == pytest.approx(0.12)
    b = {r["label"]: r for r in d["B"]["rows"]}  # FINDINGS #19, #21, #22, #26
    assert round(b["self-consistency, k = 8 samples"]["value"], 3) == 0.643
    assert round(b["hidden-state probe"]["value"], 3) == 0.547 and round(b["RL σ suffix (E1d)"]["value"], 3) == 0.512
    assert all(r["n_problems"] == 516 for r in b.values())
    assert d["B"]["cost_ratio"] == pytest.approx(8.0, abs=0.05)  # FINDINGS #37: 1,564 vs 196 tokens
    c = {p["name"]: p for p in d["C"]["points"]}  # FINDINGS #32, #35, #36
    assert c["RSS"]["slope"] == 0 and c["RSS"]["d_rss"] == 0
    assert round(c["D-pos-v2 (a)"]["slope"], 3) == 0.837 and round(c["B-v2"]["slope"], 3) == 0.416
    assert round(c["RSS fallback (post-hoc)"]["slope"], 3) == 0.136
    assert round(c["exact sensitivities (counterfactual)"]["slope"], 2) == 0.97
    assert round(1e4 * c["B-v2"]["d_rss"], 1) == -10.7 and round(1e4 * c["RSS fallback (post-hoc)"]["d_rss"], 1) == 15.9
    assert c["X1"]["d_rss_ci95"] is None  # no committed CI, none drawn


def test_the_headline_figure_renders(tmp_path):
    m = _headline()
    paths = m.draw(m.data(), tmp_path)
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)

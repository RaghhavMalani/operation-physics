"""D57 miss-count rule of the Phase 10 gate, on synthetic rows (not a gate run)."""

import numpy as np
from scipy.stats import norm

from physcal.config import load_config
from physcal.eval.sigma_star_gate import evaluate

CFG = load_config("sigma_star_gate_final")


def _rows(bad_cells: int) -> list[dict]:
    """Every cell has (near-)nominal coverage at ±1 and ±2; `bad_cells` cells are made 100% covered."""
    q = norm.ppf((np.arange(CFG["n_per_cell"]) + 0.5) / CFG["n_per_cell"])  # deterministic N(0, 1) quantiles
    rows, c = [], 0
    for d in CFG["depths"]:
        for lv in CFG["levels"]:
            dev = np.zeros_like(q) if c < bad_cells else q
            rows += [{"depth": d, "level": lv, "dev": float(x), "sigma": 1.0} for x in dev]
            c += 1
    return rows


def test_one_miss_in_15_cells_passes_two_fail():
    for bad, ok in ((0, True), (1, True), (2, False)):
        lv = evaluate(_rows(bad), CFG)["levels"]["2"]
        assert lv["n_cells_outside"] == bad
        assert lv["miss_count_ok"] is ok
    assert evaluate(_rows(1), CFG)["levels"]["2"]["share_ok"] is False  # the superseded rule failed on one miss

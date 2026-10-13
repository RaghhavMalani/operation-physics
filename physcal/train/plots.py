"""Training curves from train_metrics.jsonl (one row per GRPO step).

    uv run python -m physcal.train.plots results/train/<run_id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from physcal.eval.calibration import GRID, REFERENCE, SERIES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, _style

ROLLING = 10  # steps in the rolling mean


def _rolling(values: list[float | None], window: int) -> np.ndarray:
    v = np.array([np.nan if x is None else x for x in values], dtype=float)
    out = np.full_like(v, np.nan)
    for i in range(len(v)):
        w = v[max(0, i - window + 1): i + 1]
        if np.isfinite(w).any():
            out[i] = np.nanmean(w)
    return out


def plot_run(run_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(l) for l in (run_dir / "train_metrics.jsonl").read_text().splitlines() if l.strip()]
    meta = run_dir / "rollouts.meta.json"
    budget = json.loads(meta.read_text())["train_config"]["grpo"]["max_completion_length"] if meta.exists() else "?"
    steps = [r["step"] for r in rows]
    if len(set(steps)) < len(steps):  # older logs without a working step counter: fall back to row order
        steps = list(range(1, len(rows) + 1))
    panels = [
        ("Reward (crps_log + 0.2·steps)", [("reward_mean", SERIES[0], "mean reward")], None),
        ("Zero-variance groups", [("zero_variance_fraction", SERIES[0], "crps_log reward"),
                                  ("zero_variance_fraction_binary", SERIES[1], "binary (σ-free)")], (0, 1)),
        ("Median predicted relative σ", [("median_rel_sigma", SERIES[0], "median σ / |y|")], None),
        ("Interval coverage", [("coverage_1sigma", SERIES[0], "1σ (target 68%)"),
                               ("coverage_2sigma", SERIES[1], "2σ (target 95%)")], (0, 1)),
        ("Accuracy and format", [("accuracy_binary", SERIES[0], "correct ≤1% (σ-free)"),
                                 ("parse_fail_rate", SERIES[1], "parse failure")], (0, 1)),
        (f"Truncation at {budget} tokens", [("truncation_rate", SERIES[0], "truncated")], (0, 1)),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9.5), facecolor=SURFACE, sharex=True)
    for ax, (title, series, ylim) in zip(axes.flat, panels):
        _style(ax)
        for key, color, label in series:
            raw = [r.get(key) for r in rows]
            ax.plot(steps, [np.nan if x is None else x for x in raw], color=color, linewidth=1, alpha=0.25)
            ax.plot(steps, _rolling(raw, ROLLING), color=color, linewidth=2, label=label)
        if title == "Interval coverage":
            for target, color in ((0.68, SERIES[0]), (0.95, SERIES[1])):
                ax.axhline(target, color=color, linewidth=1, linestyle=(0, (4, 3)), alpha=0.7)
        if title.startswith("Median predicted"):
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
        if len(series) > 1:
            legend = ax.legend(frameon=False, fontsize=8, loc="best")
            for t in legend.get_texts():
                t.set_color(TEXT_PRIMARY)
    for ax in axes[-1]:
        ax.set_xlabel("GRPO step", color=TEXT_SECONDARY, fontsize=10)
    fig.suptitle(f"{run_dir.name}: thin = per step, thick = {ROLLING}-step rolling mean",
                 color=TEXT_PRIMARY, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    out = run_dir / "training_curve.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(plot_run(Path(sys.argv[1])))

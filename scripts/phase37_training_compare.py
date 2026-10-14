"""Phase 3.7: training curves of tonight's run (verifier 1.1.0) vs last night's control (1.0.0), per step.

    uv run python scripts/phase37_training_compare.py

Small multiples (one y-axis each): share of answers stating σ = 0 (from the rollout tapes; a log-σ axis can't
show σ = 0), 1σ coverage, σ-free accuracy, mean reward, each as a 20-step rolling mean. The rewards come from different verifier versions, but they differ only for σ
below 0.1% (by < 0.001 per completion), so the panel is still comparable. Writes training_vs_control.png and
training_vs_control.json into tonight's run dir.
"""

import json
from pathlib import Path

import numpy as np

from physcal.eval.calibration import GRID, REFERENCE, SERIES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, _style

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "last night (σ floor 0.1%)": ROOT / "results/train/overnight-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-512",
    "tonight (σ floor 1e-6)": ROOT / "results/train/overnight2-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512",
}
PANELS = [("sigma_zero_share", "share of answers stating σ = 0", False), ("coverage_1sigma", "1σ coverage", False),
          ("accuracy_binary", "accuracy ≤1% (σ-free)", False), ("reward_mean", "mean reward", False)]
WINDOW = 20
MATCHED_STEP = 287


def _sigma_zero_rows(path: Path) -> list[dict]:
    """Per step: share of parsed answers with a σ that state σ = 0 (needs the raw local rollouts.jsonl)."""
    zero, total = {}, {}
    for line in (path / "rollouts.jsonl").open():
        r = json.loads(line)
        b = r["breakdown"]
        if b["reason"] == "ok" and b.get("sigma_si") is not None:
            total[r["step"]] = total.get(r["step"], 0) + 1
            zero[r["step"]] = zero.get(r["step"], 0) + (b["sigma_si"] == 0)
    return [{"step": k, "sigma_zero_share": zero[k] / total[k]} for k in sorted(total)]


def _rows(path: Path, key: str) -> list[dict]:
    if key == "sigma_zero_share":
        return _sigma_zero_rows(path)
    return [json.loads(l) for l in (path / "train_metrics.jsonl").open()]


def series(path: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    rows = _rows(path, key)
    steps = np.array([r["step"] for r in rows], float)
    vals = np.array([np.nan if r.get(key) is None else r[key] for r in rows], float)
    out = np.array([np.nanmean(vals[max(0, i - WINDOW + 1): i + 1]) for i in range(len(vals))])
    return steps, out


def window_table() -> dict:
    """Per 50-step window and run: means of the per-step metrics (the numbers behind the figure)."""
    table = {}
    for name, path in RUNS.items():
        by_key = {k: _rows(path, k) for k, _, _ in PANELS}
        last = max(r["step"] for r in by_key["reward_mean"])
        out = {}
        for lo in range(0, last, 50):
            out[f"{lo + 1}-{lo + 50}"] = {
                k: float(np.nanmean([np.nan if r.get(k) is None else r[k] for r in rows if lo < r["step"] <= lo + 50]))
                for k, rows in by_key.items()}
        table[name] = out
    return table


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor=SURFACE)
    for ax, (key, title, log) in zip(axes.flat, PANELS):
        _style(ax)
        for color, (name, path) in zip(SERIES, RUNS.items()):
            x, y = series(path, key)
            ax.plot(x, y, color=color, linewidth=2, solid_capstyle="round", label=name)
            ax.plot(x[-1:], y[-1:], "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        if key == "coverage_1sigma":
            ax.axhline(0.6827, color=REFERENCE, linewidth=1)
            ax.text(2, 0.6827, "68% target", color=TEXT_SECONDARY, fontsize=8, va="bottom")
        ax.axvline(MATCHED_STEP, color=REFERENCE, linewidth=1)
        ax.text(MATCHED_STEP, ax.get_ylim()[0], f" step {MATCHED_STEP}", color=TEXT_SECONDARY, fontsize=8, va="bottom")
        if log:
            ax.set_yscale("log")
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=10, loc="left")
        ax.set_xlabel("GRPO step", color=TEXT_SECONDARY, fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY, loc="upper right", ncol=2)
    fig.suptitle(f"Tonight vs last night, {WINDOW}-step rolling means (training rollouts, T = 1.0)",
                 color=TEXT_PRIMARY, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = RUNS["tonight (σ floor 1e-6)"]
    fig.savefig(out / "training_vs_control.png", dpi=130, facecolor=SURFACE)
    (out / "training_vs_control.json").write_text(json.dumps(window_table(), indent=2))
    print(json.dumps(window_table(), indent=1))


if __name__ == "__main__":
    main()

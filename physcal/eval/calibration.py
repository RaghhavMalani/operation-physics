"""Calibration and accuracy metrics over reward breakdowns, plus PIT / reliability plots."""

from __future__ import annotations

import math

from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
from scipy import stats

from physcal.verify.scoring import RULES, floor

# Reference palette (dataviz skill, light mode).
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8f8e88"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
LABEL_GAP = 0.045  # minimum vertical gap between direct labels, in axis data units


def _mean(xs: Iterable[float]) -> float | None:
    xs = list(xs)
    return float(np.mean(xs)) if xs else None


def _median(xs: Iterable[float]) -> float | None:
    xs = list(xs)
    return float(median(xs)) if xs else None


def reliability_curve(breakdowns: list[dict], levels: list[float]) -> list[dict]:
    """Observed coverage of the central p-interval mu ± z_{(1+p)/2}·sigma, for each nominal p."""
    with_sigma = [b for b in breakdowns if b["reason"] == "ok" and b.get("sigma_si") is not None]
    out = []
    for p in levels:
        k = stats.norm.ppf(0.5 + p / 2.0)
        hits = [abs(b["mu_si"] - b["truth_si"]) <= k * b["sigma_si"] for b in with_sigma]
        out.append({"nominal": p, "observed": _mean(hits), "n": len(hits)})
    return out


def pit_histogram(breakdowns: list[dict], bins: int) -> dict:
    pits = [b["pit"] for b in breakdowns if b["reason"] == "ok" and b.get("pit") is not None]
    counts, edges = np.histogram(pits, bins=bins, range=(0.0, 1.0))
    return {"counts": counts.tolist(), "edges": edges.tolist(), "n": len(pits)}


def summarize(breakdowns: list[dict], scoring_cfg: dict, levels: list[float], pit_bins: int) -> dict[str, Any]:
    """Headline metrics. Accuracy and rule means count failures (as incorrect / as the rule's floor)."""
    n = len(breakdowns)
    reasons = Counter(b["reason"] for b in breakdowns)
    ok = [b for b in breakdowns if b["reason"] == "ok"]
    with_sigma = [b for b in ok if b.get("sigma_si") is not None]
    stepped = [b for b in breakdowns if b["n_steps_requested"] > 0]

    def rule_mean(rule: str) -> float | None:
        vals = []
        for b in breakdowns:
            s = (b.get("scores") or {}).get(rule)
            vals.append(floor(rule, scoring_cfg) if s is None else s)
        return _mean(vals)

    return {
        "n": n,
        "n_answered": len(ok),
        "n_with_sigma": len(with_sigma),
        "accuracy": (sum(b["correct"] for b in ok) / n) if n else None,
        "accuracy_among_answered": _mean(b["correct"] for b in ok),
        "median_rel_error": _median(b["abs_rel_error"] for b in ok),
        "parse_fail_rate": reasons["parse_fail"] / n if n else None,
        "unit_mismatch_rate": reasons["unit_mismatch"] / n if n else None,
        "missing_sigma_rate": (sum(1 for b in breakdowns if "no ± uncertainty" in b.get("detail", "")) / n) if n else None,
        "coverage_1sigma": _mean(b["within_1sigma"] for b in with_sigma),
        "coverage_2sigma": _mean(b["within_2sigma"] for b in with_sigma),
        "mean_rel_sigma": _mean(b["rel_sigma"] for b in with_sigma),
        "median_rel_sigma": _median(b["rel_sigma"] for b in with_sigma),
        "sigma_zero_rate": _mean(b["sigma_si"] == 0 for b in with_sigma),
        "mean_reward": _mean(b["reward"] for b in breakdowns),
        "mean_score_by_rule": {r: rule_mean(r) for r in RULES},
        "mean_step_score": _mean(b["score_steps"] for b in stepped),
        "copied_given_rate": _mean(bool(b.get("copied_given")) for b in ok),
        "steps_block_rate": _mean(bool(b.get("steps_block_found")) for b in breakdowns),
        "reliability": reliability_curve(breakdowns, levels),
        "pit_histogram": pit_histogram(breakdowns, pit_bins),
    }


def breakdown_by(records: list[dict], key: str) -> dict[str, dict]:
    """Accuracy and failure rates grouped by a problem field (family or depth)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[str(r["problem"][key])].append(r["breakdown"])
    out = {}
    for k in sorted(groups):
        bds = groups[k]
        ok = [b for b in bds if b["reason"] == "ok"]
        out[k] = {
            "n": len(bds),
            "accuracy": sum(b["correct"] for b in ok) / len(bds),
            "median_rel_error": _median(b["abs_rel_error"] for b in ok),
            "parse_fail_rate": sum(b["reason"] == "parse_fail" for b in bds) / len(bds),
            "unit_mismatch_rate": sum(b["reason"] == "unit_mismatch" for b in bds) / len(bds),
            "coverage_1sigma": _mean(b["within_1sigma"] for b in ok if b.get("within_1sigma") is not None),
        }
    return out


def noise_summary(records: list[dict], tau: float) -> dict[str, dict]:
    """Generator >= 2.0.0 (Phase 10; D50/A1): per noise level, the computation-correct rate (|ln μ − ln f(x_meas)| ≤
    tau), coverage of the clean truth, and on computation-correct answers the stated σ against σ* (log10 ratio of the
    relative σ to σ*, crps_log's scale) and coverage. Empty when no problem carries noise information."""
    groups: dict[float, list[dict]] = defaultdict(list)
    for r in records:
        n = r["problem"].get("noise")
        if n is not None:
            groups[n["level"]].append(r)
    out = {}
    for level in sorted(groups):
        rs = groups[level]
        ans = [r for r in rs if r["breakdown"]["reason"] == "ok" and r["breakdown"].get("sigma_si") is not None
               and r["breakdown"]["mu_si"] > 0]
        cc = [r for r in ans if abs(math.log(r["breakdown"]["mu_si"] / r["problem"]["noise"]["point_estimate_si"])) <= tau]
        ratio = [math.log10(max(r["breakdown"]["sigma_si"], 1e-300) / r["breakdown"]["mu_si"]
                            / r["problem"]["noise"]["sigma_star_log"])
                 for r in cc if r["problem"]["noise"]["sigma_star_log"] > 0]
        out[f"{level:g}"] = {
            "n": len(rs), "n_answered": len(ans), "computation_correct_rate": len(cc) / len(rs),
            "coverage_1sigma_answered": _mean(r["breakdown"]["within_1sigma"] for r in ans),
            "coverage_1sigma_computation_correct": _mean(r["breakdown"]["within_1sigma"] for r in cc),
            "median_log10_sigma_over_sigma_star": _median(ratio),
        }
    return out


def accuracy_at_tolerances(records: list[dict], tolerances: list[float]) -> dict[str, dict[str, float]]:
    """Fraction of *all* completions within each relative tolerance (failures count as wrong).

    strict: as scored by the main rule (± required by crps_log); sigma_free: as parsed by the binary rule.
    """
    n = len(records)
    out: dict[str, dict[str, float]] = {"strict": {}, "sigma_free": {}}
    for tol in tolerances:
        key = f"{tol:g}"
        out["strict"][key] = sum(
            r["breakdown"]["reason"] == "ok" and r["breakdown"]["abs_rel_error"] <= tol for r in records
        ) / n
        out["sigma_free"][key] = sum(
            r["by_rule"]["binary"]["abs_rel_error"] is not None and r["by_rule"]["binary"]["abs_rel_error"] <= tol
            for r in records
        ) / n
    return out


def group_stats(records: list[dict]) -> dict[str, Any]:
    """With several samples per problem: how often would a GRPO group carry no learning signal?

    A group = all samples of one problem. It has zero variance under a reward if every sample gets exactly the
    same reward (all correct / all wrong under binary; identical scores, e.g. all parse failures, under CRPS).
    Reported per scoring rule, for the final score alone and for the full reward (final + step credit).
    The binary rule doesn't need ±, so pass@k and the pass rate are σ-free.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        p = r["problem"]
        groups[(p["family"], p["depth"], p["seed"])].append(r["by_rule"])
    rules = sorted(next(iter(groups.values()))[0]) if groups else []

    def zero_var(values: list[float]) -> bool:
        return max(values) - min(values) == 0.0

    def describe(keys: list[tuple]) -> dict[str, Any]:
        out: dict[str, Any] = {"n_groups": len(keys)}
        out["zero_variance_fraction"] = {
            rule: _mean(zero_var([o[rule]["score_final"] for o in groups[k]]) for k in keys) for rule in rules
        }
        out["zero_variance_fraction_with_steps"] = {
            rule: _mean(zero_var([o[rule]["reward"] for o in groups[k]]) for k in keys) for rule in rules
        }
        correct = [[o["binary"]["score_final"] == 1.0 for o in groups[k]] for k in keys]
        out["pass_at_k"] = _mean(any(c) for c in correct)
        out["mean_pass_rate"] = _mean(sum(c) / len(c) for c in correct)
        return out

    keys = sorted(groups)
    by_depth = {str(d): describe([k for k in keys if k[1] == d]) for d in sorted({k[1] for k in keys})}
    return {"k": len(groups[keys[0]]) if keys else 0, **describe(keys), "by_depth": by_depth}


# ---------------------------------------------------------------------------------------------- plots
def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _empty(ax, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=TEXT_SECONDARY, fontsize=10, transform=ax.transAxes)


def plot_split(name: str, summary: dict, path: Path) -> None:
    """Two panels: PIT histogram (uniform = calibrated) and reliability (on the diagonal = calibrated)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_pit, ax_rel) = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    hist = summary["pit_histogram"]
    _style(ax_pit)
    if hist["n"]:
        edges = np.array(hist["edges"])
        width = edges[1] - edges[0]
        density = np.array(hist["counts"]) / hist["n"] / width
        ax_pit.bar(edges[:-1], density, width=width * 0.94, align="edge", color=SERIES[0], linewidth=0)
        ax_pit.axhline(1.0, color=REFERENCE, linewidth=1.5, linestyle=(0, (4, 3)))
        ax_pit.text(0.5, 1.0, "uniform = calibrated", color=TEXT_SECONDARY, fontsize=9, va="bottom", ha="center",
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    else:
        _empty(ax_pit, "no answers with ± uncertainty")
    ax_pit.set_xlim(0, 1)
    ax_pit.set_xlabel("PIT = Φ((y − μ)/σ)", color=TEXT_SECONDARY, fontsize=10)
    ax_pit.set_ylabel("density", color=TEXT_SECONDARY, fontsize=10)
    ax_pit.set_title(f"PIT histogram (n = {hist['n']})", color=TEXT_PRIMARY, fontsize=11, loc="left")

    _style(ax_rel)
    rel = [r for r in summary["reliability"] if r["observed"] is not None]
    ax_rel.plot([0, 1], [0, 1], color=REFERENCE, linewidth=1.5, linestyle=(0, (4, 3)))
    if rel:
        ax_rel.plot([r["nominal"] for r in rel], [r["observed"] for r in rel], color=SERIES[0], linewidth=2,
                    marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
    else:
        _empty(ax_rel, "no answers with ± uncertainty")
    ax_rel.set_xlim(0, 1)
    ax_rel.set_ylim(0, 1)
    ax_rel.set_xlabel("nominal central-interval coverage", color=TEXT_SECONDARY, fontsize=10)
    ax_rel.set_ylabel("observed coverage", color=TEXT_SECONDARY, fontsize=10)
    ax_rel.set_title("Reliability (diagonal = calibrated)", color=TEXT_PRIMARY, fontsize=11, loc="left")

    fig.suptitle(f"{name}: calibration of stated σ", color=TEXT_PRIMARY, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_reliability_all(summaries: dict[str, dict], path: Path) -> None:
    """Reliability curves of every split on one axis, direct-labelled."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 5.0), facecolor=SURFACE)
    _style(ax)
    ax.plot([0, 1], [0, 1], color=REFERENCE, linewidth=1.5, linestyle=(0, (4, 3)), label="calibrated")
    ends: list[tuple[float, float, str]] = []
    for color, (name, s) in zip(SERIES, summaries.items()):
        rel = [r for r in s["reliability"] if r["observed"] is not None]
        if not rel:
            continue
        xs, ys = [r["nominal"] for r in rel], [r["observed"] for r in rel]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{name} (n = {rel[0]['n']})")
        ends.append((ys[-1], xs[-1], name))
    # Direct labels at the line ends, nudged apart so they never overlap.
    placed: list[float] = []
    for y, x, name in sorted(ends):
        y_label = max([y] + [p + LABEL_GAP for p in placed])
        placed.append(y_label)
        ax.annotate(name, (x, y), xytext=(x + 0.015, y_label), textcoords="data",
                    color=TEXT_SECONDARY, fontsize=9, va="center")
    ax.set_xlim(0, 1.12)
    ax.set_ylim(0, 1)
    ax.set_xlabel("nominal central-interval coverage", color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel("observed coverage", color=TEXT_SECONDARY, fontsize=10)
    ax.set_title("Reliability by split", color=TEXT_PRIMARY, fontsize=12, loc="left")
    legend = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


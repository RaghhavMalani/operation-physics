"""The paper's three-panel headline figure (R5-1 framing; docs/paper_outline.md), from committed results only.

Nothing is estimated here. Every plotted number is read from a committed results file, and the only arithmetic is
X1's score difference against RSS (the difference of their committed expected scores; no CI for it is committed, so
none is drawn) and the cost ratio (committed tokens per problem). The figure's sidecar JSON lists every plotted value
with its source file and key, so each point can be traced.

  (A) Uniform vs split credit: 1σ coverage per seed, 3 vs 3 (FINDINGS #18; D71 robustness).
  (B) Within-problem pairwise accuracy (a right answer ranked above a wrong one to the same problem; X2's 516
      problems, 5,390 pairs) for every epistemic readout measured, against self-consistency, with the 8× token cost
      (FINDINGS #19, #21, #22, #26, #37). Left out: X2's problem-text-only check (0.5 by construction) and X4's
      "μ × 1.25" arm (a diagnostic of what the check reads, not a readout).
  (C) The frozen R4-5 set's A6c propagation slope against the expected crps_log difference from RSS, for RSS, X1,
      B-v2, D-pos-v2 (a), the RSS fallback (post-hoc, #36) and the exact-sensitivity swap (a counterfactual, #35;
      not a method). Seed means over 3 seeds where the source has seeds (FINDINGS #32, #35, #36).

    uv run python scripts/figures/headline.py [--out results/figures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = {
    "robust3v3": "results/corrected/robustness_3v3/report.json",
    "x2": "results/x2/results.json",
    "x2r": "results/x2/robustness/results.json",
    "e1d": "results/e1d_x2/results.json",
    "cost": "results/r5_diag/cost.json",
    "sb_dpos2": "results/stage_r4/confirmatory/stage_b_dpos2/results.json",
    "sb_b2": "results/stage_r4/confirmatory/stage_b_b2/results.json",
    "tail": "results/r5_diag/tail.json",
    "fallback": "results/r5_diag/fallback_confirmatory.json",
}
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"   # categorical slots 1-2 of the dataviz reference palette


def _load() -> dict:
    return {k: json.loads((ROOT / p).read_text()) for k, p in SRC.items()}


def panel_a(J: dict) -> dict:
    r = J["robust3v3"]
    per = r["B_seed_level"]["per_run"]
    s1b, s1a = r["A_problem_level_seed_mean"]["S1b"], r["A_problem_level_seed_mean"]["S1a"]
    return {"split": [v["cov1"] for v in per["split"].values()], "uniform": [v["cov1"] for v in per["uniform"].values()],
            "split_mean": s1b["split"], "uniform_mean": s1b["control"], "diff": s1b["diff"], "diff_ci95": s1b["diff_ci"],
            "sft_only": s1a["control"], "nominal": 0.68,
            "source": SRC["robust3v3"] + ": B_seed_level.per_run.*.cov1; A_problem_level_seed_mean.S1b / S1a"}


B_ROWS = [  # (label, source key, method key, group)
    ("SFT stated σ", "x2r", "sft_sigma", "single answer"),
    ("split-credit RL stated σ", "x2r", "split_sigma", "single answer"),
    ("SFT greedy σ suffix", "e1d", "sft_suffix", "single answer"),
    ("supervised σ suffix (X3)", "e1d", "x3_a", "single answer"),
    ("RL σ suffix (E1d)", "e1d", "e1d", "single answer"),
    ("text model p (A5)", "x2r", "a5_p", "single answer"),
    ("text model σ (A5)", "x2r", "a5_sigma", "single answer"),
    ("answer length", "x2r", "length", "single answer"),
    ("hidden-state probe", "x2r", "probe", "single answer"),
    ("hidden-state probe, cross-fitted", "x2r", "probe_crossfit", "single answer"),
    ("self-verification P(yes)", "x2r", "x4_sft", "single answer"),
    ("self-verification, no recompute", "x2r", "x4_direct", "single answer"),
    ("control: probe on random-init model", "x2r", "random_init_probe", "control"),
    ("control: self-verification, random init", "x2r", "x4_random_init", "control"),
    ("control: self-verification, other problem", "x2r", "x4_shuffled", "control"),
    ("self-consistency, k = 8 samples", "x2r", "self_consistency", "several samples"),
]


def panel_b(J: dict) -> dict:
    rows = []
    for label, src, key, group in B_ROWS:
        m = J[src]["methods"][key]["pairwise_accuracy"]
        rows.append({"label": label, "group": group, "value": m["pairwise_accuracy"], "ci95": m["ci95"],
                     "n_problems": m["n_problems"], "source": f"{SRC[src]}: methods.{key}.pairwise_accuracy"})
    pp = J["cost"]["x2"]["per_problem"]
    return {"rows": rows, "tokens_k": pp["tokens_k"], "tokens_one": pp["tokens_one"],
            "cost_ratio": pp["tokens_k"] / pp["tokens_one"], "cost_source": SRC["cost"] + ": x2.per_problem"}


def panel_c(J: dict) -> dict:
    cd, cb, t, fb = J["sb_dpos2"], J["sb_b2"], J["tail"], J["fallback"]
    comp = cd["comparators"]
    if comp != cb["comparators"]:
        raise ValueError("the two stage B results disagree on the comparators")
    rss_e, x1_e = comp["rss"]["oracle_centred"]["expected"][0], comp["x1"]["oracle_centred"]["expected"][0]
    pt = lambda name, group, sx, sx_ci, dy, dy_ci, src, note="": {
        "name": name, "group": group, "slope": sx, "slope_ci": sx_ci, "d_rss": dy, "d_rss_ci95": dy_ci, "source": src,
        "note": note}
    return {"points": [
        pt("RSS", "heuristic", comp["rss"]["a6c_all_pairs"]["slope"], comp["rss"]["a6c_all_pairs"]["ci"], 0.0, None,
           SRC["sb_dpos2"] + ": comparators.rss", "the reference (Δ = 0); slope 0 by A6c's construction"),
        pt("X1", "heuristic", comp["x1"]["a6c_all_pairs"]["slope"], comp["x1"]["a6c_all_pairs"]["ci"], x1_e - rss_e, None,
           SRC["sb_dpos2"] + ": comparators.x1 (slope CI 95%); Δ = oracle_centred.expected(x1) − (rss)",
           "no bootstrap CI for X1 − RSS is committed"),
        pt("B-v2", "trained", cb["tests"]["B2"]["value"], cb["tests"]["B2"]["ci90"], cb["tests"]["B3"]["value"],
           cb["tests"]["B3"]["ci95"], SRC["sb_b2"] + ": tests.B2 (K4), tests.B3 (K8)",
           "slope on answered pairs (509 / 510 / 510)"),
        pt("D-pos-v2 (a)", "trained", t["swaps"]["a"]["a6c_slope"]["value"], t["swaps"]["a"]["a6c_slope"]["ci90"],
           t["swaps"]["a"]["minus_rss"]["value"], t["swaps"]["a"]["minus_rss"]["ci95"], SRC["tail"] + ": swaps.a",
           "all 510 pairs (unanswered members at the floor); K1 on answered pairs is +0.816"),
        pt("RSS fallback (post-hoc)", "post-hoc", fb["a6c_slope"]["value"], fb["a6c_slope"]["ci90"],
           fb["minus_rss"]["value"], fb["minus_rss"]["ci95"], SRC["fallback"] + ": a6c_slope, minus_rss"),
        pt("exact sensitivities (counterfactual)", "post-hoc", t["swaps"]["S"]["a6c_slope"]["value"],
           t["swaps"]["S"]["a6c_slope"]["ci90"], t["swaps"]["S"]["minus_rss"]["value"], t["swaps"]["S"]["minus_rss"]["ci95"],
           SRC["tail"] + ": swaps.S"),
    ], "sigma_star_gap": t["rss_to_star_gap"], "gap_source": SRC["tail"] + ": rss_to_star_gap"}


def data() -> dict:
    J = _load()
    return {"A": panel_a(J), "B": panel_b(J), "C": panel_c(J), "sources": SRC}


def draw(d: dict, out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(14.5, 4.9))
    gs = fig.add_gridspec(1, 5, width_ratios=[0.62, 0.62, 1.15, 0.2, 1.15], wspace=0.05)  # columns 1, 3: gutters
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in (0, 2, 4))

    # (A) coverage, 3 vs 3 seeds
    a = d["A"]
    for x, key, mean, color in ((0, "uniform", a["uniform_mean"], ORANGE), (1, "split", a["split_mean"], BLUE)):
        vals = a[key]
        ax_a.scatter([x - 0.09, x, x + 0.09], [100 * v for v in vals], s=26, color=color, zorder=3,
                     edgecolor="white", linewidth=0.8)
        ax_a.hlines(100 * mean, x - 0.22, x + 0.22, color=color, linewidth=2, zorder=2)
    ax_a.axhline(100 * a["nominal"], color=MUTED, linestyle="--", linewidth=1)
    ax_a.text(1.45, 100 * a["nominal"] + 1, "nominal 68%", color=MUTED, ha="right", va="bottom", fontsize=7)
    ax_a.axhline(100 * a["sft_only"], color=MUTED, linestyle=":", linewidth=1)
    ax_a.text(1.45, 100 * a["sft_only"] + 1, f"SFT only {100 * a['sft_only']:.0f}%", color=MUTED, ha="right",
              va="bottom", fontsize=7)
    lo, hi = (round(100 * v + 1e-9, 1) for v in a["diff_ci95"])  # half up, as FINDINGS #18 prints it
    ax_a.text(0.5, 100 * a["split_mean"] + 9, f"Δ {100 * a['diff']:+.1f} pts\n95% CI [{lo:+.1f}, {hi:+.1f}]",
              ha="center", fontsize=7, color=INK)
    ax_a.set_xticks([0, 1], ["uniform\ncredit", "split\ncredit"])
    ax_a.set_xlim(-0.5, 1.5)
    ax_a.set_ylim(0, 80)
    ax_a.set_ylabel("1σ interval coverage (%)")
    ax_a.set_title("(A) σ credit: coverage, 3 vs 3 seeds", loc="left", fontsize=9)
    ax_a.grid(axis="y", color=GRID, linewidth=0.6)
    ax_a.set_axisbelow(True)

    # (B) within-problem pairwise accuracy
    b = d["B"]
    rows = b["rows"]
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        color = BLUE if r["group"] == "several samples" else (MUTED if r["group"] == "control" else INK)
        lo, hi = r["ci95"]
        ax_b.hlines(y, lo, hi, color=color, linewidth=1.4)
        ax_b.scatter([r["value"]], [y], s=22, color=color, zorder=3, edgecolor="white", linewidth=0.6)
        ax_b.text(hi + 0.004, y, f"{r['value']:.3f}", va="center", fontsize=6.5, color=MUTED)
    ax_b.axvline(0.5, color=MUTED, linestyle="--", linewidth=1)
    ax_b.set_yticks(ys, [r["label"] for r in rows], fontsize=7)
    for tick, r in zip(ax_b.get_yticklabels(), rows):
        tick.set_color(MUTED if r["group"] == "control" else INK)
    sc = rows[-1]
    ax_b.annotate(f"{b['cost_ratio']:.1f}× the tokens of one answer\n({b['tokens_k']:,.0f} vs {b['tokens_one']:,.0f} per problem)",
                  xy=(sc["value"], ys[-1]), xytext=(0.66, ys[-1] + 2.3), fontsize=6.5, color=INK,
                  ha="center", arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6})
    ax_b.set_xlim(0.44, 0.73)
    ax_b.set_xlabel("within-problem pairwise accuracy, 95% CI (0.5 = chance)")
    ax_b.set_title("(B) Which answer to the same problem is right?", loc="left", fontsize=9)
    ax_b.grid(axis="x", color=GRID, linewidth=0.6)
    ax_b.set_axisbelow(True)

    # (C) propagation vs score, frozen set
    c = d["C"]
    style = {"heuristic": (MUTED, "o", True), "trained": (BLUE, "o", True), "post-hoc": (ORANGE, "D", False)}
    offs = {"RSS": (6, -10), "X1": (6, 5), "B-v2": (6, -11), "D-pos-v2 (a)": (-8, 8),
            "RSS fallback (post-hoc)": (8, -14), "exact sensitivities (counterfactual)": (-10, 34)}
    for p in c["points"]:
        color, marker, filled = style[p["group"]]
        x, y = p["slope"], 1e3 * p["d_rss"]
        xe = [[x - p["slope_ci"][0]], [p["slope_ci"][1] - x]]
        ye = None if p["d_rss_ci95"] is None else [[y - 1e3 * p["d_rss_ci95"][0]], [1e3 * p["d_rss_ci95"][1] - y]]
        ax_c.errorbar([x], [y], xerr=xe, yerr=ye, fmt="none", ecolor=color, elinewidth=1, alpha=0.8)
        ax_c.scatter([x], [y], s=34, marker=marker, color=color if filled else "white", edgecolor=color, linewidth=1.3,
                     zorder=3)
        dx, dy = offs[p["name"]]
        ax_c.annotate(p["name"], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7, color=INK,
                      ha="right" if dx < 0 else "left",
                      arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6} if abs(dy) > 20 else None)
    ax_c.axhline(0, color=MUTED, linewidth=0.8)
    ax_c.axhline(1e3 * c["sigma_star_gap"], color=MUTED, linestyle=":", linewidth=1)
    ax_c.text(-0.08, 1e3 * c["sigma_star_gap"] + 0.06, "σ* (oracle)", color=MUTED, fontsize=7, va="bottom")
    ax_c.axvline(1, color=MUTED, linestyle="--", linewidth=1)
    ax_c.text(0.985, 3.3, "ideal slope 1", color=MUTED, fontsize=7, ha="right")
    ax_c.set_xlim(-0.1, 1.15)
    ax_c.set_ylim(-2.4, 3.5)
    ax_c.set_xlabel("A6c propagation slope (90% CI; 0 = no propagation)")
    ax_c.set_ylabel("expected crps_log − RSS (×10⁻³, 95% CI)")
    ax_c.set_title("(C) Propagation vs forecast score (frozen set)", loc="left", fontsize=9)
    ax_c.grid(color=GRID, linewidth=0.6)
    ax_c.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker=m, linestyle="none", color=col, markerfacecolor=col if f else "white",
                          label=lab) for lab, (col, m, f) in
               (("heuristic (no model)", style["heuristic"]), ("trained stated σ", style["trained"]),
                ("post-hoc / counterfactual", style["post-hoc"]))]
    ax_c.legend(handles=handles, loc="lower left", fontsize=6.5, frameon=False)

    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "headline.png", out / "headline.pdf"]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    (out / "headline_data.json").write_text(json.dumps(d, indent=1))
    return paths + [out / "headline_data.json"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    for p in draw(data(), ROOT / args.out):
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()

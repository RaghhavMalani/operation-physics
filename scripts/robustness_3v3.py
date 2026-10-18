"""R2-4 / R2-6 (DECISIONS.md D71): split credit vs uniform credit at step 287, 3 seeds each, on the corrected evals.

    uv run python scripts/robustness_3v3.py

A ROBUSTNESS analysis, reported next to the pre-registered family S verdicts (FINDINGS #16: three split-credit seeds
against the single uniform-credit seed). It does not replace them.
(A) Problem level: D62's seed-mean tests with the uniform arm also averaged over its seeds. S1a (vs SFT-only, which
    has no seed) is unchanged. One-sided paired bootstrap over problems (B = 10,000, seed 0; one index set per
    resample for every arm), Holm over {S1a, S1b, S2, S3} at α = 0.05. "Conditional on these 3 + 3 seeds".
(B) Seed level, descriptive: per-run metrics; Δ = mean(split) − mean(uniform); an exact permutation test over every
    relabelling of the runs into the two arm sizes (3 + 3: 20, one-sided p >= 0.05, two-sided >= 0.10); Hedges' g;
    the cross-seed differences split_i − uniform_j. No verdict comes from (B) (R2-4).
(C) Matched-step trajectories from the training logs (steps 1–287; the training σ is the unmerged 4-bit policy's
    own): per 25-step window, median relative σ (tape), σ = 0 share, mean reward, binary pass rate, zero-variance
    group fraction (reward and binary). A table and one figure.
A uniform seed without a step-287 corrected eval is left out and the report says so (D71: 3 vs 2 if one is missing).
Writes results/corrected/robustness_3v3/{report.json, report.md, trajectories.png}.
"""

from __future__ import annotations

import gzip
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase95_compare import CORRECTED, MARGIN, N_BOOT, ROOT, SEED, holm, load, rho  # noqa: E402

T = ROOT / "results/train"
SPLIT = [T / f"split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed{s}" for s in (0, 1, 2)]
UNIFORM = [T / "overnight2-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512"] + \
          [T / f"overnight2-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed{s}" for s in (1, 2)]
EVAL = "compare/rl287_4bit/rollouts.jsonl"
SFT = UNIFORM[0] / "compare/sft_4bit/rollouts.jsonl"
OUT = CORRECTED / "robustness_3v3"
STEP, WINDOW = 287, 25
TESTS = ("S1a", "S1b", "S2", "S3")
# (B): the direction of each hypothesis (None: two-sided only)
METRICS = {"cov1": "greater", "rho": "greater", "acc": "greater", "sigma_zero": None, "median_rel_sigma": None}


def seed_of(name: str) -> int:
    """The seed in a run name ("...-seed2"); the uniform-credit seed-0 control's name has no suffix."""
    return int(name.rsplit("-seed", 1)[1]) if "-seed" in name else 0


def cov(d: dict, ks: list) -> float:
    return float(np.mean([d[k]["cov1"] for k in ks]))


def acc(d: dict, ks: list) -> float:
    return float(np.mean([d[k]["correct"] for k in ks]))


def seed_mean_tests(split: list[dict], uni: list[dict], sft: dict, keys: list, n_boot: int = N_BOOT,
                    seed: int = SEED) -> dict:
    """(A): D62's statistics, with the uniform arm a seed mean as well."""
    mean_over = lambda arms, stat: (lambda ks: float(np.mean([stat(d, ks) for d in arms])))
    tests = {"S1a": (mean_over(split, cov), lambda ks: cov(sft, ks), 0.0),
             "S1b": (mean_over(split, cov), mean_over(uni, cov), 0.0),
             "S2": (mean_over(split, rho), mean_over(uni, rho), 0.0),
             "S3": (mean_over(split, acc), mean_over(uni, acc), -MARGIN)}
    rng = np.random.default_rng(seed)
    karr = np.array(keys, dtype=object)
    resamples = [[tuple(k) for k in karr[rng.integers(0, len(keys), len(keys))]] for _ in range(n_boot)]
    out = {}
    for name, (a, b, thr) in tests.items():
        d = np.array([a(ks) - b(ks) for ks in resamples])
        out[name] = {"split": a(keys), "control": b(keys), "diff": a(keys) - b(keys),
                     "diff_ci": np.percentile(d, [2.5, 97.5]).tolist(), "p_threshold": thr,
                     "p_one_sided": float(np.mean(d <= thr))}
    out["holm"] = holm({k: v["p_one_sided"] for k, v in out.items()})
    return out


def run_metrics(d: dict, keys: list) -> dict:
    sig = [d[k]["rel_sigma"] for k in keys if d[k]["rel_sigma"] is not None]
    return {"cov1": cov(d, keys), "rho": rho(d, keys), "acc": acc(d, keys),
            "sigma_zero": float(np.mean([s == 0 for s in sig])), "median_rel_sigma": float(np.median(sig))}


def permutation(a: list[float], b: list[float], direction: str | None) -> dict:
    """Exact permutation test of Δ = mean(a) − mean(b) over every relabelling into groups of len(a) and len(b)."""
    v, n = np.array(a + b, dtype=float), len(a)
    obs = float(np.mean(a) - np.mean(b))
    perm = np.array([v[list(c)].mean() - np.delete(v, list(c)).mean()
                     for c in itertools.combinations(range(len(v)), n)])
    eps = 1e-12
    out = {"delta": obs, "n_relabellings": len(perm),
           "p_two_sided": float(np.mean(np.abs(perm) >= abs(obs) - eps)),
           "min_p_two_sided": float(np.mean(np.abs(perm) >= np.abs(perm).max() - eps))}
    if direction == "greater":
        out["p_one_sided"] = float(np.mean(perm >= obs - eps))
        out["min_p_one_sided"] = 1 / len(perm)
    na, nb = len(a), len(b)
    s2 = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2) if na > 1 and nb > 1 else 0.0
    out["hedges_g"] = (1 - 3 / (4 * (na + nb) - 9)) * obs / math.sqrt(s2) if s2 > 0 else None
    cross = [x - y for x in a for y in b]
    out["cross_differences"] = cross
    if direction == "greater":
        out["share_cross_positive"] = float(np.mean([c > 0 for c in cross]))
    return out


def _tape_lines(run: Path):
    raw, gz = run / "rollouts.jsonl", run / "rollouts.jsonl.gz"
    if raw.exists():
        yield from raw.open()
    elif gz.exists():
        with gzip.open(gz, "rt") as f:
            yield from f


def trajectory(run: Path, step: int | None = None, window: int | None = None) -> list[dict]:
    """(C): per window of training steps, from train_metrics.jsonl and the tape (σ and σ = 0)."""
    step, window = step or STEP, window or WINDOW
    tm = [json.loads(line) for line in (run / "train_metrics.jsonl").open()]
    tm = [r for r in tm if r["step"] <= step]
    sig: dict[int, list[float]] = {}
    for line in _tape_lines(run):
        r = json.loads(line)
        if r["step"] <= step and r["breakdown"]["reason"] == "ok" and r["breakdown"].get("rel_sigma") is not None:
            sig.setdefault(r["step"], []).append(r["breakdown"]["rel_sigma"])
    out = []
    for lo in range(1, step + 1, window):
        hi = min(lo + window - 1, step)
        rows = [r for r in tm if lo <= r["step"] <= hi]
        s = [x for st in range(lo, hi + 1) for x in sig.get(st, [])]
        out.append({"steps": [lo, hi], "n_steps": len(rows),
                    "median_rel_sigma": float(np.median(s)) if s else None,
                    "sigma_zero_share": float(np.mean([x == 0 for x in s])) if s else None,  # tape: older logs lack it
                    "reward_mean": float(np.mean([r["reward_mean"] for r in rows])),
                    "pass_rate_binary": float(np.mean([r["accuracy_binary"] for r in rows])),
                    "zero_variance_fraction": float(np.mean([r["zero_variance_fraction"] for r in rows])),
                    "zero_variance_fraction_binary": float(np.mean([r["zero_variance_fraction_binary"] for r in rows]))})
    return out


def figure(traj: dict, path: Path) -> None:
    """2 x 2 small multiples; split credit blue, uniform credit orange (palette slots 1 and 2), seeds by line style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#1a1a19", "#6b6a64", "#e6e5e0", "#fcfcfb"
    color = {"split": "#2a78d6", "uniform": "#eb6834"}
    styles = ["-", "--", ":"]
    panels = [("median_rel_sigma", "median relative σ (training samples)", True),
              ("sigma_zero_share", "σ = 0 share", False), ("reward_mean", "mean reward", False),
              ("pass_rate_binary", "binary pass rate (≤ 1%)", False)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), facecolor=surface)
    for ax, (key, title, log) in zip(axes.flat, panels):
        ax.set_facecolor(surface)
        for arm in ("split", "uniform"):
            for i, (name, rows) in enumerate(traj[arm].items()):
                x = [np.mean(r["steps"]) for r in rows if r[key] is not None]
                y = [r[key] for r in rows if r[key] is not None]
                ax.plot(x, y, styles[i % 3], color=color[arm], lw=2,
                        label=f"{arm} credit, seed {seed_of(name)}")
        if log:
            ax.set_yscale("log")
        ax.set_title(title, color=ink, fontsize=10, loc="left")
        ax.grid(True, color=grid, lw=0.8)
        ax.tick_params(colors=muted, labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("training step (25-step windows)", color=muted, fontsize=8)
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=ink)
    fig.suptitle("Split vs uniform credit, matched steps 1–287 (D71 (C), descriptive)", color=ink, fontsize=11, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def main() -> dict:
    uni_runs = [r for r in UNIFORM if (r / EVAL).exists()]
    missing = [str(r.relative_to(ROOT)) for r in UNIFORM if r not in uni_runs]
    if len(uni_runs) < 2:
        raise SystemExit(f"need at least 2 uniform-credit seeds with a corrected step-287 eval; missing {missing}")
    split = [load(r / EVAL) for r in SPLIT]
    uni = [load(r / EVAL) for r in uni_runs]
    sft = load(SFT)
    keys = sorted(split[0])
    assert all(set(keys) == set(d) for d in split + uni + [sft]), "different eval problems"
    a = seed_mean_tests(split, uni, sft, keys)
    per_run = {"split": {r.name: run_metrics(d, keys) for r, d in zip(SPLIT, split)},
               "uniform": {r.name: run_metrics(d, keys) for r, d in zip(uni_runs, uni)}}
    b = {m: permutation([v[m] for v in per_run["split"].values()], [v[m] for v in per_run["uniform"].values()], direc)
         for m, direc in METRICS.items()}
    traj = {"split": {r.name: trajectory(r) for r in SPLIT}, "uniform": {r.name: trajectory(r) for r in uni_runs}}
    sanity = {r.name: (json.loads((r / "sanity.json").read_text()).get("sane") if (r / "sanity.json").exists() else None)
              for r in uni_runs[1:]}
    out = {"decision": "D71 (R2-4 / R2-6): robustness, not the pre-registered verdict",
           "split_runs": [str(r.relative_to(ROOT)) for r in SPLIT], "uniform_runs": [str(r.relative_to(ROOT)) for r in uni_runs],
           "uniform_missing": missing, "uniform_sanity": sanity, "eval": EVAL,
           "A_problem_level_seed_mean": a, "B_seed_level": {"per_run": per_run, "tests": b}, "C_trajectories": traj}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(out, indent=1, default=float))
    figure(traj, OUT / "trajectories.png")
    (OUT / "report.md").write_text(render(out))
    print(render(out))
    return out


def render(o: dict) -> str:
    a, b = o["A_problem_level_seed_mean"], o["B_seed_level"]
    nu = len(o["uniform_runs"])
    lines = [f"# Split vs uniform credit, 3 vs {nu} seeds: robustness analysis (D71, R2-4 / R2-6)", "",
             "Not the pre-registered verdict: FINDINGS #16's family S tests (3 split-credit seeds vs 1 uniform seed) "
             "stand. Problem-level p-values are conditional on these seeds; they are not seed-level significance.", ""]
    if o["uniform_missing"]:
        lines += [f"**Missing uniform-credit runs (no corrected step-287 eval):** {', '.join(o['uniform_missing'])}", ""]
    lines += ["## (A) Problem level, both arms as seed means", "",
              "| test | split (seed mean) | control | difference [95% CI] | p (one-sided) | Holm threshold | supported |",
              "|---|---|---|---|---|---|---|"]
    ctrl = {"S1a": "SFT-only", "S1b": f"uniform ({nu}-seed mean)", "S2": f"uniform ({nu}-seed mean)",
            "S3": f"uniform ({nu}-seed mean), margin −5 pts"}
    for k in TESTS:
        v, h = a[k], a["holm"][k]
        lines.append(f"| {k} (vs {ctrl[k]}) | {v['split']:.4f} | {v['control']:.4f} | {v['diff']:+.4f} "
                     f"[{v['diff_ci'][0]:+.4f}, {v['diff_ci'][1]:+.4f}] | {h['p']:.3g} | {h['holm_threshold']:.3g} "
                     f"| {bool(h['supported'])} |")
    lines += ["", "## (B) Seed level (descriptive; no verdict)", "",
              "| metric | split runs | uniform runs | Δ (means) | permutation p one-sided / two-sided (min possible) "
              "| Hedges' g | cross-seed differences with the hypothesis's sign |", "|---|---|---|---|---|---|---|"]
    for m, t in b["tests"].items():
        sv = " / ".join(f"{v[m]:.4g}" for v in b["per_run"]["split"].values())
        uv = " / ".join(f"{v[m]:.4g}" for v in b["per_run"]["uniform"].values())
        p1 = f"{t['p_one_sided']:.3g}" if "p_one_sided" in t else "–"
        mins = f"{t.get('min_p_one_sided', float('nan')):.3g} / {t['min_p_two_sided']:.3g}"
        g = "–" if t["hedges_g"] is None else f"{t['hedges_g']:+.2f}"
        sh = f"{t['share_cross_positive']:.0%} of {len(t['cross_differences'])}" if "share_cross_positive" in t else "–"
        lines.append(f"| {m} | {sv} | {uv} | {t['delta']:+.4g} | {p1} / {t['p_two_sided']:.3g} ({mins}) | {g} | {sh} |")
    lines += ["", "## (C) Matched-step trajectories (training logs, steps 1–287)", "",
              "| arm | run | window | median rel. σ | σ = 0 share | mean reward | pass rate | zero-var (reward) "
              "| zero-var (binary) |", "|---|---|---|---|---|---|---|---|---|"]
    for arm, runs in o["C_trajectories"].items():
        for name, rows in runs.items():
            for r in rows:
                ms = "–" if r["median_rel_sigma"] is None else f"{r['median_rel_sigma']:.4%}"
                zs = "–" if r["sigma_zero_share"] is None else f"{r['sigma_zero_share']:.1%}"
                lines.append(f"| {arm} | seed {seed_of(name)} "
                             f"| {r['steps'][0]}–{r['steps'][1]} | {ms} | {zs} "
                             f"| {r['reward_mean']:+.3f} | {r['pass_rate_binary']:.1%} | {r['zero_variance_fraction']:.1%} "
                             f"| {r['zero_variance_fraction_binary']:.1%} |")
    lines += ["", "Figure: `trajectories.png`."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

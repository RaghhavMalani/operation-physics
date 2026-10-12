"""Markdown report for the phase-3.6 overnight run: training windows, curriculum, 3-way eval, paired tests.

    uv run python scripts/summarize_overnight.py results/train/<run_id>
Writes <run>/report.md and prints it.
"""

import json
import statistics as st
import sys
from pathlib import Path

PARTS = (("base", "raw base"), ("sft", "SFT only"), ("rl", "SFT + RL"))


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def training_table(run: Path, window: int = 50) -> list[str]:
    rows = [json.loads(l) for l in (run / "train_metrics.jsonl").read_text().splitlines() if l.strip()]
    out = ["| steps | mean reward | zero-var (crps_log) | zero-var (binary) | accuracy ≤1% | median rel. σ | 1σ cov | parse fail | tokens |",
           "|---|---|---|---|---|---|---|---|---|"]
    for a in range(0, len(rows), window):
        w = rows[a:a + window]
        sig = [r["median_rel_sigma"] for r in w if r["median_rel_sigma"] is not None]
        cov = [r["coverage_1sigma"] for r in w if r["coverage_1sigma"] is not None]
        out.append(f"| {a + 1}–{a + len(w)} | {st.mean(r['reward_mean'] for r in w):+.3f} "
                   f"| {pct(st.mean(r['zero_variance_fraction'] for r in w))} "
                   f"| {pct(st.mean(r['zero_variance_fraction_binary'] for r in w))} "
                   f"| {pct(st.mean(r['accuracy_binary'] for r in w))} | {st.median(sig) * 100:.3f}% "
                   f"| {pct(st.mean(cov) if cov else None)} | {pct(st.mean(r['parse_fail_rate'] for r in w))} "
                   f"| {st.mean(r['mean_completion_tokens'] for r in w):.0f} |")
    return out


def curriculum_lines(run: Path) -> list[str]:
    states = [json.loads(l) for l in (run / "curriculum.jsonl").read_text().splitlines() if l.strip()]
    last = states[-1]
    cells = {}
    for s in states:
        for c in s["cells"]:
            cells[c] = cells.get(c, 0) + 1
    return [f"- Unlock log (observation #, cell): {last['unlock_log']}",
            f"- Final unlocked cells: {', '.join(last['unlocked'])}",
            "- Final EMA pass rate per cell: " + ", ".join(f"{k} {v:.2f}" for k, v in last["p"].items() if last["n"][k]),
            "- Groups trained per cell: " + ", ".join(f"{k} {v}" for k, v in sorted(cells.items()))]


def eval_table(run: Path) -> list[str]:
    m = {p: json.loads((run / "compare" / p / "metrics.json").read_text()) for p, _ in PARTS
         if (run / "compare" / p / "metrics.json").exists()}
    out = ["| split | model | acc ≤1% (σ-free) | ≤2% | ≤5% | ≤10% | strict ≤1% | parse fail | 1σ cov | 2σ cov | median rel. σ | mean reward (crps_log) | tokens |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for split in ("id", "depth_extrapolation"):
        for p, name in PARTS:
            if p not in m:
                continue
            s = m[p]["splits"][split]
            a = s["accuracy_at_tolerance"]["sigma_free"]
            sigma = "–" if s["median_rel_sigma"] is None else f"{100 * s['median_rel_sigma']:.3f}%"
            out.append(f"| {split} | {name} | {pct(a['0.01'])} | {pct(a['0.02'])} | {pct(a['0.05'])} | {pct(a['0.1'])} "
                       f"| {pct(s['accuracy_at_tolerance']['strict']['0.01'])} | {pct(s['parse_fail_rate'])} "
                       f"| {pct(s['coverage_1sigma'])} | {pct(s['coverage_2sigma'])} | {sigma} "
                       f"| {s['mean_reward']:+.3f} | {s['mean_completion_tokens']:.0f} |")
    return out


def tests_table(run: Path) -> list[str]:
    path = run / "compare" / "paired_tests.json"
    if not path.exists():
        return ["(paired tests missing)"]
    t = json.loads(path.read_text())
    out = ["| comparison | split | strict ≤1%: before → after (only-after / only-before), McNemar p | σ-free ≤1%, McNemar p | parsed, McNemar p | reward Δ, Wilcoxon p |",
           "|---|---|---|---|---|---|"]
    for name, res in t.items():
        if res == "missing":
            out.append(f"| {name} | – | missing | | | |")
            continue
        for split, r in res.items():
            cs, cf, pa, rw = r["correct_strict"], r["correct_sigma_free"], r["parsed"], r["reward"]
            out.append(f"| {name.replace('_vs_', ' vs ')} | {split} | {cs['before']} → {cs['after']} ({cs['only_after']}/{cs['only_before']}), p = {cs['mcnemar_p']:.3g} "
                       f"| {cf['before']} → {cf['after']}, p = {cf['mcnemar_p']:.3g} | {pa['before']} → {pa['after']}, p = {pa['mcnemar_p']:.3g} "
                       f"| {rw['mean_diff']:+.3f}, p = {rw['wilcoxon_p']:.3g} |")
    return out


def main(run_dir: str) -> None:
    run = Path(run_dir)
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    lines = [f"# {run.name}", "", "## Training", ""]
    if summary:
        lines.append(f"{summary.get('steps_completed')} steps, {summary.get('seconds_per_step', 0):.1f} s/step, "
                     f"peak {summary['peak_vram']['torch_max_allocated_gib']:.2f} GiB torch / "
                     f"{summary['peak_vram'].get('nvidia_smi_peak_used_gib_device_wide', float('nan')):.2f} GiB device.")
        lines.append("")
    lines += training_table(run) + ["", "## Curriculum", ""] + curriculum_lines(run)
    lines += ["", "## Final eval (frozen eval sets, identical settings)", ""] + eval_table(run)
    lines += ["", "## Paired tests (same problems, greedy)", ""] + tests_table(run)
    text = "\n".join(lines) + "\n"
    (run / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])

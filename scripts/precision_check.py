"""Exploratory, no tests (DECISIONS.md D65): RL adapters evaluated on the 4-bit training base vs the bf16 merge.

    uv run python scripts/precision_check.py

Arms (same frozen eval problems / prefixes, same decoding; only how the adapter is applied differs):
- split-credit seed 0 @287: <run>/compare/rl287{,_sampled} (bf16, LoRA merged) vs rl287_4bit{,_sampled}.
- E1a final: <run>/e1_eval/final{,_sampled} vs final_4bit{,_sampled}. μ is fixed (SFT prefix).
Per arm × precision × decoding: the D62 summary (failures, median relative σ, σ = 0, coverage, crps_log, ρ), plus
accuracy (≤ 1%) for split credit. Paired, greedy: identical completions, same μ (split credit), and the median of
ln(σ_4bit / σ_bf16) on problems answered in both. Training reference: training samples' median relative σ in the last
25 steps, overall and per depth. Reading rule (D65, fixed before results): g = ln(σ̃_4bit,s / σ̃_bf16,s) /
ln(σ̃_train / σ̃_bf16,s); g ≥ 0.75 "precision explains most of the gap", 0.25 ≤ g < 0.75 "partly", g < 0.25 "does not
explain the gap". Writes results/precision_check/report.{json,md}.
"""

import gzip
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from sigma_greedy_vs_sampled import SIGMA_FLOOR_REL, describe, e1_rows, tape_rows, within_problem  # noqa: E402
from physcal.tape.tape import read_tape  # noqa: E402

SPLIT = ROOT / "results/train/split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed0"
E1A = ROOT / "results/train/e1a-sigma-only-free-seed0"
OUT = ROOT / "results/precision_check"
ACC_TOL = 0.01                     # accuracy: |relative error| <= 1% (phase95_compare.py)
TRAIN_WINDOW = 25                  # D65: the last 25 training steps
GAP_LABELS = ((0.75, "precision explains most of the gap"), (0.25, "partly"))  # else "does not explain the gap"

ARMS = {
    "split-credit seed 0 @287": {"reader": "tape", "run": SPLIT, "last_step": 287, "depth_at": 2,
                                 "paths": {("bf16", "greedy"): "compare/rl287", ("bf16", "sampled"): "compare/rl287_sampled",
                                           ("4bit", "greedy"): "compare/rl287_4bit",
                                           ("4bit", "sampled"): "compare/rl287_4bit_sampled"}},
    "E1a final": {"reader": "e1", "run": E1A, "last_step": 300, "depth_at": 1,
                  "paths": {("bf16", "greedy"): "e1_eval/final", ("bf16", "sampled"): "e1_eval/final_sampled",
                            ("4bit", "greedy"): "e1_eval/final_4bit", ("4bit", "sampled"): "e1_eval/final_4bit_sampled"}},
}


def rows_of(arm: dict, rel: str) -> list[dict] | None:
    d = arm["run"] / rel
    if arm["reader"] == "tape":
        if not (d / "rollouts.jsonl").exists():
            return None
        rows = tape_rows(d / "rollouts.jsonl")
        for r, rec in zip(rows, read_tape(d / "rollouts.jsonl")):  # accuracy, μ and the text, same order
            b = rec["breakdown"]
            r["correct"] = b["reason"] == "ok" and b["abs_rel_error"] <= ACC_TOL
            r["mu_si"], r["text"] = b.get("mu_si"), rec["completion"]
        return rows
    if not (d / "rows.jsonl").exists():
        return None
    rows = e1_rows(d / "rows.jsonl")
    for r, rec in zip(rows, read_tape(d / "rollouts.jsonl")):
        r["text"] = rec["suffix"]
    return rows


def median_sigma(rows: list[dict]) -> float | None:
    s = [r["rel_sigma"] for r in rows if not r["fail"] and r["rel_sigma"] is not None]
    return st.median(s) if s else None


def by_depth(rows: list[dict], depth_at: int) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[r["key"][depth_at]].append(r)
    return {d: median_sigma(groups[d]) for d in sorted(groups)}


def training_rows(arm: dict) -> list[dict]:
    """Training samples of the last TRAIN_WINDOW steps, in the eval rows' σ convention."""
    raw = arm["run"] / "rollouts.jsonl"
    opener = (lambda: raw.open()) if raw.exists() else (lambda: gzip.open(raw.with_suffix(".jsonl.gz"), "rt"))
    first = arm["last_step"] - TRAIN_WINDOW + 1
    rows = []
    with opener() as fh:
        for line in fh:
            r = json.loads(line)
            if not first <= r["step"] <= arm["last_step"]:
                continue
            b, p = r["breakdown"], r["problem"]
            if arm["reader"] == "tape":
                ok = b["reason"] == "ok" and b.get("rel_sigma") is not None
                sigma = b["rel_sigma"] if ok else None
            else:  # E1: the σ the verifier scored; a changed μ is a failure, as in e1_row
                ok = b["reason"] == "ok" and not r["mu_changed"] and r.get("sigma_used_rel") is not None
                sigma = r["sigma_used_rel"] if ok else None
            key = (p["family"], p["depth"], p["seed"]) if arm["reader"] == "e1" else ("train", p["family"], p["depth"], p["seed"])
            rows.append({"key": key, "fail": not ok, "rel_sigma": sigma, "step": r["step"]})
    return rows


def paired_greedy(a: list[dict], b: list[dict], reader: str) -> dict:
    """bf16 greedy (a) vs 4-bit greedy (b), matched by problem key."""
    bk = {r["key"]: r for r in b}
    pairs = [(x, bk[x["key"]]) for x in a if x["key"] in bk]
    both = [(x, y) for x, y in pairs if not x["fail"] and not y["fail"] and x["rel_sigma"] is not None
            and y["rel_sigma"] is not None]
    ratio = [math.log(max(y["rel_sigma"], SIGMA_FLOOR_REL) / max(x["rel_sigma"], SIGMA_FLOOR_REL)) for x, y in both]
    out = {"n_pairs": len(pairs), "identical_completion": sum(x["text"] == y["text"] for x, y in pairs) / len(pairs),
           "n_both_answered": len(both),
           "median_ln_sigma_ratio_4bit_over_bf16": st.median(ratio) if ratio else None,
           "share_4bit_sigma_wider": sum(v > 0 for v in ratio) / len(ratio) if ratio else None,
           "share_4bit_sigma_narrower": sum(v < 0 for v in ratio) / len(ratio) if ratio else None}
    if reader == "tape":
        out["same_mu"] = sum(x["mu_si"] == y["mu_si"] for x, y in pairs) / len(pairs)
    return out


def gap_share(s_bf16: float, s_4bit: float, s_train: float) -> tuple[float | None, str]:
    if not (s_bf16 and s_4bit and s_train) or s_train == s_bf16:
        return None, "undefined"
    g = math.log(s_4bit / s_bf16) / math.log(s_train / s_bf16)
    return g, next((label for cut, label in GAP_LABELS if g >= cut), "does not explain the gap")


def main() -> dict:
    out = {}
    for name, arm in ARMS.items():
        rows = {k: rows_of(arm, rel) for k, rel in arm["paths"].items()}
        missing = [arm["paths"][k] for k, v in rows.items() if v is None]
        if missing:
            out[name] = {"missing": missing}
            continue
        train = training_rows(arm)
        res = {"describe": {f"{p} {d}": describe(v) for (p, d), v in rows.items()},
               "paired_greedy": paired_greedy(rows[("bf16", "greedy")], rows[("4bit", "greedy")], arm["reader"]),
               "within_problem": {p: within_problem(rows[(p, "greedy")], rows[(p, "sampled")]) for p in ("bf16", "4bit")},
               "training": {"steps": [arm["last_step"] - TRAIN_WINDOW + 1, arm["last_step"]], "n": len(train),
                            "fail_rate": sum(r["fail"] for r in train) / len(train),
                            "median_rel_sigma": median_sigma(train)},
               "median_rel_sigma_by_depth": {"training": by_depth(train, arm["depth_at"]),
                                             **{f"{p} {d}": by_depth(v, arm["depth_at"]) for (p, d), v in rows.items()}}}
        if arm["reader"] == "tape":
            for (p, d), v in rows.items():
                res["describe"][f"{p} {d}"]["accuracy"] = sum(r["correct"] for r in v) / len(v)
        s = {k: res["describe"][k]["median_rel_sigma"] for k in ("bf16 sampled", "4bit sampled")}
        g, label = gap_share(s["bf16 sampled"], s["4bit sampled"], res["training"]["median_rel_sigma"])
        res["gap"] = {"g": g, "label": label, "sigma_bf16_sampled": s["bf16 sampled"],
                      "sigma_4bit_sampled": s["4bit sampled"], "sigma_train": res["training"]["median_rel_sigma"]}
        g_greedy, label_greedy = gap_share(res["describe"]["bf16 greedy"]["median_rel_sigma"],
                                           res["describe"]["4bit greedy"]["median_rel_sigma"],
                                           res["training"]["median_rel_sigma"])
        res["gap_greedy_descriptive"] = {"g": g_greedy, "label": label_greedy}
        out[name] = res
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(out, indent=2, default=str))
    (OUT / "report.md").write_text(render(out))
    print(render(out))
    return out


def render(out: dict) -> str:
    fmt = lambda x, f: "–" if x is None else format(x, f)
    lines = ["# RL adapters on the 4-bit training base vs the bf16 merge (exploratory, D65; no tests)", "",
             "| model | base | decoding | rows | fail | acc ≤ 1% | median rel. σ | σ = 0 | 1σ cov | 2σ cov | mean crps_log | ρ(σ, err) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, a in out.items():
        if "missing" in a:
            lines.append(f"| {name} | missing: {', '.join(a['missing'])} | | | | | | | | | | |")
            continue
        for k in ("bf16 greedy", "4bit greedy", "bf16 sampled", "4bit sampled"):
            d = a["describe"][k]
            base, dec = k.split()
            lines.append(f"| {name} | {base} | {dec} | {d['n']} | {d['fail_rate']:.1%} | {fmt(d.get('accuracy'), '.1%')} | "
                         f"{fmt(d['median_rel_sigma'], '.3%')} | {fmt(d['sigma_zero'], '.1%')} | {d['cov1']:.1%} | "
                         f"{d['cov2']:.1%} | {d['mean_crps_log']:.4f} | {fmt(d['rho_sigma_err'], '.2f')} |")
        t = a["training"]
        lines.append(f"| {name} | 4bit (training) | sampled, steps {t['steps'][0]}–{t['steps'][1]} | {t['n']} | "
                     f"{t['fail_rate']:.1%} | | {fmt(t['median_rel_sigma'], '.3%')} | | | | | |")
    lines += ["", "**Share of the train/eval σ gap closed** (D65 rule, sampled medians):", "",
              "| model | σ̃ bf16 sampled | σ̃ 4-bit sampled | σ̃ training | g | label | greedy g (descriptive) |",
              "|---|---|---|---|---|---|---|"]
    for name, a in out.items():
        if "missing" not in a:
            gp, gg = a["gap"], a["gap_greedy_descriptive"]
            lines.append(f"| {name} | {fmt(gp['sigma_bf16_sampled'], '.3%')} | {fmt(gp['sigma_4bit_sampled'], '.3%')} | "
                         f"{fmt(gp['sigma_train'], '.3%')} | {fmt(gp['g'], '.2f')} | {gp['label']} | {fmt(gg['g'], '.2f')} |")
    lines += ["", "**Paired, greedy, 4-bit vs bf16** (same problems / prefixes):", "",
              "| model | pairs | identical completion | same μ | both answered | median ln(σ_4bit / σ_bf16) | 4-bit σ wider | 4-bit σ narrower |",
              "|---|---|---|---|---|---|---|---|"]
    for name, a in out.items():
        if "missing" not in a:
            p = a["paired_greedy"]
            lines.append(f"| {name} | {p['n_pairs']} | {p['identical_completion']:.1%} | {fmt(p.get('same_mu'), '.1%')} | "
                         f"{p['n_both_answered']} | {fmt(p['median_ln_sigma_ratio_4bit_over_bf16'], '+.3f')} | "
                         f"{fmt(p['share_4bit_sigma_wider'], '.1%')} | {fmt(p['share_4bit_sigma_narrower'], '.1%')} |")
    lines += ["", "**Median relative σ by depth** (controls for the problem mix):", ""]
    for name, a in out.items():
        if "missing" in a:
            continue
        cols = ["training", "bf16 greedy", "4bit greedy", "bf16 sampled", "4bit sampled"]
        depths = sorted({d for c in cols for d in a["median_rel_sigma_by_depth"][c]}, key=int)
        lines += [f"{name}:", "", "| depth | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for d in depths:
            lines.append(f"| {d} | " + " | ".join(fmt(a["median_rel_sigma_by_depth"][c].get(d), ".3%") for c in cols) + " |")
        lines.append("")
    lines += ["**Within-problem spread of sampled σ** (k = 4):", "",
              "| model | base | median SD of ln σ | greedy σ mean mid-rank | greedy σ below all samples |", "|---|---|---|---|---|"]
    for name, a in out.items():
        if "missing" not in a:
            for p, w in a["within_problem"].items():
                lines.append(f"| {name} | {p} | {fmt(w['median_sd_ln_sigma'], '.2f')} | {fmt(w['greedy_mean_midrank'], '.2f')} | "
                             f"{fmt(w['greedy_below_all_samples'], '.1%')} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

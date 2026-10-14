"""PREREGISTRATION.md family S: split credit vs SFT-only and vs uniform-credit RL, at matched step 287.

    uv run python scripts/phase95_compare.py results/train/<split-credit run dir>

Written before any split-credit result exists. Pooled frozen eval sets (200 ID + 200 depth 5), verifier 1.1.0.
- S1a: split-credit 1σ coverage > SFT-only (McNemar, one-sided; failures count as not covered).
- S1b: split-credit 1σ coverage > uniform-credit RL@287 (McNemar, one-sided).
- S2: Spearman ρ(stated relative σ, |relative error|) of split-credit > uniform-credit (paired bootstrap).
- S3: strict accuracy (≤ 1%) not lower than uniform-credit by more than 5 points (one-sided bootstrap: p = share
  of resamples with difference ≤ −5 points).
Holm over {S1a, S1b, S2, S3} at α = 0.05. Also reported (not tested): 2σ coverage, σ = 0 share, crps_log vs the
post-hoc control (Wilcoxon, two-sided), and the final checkpoint as "longer training".
Writes <run>/compare/phase95_family_S.json and .md.
"""

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/train/overnight-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-512/compare_verifier_1.1.0"
UNIFORM = ROOT / "results/train/overnight2-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512/compare"
POSTHOC = ROOT / "results/posthoc_calibration/sft_eval_scaled_crps_fit.jsonl"
N_BOOT, SEED, ALPHA, MARGIN = 10_000, 0, 0.05, 0.05


def load(path: Path) -> dict:
    out = {}
    for line in path.open():
        r = json.loads(line)
        p, b = r["problem"], r["breakdown"]
        ok = b["reason"] == "ok"
        out[(r["split"], p["family"], p["depth"], p["seed"])] = {
            "cov1": ok and bool(b.get("within_1sigma")), "cov2": ok and bool(b.get("within_2sigma")),
            "correct": ok and b["abs_rel_error"] <= 0.01,
            "rel_sigma": b.get("rel_sigma") if ok else None, "abs_err": b.get("abs_rel_error") if ok else None,
            "crps_log": b["scores"]["crps_log"] if ok else -5.0}
    return out


def mcnemar_greater(after: dict, before: dict, keys: list, m: str) -> dict:
    up = sum(after[k][m] and not before[k][m] for k in keys)
    down = sum(before[k][m] and not after[k][m] for k in keys)
    return {"before": sum(before[k][m] for k in keys), "after": sum(after[k][m] for k in keys), "only_after": up,
            "only_before": down, "p_one_sided": binomtest(up, up + down, 0.5, alternative="greater").pvalue if up + down else 1.0}


def rho(d: dict, keys: list) -> float:
    ks = [k for k in keys if d[k]["rel_sigma"] is not None]
    return float(spearmanr([d[k]["rel_sigma"] for k in ks], [d[k]["abs_err"] for k in ks]).statistic)


def bootstrap(a: dict, b: dict, keys: list, stat) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    keys = np.array(keys, dtype=object)
    out = []
    for _ in range(N_BOOT):
        ks = [tuple(k) for k in keys[rng.integers(0, len(keys), len(keys))]]
        out.append(stat(a, ks) - stat(b, ks))
    return np.array(out)


def holm(p: dict) -> dict:
    order, still, out = sorted(p, key=p.get), True, {}
    for i, name in enumerate(order):
        thr = ALPHA / (len(order) - i)
        still = still and p[name] <= thr
        out[name] = {"p": p[name], "holm_threshold": thr, "supported": still}
    return out


def headline(d: dict, keys: list) -> dict:
    sig = [d[k] for k in keys if d[k]["rel_sigma"] is not None]
    return {"n": len(keys), "acc": st.mean(d[k]["correct"] for k in keys), "cov1": st.mean(d[k]["cov1"] for k in keys),
            "cov2": st.mean(d[k]["cov2"] for k in keys), "sigma_zero": st.mean(x["rel_sigma"] == 0 for x in sig),
            "median_rel_sigma": st.median(x["rel_sigma"] for x in sig), "rho_sigma_err": rho(d, keys),
            "mean_crps_log": st.mean(d[k]["crps_log"] for k in keys)}


def main(run: Path) -> None:
    split = load(run / "compare/rl287/rollouts.jsonl")
    sft, uni, post = load(OLD / "sft/rollouts.jsonl"), load(UNIFORM / "rl287/rollouts.jsonl"), load(POSTHOC)
    keys = sorted(split)
    assert set(keys) == set(sft) == set(uni) == set(post), "different eval problems"
    acc = lambda d, ks: float(np.mean([d[k]["correct"] for k in ks]))
    t = {"S1a_cov1_vs_sft": mcnemar_greater(split, sft, keys, "cov1"),
         "S1b_cov1_vs_uniform": mcnemar_greater(split, uni, keys, "cov1")}
    d_rho = bootstrap(split, uni, keys, rho)
    t["S2_rho_vs_uniform"] = {"split": rho(split, keys), "uniform": rho(uni, keys), "diff_ci": np.percentile(d_rho, [2.5, 97.5]).tolist(),
                              "p_one_sided": float(np.mean(d_rho <= 0))}
    d_acc = bootstrap(split, uni, keys, acc)
    t["S3_accuracy_noninferiority"] = {"split": acc(split, keys), "uniform": acc(uni, keys), "margin": MARGIN,
                                       "diff_ci": np.percentile(d_acc, [2.5, 97.5]).tolist(),
                                       "p_one_sided": float(np.mean(d_acc <= -MARGIN))}
    t["holm"] = holm({"S1a": t["S1a_cov1_vs_sft"]["p_one_sided"], "S1b": t["S1b_cov1_vs_uniform"]["p_one_sided"],
                      "S2": t["S2_rho_vs_uniform"]["p_one_sided"], "S3": t["S3_accuracy_noninferiority"]["p_one_sided"]})
    diffs = [split[k]["crps_log"] - post[k]["crps_log"] for k in keys]
    reported = {"crps_log_vs_posthoc": {"mean_diff": st.mean(diffs), "wilcoxon_p_two_sided": float(wilcoxon(diffs).pvalue)}}
    rows = {"sft": sft, "posthoc": post, "uniform_rl287": uni, "split_rl287": split}
    final = run / "compare/rlfinal/rollouts.jsonl"
    if final.exists():
        rows["split_final"] = load(final)
    head = {name: {s: headline(d, [k for k in keys if k[0] == s]) for s in ("id", "depth_extrapolation")} | {"pooled": headline(d, keys)}
            for name, d in rows.items()}
    out = {"tests": t, "reported": reported, "headline": head}
    (run / "compare/phase95_family_S.json").write_text(json.dumps(out, indent=2, default=float))
    lines = ["| model | split | acc ≤1% | 1σ cov | 2σ cov | σ = 0 | median rel. σ | ρ(σ, err) | mean crps_log |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, h in head.items():
        for s, v in h.items():
            lines.append(f"| {name} | {s} | {v['acc']:.1%} | {v['cov1']:.1%} | {v['cov2']:.1%} | {v['sigma_zero']:.1%} | "
                         f"{v['median_rel_sigma']:.4%} | {v['rho_sigma_err']:.2f} | {v['mean_crps_log']:.4f} |")
    lines += ["", "| hypothesis | p (one-sided) | Holm threshold | supported |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['p']:.3g} | {v['holm_threshold']:.3g} | {v['supported']} |" for k, v in t["holm"].items()]
    (run / "compare/phase95_family_S.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if Path(sys.argv[1]).is_absolute() else ROOT / sys.argv[1])

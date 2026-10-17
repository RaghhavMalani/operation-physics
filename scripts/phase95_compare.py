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

Amendment family A (DECISIONS.md D49, committed before any split-credit result): RL calibration claims are judged
against SFT-only + (a), the text-feature σ scale fitted on the calibration split. Paired bootstrap, one-sided,
Holm over {A1, A2, A3}:
- A1: |1σ coverage − 68.27%| of split-credit < that of SFT-only + (a) (failures count as not covered).
- A2: Spearman ρ(relative σ, |relative error|) of split-credit > that of SFT-only + (a).
- A3: mean crps_log of split-credit > that of SFT-only + (a).
"RL calibration beats post-hoc" = A1 and A2 supported. The same three tests against (a') (text direct, σ' = c(x)·|μ|)
are reported as a sensitivity analysis; (b) (oracle per cell, in-sample) is a headline row only, never tested.
D49's family A is SUPERSEDED by D50/A5 (still computed and reported, labelled so).

D50/A5 family N (the judgement since D50): the strongest nested baseline (results/a5_baselines, chosen on the
calibration split) applied to the SAME split-credit completions, so μ is identical and only σ differs:
- N1: |1σ coverage − 68.27%| smaller; N2: Spearman ρ(σ, |error|) higher; N3: mean crps_log higher.
Holm over {N1, N2, N3}; "RL σ beats the baseline" = N2 and N3 supported (N1 reported).
Writes <run>/compare/phase95_family_S.json and .md.

`--corrected` (DECISIONS.md D67): the same analysis on the corrected evals, every adapter unmerged on its 4-bit
training base: <run>/compare/rl287_4bit, the SFT-only control and uniform credit @287 on the same base
(overnight2…/compare/{sft_4bit,rl287_4bit}). Only the inputs and outputs change:
- the A5 σ is the same fitted model, applied to the corrected completions;
- the standing post-hoc scale (c from posthoc_fit.json, not re-fitted) is applied to the 4-bit SFT answers;
- D49's superseded family A is not re-run.
Writes results/corrected/family_S/<run name>.{json,md}; derived tapes go to results/corrected/derived/.
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
COND = {"text_scale": ROOT / "results/posthoc_calibration/sft_eval_text_scale.jsonl",    # (a), the D49 control
        "text_direct": ROOT / "results/posthoc_calibration/sft_eval_text_direct.jsonl",  # (a'), sensitivity
        "oracle_cell": ROOT / "results/posthoc_calibration/sft_eval_oracle_cell.jsonl"}  # (b), upper bound only
N_BOOT, SEED, ALPHA, MARGIN = 10_000, 0, 0.05, 0.05
COVERAGE_TARGET = 0.6827
CORRECTED = ROOT / "results/corrected"
POSTHOC_FIT = ROOT / "results/posthoc_calibration/posthoc_fit.json"


def paths(run: Path, corrected: bool) -> dict:
    """Inputs and outputs of one analysis: the superseded (bf16-merged) evals, or the corrected ones (D67)."""
    if not corrected:
        return {"split": run / "compare/rl287/rollouts.jsonl", "sft": OLD / "sft/rollouts.jsonl",
                "uniform": UNIFORM / "rl287/rollouts.jsonl", "posthoc": POSTHOC,
                "sft_a5": ROOT / "results/a5_baselines/sft_eval_a5_sigma.jsonl", "sft_a5_from": None,
                "a5_out": run / "compare/rl287_a5_sigma.jsonl", "final": run / "compare/rlfinal/rollouts.jsonl",
                "out": run / "compare/phase95_family_S", "family_a": True}
    d = CORRECTED / "derived"
    return {"split": run / "compare/rl287_4bit/rollouts.jsonl", "sft": UNIFORM / "sft_4bit/rollouts.jsonl",
            "uniform": UNIFORM / "rl287_4bit/rollouts.jsonl", "posthoc": d / "sft_4bit_posthoc_crps_fit.jsonl",
            "sft_a5": d / "sft_4bit_a5_sigma.jsonl", "sft_a5_from": UNIFORM / "sft_4bit/rollouts.jsonl",
            "a5_out": d / f"{run.name}_rl287_4bit_a5_sigma.jsonl", "final": run / "compare/rlfinal_4bit/rollouts.jsonl",
            "out": CORRECTED / "family_S" / run.name, "family_a": False}


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


def family_a(rl: dict, ctrl: dict, keys: list, prefix: str = "A", beats: tuple = ("1", "2")) -> dict:
    """Coverage-error, ρ and crps_log tests of `rl` against a control on the pooled eval sets (D49 family A, or
    D50 family N with prefix "N" and beats = ("2", "3"))."""
    cov_err = lambda d, ks: abs(float(np.mean([d[k]["cov1"] for k in ks])) - COVERAGE_TARGET)
    crps = lambda d, ks: float(np.mean([d[k]["crps_log"] for k in ks]))
    t = {}
    for name, stat, better_low in ((f"{prefix}1_cov1_error", cov_err, True), (f"{prefix}2_rho", rho, False),
                                   (f"{prefix}3_crps_log", crps, False)):
        d = bootstrap(rl, ctrl, keys, stat)
        t[name] = {"rl": stat(rl, keys), "control": stat(ctrl, keys), "diff_ci": np.percentile(d, [2.5, 97.5]).tolist(),
                   "p_one_sided": float(np.mean(d >= 0) if better_low else np.mean(d <= 0))}
    t["holm"] = holm({k.split("_")[0]: v["p_one_sided"] for k, v in t.items()})
    t["rl_calibration_beats_posthoc"] = all(t["holm"][f"{prefix}{i}"]["supported"] for i in beats)
    return t


def headline(d: dict, keys: list) -> dict:
    sig = [d[k] for k in keys if d[k]["rel_sigma"] is not None]
    return {"n": len(keys), "acc": st.mean(d[k]["correct"] for k in keys), "cov1": st.mean(d[k]["cov1"] for k in keys),
            "cov2": st.mean(d[k]["cov2"] for k in keys), "sigma_zero": st.mean(x["rel_sigma"] == 0 for x in sig),
            "median_rel_sigma": st.median(x["rel_sigma"] for x in sig), "rho_sigma_err": rho(d, keys),
            "mean_crps_log": st.mean(d[k]["crps_log"] for k in keys)}


def main(run: Path, corrected: bool = False) -> None:
    P = paths(run, corrected)
    from physcal.eval.nested_baselines import load_models, write_scaled_tape

    models, sel = load_models()
    if corrected:  # derived inputs on the 4-bit SFT answers: the standing post-hoc scale and the A5 σ
        from physcal.eval.posthoc import scaled_tape

        P["posthoc"].parent.mkdir(parents=True, exist_ok=True)
        c = json.loads(POSTHOC_FIT.read_text())["scales"]["crps_fit"]
        scaled_tape(P["sft"], c, P["posthoc"])
        write_scaled_tape(P["sft_a5_from"], models["sigma"][sel["strongest"]["sigma"]], P["sft_a5"])
    split = load(P["split"])
    sft, uni, post = load(P["sft"]), load(P["uniform"]), load(P["posthoc"])
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
    cond = {name: load(path) for name, path in COND.items()} if P["family_a"] else {}
    assert all(set(keys) == set(c) for c in cond.values()), "different eval problems (conditional post-hoc)"
    a5_tape = P["a5_out"]
    write_scaled_tape(P["split"], models["sigma"][sel["strongest"]["sigma"]], a5_tape)
    a5_same = load(a5_tape)
    amendment = {"N_vs_A5_strongest_same_completions": family_a(split, a5_same, keys, "N", ("2", "3"))}
    if P["family_a"]:
        amendment |= {"superseded_D49_A_vs_text_scale": family_a(split, cond["text_scale"], keys),
                      "superseded_D49_sensitivity_A_vs_text_direct": family_a(split, cond["text_direct"], keys)}
    rows = {"sft": sft, "posthoc": post, **{f"sft_{k}": v for k, v in cond.items()},
            "sft_a5_strongest": load(P["sft_a5"]), "uniform_rl287": uni,
            "split_rl287": split, "split_rl287_a5_strongest": a5_same}
    final = P["final"]
    if final.exists():
        rows["split_final"] = load(final)
    head = {name: {s: headline(d, [k for k in keys if k[0] == s]) for s in ("id", "depth_extrapolation")} | {"pooled": headline(d, keys)}
            for name, d in rows.items()}
    out = {"tests": t, "amendment_D49": amendment, "reported": reported, "headline": head,
           "inputs": {k: str(v) for k, v in P.items() if isinstance(v, Path)}, "corrected_D67": corrected}
    P["out"].parent.mkdir(parents=True, exist_ok=True)
    Path(f"{P['out']}.json").write_text(json.dumps(out, indent=2, default=float))  # run names contain dots
    lines = ["| model | split | acc ≤1% | 1σ cov | 2σ cov | σ = 0 | median rel. σ | ρ(σ, err) | mean crps_log |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, h in head.items():
        for s, v in h.items():
            lines.append(f"| {name} | {s} | {v['acc']:.1%} | {v['cov1']:.1%} | {v['cov2']:.1%} | {v['sigma_zero']:.1%} | "
                         f"{v['median_rel_sigma']:.4%} | {v['rho_sigma_err']:.2f} | {v['mean_crps_log']:.4f} |")
    lines += ["", "| hypothesis | p (one-sided) | Holm threshold | supported |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['p']:.3g} | {v['holm_threshold']:.3g} | {v['supported']} |" for k, v in t["holm"].items()]
    for name, a in amendment.items():
        lines += ["", f"{name} (split-credit RL@287 stated σ vs the control; RL σ beats it: "
                  f"{a['rl_calibration_beats_posthoc']})", "",
                  "| test | RL | control | 95% CI of RL − control | p (one-sided) | Holm threshold | supported |",
                  "|---|---|---|---|---|---|---|"]
        for k, v in a.items():
            if k[0] in "AN" and k[1].isdigit():
                h = a["holm"][k.split("_")[0]]
                lines.append(f"| {k} | {v['rl']:.4f} | {v['control']:.4f} | [{v['diff_ci'][0]:.4f}, {v['diff_ci'][1]:.4f}] "
                             f"| {h['p']:.3g} | {h['holm_threshold']:.3g} | {h['supported']} |")
    Path(f"{P['out']}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--corrected"]
    main(Path(args[0]) if Path(args[0]).is_absolute() else ROOT / args[0], corrected="--corrected" in sys.argv[1:])

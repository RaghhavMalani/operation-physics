"""PREREGISTRATION.md family S on the seed mean (DECISIONS.md D62), once split-credit seeds 0, 1 and 2 exist.

    uv run python scripts/phase95_seed_mean.py <seed-0 run dir> <seed-1 run dir> <seed-2 run dir>

Per problem i (the pooled 200 ID + 200 depth-5 frozen problems) and seed s: the step-287 greedy outcomes
(<run>/compare/rl287). Controls as pre-registered: SFT-only and the single uniform-credit RL@287 run. One-sided
paired bootstrap over problems (B = 10,000, seed 0; one index set per resample, shared by every arm and test),
Holm over {S1a, S1b, S2, S3} at α = 0.05:
- S1a / S1b: mean over i of (mean over s of 1σ coverage) − the control's coverage; p = share of resamples ≤ 0.
- S2: (mean over s of ρ_s) − ρ_uniform, ρ = phase95_compare.rho; p = share ≤ 0.
- S3: seed-mean strict accuracy − uniform accuracy; p = share ≤ −5 points.
A hypothesis is supported iff it is supported here (PREREGISTRATION.md family S). The per-seed tests (McNemar for
S1, bootstrap for S2/S3, exactly as phase95_compare.py) and the range across seeds are reported alongside.
Writes results/split_credit_seed_mean/family_S_seed_mean.{json,md}.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase95_compare import MARGIN, N_BOOT, OLD, ROOT, SEED, UNIFORM, bootstrap, holm, load, mcnemar_greater, rho  # noqa: E402

OUT = ROOT / "results/split_credit_seed_mean"
EVAL_TAPE = "compare/rl287/rollouts.jsonl"


def cov(d: dict, ks: list) -> float:
    return float(np.mean([d[k]["cov1"] for k in ks]))


def acc(d: dict, ks: list) -> float:
    return float(np.mean([d[k]["correct"] for k in ks]))


def seed_mean_tests(seeds: list[dict], sft: dict, uni: dict, keys: list, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """The D62 seed-mean tests. `seeds` holds one loaded eval tape per split-credit seed."""
    mean_over = lambda stat: (lambda ks: float(np.mean([stat(d, ks) for d in seeds])))
    tests = {"S1a": (mean_over(cov), lambda ks: cov(sft, ks), 0.0),
             "S1b": (mean_over(cov), lambda ks: cov(uni, ks), 0.0),
             "S2": (mean_over(rho), lambda ks: rho(uni, ks), 0.0),
             "S3": (mean_over(acc), lambda ks: acc(uni, ks), -MARGIN)}
    rng = np.random.default_rng(seed)
    karr = np.array(keys, dtype=object)
    resamples = [[tuple(k) for k in karr[rng.integers(0, len(keys), len(keys))]] for _ in range(n_boot)]
    out = {}
    for name, (a, b, threshold) in tests.items():
        d = np.array([a(ks) - b(ks) for ks in resamples])
        out[name] = {"seed_mean": a(keys), "control": b(keys), "diff": a(keys) - b(keys),
                     "diff_ci": np.percentile(d, [2.5, 97.5]).tolist(), "p_threshold": threshold,
                     "p_one_sided": float(np.mean(d <= threshold))}
    out["holm"] = holm({k: v["p_one_sided"] for k, v in out.items()})
    return out


def per_seed_tests(split: dict, sft: dict, uni: dict, keys: list) -> dict:
    """phase95_compare.py's per-seed tests (McNemar for S1, bootstrap for S2/S3), without Holm."""
    d_rho, d_acc = bootstrap(split, uni, keys, rho), bootstrap(split, uni, keys, acc)
    return {"S1a": mcnemar_greater(split, sft, keys, "cov1"), "S1b": mcnemar_greater(split, uni, keys, "cov1"),
            "S2": {"split": rho(split, keys), "uniform": rho(uni, keys), "p_one_sided": float(np.mean(d_rho <= 0))},
            "S3": {"split": acc(split, keys), "uniform": acc(uni, keys), "p_one_sided": float(np.mean(d_acc <= -MARGIN))}}


def main(runs: list[Path]) -> dict:
    if len(runs) != 3:
        raise SystemExit("family S on the seed mean needs exactly 3 split-credit runs (seeds 0, 1, 2)")
    seeds = [load(r / EVAL_TAPE) for r in runs]
    sft, uni = load(OLD / "sft/rollouts.jsonl"), load(UNIFORM / "rl287/rollouts.jsonl")
    keys = sorted(seeds[0])
    assert all(set(keys) == set(d) for d in seeds + [sft, uni]), "different eval problems"
    sanity = [json.loads((r / "sanity.json").read_text()) if (r / "sanity.json").exists() else None for r in runs]
    per_seed = {r.name: per_seed_tests(d, sft, uni, keys) for r, d in zip(runs, seeds)}
    point = {"cov1": [cov(d, keys) for d in seeds], "acc": [acc(d, keys) for d in seeds], "rho": [rho(d, keys) for d in seeds]}
    out = {"runs": [str(r.relative_to(ROOT)) for r in runs], "sanity": [s and s.get("sane") for s in sanity],
           "seed_mean": seed_mean_tests(seeds, sft, uni, keys), "per_seed": per_seed,
           "range_across_seeds": {k: [min(v), max(v)] for k, v in point.items()}, "per_seed_point": point}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "family_S_seed_mean.json").write_text(json.dumps(out, indent=2, default=float))
    t = out["seed_mean"]
    lines = ["# Family S on the seed mean (D62)", "", f"Runs: {', '.join(out['runs'])}; sanity: {out['sanity']}", "",
             "| test | split credit, seed mean | control | difference | 95% CI | p (one-sided) | Holm threshold | supported |",
             "|---|---|---|---|---|---|---|---|"]
    for k in ("S1a", "S1b", "S2", "S3"):
        v, h = t[k], t["holm"][k]
        lines.append(f"| {k} | {v['seed_mean']:.4f} | {v['control']:.4f} | {v['diff']:+.4f} | "
                     f"[{v['diff_ci'][0]:+.4f}, {v['diff_ci'][1]:+.4f}] | {h['p']:.3g} | {h['holm_threshold']:.3g} | "
                     f"{bool(h['supported'])} |")
    lines += ["", "Per seed (as phase95_compare.py; one-sided p, no Holm):", "",
              "| run | S1a covered vs SFT | p | S1b covered vs uniform | p | S2 ρ split vs uniform | p | S3 acc split vs uniform | p |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, s in per_seed.items():
        lines.append(f"| {name[-5:]} | {s['S1a']['after']} vs {s['S1a']['before']} | {s['S1a']['p_one_sided']:.3g} | "
                     f"{s['S1b']['after']} vs {s['S1b']['before']} | {s['S1b']['p_one_sided']:.3g} | "
                     f"{s['S2']['split']:.3f} vs {s['S2']['uniform']:.3f} | {s['S2']['p_one_sided']:.3g} | "
                     f"{s['S3']['split']:.1%} vs {s['S3']['uniform']:.1%} | {s['S3']['p_one_sided']:.3g} |")
    r = out["range_across_seeds"]
    lines += ["", f"Range across seeds: 1σ coverage {r['cov1'][0]:.1%}–{r['cov1'][1]:.1%}, accuracy "
              f"{r['acc'][0]:.1%}–{r['acc'][1]:.1%}, ρ {r['rho'][0]:.3f}–{r['rho'][1]:.3f}."]
    (OUT / "family_S_seed_mean.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


if __name__ == "__main__":
    main([Path(a) if Path(a).is_absolute() else ROOT / a for a in sys.argv[1:]])

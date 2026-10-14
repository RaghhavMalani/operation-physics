"""Phase 10 step 2: rounding audit of the frozen eval sets (CPU; FINDINGS #8).

    uv run python -m physcal.eval.rounding_audit

Three questions:
1. **Does ground truth use the displayed (rounded) givens?** For every eval problem, re-parse each given's display
   string, convert it to SI, re-run the chain and compare with the stored truth.
2. **Counterfactual σ\\*:** if the displayed givens were measurements rounded to 3 s.f. (true value uniform within
   half a unit in the last displayed place), how uncertain would the target be? Monte Carlo through the chain;
   σ\\* is reported as a relative standard deviation and in log space, against the 1% accuracy band.
3. **Solver-side rounding:** the error a perfect solver makes if it rounds every intermediate to k significant
   figures (the SFT targets use 4) and reports the final answer to k s.f. This is the irreducible error behind
   the "exactly right" spike of the observed errors.
Writes `results/rounding_audit/rounding_audit.json`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.env.formulas import get_family
from physcal.env.generator import chain_of, env_config, generator_version
from physcal.env.units import to_si

QUANTILES = (0.1, 0.5, 0.9, 0.99)


def half_ulp_rel(display: str, sig_figs: int) -> float:
    """Half a unit in the last displayed place, relative to the displayed value."""
    x = abs(float(display))
    exp = math.floor(math.log10(x))
    return 0.5 * 10 ** (exp - sig_figs + 1) / x


def round_sig(x: np.ndarray | float, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    exp = np.floor(np.log10(np.abs(x)))
    scale = 10.0 ** (exp - k + 1)
    return np.round(x / scale) * scale


def chain_values(fam, chain, givens: dict[str, np.ndarray], round_k: int | None = None) -> dict[str, np.ndarray]:
    env = {**givens, **fam.si_constants}
    for name, out in chain.steps:
        v = np.asarray(fam.formula(name).solved[out](env), dtype=float)
        env[out] = round_sig(v, round_k) if round_k else v
    return env


def audit_problem(p: dict, sig_figs: int, n_mc: int, rng: np.random.Generator, solver_sig_figs: list[int]) -> dict:
    fam, chain = get_family(p["family"]), chain_of(p)
    used = [g for g in p["givens"] if g["symbol"] in chain.givens]
    # 1. truth from the displayed strings
    reparsed = {g["symbol"]: to_si(float(g["display"]), g["unit"]) for g in used}
    truth = float(chain_values(fam, chain, {k: np.array(v) for k, v in reparsed.items()})[chain.target])
    stored_vs_reparsed = max(abs(reparsed[g["symbol"]] - g["value_si"]) / g["value_si"] for g in used)
    # 2. counterfactual sigma*: uniform rounding error on every used given
    mc = {g["symbol"]: g["value_si"] * (1 + rng.uniform(-1, 1, n_mc) * half_ulp_rel(g["display"], sig_figs))
          for g in used}
    y = chain_values(fam, chain, mc)[chain.target]
    y = y[np.isfinite(y) & (y > 0)]
    # 3. solver rounding: intermediates and the final answer rounded to k s.f.
    solver = {}
    for k in solver_sig_figs:
        g_k = {g["symbol"]: np.array(g["value_si"]) for g in used}
        out = chain_values(fam, chain, g_k, round_k=k)[chain.target]
        solver[str(k)] = float(abs(out - p["target_value_si"]) / p["target_value_si"])
    return {
        "key": [p["family"], p["depth"], p["seed"]],
        "depth": p["depth"],
        "truth_rel_diff": abs(truth - p["target_value_si"]) / p["target_value_si"],
        "given_si_rel_diff": stored_vs_reparsed,
        "max_given_half_ulp_rel": max(half_ulp_rel(g["display"], sig_figs) for g in used),
        "sigma_star_rel": float(np.std(y) / p["target_value_si"]),
        "sigma_star_log": float(np.std(np.log(y))),
        "mc_max_abs_rel_dev": float(np.max(np.abs(y / p["target_value_si"] - 1))),
        "solver_rounding_rel_error": solver,
    }


def quantiles(xs: list[float]) -> dict[str, float]:
    return {f"q{int(q * 100)}": float(np.quantile(xs, q)) for q in QUANTILES}


def observed_errors(tape: Path) -> dict[str, dict]:
    """The SFT-only model's eval errors, for comparison with the rounding floors."""
    out = {}
    for line in tape.open():
        r = json.loads(line)
        b = r["breakdown"]
        if b["reason"] == "ok":
            out.setdefault(r["split"], []).append(b["abs_rel_error"])
    return {s: {"n": len(v), **quantiles(v), "share_le_1e-3": float(np.mean(np.array(v) <= 1e-3)),
                "share_le_1e-2": float(np.mean(np.array(v) <= 1e-2))} for s, v in out.items()}


def main() -> None:
    cfg = load_config("rounding_audit")
    env = env_config()
    version = generator_version(env)
    sig = env["generator"]["display"]["sig_figs"]
    rng = np.random.default_rng(cfg["seed"])
    root = REPO_ROOT / "data" / "eval_sets" / version
    rows = []
    for split in ("id", "depth_extrapolation"):
        for line in (root / f"{split}.jsonl").open():
            rows.append({"split": split, **audit_problem(json.loads(line), sig, cfg["n_mc"], rng, cfg["solver_sig_figs"])})
    band = cfg["accuracy_band"]
    summary: dict = {
        "generator_version": version, "display_sig_figs": sig, "n_problems": len(rows), "n_mc": cfg["n_mc"],
        "q1_truth_from_displayed_givens": {
            "max_truth_rel_diff": max(r["truth_rel_diff"] for r in rows),
            "max_given_si_rel_diff": max(r["given_si_rel_diff"] for r in rows),
        },
        "q2_counterfactual_sigma_star": {}, "q3_solver_rounding": {},
        "observed_sft_only_errors": observed_errors(REPO_ROOT / cfg["sft_eval_tape"]),
    }
    for d in sorted({r["depth"] for r in rows}) + ["all"]:
        sub = [r for r in rows if d == "all" or r["depth"] == d]
        s = [r["sigma_star_rel"] for r in sub]
        summary["q2_counterfactual_sigma_star"][str(d)] = {
            "n": len(sub), "sigma_star_rel": quantiles(s), "sigma_star_over_band_median": float(np.median(s)) / band,
            "share_sigma_star_gt_0.1pct": float(np.mean(np.array(s) > 1e-3)),
            "share_2sigma_star_gt_band": float(np.mean(2 * np.array(s) > band)),
            "max_given_half_ulp_rel": quantiles([r["max_given_half_ulp_rel"] for r in sub]),
        }
        summary["q3_solver_rounding"][str(d)] = {
            k: {**quantiles([r["solver_rounding_rel_error"][k] for r in sub]),
                "share_gt_band": float(np.mean(np.array([r["solver_rounding_rel_error"][k] for r in sub]) > band))}
            for k in map(str, cfg["solver_sig_figs"])}
    out = REPO_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "rounding_audit.json").write_text(json.dumps({"summary": summary, "problems": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

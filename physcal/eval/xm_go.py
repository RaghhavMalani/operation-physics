"""The cross-model pilot's go rule on D-pos-v2 alone (DECISIONS.md D139, your instruction of 2026-10-29), run once as
soon as the Llama D-pos-v2 eval exists, so the queue can decide on seed 1 the same night:

    PHYSCAL_CONFIRMATORY=1 OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.eval.xm_go analyze

The rule and its estimator are D137's, through xm_pilot's own code path for D-pos-v2's answer σ (a): the frozen A6c +
A6b problems, `r4_confirm.seed_sources`, `r4_analysis.model_pairs`, `r4_analysis.slope` (OLS within pairs, pairs
bootstrap B = 10,000, seed 0) and `xm_pilot.go_rule` (K1 ∧ C2 ∧ C3). It needs no B-v2 output, and it skips xm_pilot's
descriptive parts (other σ sources, expected scores, tails). The one-time pilot analysis (xm_pilot) still runs when
both Llama evals exist; it computes the same rule from the same logs and must reproduce this file's numbers.
"""

from __future__ import annotations

import argparse
import json

from physcal.config import REPO_ROOT, load_config
from physcal.eval import r4_analysis as R
from physcal.eval import r4_confirm as RC
from physcal.eval import stage_a
from physcal.eval import stage_b_analysis as B
from physcal.eval import xm_pilot as P


def complete_eval(d, n_problems: int) -> str | None:
    """None if `d` holds a finished eval (metrics.json is written after the tape) with every problem of both sets;
    otherwise why the go rule cannot be evaluated."""
    if not (d / "metrics.json").exists():
        return f"no finished eval in {d} (no metrics.json)"
    recs = B.read_records(d)
    n = len([r for r in recs or [] if r["split"] in P.SETS])
    return None if n == n_problems else f"{d} has {n} records on A6c + A6b, expected {n_problems}"


def go_dpos2(dirs: dict, cfg: dict | None = None) -> dict:
    """The go rule on `dirs["dpos2"]` (a model's confirmatory eval dir), with xm_pilot's problems and estimator."""
    cfg = cfg or P._config()
    RC._guard()
    r4 = load_config(cfg["stage_r4_config"])
    ac = r4["analysis"]
    sa = RC.stage_a_config(r4)
    sa["pair_sets"] = [s for s in sa["pair_sets"] if s in P.SETS]
    pc = load_config(load_config(ac["stage_b_config"])["analysis"]["pairs_config"])
    dcfg = load_config(r4["sft_phase12_config"])
    probs = stage_a.load_sets(sa)
    why = complete_eval(dirs["dpos2"], len(probs))
    if why:
        return {"evaluable": False, "reason": why}
    recs = [r for r in B.read_records(dirs["dpos2"]) if r["split"] in P.SETS]
    src = RC.seed_sources(probs, recs, "dpos2", R.references(probs, dcfg), dcfg, ac)
    pairs = R.model_pairs(probs, src["bds"], pc)
    c, b = R.slope(pairs["a6c"], ac), R.slope(pairs["a6b"], ac, "truth")
    return {"evaluable": True, **P.go_rule(c, b, ac, pc["invariance_analysis"]["equivalence_margin"]),
            "n_problems": len(probs)}


def analyze(cfg: dict | None = None) -> dict:
    cfg = cfg or P._config()
    out = REPO_ROOT / cfg["output_dir"] / "go_dpos2"
    if (out / "go.json").exists():
        raise RuntimeError(f"D139: the go rule runs once; {out / 'go.json'} exists")
    g = go_dpos2(P.models(cfg)["llama"], cfg)
    if not g["evaluable"]:
        raise RuntimeError(f"D139: the go rule cannot be evaluated: {g['reason']}")
    res = {"decision": "D139 (D137's go rule on the Llama D-pos-v2 eval, before B-v2's)", **g}
    out.mkdir(parents=True, exist_ok=True)
    (out / "go.json").write_text(json.dumps(res, indent=1, default=B._json_default))
    (out / "summary.md").write_text(render(res))
    return res


def render(g: dict) -> str:
    ci = lambda c: "–" if not c else f"[{c[0]:+.3f}, {c[1]:+.3f}]"
    f = lambda v: "–" if v is None else f"{v:+.3f}"
    return "\n".join([
        "# The pilot's go rule on D-pos-v2 (D137's rule, run early per D139): Llama-3.2-3B-Instruct, seed 0", "",
        f"**{'GO' if g['go'] else 'NO-GO'}** (K1 {g['K1']}, C2 {g['C2']}, C3 {g['C3']}; C1 {g['C1']}). "
        f"{g['n_problems']} frozen A6c + A6b problems.", "",
        f"- A6c slope (answer σ, OLS within pairs): {f(g['a6c_slope'])} {ci(g['a6c_ci90'])} (90% CI), "
        f"one-sided p {g['a6c_p_one_sided']}, {g['a6c_n_pairs']} pairs (K1: p < 0.05; C2: slope ≥ 0.5)",
        f"- A6b slope on computation-correct pairs: {f(g['a6b_slope'])} {ci(g['a6b_ci90'])} (90% CI), "
        f"{g['a6b_n_pairs']} pairs (C3: CI inside ±0.3; fewer than 3 pairs = not met)", ""])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["analyze"])
    ap.parse_args()
    print(render(analyze()))


if __name__ == "__main__":
    main()

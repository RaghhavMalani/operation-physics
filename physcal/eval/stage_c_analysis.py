"""Phase 12 stage C pilot (DECISIONS.md D106, pre-registered): the analysis, CPU only.

    OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.eval.stage_c_analysis analyze

Each eval arm (final adapter, step 150, the SFT-only control; one seed) goes through stage B's analysis (D104's code,
`stage_b_analysis.analyze`, unchanged). C1–C5 are the final adapter's B1–B5; C6 is the RL effect: the mean
per-problem oracle-centred expected crps_log, final − control, > 0, with the bootstrap over pairs. Holm over C1–C6.
"""

from __future__ import annotations

import argparse
import copy
import json

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.eval import oracle_centred as oc
from physcal.eval import stage_a
from physcal.eval import stage_b_analysis as B
from physcal.eval.stats import holm
from physcal.verify.reward import reward_config

ARMS = ("final", "step150", "sft_control")
TESTS = ("C1", "C2", "C3", "C4", "C5", "C6")


def _config() -> dict:
    return load_config("stage_c")


def rl_effect(root, cfg: dict) -> dict | None:
    """C6: expected crps_log (final − SFT-only control) per problem, mean over problems with draws, bootstrap over
    pairs (stage A's MC streams, as D104)."""
    ac = cfg["analysis"]
    recs = {a: B.read_records(root / a / "seed0" / "eval") for a in ("final", "sft_control")}
    if any(r is None for r in recs.values()):
        return None
    sa = load_config(load_config(ac["stage_b_config"])["analysis"]["stage_a_config"])
    sc = reward_config()["scoring"]
    problems = stage_a.load_sets(sa)
    terms = [oc.problem_terms(p, sa["mc"]["n_draws"], sa["mc"]["seed_tag"]) for p in problems]
    exp = {a: oc.score_source(terms, B.model_sigma(problems, r)[0], sc)["expected"] for a, r in recs.items()}
    mask = np.array([t["draws"] is not None for t in terms])
    boot = oc.ClusterBootstrap([(p["_set"], p["noise"]["pair"]["pair_seed"]) for p in problems], ac["bootstrap"], ac["seed"])
    pt, dist = B.boot_mean_dist(boot, exp["final"] - exp["sft_control"], mask)
    return {"value": pt, "ci95": np.percentile(dist, [2.5, 97.5]).tolist(), "p_one_sided": float(np.mean(dist <= 0)),
            "n_problems": int(mask.sum())}


def reading(t: dict, final: dict | None) -> list[str]:
    """D106's reading rules."""
    sup = {k: bool(t[k]["holm"]["rejected"]) for k in TESTS}
    sm = (final or {}).get("seed_mean") or {}
    out = []
    if sup["C1"] and sup["C2"] and sm.get("a6b_equivalent_to_zero") and sm.get("a6_label") in B.PROPAGATING:
        out.append(f"σ-RL propagates (C1, C2 supported; A6b equivalent to 0; A6 label {sm['a6_label']})")
    if sup["C6"]:
        out.append("RL moved σ toward the predictive distribution (C6 supported)")
    if sup["C6"] and not sup["C2"]:
        out.append("C6 supported, C2 not: a better score without learning allocation")
    if sup["C6"] or sup["C2"]:
        out.append("signal: 3 seeds are warranted (R3-4 C)")
    if not out:
        out.append("no signal at the pilot budget: reported as observed")
    return out


def analyze(cfg: dict | None = None, root=None) -> dict:
    cfg = cfg or _config()
    ac = cfg["analysis"]
    root = root or REPO_ROOT / cfg["output_dir"]
    bcfg = copy.deepcopy(load_config(ac["stage_b_config"]))
    bcfg["seeds"] = [0]
    bcfg["analysis"].update(bootstrap=ac["bootstrap"], seed=ac["seed"], alpha=ac["alpha"])
    arms = {a: (B.analyze(bcfg, root / a) if B.read_records(root / a / "seed0" / "eval") is not None else None)
            for a in ARMS}
    final = arms["final"]
    t = {f"C{i}": (copy.deepcopy(final["tests"][f"B{i}"]) if final else {"missing": True}) for i in range(1, 6)}
    t["C6"] = rl_effect(root, cfg) or {"missing": True}
    h = holm({k: (1.0 if v.get("missing") else v["p_one_sided"]) for k, v in t.items()}, ac["alpha"])
    for k in TESTS:
        t[k]["holm"] = h[k]
    res = {"decision": "D106 (stage C pilot, pre-registered)", "tests": t, "reading": reading(t, final),
           "arms": {a: (None if r is None else {"tests": r["tests"], "seed_mean": r["seed_mean"],
                                                 "seed0": r["seeds"][0]}) for a, r in arms.items()},
           "comparators": (final or arms["sft_control"] or {}).get("comparators")}
    sb = REPO_ROOT / bcfg["output_dir"] / "results.json"
    if sb.exists():  # descriptive only: stage B (D104) has another base
        b = json.loads(sb.read_text())
        res["stage_b_seed_mean"] = {k: b["tests"][k] for k in B.TESTS} | {"seed_mean": b["seed_mean"]}
    root.mkdir(parents=True, exist_ok=True)
    (root / "results.json").write_text(json.dumps(res, indent=1, default=B._json_default))
    (root / "summary.md").write_text(render(res))
    return res


def render(res: dict) -> str:
    t = res["tests"]
    names = {"C1": "C1: A6 slope > 0", "C2": "C2: A6c slope > 0", "C3": "C3: expected crps_log − RSS > 0",
             "C4": "C4: expected crps_log − X1 > 0", "C5": "C5: A6 slope − X1's > 0",
             "C6": "C6: expected crps_log, RL − SFT-only control > 0"}
    L = ["# Phase 12 stage C pilot (D106): E1d-style σ-only RL with verifier 1.3.0's σ reward", "",
         "**Reading (D106):** " + "; ".join(res["reading"]) + ".", "",
         "| test | estimate [95% CI] | p (one-sided) | Holm |", "|---|---|---|---|"]
    for k in TESTS:
        d = t[k]
        if d.get("missing"):
            L.append(f"| {names[k]} | missing | 1 | not supported |")
            continue
        L.append(f"| {names[k]} | {B._ci(d)} | {d['p_one_sided']:.4g} | "
                 f"{'supported' if d['holm']['rejected'] else 'not supported'} |")
    L += ["", "| arm | A6 slope (label) | A6b slope | A6c slope | expected crps_log | median abs ln(σ/σ*) | vol beta | "
          "1σ / 2σ cov. | μ-changed / failed |", "|---|---|---|---|---|---|---|---|---|"]
    for a, r in res["arms"].items():
        if r is None:
            L.append(f"| {a} | not run | | | | | | | |")
            continue
        s = r["seed0"]
        o = s["oracle_centred"]
        L.append(f"| {a} | {s['a6']['primary_computation_correct']['slope'] or float('nan'):+.3f} ({s['a6']['label']}) | "
                 f"{s['a6b']['primary_computation_correct']['slope'] or float('nan'):+.3f} | "
                 f"{s['a6c']['primary_all_answered']['slope'] or float('nan'):+.3f} | {o['expected'][0]:.4f} | "
                 f"{o['median_abs_log_ratio']:.3f} | {o['vol_beta'][0]:.3f} | {o['coverage_1'][0]:.1%} / "
                 f"{o['coverage_2'][0]:.1%} | {1 - s['tape']['answered_rate']:.1%} |")
    if res.get("stage_b_seed_mean"):
        b = res["stage_b_seed_mean"]
        L += ["", f"Stage B (D104; another base, descriptive): B1 {B._ci(b['B1'])}, B2 {B._ci(b['B2'])}, "
                  f"B3 {B._ci(b['B3'])}."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["analyze"])
    ap.parse_args()
    print(render(analyze()))


if __name__ == "__main__":
    main()

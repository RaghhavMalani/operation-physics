"""Phase 12 stage A (DECISIONS.md D100; R3-4 A, R3-6): the scoring audit, with the oracle-centred diagnostic. CPU only,
descriptive.

    OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.eval.stage_a run

Does the scoring Phase 12 will use reward σ propagation on noisy problems? Every forecast is centred at μ = f(x_meas):
1. the optimum: σ = c·σ* on a log grid, mean realized crps_log on the capped gate sample (10,005 noisy problems);
2. the resolution: per-problem score differences σ* − alternative (c·σ*, the noise heuristics, X1, an oracle constant),
   realized and expected, on the frozen A6 / A6b / A6c sets: mean, SD, n₂ = (2·SD / mean)², variance ratio;
3. the ranking: sources by mean realized crps_log vs by median |ln(σ / σ*)| (Kendall τ);
4. the verifier's details: clip and floor shares, and the diagnostic's realized score against reward()'s score_final.
Plus: μ = f(x_meas)·(1 + δ), to show how quickly μ error swamps the σ signal.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy.stats import kendalltau

from physcal.config import REPO_ROOT, load_config
from physcal.env.generator import problem_from_dict
from physcal.eval import noise_heuristics, oracle_centred as oc
from physcal.eval.x1_text_sigma import load_pairs
from physcal.verify.reward import reward, reward_config

STAR = "sigma_star"


def _config() -> dict:
    return load_config("stage_a")


# ------------------------------------------------------------------------------------------- question 1
def optimum(rows: list[dict], cfg: dict, sc: dict) -> dict:
    """argmax_c of the mean realized crps_log with σ = c·σ*, pooled and per (depth, level) cell, bootstrap over problems."""
    oc_cfg = cfg["optimum"]
    rows = [r for r in rows if not r["exact"] and r["level"] > 0]
    cs = np.exp(np.linspace(math.log(oc_cfg["c_min"]), math.log(oc_cfg["c_max"]), oc_cfg["n_grid"]))
    ln_ratio = -np.array([r["dev"] for r in rows])  # dev = ln truth − ln f(x_meas)
    star = np.array([r["sigma"] for r in rows])
    S = np.stack([oc.crps_log_vec(ln_ratio, c * star, sc) for c in cs], axis=1)  # problems × grid
    exact_c1 = [oc.verifier_crps_log(math.exp(lr), s, 1.0, sc) for lr, s in zip(ln_ratio[:500], star[:500])]
    ident = float(np.max(np.abs(np.array(exact_c1) - S[:500, np.argmin(np.abs(cs - 1.0))])))

    def best(idx: np.ndarray) -> dict:
        m = S[idx].mean(axis=0)
        boot = oc.ClusterBootstrap(list(idx), cfg["bootstrap"], cfg["seed"])
        bm = (boot.counts @ S[idx]) / len(idx)
        arg = cs[np.argmax(bm, axis=1)]
        at = {str(c): float(m[np.argmin(np.abs(cs - c))]) for c in oc_cfg["report_c"]}
        return {"n": int(len(idx)), "argmax_c": float(cs[np.argmax(m)]), "argmax_c_ci95": np.percentile(arg, [2.5, 97.5]).tolist(),
                "mean_score_at_c": at, "loss_at_half_vs_1": at["1.0"] - at["0.5"], "loss_at_2_vs_1": at["1.0"] - at["2.0"]}

    out = {"n_problems": len(rows), "grid": [float(cs[0]), float(cs[-1]), len(cs)], "vectorized_vs_scalar_max_abs": ident,
           "pooled": best(np.arange(len(rows))), "cells": {}}
    for d in sorted({r["depth"] for r in rows}):
        for lv in sorted({r["level"] for r in rows}):
            idx = np.array([i for i, r in enumerate(rows) if r["depth"] == d and r["level"] == lv])
            out["cells"][f"{d}|{lv}"] = best(idx)
    return out


# ------------------------------------------------------------------------------------------- the sources
def load_sets(cfg: dict) -> list[dict]:
    root = f"data/eval_sets/{cfg['generator_version']}"
    out = []
    for name in cfg["pair_sets"]:
        for d in load_pairs(f"{root}/{name}.jsonl"):  # checksum checked against the manifest
            out.append({**d, "_set": name})
    return out


def sources(problems: list[dict], cfg: dict) -> tuple[dict[str, np.ndarray], dict]:
    """Stated relative σ (log-space scale) per source and problem; X1 only if D99's reproduction passed."""
    a = cfg["alternatives"]
    star = np.array([p["noise"]["sigma_star_log"] for p in problems])
    src = {STAR: star}
    for c in a["scales"]:
        src[f"{c:g}x_sigma_star"] = c * star
    for h in a["heuristics"]:
        src[f"heuristic_{h}"] = np.array([noise_heuristics.sigma(p, h, a["heuristic_scope"]) or 0.0 for p in problems])
    src["constant_median_sigma_star"] = np.full(len(problems), float(np.median(star)))
    note = {"x1": "not used: no predictions"}
    rep_path, pred_path = REPO_ROOT / a["x1_reproduction"], REPO_ROOT / a["x1_predictions"]
    if rep_path.exists() and pred_path.exists():
        if json.loads(rep_path.read_text())["ok"]:
            preds = {}
            for line in pred_path.read_text().splitlines():
                r = json.loads(line)
                preds[(f"{r['set']}_pairs", r["seed"])] = r["ln_sigma_hat"]
            for fs in next(iter(preds.values())):
                src[f"x1_{fs}"] = np.array([math.exp(preds[(p["_set"], p["seed"])][fs]) for p in problems])
            note["x1"] = "used: D99's reproduction passed"
        else:
            note["x1"] = "not used: D99's reproduction failed"
    return src, note


# ------------------------------------------------------------------------------------------- questions 2-4
def resolution(scores: dict, sets: np.ndarray) -> dict:
    """σ* − alternative, per problem: mean, SD, n₂ = (2·SD / mean)²; realized and expected; pooled and per set."""
    out = {}
    for name, s in scores.items():
        if name == STAR:
            continue
        out[name] = {}
        for scope in ["pooled", *sorted(set(sets))]:
            m = np.ones(len(sets), bool) if scope == "pooled" else sets == scope
            e = {}
            for kind in ("realized", "expected"):
                d = scores[STAR][kind][m] - s[kind][m]
                d = d[np.isfinite(d)]
                mu, sd = float(d.mean()), float(d.std(ddof=1))
                e[kind] = {"n": int(d.size), "mean_diff": mu, "sd_diff": sd,
                           "n2": (2 * sd / mu) ** 2 if mu != 0 else math.inf, "share_star_better": float(np.mean(d > 0))}
            e["variance_ratio_realized_over_expected"] = (e["realized"]["sd_diff"] / e["expected"]["sd_diff"]) ** 2 \
                if e["expected"]["sd_diff"] > 0 else None
            out[name][scope] = e
    return out


def ranking(summary: dict) -> dict:
    names = list(summary)
    by_score = sorted(names, key=lambda n: -summary[n]["realized"][0])
    by_dist = sorted(names, key=lambda n: summary[n]["median_abs_log_ratio"])
    tau = kendalltau([-summary[n]["realized"][0] for n in names], [summary[n]["median_abs_log_ratio"] for n in names])
    return {"by_mean_realized_score": by_score, "by_median_abs_log_ratio": by_dist, "kendall_tau": float(tau.statistic),
            "kendall_p": float(tau.pvalue)}


def identity_check(problems: list[dict], src: dict, cfg: dict, rw_cfg: dict) -> dict:
    """The diagnostic's realized score against reward()'s score_final on a rendered answer μ ± σ in SI units."""
    ic, sc = cfg["identity_check"], rw_cfg["scoring"]
    diffs, fails = [], 0
    for name in ic["sources"]:
        seen: dict = {}
        for i, p in enumerate(problems):
            if seen.get(p["_set"], 0) >= ic["n_per_set"]:
                continue
            seen[p["_set"]] = seen.get(p["_set"], 0) + 1
            mu = p["noise"]["point_estimate_si"]
            sig = float(src[name][i]) * abs(mu)
            text = f"<steps>\n</steps>\n<answer>{mu!r} ± {sig!r} {p['target_si_unit']}</answer>"
            _, bd = reward(problem_from_dict({k: v for k, v in p.items() if k != "_set"}), text, rw_cfg)
            if bd["reason"] != "ok":
                fails += 1
                continue
            diffs.append(abs(bd["score_final"] - oc.verifier_crps_log(mu, float(src[name][i]), p["target_value_si"], sc)))
    return {"n": len(diffs), "n_parse_or_unit_fail": fails, "max_abs_diff": float(max(diffs)) if diffs else None,
            "n_exact": int(sum(d == 0.0 for d in diffs))}


def mu_error(terms: list[dict], src: dict, cfg: dict, sc: dict) -> dict:
    out = {}
    for delta in cfg["mu_error"]["deltas"]:
        row = {n: float(np.mean(oc.score_source(terms, src[n], sc, 1.0 + delta)["realized"])) for n in cfg["mu_error"]["sources"]}
        a, b = cfg["mu_error"]["sources"]
        row["difference"] = row[a] - row[b]
        out[str(delta)] = row
    return out


def run() -> dict:
    cfg = _config()
    rw_cfg = reward_config()
    sc = rw_cfg["scoring"]
    out = REPO_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    gate = [json.loads(line) for line in (REPO_ROOT / cfg["gate_rows"]).read_text().splitlines()]
    res = {"decision": "D100 (Phase 12 stage A)", "verifier": rw_cfg.get("version"), "optimum": optimum(gate, cfg, sc)}
    problems = load_sets(cfg)
    terms = [oc.problem_terms(p, cfg["mc"]["n_draws"], cfg["mc"]["seed_tag"]) for p in problems]
    src, note = sources(problems, cfg)
    sets = np.array([p["_set"] for p in problems])
    boot = oc.ClusterBootstrap([(p["_set"], p["noise"]["pair"]["pair_seed"]) for p in problems], cfg["bootstrap"], cfg["seed"])
    scores = {n: oc.score_source(terms, s, sc) for n, s in src.items()}
    summary = {n: oc.summarize(terms, s, scores[n], boot, sc, cfg["coverage_k"]) for n, s in src.items()}
    fresh = [oc_star for t in terms if t["draws"] is not None
             for oc_star in [math.log(oc.noise_mode.crps_optimal_scale(t["draws"], math.log(t["mu"])) / t["sigma_star"])]]
    per_set = {}
    for name in sorted(set(sets)):
        idx = np.flatnonzero(sets == name)
        b = oc.ClusterBootstrap([problems[i]["noise"]["pair"]["pair_seed"] for i in idx], cfg["bootstrap"], cfg["seed"])
        per_set[name] = {n: oc.summarize([terms[i] for i in idx], s[idx], {k: v[idx] for k, v in scores[n].items()}, b, sc,
                                         cfg["coverage_k"]) for n, s in src.items()}
    res.update(n_problems=len(problems), n_pairs=boot.k, sources_note=note, summary=summary, per_set=per_set,
               mc_check={"n": len(fresh), "median_abs_ln_fresh_over_stored_sigma_star": float(np.median(np.abs(fresh))),
                         "max_abs_ln_fresh_over_stored_sigma_star": float(np.max(np.abs(fresh)))},
               resolution=resolution(scores, sets), ranking=ranking(summary),
               identity_check=identity_check(problems, src, cfg, rw_cfg), mu_error=mu_error(terms, src, cfg, sc))
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "summary.md").write_text(render(res))
    return res


# ------------------------------------------------------------------------------------------- report
def _ci(t) -> str:
    return f"{t[0]:+.4f} [{t[1][0]:+.4f}, {t[1][1]:+.4f}]"


def render(res: dict) -> str:
    o = res["optimum"]
    L = ["# Phase 12 stage A: scoring audit and oracle-centred diagnostic (D100)", "",
         f"Verifier {res['verifier']} crps_log, every forecast centred at μ = f(x_meas). Descriptive, no tests.", "",
         "## 1. Optimum: σ = c·σ*, mean realized crps_log (capped gate sample)", "",
         f"{o['n_problems']} noisy problems; c on a log grid {o['grid'][0]:g}–{o['grid'][1]:g} ({o['grid'][2]} points); "
         f"vectorized vs scalar verifier score: max abs diff {o['vectorized_vs_scalar_max_abs']:.1e}.", "",
         "| cell (depth, level) | n | argmax c [95% CI] | score at c = 1 | loss at c = 0.5 | loss at c = 2 |", "|---|---|---|---|---|---|"]
    for name, b in [("pooled", o["pooled"]), *o["cells"].items()]:
        L.append(f"| {name} | {b['n']} | {b['argmax_c']:.3f} [{b['argmax_c_ci95'][0]:.3f}, {b['argmax_c_ci95'][1]:.3f}] "
                 f"| {b['mean_score_at_c']['1.0']:+.5f} | {b['loss_at_half_vs_1']:+.5f} | {b['loss_at_2_vs_1']:+.5f} |")
    L += ["", f"## Oracle-centred diagnostic, pooled A6 + A6b + A6c ({res['n_problems']} problems, {res['n_pairs']} pairs; "
          "CIs: bootstrap over pairs)", "", f"X1: {res['sources_note']['x1']}. MC check: fresh CRPS-optimal σ* vs stored, "
          f"median |ln ratio| {res['mc_check']['median_abs_ln_fresh_over_stored_sigma_star']:.4f} "
          f"(max {res['mc_check']['max_abs_ln_fresh_over_stored_sigma_star']:.4f}).", "",
          "| source | realized crps_log | expected crps_log | median ln(σ/σ*) | median abs ln(σ/σ*) | vol beta | 1σ cov. | 2σ cov. |",
          "|---|---|---|---|---|---|---|---|"]
    for n, s in res["summary"].items():
        L.append(f"| {n} | {_ci(s['realized'])} | {_ci(s['expected'])} | {s['median_log_ratio']:+.3f} | "
                 f"{s['median_abs_log_ratio']:.3f} | {s['vol_beta'][0]:+.3f} [{s['vol_beta'][1][0]:+.3f}, {s['vol_beta'][1][1]:+.3f}] "
                 f"| {s['coverage_1'][0]:.3f} | {s['coverage_2'][0]:.3f} |")
    L += ["", "## 2. Resolution: σ* − alternative, per problem (pooled)", "",
          "n₂ = (2·SD / mean)²: problems for the mean difference to sit two standard errors from 0 (problems treated as "
          "independent; members of a pair are not).", "",
          "| alternative | realized mean diff | realized SD | realized n₂ | expected mean diff | expected SD | expected n₂ "
          "| Var ratio (realized / expected) | σ* better: realized | σ* better: expected |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n, e in res["resolution"].items():
        r, x = e["pooled"]["realized"], e["pooled"]["expected"]
        vr = e["pooled"]["variance_ratio_realized_over_expected"]
        L.append(f"| {n} | {r['mean_diff']:+.5f} | {r['sd_diff']:.5f} | {r['n2']:.0f} | {x['mean_diff']:+.5f} | {x['sd_diff']:.5f} "
                 f"| {x['n2']:.0f} | {'–' if vr is None else f'{vr:.1f}'} | {r['share_star_better']:.1%} "
                 f"| {x['share_star_better']:.1%} |")
    L += ["", "## Per set (oracle-centred)", "", "| set | source | realized crps_log | expected crps_log | median abs ln(σ/σ*) "
          "| vol beta | 1σ cov. |", "|---|---|---|---|---|---|---|"]
    for name, rows in res["per_set"].items():
        for n, s in rows.items():
            L.append(f"| {name} | {n} | {s['realized'][0]:+.4f} | {s['expected'][0]:+.4f} | {s['median_abs_log_ratio']:.3f} "
                     f"| {s['vol_beta'][0]:+.3f} | {s['coverage_1'][0]:.3f} |")
    rk = res["ranking"]
    L += ["", "## 3. Ranking", "", f"- by mean realized crps_log: {', '.join(rk['by_mean_realized_score'])}",
          f"- by median abs ln(σ/σ*): {', '.join(rk['by_median_abs_log_ratio'])}",
          f"- Kendall τ = {rk['kendall_tau']:+.3f} (p = {rk['kendall_p']:.3g})", "",
          "## 4. Verifier details", "", "| source | share at the −5 clip | share where the 1e-6 floor binds | share σ = 0 |",
          "|---|---|---|---|"]
    for n, s in res["summary"].items():
        L.append(f"| {n} | {s['share_at_clip']:.4f} | {s['share_floor_binds']:.4f} | {s['share_sigma_zero']:.4f} |")
    ic = res["identity_check"]
    L += ["", f"Identity check (reward()'s score_final vs the diagnostic, rendered answers in SI): {ic['n']} answers, "
          f"{ic['n_exact']} exact, max abs diff {ic['max_abs_diff']}, {ic['n_parse_or_unit_fail']} parse/unit failures.", "",
          "## μ error against σ signal (μ = f(x_meas)·(1 + δ), pooled mean realized crps_log)", "",
          "| δ | " + " | ".join(next(iter(res["mu_error"].values()))) + " |", "|---|---|---|---|"]
    for d, row in res["mu_error"].items():
        L.append(f"| {float(d):.1%} | " + " | ".join(f"{v:+.5f}" for v in row.values()) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["run"])
    ap.parse_args()
    print(render(run()))


if __name__ == "__main__":
    main()

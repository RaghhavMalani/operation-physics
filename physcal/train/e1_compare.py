"""E1 family-E tests (DECISIONS.md D52), CPU only, after the E1 evals exist.

    uv run python -m physcal.train.e1 compare

On the frozen eval prefixes (SFT-only completions with a ± in their answer), paired by problem, so μ is the same
in every arm:
- E1a-P / E1b-P: mean crps_log of the E1 σ > the SFT stated σ;
- E1a-N / E1b-N: mean crps_log of the E1 σ > the strongest A5 σ baseline on the same completions.
One-sided paired bootstrap over problems; Holm over all four (a run that does not exist counts as p = 1, so the
correction never gets weaker than pre-registered). Writes results/e1/family_E.{json,md}.

`compare_pilots()` (DECISIONS.md D94, R3-3): E1c / E1d against E1a on ρ(σ, |error|), the corrected eval, one-sided
paired bootstrap over problems, Holm over the two; plus each pilot's family E and the collapse check. Writes
results/e1/pilots/pilots.{json,md}.

`compare(corrected=True)` (DECISIONS.md D67): the same tests on the corrected evals. The E1 adapters are applied
unmerged to the 4-bit training base (e1_eval/final_4bit), and the SFT stated σ is the SFT-only model's own σ suffix
on the same prefixes and base (results/e1/sft_4bit). The A5 σ (on the fixed prefixes) is unchanged. Writes
results/corrected/e1/family_E.{json,md}.

    uv run python -c "from physcal.train.e1_compare import compare; print(compare(corrected=True))"
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from physcal.config import REPO_ROOT, load_config
from physcal.eval.stats import holm
from physcal.tape.tape import read_tape
from physcal.train.e1 import bin_entropy_bits, read_rows


def tape_arm(path: Path) -> dict[tuple, dict]:
    """Per problem: crps_log, coverage and relative σ of a (derived) eval tape; failures score the floor."""
    out = {}
    for r in read_tape(path):
        p, b = r["problem"], r["breakdown"]
        ok = b["reason"] == "ok"
        out[(p["family"], p["depth"], p["seed"])] = {
            "crps_log": b["scores"]["crps_log"] if ok else b["score_final"], "fail": not ok,
            "cov1": ok and bool(b["within_1sigma"]), "cov2": ok and bool(b["within_2sigma"]),
            "rel_sigma": b["sigma_si"] / abs(b["mu_si"]) if ok else None,
            "abs_log_error": abs(math.log(b["mu_si"] / b["truth_si"])) if ok and b["mu_si"] > 0 else None}
    return out


def rows_arm(path: Path) -> dict[tuple, dict]:
    out = {}
    for line in path.open():
        r = json.loads(line)
        out[tuple(r["key"])] = {"crps_log": r["crps_log"], "fail": r["fail"], "cov1": r["cov1"], "cov2": r["cov2"],
                                "rel_sigma": r["rel_sigma_used"], "abs_log_error": r["abs_log_error"],
                                "mu_changed": r["mu_changed"]}
    return out


def describe(arm: dict, keys: list, bins: list[float], target: float) -> dict:
    ok = [arm[k] for k in keys if not arm[k]["fail"] and arm[k]["rel_sigma"] is not None
          and arm[k]["abs_log_error"] is not None]  # μ <= 0 answers: crps_log already gives them the clip
    rho = spearmanr([x["rel_sigma"] for x in ok], [x["abs_log_error"] for x in ok]).statistic if len(ok) > 2 else None
    cov1 = float(np.mean([arm[k]["cov1"] for k in keys]))
    return {"n": len(keys), "n_failed": sum(arm[k]["fail"] for k in keys),
            "mu_changed": sum(arm[k].get("mu_changed", False) for k in keys),
            "mean_crps_log": float(np.mean([arm[k]["crps_log"] for k in keys])),
            "coverage_1sigma": cov1, "coverage_1sigma_error": abs(cov1 - target),
            "coverage_2sigma": float(np.mean([arm[k]["cov2"] for k in keys])),
            "spearman_sigma_error": None if rho is None else float(rho),
            "median_rel_sigma": float(np.median([x["rel_sigma"] for x in ok])) if ok else None,
            "sigma_zero_share": float(np.mean([x["rel_sigma"] == 0 for x in ok])) if ok else None,
            "bin_entropy_bits": bin_entropy_bits([x["rel_sigma"] for x in ok], bins)}


def paired_bootstrap(a: dict, b: dict, keys: list, n_boot: int, seed: int) -> dict:
    """One-sided: mean crps_log of a > that of b. p = share of resamples with a − b ≤ 0."""
    d = np.array([a[k]["crps_log"] - b[k]["crps_log"] for k in keys])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return {"mean_diff": float(d.mean()), "ci95": np.percentile(boots, [2.5, 97.5]).tolist(),
            "p_one_sided": float(np.mean(boots <= 0))}


CORRECTED = {"eval_name": "final_4bit", "sft_rows": "results/e1/sft_4bit/rows.jsonl", "output_dir": "results/corrected/e1",
             "title": " — corrected evals: 4-bit training base, adapters unmerged (D67)"}


def compare(corrected: bool = False, a5_tape: str | None = None, output_dir: str | None = None) -> str:
    """`a5_tape` / `output_dir`: D69's sensitivity rerun with the A5 re-fitted on 4-bit SFT (defaults: as before)."""
    e1 = load_config("e1")
    cc = e1["compare"]
    keys = [tuple(r["key"]) for r in read_rows("eval")]
    eval_name, title = (CORRECTED["eval_name"], CORRECTED["title"]) if corrected else ("final", "")
    sft = rows_arm(REPO_ROOT / CORRECTED["sft_rows"]) if corrected else tape_arm(REPO_ROOT / e1["eval_tapes"][0])
    a5 = tape_arm(REPO_ROOT / (a5_tape or cc["a5_sigma_tape"]))
    if a5_tape:
        title += " — A5 re-fitted on 4-bit SFT completions (D69 sensitivity)"
    arms = {"SFT stated σ": sft, "A5 strongest σ": a5}
    tests, p = {}, {}
    for name, run in cc["runs"].items():
        path = REPO_ROOT / run / "e1_eval" / eval_name / "rows.jsonl"
        if not path.exists():
            p[f"{name}-P"] = p[f"{name}-N"] = 1.0
            tests[name] = "not run"
            continue
        arm = rows_arm(path)
        missing = [k for k in keys if k not in arm]
        if missing:
            raise ValueError(f"{name}: {len(missing)} eval prefixes without an E1 row")
        arms[name] = arm
        tests[f"{name}-P"] = paired_bootstrap(arm, sft, keys, cc["bootstrap"], cc["seed"])
        tests[f"{name}-N"] = paired_bootstrap(arm, a5, keys, cc["bootstrap"], cc["seed"])
        p[f"{name}-P"], p[f"{name}-N"] = tests[f"{name}-P"]["p_one_sided"], tests[f"{name}-N"]["p_one_sided"]
    h = holm(p, cc["alpha"])
    desc = {name: describe(arm, keys, e1["bins"], cc["coverage_target"]) for name, arm in arms.items()}
    sft_binned = {k: {**v, "rel_sigma": None if v["rel_sigma"] is None else
                      min(e1["bins"], key=lambda b: abs(math.log(b) - math.log(max(v["rel_sigma"], 1e-300))))}
                  for k, v in sft.items()}
    desc["SFT stated σ, binned (descriptive)"] = {"bin_entropy_bits": describe(sft_binned, keys, e1["bins"],
                                                                              cc["coverage_target"])["bin_entropy_bits"]}
    out = {"decision": "D52", "n_problems": len(keys), "tests": tests, "holm": h, "describe": desc,
           "corrected_D67": corrected, "eval_name": eval_name, "a5_sigma_tape": a5_tape or cc["a5_sigma_tape"]}
    od = REPO_ROOT / (output_dir or (CORRECTED["output_dir"] if corrected else cc["output_dir"]))
    od.mkdir(parents=True, exist_ok=True)
    (od / "family_E.json").write_text(json.dumps(out, indent=1, default=float))
    lines = [f"# E1 family E (D52), {len(keys)} eval prefixes{title}", "",
             "| test | mean Δcrps_log | 95% CI | p (one-sided) | Holm threshold | supported |", "|---|---|---|---|---|---|"]
    for name in p:
        t = tests.get(name)
        if isinstance(t, dict):
            lines.append(f"| {name} | {t['mean_diff']:+.4f} | [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] | {h[name]['p']:.3g} "
                         f"| {h[name]['threshold']:.3g} | {h[name]['rejected']} |")
        else:
            lines.append(f"| {name} | – | – | not run | {h[name]['threshold']:.3g} | False |")
    lines += ["", "| σ source | mean crps_log | 1σ cov | 2σ cov | ρ(σ, err) | median rel σ | σ = 0 | bin entropy | failed | μ changed |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, d in desc.items():
        if "mean_crps_log" not in d:
            continue
        f = lambda x, pct=False: "–" if x is None else (f"{100 * x:.1f}%" if pct else f"{x:.4g}")
        lines.append(f"| {name} | {d['mean_crps_log']:.4f} | {f(d['coverage_1sigma'], True)} | {f(d['coverage_2sigma'], True)} "
                     f"| {f(d['spearman_sigma_error'])} | {f(d['median_rel_sigma'])} | {f(d['sigma_zero_share'], True)} "
                     f"| {f(d['bin_entropy_bits'])} | {d['n_failed']} | {d['mu_changed']} |")
    md = "\n".join(lines) + "\n"
    (od / "family_E.md").write_text(md)
    return md


# ------------------------------------------------------------------------------------------- D94: the R3-3 pilots
def _ok(arm: dict, keys: list) -> list:
    return [k for k in keys if not arm[k]["fail"] and arm[k]["rel_sigma"] is not None
            and arm[k]["abs_log_error"] is not None]  # describe()'s rows


def rho_of(arm: dict, keys: list) -> float | None:
    ok = _ok(arm, keys)
    if len(ok) <= 2:
        return None
    r = spearmanr([arm[k]["rel_sigma"] for k in ok], [arm[k]["abs_log_error"] for k in ok]).statistic
    return None if r is None or math.isnan(r) else float(r)


def rho_bootstrap(a: dict, b: dict, keys: list, n_boot: int, seed: int) -> dict:
    """D94: one-sided ρ(a) > ρ(b); each resample of problems recomputes both ρ on the same problems. p = the share
    of resamples with ρ(a) − ρ(b) <= 0 (a resample where either ρ is undefined counts against a)."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        ks = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        ra, rb = rho_of(a, ks), rho_of(b, ks)
        diffs.append(ra - rb if ra is not None and rb is not None else -math.inf)
    d = np.array(diffs)
    fin = d[np.isfinite(d)]
    return {"rho": rho_of(a, keys), "rho_reference": rho_of(b, keys), "diff": rho_of(a, keys) - rho_of(b, keys),
            "ci95": np.percentile(fin, [2.5, 97.5]).tolist() if len(fin) else [None, None],
            "p_one_sided": float(np.mean(d <= 0))}


def collapse(tape: Path, share: float) -> dict:
    """D94: the most common σ suffix string (stripped) and its share of the eval suffixes."""
    from collections import Counter

    sfx = [r.get("suffix", "").strip() for r in read_tape(tape)]
    top, n = Counter(sfx).most_common(1)[0] if sfx else ("", 0)
    return {"top_suffix": top, "top_share": n / len(sfx) if sfx else None, "collapsed": bool(sfx) and n / len(sfx) > share}


def compare_pilots() -> dict:
    e1 = load_config("e1")
    cc = e1["compare"]
    pc = cc["pilots"]
    keys = [tuple(r["key"]) for r in read_rows("eval")]
    ref = rows_arm(REPO_ROOT / pc["reference"] / "e1_eval" / pc["eval_name"] / "rows.jsonl")
    sft = rows_arm(REPO_ROOT / CORRECTED["sft_rows"])
    a5 = tape_arm(REPO_ROOT / cc["a5_sigma_tape"])
    tests, p, desc, fam, col = {}, {}, {}, {}, {}
    for name, run in pc["runs"].items():
        d = REPO_ROOT / run / "e1_eval" / pc["eval_name"]
        if not (d / "rows.jsonl").exists():
            p[name], tests[name] = 1.0, "not run"
            continue
        arm = rows_arm(d / "rows.jsonl")
        missing = [k for k in keys if k not in arm]
        if missing:
            raise ValueError(f"{name}: {len(missing)} eval prefixes without a row")
        tests[name] = rho_bootstrap(arm, ref, keys, cc["bootstrap"], cc["seed"])
        p[name] = tests[name]["p_one_sided"]
        desc[name] = describe(arm, keys, e1["bins"], cc["coverage_target"])
        fam[name] = {"P (vs SFT σ)": paired_bootstrap(arm, sft, keys, cc["bootstrap"], cc["seed"]),
                     "N (vs A5 σ)": paired_bootstrap(arm, a5, keys, cc["bootstrap"], cc["seed"])}
        col[name] = collapse(d / "rollouts.jsonl", pc["collapse_share"])
    desc["E1a (reference)"] = describe(ref, keys, e1["bins"], cc["coverage_target"])
    h = holm(p, cc["alpha"])
    sup = [n for n in p if h[n]["rejected"]]
    if not sup:
        rule = "neither moves ρ: Phase 12 RL keeps E1's settings (reported as a negative result)"
    else:
        best = max(sup, key=lambda n: tests[n]["rho"])
        rule = f"{best} moves ρ: its setting becomes Phase 12's RL setting" + (
            f" (both supported; {best} has the higher ρ)" if len(sup) > 1 else "")
    out = {"decision": "D94", "n_problems": len(keys), "tests": tests, "holm": h, "describe": desc,
           "family_E": fam, "collapse": col, "decision_rule": rule}
    od = REPO_ROOT / pc["output_dir"]
    od.mkdir(parents=True, exist_ok=True)
    (od / "pilots.json").write_text(json.dumps(out, indent=1, default=float))
    L = [f"# E1c / E1d, the R3-3 pilots (D94): ρ(σ, |error|) vs E1a, {len(keys)} eval prefixes, corrected eval", "",
         f"**Decision rule:** {rule}.", "",
         "| pilot | ρ | ρ E1a | Δρ [95% CI] | p (one-sided) | Holm | collapsed (top suffix share) |", "|---|---|---|---|---|---|---|"]
    for name in p:
        t = tests[name]
        if t == "not run":
            L.append(f"| {name} | not run | – | – | 1 | not supported | – |")
            continue
        c = col[name]
        L.append(f"| {name} | {t['rho']:.3f} | {t['rho_reference']:.3f} | {t['diff']:+.3f} [{t['ci95'][0]:+.3f}, "
                 f"{t['ci95'][1]:+.3f}] | {t['p_one_sided']:.3g} | {'supported' if h[name]['rejected'] else 'not supported'} "
                 f"| {c['collapsed']} ({c['top_share']:.1%}) |")
    (od / "pilots.md").write_text("\n".join(L) + "\n")
    return out

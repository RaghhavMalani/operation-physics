"""R4-4 (DECISIONS.md D112; pre-registered in D113 §7): MC-A, Monte Carlo over the inputs with the model as the
function, a diagnostic (inference only).

    uv run python -m physcal.eval.mc_a run --set a6c_pairs        # GPU: k perturbed runs per problem (resumable)
    OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.eval.mc_a analyze --set a6c_pairs   # CPU

For run j = 1..k and each noisy given i: z_ij ~ N(0, 1) from SHA-256 of `mc-a|set|family|depth|seed|j`, and
x_ij = x_meas,i / (1 + s_i·z_ij) (the generator's predictive draw), shown in the given's unit at the noise mode's
significant figures; the problem text is the original with each noisy given's number replaced ("± p%" stays; each
replaced string must occur once, else the problem is excluded). Exact givens and distractors are unchanged. Run 0
is the unperturbed problem's completion from the model's own eval tape. The model is B-v2 (stage R4 `b2`, seed 0)
with its eval settings, or stage B seed 0 if B-v2 has no adapter.

σ per problem (log space, used as σ_rel), from the first k runs: raw = SD (ddof 1) of ln μ_j over the valid runs;
robust = 1.4826 × MAD of ln μ_j over the valid runs whose step-name sequence equals run 0's; oracle = SD of
ln f(x_j), the exact chain on the same displayed perturbations (also its MAD). Fewer than `min_runs` -> 0, the floor.
The stopping rule: robust (k = 8) beats RSS on A6c iff the mean oracle-centred expected crps_log difference has a
one-sided cluster-bootstrap p < 0.05 (pairs); otherwise k is not increased.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config

STEP_NAME_RE = re.compile(r"^[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*=[ \t]", re.M)
VARIANTS = ("raw", "robust", "oracle", "oracle_mad")


def _config() -> dict:
    return load_config("mc_a")


def set_dir(set_name: str, cfg: dict | None = None) -> Path:
    return REPO_ROOT / (cfg or _config())["output_dir"] / set_name


# ------------------------------------------------------------------------------------------- perturbations
def z_draws(problem: dict, set_name: str, j: int, n: int, tag: str) -> np.ndarray:
    key = f"{tag}|{set_name}|{problem['family']}|{problem['depth']}|{problem['seed']}|{j}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:16], "little")).standard_normal(n)


def _shown_text(display: str, unit: str, rel: float) -> str:
    """render.ShownValue.text for a measured value."""
    return f"{f'{display} {unit}'.strip()} ± {100 * rel:g}%"


def perturb(problem: dict, set_name: str, j: int, cfg: dict, env_cfg: dict) -> dict | None:
    """Run j's text, displayed values and the exact chain's ln f at them; None if a replaced string is not unique or
    a draw leaves the domain (1 + s·z <= 0)."""
    from physcal.env import noise as noise_mode
    from physcal.env import render
    from physcal.env.formulas import get_family
    from physcal.env.generator import chain_of
    from physcal.env.units import from_si, to_si

    gen = env_cfg["generator"]
    fam = get_family(problem["family"])
    noisy = [g for g in problem["givens"] if g.get("noise_rel")]
    z = z_draws(problem, set_name, j, len(noisy), cfg["seed_tag"])
    text, si, shown = problem["text"], {g["symbol"]: g["value_si"] for g in problem["givens"]}, {}
    for g, zi in zip(noisy, z):
        denom = 1.0 + g["noise_rel"] * float(zi)
        if denom <= 0:
            return None
        q = fam.quantities[g["symbol"]]
        disp = render.format_number(from_si(g["value_si"] / denom, q.si_unit, g["unit"]), gen["noise"]["sig_figs"],
                                    tuple(gen["display"]["fixed_exponent_range"]))
        old = _shown_text(g["display"], g["unit"], g["noise_rel"])
        if text.count(old) != 1:
            return None
        text = text.replace(old, _shown_text(disp, g["unit"], g["noise_rel"]))
        si[g["symbol"]], shown[g["symbol"]] = to_si(float(disp), g["unit"]), disp
    y = float(noise_mode.chain_target(fam, chain_of(problem), si))
    if not (math.isfinite(y) and y > 0):
        return None
    return {"j": j, "text": text, "displays": shown, "ln_oracle": math.log(y)}


def step_names(completion: str) -> tuple[str, ...]:
    """The left-hand names of the lines before <steps> that read `name = ...` (the chain's computation lines; the
    `SI values:` / `Not needed:` / `In <unit>:` lines do not match)."""
    return tuple(STEP_NAME_RE.findall(completion.split("<steps>", 1)[0]))


# ------------------------------------------------------------------------------------------- the model
def model_choice(cfg: dict | None = None) -> dict:
    """B-v2's eval config and tape, or stage B seed 0's if B-v2 has no adapter (D113 §7)."""
    from physcal.train import stage_b, stage_r4

    cfg = cfg or _config()
    ev = stage_r4.eval_config(cfg["model"]["arm"])
    label = f"stage R4 {cfg['model']['arm']} (B-v2), seed 0"
    if not (Path(ev["model"]["lora"]) / "adapter_model.safetensors").exists():
        s = cfg["model"]["fallback_stage_b_seed"]
        ev = stage_b.eval_config(s)
        ev["generation"]["max_new_tokens"] = stage_r4.token_budget()["max_new_tokens"]
        ev["model"]["max_reserved_gib"] = stage_r4._config()["eval"]["max_reserved_gib"]
        ev["generation"]["batch_size"] = stage_r4._config()["eval"]["batch_size"]  # D116
        label = f"stage B seed {s} (fallback: B-v2 has no adapter)"
    return {"eval": ev, "label": label, "tape_dir": str(Path(ev["output_dir"]).relative_to(REPO_ROOT))}


def problems_of(set_name: str) -> list[dict]:
    from physcal.eval import stage_a

    return [p for p in stage_a.load_sets(load_config("stage_a")) if p["_set"] == set_name]


def run(set_name: str, limit: int | None = None) -> dict:
    """GPU: every (problem, j) not yet in runs.jsonl, in chunks; each chunk is appended as it finishes."""
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    from physcal.env.generator import problem_from_dict
    from physcal.eval.run_eval import HFBackend
    from physcal.prompt import build_messages
    from physcal.verify.reward import reward, reward_config

    cfg = _config()
    env_cfg = load_config("env_v2")
    choice = model_choice(cfg)
    ev = choice["eval"]
    out = set_dir(set_name, cfg)
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.jsonl"
    done = {(r["seed"], r["j"]) for r in map(json.loads, runs_path.open())} if runs_path.exists() else set()
    probs = problems_of(set_name)[:limit]
    todo, excluded = [], []
    for p in probs:
        per = [perturb(p, set_name, j, cfg, env_cfg) for j in range(1, cfg["k"] + 1)]
        if any(x is None for x in per):
            excluded.append(p["seed"])
            continue
        todo += [(p, x) for x in per if (p["seed"], x["j"]) not in done]
    t0 = time.time()
    backend = HFBackend(ev["model"]["id"], ev["model"], ev["generation"])
    rw_cfg = reward_config()
    meta = {"decision": "D113 §7 (R4-4: MC-A)", "set": set_name, "model": choice["label"], "lora": ev["model"]["lora"],
            "lora_hash": backend.lora_hash, "provenance": backend.provenance, "generation": ev["generation"],
            "prompt": ev["prompt"], "run0_tape": choice["tape_dir"], "k": cfg["k"], "n_problems": len(probs),
            "excluded_problems": excluded, "already_done": len(done), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    step = cfg["chunk_problems"] * cfg["k"]
    with runs_path.open("a") as fh:
        for c in range(0, len(todo), step):
            chunk = todo[c:c + step]
            gens = backend([build_messages(x["text"], ev["prompt"]) for _, x in chunk])
            for (p, x), g in zip(chunk, gens):
                _, bd = reward(problem_from_dict({k: v for k, v in p.items() if k != "_set"}), g.text, rw_cfg)
                fh.write(json.dumps({"seed": p["seed"], "family": p["family"], "depth": p["depth"], "j": x["j"],
                                     "displays": x["displays"], "ln_oracle": x["ln_oracle"], "completion": g.text,
                                     "n_tokens": g.n_tokens, "truncated": g.truncated, "reason": bd["reason"],
                                     "mu_si": bd.get("mu_si")}) + "\n")
            fh.flush()
            print(f"  mc-a {set_name}: {c + len(chunk)}/{len(todo)} runs ({time.time() - t0:.0f}s)", flush=True)
    meta.update(seconds=time.time() - t0, n_runs_new=len(todo), finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


# ------------------------------------------------------------------------------------------- estimators
def sd(v: list[float]) -> float | None:
    return float(np.std(v, ddof=1)) if len(v) >= 2 else None


def mad(v: list[float], scale: float) -> float | None:
    if len(v) < 2:
        return None
    a = np.asarray(v)
    return float(scale * np.median(np.abs(a - np.median(a))))


def estimates(run0: dict | None, runs: list[dict], k: int, cfg: dict) -> dict:
    """The four σ variants from the first k runs (None = undefined -> the floor), and the drop bookkeeping."""
    rs = [r for r in runs if r["j"] <= k]
    valid = [r for r in rs if r["reason"] == "ok" and (r["mu_si"] or 0) > 0]
    ok0 = run0 is not None and run0["breakdown"]["reason"] == "ok" and (run0["breakdown"].get("mu_si") or 0) > 0
    names0 = step_names(run0["completion"]) if run0 is not None else None
    kept = [r for r in valid if ok0 and step_names(r["completion"]) == names0]
    m = cfg["min_runs"]
    ln = lambda xs: [math.log(r["mu_si"]) for r in xs]
    return {"raw": sd(ln(valid)) if len(valid) >= m else None,
            "robust": mad(ln(kept), cfg["mad_scale"]) if ok0 and len(kept) >= m else None,
            "oracle": sd([r["ln_oracle"] for r in rs]), "oracle_mad": mad([r["ln_oracle"] for r in rs], cfg["mad_scale"]),
            "n": len(rs), "n_invalid": len(rs) - len(valid), "n_filtered": len(valid) - len(kept) if ok0 else None,
            "run0_valid": ok0}


def auroc(score: np.ndarray, positive: np.ndarray) -> float | None:
    """P(score of a positive > score of a negative), ties half (Mann-Whitney)."""
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return None
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ------------------------------------------------------------------------------------------- the analysis
def analyze(set_name: str, cfg: dict | None = None) -> dict:
    from physcal.eval import a6_pairs, oracle_centred as oc
    from physcal.eval import r4_analysis as R
    from physcal.eval import stage_a
    from physcal.eval import stage_b_analysis as B
    from physcal.eval import stage_d_analysis as Dn
    from physcal.train import sft_r4
    from physcal.verify.reward import reward_config

    cfg = cfg or _config()
    ac = cfg["analysis"]
    out = set_dir(set_name, cfg)
    meta = json.loads((out / "meta.json").read_text())
    sa, bcfg, pc = load_config("stage_a"), load_config("stage_b")["analysis"], load_config("pairs")
    sc = reward_config()["scoring"]
    floor, tau = pc["analysis"]["sigma_floor_rel"], pc["analysis"]["tau"]
    allp = stage_a.load_sets(sa)
    src, _ = stage_a.sources(allp, sa)
    idx = [i for i, p in enumerate(allp) if p["_set"] == set_name]
    probs = [allp[i] for i in idx]
    runs: dict = {}
    raw = out / "runs.jsonl"
    with (open(raw) if raw.exists() else gzip.open(out / "runs.jsonl.gz", "rt")) as fh:
        for r in map(json.loads, fh):
            runs.setdefault(r["seed"], {})[r["j"]] = r
    tape = {(r["split"], *B._key(r["problem"])): r for r in B.read_records(Path(REPO_ROOT / meta["run0_tape"]))}
    complete = np.array([len(runs.get(p["seed"], {})) == cfg["k"] for p in probs])
    pair_ok: dict = {}
    for p, c in zip(probs, complete):
        pair_ok.setdefault(p["noise"]["pair"]["pair_seed"], []).append(bool(c))
    use = np.array([all(pair_ok[p["noise"]["pair"]["pair_seed"]]) and len(pair_ok[p["noise"]["pair"]["pair_seed"]]) == 2
                    for p in probs])
    sub = [p for p, u in zip(probs, use) if u]
    terms = [oc.problem_terms(p, sa["mc"]["n_draws"], sa["mc"]["seed_tag"]) for p in sub]
    has = np.array([t["draws"] is not None for t in terms])
    boot = oc.ClusterBootstrap([(set_name, p["noise"]["pair"]["pair_seed"]) for p in sub], ac["bootstrap"], ac["seed"])
    types = R.pair_types(sub, R.references(sub, sft_r4._config()))
    rac = {"bootstrap": ac["bootstrap"], "seed": ac["seed"], "criterion_ci_level": ac["criterion_ci_level"]}
    run0 = [tape.get((set_name, *B._key(p))) for p in sub]
    cc0 = np.array([r is not None and r["breakdown"]["reason"] == "ok" and (r["breakdown"].get("mu_si") or 0) > 0 and
                    abs(math.log(r["breakdown"]["mu_si"] / p["noise"]["point_estimate_si"])) <= tau
                    for r, p in zip(run0, sub)])

    def score(sigma: np.ndarray) -> dict:
        scores = oc.score_source(terms, sigma, sc)
        pairs = a6_pairs.pairs_from_rows(B.source_rows(sub, sigma, floor)[set_name])
        reg = "truth" if set_name == "a6b_pairs" else "sigma_star"
        mu0 = [r["breakdown"]["mu_si"] if r is not None and r["breakdown"]["reason"] == "ok" and
               (r["breakdown"].get("mu_si") or 0) > 0 else None for r in run0]
        e2e = np.array([-sc["crps_clip"] if m is None else oc.verifier_crps_log(m, float(s), p["target_value_si"], sc)
                        for m, s, p in zip(mu0, sigma, sub)])
        return {"oracle_centred": oc.summarize(terms, sigma, scores, boot, sc, sa["coverage_k"]),
                "slope": R.slope(pairs, rac, reg), "splits": R.splits(pairs, types, rac),
                "aleatoric_expected_on_run0_correct": boot.mean(np.nan_to_num(scores["expected"]), has & cc0),
                "epistemic_auroc_run0_wrong": auroc(sigma, ~cc0),
                "end_to_end_realized": boot.mean(e2e), "_expected": scores["expected"]}

    res: dict = {"decision": "D113 §7 (R4-4: MC-A, diagnostic)", "set": set_name, "model": meta["model"],
                 "n_problems": len(probs), "n_excluded": len(meta["excluded_problems"]),
                 "n_complete": int(complete.sum()), "n_used": len(sub), "run0_computation_correct": float(cc0.mean()),
                 "variants": {}, "comparators": {}}
    est = {k: [estimates(r0, list(runs[p["seed"]].values()), k, cfg) for r0, p in zip(run0, sub)] for k in cfg["k_curve"]}
    for k in cfg["k_curve"]:
        for v in VARIANTS:
            sigma = np.array([e[v] or 0.0 for e in est[k]])
            res["variants"][f"{v}_k{k}"] = {**score(sigma), "share_undefined": float(np.mean([e[v] is None for e in est[k]]))}
    k = max(cfg["k_curve"])
    e8 = est[k]
    res["drops"] = {"perturbed_runs": int(sum(e["n"] for e in e8)),
                    "invalid_share": float(sum(e["n_invalid"] for e in e8) / sum(e["n"] for e in e8)),
                    "filtered_share": float(sum(e["n_filtered"] or 0 for e in e8) / sum(e["n"] for e in e8)),
                    "run0_invalid_share": float(np.mean([not e["run0_valid"] for e in e8])),
                    "truncated_share": float(np.mean([r["truncated"] for p in sub for r in runs[p["seed"]].values()]))}
    for n, s in (("sigma_star", src[stage_a.STAR]), ("rss", src[bcfg["rss_source"]]), ("x1", src[bcfg["x1_source"]])):
        sel = np.array([s[i] for i, u in zip(idx, use) if u])
        res["comparators"][n] = score(sel)
    rule = Dn.paired(boot, res["variants"][f"robust_k{k}"]["_expected"], res["comparators"]["rss"]["_expected"], has)
    res["stopping_rule"] = {"robust_minus_rss": rule, "applies": set_name == "a6c_pairs",
                            "beats_rss": bool(not rule.get("missing") and rule["p_one_sided"] < ac["alpha"]),
                            "rule": "robust MC-A (k = 8) beats RSS on A6c iff the one-sided p < 0.05; otherwise k is "
                                    "not increased (R4-4)"}
    res["robust_minus_x1"] = Dn.paired(boot, res["variants"][f"robust_k{k}"]["_expected"],
                                       res["comparators"]["x1"]["_expected"], has)
    res["oracle_minus_rss"] = Dn.paired(boot, res["variants"][f"oracle_k{k}"]["_expected"],
                                        res["comparators"]["rss"]["_expected"], has)
    for g in list(res["variants"].values()) + list(res["comparators"].values()):
        g.pop("_expected")
    (out / "results.json").write_text(json.dumps(res, indent=1, default=B._json_default))
    (out / "summary.md").write_text(render(res))
    return res


def render(res: dict) -> str:
    sl = lambda d: "–" if not d or d.get("slope") is None else f"{d['slope']:+.3f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"
    f = lambda v: "–" if v is None else f"{v:.3f}"
    sr = res["stopping_rule"]
    rr = sr["robust_minus_rss"]
    L = [f"# MC-A on {res['set']} (R4-4, D113 §7; a diagnostic)", "",
         f"Model: {res['model']}. Problems: {res['n_problems']} ({res['n_excluded']} excluded, {res['n_complete']} with "
         f"all runs; {res['n_used']} in complete pairs). Run 0 computation-correct: {res['run0_computation_correct']:.1%}.",
         "", f"**Stopping rule** ({'applies' if sr['applies'] else 'A6c only; descriptive here'}): robust (k = 8) − RSS, "
         "expected crps_log = " + ("missing" if rr.get("missing") else
                                   f"{rr['value']:+.4f} [{rr['ci95'][0]:+.4f}, {rr['ci95'][1]:+.4f}], one-sided p "
                                   f"{rr['p_one_sided']:.4g}") +
         f": **{'beats RSS' if sr['beats_rss'] else 'does not beat RSS; k is not increased'}**.", "",
         f"Drops (k = 8): invalid {res['drops']['invalid_share']:.1%} of perturbed runs, filtered by the step-name rule "
         f"{res['drops']['filtered_share']:.1%}, run 0 invalid {res['drops']['run0_invalid_share']:.1%}, truncated "
         f"{res['drops']['truncated_share']:.1%}.", "",
         "| σ source | expected crps_log | median ln(σ/σ*) | 1σ / 2σ cov. | slope [90% CI] | undefined | aleatoric "
         "(run 0 correct) | epistemic AUROC | end-to-end |", "|---|---|---|---|---|---|---|---|---|"]
    rows = list(res["variants"].items()) + list(res["comparators"].items())
    for name, g in rows:
        o = g["oracle_centred"]
        L.append(f"| {name} | {o['expected'][0]:.4f} | {o['median_log_ratio']:+.3f} | {o['coverage_1'][0]:.1%} / "
                 f"{o['coverage_2'][0]:.1%} | {sl(g['slope'])} | {g.get('share_undefined', 0):.1%} | "
                 f"{g['aleatoric_expected_on_run0_correct'][0]:.4f} | {f(g['epistemic_auroc_run0_wrong'])} | "
                 f"{g['end_to_end_realized'][0]:.4f} |")
    L += ["", "**Slope by depth and chain type:**", "", "| σ source | d1 | d2 | d3 | d4 | d5 | power-law | value-dependent |",
          "|---|---|---|---|---|---|---|---|"]
    for name, g in rows:
        s = g["splits"]
        L.append(f"| {name} | " + " | ".join(sl(s["by_depth"].get(str(d))) for d in range(1, 6)) +
                 f" | {sl(s['by_type']['power_law'])} | {sl(s['by_type']['value_dependent'])} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["run", "analyze"])
    ap.add_argument("--set", required=True, choices=("a6c_pairs", "a6_pairs", "a6b_pairs"))
    ap.add_argument("--limit", type=int, default=None, help="run: the first N problems only (smoke tests)")
    args = ap.parse_args()
    if args.action == "run":
        print(json.dumps({k: v for k, v in run(args.set, args.limit).items() if k != "provenance"}, indent=1, default=str))
    else:
        print(render(analyze(args.set)))


if __name__ == "__main__":
    main()

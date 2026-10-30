"""R4-7 (DECISIONS.md D112; pre-registered in D119): MC-B, the model writes the problem's program and a sandboxed
interpreter runs it on MC-A's perturbations. Inference only, development sets.

    uv run python -m physcal.eval.mc_b prompt                                   # CPU: the few-shot prompt + its length
    uv run python -m physcal.eval.mc_b generate --set a6c_pairs                 # GPU: one greedy program per problem
    OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.eval.mc_b execute --set a6c_pairs   # CPU: the sandbox
    OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.eval.mc_b analyze --set a6c_pairs   # CPU: D119 §6-§8

The model is the pinned 4-bit base (Qwen2.5-Math-1.5B-Instruct, D78) without adapter, prompted with a system message
and two worked examples whose programs the builder writes (`reference_program`). The program is a function
`solve(<given symbols>)` taking the displayed values in their displayed units and returning the target in the
requested unit. Run 0 calls it with the displayed values; runs 1..k with MC-A's perturbed displayed values (same z,
`mc_a.perturb`). σ (log space) = SD of ln y_j over the k perturbed runs (`mc_b`), or 1.4826 × MAD (`mc_b_mad`);
μ = y_0. A program fails if extraction, the static check, the signature or any of its k + 1 calls fails; a failed
program has σ = 0 (the floor) and no μ, and every problem stays in every denominator.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.eval import sandbox

STEPS_SENTENCE_RE = re.compile(r"[ \t]*[^.\n]*<steps>[^.\n]*\.")
VARIANTS = ("mc_b", "mc_b_mad")
TESTS = ("M1", "M2", "M3", "M4", "M5")
TEST_NAMES = {"M1": "MC-B − RSS > 0", "M2": "MC-B − X1 > 0", "M3": "MC-B − D-pos-v2 (a) > 0",
              "M4": "MC-B − MC-A robust > 0", "M5": "MC-B's {set} slope > 0"}  # {set}: the set it is computed on (D146)


def _config() -> dict:
    return load_config("mc_b")


def set_dir(set_name: str, cfg: dict | None = None) -> Path:
    return REPO_ROOT / (cfg or _config())["output_dir"] / set_name


# ------------------------------------------------------------------------------------------- the builder's program
def _factor(unit: str) -> float:
    from physcal.env.units import to_si

    return to_si(1.0, unit) if unit else 1.0


def reference_program(problem: dict) -> str:
    """The exact program for a problem: the chain's givens (problem order) converted to SI by their pint factors, the
    family's constants it uses, one line per step (sympy's pycode of the solved expression), the target divided by the
    requested unit's factor."""
    import sympy as sp

    from physcal.env.formulas import get_family
    from physcal.env.generator import chain_of

    fam = get_family(problem["family"])
    chain = chain_of(problem)
    steps = [(name, out, fam.formula(name).solved[out]) for name, out in chain.steps]
    used = {a for _, _, s in steps for a in s.args}
    q = fam.quantities
    lines = ["import math", "", "", f"def solve({', '.join(g['symbol'] for g in problem['givens'])}):"]
    for g in problem["givens"]:
        f = _factor(g["unit"])
        if f != 1.0:
            lines.append(f"    {g['symbol']} = {g['symbol']} * {f!r}  # {g['unit']} -> {q[g['symbol']].si_unit}")
    for c in sorted(used & set(fam.si_constants)):
        lines.append(f"    {c} = {fam.si_constants[c]!r}  # {fam.constants[c][1]}")
    for name, out, s in steps:
        lines.append(f"    {out} = {sp.pycode(s.expr)}  # {name}")
    f = _factor(problem["target_unit"])
    tgt = problem["target_symbol"]
    lines.append(f"    return {tgt} / {f!r}  # {problem['target_si_unit']} -> {problem['target_unit']}" if f != 1.0
                 else f"    return {tgt}  # {problem['target_unit']}")
    return "\n".join(lines) + "\n"


def prompt_text(text: str, cfg: dict | None = None) -> str:
    """The problem text the model sees: the answer format's `<steps>` sentences removed (D119 §2)."""
    cfg = cfg or _config()
    return STEPS_SENTENCE_RE.sub("", text).rstrip() if cfg["prompt"]["strip_steps_sentences"] else text


@lru_cache(maxsize=None)
def _examples_cached(key: str) -> tuple:
    cfg = json.loads(key)
    ex = cfg["prompt"]["examples"]
    env_cfg = load_config(ex["env_config"])
    from physcal.env.generator import generate_problem

    out = []
    for fam, depth in ex["cells"]:
        for i in range(ex["max_tries"]):
            p = generate_problem(fam, depth, ex["seed_offset"] + i, split="train", cfg=env_cfg,
                                 mixed_levels=ex["mixed_levels"]).to_dict()
            if (p["distractors"] and any(_factor(g["unit"]) != 1.0 for g in p["givens"])
                    and any(g.get("noise_rel") for g in p["givens"])):
                out.append(p)
                break
        else:
            raise RuntimeError(f"no example for {fam} depth {depth} in {ex['max_tries']} seeds")
    return tuple(json.dumps(p) for p in out)


def examples(cfg: dict | None = None) -> list[dict]:
    """D119 §2: per cell, the first seed from the offset whose problem has a distractor, a non-SI given unit and a
    noisy given (training split, mixed levels)."""
    return [json.loads(p) for p in _examples_cached(json.dumps(cfg or _config(), sort_keys=True))]


def messages(problem_text: str, cfg: dict | None = None) -> list[dict]:
    cfg = cfg or _config()
    msgs = [{"role": "system", "content": cfg["prompt"]["system"]}]
    for p in examples(cfg):
        msgs += [{"role": "user", "content": prompt_text(p["text"], cfg)},
                 {"role": "assistant", "content": f"```python\n{reference_program(p)}```"}]
    return msgs + [{"role": "user", "content": prompt_text(problem_text, cfg)}]


def model_config(cfg: dict | None = None, cap_gib: float | None = None) -> tuple[str, dict, dict]:
    """(model id, run_eval model config, generation config): the pinned base, no adapter, the unsloth 4-bit loader."""
    from physcal.base import base_config, pinned_base_dir

    cfg = cfg or _config()
    pin_cfg = base_config(cfg["model"]["pinned_base"])
    mid = pinned_base_dir(pin_cfg)
    mc = {"id": mid, "pinned_base": cfg["model"]["pinned_base"], "revision": None, "fallback_id": None, "lora": None,
          "loader": "unsloth_4bit", "model_class": "causal_lm", "max_seq_length": pin_cfg["model"]["max_seq_length"],
          "text_only": pin_cfg["model"].get("text_only", False),
          "chat_template_kwargs": pin_cfg["model"].get("chat_template_kwargs") or {},
          "max_reserved_gib": cfg["model"]["max_reserved_gib"] if cap_gib is None else cap_gib, "device": "cuda"}
    return mid, mc, dict(cfg["generation"])


# ------------------------------------------------------------------------------------------- generation (GPU)
def problems_of(set_name: str) -> list[dict]:
    from physcal.eval import mc_a

    return mc_a.problems_of(set_name)


def generate(set_name: str, limit: int | None = None, backend=None, cap_gib: float | None = None,
             cfg: dict | None = None) -> dict:
    """GPU: one greedy completion per problem not yet in generations.jsonl, in chunks, each appended as it finishes.
    `backend` (tests): a callable taking conversations and returning objects with text / n_tokens / truncated."""
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    cfg = cfg or _config()
    out = set_dir(set_name, cfg)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "generations.jsonl"
    done = {r["seed"] for r in map(json.loads, path.open())} if path.exists() else set()
    probs = problems_of(set_name)[:limit]
    todo = [p for p in probs if p["seed"] not in done]
    mid, mc, gc = model_config(cfg, cap_gib)
    t0 = time.time()
    meta = {"decision": "D119 (R4-7: MC-B)", "set": set_name, "model": mid, "generation": gc,
            "examples": [{"family": p["family"], "depth": p["depth"], "seed": p["seed"]} for p in examples(cfg)],
            "n_problems": len(probs), "already_done": len(done), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if backend is None:
        from physcal.eval.run_eval import HFBackend

        backend = HFBackend(mid, mc, gc)
        meta.update(provenance=backend.provenance, loader_check=backend.loader_check)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    step = cfg["chunk_problems"]
    with path.open("a") as fh:
        for c in range(0, len(todo), step):
            chunk = todo[c:c + step]
            gens = backend([messages(p["text"], cfg) for p in chunk])
            for p, g in zip(chunk, gens):
                fh.write(json.dumps({"seed": p["seed"], "family": p["family"], "depth": p["depth"], "completion": g.text,
                                     "n_tokens": g.n_tokens, "truncated": g.truncated}) + "\n")
            fh.flush()
            print(f"  mc-b {set_name}: {c + len(chunk)}/{len(todo)} programs ({time.time() - t0:.0f}s)", flush=True)
    meta.update(seconds=time.time() - t0, n_new=len(todo), finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


# ------------------------------------------------------------------------------------------- execution (CPU)
def run_inputs(problem: dict, set_name: str, mc_a_cfg: dict, env_cfg: dict) -> tuple[list[dict], list[float]] | None:
    """Run 0's and runs 1..k's keyword values (displayed numbers, displayed units) and the exact chain's ln f per run;
    None if MC-A excluded the problem."""
    from physcal.eval import mc_a

    base = {g["symbol"]: float(g["display"]) for g in problem["givens"] + problem["distractors"]}
    calls, ln_oracle = [base], [math.log(problem["noise"]["point_estimate_si"])]
    for j in range(1, mc_a_cfg["k"] + 1):
        x = mc_a.perturb(problem, set_name, j, mc_a_cfg, env_cfg)
        if x is None:
            return None
        calls.append({**base, **{s: float(v) for s, v in x["displays"].items()}})
        ln_oracle.append(x["ln_oracle"])
    return calls, ln_oracle


def execute_one(problem: dict, completion: str, calls: list[dict], cfg: dict) -> dict:
    """One program on k + 1 inputs: the failure type (None = every call succeeded) and y_j in SI."""
    rec: dict = {"failure": None, "reason": None, "params": None, "y_si": None}
    code = sandbox.extract(completion)
    if code is None:
        return {**rec, "failure": "no_program"}
    tree, why = sandbox.static_check(code)
    if tree is None:
        return {**rec, "failure": "rejected", "reason": why}
    displayed = {g["symbol"] for g in problem["givens"] + problem["distractors"]}
    params, why = sandbox.signature(tree, displayed)
    if params is None:
        return {**rec, "failure": "signature", "reason": why}
    sb = cfg["sandbox"]
    res = sandbox.run_program(code, [{p: c[p] for p in params} for c in calls], sb["cpu_seconds"], sb["memory_bytes"],
                              sb["wall_seconds"])
    f = _factor(problem["target_unit"])
    y = [None if v is None else v * f for v in res["values"]]
    return {**rec, "failure": res["failure"], "reason": next((e for e in res["errors"] if e), None), "params": params,
            "y_si": y}


def execute(set_name: str, limit: int | None = None, cfg: dict | None = None) -> dict:
    """CPU: every generated program not yet in executions.jsonl, run on its k + 1 inputs."""
    cfg = cfg or _config()
    mc_a_cfg, env_cfg = load_config(cfg["mc_a_config"]), load_config("env_v2")
    out = set_dir(set_name, cfg)
    gens = {r["seed"]: r for r in map(json.loads, (out / "generations.jsonl").open())}
    path = out / "executions.jsonl"
    done = {r["seed"] for r in map(json.loads, path.open())} if path.exists() else set()
    probs = [p for p in problems_of(set_name)[:limit] if p["seed"] in gens and p["seed"] not in done]
    t0, n_excluded = time.time(), 0
    with path.open("a") as fh:
        for i, p in enumerate(probs):
            ri = run_inputs(p, set_name, mc_a_cfg, env_cfg)
            if ri is None:
                n_excluded += 1
                rec = {"seed": p["seed"], "excluded": True}
            else:
                calls, ln_oracle = ri
                rec = {"seed": p["seed"], "family": p["family"], "depth": p["depth"], "excluded": False,
                       "ln_oracle": ln_oracle, **execute_one(p, gens[p["seed"]]["completion"], calls, cfg)}
            fh.write(json.dumps(rec) + "\n")
            if (i + 1) % 200 == 0:
                fh.flush()
                print(f"  mc-b execute {set_name}: {i + 1}/{len(probs)} ({time.time() - t0:.0f}s)", flush=True)
    return {"set": set_name, "n_executed": len(probs), "n_excluded": n_excluded, "seconds": time.time() - t0}


def reference_check(set_name: str, stride: int = 1, cfg: dict | None = None) -> dict:
    """D119 §3: the builder's reference program, run through the same harness on MC-A's inputs, reproduces MC-A's
    oracle σ (k = 8) on every problem: max |σ_ref − σ_oracle| and max |ln y − ln f| over all runs."""
    from physcal.eval import mc_a

    cfg = cfg or _config()
    mc_a_cfg, env_cfg = load_config(cfg["mc_a_config"]), load_config("env_v2")
    k = max(cfg["k_curve"])
    worst_sigma, worst_ln, n, fails, excluded = 0.0, 0.0, 0, [], 0
    for p in problems_of(set_name)[::stride]:
        ri = run_inputs(p, set_name, mc_a_cfg, env_cfg)
        if ri is None:
            excluded += 1
            continue
        calls, ln_oracle = ri
        ex = execute_one(p, f"```python\n{reference_program(p)}```", calls, cfg)
        n += 1
        if ex["failure"] is not None:
            fails.append((p["seed"], ex["failure"], ex["reason"]))
            continue
        worst_ln = max(worst_ln, max(abs(math.log(y) - lo) for y, lo in zip(ex["y_si"], ln_oracle)))
        worst_sigma = max(worst_sigma, abs(estimates({**ex, "ln_oracle": ln_oracle}, k, cfg)["mc_b"] -
                                           mc_a.sd(ln_oracle[1:k + 1])))
    return {"set": set_name, "n": n, "excluded": excluded, "failures": fails, "max_abs_sigma_diff": worst_sigma,
            "max_abs_ln_diff": worst_ln}


# ------------------------------------------------------------------------------------------- estimators
def estimates(ex: dict, k: int, cfg: dict) -> dict:
    """MC-B's σ from the first k perturbed runs (None = undefined -> the floor) and μ = y_0 (None if the program
    failed). A failed program (any of its k + 1 calls) has neither."""
    from physcal.eval import mc_a

    ok = ex["failure"] is None
    ln = [math.log(y) for y in ex["y_si"][1:k + 1]] if ok else []
    return {"mc_b": mc_a.sd(ln) if ok and len(ln) >= cfg["min_runs"] else None,
            "mc_b_mad": mc_a.mad(ln, cfg["mad_scale"]) if ok and len(ln) >= cfg["min_runs"] else None,
            "mu0_si": ex["y_si"][0] if ok else None, "failed": not ok}


def exact_program(ex: dict, tau: float) -> bool:
    """All k + 1 outputs within τ of the exact chain on the same inputs (log space)."""
    return ex["failure"] is None and all(abs(math.log(y) - lo) <= tau for y, lo in zip(ex["y_si"], ex["ln_oracle"]))


def _read_jsonl(path: Path) -> list[dict]:
    if path.exists():
        return [json.loads(line) for line in path.open() if line.strip()]
    gz = path.with_suffix(path.suffix + ".gz")
    with gzip.open(gz, "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ------------------------------------------------------------------------------------------- the analysis (D119 §6-§8)
def analyze(set_name: str, cfg: dict | None = None) -> dict:
    from physcal.eval import a6_pairs, mc_a, oracle_centred as oc
    from physcal.eval import r4_analysis as R
    from physcal.eval import stage_a
    from physcal.eval import stage_b_analysis as B
    from physcal.eval import stage_d_analysis as Dn
    from physcal.eval.stats import holm
    from physcal.train import sft_r4
    from physcal.verify.reward import reward_config

    cfg = cfg or _config()
    ac = cfg["analysis"]
    mc_a_cfg = load_config(cfg["mc_a_config"])
    out = set_dir(set_name, cfg)
    sa, bcfg, pc = load_config("stage_a"), load_config("stage_b")["analysis"], load_config("pairs")
    sc = reward_config()["scoring"]
    floor, tau = pc["analysis"]["sigma_floor_rel"], pc["analysis"]["tau"]
    allp = stage_a.load_sets(sa)
    src, _ = stage_a.sources(allp, sa)
    idx = [i for i, p in enumerate(allp) if p["_set"] == set_name]
    probs = [allp[i] for i in idx]
    ex = {r["seed"]: r for r in _read_jsonl(out / "executions.jsonl")}
    gens = {r["seed"]: r for r in _read_jsonl(out / "generations.jsonl")}

    # MC-A's runs and run-0 tape (its used problems: k runs complete, both pair members)
    a_dir = mc_a.set_dir(set_name, mc_a_cfg)
    a_meta = json.loads((a_dir / "meta.json").read_text())
    a_runs: dict = {}
    for r in _read_jsonl(a_dir / "runs.jsonl"):
        a_runs.setdefault(r["seed"], {})[r["j"]] = r
    tape = {(r["split"], *B._key(r["problem"])): r for r in B.read_records(REPO_ROOT / a_meta["run0_tape"])}
    avail = {p["seed"]: len(a_runs.get(p["seed"], {})) == mc_a_cfg["k"] and p["seed"] in ex
             and not ex[p["seed"]].get("excluded") for p in probs}
    by_pair: dict = {}
    for p in probs:
        by_pair.setdefault(p["noise"]["pair"]["pair_seed"], []).append(avail[p["seed"]])
    use = np.array([len(by_pair[p["noise"]["pair"]["pair_seed"]]) == 2 and all(by_pair[p["noise"]["pair"]["pair_seed"]])
                    for p in probs])
    sub = [p for p, u in zip(probs, use) if u]
    sub_idx = [i for i, u in zip(idx, use) if u]
    terms = [oc.problem_terms(p, sa["mc"]["n_draws"], sa["mc"]["seed_tag"]) for p in sub]
    has = np.array([t["draws"] is not None for t in terms])
    boot = oc.ClusterBootstrap([(set_name, p["noise"]["pair"]["pair_seed"]) for p in sub], ac["bootstrap"], ac["seed"])
    types = R.pair_types(sub, R.references(sub, sft_r4._config()))
    rac = {"bootstrap": ac["bootstrap"], "seed": ac["seed"], "criterion_ci_level": ac["criterion_ci_level"]}
    reg = "truth" if set_name == "a6b_pairs" else "sigma_star"
    exs = [ex[p["seed"]] for p in sub]
    point = np.array([math.log(p["noise"]["point_estimate_si"]) for p in sub])
    k = max(cfg["k_curve"])
    est = {kk: [estimates(e, kk, cfg) for e in exs] for kk in cfg["k_curve"]}
    mu_b = [e["mu0_si"] for e in est[k]]
    cc_b = np.array([m is not None and abs(math.log(m) - lp) <= tau for m, lp in zip(mu_b, point)])
    run0 = [tape.get((set_name, *B._key(p))) for p in sub]
    mu_a = [r["breakdown"]["mu_si"] if r is not None and r["breakdown"]["reason"] == "ok" and
            (r["breakdown"].get("mu_si") or 0) > 0 else None for r in run0]
    dpos_recs = B.read_records(REPO_ROOT / cfg["comparators"]["dpos2_eval"])

    def e2e(sigma: np.ndarray, mu: list) -> tuple[float, list[float]]:
        v = np.array([-sc["crps_clip"] if m is None else oc.verifier_crps_log(m, float(s), p["target_value_si"], sc)
                      for m, s, p in zip(mu, sigma, sub)])
        return boot.mean(v)

    def score(sigma: np.ndarray, own_mu: list | None, cc: np.ndarray) -> dict:
        scores = oc.score_source(terms, sigma, sc)
        pairs = a6_pairs.pairs_from_rows(B.source_rows(sub, sigma, floor)[set_name])
        return {"oracle_centred": oc.summarize(terms, sigma, scores, boot, sc, sa["coverage_k"]),
                "slope": R.slope(pairs, rac, reg), "splits": R.splits(pairs, types, rac),
                "aleatoric_expected_on_own_run0_correct": boot.mean(np.nan_to_num(scores["expected"]), has & cc),
                "epistemic_auroc_own_run0_wrong": mc_a.auroc(sigma, ~cc),
                "end_to_end_own_mu": None if own_mu is None else e2e(sigma, own_mu),
                "end_to_end_with_mc_b_mu": e2e(sigma, mu_b), "_expected": scores["expected"]}

    fails = [e["failure"] for e in exs]
    res: dict = {"decision": "D119 (R4-7: MC-B, prototype, development set)", "set": set_name,
                 "model": json.loads((out / "meta.json").read_text()).get("model"), "n_problems": len(probs),
                 "n_generated": sum(p["seed"] in gens for p in probs),
                 "n_excluded_by_mc_a": sum(bool(ex.get(p["seed"], {}).get("excluded")) for p in probs),
                 "n_used": len(sub), "variants": {}, "comparators": {}}
    res["programs"] = {
        "failure_rate": float(np.mean([f is not None for f in fails])),
        "failure_by_type": {t: float(np.mean([f == t for f in fails])) for t in sandbox.FAILURES},
        "failure_by_depth": {str(d): float(np.mean([f is not None for f, p in zip(fails, sub) if p["depth"] == d]))
                             for d in sorted({p["depth"] for p in sub})},
        "run0_computation_correct": float(cc_b.mean()),
        "run0_computation_correct_by_depth": {str(d): float(np.mean([c for c, p in zip(cc_b, sub) if p["depth"] == d]))
                                              for d in sorted({p["depth"] for p in sub})},
        "exact_program": float(np.mean([exact_program(e, tau) for e in exs])),
        "truncated": float(np.mean([gens[p["seed"]]["truncated"] for p in sub])),
        "params_with_distractor": float(np.mean([bool(set(e["params"] or []) & {d["symbol"] for d in p["distractors"]})
                                                 for e, p in zip(exs, sub)]))}
    for kk in cfg["k_curve"]:
        for v in VARIANTS:
            sigma = np.array([e[v] or 0.0 for e in est[kk]])
            res["variants"][f"{v}_k{kk}"] = {**score(sigma, mu_b, cc_b),
                                             "share_undefined": float(np.mean([e[v] is None for e in est[kk]]))}
    # comparators on the same problems
    cc_a = np.array([m is not None and abs(math.log(m) - lp) <= tau for m, lp in zip(mu_a, point)])
    a_est = [mc_a.estimates(r0, list(a_runs[p["seed"]].values()), k, mc_a_cfg) for r0, p in zip(run0, sub)]
    for v in ("oracle", "robust", "raw"):
        res["comparators"][f"mc_a_{v}"] = score(np.array([e[v] or 0.0 for e in a_est]), mu_a, cc_a)
    for n, s in (("sigma_star", src[stage_a.STAR]), ("rss", src[bcfg["rss_source"]]), ("x1", src[bcfg["x1_source"]])):
        res["comparators"][n] = score(np.array([s[i] for i in sub_idx]), None, cc_b)
    if dpos_recs is not None:
        sig_d, bds_d = B.model_sigma(allp, dpos_recs)
        mu_d = [b["mu_si"] if b.get("reason") == "ok" and (b.get("mu_si") or 0) > 0 else None
                for b in (bds_d[i] for i in sub_idx)]
        cc_d = np.array([m is not None and abs(math.log(m) - lp) <= tau for m, lp in zip(mu_d, point)])
        res["comparators"]["dpos2_a"] = score(np.array([sig_d[i] for i in sub_idx]), mu_d, cc_d)
    exp = lambda name: (res["variants"][name] if name in res["variants"] else res["comparators"].get(name) or {}).get("_expected")
    b = exp(f"mc_b_k{k}")
    t = {"M1": Dn.paired(boot, b, exp("rss"), has), "M2": Dn.paired(boot, b, exp("x1"), has),
         "M3": Dn.paired(boot, b, exp("dpos2_a"), has), "M4": Dn.paired(boot, b, exp("mc_a_robust"), has)}
    sl = res["variants"][f"mc_b_k{k}"]["slope"]
    t["M5"] = {"missing": True} if sl.get("slope") is None else {
        "value": sl["slope"], "ci90": sl["ci"], "ci95": sl.get("ci95"), "p_one_sided": sl["p_one_sided"],
        "n_pairs": sl.get("n_pairs")}
    tested = set_name == "a6c_pairs"
    if tested:
        h = holm({kk: (1.0 if t[kk].get("missing") else t[kk]["p_one_sided"]) for kk in TESTS}, ac["alpha"])
        for kk in TESTS:
            t[kk]["holm"] = h[kk]
    res["tests"] = {"applies": tested, "note": "Holm over M1-M5 on A6c only; A6 / A6b are descriptive", **t}
    res["mc_b_minus_mc_a_oracle"] = Dn.paired(boot, b, exp("mc_a_oracle"), has)
    for g in list(res["variants"].values()) + list(res["comparators"].values()):
        g.pop("_expected", None)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=B._json_default))
    (out / "summary.md").write_text(render(res))
    return res


def render(res: dict) -> str:
    sl = lambda d: "–" if not d or d.get("slope") is None else f"{d['slope']:+.3f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"
    f = lambda v, fmt=".3f": "–" if v is None else format(v, fmt)
    k = max(int(n.rsplit("_k", 1)[1]) for n in res["variants"])
    pr = res["programs"]
    L = [f"# MC-B on {res['set']} (R4-7, D119; a prototype on a development set)", "",
         f"Model: {res['model']}. Problems {res['n_problems']}, generated {res['n_generated']}, excluded by MC-A "
         f"{res['n_excluded_by_mc_a']}, used {res['n_used']} (both pair members with MC-A's k runs and an executed "
         "MC-B program).", "",
         f"**Programs (every used problem counted):** failure rate {pr['failure_rate']:.1%} ("
         + (", ".join(f"{t} {v:.1%}" for t, v in pr["failure_by_type"].items() if v) or "none") + "); "
         f"run 0 computation-correct {pr['run0_computation_correct']:.1%}; exact program {pr['exact_program']:.1%}; "
         f"truncated {pr['truncated']:.1%}; a distractor among the parameters {pr['params_with_distractor']:.1%}.", "",
         "Failure rate by depth: " + ", ".join(f"{d}: {v:.1%}" for d, v in pr["failure_by_depth"].items()) +
         ". Run 0 computation-correct by depth: " +
         ", ".join(f"{d}: {v:.1%}" for d, v in pr["run0_computation_correct_by_depth"].items()) + ".", "",
         "| σ source | expected crps_log | median ln(σ/σ*) | 1σ / 2σ cov. | slope [90% CI] | aleatoric (own run 0 "
         "correct) | epistemic AUROC | end-to-end, own μ | end-to-end, MC-B's μ |", "|---|---|---|---|---|---|---|---|---|"]
    rows = [(n, g) for n, g in res["variants"].items()] + list(res["comparators"].items())
    for name, g in rows:
        o = g["oracle_centred"]
        own = g["end_to_end_own_mu"]
        L.append(f"| {name} | {o['expected'][0]:.4f} | {o['median_log_ratio']:+.3f} | {o['coverage_1'][0]:.1%} / "
                 f"{o['coverage_2'][0]:.1%} | {sl(g['slope'])} | {g['aleatoric_expected_on_own_run0_correct'][0]:.4f} "
                 f"| {f(g['epistemic_auroc_own_run0_wrong'])} | {'–' if own is None else f(own[0], '.4f')} "
                 f"| {g['end_to_end_with_mc_b_mu'][0]:.4f} |")
    t = res["tests"]
    name = lambda kk: TEST_NAMES[kk].format(set=res["set"].split("_")[0].capitalize())  # a6b_pairs -> A6b
    L += ["", "## Tests (A6c only; Holm over M1–M5 at α = 0.05)" if t["applies"] else
          "## The same differences (descriptive on this set)", "",
          "| test | value | 95% CI | p (one-sided) | Holm |", "|---|---|---|---|---|"]
    for kk in TESTS:
        x = t[kk]
        if x.get("missing"):
            L.append(f"| {kk}: {name(kk)} | missing | | | |")
            continue
        ci = x.get("ci95") or x.get("ci90")
        L.append(f"| {kk}: {name(kk)} | {x['value']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {x['p_one_sided']:.4g} "
                 f"| {'rejected' if (x.get('holm') or {}).get('rejected') else '–'} |")
    m = res["mc_b_minus_mc_a_oracle"]
    if not m.get("missing"):
        L += ["", f"MC-B − MC-A oracle (the attainable bound; untested): {m['value']:+.4f} "
                  f"[{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}]."]
    L += ["", "**Slope by depth and chain type:**", "", "| σ source | d1 | d2 | d3 | d4 | d5 | power-law | value-dependent |",
          "|---|---|---|---|---|---|---|---|"]
    for name, g in rows:
        s = g["splits"]
        L.append(f"| {name} | " + " | ".join(sl(s["by_depth"].get(str(d))) for d in range(1, 6)) +
                 f" | {sl(s['by_type']['power_law'])} | {sl(s['by_type']['value_dependent'])} |")
    return "\n".join(L) + "\n"


def prompt_report(cfg: dict | None = None) -> dict:
    """The few-shot examples, and the longest prompt on the three sets in tokens (D119 §2's budget check)."""
    from transformers import AutoTokenizer

    from physcal.base import base_config, pinned_base_dir

    cfg = cfg or _config()
    pin = base_config(cfg["model"]["pinned_base"])
    tok = AutoTokenizer.from_pretrained(str(REPO_ROOT / pinned_base_dir(pin)))
    n = {}
    for s in cfg["sets"]:
        for p in problems_of(s):
            text = tok.apply_chat_template(messages(p["text"], cfg), tokenize=False, add_generation_prompt=True,
                                           **(pin["model"].get("chat_template_kwargs") or {}))
            n[(s, p["seed"])] = len(tok(text, add_special_tokens=False)["input_ids"])
    longest = max(n.values())
    return {"examples": [{k: p[k] for k in ("family", "depth", "seed")} for p in examples(cfg)],
            "longest_prompt_tokens": longest, "max_new_tokens": cfg["generation"]["max_new_tokens"],
            "max_seq_length": pin["model"]["max_seq_length"],
            "fits": longest + cfg["generation"]["max_new_tokens"] <= pin["model"]["max_seq_length"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["prompt", "generate", "execute", "analyze"])
    ap.add_argument("--set", default="a6c_pairs", choices=["a6c_pairs", "a6_pairs", "a6b_pairs"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-reserved-gib", type=float, default=None)
    args = ap.parse_args()
    if args.action == "prompt":
        print(json.dumps(prompt_report(), indent=1))
    elif args.action == "generate":
        print(json.dumps({k: v for k, v in generate(args.set, args.limit, cap_gib=args.max_reserved_gib).items()
                          if k in ("seconds", "n_new", "n_problems")}, indent=1))
    elif args.action == "execute":
        print(json.dumps(execute(args.set, args.limit), indent=1))
    else:
        analyze(args.set)
        print((set_dir(args.set) / "summary.md").read_text())


if __name__ == "__main__":
    main()

"""D151 item 1 (EXPLORATORY, follow-up project; not the paper): uncertainty-guided test-time compute, offline.

    OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.eval.followup_ttc x2     # X2's k = 8 samples (SFT 1.5B)
    OMP_NUM_THREADS=1 nice -n 19 uv run python -m physcal.eval.followup_ttc mc_a   # MC-A's raw spread (B-v2), secondary

X2: adaptive sampling policies (draw a few samples; stop when they agree; else draw more, up to 8; majority answer)
against fixed-k majority voting, on accuracy (within 1%) and completion tokens per problem. Every quantity is the
mean over random draw orders of the 8 exchangeable samples. The headline policy is chosen on a hash half of the
problems and reported on the other half (matched-accuracy token saving against the fixed-k curve, bootstrap CI).
MC-A: run 0's answer is kept; the raw spread of the perturbed re-asks flags it as wrong (MC-A's AUROC); a two-stage
rule (2 re-asks, all 8 only when the spread is large) against fixed k. Outputs: <output_dir>/ttc/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math

import numpy as np

from physcal.config import REPO_ROOT, load_config


def _config() -> dict:
    return load_config("followup")


def selection_half(seed: int, salt: str) -> bool:
    return hashlib.sha256(f"{salt}{seed}".encode()).digest()[0] % 2 == 0


# ------------------------------------------------------------------------------------------- X2's samples
def x2_problems(tc: dict) -> list[dict]:
    """Per pool problem (in order): k answers' μ (nan = not an answer), correct, stated relative σ (nan = none) and
    completion tokens, by sample index."""
    from physcal.eval import x2_pairs

    x2 = load_config(tc["x2_config"])
    recs, _ = x2_pairs.sampled_records(x2)
    tol, floor = x2["selection"]["tolerance"], x2["analysis"]["sigma_floor_rel"]
    by: dict = {}
    for r in recs:
        p = r["problem"]
        by.setdefault(p["seed"], {"seed": p["seed"], "family": p["family"], "depth": p["depth"], "rows": {}})
        by[p["seed"]]["rows"][r.get("sample", 0)] = (x2_pairs.answer(r["breakdown"], tol, floor), r["n_tokens"])
    out = []
    for i, (seed, d) in enumerate(by.items()):
        k = tc["k_max"]
        if sorted(d["rows"]) != list(range(k)):
            raise ValueError(f"problem {seed}: samples {sorted(d['rows'])}")
        a = [d["rows"][s][0] for s in range(k)]
        out.append({"index": i, "seed": seed, "family": d["family"], "depth": d["depth"],
                    "mu": np.array([np.nan if x is None else x["mu_si"] for x in a]),
                    "correct": np.array([x is not None and x["correct"] for x in a]),
                    "sigma": np.array([np.nan if x is None or x["rel_sigma"] is None else x["rel_sigma"] for x in a]),
                    "tokens": np.array([d["rows"][s][1] for s in range(k)], dtype=float)})
    return out


def agreement(mu: np.ndarray, tol: float) -> np.ndarray:
    """A[i, j]: answer j agrees with answer i (|μ_j / μ_i − 1| <= tol; μ_i = 0 only with μ_j = 0); both answers."""
    k = len(mu)
    A = np.zeros((k, k), dtype=bool)
    for i in range(k):
        if np.isnan(mu[i]):
            continue
        for j in range(k):
            if np.isnan(mu[j]):
                continue
            A[i, j] = (mu[j] == 0) if mu[i] == 0 else abs(mu[j] / mu[i] - 1) <= tol
    return A


def prefix_states(p: dict, orders: np.ndarray, tol: float) -> dict:
    """For every order (rows) and prefix length m = 1..k (columns m - 1): the majority answer's correctness, its
    votes (0 = no answer drawn), the runner-up's votes, cumulative tokens."""
    A = agreement(p["mu"], tol)
    n, k = orders.shape
    Ap = A[orders[:, :, None], orders[:, None, :]]                  # (n, k, k) in draw order
    C = np.cumsum(Ap, axis=2)                                      # votes of position a among the first m draws
    parsed = ~np.isnan(p["mu"])[orders]
    corr = p["correct"][orders]
    rows = np.arange(n)
    top, second, ok = np.zeros((n, k), int), np.zeros((n, k), int), np.zeros((n, k), bool)
    for m in range(1, k + 1):
        v = np.where(parsed[:, :m], C[:, :m, m - 1], -1)
        lead = v.argmax(axis=1)                                     # ties: the earliest drawn
        t = v[rows, lead]
        same = Ap[rows, lead, :m] | Ap[rows, :m, lead]
        rest = np.where(same, -1, v)
        top[:, m - 1] = np.maximum(t, 0)
        second[:, m - 1] = np.maximum(rest.max(axis=1), 0)
        ok[:, m - 1] = (t >= 1) & corr[rows, lead]
    return {"top": top, "second": second, "ok": ok, "tokens": np.cumsum(p["tokens"][orders], axis=1),
            "first_sigma": p["sigma"][orders[:, 0]]}


def stop_index(st: dict, pol: dict, k: int, theta: dict) -> np.ndarray:
    """The number of samples drawn by a policy, per order."""
    top, second = st["top"], st["second"]
    n = top.shape[0]
    kind = pol["kind"]
    if kind == "fixed":
        return np.full(n, pol["k"])

    def first_where(cond: np.ndarray, start: int) -> np.ndarray:
        c = cond.copy()
        c[:, :start - 1] = False
        c[:, k - 1] = True
        return c.argmax(axis=1) + 1

    if kind == "two_then_all":
        return np.where(top[:, pol["first"] - 1] >= pol["first"], pol["first"], k)
    if kind == "sequential":
        return first_where(top >= pol["votes"], pol["first"])
    if kind == "no_loss":
        left = k - np.arange(1, k + 1)
        return first_where(top > second + left, 1)
    if kind == "sigma_gate":
        then = first_where(top >= pol["then_votes"], 2)
        s = st["first_sigma"]
        return np.where(~np.isnan(s) & (s <= theta[pol["name"]]), 1, then)
    raise ValueError(kind)


def simulate(probs: list[dict], policies: list[dict], tc: dict, theta: dict, identity: bool = False) -> dict:
    """Per problem and policy: mean accuracy, tokens and samples over the draw orders."""
    k = tc["k_max"]
    out = {pol["name"]: {"acc": [], "tok": [], "n": []} for pol in policies}
    for p in probs:
        if identity:
            orders = np.arange(k)[None, :]
        else:
            rng = np.random.default_rng(tc["seed"] + p["index"])
            orders = np.array([rng.permutation(k) for _ in range(tc["n_orders"])])
        st = prefix_states(p, orders, tc["agree_tolerance"])
        rows = np.arange(len(orders))
        for pol in policies:
            m = stop_index(st, pol, k, theta)
            out[pol["name"]]["acc"].append(st["ok"][rows, m - 1].mean())
            out[pol["name"]]["tok"].append(st["tokens"][rows, m - 1].mean())
            out[pol["name"]]["n"].append(m.mean())
    return {n: {f: np.array(v) for f, v in d.items()} for n, d in out.items()}


# ------------------------------------------------------------------------------------------- matched saving
def t_fix(acc_k: np.ndarray, tok_k: np.ndarray, a: float) -> tuple[float, str]:
    """The token cost at which the monotone fixed-k curve reaches accuracy a (linear interpolation), and a flag."""
    env = np.maximum.accumulate(acc_k)
    if a < env[0]:
        return float(tok_k[0]), "below_k1"
    if a > env[-1]:
        return float(tok_k[int(np.argmax(env >= env[-1]))]), "above_curve"
    j = int(np.argmax(env >= a))
    if j == 0:
        return float(tok_k[0]), "ok"
    lo, hi = env[j - 1], env[j]
    w = 0.0 if hi == lo else (a - lo) / (hi - lo)
    return float(tok_k[j - 1] + w * (tok_k[j] - tok_k[j - 1])), "ok"


def saving(acc: float, tok: float, acc_k: np.ndarray, tok_k: np.ndarray) -> tuple[float, str]:
    t, flag = t_fix(acc_k, tok_k, acc)
    return 1 - tok / t, flag


def summarize(sim: dict, idx: np.ndarray, names: list[str], fixed: list[str], n_boot: int, seed: int,
              ref: str) -> dict:
    """Point estimates and problem-bootstrap CIs on the problems `idx`: accuracy, tokens, samples, Δ accuracy and
    token ratio vs `ref` (majority@8), and the matched-accuracy saving S_P (the fixed-k curve recomputed per draw)."""
    rng = np.random.default_rng(seed)
    B = rng.integers(0, len(idx), (n_boot, len(idx)))
    get = lambda n, f: sim[n][f][idx]
    curve = lambda sel=None: (np.array([get(f, "acc").mean() if sel is None else get(f, "acc")[sel].mean(axis=1)
                                        for f in fixed]),
                              np.array([get(f, "tok").mean() if sel is None else get(f, "tok")[sel].mean(axis=1)
                                        for f in fixed]))
    acc_k, tok_k = curve()
    bacc_k, btok_k = curve(B)                                       # (len(fixed), n_boot)
    out = {"n_problems": int(len(idx)), "fixed_curve": {"acc": acc_k.tolist(), "tokens": tok_k.tolist()}, "policies": {}}
    ci = lambda v: np.percentile(v, [2.5, 97.5]).tolist()
    for n in names:
        a, t, s = get(n, "acc"), get(n, "tok"), get(n, "n")
        ba, bt = a[B].mean(axis=1), t[B].mean(axis=1)
        sv, flag = saving(a.mean(), t.mean(), acc_k, tok_k)
        bs = np.array([saving(ba[b], bt[b], bacc_k[:, b], btok_k[:, b])[0] for b in range(n_boot)])
        ra, rt = get(ref, "acc"), get(ref, "tok")
        out["policies"][n] = {
            "accuracy": float(a.mean()), "accuracy_ci95": ci(ba), "tokens": float(t.mean()), "tokens_ci95": ci(bt),
            "samples": float(s.mean()),
            "delta_accuracy_vs_maj8": float(a.mean() - ra.mean()), "delta_accuracy_ci95": ci(ba - ra[B].mean(axis=1)),
            "token_ratio_vs_maj8": float(t.mean() / rt.mean()), "token_ratio_ci95": ci(bt / rt[B].mean(axis=1)),
            "matched_saving": float(sv), "matched_saving_flag": flag, "matched_saving_ci95": ci(bs)}
    return out


def run_x2(cfg: dict | None = None) -> dict:
    cfg = cfg or _config()
    tc = cfg["ttc"]
    probs = x2_problems(tc)
    sel = np.array([selection_half(p["seed"], tc["split_salt"]) for p in probs])
    k = tc["k_max"]
    sig_sel = np.concatenate([p["sigma"] for p, s in zip(probs, sel) if s])
    sig_sel = sig_sel[~np.isnan(sig_sel)]
    theta = {pol["name"]: float(np.quantile(sig_sel, pol["quantile"]))
             for pol in tc["policies"] if pol["kind"] == "sigma_gate"}
    fixed = [{"name": f"maj@{j}", "kind": "fixed", "k": j} for j in range(1, k + 1)]
    pols = fixed + tc["policies"]
    names_f, names_p = [p["name"] for p in fixed], [p["name"] for p in tc["policies"]]
    sim = simulate(probs, pols, tc, theta)
    sim_id = simulate(probs, pols, tc, theta, identity=True)
    nb, sd = tc["bootstrap"], tc["seed"]
    ref = f"maj@{k}"
    half = {"selection": np.flatnonzero(sel), "evaluation": np.flatnonzero(~sel), "all": np.arange(len(probs))}
    s_sel = summarize(sim, half["selection"], names_p, names_f, 1, sd, ref)  # point estimates only
    best = max(names_p, key=lambda n: s_sel["policies"][n]["matched_saving"])
    res = {"kind": "D151 item 1, exploratory (offline; X2's k = 8 samples of the corrected SFT 1.5B)",
           "n_problems": len(probs), "n_selection": int(sel.sum()), "n_evaluation": int((~sel).sum()),
           "sigma_gate_thresholds": theta,
           "selection": {n: s_sel["policies"][n]["matched_saving"] for n in names_p}, "best_on_selection": best,
           "evaluation": summarize(sim, half["evaluation"], [best] + names_f, names_f, nb, sd, ref),
           "all_in_sample": summarize(sim, half["all"], names_f + names_p, names_f, nb, sd, ref),
           "identity_order_check": {n: {"accuracy": float(sim_id[n]["acc"].mean()),
                                        "tokens": float(sim_id[n]["tok"].mean())} for n in names_f + names_p},
           "by_depth_in_sample": {}}
    for d in sorted({p["depth"] for p in probs}):
        idx = np.array([i for i, p in enumerate(probs) if p["depth"] == d])
        res["by_depth_in_sample"][str(d)] = {n: {"accuracy": float(sim[n]["acc"][idx].mean()),
                                                 "tokens": float(sim[n]["tok"][idx].mean())}
                                             for n in names_f + names_p}
    return res


# ------------------------------------------------------------------------------------------- MC-A
def mc_a_set(set_name: str, tc: dict) -> dict:
    """MC-A's problems in complete pairs (its own rule), run 0's computation-correctness, the re-asks' ln μ (nan =
    invalid) and tokens in MC-A's draw order j = 1..8, and the pair clusters."""
    import gzip
    from pathlib import Path

    from physcal.eval import mc_a, stage_a
    from physcal.eval import stage_b_analysis as B

    mc = tc["mc_a"]
    cfg = load_config(mc["config"])
    out = mc_a.set_dir(set_name, cfg)
    meta = json.loads((out / "meta.json").read_text())
    probs = [p for p in stage_a.load_sets(load_config("stage_a")) if p["_set"] == set_name]
    runs: dict = {}
    raw = out / "runs.jsonl"
    with (open(raw) if raw.exists() else gzip.open(out / "runs.jsonl.gz", "rt")) as fh:
        for r in map(json.loads, fh):
            runs.setdefault(r["seed"], {})[r["j"]] = r
    tape = {(r["split"], *B._key(r["problem"])): r for r in B.read_records(Path(REPO_ROOT / meta["run0_tape"]))}
    complete = {p["seed"]: len(runs.get(p["seed"], {})) == cfg["k"] for p in probs}
    pair_ok: dict = {}
    for p in probs:
        pair_ok.setdefault(p["noise"]["pair"]["pair_seed"], []).append(complete[p["seed"]])
    sub = [p for p in probs if all(pair_ok[p["noise"]["pair"]["pair_seed"]]) and
           len(pair_ok[p["noise"]["pair"]["pair_seed"]]) == 2]
    k = cfg["k"]
    lnmu = np.full((len(sub), k), np.nan)
    tok = np.zeros((len(sub), k))
    cc0 = np.zeros(len(sub), bool)
    for i, p in enumerate(sub):
        r0 = tape.get((set_name, *B._key(p)))
        b = r0["breakdown"] if r0 is not None else {"reason": "missing"}
        cc0[i] = b["reason"] == "ok" and (b.get("mu_si") or 0) > 0 and \
            abs(math.log(b["mu_si"] / p["noise"]["point_estimate_si"])) <= mc["tau"]
        for j in range(1, k + 1):
            r = runs[p["seed"]][j]
            tok[i, j - 1] = r["n_tokens"]
            if r["reason"] == "ok" and (r["mu_si"] or 0) > 0:
                lnmu[i, j - 1] = math.log(r["mu_si"])
    return {"problems": sub, "lnmu": lnmu, "tokens": tok, "wrong": ~cc0, "min_runs": cfg["min_runs"],
            "clusters": [p["noise"]["pair"]["pair_seed"] for p in sub]}


def raw_spread(lnmu: np.ndarray, m: int, min_runs: int) -> np.ndarray:
    """MC-A's raw σ from the first m re-asks: SD (ddof 1) of the valid ln μ; fewer than min_runs -> 0 (the floor)."""
    x = lnmu[:, :m]
    n = (~np.isnan(x)).sum(axis=1)
    sd = np.nanstd(np.where(n[:, None] >= 2, x, 0.0), axis=1, ddof=1) if m >= 2 else np.zeros(len(x))
    return np.where(n >= min_runs, sd, 0.0)


def run_mc_a(cfg: dict | None = None) -> dict:
    from physcal.eval.mc_a import auroc

    cfg = cfg or _config()
    tc = cfg["ttc"]
    mc = tc["mc_a"]
    res: dict = {"kind": "D151 item 1, secondary (descriptive): MC-A's raw spread as a two-stage stopping rule; "
                         "MC-A's own draw order j = 1..8 (one realization: it reproduces #37's k curve)", "sets": {}}
    data = {s: mc_a_set(s, tc) for s in mc["sets"]}
    d0 = data[mc["select_set"]]
    sel0 = np.array([selection_half(p["seed"], tc["split_salt"]) for p in d0["problems"]])
    s2 = raw_spread(d0["lnmu"], mc["first"], d0["min_runs"])
    thetas = {f"q{q}": float(np.quantile(s2[sel0], q)) for q in mc["quantiles"]}
    fixed = list(range(mc["first"], mc["k_max"] + 1))
    rng = np.random.default_rng(tc["seed"])
    for s, d in data.items():
        n = len(d["problems"])
        spreads = {f"k{m}": raw_spread(d["lnmu"], m, d["min_runs"]) for m in fixed}
        tokens = {f"k{m}": d["tokens"][:, :m].sum(axis=1) for m in fixed}
        sk, s8 = spreads[f"k{mc['first']}"], spreads[f"k{mc['k_max']}"]
        for name, th in thetas.items():
            stop = sk <= th
            spreads[name] = np.where(stop, sk, s8)
            tokens[name] = np.where(stop, tokens[f"k{mc['first']}"], tokens[f"k{mc['k_max']}"])
        sel = np.array([selection_half(p["seed"], tc["split_salt"]) for p in d["problems"]])
        cl = np.array(d["clusters"])
        uniq = np.unique(cl)
        members = {c: np.flatnonzero(cl == c) for c in uniq}

        def stats(idx: np.ndarray, n_boot: int) -> dict:
            au = {nm: auroc(v[idx], d["wrong"][idx]) for nm, v in spreads.items()}
            tk = {nm: float(v[idx].mean()) for nm, v in tokens.items()}
            curve_a = np.array([au[f"k{m}"] for m in fixed])
            curve_t = np.array([tk[f"k{m}"] for m in fixed])
            boots = {nm: [] for nm in spreads}
            cl_idx = np.unique(cl[idx])
            for _ in range(n_boot):
                pick = np.concatenate([members[c] for c in rng.choice(cl_idx, len(cl_idx))])
                pick = pick[np.isin(pick, idx)]
                ba = {nm: auroc(v[pick], d["wrong"][pick]) for nm, v in spreads.items()}
                bt = {nm: float(v[pick].mean()) for nm, v in tokens.items()}
                ca, ct = np.array([ba[f"k{m}"] for m in fixed]), np.array([bt[f"k{m}"] for m in fixed])
                for nm in spreads:
                    boots[nm].append((ba[nm], bt[nm], saving(ba[nm], bt[nm], ca, ct)[0]))
            out = {}
            for nm in spreads:
                b = np.array(boots[nm]) if n_boot else None
                sv, flag = saving(au[nm], tk[nm], curve_a, curve_t)
                out[nm] = {"auroc": au[nm], "reask_tokens": tk[nm], "matched_saving": float(sv), "flag": flag,
                           **({"auroc_ci95": np.percentile(b[:, 0], [2.5, 97.5]).tolist(),
                               "tokens_ci95": np.percentile(b[:, 1], [2.5, 97.5]).tolist(),
                               "matched_saving_ci95": np.percentile(b[:, 2], [2.5, 97.5]).tolist()} if n_boot else {})}
            return out

        sel_stats = stats(np.flatnonzero(sel), 0) if s == mc["select_set"] else None
        res["sets"][s] = {"n_problems": n, "run0_wrong_share": float(d["wrong"].mean()),
                          "selection_half": sel_stats,
                          "evaluation_half": stats(np.flatnonzero(~sel), tc["bootstrap"]) if s == mc["select_set"] else None,
                          "all": stats(np.arange(n), tc["bootstrap"])}
    sel_res = res["sets"][mc["select_set"]]["selection_half"]
    res["thetas"] = thetas
    res["best_on_selection"] = max(thetas, key=lambda nm: sel_res[nm]["matched_saving"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["x2", "mc_a"])
    a = ap.parse_args()
    cfg = _config()
    out = REPO_ROOT / cfg["output_dir"] / "ttc"
    out.mkdir(parents=True, exist_ok=True)
    res = run_x2(cfg) if a.what == "x2" else run_mc_a(cfg)
    (out / f"{a.what}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("all_in_sample", "evaluation", "sets")}, indent=1,
                     default=float))


if __name__ == "__main__":
    main()

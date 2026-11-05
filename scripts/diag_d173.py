"""Post-hoc diagnostics of D173's placebo (read-only on results/): why do same-value texts move at all?"""
import json, sys
import numpy as np
from physcal.eval import followup_r6 as R6, followup_r6_controls as C
from physcal.eval import followup_scale as F, followup_scale_analysis as A
cfg = F._config(); sc = cfg["scale"]
rows, items = C._kept(cfg)
corr = {R6._key_sample(it): it["correct"] for it in items}
index, _, cache = A.probe_inputs(C.MODEL, "hidden", corr, cfg)
pc = json.loads(cache.read_text())
opp, ol = C._ordered("swap_1.5b", "1.5b", cfg)
tx, tl = C._ordered(C.SET, "1.5b", cfg)
po, pp, pt = C.score_probe(items, index, pc, opp, ol, tx, tl, cfg); pt.pop("_gap")
e = C.box_effects(tx, pt, po)
by = {it["id"]: it for it in items}
box = lambda it: F.last_boxed(it["completion"])[0]
out = {}
for kind in ("correct", "same_value", "diff_value"):
    ts = [t for t in tx if t["kind"] == kind]
    ident = [t for t in ts if box(by[t["orig_id"]]) == box(by[t["donor_id"]])]
    other = [t for t in ts if box(by[t["orig_id"]]) != box(by[t["donor_id"]])]
    v = np.array([e[t["id"]] for t in ident]) if ident else np.array([0.0])
    w = np.array([e[t["id"]] for t in other]) if other else np.array([0.0])
    out[kind] = {"identical_string": {"n": len(ident), "mean": float(v.mean()), "max_abs": float(np.abs(v).max())},
                 "other_string": {"n": len(other), "mean": float(w.mean()), "median": float(np.median(w))}}
# M with a pure placebo: the same_value donor's box string is identical to a's
pairs = {}
for t in tx:
    if t["kind"] != "correct":
        pairs.setdefault(t["orig_id"], {})[t["kind"]] = t
both = [v for v in pairs.values() if len(v) == 2]
pure = [v for v in both if box(by[v["same_value"]["orig_id"]]) == box(by[v["same_value"]["donor_id"]])]
def per_problem(sel, f):
    d = {}
    for v in sel:
        d.setdefault(tuple(v["diff_value"]["key"]), []).append(f(v))
    return np.array([np.mean(x) for x in d.values()])
m_pure = per_problem(pure, lambda v: e[v["diff_value"]["id"]] - e[v["same_value"]["id"]])
out["M, identical-string placebo"] = A.paired(m_pure, np.zeros_like(m_pure), sc["bootstrap"], sc["seed"]) | {"n_answers": len(pure)}
base = lambda v, k: po[v[k]["donor_id"]]
out["donor p_orig on M's answers"] = {"d_diff": float(np.mean([base(v, "diff_value") for v in both])),
                                      "d_same": float(np.mean([base(v, "same_value") for v in both]))}
lg = lambda p: float(np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))))
el = lambda t: lg(pt[t["id"]]) - lg(po[t["donor_id"]])
m_log = per_problem(both, lambda v: el(v["diff_value"]) - el(v["same_value"]))
out["M in log-odds"] = A.paired(m_log, np.zeros_like(m_log), sc["bootstrap"], sc["seed"])
ex = [(box(by[t["orig_id"]]), box(by[t["donor_id"]]), round(e[t["id"]], 3)) for t in tx if t["kind"] == "correct" and box(by[t["orig_id"]]) != box(by[t["donor_id"]])][:12]
out["examples (a box, donor box, e)"] = ex
ca = json.loads((R6.r6_dir(cfg) / "cache" / "answer_only_probe_1.5b.json").read_text())["info"]["folds"]
out["trained answer-only folds"] = [(f["layer"], f["C"]) for f in ca]
import transformers
from physcal.config import REPO_ROOT
from physcal.eval import x2_hidden as XH
tok = transformers.AutoTokenizer.from_pretrained(str(REPO_ROOT / cfg["scale"]["models"]["1.5b"]["dir"]))
def brace_tok(comp, brace):
    en = tok(comp, add_special_tokens=False, return_offsets_mapping=True)
    return tok.decode(en["input_ids"][XH.token_at(en["offset_mapping"], brace)])
ident = [t for t in tx if t["kind"] != "diff_value" and box(by[t["orig_id"]]) == box(by[t["donor_id"]])]
own = [brace_tok(by[t["donor_id"]]["completion"], by[t["donor_id"]]["p2"]) for t in ident]
plain = np.array([e[t["id"]] for t, o in zip(ident, own) if o == "}"])
merged = np.array([e[t["id"]] for t, o in zip(ident, own) if o != "}"])
out["identical strings by the donor's brace token"] = {
    "plain": {"n": len(plain), "mean": float(plain.mean()), "max_abs": float(np.abs(plain).max())},
    "merged": {"n": len(merged), "mean": float(merged.mean()), "mean_abs": float(np.abs(merged).mean()),
               "max_abs": float(np.abs(merged).max())}}
print(out["identical strings by the donor's brace token"])
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)

"""The paper's tables, each built from committed result files only (paper/generated/tables/<name>.tex).

Every table starts with a comment block naming its source files and their git blobs (paperlib.Sources.header).
"""

from __future__ import annotations

import re

from paperlib import OUT, Sources, ascii_only, ci, est_ci, integer, num, pct, pval, tex_escape

SCRIPT = "scripts/paper/tables.py"
CONF = "results/stage_r4/confirmatory/results.json"
J = "results/r5_6/results.json"


def _write(S: Sources, artifact: str, name: str, body: str) -> None:
    """tables/<name>.tex, headed by the sources registered for `artifact` (also recorded under the file's name)."""
    d = OUT / "tables"
    d.mkdir(parents=True, exist_ok=True)
    S.by_artifact[f"table:{name}"] = dict(S.by_artifact.get(artifact, {}))
    (d / f"{name}.tex").write_text(ascii_only(name, S.header(f"table:{name}", SCRIPT) + body))


def _table(caption: str, label: str, spec: str, head: str, rows: list[str], note: str = "",
           size: str = "\\small") -> str:
    out = ["\\begin{table}[t]", "\\centering", size, f"\\caption{{{caption}}}",
           f"\\label{{{label}}}", f"\\begin{{tabular}}{{{spec}}}", "\\toprule", head + " \\\\", "\\midrule"]
    out += [r + " \\\\" if r != "\\midrule" else r for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    if note:
        out += [f"\\par\\smallskip\\begin{{minipage}}{{0.95\\linewidth}}\\footnotesize {note}\\end{{minipage}}"]
    return "\n".join(x for x in out if x) + "\n\\end{table}\n"


def yes(b: bool) -> str:
    return "yes" if b else "no"


# ------------------------------------------------------------------------------------------- Setup: the pair sets
def pair_sets(S: Sources) -> None:
    A = "tab_pair_sets"
    pc = S.yaml(A, "configs/pairs.yaml")
    man = lambda rel: S.json(A, f"data/eval_sets/2.0.0+bdbb2279/{rel}_manifest.json")
    rows = []
    spec = [  # (name, decision, changes within a pair, held fixed, ideal slope, regressor, files)
        ("A6", "D58", "the world (values); same template", "template, phrasing, level, $z$", "1",
         r"$\Delta\ln\sigma^\star$", ["a6_pairs", "confirmatory/a6_pairs"]),
        ("A6b", "D59", "the target's magnitude ($\\geq\\times$" + num(pc["invariance_construction"]["min_target_ratio"], 0)
         + "); power-law template", "relative $\\sigma^\\star$ (MC-checked)", "0", r"$\Delta\ln y$",
         ["a6b_pairs", "confirmatory/a6b_pairs"]),
        ("A6c", "D95", "which given carries the high level", "displayed values, level multiset", "1",
         r"$\Delta\ln\sigma^\star$", ["a6c_pairs", "confirmatory/a6c_pairs"]),
        ("J sensitivity", "D147", "as A6c, on a Jacobian outside the training range", "as A6c", "1",
         r"$\Delta\ln\sigma^\star$", ["confirmatory/r5_6/a6c_pairs"]),
        ("J invariance", "D147", "a dimensional rescaling (target $\\times$" + integer(min(f for f in pc["j_invariance_construction"]["scale_factors"] if f > 1))
         + "--" + integer(max(pc["j_invariance_construction"]["scale_factors"])) + ")",
         "every log-sensitivity, so relative $\\sigma^\\star$", "0", r"$\Delta\ln y$", ["confirmatory/r5_6/a6b_pairs"]),
    ]
    for name, dec, change, fixed, ideal, reg, files in spec:
        counts = " / ".join(integer(man(f)["n_problems"]) for f in files)
        rows.append(f"{name} & {change} & {fixed} & {reg} & {ideal} & {counts}")
    note = ("Ideal: the within-pair OLS slope of $\\Delta\\ln$(stated $\\sigma$) on the regressor for an exact "
            "propagator; a reader of surface features that are identical within a pair scores 0 on A6c and J. "
            "Problems: development / frozen confirmatory set (A6, A6b, A6c; D113), or the frozen J set (D147). "
            "Sources: the sets' manifests.")
    _write(S, A, "pair_sets", _table("The counterfactual pair sets and what each isolates.", "tab:pairs",
                                  "@{}L{2.1cm}L{3.6cm}L{3.1cm}cc r@{}",
                                  "set & changes within a pair & held fixed & regressor & ideal & problems", rows, note))


# ------------------------------------------------------------------------------------------- Protocol: deviations
def deviations(S: Sources) -> None:
    A = "tab_deviations"
    tl = S.text(A, "docs/decision_timeline.md")
    rows = []
    for line in tl.splitlines():
        if not line.startswith("| 10-"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        date, entry, what, kind = cells[0], cells[1], cells[2], cells[3]
        if not re.search(r"amendment|deviation|\*\*after\*\*", kind):
            continue
        clean = lambda s: re.sub(r"\\([*_#])", r"\1", re.sub(r"\[(#\d+)\]\[f\d+\]", r"\1",
                                                               re.sub(r"`[0-9a-f]{7}`", "", s)).replace("**", "\x00"))
        entry = re.sub(r"\s*\(.*?\)", "", clean(entry))
        what = clean(what)
        if len(what) > 150:
            what = what[:147].rsplit(" ", 1)[0] + " …"
        bold = lambda t: re.sub("\x00(.+?)\x00", r"\\textbf{\1}", t).replace("\x00", "")  # a cut-off marker
        rows.append(f"{tex_escape(entry)} & {tex_escape(date)} & {bold(tex_escape(what))} & {bold(tex_escape(clean(kind)))}")
    note = ("Every amendment and deviation in the decision timeline (dates IST, 2026), with when it was made relative "
            "to the data it affects. Source: docs/decision\\_timeline.md.")
    head = "entry & date & what & timing \\\\"
    body = ["\\begingroup\\footnotesize", "\\begin{longtable}{@{}l l L{7.8cm} L{3.9cm}@{}}",
            "\\caption{Amendments and deviations from the pre-registered plans.}\\label{tab:deviations} \\\\",
            "\\toprule", head, "\\midrule", "\\endfirsthead", "\\toprule", head, "\\midrule", "\\endhead",
            "\\midrule", f"\\multicolumn{{4}}{{@{{}}p{{15.5cm}}@{{}}}}{{{note}}} \\\\", "\\endlastfoot"]
    body += [r + " \\\\" for r in rows]
    body += ["\\end{longtable}", "\\endgroup", ""]
    _write(S, A, "deviations", "\n".join(body))


# ------------------------------------------------------------------------------------------- R4-3 confirmatory (#32)
K_NAMES = {"K1": "D-pos-v2 (a): A6c slope $>0$ (primary)", "K2": "D-pos-v2 (b) line's $X$: A6c slope $>0$",
           "K3": "D-pos-v2 (c) exact from terms: A6c slope $>0$", "K4": "B-v2: A6c slope $>0$",
           "K5": "D-pos-v2 $-$ B-v2 (expected crps\\_log)", "K6": "D-pos-v2 $-$ RSS", "K7": "D-pos-v2 $-$ X1",
           "K8": "B-v2 $-$ RSS"}


def confirmatory(S: Sources) -> None:
    A = "tab_confirmatory"
    r = S.json(A, CONF)
    t, pr = r["main"]["tests"], r["primary"]
    rows = []
    for k, name in K_NAMES.items():
        nd = 3 if k in ("K1", "K2", "K3", "K4") else 4
        rows.append(f"{k} & {name} & {est_ci(t[k], nd=nd)} & {pval(t[k]['p_one_sided'])} & "
                    f"{'rejected' if t[k]['holm']['rejected'] else '--'}")
    crit = r["main"]["criteria"]
    rows.append("\\midrule")
    for src, label in (("dpos2_a", "D-pos-v2 (a)"), ("dpos2_b", "D-pos-v2 (b)"), ("dpos2_c", "D-pos-v2 (c)"),
                       ("b2_a", "B-v2 (a)")):
        c = crit[src]
        rows.append(f"C1--C3 & {label}: A6b slope {est_ci(c['a6b'])} & C1 {yes(c['C1'])}, C2 {yes(c['C2'])}, "
                    f"C3 {yes(c['C3'])} & & ")
    note = (f"Seed means over {integer(len(r['seeds']['dpos2']))} seeds per arm, frozen R4-5 set "
            f"({integer(r['n_problems'])} problems); stage B's joint pairs bootstrap, $B={integer(r['bootstrap'])}$; "
            "90\\% CIs; one-sided $p$; Holm over K1--K8 at 0.05. Expected-score rows: oracle-centred expected "
            "crps\\_log, pooled over the three sets, clustered over pairs. "
            f"P = K1 $\\wedge$ C2 $\\wedge$ C3 (D125): {'confirmed' if pr['confirmed'] else 'not confirmed'}. "
            f"Source: {tex_escape(CONF)} (D118, D125; FINDINGS \\#32).")
    _write(S, A, "confirmatory", _table("The R4-3 confirmatory run: D-pos-v2 and B-v2 on the frozen set.", "tab:confirmatory",
                                     "@{}l L{5.2cm} l l l@{}", "test & statistic & estimate [90\\% CI] & $p$ & Holm",
                                     rows, note))


# ------------------------------------------------------------------------------------------- R5-6 part J (#39)
J_NAMES = {"J1": "D-pos-v2 (a): J-sensitivity slope $>0$ (primary)", "J2": "explicit $-$ implicit slope",
           "J3": "D-pos-v2 $-$ RSS (expected crps\\_log)", "J4": "D-pos-v2 $-$ X1", "J5": "B-v2 $-$ RSS",
           "J6": "B-v2 $-$ X1"}


def j_results(S: Sources) -> None:
    A = "tab_j"
    r = S.json(A, J)
    m, pr = r["main"], r["primary"]
    rows = []
    for k, name in J_NAMES.items():
        nd = 3 if k in ("J1", "J2") else 4
        rows.append(f"{k} & {name} & {est_ci(m['tests'][k], nd=nd)} & {pval(m['tests'][k]['p_one_sided'])} & "
                    f"{'rejected' if m['tests'][k]['holm']['rejected'] else '--'}")
    rows.append("\\midrule")
    for src, label in (("dpos2_a", "D-pos-v2 (a)"), ("dpos2_b", "D-pos-v2 (b)"), ("dpos2_c", "D-pos-v2 (c)"),
                       ("b2_a", "B-v2 (a)")):
        c = m["criteria"][src]
        rows.append(f"source & {label} & sens.\\ {est_ci(c['a6c'])}; inv.\\ {est_ci(c['a6b'])} & "
                    f"\\multicolumn{{2}}{{l}}{{C1--C3: {yes(c['C1'] and c['C2'] and c['C3'])}}}")
    for n, label in (("sigma_star", "$\\sigma^\\star$"), ("rss", "RSS"), ("x1", "X1"),
                     ("clamped", "clamped to the training range")):
        c = m["comparators"][n]
        rows.append(f"comparator & {label} & sens.\\ {est_ci(c['sensitivity'])} & & ")
    note = (f"Seed means over {integer(len(r['seeds']['dpos2']))} seeds per arm on the frozen J set "
            f"({integer(r['n_problems'])} problems); bootstrap as Table~\\ref{{tab:confirmatory}}; Holm over J1--J6. "
            "C3 uses computation-correct invariance pairs. "
            f"P$_J$ = J1 $\\wedge$ C2 $\\wedge$ C3 (D147): {'confirmed' if pr['confirmed'] else 'not confirmed'}. "
            "The clamped comparator propagates exactly but with every sensitivity clipped into its chain's training "
            f"range. Source: {tex_escape(J)} (D147; FINDINGS \\#39).")
    _write(S, A, "j_results", _table("R5-6 part J: Jacobians outside the training range of trained chains.", "tab:j",
                                  "@{}l L{4.3cm} L{5.6cm} l l@{}", "test & statistic & estimate [90\\% CI] & $p$ & Holm",
                                  rows, note))
    # the pre-specified subgroups where clamping matters (descriptive)
    rows = []
    for key, label in (("exceedance_below_median", "exceedance below the median"),
                       ("exceedance_above_median", "exceedance above the median"), ("strict", "strict"),
                       ("rank_flip_sensitivity", "rank flip")):
        sp = r["splits"][key]
        s = sp["slopes"]
        rows.append(f"{label} & {integer(sp['n_pairs'])} & {est_ci(s['dpos2_a']['sensitivity'], nd=2)} & "
                    f"{est_ci(s['b2_a']['sensitivity'], nd=2)} & {est_ci(s['comparator_clamped']['sensitivity'], nd=2)}")
    note = ("Descriptive, outside the Holm family (D147 \\S10). Pairs: J-sensitivity and J-invariance pairs in the "
            "subgroup; slopes on the J-sensitivity pairs. Strict: also outside the range over every training chain to "
            f"the same target. Rank flip: the dominant given was never dominant for that chain in training. Source: {tex_escape(J)}.")
    _write(S, A, "j_subgroups", _table("R5-6 part J: J-sensitivity slopes where clamping matters most.", "tab:jsub",
                                    "@{}l r l l l@{}", "subgroup & pairs & D-pos-v2 (a) & B-v2 (a) & clamped", rows, note,
                                    size="\\footnotesize"))


# ------------------------------------------------------------------------------------------- Llama pilot (#38)
def llama(S: Sources) -> None:
    A = "tab_llama"
    p = S.json(A, "results/stage_xm/pilot/results.json")
    go = S.json(A, "results/stage_xm/go_dpos2/go.json")
    ll, qw = p["models"]["llama"], p["models"]["qwen_seed0"]
    a = ll["arms"]["dpos2"]["sources"]["a"]
    rows = [f"K1, C2 & D-pos-v2 (a): A6c slope & {num(a['a6c_slope'], 3, True)} {ci(*a['a6c_ci90'])} "
            f"({integer(a['a6c_n_pairs'])} pairs) & {yes(go['K1'])}, {yes(go['C2'])}",
            f"C3 & D-pos-v2 (a): A6b slope, computation-correct pairs & {integer(go['a6b_n_pairs'])} pairs: "
            f"not measurable & {yes(go['C3'])}",
            f"go & K1 $\\wedge$ C2 $\\wedge$ C3 (D137) & & {'GO' if go['go'] else 'NO-GO'}", "\\midrule"]
    d = a["a6b_answered_pairs_slope"]
    qd = qw["arms"]["dpos2"]["sources"]["a"]["a6b_answered_pairs_slope"]
    rows.append(f"desc. & A6b slope, all answered pairs (Llama / Qwen s0) & {num(d['slope'], 3, True)} {ci(*d['ci90'])} "
                f"/ {num(qd['slope'], 3, True)} & ")
    b = ll["arms"]["b2"]["sources"]["a"]
    rows.append(f"desc. & B-v2 (a): A6c slope & {num(b['a6c_slope'], 3, True)} {ci(*b['a6c_ci90'])} & ")
    dd = ll["differences"]["dpos2_minus_b2"]
    rows.append(f"desc. & D-pos-v2 $-$ B-v2 (expected crps\\_log) & {num(dd['value'], 4, True)} {ci(*dd['ci90'], nd=4)} & ")
    for arm, label in (("dpos2", "D-pos-v2"), ("b2", "B-v2")):
        fa = ll["arms"][arm]["format_arithmetic"]["all"]
        rows.append(f"format & {label}: parsed / units / computation-correct & {pct(fa['parsed'])} / "
                    f"{pct(fa['units_ok'])} / {pct(fa['computation_correct'])} & ")
    note = ("Seed 0, frozen A6c + A6b (D137, D146). The go rule is R4-3's criterion on D-pos-v2's answer $\\sigma$; "
            "C3 needs at least three computation-correct A6b pairs. Descriptive rows are D137's, outside the go rule. "
            "Sources: results/stage\\_xm/pilot/results.json, results/stage\\_xm/go\\_dpos2/go.json (FINDINGS \\#38).")
    _write(S, A, "llama", _table(f"The cross-model pilot on {tex_escape('Llama-3.2-3B-Instruct')}.", "tab:llama",
                              "@{}l L{5.6cm} L{5.0cm} l@{}", "item & statistic & value & met", rows, note))


# ------------------------------------------------------------------------------------------- tail autopsy (#35)
def tail(S: Sources) -> None:
    A = "tab_tail"
    t = S.json(A, "results/r5_diag/tail.json")
    pooled = t["causes_worst_pooled"]
    n = sum(pooled.values())
    group = lambda c: ("copy: decimal shift" if c.startswith("copy: decimal shift") else c)
    agg: dict = {}
    for c, k in pooled.items():
        agg[group(c)] = agg.get(group(c), 0) + k
    rows = [f"{tex_escape(c)} & {integer(k)} & {pct(k / n, 0)}" for c, k in sorted(agg.items(), key=lambda x: -x[1])]
    shares = [v["worst_regret_share"] for v in t["per_seed"].values()]
    note = (f"D-pos-v2 (a), frozen R4-5 set, {integer(len(t['per_seed']))} seeds; the worst 5\\% of problems by regret "
            f"against $\\sigma^\\star$ per seed ({integer(n)} problem-seeds), classified by primary cause. They hold "
            f"{pct(min(shares), 0)}--{pct(max(shares), 0)} of each seed's regret. Descriptive (R5-2, D133). "
            "Source: results/r5\\_diag/tail.json (FINDINGS \\#35).")
    _write(S, A, "tail", _table("Tail autopsy: what the worst problems get wrong.", "tab:tail", "@{}l r r@{}",
                             "primary cause & problem-seeds & share", rows, note))
    rows = []
    gap = t["rss_to_star_gap"]
    for k in ("a", "S", "L", "A", "C", "AC", "exact"):
        s = t["swaps"][k]
        mr = s["minus_rss"]
        sl = s["a6c_slope"]
        slope = num(sl["value"], 3, True) if isinstance(sl, dict) and sl.get("value") is not None else "--"
        mrs = "--" if mr is None else f"{num(mr['value'], 4, True)} {ci(*mr['ci90'], nd=4)}"
        rows.append(f"{tex_escape(s['name'])} & {mrs} & {pct(s['share_of_rss_to_star_gap'], 0)} & {slope}")
    note = ("One component of D-pos-v2's $\\sigma$ replaced by its exact value at a time (R5-3, D133): a diagnostic of "
            "where the loss comes from, never a method. Slopes on every A6c pair, an unanswered member at the floor. "
            f"RSS-to-$\\sigma^\\star$ gap {num(gap, 5)} (expected crps\\_log). "
            "Source: results/r5\\_diag/tail.json (FINDINGS \\#35).")
    _write(S, A, "swaps", _table("Component swaps: which part of the explicit $\\sigma$ costs the score.", "tab:swaps",
                              "@{}L{6.2cm} l r r@{}", "$\\sigma$ with & $-$ RSS [90\\% CI] & gap closed & A6c slope",
                              rows, note))


# ------------------------------------------------------------------------------------------- cost (#37)
SC_LABELS = {"self_consistency": "self-consistency (agreement of the other samples)", "probe": "hidden-state probe",
             "random_init_probe": "the same probe, random-init model", "length": "completion length",
             "x4_sft": "self-verification P(yes) (X4)", "sft_sigma": "stated $\\sigma$ (SFT)"}


def cost(S: Sources) -> None:
    A = "tab_cost"
    c = S.json(A, "results/r5_diag/cost.json")
    x2 = c["x2"]
    pp = x2["per_problem"]
    rows = []
    for k, label in SC_LABELS.items():
        m = x2["methods"][k]
        tok = integer(round(pp["tokens_k"])) if k == "self_consistency" else integer(round(pp["tokens_one"]))
        rows.append(f"{label} & {num(m['pairwise_accuracy'], 3)} {ci(*m['ci95'], sign=False)} & {tok}")
    note = (f"Within-problem pairwise accuracy (a right answer ranked above a wrong one to the same problem; "
            f"{integer(x2['methods']['self_consistency']['n_problems'])} problems, 95\\% CIs clustered over problems) "
            f"against generated tokens per problem (k = {integer(x2['k'])} samples for self-consistency). "
            "Source: results/r5\\_diag/cost.json (FINDINGS \\#22, \\#37).")
    _write(S, A, "cost", _table("Correctness signals against their inference cost.", "tab:cost", "@{}L{6.4cm} l r@{}",
                             "readout & pairwise accuracy [95\\% CI] & tokens / problem", rows, note))
    rows = []
    ma = c["mc_a"]["a6c_pairs"]
    for name, v in (("$\\sigma^\\star$", ma["comparators_auroc"]["sigma_star"]), ("RSS", ma["comparators_auroc"]["rss"]),
                    ("X1", ma["comparators_auroc"]["x1"]), ("B-v2 stated $\\sigma$", ma["stated_sigma_auroc"])):
        rows.append(f"{name} & {num(v, 3)} & {integer(0)}")
    for k, v in ma["k"].items():
        rows.append(f"MC-A raw spread, $k={integer(int(k))}$ & {num(v['raw_auroc'], 3)} & {integer(round(v['extra_tokens_per_problem']))}")
    sp = c["mcb_split"]["a6c_pairs"]["share"]
    note = ("A6c (development), AUROC of the signal for B-v2's run-0 answer being wrong, against the extra generated "
            "tokens per problem. MC-B's programs on the same set: "
            f"{pct(sp['interface'])} fail at the interface, {pct(sp['runtime'])} at run time, {pct(sp['semantic'])} run "
            f"but compute the wrong chain, {pct(sp['correct'])} are correct. "
            "Source: results/r5\\_diag/cost.json (FINDINGS \\#31, \\#33, \\#37).")
    _write(S, A, "cost_mca", _table("MC-A's spread against its cost, and MC-B's failure split.", "tab:costmca",
                                 "@{}l r r@{}", "signal & AUROC & extra tokens / problem", rows, note))


def build(S: Sources) -> list[str]:
    for f in (pair_sets, deviations, confirmatory, j_results, llama, tail, cost):
        f(S)
    return sorted(p.stem for p in (OUT / "tables").glob("*.tex"))

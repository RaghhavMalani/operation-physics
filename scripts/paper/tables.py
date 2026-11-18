"""The paper's tables, each built from committed result files only (paper/generated/tables/<name>.tex).

Every table starts with a comment block naming its source files and their git blobs (paperlib.Sources.header).
"""

from __future__ import annotations

import re

from paperlib import OUT, UNICODE, Sources, ascii_only, ci, est_ci, integer, num, pct, pval, tex_escape

SCRIPT = "scripts/paper/tables.py"
CONF = "results/stage_r4/confirmatory/results.json"
J = "results/r5_6/results.json"
MINUS = "\\nobreak\\enspace\\textminus\\nobreak\\enspace "  # a text minus with tied spaces in comparison labels (G7-1, D185)


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
        what = clean(what)  # never cut (G7-4, D185)
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
           "K5": f"D-pos-v2{MINUS}B-v2 (expected crps\\_log)", "K6": f"D-pos-v2{MINUS}RSS", "K7": f"D-pos-v2{MINUS}X1",
           "K8": f"B-v2{MINUS}RSS"}


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
J_NAMES = {"J1": "D-pos-v2 (a): J-sensitivity slope $>0$ (primary)", "J2": f"explicit{MINUS}implicit slope",
           "J3": f"D-pos-v2{MINUS}RSS (expected crps\\_log)", "J4": f"D-pos-v2{MINUS}X1", "J5": f"B-v2{MINUS}RSS",
           "J6": f"B-v2{MINUS}X1"}


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
    rows.append(f"desc. & D-pos-v2{MINUS}B-v2 (expected crps\\_log) & {num(dd['value'], 4, True)} {ci(*dd['ci90'], nd=4)} & ")
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
                              "@{}L{5.0cm} L{4.7cm} r r@{}", "$\\sigma$ with & $\\Delta$ expected crps\\_log vs RSS [90\\% CI] & gap closed & A6c slope",
                              rows, note))


# ------------------------------------------------------------------------------------------- cost (#37)
SC_LABELS = {"self_consistency": "self-consistency (agreement of the other samples)", "probe": "hidden-state probe",
             "random_init_probe": "the same probe, random-init model", "length": "completion length",
             "x4_sft": "self-verification P(yes) (X4)", "sft_sigma": "stated $\\sigma$ (SFT)"}


def cost(S: Sources) -> None:
    A = "tab_cost"
    c = S.json(A, "results/r5_diag/cost.json")
    x4 = S.json(A, "results/x4/results.json")["tokens"]["sft"]  # D91: tokens per checked answer beyond the answer
    x2 = c["x2"]
    pp = x2["per_problem"]
    rows = []
    for k, label in SC_LABELS.items():
        m = x2["methods"][k]
        tok = integer(round(pp["tokens_k"])) if k == "self_consistency" else integer(round(pp["tokens_one"]))
        extra = "--"
        if k == "x4_sft":  # R7-5 (D185): the verification overhead, shown apart from the answer's own tokens
            extra = (f"{integer(round(x4['total_extra']['mean']))} ({integer(round(x4['generated']['mean']))} generated "
                     f"+ {integer(sum(x4['overhead'].values()))} prompt)")
        rows.append(f"{label} & {num(m['pairwise_accuracy'], 3)} {ci(*m['ci95'], sign=False)} & {tok} & {extra}")
    note = (f"Within-problem pairwise accuracy (a right answer ranked above a wrong one to the same problem; "
            f"{integer(x2['methods']['self_consistency']['n_problems'])} problems, 95\\% CIs clustered over problems). "
            "Answer-generation tokens: mean generated tokens of the answers a readout needs, one answer or, for "
            f"self-consistency, k = {integer(x2['k'])} answers per problem. Extra tokens: mean tokens per checked answer "
            "beyond the answer itself; for X4, its verification prompt plus the generated verdict (D91). The probes and "
            "length read the answer already generated. "
            "Sources: results/r5\\_diag/cost.json, results/x4/results.json (FINDINGS \\#21, \\#22, \\#37).")
    _write(S, A, "cost", _table("Correctness signals against their inference cost.", "tab:cost",
                                "@{}L{4.4cm} l L{2.0cm} L{3.0cm}@{}",
                                "readout & pairwise accuracy [95\\% CI] & answer-generation tokens & extra tokens",
                                rows, note))
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


# ------------------------------------------------------------------------------------------- appendices (D185)
def _greek(s: str) -> str:
    """Ledger prose is plain ASCII: sigma*, sigma and P_J become math."""
    s = re.sub(r"\b([A-Z]+\d)-([A-Z]+\d)\b", "\\1\x04\\2", s)  # ranges: P1-P3 -> P1--P3
    s = s.replace("1-sigma", "\x05")
    s = tex_escape(s.replace("PREREGISTRATION.md", "pre-registration").replace("sigma*", "\x01").replace("sigma", "\x02").replace("P_J", "\x03"))
    return (s.replace("\x01", r"$\sigma^\star$").replace("\x02", r"$\sigma$").replace("\x03", r"P$_J$")
            .replace("\x04", "--").replace("\x05", r"$1\sigma$"))


def _longtable(spec: str, caption: str, label: str, head: str, rows: list[str], note: str, width: str) -> str:
    ncol = head.count("&") + 1
    body = ["\\begingroup\\footnotesize", f"\\begin{{longtable}}{{{spec}}}",
            f"\\caption{{{caption}}}\\label{{{label}}} \\\\",
            "\\toprule", head + " \\\\", "\\midrule", "\\endfirsthead", "\\toprule", head + " \\\\", "\\midrule", "\\endhead",
            "\\midrule", f"\\multicolumn{{{ncol}}}{{@{{}}p{{{width}}}@{{}}}}{{{note}}} \\\\", "\\endlastfoot"]
    return "\n".join(body + [r + " \\\\" for r in rows] + ["\\end{longtable}", "\\endgroup", ""])


def ledger(S: Sources) -> None:
    """Appendix A: every pre-registered hypothesis with its verdict (docs/hypothesis_ledger.yaml; R7-3)."""
    A = "tab_ledger"
    L = S.yaml(A, "docs/hypothesis_ledger.yaml")
    rows = [" & ".join(_greek(str(r[k])) for k in ("tests", "entry", "question", "label", "verdict", "findings"))
            for r in L["rows"]]
    note = ("Every pre-registered hypothesis or test family, in the order of the pre-registration and the decision log, "
            "with its verdict as the cited FINDINGS entry states it. Supported or rejected: the registered one-sided "
            "test passed (Holm-adjusted where the family is); not rejected: it did not, which is not evidence of "
            "equivalence. Source: docs/hypothesis\\_ledger.yaml.")
    _write(S, A, "ledger", _longtable("@{}L{1.7cm} L{1.8cm} L{3.7cm} L{1.9cm} L{3.7cm} L{1.2cm}@{}",
                                      "The pre-registration ledger.", "tab:ledger",
                                      "tests & entry & question & label & verdict & finding", rows, note, "15.4cm"))


FROZEN = [  # (set, manifest, split key or None, what it is for)
    ("in-distribution, noise-free", "data/eval_sets/1.1.0+c735a0d4/manifest.json", "id", "Results I"),
    ("depth 5, noise-free", "data/eval_sets/1.1.0+c735a0d4/manifest.json", "depth_extrapolation", "Results I"),
    ("X2 candidates", "data/eval_sets/1.1.0+c735a0d4/x2_candidates_manifest.json", None, "Results II"),
    ("R4-5 A6", "data/eval_sets/2.0.0+bdbb2279/confirmatory/a6_pairs_manifest.json", None, "Results III"),
    ("R4-5 A6b", "data/eval_sets/2.0.0+bdbb2279/confirmatory/a6b_pairs_manifest.json", None, "Results III"),
    ("R4-5 A6c", "data/eval_sets/2.0.0+bdbb2279/confirmatory/a6c_pairs_manifest.json", None, "Results III"),
    ("J sensitivity", "data/eval_sets/2.0.0+bdbb2279/confirmatory/r5_6/a6c_pairs_manifest.json", None, "Generalization"),
    ("J invariance", "data/eval_sets/2.0.0+bdbb2279/confirmatory/r5_6/a6b_pairs_manifest.json", None, "Generalization"),
]


def frozen_sets(S: Sources) -> None:
    """Appendix A: the frozen evaluation sets, their manifests and SHA-256 checksums (R7-3)."""
    A = "tab_frozen"
    rows = []
    for name, rel, split, used in FROZEN:
        m = S.json(A, rel)
        d = m["splits"][split] if split else m
        n = d.get("n_problems", d.get("n"))
        sha = d["sha256"]
        rows.append(f"{tex_escape(name)} & {tex_escape(m['generator_version'])} & {integer(n)} & {used} & "
                    f"\\texttt{{{sha[:16]}\\ldots{{}}}} \\\\ \\multicolumn{{5}}{{@{{}}l@{{}}}}{{\\quad\\scriptsize\\path{{{rel}}}}}")
    note = ("Each set is regenerated from its seeds and compared with its file before any generation; the loader "
            "refuses the confirmatory sets without an explicit flag (D113). The first 16 hexadecimal digits of each "
            "file's SHA-256 are shown; the full digest is in the manifest named under it.")
    _write(S, A, "frozen", _table("The frozen evaluation sets.", "tab:frozen", "@{}l l r l l@{}",
                                  "set & generator & problems & used in & SHA-256", rows, note, size="\\footnotesize"))


def timeline(S: Sources) -> None:
    """Appendix C: every row of the decision timeline (docs/decision_timeline.md; R5-8, R7-3)."""
    A = "tab_timeline"
    tl = S.text(A, "docs/decision_timeline.md")
    clean = lambda s: re.sub(r"\\([*_#])", r"\1", re.sub(r"\[(#\d+)\]\[f\d+\]", r"\1", s)).replace("**", "\x00")
    bold = lambda x: re.sub("\x00(.+?)\x00", r"\\textbf{\1}", x).replace("\x00", "")
    rows = []
    for line in tl.splitlines():
        if not line.startswith("| 10-"):
            continue
        date, entry, what, kind, analysis, _ = [c.strip() for c in line.strip("|").split("|")]
        commit = re.search(r"`([0-9a-f]{7})`", entry)
        entry = re.sub(r"\s*\(.*?\)", "", clean(entry))
        what = clean(what)  # never cut (G7-4, D185)
        entry = entry.replace("PREREGISTRATION.md", "pre-registration")
        cells = [tex_escape(entry) + (f" \\texttt{{{commit.group(1)}}}" if commit else ""), tex_escape(date),
                 bold(tex_escape(what)), bold(tex_escape(clean(kind))), bold(tex_escape(clean(analysis)))]
        rows.append(" & ".join(cells))
    note = ("Every pre-registration and amendment in date order (IST, 2026), with its first commit, what it fixed, its "
            "timing relative to the data it governs, and the analysis label. Confirmatory: committed before its data "
            "and run as registered; pilot: the same at one seed; exploratory: post hoc, descriptive or diagnostic. "
            "Results before the pre-registration (FINDINGS \\#1--\\#6) are exploratory. Source: "
            "docs/decision\\_timeline.md.")
    _write(S, A, "timeline", _longtable("@{}L{1.8cm} l L{5.0cm} L{3.1cm} L{3.1cm}@{}", "The decision timeline.",
                                        "tab:timeline", "entry & date & what it fixed & kind & analysis", rows, note,
                                        "15.4cm"))


TT = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#", "^": r"\^{}", "_": r"\_",
      "%": r"\%", "~": r"\~{}", "<": r"\textless{}", ">": r"\textgreater{}", **UNICODE}


def _tt(s: str) -> str:
    """A completion line, verbatim in typewriter type (no math substitutions for * or ^)."""
    return "".join(TT.get(c, c) for c in s)


def worked_example(S: Sources) -> None:
    """G7-11 (D185): one frozen A6c pair and D-pos-v2's completion, chosen by configs/paper.yaml's rule."""
    import ast
    import gzip
    import json
    import math
    from collections import defaultdict

    A = "tab_example"
    cfg = S.yaml(A, "configs/paper.yaml")["worked_example"]
    tau = S.yaml(A, "configs/pairs.yaml")["analysis"]["tau"]
    pairs = defaultdict(dict)
    with gzip.open(S.path(A, cfg["tape"]), "rt") as fh:
        for line in fh:
            r = json.loads(line)
            pr = r["problem"] if isinstance(r["problem"], dict) else ast.literal_eval(r["problem"])
            pa = pr["noise"]["pair"]
            if pa.get("kind") != cfg["kind"] or pr["depth"] != cfg["depth"] or pa["construction"]["levels"] != cfg["levels"]:
                continue
            b = r["breakdown"] if isinstance(r["breakdown"], dict) else ast.literal_eval(r["breakdown"])
            pairs[pa["pair_seed"]][pa["member"]] = (r, pr, b)
    ok = lambda m: m[2]["reason"] == "ok" and (m[2].get("mu_si") or 0) > 0 and \
        abs(math.log(m[2]["mu_si"] / m[1]["noise"]["point_estimate_si"])) <= tau
    cands = sorted((len(m["lo"][0]["completion"]) + len(m["hi"][0]["completion"]), seed)
                   for seed, m in pairs.items() if set(m) == {"lo", "hi"} and ok(m["lo"]) and ok(m["hi"]))
    if cfg["pick"] != "shortest" or not cands:
        raise ValueError("worked_example: no pair matches the rule")
    seed = cands[0][1]
    m = pairs[seed]
    para = lambda pr, i: [x for x in pr["text"].split("\n\n") if x.strip()][i]
    hi_given = lambda pr: max(pr["noise"]["givens"], key=lambda g: pr["noise"]["givens"][g]["rel"])
    r_lo, p_lo, b_lo = m["lo"]
    r_hi, p_hi, b_hi = m["hi"]
    rows = []
    for k, (r, pr, b) in enumerate((m["lo"], m["hi"]), 1):
        rows.append(f"{k} & \\texttt{{{_tt(hi_given(pr))}}} & {num(pr['noise']['sigma_star_log'], 3)} & "
                    f"{num(b['sigma_si'] / b['mu_si'], 3)} \\\\")
    # the caption's claim, checked: apart from the propagation line and the answer, the two completions are identical
    strip = lambda c: [ln for ln in c.splitlines() if not ln.startswith(("sigma_rel", "<answer>"))]
    if strip(r_lo["completion"]) != strip(r_hi["completion"]):
        raise ValueError("worked_example: the two completions differ outside the propagation line and the answer")
    comp = "\\\\\n".join(_tt(ln) if ln.strip() else "\\mbox{}" for ln in r_hi["completion"].splitlines())
    body = [
        "\\begin{figure}[tp]", "\\small",
        "\\fbox{\\begin{minipage}{\\dimexpr\\linewidth-2\\fboxsep-2\\fboxrule\\relax}",
        f"\\textbf{{The pair}} (frozen A6c set, {tex_escape(p_lo['family'])}, depth {integer(p_lo['depth'])}, pair seed "
        f"\\texttt{{{seed}}}). The two members share the scenario, every displayed value and the question; only which "
        f"given carries {pct(cfg['levels'][1], 0)} changes (every other given carries {pct(cfg['levels'][0], 0)}).\\par\\smallskip",
        f"\\emph{{Member 1:}} {tex_escape(para(p_lo, 1))}\\par\\smallskip",
        f"\\emph{{Member 2:}} {tex_escape(para(p_hi, 1))}\\par\\smallskip",
        f"\\emph{{Question (both):}} {tex_escape(para(p_hi, -1))}\\par\\medskip",
        "\\textbf{D-pos-v2's completion for member 2} (seed 0, greedy, verbatim):\\par\\smallskip",
        "{\\ttfamily\\footnotesize\\raggedright " + comp + "\\par}\\medskip",
        "\\centering\\begin{tabular}{@{}c l r r@{}}", "\\toprule",
        "member & given at the higher level & $\\sigma^\\star$ (log scale) & stated relative $\\sigma$ \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}",
        "\\end{minipage}}",
        "\\caption{A worked example: one A6c pair and D-pos-v2's answer. D-pos-v2's two completions are identical except "
        "for the propagation line and the answer: the derivation and the sensitivities $S_i$ are the same, only the level "
        "each $S_i$ is multiplied by moves, and the stated $\\sigma$ moves with $\\sigma^\\star$. Chosen by a fixed rule (configs/paper.yaml): the depth-" + integer(cfg["depth"]) +
        " pair with levels " + ", ".join(pct(x, 0) for x in cfg["levels"]) + " whose two completions are both "
        "computation-correct and shortest in total. It is illustrative, not typical: most answers are not "
        "computation-correct (Section~\\ref{sec:act3}). The trailing confidence tag is part of the R4 output format "
        "(D83) and is not used here. Source: " + tex_escape(cfg["tape"]) + ".}",
        "\\label{fig:example}", "\\end{figure}", ""]
    _write(S, A, "example", "\n".join(body))


def build(S: Sources) -> list[str]:
    for f in (pair_sets, deviations, confirmatory, j_results, llama, tail, cost, ledger, frozen_sets, timeline,
              worked_example):
        f(S)
    return sorted(p.stem for p in (OUT / "tables").glob("*.tex"))

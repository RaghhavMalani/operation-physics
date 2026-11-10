"""The values the paper's prose uses (\\pv{key}), each read from a committed config, data manifest or result file.

Setup, Methods and the evaluation protocol mostly cite design constants; they are read from the configs and frozen
manifests that defined them, never typed by hand.
"""

from __future__ import annotations

import re

from paperlib import Macros, Sources, ci, integer, num, pct, pval

A = "macros"  # the artifact name in sources.json


def _lv(x: float) -> str:
    return f"\\SI{{{100 * x:g}}}{{\\percent}}"


def _set(values) -> str:
    return ", ".join(_lv(v) for v in values)


def build(S: Sources) -> Macros:
    M = Macros()
    add = M.add

    # ---- models (D78, D137)
    base = S.yaml(A, "configs/base_4bit.yaml")
    llama = S.yaml(A, "configs/base_llama32_3b_4bit.yaml")
    strip = lambda r: r.split("/")[-1].replace("-bnb-4bit", "")
    add("model.base", strip(base["repo_id"]), "configs/base_4bit.yaml repo_id")
    add("model.llama", strip(llama["repo_id"]), "configs/base_llama32_3b_4bit.yaml repo_id")
    add("model.bits", integer(int(re.search(r"(\d+)bit", base["repo_id"]).group(1))), "configs/base_4bit.yaml repo_id (bnb-Nbit)")
    r4 = S.yaml(A, "configs/stage_r4.yaml")
    add("lora.r", integer(r4["lora"]["r"]), "stage_r4 lora.r")
    add("lora.alpha", integer(r4["lora"]["alpha"]), "stage_r4 lora.alpha")
    add("lora.modules", integer(len(r4["lora"]["target_modules"])), "stage_r4 lora.target_modules")

    # ---- the generator (D51, D54; configs/env_v2.yaml, configs/pairs.yaml)
    env = S.yaml(A, "configs/env_v2.yaml")
    gen = env["generator"]
    add("gen.version", gen["version"], "env_v2 generator.version")
    add("gen.maxdepth", integer(max(env["splits"]["depths"])), "env_v2 splits.depths")
    add("gen.heldoutdepth", integer(env["splits"]["heldout_depths"][0]), "env_v2 splits.heldout_depths")
    add("gen.distractors", f"{integer(gen['distractors'][0])}--{integer(gen['distractors'][1])}", "env_v2 distractors")
    add("gen.sigfigs", integer(gen["display"]["sig_figs"]), "env_v2 display.sig_figs")
    add("gen.noisesigfigs", integer(gen["noise"]["sig_figs"]), "env_v2 noise.sig_figs")
    add("gen.kappa", integer(gen["max_condition_number"]), "env_v2 max_condition_number")
    add("gen.nmc", integer(gen["noise"]["n_mc"]), "env_v2 noise.n_mc")
    pc = S.yaml(A, "configs/pairs.yaml")
    fams = pc["export"]["families"]
    add("gen.nfamilies", integer(len(fams)), "pairs.yaml export.families")
    add("gen.families", ", ".join(fams), "pairs.yaml export.families")
    from physcal.env.formulas import get_family  # the committed formula modules
    for f in fams:
        S.path(A, f"physcal/env/formulas/{f}.py")
        add(f"gen.nformulas.{f}", integer(len(get_family(f).formulas)), f"physcal/env/formulas/{f}.py FORMULAS")
    add("gen.levels", _set(pc["export"]["levels"]), "pairs.yaml export.levels (D54)")
    add("gen.maxlevel", _lv(max(pc["export"]["levels"])), "pairs.yaml export.levels (D54)")
    sft_r4 = S.yaml(A, "configs/sft_r4.yaml")
    add("gen.mixedlevels", _set(sft_r4["mixed_levels"]), "sft_r4.yaml mixed_levels (D113)")

    # ---- the self-consistency gate, FINAL run (D57; FINDINGS #12)
    g = S.json(A, "results/sigma_star_gate_final/gate.json")
    gc = g["config"]
    add("gate.n", integer(g["n_noisy"]), "gate.json n_noisy")
    add("gate.ncells", integer(len(gc["depths"]) * len(gc["levels"])), "gate.json config depths x levels")
    add("gate.percell", integer(gc["n_per_cell"]), "gate.json config n_per_cell")
    add("gate.nzero", integer(g["noise_zero"]["n"]), "gate.json noise_zero.n")
    for k, name in (("1", "one"), ("2", "two")):
        pooled = g["primary_sigma_crps"]["levels"][k]["pooled"]
        add(f"gate.cov.{name}", pct(pooled["coverage"], 2), f"gate.json primary_sigma_crps.levels.{k}.pooled.coverage")
        add(f"gate.nominal.{name}", pct(pooled["nominal"], 2), f"gate.json primary_sigma_crps.levels.{k}.pooled.nominal")
        add(f"gate.kupiec.{name}", num(pooled["p"], 2), f"gate.json primary_sigma_crps.levels.{k}.pooled.p")
    add("gate.pitp", num(g["pit"]["ks_p"], 2), "gate.json pit.ks_p")

    # ---- the verifier (D15, D29, D38; configs/reward.yaml)
    rw = S.yaml(A, "configs/reward.yaml")
    sc = rw["scoring"]
    add("ver.version", rw["version"], "reward.yaml version")
    add("ver.floor", num(-sc["crps_clip"], 0), "reward.yaml scoring.crps_clip")
    add("ver.sigmafloor", f"\\num{{{sc['crps_log_sigma_floor']:.0e}}}", "reward.yaml scoring.crps_log_sigma_floor")
    add("ver.binary", _lv(sc["binary_tolerance"]), "reward.yaml scoring.binary_tolerance")
    add("ver.steptol", _lv(rw["steps"]["tolerance"]), "reward.yaml steps.tolerance")
    add("ver.stepweight", num(rw["weights"]["steps"], 1), "reward.yaml weights.steps")
    add("ver.tau", f"\\num{{{pc['analysis']['tau']:.0e}}}", "pairs.yaml analysis.tau (D50/A1)")

    # ---- the oracle-centred expected score (D100; FINDINGS #24)
    sa = S.yaml(A, "configs/stage_a.yaml")
    add("sa.draws", integer(sa["mc"]["n_draws"]), "stage_a.yaml mc.n_draws")
    st = S.json(A, "results/stage_a/results.json")
    vr = [v["pooled"]["variance_ratio_realized_over_expected"] for v in st["resolution"].values()]
    add("sa.varratio.min", num(min(vr), 0), "stage_a results.json resolution.*.pooled.variance_ratio (min)")
    add("sa.varratio.max", num(max(vr), 0), "stage_a results.json resolution.*.pooled.variance_ratio (max)")
    half = st["resolution"]["0.5x_sigma_star"]["pooled"]["realized"]["share_star_better"]
    add("sa.halfwins", pct(1 - half, 0), "stage_a results.json resolution.0.5x_sigma_star realized share (1 - star better)")
    add("sa.scale", integer(round(1 / min(sa["alternatives"]["scales"]))), "stage_a.yaml alternatives.scales")
    tail = S.json(A, "results/r5_diag/tail.json")
    add("gap.rss", num(tail["rss_to_star_gap"], 5), "r5_diag/tail.json rss_to_star_gap")

    # ---- the pair sets (D58, D59, D95, D113, D147)
    for key, con in (("aSix", "construction"), ("aSixB", "invariance_construction"), ("aSixC", "allocation_construction")):
        c = pc[con]
        add(f"pairs.{key}.worlds", integer(c["candidate_worlds"]), f"pairs.yaml {con}.candidate_worlds")
    add("pairs.aSix.ratio", num(pc["construction"]["min_ratio"], 1), "pairs.yaml construction.min_ratio")
    add("pairs.aSixB.ratio", num(pc["invariance_construction"]["min_target_ratio"], 1), "pairs.yaml invariance min_target_ratio")
    add("pairs.aSixB.mc", f"\\num{{{pc['invariance_construction']['max_log_sigma_ratio']:.0e}}}", "pairs.yaml invariance max_log_sigma_ratio")
    add("pairs.aSixC.ratio", num(pc["allocation_construction"]["min_ratio"], 1), "pairs.yaml allocation min_ratio")
    ex = pc["export"]
    ncell = len(ex["depths"]) * len(ex["levels"])
    add("pairs.cells", integer(ncell), "pairs.yaml export depths x levels")
    add("pairs.percell", integer(ex["pairs_per_cell"]), "pairs.yaml export.pairs_per_cell")
    for name, rel in (("aSix", "a6_pairs"), ("aSixB", "a6b_pairs"), ("aSixC", "a6c_pairs")):
        m = S.json(A, f"data/eval_sets/2.0.0+bdbb2279/confirmatory/{rel}_manifest.json")
        add(f"frozen.{name}.problems", integer(m["n_problems"]), f"confirmatory/{rel}_manifest.json n_problems")
    jc = pc["j_allocation_construction"]
    add("j.margin", num(jc["jacobian"]["margin"], 1), "pairs.yaml j_jacobian.margin (D147 sec. 2)")
    add("j.worlds", integer(jc["candidate_worlds"]), "pairs.yaml j_allocation_construction.candidate_worlds")
    sf = pc["j_invariance_construction"]["scale_factors"]
    add("j.scale.max", integer(max(sf)), "pairs.yaml j_invariance_construction.scale_factors")
    add("j.scale.min", integer(min(f for f in sf if f > 1)), "pairs.yaml j_invariance_construction.scale_factors")
    jex = pc["j_allocation_export"]
    add("j.maxdepth", integer(max(jex["depths"])), "pairs.yaml j_allocation_export.depths")
    for name, rel in (("sens", "a6c_pairs"), ("inv", "a6b_pairs")):
        m = S.json(A, f"data/eval_sets/2.0.0+bdbb2279/confirmatory/r5_6/{rel}_manifest.json")
        add(f"j.{name}.pairs", integer(m["n_pairs"]), f"confirmatory/r5_6/{rel}_manifest.json n_pairs")
        add(f"j.{name}.problems", integer(m["n_problems"]), f"confirmatory/r5_6/{rel}_manifest.json n_problems")
        d = m["distances"]
        add(f"j.{name}.exc", num(d["exceedance_q25_50_75"][1], 2), f"r5_6/{rel}_manifest.json distances exceedance median")
        add(f"j.{name}.ntrain", integer(d["n_training_q25_50_75"][1]), f"r5_6/{rel}_manifest.json distances n_training median")
    sup = S.json(A, "data/jacobian_support/r4_1.json")
    add("j.support.chains", integer(len(sup["chains"])), "data/jacobian_support/r4_1.json chains")

    # ---- training (D35, D45/D46, D113)
    sft = S.yaml(A, "configs/sft.yaml")
    add("sft.n", integer(sft["data"]["n_examples"]), "sft.yaml data.n_examples")
    add("sft.sigmarel", _lv(sft["data"]["sigma_rel"]), "sft.yaml data.sigma_rel")
    add("sft.minutes", integer(sft["train"]["wall_clock_minutes"]), "sft.yaml train.wall_clock_minutes")
    add("sft.lr", f"\\num{{{sft['train']['learning_rate']:.0e}}}", "sft.yaml train.learning_rate")
    gr = S.yaml(A, "configs/grpo_split_credit.yaml")
    g2 = gr["grpo"]
    add("grpo.G", integer(g2["num_generations"]), "grpo_split_credit.yaml grpo.num_generations")
    add("grpo.prompts", integer(g2["prompts_per_step"]), "grpo_split_credit.yaml grpo.prompts_per_step")
    add("grpo.lr", f"\\num{{{g2['learning_rate']:.0e}}}", "grpo_split_credit.yaml grpo.learning_rate")
    add("grpo.maxlen", integer(g2["max_completion_length"]), "grpo_split_credit.yaml grpo.max_completion_length")
    add("grpo.beta", num(g2["beta"], 0), "grpo_split_credit.yaml grpo.beta")
    add("grpo.temp", num(g2["temperature"], 1), "grpo_split_credit.yaml grpo.temperature")
    add("grpo.step", integer(g2["permanent_save_steps"][0]), "grpo_split_credit.yaml grpo.permanent_save_steps")
    add("grpo.floor", num(gr["curriculum"]["weight_floor"], 2), "grpo_split_credit.yaml curriculum.weight_floor")
    add("grpo.ema", num(gr["curriculum"]["alpha"], 1), "grpo_split_credit.yaml curriculum.alpha")
    add("split.ref", _lv(gr["split_credit"]["sigma_ref_rel"]), "grpo_split_credit.yaml split_credit.sigma_ref_rel")
    import glob
    add("split.seeds", integer(len(glob.glob(str(S.path(A, "configs/grpo_split_credit.yaml").parent / "grpo_split_credit*.yaml")))),
        "configs/grpo_split_credit*.yaml (one config per seed)")
    e1a, e1d = S.yaml(A, "configs/grpo_e1a.yaml")["grpo"], S.yaml(A, "configs/grpo_e1d.yaml")["grpo"]
    add("eone.suffix", integer(e1a["max_completion_length"]), "grpo_e1a.yaml grpo.max_completion_length")
    add("eone.steps", integer(e1a["max_steps"]), "grpo_e1a.yaml grpo.max_steps")
    add("eone.lr", f"\\num{{{e1a['learning_rate']:.0e}}}", "grpo_e1a.yaml grpo.learning_rate")
    add("eone.lrd", f"\\num{{{e1d['learning_rate']:.0e}}}", "grpo_e1d.yaml grpo.learning_rate")
    t = r4["train"]
    man = S.json(A, "results/phase12_sft_data_r4/manifest.json")
    add("rfour.n", integer(man["n_examples"]), "phase12_sft_data_r4/manifest.json n_examples")
    add("rfour.mixing", integer(man["problems_mixing_noisy_levels"]), "phase12_sft_data_r4/manifest.json")
    summ = S.json(A, "results/stage_r4/dpos2/seed0/summary.json")
    add("rfour.steps", integer(summ["steps_planned"]), "stage_r4/dpos2/seed0/summary.json steps_planned")
    add("rfour.epochs", integer(t["epochs"]), "stage_r4.yaml train.epochs")
    add("rfour.lr", f"\\num{{{t['learning_rate']:.0e}}}", "stage_r4.yaml train.learning_rate")
    add("rfour.batch", integer(t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]),
        "stage_r4.yaml train batch x accumulation")
    add("rfour.tokens", integer(r4["eval"]["long_max_new_tokens"]), "stage_r4.yaml eval.long_max_new_tokens (D114)")
    add("rfour.evalbatch", integer(r4["eval"]["batch_size"]), "stage_r4.yaml eval.batch_size (D116)")
    add("rfour.seeds", integer(len(r4["confirmatory"]["seeds"])), "stage_r4.yaml confirmatory.seeds")
    add("rfour.sfmt", integer(int(sft_r4["sensitivity_format"].strip(".g"))), "sft_r4.yaml sensitivity_format (.Ng)")
    mca = S.yaml(A, "configs/mc_a.yaml")
    add("mca.k", integer(mca["k"]), "mc_a.yaml k")
    add("mca.mad", num(mca["mad_scale"], 4), "mc_a.yaml mad_scale")
    mcb = S.yaml(A, "configs/mc_b.yaml")
    add("mcb.tokens", integer(mcb["generation"]["max_new_tokens"]), "mc_b.yaml generation.max_new_tokens")
    add("mcb.nexamples", integer(len(mcb["prompt"]["examples"]["cells"])), "mc_b.yaml prompt.examples.cells")
    add("mcb.exdepth", integer(mcb["prompt"]["examples"]["cells"][0][1]), "mc_b.yaml prompt.examples.cells depth")
    add("stat.floor", f"\\num{{{pc['analysis']['sigma_floor_rel']:.0e}}}", "pairs.yaml analysis.sigma_floor_rel")

    # ---- statistics (D104, D113, D118, D125, D147)
    an = r4["analysis"]
    add("stat.B", integer(an["bootstrap"]), "stage_r4.yaml analysis.bootstrap")
    add("stat.alpha", num(an["alpha"], 2), "stage_r4.yaml analysis.alpha")
    add("stat.cilevel", pct(an["criterion_ci_level"], 0), "stage_r4.yaml analysis.criterion_ci_level")
    add("stat.practical", num(an["practical_slope"], 1), "stage_r4.yaml analysis.practical_slope (C2)")
    add("stat.margin", num(pc["invariance_analysis"]["equivalence_margin"], 1), "pairs.yaml invariance_analysis.equivalence_margin")
    # ---- the post-freeze exploratory comparison (D166, R6-1): the follow-up track's item 3 (D157-D162), not a result
    # of this paper's frozen experiments; the 7B's numbers stay out (reserved for a separate paper)
    fs = S.json(A, "results/followup/scale/results.json")
    fu = S.yaml(A, "configs/followup.yaml")
    add("post.model.large", fu["scale"]["kaggle"]["repo_id"].split("/")[-1], "followup.yaml scale.kaggle repo_id (D157)")
    add("post.probe.native", num(fs["models"]["1.5b"]["readouts"]["probe"]["pairwise_accuracy"]),
        "followup scale results.json models.1.5b probe pairwise_accuracy (D162)")
    add("post.probe.random", num(fs["models"]["1.5b"]["readouts"]["probe_random"]["pairwise_accuracy"]),
        "followup scale results.json models.1.5b probe_random pairwise_accuracy (D160, D162)")
    add("post.probe.sft", num(fs["x2_sft_on_prefix"]["methods"]["probe"]["pairwise_accuracy"]),
        "followup scale results.json x2_sft_on_prefix probe (D161's refit, the same 1,176 candidates)")

    # ---- the protocol's record
    dec = S.text(A, "DECISIONS.md")
    add("record.dentries", integer(len(re.findall(r"^### D\d+\.", dec, flags=re.M))), "DECISIONS.md: '### D<n>.' headers")
    fin = S.text(A, "FINDINGS.md")
    add("record.findings", integer(max(int(n) for n in re.findall(r"^## #(\d+)\.", fin, flags=re.M))),
        "FINDINGS.md: the highest '## #<n>.' header")
    for part in (_act1, _act2, _act3, _generalization):
        part(S, add)
    return M


# ===================================================================================== the results sections (D178)
def _level(key: str) -> str:
    """A CI's level from the source's own key (ci95 / ci90), never typed."""
    return pct(int(re.fullmatch(r"ci(\d+)", key).group(1)) / 100, 0)


def _ci_named(lo: float, hi: float, key: str, nd: int = 3, sign: bool = True) -> str:
    """'[95\\% CI lo, hi]'."""
    return f"[{_level(key)} CI {num(lo, nd, sign)}, {num(hi, nd, sign)}]"


def _est(d: dict, key: str = "ci90", nd: int = 3, sign: bool = True, value: str = "value") -> str:
    """'+0.816 [90\\% CI +0.707, +0.923]' from a statistic dict."""
    return f"{num(d[value], nd, sign)} {_ci_named(*d[key], key, nd=nd, sign=sign)}"


def _span(values, fmt) -> str:
    lo, hi = min(values), max(values)
    return fmt(lo) if fmt(lo) == fmt(hi) else f"{fmt(lo)}--{fmt(hi)}"


def _act1(S: Sources, add) -> None:
    """Results I: uniform vs split credit, 3 vs 3 seeds (D71, FINDINGS #16, #18); the collapse (#6, #16); the post-hoc
    scale (D69); E1 / E1d (#14, #16, #25; E1d exploratory, R5-7)."""
    A_ = "macros"
    rel = "results/corrected/robustness_3v3/report.json"
    R = S.json(A_, rel)
    a, b = R["A_problem_level_seed_mean"], R["B_seed_level"]
    add("ra.cov.split", pct(a["S1b"]["split"]), f"{rel} A.S1b.split (3-seed mean)")
    add("ra.cov.uniform", pct(a["S1b"]["control"]), f"{rel} A.S1b.control (3-seed mean)")
    add("ra.cov.sft", pct(a["S1a"]["control"]), f"{rel} A.S1a.control")
    d = a["S1b"]
    add("ra.cov.diff", f"{num(100 * d['diff'], 1, True)} percentage points {_ci_named(*[100 * x for x in d['diff_ci']], 'ci95', nd=1)}",
        f"{rel} A.S1b.diff, diff_ci (problem-level bootstrap, 95%)")
    cross = b["tests"]["cov1"]["cross_differences"]
    add("ra.cov.pairs", f"{integer(sum(x > 0 for x in cross))} of {integer(len(cross))}", f"{rel} B.tests.cov1.cross_differences")
    runs = b["per_run"]
    sp, un = list(runs["split"].values()), list(runs["uniform"].values())
    add("ra.cov.split.seeds", _span([r["cov1"] for r in sp], pct), f"{rel} B.per_run.split.*.cov1")
    add("ra.cov.uniform.seeds", _span([r["cov1"] for r in un], pct), f"{rel} B.per_run.uniform.*.cov1")
    add("ra.zero.split", _span([r["sigma_zero"] for r in sp], pct), f"{rel} B.per_run.split.*.sigma_zero")
    add("ra.zero.uniform", _span([r["sigma_zero"] for r in un], pct), f"{rel} B.per_run.uniform.*.sigma_zero")
    add("ra.sigma.split", _span([r["median_rel_sigma"] for r in sp], lambda x: pct(x, 2)), f"{rel} B.per_run.split.*.median_rel_sigma")
    add("ra.sigma.uniform", _span([r["median_rel_sigma"] for r in un], lambda x: pct(x, 3)), f"{rel} B.per_run.uniform.*.median_rel_sigma")
    add("ra.acc.split", pct(a["S3"]["split"]), f"{rel} A.S3.split")
    add("ra.acc.uniform", pct(a["S3"]["control"]), f"{rel} A.S3.control")
    s2 = a["S2"]
    add("ra.rho.split", num(s2["split"]), f"{rel} A.S2.split")
    add("ra.rho.uniform", num(s2["control"]), f"{rel} A.S2.control")
    add("ra.rho.diff", f"{num(s2['diff'], 3, True)} {_ci_named(*s2['diff_ci'], 'ci95')}", f"{rel} A.S2.diff, diff_ci")
    add("ra.rho.p", num(s2["p_one_sided"], 2), f"{rel} A.S2.p_one_sided")
    add("ra.rho.split.seeds", _span([r["rho"] for r in sp], lambda x: num(x, 2)), f"{rel} B.per_run.split.*.rho")
    rhos = b["tests"]["rho"]["cross_differences"]
    add("ra.rho.pairs", f"{integer(sum(x > 0 for x in rhos))} of {integer(len(rhos))}", f"{rel} B.tests.rho.cross_differences")
    # family N on the corrected evals (D49, D50/A5, D67; FINDINGS #16)
    rel = "results/corrected/side_by_side.json"
    sbs = S.json(A_, rel)
    corr = [v for k, v in sbs["per_seed"].items() if k.endswith("corrected")]
    add("ra.nfail", f"{integer(sum(not t['supported'] for c in corr for t in c['N'].values()))} of "
        f"{integer(sum(len(c['N']) for c in corr))}", f"{rel} per_seed.*corrected.N.*.supported")
    add("ra.rho.afive", _span([c["N"]["N2"]["control"] for c in corr], lambda x: num(x, 2)), f"{rel} per_seed.*corrected.N.N2.control")
    key = next(k for k in sbs["extra_headlines"] if k.startswith("#6"))
    col = sbs["extra_headlines"][key]["corrected"]
    add("ra.collapse.step", integer(int(re.search(r"step (\d+)", key).group(1))), f"{rel} extra_headlines '{key}'")
    add("ra.collapse.zero", pct(col["sigma_zero"]), f"{rel} extra_headlines #6 corrected.sigma_zero")
    add("ra.collapse.cov", pct(col["cov1"]), f"{rel} extra_headlines #6 corrected.cov1")
    rel = "results/posthoc_calibration_4bit/posthoc_fit.json"
    ph = S.json(A_, rel)
    add("ra.posthoc.fit", num(ph["scales"]["crps_fit"], 1), f"{rel} scales.crps_fit (D69)")
    add("ra.posthoc.cov", integer(round(ph["scales"]["coverage_matched"])), f"{rel} scales.coverage_matched (D69)")
    # E1 / E1d (D52, D94, D101; FINDINGS #14, #16, #25)
    rel = "results/e1/pilots/pilots.json"
    pl = S.json(A_, rel)
    add("ra.evaln", integer(pl["n_problems"]), f"{rel} n_problems (the frozen noise-free evaluation set, D67)")
    de = pl["describe"]
    add("ra.eone.rho", num(de["E1a (reference)"]["spearman_sigma_error"], 2), f"{rel} describe.E1a.spearman_sigma_error")
    add("ra.eone.cov", pct(de["E1a (reference)"]["coverage_1sigma"]), f"{rel} describe.E1a.coverage_1sigma")
    add("ra.eonec.rho", num(de["E1c"]["spearman_sigma_error"], 3), f"{rel} describe.E1c.spearman_sigma_error")
    add("ra.eoned.rho", num(de["E1d"]["spearman_sigma_error"], 3), f"{rel} describe.E1d.spearman_sigma_error")
    t = pl["tests"]["E1d"]
    add("ra.eoned.diff", f"{num(t['diff'], 3, True)} {_ci_named(*t['ci95'], 'ci95')}", f"{rel} tests.E1d.diff, ci95")
    add("ra.eoned.cov", pct(de["E1d"]["coverage_1sigma"]), f"{rel} describe.E1d.coverage_1sigma")
    add("ra.eoned.sigma", pct(de["E1d"]["median_rel_sigma"], 2), f"{rel} describe.E1d.median_rel_sigma")
    fe = pl["family_E"]["E1d"]
    add("ra.eoned.sft", f"{num(fe['P (vs SFT σ)']['mean_diff'], 3, True)} {_ci_named(*fe['P (vs SFT σ)']['ci95'], 'ci95')}",
        f"{rel} family_E.E1d.P")
    add("ra.eoned.afive", f"{num(fe['N (vs A5 σ)']['mean_diff'], 3, True)} {_ci_named(*fe['N (vs A5 σ)']['ci95'], 'ci95')}",
        f"{rel} family_E.E1d.N")
    rel = "results/x3/pilot.json"
    add("ra.xthree.rho", num(S.json(A_, rel)["arms"]["a"]["describe"]["spearman_sigma_error"], 3), f"{rel} arms.a.describe.spearman_sigma_error")


def _act2(S: Sources, add) -> None:
    """Results II: within-problem correctness signals (D74, D89, D91, D92, D102; FINDINGS #19, #21, #22, #26, #37)."""
    A_ = "macros"
    x2 = S.json(A_, "results/x2/results.json")
    add("rb.problems", integer(x2["n_kept"]), "results/x2/results.json n_kept")
    add("rb.pairs", integer(x2["n_pairs"]), "results/x2/results.json n_pairs")
    add("rb.candidates", integer(x2["n_candidates_sampled"]), "results/x2/results.json n_candidates_sampled")
    rel = "results/x2/robustness/results.json"
    rob = S.json(A_, rel)
    m = {k: v["pairwise_accuracy"] for k, v in rob["methods"].items()}
    add("rb.cilevel", _level("ci95"), f"{rel} methods.*.pairwise_accuracy.ci95")
    add("rb.sc.v", num(m["self_consistency"]["pairwise_accuracy"]), f"{rel} methods.self_consistency (point; D179)")
    xr = S.yaml(A_, "configs/x2_robustness.yaml")
    add("rb.agree", _lv(xr["self_consistency"]["agree_tolerance"]), "configs/x2_robustness.yaml self_consistency.agree_tolerance")
    acc = lambda d: f"{num(d['pairwise_accuracy'])} {ci(*d['ci95'], sign=False)}"
    names = {"sft_sigma": "sftsigma", "split_sigma": "splitsigma", "a5_p": "afivep", "a5_sigma": "afivesigma",
             "probe": "probe", "random_init_probe": "randominitprobe", "length": "length",
             "self_consistency": "selfconsistency", "x4_sft": "xfoursft"}
    for k, name in names.items():
        add(f"rb.acc.{name}", acc(m[k]), f"{rel} methods.{k}.pairwise_accuracy, ci95")
    e = S.json(A_, "results/e1d_x2/results.json")["methods"]
    for k, name in (("e1d", "eoned"), ("x3_a", "xthreea")):
        add(f"rb.acc.{name}", acc(e[k]["pairwise_accuracy"]), f"results/e1d_x2/results.json methods.{k}.pairwise_accuracy")
    single = [m[k]["pairwise_accuracy"] for k in ("sft_sigma", "split_sigma", "a5_p", "a5_sigma", "x4_sft")]
    single += [e[k]["pairwise_accuracy"]["pairwise_accuracy"] for k in ("e1d", "x3_a")]
    add("rb.single.span", _span(single, num), f"{rel} + e1d_x2: the stated / self-reported readouts' range")
    pr = rob["paired"]
    for k, name in (("probe - random_init_probe", "probe.random"), ("self_consistency - probe", "sc.probe"),
                    ("length - probe", "length.probe")):
        add(f"rb.diff.{name}", f"{num(pr[k]['mean_diff'], 3, True)} {_ci_named(*pr[k]['ci95'], 'ci95')}", f"{rel} paired.'{k}'")
    ep = S.json(A_, "results/e1d_x2/results.json")["reported_paired"]["e1d - self_consistency"]
    add("rb.diff.eoned.sc", f"{num(ep['mean_diff'], 3, True)} {_ci_named(*ep['ci95'], 'ci95')}",
        "results/e1d_x2/results.json reported_paired.'e1d - self_consistency'")
    sep = rob["methods"]["self_consistency"]["separated"]
    add("rb.sc.separated", num(sep["pairwise_accuracy"]), f"{rel} methods.self_consistency.separated")
    ac = e["e1d"]["across_problems"]
    add("rb.eoned.across", num(ac["spearman_sigma_error"]["value"], 3), "results/e1d_x2/results.json methods.e1d.across_problems")
    add("rb.xthree.across", num(e["x3_a"]["across_problems"]["spearman_sigma_error"]["value"], 3),
        "results/e1d_x2/results.json methods.x3_a.across_problems")
    # the cost (R5-8; FINDINGS #37)
    rel = "results/r5_diag/cost.json"
    c = S.json(A_, rel)
    pp = c["x2"]["per_problem"]
    add("rb.k", integer(c["x2"]["k"]), f"{rel} x2.k")
    add("rb.tokens.k", integer(round(pp["tokens_k"])), f"{rel} x2.per_problem.tokens_k")
    add("rb.tokens.one", integer(round(pp["tokens_one"])), f"{rel} x2.per_problem.tokens_one")
    add("rb.cost.ratio", num(pp["tokens_k"] / pp["tokens_one"], 1), f"{rel} x2.per_problem tokens_k / tokens_one")
    add("rb.seconds.k", num(pp["seconds_k"], 2), f"{rel} x2.per_problem.seconds_k")
    add("rb.seconds.one", num(pp["seconds_one"], 2), f"{rel} x2.per_problem.seconds_one")
    # X4 (D89, D91; FINDINGS #21)
    rel = "results/x4/results.json"
    x4 = S.json(A_, rel)
    dg = x4["diagnostics"]["sft"]
    add("rb.xfour.verdict", pct(dg["share_wrote_verdict"], 0), f"{rel} diagnostics.sft.share_wrote_verdict")
    add("rb.xfour.yes.correct", pct(dg["natural_yes_rate_correct"]), f"{rel} diagnostics.sft.natural_yes_rate_correct")
    add("rb.xfour.yes.incorrect", pct(dg["natural_yes_rate_incorrect"]), f"{rel} diagnostics.sft.natural_yes_rate_incorrect")
    add("rb.xfour.pyes.correct", num(dg["mean_p_yes_correct"]), f"{rel} diagnostics.sft.mean_p_yes_correct")
    add("rb.xfour.pyes.incorrect", num(dg["mean_p_yes_incorrect"]), f"{rel} diagnostics.sft.mean_p_yes_incorrect")
    add("rb.xfour.p", num(x4["holm"]["sft"]["p"]), f"{rel} holm.sft.p")
    add("rb.xfour.threshold", num(x4["holm"]["sft"]["threshold"], 4), f"{rel} holm.sft.threshold")
    add("rb.xfour.tokens", integer(int(x4["tokens"]["sft"]["generated"]["median"])), f"{rel} tokens.sft.generated.median")
    add("rb.xfour.factor", num(x4["meta"]["perturbed"]["perturb_factor"], 2), f"{rel} meta.perturbed.perturb_factor")
    add("rb.xfour.perturbed", num(x4["methods"]["perturbed"]["pairwise_accuracy"]), f"{rel} methods.perturbed.pairwise_accuracy")
    au = x4["methods"]["sft"]["auroc_across_problems"]
    add("rb.xfour.auroc", f"{num(au['auroc'])} {ci(*au['ci95'], sign=False)}", f"{rel} methods.sft.auroc_across_problems")


def _act3(S: Sources, add) -> None:
    """Results III: implicit vs explicit and R4-3 (D118, D125; FINDINGS #32); depth 5 (#34); the tail and the swaps
    (#35); the RSS fallback (#36, post-hoc); MC-A (#31) and MC-B (#33, #37); the routes that failed (#17, #23, #27-#29)."""
    A_ = "macros"
    rel = "results/stage_r4/confirmatory/results.json"
    r = S.json(A_, rel)
    t = r["main"]["tests"]
    for k in ("K1", "K2", "K3", "K4"):
        add(f"rc.{k.lower()}", _est(t[k]), f"{rel} main.tests.{k} (ci90)")
    add("rc.k1.v", num(t["K1"]["value"], 2, True), f"{rel} main.tests.K1.value (the point estimate, for the abstract; D179)")
    for k in ("K5", "K6", "K7", "K8"):
        add(f"rc.{k.lower()}", _est(t[k], nd=4), f"{rel} main.tests.{k} (ci90)")
        add(f"rc.{k.lower()}.p", pval(t[k]["p_one_sided"]), f"{rel} main.tests.{k}.p_one_sided")
    add("rc.ksix.seeds", ", ".join(num(v, 4, True) for _, v in sorted(t["K6"]["per_seed"].items())), f"{rel} main.tests.K6.per_seed")
    crit = r["main"]["criteria"]
    add("rc.cthree", _est(crit["dpos2_a"]["a6b"]), f"{rel} main.criteria.dpos2_a.a6b (ci90)")
    add("rc.ctwo", num(r["main"]["criteria"]["dpos2_a"]["a6c"]["value"], 3, True), f"{rel} main.criteria.dpos2_a.a6c.value")
    add("rc.kone.seeds", ", ".join(num(v["value"], 3, True) for _, v in sorted(t["K1"]["per_seed"].items())), f"{rel} K1.per_seed")
    add("rc.p.confirmed", "confirmed" if r["primary"]["confirmed"] else "not confirmed", f"{rel} primary.confirmed")
    sub = r["subsets"]
    add("rc.dfive.kone", _est(sub["depth_5"]["tests"]["K1"]), f"{rel} subsets.depth_5.tests.K1")
    for k in ("K6", "K7"):
        add(f"rc.dfive.{k.lower()}", _est(sub["depth_5"]["tests"][k], nd=4), f"{rel} subsets.depth_5.tests.{k}")
        add(f"rc.dfour.{k.lower()}", _est(sub["depths_1_4"]["tests"][k], nd=4), f"{rel} subsets.depths_1_4.tests.{k}")
    regret = [v["share_in_worst"] for v in r["regret"].values()]
    add("rc.regret.share", _span(regret, lambda x: pct(x, 0)), f"{rel} regret.*.share_in_worst")
    # oracle-centred description, per seed and the references (D118 sec. 9)
    rel = "results/stage_r4/confirmatory/stage_b_dpos2/results.json"
    sb = S.json(A_, rel)
    oc = [v["oracle_centred"] for v in sb["seeds"].values()]
    add("rc.cov.seeds", _span([o["coverage_1"][0] for o in oc], lambda x: pct(x, 0)), f"{rel} seeds.*.oracle_centred.coverage_1")
    add("rc.covtwo.seeds", _span([o["coverage_2"][0] for o in oc], lambda x: pct(x, 0)), f"{rel} seeds.*.oracle_centred.coverage_2")
    add("rc.beta.seeds", _span([o["vol_beta"][0] for o in oc], lambda x: num(x, 2)), f"{rel} seeds.*.oracle_centred.vol_beta")
    add("rc.mal.seeds", _span([o["median_abs_log_ratio"] for o in oc], lambda x: num(x, 2)), f"{rel} seeds.*.median_abs_log_ratio")
    for n in ("rss", "x1", "sigma_star"):
        o = sb["comparators"][n]["oracle_centred"]
        key = {"rss": "rss", "x1": "xone", "sigma_star": "star"}[n]
        add(f"rc.cov.{key}", pct(o["coverage_1"][0], 0), f"{rel} comparators.{n}.oracle_centred.coverage_1")
        add(f"rc.beta.{key}", num(o["vol_beta"][0], 2), f"{rel} comparators.{n}.oracle_centred.vol_beta")
        add(f"rc.mal.{key}", num(o["median_abs_log_ratio"], 2), f"{rel} comparators.{n}.oracle_centred.median_abs_log_ratio")
    add("rc.xone.slope", num(sb["comparators"]["x1"]["a6c_all_pairs"]["slope"], 3, True), f"{rel} comparators.x1.a6c_all_pairs.slope")
    # the RSS-to-sigma* gap K6 closes (D118 sec. 9; FINDINGS #32)
    rel = "results/r5_diag/tail.json"
    tl = S.json(A_, rel)
    gap = tl["rss_to_star_gap"]
    k6 = t["K6"]
    lo, hi = (x / gap for x in k6["ci90"])
    add("rc.ksix.gap", f"{pct(k6['value'] / gap, 0)} [{_level('ci90')} CI {pct(lo, 0)}, {pct(hi, 0)}]",
        f"{rel} rss_to_star_gap; confirmatory K6 value and ci90 over the gap")
    add("rc.ksix.gap.v", pct(k6["value"] / gap, 0), f"{rel} rss_to_star_gap; K6 value over the gap (point; D179)")
    # the tail autopsy (R5-2) and the component swaps (R5-3)
    ps = tl["per_seed"]
    add("rc.tail.worst", pct(next(iter(ps.values()))["n_worst"] / tl["n_problems"], 0), f"{rel} per_seed.*.n_worst / n_problems")
    add("rc.tail.share", _span([v["worst_regret_share"] for v in ps.values()], lambda x: pct(x, 0)), f"{rel} per_seed.*.worst_regret_share")
    pooled = tl["causes_worst_pooled"]
    n = sum(pooled.values())
    share = lambda pre: sum(k for c, k in pooled.items() if c.startswith(pre)) / n
    add("rc.tail.formula", pct(share("sensitivity: value-dependent: formula"), 0), f"{rel} causes_worst_pooled (formula, with and without an expression)")
    add("rc.tail.subst", pct(share("sensitivity: value-dependent: substitution (own wrong"), 0), f"{rel} causes_worst_pooled (own wrong intermediates)")
    add("rc.tail.copy", pct(share("copy: decimal shift"), 0), f"{rel} causes_worst_pooled (decimal shifts)")
    add("rc.tail.chain", pct(share("chain selection"), 0), f"{rel} causes_worst_pooled (chain selection)")
    sw = tl["swaps"]
    add("rc.swap.s.rss", _est(sw["S"]["minus_rss"], nd=4), f"{rel} swaps.S.minus_rss")
    add("rc.swap.s.xone", _est(sw["S"]["minus_x1"], nd=4), f"{rel} swaps.S.minus_x1")
    add("rc.swap.s.gap", pct(sw["S"]["share_of_rss_to_star_gap"], 0), f"{rel} swaps.S.share_of_rss_to_star_gap")
    add("rc.swap.s.slope", num(sw["S"]["a6c_slope"]["value"], 2, True), f"{rel} swaps.S.a6c_slope")
    add("rc.swap.stated.slope", num(sw["a"]["a6c_slope"]["value"], 3, True), f"{rel} swaps.a.a6c_slope (every pair)")
    for k, name in (("L", "levels"), ("A", "aggregation"), ("C", "copy")):
        add(f"rc.swap.{name}.gap", pct(sw[k]["share_of_rss_to_star_gap"], 0), f"{rel} swaps.{k}.share_of_rss_to_star_gap")
    # per-depth slopes, OLS and Theil-Sen (D137 item 4; FINDINGS #34)
    rel = "results/r5_diag/depth_ts.json"
    dt = S.json(A_, rel)["arms"]
    depths = sorted(next(iter(dt["dpos2"].values())), key=int)
    mean = lambda arm, d, k: sum(s[d][k] for s in dt[arm].values()) / len(dt[arm])
    for d in depths:
        add(f"rc.depth.{d}", f"{num(mean('dpos2', d, 'ols'), 2, True)} / {num(mean('dpos2', d, 'theil_sen'), 2, True)}",
            f"{rel} arms.dpos2.*.{d} ols / theil_sen (mean over seeds)")
    add("rc.depth.first", integer(int(depths[0])), f"{rel} depths")
    add("rc.rev.first", _span([s[depths[0]]["n_reversed"] for s in dt["dpos2"].values()], integer), f"{rel} dpos2 n_reversed, first depth")
    add("rc.rev.last", _span([s[depths[-1]]["n_reversed"] for s in dt["dpos2"].values()], integer), f"{rel} dpos2 n_reversed, last depth")
    add("rc.depth.pairs", integer(next(iter(dt["dpos2"].values()))[depths[-1]]["n"]), f"{rel} dpos2 n, last depth")
    add("rc.btwo.ols", _span([mean("b2", d, "ols") for d in depths], lambda x: num(x, 2)), f"{rel} arms.b2 ols (seed means by depth)")
    add("rc.btwo.ts", _span([mean("b2", d, "theil_sen") for d in depths], lambda x: num(x, 2)), f"{rel} arms.b2 theil_sen (seed means by depth)")
    rel = "results/r5_diag/depth5.json"
    d5 = S.json(A_, rel)
    last = depths[-1]
    for s, key in (("development", "dev"), ("frozen", "frozen")):
        b = d5[s]["by_depth"][last]
        add(f"rc.dfive.{key}.ols", num(b["slope"], 3, True), f"{rel} {s}.by_depth.{last}.slope")
        add(f"rc.dfive.{key}.ts", num(b["theil_sen"]["slope"], 3, True), f"{rel} {s}.by_depth.{last}.theil_sen.slope")
        add(f"rc.dfive.{key}.rev", integer(b["n_reversed"]), f"{rel} {s}.by_depth.{last}.n_reversed")
    # the RSS fallback (R5-4, post-hoc exploratory; FINDINGS #36)
    rel = "results/r5_diag/fallback_confirmatory.json"
    fb = S.json(A_, rel)
    add("rc.fb.bounds", f"[{num(fb['bounds'][0], 3)}, {num(fb['bounds'][1], 0)}]", f"{rel} bounds (D135)")
    add("rc.fb.rate", _span(list(fb["fallback_rate"].values()), lambda x: pct(x, 0)), f"{rel} fallback_rate")
    add("rc.fb.rss", _est(fb["minus_rss"], nd=4), f"{rel} minus_rss")
    add("rc.fb.xone", _est(fb["minus_x1"], nd=4), f"{rel} minus_x1")
    add("rc.fb.gap", pct(fb["share_of_rss_to_star_gap"], 0), f"{rel} share_of_rss_to_star_gap")
    add("rc.fb.slope", _est(fb["a6c_slope"]), f"{rel} a6c_slope")
    add("rc.fb.slope.v", num(fb["a6c_slope"]["value"], 2, True), f"{rel} a6c_slope.value (point; D179)")
    # MC-A (R4-4; FINDINGS #31) and MC-B (D119 / D122; FINDINGS #33, #37)
    rel = "results/mc_a/a6c_pairs/results.json"
    ma = S.json(A_, rel)
    add("rc.mca.rss", _est(ma["stopping_rule"]["robust_minus_rss"], key="ci95", nd=4), f"{rel} stopping_rule.robust_minus_rss")
    add("rc.mca.p", pval(ma["stopping_rule"]["robust_minus_rss"]["p_one_sided"]), f"{rel} stopping_rule p_one_sided")
    add("rc.mca.oracle", _est(ma["oracle_minus_rss"], key="ci95", nd=4), f"{rel} oracle_minus_rss")
    rel = "results/r5_diag/cost.json"
    c = S.json(A_, rel)
    mk = c["mc_a"]["a6c_pairs"]
    kmax = max(mk["k"], key=int)
    add("rc.mca.auroc", num(mk["k"][kmax]["raw_auroc"]), f"{rel} mc_a.a6c_pairs.k.{kmax}.raw_auroc")
    add("rc.mca.k", integer(int(kmax)), f"{rel} mc_a.a6c_pairs.k")
    free = list(mk["comparators_auroc"].values()) + [mk["stated_sigma_auroc"]]
    add("rc.mca.free", num(max(free)), f"{rel} mc_a.a6c_pairs: the best free or single-sample AUROC")
    add("rc.mca.tokens", integer(round(mk["k"][kmax]["extra_tokens_per_problem"])), f"{rel} mc_a.a6c_pairs.k.{kmax}.extra_tokens_per_problem")
    sh = c["mcb_split"]["a6c_pairs"]["share"]
    add("rc.mcb.interface", pct(sh["interface"]), f"{rel} mcb_split.a6c_pairs.share.interface")
    add("rc.mcb.semantic", pct(sh["semantic"] / (sh["semantic"] + sh["correct"]), 0), f"{rel} mcb_split semantic / (semantic + correct)")
    add("rc.mcb.correct", pct(sh["correct"]), f"{rel} mcb_split.a6c_pairs.share.correct")
    rel = "results/mc_b/a6c_pairs/results.json"
    mb = S.json(A_, rel)["tests"]
    ms = [k for k in mb if re.fullmatch(r"M\d", k)]
    add("rc.mcb.rejected", f"{integer(sum(mb[k]['holm']['rejected'] for k in ms))} of {integer(len(ms))}", f"{rel} tests.M*.holm.rejected")
    add("rc.mcb.rss", _est(mb["M1"], key="ci95", nd=4), f"{rel} tests.M1 (MC-B - RSS)")
    # stage B: SFT on sigma* (D104; FINDINGS #27)
    rel = "results/stage_b/results.json"
    sbt = S.json(A_, rel)["tests"]
    add("rc.stageb.slope", _est(sbt["B2"], key="ci95"), f"{rel} tests.B2 (A6c slope)")
    add("rc.stageb.rss", _est(sbt["B3"], key="ci95", nd=4), f"{rel} tests.B3 (expected crps_log - RSS)")


def _generalization(S: Sources, add) -> None:
    """R5-6 part J (D147-D149; FINDINGS #39) and the Llama pilot (D137, D139, D146; FINDINGS #38)."""
    A_ = "macros"
    rel = "results/r5_6/results.json"
    r = S.json(A_, rel)
    m = r["main"]
    cr = m["criteria"]["dpos2_a"]
    add("rd.jone", _est(cr["a6c"]), f"{rel} main.criteria.dpos2_a.a6c (J1)")
    add("rd.jone.v", num(cr["a6c"]["value"], 2, True), f"{rel} main.criteria.dpos2_a.a6c.value (point; D179)")
    add("rd.clamped.v", num(m["comparators"]["clamped"]["sensitivity"]["value"], 2, True),
        f"{rel} main.comparators.clamped.sensitivity.value (point; D179)")
    add("rd.cthree", _est(cr["a6b"]), f"{rel} main.criteria.dpos2_a.a6b (C3)")
    add("rd.cthree.pairs", _span([v["n_pairs"] for v in cr["a6b"]["per_seed"].values()], integer), f"{rel} criteria.dpos2_a.a6b.per_seed.n_pairs")
    add("rd.p.confirmed", "confirmed" if r["primary"]["confirmed"] else "not confirmed", f"{rel} primary.confirmed")
    add("rd.jtwo", _est(m["tests"]["J2"]), f"{rel} main.tests.J2")
    for k, name in (("J3", "jthree"), ("J4", "jfour")):
        add(f"rd.{name}", _est(m["tests"][k], nd=4), f"{rel} main.tests.{k}")
        add(f"rd.{name}.holm", "rejected" if m["tests"][k]["holm"]["rejected"] else "not rejected", f"{rel} main.tests.{k}.holm")
    add("rd.clamped", _est(m["comparators"]["clamped"]["sensitivity"]), f"{rel} main.comparators.clamped.sensitivity")
    rf = r["splits"]["rank_flip_sensitivity"]
    add("rd.flip.pairs", integer(rf["n_pairs"]), f"{rel} splits.rank_flip_sensitivity.n_pairs")
    add("rd.flip.dpos", num(rf["slopes"]["dpos2_a"]["sensitivity"]["value"], 3, True), f"{rel} splits.rank_flip_sensitivity dpos2_a")
    add("rd.flip.clamped", num(rf["slopes"]["comparator_clamped"]["sensitivity"]["value"], 3, True), f"{rel} splits.rank_flip_sensitivity clamped")
    # the Llama pilot
    rel = "results/stage_xm/pilot/results.json"
    p = S.json(A_, rel)
    go = S.json(A_, "results/stage_xm/go_dpos2/go.json")
    a = p["models"]["llama"]["arms"]["dpos2"]["sources"]["a"]
    add("rd.xm.slope", f"{num(a['a6c_slope'], 3, True)} {_ci_named(*a['a6c_ci90'], 'ci90')}", f"{rel} llama dpos2 a a6c_slope")
    add("rd.xm.pairs", integer(a["a6c_n_pairs"]), f"{rel} llama dpos2 a a6c_n_pairs")
    add("rd.xm.cthree.pairs", integer(go["a6b_n_pairs"]), "results/stage_xm/go_dpos2/go.json a6b_n_pairs")
    add("rd.xm.go", "GO" if go["go"] else "NO-GO", "results/stage_xm/go_dpos2/go.json go")
    d = a["a6b_answered_pairs_slope"]
    add("rd.xm.desc", f"{num(d['slope'], 3, True)} {_ci_named(*d['ci90'], 'ci90')}", f"{rel} llama dpos2 a a6b_answered_pairs_slope")
    add("rd.xm.desc.pairs", integer(d["n"]), f"{rel} llama dpos2 a a6b_answered_pairs_slope.n")
    q = p["models"]["qwen_seed0"]["arms"]["dpos2"]["sources"]["a"]["a6b_answered_pairs_slope"]
    add("rd.xm.qwen", num(q["slope"], 3, True), f"{rel} qwen_seed0 dpos2 a a6b_answered_pairs_slope")

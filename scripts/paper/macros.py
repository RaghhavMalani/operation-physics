"""The values the paper's prose uses (\\pv{key}), each read from a committed config, data manifest or result file.

Setup, Methods and the evaluation protocol mostly cite design constants; they are read from the configs and frozen
manifests that defined them, never typed by hand.
"""

from __future__ import annotations

import re

from paperlib import Macros, Sources, ci, integer, num, pct

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
    # ---- the protocol's record
    dec = S.text(A, "DECISIONS.md")
    add("record.dentries", integer(len(re.findall(r"^### D\d+\.", dec, flags=re.M))), "DECISIONS.md: '### D<n>.' headers")
    fin = S.text(A, "FINDINGS.md")
    add("record.findings", integer(max(int(n) for n in re.findall(r"^## #(\d+)\.", fin, flags=re.M))),
        "FINDINGS.md: the highest '## #<n>.' header")
    return M

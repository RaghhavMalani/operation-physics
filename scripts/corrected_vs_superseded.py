"""Corrected (D67: adapters unmerged on the 4-bit training base) vs superseded (bf16-merged) results, side by side.

    uv run python scripts/corrected_vs_superseded.py

Reads the outputs of phase95_compare.py / phase95_seed_mean.py / e1 compare in both modes, plus the eval tapes for
the descriptive rows. Every part whose inputs do not exist yet is reported as missing. Writes
results/corrected/side_by_side.{json,md}. Descriptive only: the tests themselves are in the files it reads.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from phase95_compare import headline, load  # noqa: E402
from sigma_greedy_vs_sampled import e1_rows, tape_rows, within_problem  # noqa: E402
from physcal.tape.tape import read_tape  # noqa: E402

T = ROOT / "results/train"
CORR = ROOT / "results/corrected"
SPLIT = {s: T / f"split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed{s}" for s in (0, 1, 2)}
UNI = T / "overnight2-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512"
N5 = T / "overnight-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-512"
E1 = {"E1a": T / "e1a-sigma-only-free-seed0", "E1b": T / "e1b-sigma-only-bins-seed0"}
S_TESTS = ("S1a", "S1b", "S2", "S3")
N_TESTS = ("N1", "N2", "N3")
E_TESTS = ("E1a-P", "E1b-P", "E1a-N", "E1b-N")


def jload(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def seed_mean() -> dict:
    out = {}
    for mode, p in (("superseded", ROOT / "results/split_credit_seed_mean/family_S_seed_mean.json"),
                    ("corrected", CORR / "split_credit_seed_mean/family_S_seed_mean.json")):
        d = jload(p)
        out[mode] = None if d is None else {k: {**{f: d["seed_mean"][k][f] for f in ("seed_mean", "control", "diff", "diff_ci")},
                                               **d["seed_mean"]["holm"][k], "per_seed_p": d.get("per_seed_p", {}).get(k),
                                               "reported_as": d.get("reported_as", {}).get(k)} for k in S_TESTS}
    return out


def per_seed() -> dict:
    out = {}
    for s, run in SPLIT.items():
        for mode, p in (("superseded", run / "compare/phase95_family_S.json"), ("corrected", CORR / f"family_S/{run.name}.json")):
            d = jload(p)
            if d is None:
                out[(s, mode)] = None
                continue
            n = d["amendment_D49"]["N_vs_A5_strongest_same_completions"]
            out[(s, mode)] = {"S": {k: d["tests"]["holm"][k] for k in S_TESTS},
                              "N": {k: {**n[[x for x in n if x.startswith(k + "_")][0]], **n["holm"][k]} for k in N_TESTS},
                              "headline": {name: h["pooled"] for name, h in d["headline"].items()}}
    return out


def family_e() -> dict:
    out = {}
    for mode, p in (("superseded", ROOT / "results/e1/family_E.json"), ("corrected", CORR / "e1/family_E.json")):
        d = jload(p)
        out[mode] = None if d is None else {"tests": {k: {**(d["tests"][k] if isinstance(d["tests"].get(k), dict) else {}),
                                                          **d["holm"][k]} for k in E_TESTS},
                                            "describe": d["describe"]}
    return out


def extra_headlines() -> dict:
    """Descriptive: the FINDINGS #5 / #6 adapters, superseded vs corrected (pooled 400 problems)."""
    pairs = {"#5 overnight RL, final (step 287)": (N5 / "compare_verifier_1.1.0/rl", N5 / "compare/rl_4bit"),
             "#6 overnight2 (uniform), final (step 759)": (UNI / "compare/rlfinal", UNI / "compare/rlfinal_4bit")}
    out = {}
    for name, (a, b) in pairs.items():
        row = {}
        for mode, d in (("superseded", a), ("corrected", b)):
            t = d / "rollouts.jsonl"
            if t.exists():
                x = load(t)
                row[mode] = headline(x, sorted(x))
        out[name] = row
    return out


def e1_step150() -> dict:
    from physcal.train.e1_compare import describe, rows_arm
    from physcal.config import load_config

    e1 = load_config("e1")
    out = {}
    for name, run in E1.items():
        for mode, sub in (("superseded", "step150"), ("corrected", "step150_4bit")):
            p = run / "e1_eval" / sub / "rows.jsonl"
            if p.exists():
                arm = rows_arm(p)
                out[f"{name} step 150 ({mode})"] = describe(arm, sorted(arm), e1["bins"], e1["compare"]["coverage_target"])
    return out


def determinism() -> dict:
    """The D65 tapes (primary, D67) vs tonight's reruns: share of byte-identical completions / σ suffixes."""
    pairs = {"split seed 0 @287 greedy": (SPLIT[0] / "compare/rl287_4bit", SPLIT[0] / "compare/rl287_4bit_rerun", "completion"),
             "E1a final greedy": (E1["E1a"] / "e1_eval/final_4bit", E1["E1a"] / "e1_eval/final_4bit_rerun", "suffix"),
             "split seed 0 @287 sampled k=4": (SPLIT[0] / "compare/rl287_4bit_sampled",
                                               SPLIT[0] / "compare/rl287_4bit_sampled_rerun", "completion"),
             "E1a final sampled k=4": (E1["E1a"] / "e1_eval/final_4bit_sampled",
                                       E1["E1a"] / "e1_eval/final_4bit_sampled_rerun", "suffix")}
    out = {}
    for name, (a, b, field) in pairs.items():
        if not ((a / "rollouts.jsonl").exists() and (b / "rollouts.jsonl").exists()):
            out[name] = "missing"
            continue
        ta, tb = list(read_tape(a / "rollouts.jsonl")), list(read_tape(b / "rollouts.jsonl"))
        same = sum(x[field] == y[field] for x, y in zip(ta, tb))
        out[name] = {"n": len(ta), "identical": same / len(ta), "same_length": len(ta) == len(tb),
                     "rewards_identical": sum(x["reward"] == y["reward"] for x, y in zip(ta, tb)) / len(ta)}
    return out


def instability() -> dict:
    """σ instability (D62): within-problem SD of ln σ over the k = 4 samples, 4-bit base (D65 run and tonight's rerun)."""
    arms = {"split seed 0 @287": (tape_rows, SPLIT[0] / "compare/rl287_4bit/rollouts.jsonl",
                                  {"D65": SPLIT[0] / "compare/rl287_4bit_sampled/rollouts.jsonl",
                                   "rerun": SPLIT[0] / "compare/rl287_4bit_sampled_rerun/rollouts.jsonl",
                                   "superseded (bf16 merged)": SPLIT[0] / "compare/rl287_sampled/rollouts.jsonl"}),
            "E1a final": (e1_rows, E1["E1a"] / "e1_eval/final_4bit/rows.jsonl",
                          {"D65": E1["E1a"] / "e1_eval/final_4bit_sampled/rows.jsonl",
                           "rerun": E1["E1a"] / "e1_eval/final_4bit_sampled_rerun/rows.jsonl",
                           "superseded (bf16 merged)": E1["E1a"] / "e1_eval/final_sampled/rows.jsonl"})}
    out = {}
    for name, (reader, greedy, sampled) in arms.items():
        g = reader(greedy) if greedy.exists() else None
        for label, p in sampled.items():
            if g is not None and p.exists():
                g_ref = g if "superseded" not in label else reader(greedy.parent.parent / greedy.parent.name.replace("_4bit", "") / greedy.name)
                out[f"{name}, {label}"] = within_problem(g_ref, reader(p))
    return out


def fmt(x, f=".3g"):
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    return format(x, f)


def render(r: dict) -> str:
    L = ["# Corrected (4-bit training base, adapters unmerged; D67) vs superseded (bf16-merged) results", ""]
    sm = r["seed_mean"]
    L += ["## Family S on the seed mean (D62)", "",
          "| test | superseded: split / control / diff / p / verdict | corrected: split / control / diff / p / verdict | corrected per-seed p |",
          "|---|---|---|---|"]
    for k in S_TESTS:
        cells = []
        for mode in ("superseded", "corrected"):
            v = (sm[mode] or {}).get(k)
            cells.append("missing" if v is None else
                         f"{v['seed_mean']:.4f} / {v['control']:.4f} / {v['diff']:+.4f} / {fmt(v['p'])} / "
                         f"{v.get('reported_as') or ('supported' if v['supported'] else 'not supported')}")
        ps = ((sm["corrected"] or {}).get(k) or {}).get("per_seed_p")
        L.append(f"| {k} | {cells[0]} | {cells[1]} | {' / '.join(fmt(x) for x in ps) if ps else '–'} |")
    L += ["", "## Families S and N per seed (Holm within each family)", "",
          "| seed | eval | S1a | S1b | S2 | S3 | N1 (RL vs A5) | N2 | N3 |", "|---|---|---|---|---|---|---|---|---|"]
    for (s, mode), v in r["per_seed"].items():
        if v is None:
            L.append(f"| {s} | {mode} | missing | | | | | | |")
            continue
        sc = [f"p {fmt(v['S'][k]['p'])} {'✓' if v['S'][k]['supported'] else '✗'}" for k in S_TESTS]
        nc = [f"{v['N'][k]['rl']:.3f} vs {v['N'][k]['control']:.3f}, p {fmt(v['N'][k]['p'])} {'✓' if v['N'][k]['supported'] else '✗'}"
              for k in N_TESTS]
        L.append(f"| {s} | {mode} | " + " | ".join(sc + nc) + " |")
    L += ["", "N1 = |1σ coverage − 68.3%| (lower is better), N2 = ρ(σ, err), N3 = mean crps_log; ✓ = supported "
          "(RL σ better than the A5 σ on the same answers).", "",
          "## Headline metrics, pooled 400 problems (greedy)", "",
          "| arm | eval | acc ≤ 1% | 1σ cov | 2σ cov | σ = 0 | median rel. σ | ρ(σ, err) | mean crps_log |",
          "|---|---|---|---|---|---|---|---|---|"]
    names = ("sft", "posthoc", "sft_a5_strongest", "uniform_rl287", "split_rl287", "split_rl287_a5_strongest")
    for name in names:
        for s in SPLIT:
            for mode in ("superseded", "corrected"):
                v = r["per_seed"].get((s, mode))
                if v is None or name not in v["headline"] or (name not in ("split_rl287", "split_rl287_a5_strongest") and s != 0):
                    continue
                h = v["headline"][name]
                label = f"{name} (seed {s})" if name.startswith("split") else name
                L.append(f"| {label} | {mode} | {h['acc']:.1%} | {h['cov1']:.1%} | {h['cov2']:.1%} | {h['sigma_zero']:.1%} | "
                         f"{h['median_rel_sigma']:.3%} | {h['rho_sigma_err']:.2f} | {h['mean_crps_log']:.4f} |")
    for name, row in r["extra_headlines"].items():
        for mode, h in row.items():
            L.append(f"| {name} | {mode} | {h['acc']:.1%} | {h['cov1']:.1%} | {h['cov2']:.1%} | {h['sigma_zero']:.1%} | "
                     f"{h['median_rel_sigma']:.3%} | {h['rho_sigma_err']:.2f} | {h['mean_crps_log']:.4f} |")
    fe = r["family_e"]
    L += ["", "## Family E (D52; μ fixed, the same 400 prefixes)", "",
          "| test | superseded: Δcrps_log [95% CI], p, verdict | corrected: Δcrps_log [95% CI], p, verdict |", "|---|---|---|"]
    for k in E_TESTS:
        cells = []
        for mode in ("superseded", "corrected"):
            v = (fe[mode] or {}).get("tests", {}).get(k)
            cells.append("missing" if v is None or "mean_diff" not in v else
                         f"{v['mean_diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}], p {fmt(v['p'])}, "
                         f"{'supported' if v['rejected'] else 'not supported'}")
        L.append(f"| {k} | {cells[0]} | {cells[1]} |")
    L += ["", "| σ source | eval | mean crps_log | 1σ cov | ρ(σ, err) | median rel. σ | σ = 0 |", "|---|---|---|---|---|---|---|"]
    for mode in ("superseded", "corrected"):
        for name, d in ((fe[mode] or {}).get("describe") or {}).items():
            if "mean_crps_log" in d:
                L.append(f"| {name} | {mode} | {d['mean_crps_log']:.4f} | {fmt(d['coverage_1sigma'], '.1%')} | "
                         f"{fmt(d['spearman_sigma_error'], '.2f')} | {fmt(d['median_rel_sigma'], '.3%')} | {fmt(d['sigma_zero_share'], '.1%')} |")
    for name, d in r["e1_step150"].items():
        L.append(f"| {name} | (descriptive) | {d['mean_crps_log']:.4f} | {fmt(d['coverage_1sigma'], '.1%')} | "
                 f"{fmt(d['spearman_sigma_error'], '.2f')} | {fmt(d['median_rel_sigma'], '.3%')} | {fmt(d['sigma_zero_share'], '.1%')} |")
    L += ["", "## Determinism: D65 tapes (primary, D67) vs tonight's reruns", "",
          "| eval | rows | identical completions | identical rewards |", "|---|---|---|---|"]
    for name, v in r["determinism"].items():
        L.append(f"| {name} | missing | | |" if v == "missing" else
                 f"| {name} | {v['n']} | {v['identical']:.1%} | {v['rewards_identical']:.1%} |")
    L += ["", "## σ instability (D62): within-problem spread of sampled σ (k = 4)", "",
          "| model, sampled run | median SD of ln σ | greedy σ mean mid-rank | greedy σ below all samples |", "|---|---|---|---|"]
    for name, w in r["instability"].items():
        L.append(f"| {name} | {fmt(w['median_sd_ln_sigma'], '.2f')} | {fmt(w['greedy_mean_midrank'], '.2f')} | "
                 f"{fmt(w['greedy_below_all_samples'], '.1%')} |")
    return "\n".join(L) + "\n"


def main() -> dict:
    r = {"seed_mean": seed_mean(), "per_seed": per_seed(), "family_e": family_e(), "extra_headlines": extra_headlines(),
         "e1_step150": e1_step150(), "determinism": determinism(), "instability": instability()}
    CORR.mkdir(parents=True, exist_ok=True)
    md = render(r)
    (CORR / "side_by_side.md").write_text(md)
    (CORR / "side_by_side.json").write_text(json.dumps({**r, "per_seed": {f"seed{s} {m}": v for (s, m), v in r["per_seed"].items()}},
                                                       indent=1, default=str))
    print(md)
    return r


if __name__ == "__main__":
    main()

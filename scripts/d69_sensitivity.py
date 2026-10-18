"""D69 sensitivity: families N and E rerun with the A5 baseline re-fitted on 4-bit SFT completions.

    uv run python scripts/d69_sensitivity.py

FINDINGS #16's verdicts stand (D67 fixed them on the bf16-fitted A5). This reruns the same tests with only the A5
model swapped for results/a5_baselines_4bit (D69): family N per split-credit seed (phase95_compare.family_a on the
corrected step-287 completions) and family E (e1_compare.compare on the fixed E1 prefixes; the A5 σ is applied to the
same SFT completions the prefixes come from). Writes results/corrected/d69_sensitivity/report.{json,md}; a verdict
that differs from the primary one is flagged.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase95_compare import CORRECTED, ROOT, family_a, load  # noqa: E402

from physcal.config import load_config  # noqa: E402
from physcal.eval.nested_baselines import load_models, write_scaled_tape  # noqa: E402
from physcal.train.e1_compare import compare  # noqa: E402

T = ROOT / "results/train"
SPLIT = [T / f"split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed{s}" for s in (0, 1, 2)]
OUT = CORRECTED / "d69_sensitivity"


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    models, sel = load_models("nested_baselines_4bit")
    cand = models["sigma"][sel["strongest"]["sigma"]]
    rep = {"decision": "D69 sensitivity (not the primary verdict)", "a5_strongest_sigma": sel["strongest"]["sigma"],
           "family_N": {}, "family_E": {}}
    for run in SPLIT:
        tape = run / "compare/rl287_4bit/rollouts.jsonl"
        a5 = OUT / f"{run.name}_rl287_4bit_a5_4bit_sigma.jsonl"
        write_scaled_tape(tape, cand, a5)
        split = load(tape)
        sens = family_a(split, load(a5), sorted(split), "N", ("2", "3"))
        prim = json.loads((CORRECTED / "family_S" / f"{run.name}.json").read_text())["amendment_D49"][
            "N_vs_A5_strongest_same_completions"]
        rep["family_N"][run.name] = {"primary_bf16_a5": prim, "sensitivity_4bit_a5": sens}
    e1_tape = ROOT / load_config("e1")["eval_tapes"][0]
    e1_a5 = OUT / "e1_sft_eval_a5_4bit_sigma.jsonl"
    write_scaled_tape(e1_tape, cand, e1_a5)
    compare(corrected=True, a5_tape=str(e1_a5.relative_to(ROOT)), output_dir=str((OUT / "e1").relative_to(ROOT)))
    rep["family_E"] = {"primary_bf16_a5": json.loads((CORRECTED / "e1" / "family_E.json").read_text()),
                       "sensitivity_4bit_a5": json.loads((OUT / "e1" / "family_E.json").read_text())}
    lines = ["# D69 sensitivity: families N and E with A5 re-fitted on 4-bit SFT completions", "",
             "FINDINGS #16's verdicts (bf16-fitted A5) stand; this is a sensitivity rerun. Holm-supported = the RL σ beats "
             "the A5 σ on that test.", "",
             "| family / seed | test | RL | A5 (bf16 fit) | A5 (4-bit fit) | supported, bf16 fit | supported, 4-bit fit | changed |",
             "|---|---|---|---|---|---|---|---|"]
    flips = []
    for name, v in rep["family_N"].items():
        p, s = v["primary_bf16_a5"], v["sensitivity_4bit_a5"]
        for k in sorted(x for x in p if x.startswith("N")):
            short = k.split("_")[0]
            a, b = p["holm"][short]["supported"], s["holm"][short]["supported"]
            flips += [f"N {name[-5:]} {short}"] if a != b else []
            lines.append(f"| N {name[-5:]} | {k} | {s[k]['rl']:.4f} | {p[k]['control']:.4f} | {s[k]['control']:.4f} "
                         f"| {a} | {b} | {'**yes**' if a != b else 'no'} |")
    pe, se = rep["family_E"]["primary_bf16_a5"], rep["family_E"]["sensitivity_4bit_a5"]
    for k in (x for x in se["tests"] if x.endswith("-N")):
        a, b = pe["holm"][k]["rejected"], se["holm"][k]["rejected"]
        flips += [f"E {k}"] if a != b else []
        t1, t2 = pe["tests"][k], se["tests"][k]
        lines.append(f"| E | {k} (Δcrps_log, RL − A5) | – | {t1['mean_diff']:+.4f} | {t2['mean_diff']:+.4f} | {a} | {b} "
                     f"| {'**yes**' if a != b else 'no'} |")
    rep["changed_verdicts"] = flips
    lines += ["", f"Changed verdicts: {', '.join(flips) if flips else 'none'}."]
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return rep


if __name__ == "__main__":
    main()

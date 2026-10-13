"""Post-hoc calibration control: a single multiplicative σ scale fitted on training-seed problems.

    uv run python -m physcal.eval.posthoc generate     # GPU: SFT-only completions on the calibration set
    uv run python -m physcal.eval.posthoc fit --eval results/.../compare_v1.1.0/sft/rollouts.jsonl

`generate` runs the SFT-only model (greedy, prompt v4, the GRPO token budget) on `calibration_split`, which is
built from training seeds with the generator's train split. `fit` chooses c to maximize the mean crps_log score
(the RL objective) on those completions, applies σ' = c·σ to the given eval tape, and reports 1σ/2σ coverage and
crps_log per split. A second variant matches 1σ coverage to 68.27% on the calibration set (sensitivity check).
Failures (parse / unit) keep the rule's floor and count as not covered, as in the eval.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.tape.tape import read_meta, read_tape
from physcal.verify.reward import verifier_version
from physcal.verify.scoring import floor, relative_terms, score, sigma_floor

RULE = "crps_log"


def _config() -> dict:
    return load_config("posthoc")


def generate() -> dict:
    from physcal.eval.run_eval import run
    from physcal.train.compare import eval_config

    cfg_ph = _config()
    out = REPO_ROOT / cfg_ph["output_dir"] / "sft_calibration_set"
    sft_dir = REPO_ROOT / "results" / "sft" / cfg_ph["label"] / "merged"
    cfg = eval_config("sft", cfg_ph["label"], cfg_ph["budget"], out, sft_dir)
    cfg["output_dir"] = str(out)
    cfg["run_id"] = "posthoc-sft-calibration-set"
    cfg["splits"] = {"calibration": copy.deepcopy(cfg_ph["calibration_split"])}
    return run(cfg)


def _records(tape: Path) -> list[dict]:
    """(split, key, mu, sigma, truth) per completion; mu/sigma None for failures or missing σ."""
    out = []
    for r in read_tape(tape):
        b, p = r["breakdown"], r["problem"]
        ok = b["reason"] == "ok" and b.get("sigma_si") is not None
        out.append({"split": r.get("split", "all"), "key": (p["family"], p["depth"], p["seed"]),
                    "mu": b["mu_si"] if ok else None, "sigma": b["sigma_si"] if ok else None,
                    "truth": b["truth_si"]})
    return out


def scaled_outcomes(records: list[dict], c: float, sc: dict) -> list[dict]:
    """Per completion: crps_log score and 1σ/2σ coverage after σ' = c·σ (failures: floor, not covered)."""
    out = []
    for r in records:
        if r["mu"] is None:
            out.append({"score": floor(RULE, sc), "within_1sigma": False, "within_2sigma": False, "answered": False})
            continue
        sig = c * r["sigma"]
        e, s = relative_terms(r["mu"], sig, r["truth"], sigma_floor(RULE, sc))
        err = abs(r["mu"] - r["truth"])
        out.append({"score": score(RULE, e, s, sc), "within_1sigma": err <= sig, "within_2sigma": err <= 2 * sig,
                    "answered": True})
    return out


def summary(records: list[dict], c: float, sc: dict) -> dict:
    o = scaled_outcomes(records, c, sc)
    ans = [x for x in o if x["answered"]]
    return {"n": len(o), "n_with_sigma": len(ans),
            "coverage_1sigma": float(np.mean([x["within_1sigma"] for x in ans])) if ans else None,
            "coverage_2sigma": float(np.mean([x["within_2sigma"] for x in ans])) if ans else None,
            "mean_crps_log": float(np.mean([x["score"] for x in o]))}


def fit_crps_scale(records: list[dict], sc: dict, fit_cfg: dict) -> float:
    """argmax_c mean crps_log: a log-spaced grid, then golden-section search between the best point's neighbours."""
    lo, hi = fit_cfg["log10_scale_bounds"]
    grid = np.linspace(lo, hi, fit_cfg["grid_points"])
    mean = lambda lc: summary(records, 10 ** lc, sc)["mean_crps_log"]
    vals = [mean(g) for g in grid]
    i = int(np.argmax(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    inv_phi = (math.sqrt(5) - 1) / 2
    for _ in range(60):
        x1, x2 = b - inv_phi * (b - a), a + inv_phi * (b - a)
        if mean(x1) >= mean(x2):
            b = x2
        else:
            a = x1
    best = (a + b) / 2
    return float(10 ** (best if mean(best) >= vals[i] else grid[i]))


def _ratio(r: dict) -> float:
    err = abs(r["mu"] - r["truth"])
    if r["sigma"] > 0:
        return err / r["sigma"]
    return 0.0 if err == 0 else math.inf  # a stated σ = 0 is covered by no finite scale unless exact


def coverage_scale(records: list[dict], target: float) -> float:
    """Smallest c with 1σ coverage >= target among *all* completions that stated σ (same denominator as the
    reported coverage): the target quantile of |err|/σ. Raises if σ = 0 answers make the target unreachable."""
    ratios = sorted(_ratio(r) for r in records if r["mu"] is not None)
    c = ratios[max(math.ceil(target * len(ratios)) - 1, 0)]
    if not math.isfinite(c):
        raise ValueError(f"1σ coverage {target} is unreachable: too many answers state σ = 0")
    return float(c)


def fit(eval_tape: Path) -> dict:
    cfg_ph = _config()
    out_dir = REPO_ROOT / cfg_ph["output_dir"]
    calib_tape = out_dir / "sft_calibration_set" / "rollouts.jsonl"
    meta = read_meta(calib_tape)
    sc = meta["reward_config"]["scoring"]
    eval_meta = read_meta(eval_tape)
    if verifier_version(meta["reward_config"]) != verifier_version(eval_meta["reward_config"]):
        raise ValueError("calibration and eval tapes were scored by different verifier versions")
    calib = _records(calib_tape)
    seeds = [r["key"][2] for r in calib]
    eval_offset = meta["env_config"]["splits"]["eval_seed_offset"]
    if max(seeds) >= eval_offset:
        raise ValueError("calibration set contains eval seeds")
    evals = _records(eval_tape)
    scales = {"crps_fit": fit_crps_scale(calib, sc, cfg_ph["fit"]),
              "coverage_matched": coverage_scale(calib, cfg_ph["fit"]["coverage_target"])}
    result = {
        "verifier_version": verifier_version(meta["reward_config"]),
        "calibration_tape": str(calib_tape.relative_to(REPO_ROOT)),
        "eval_tape": str(Path(eval_tape).resolve().relative_to(REPO_ROOT)),
        "calibration_seeds": [min(seeds), max(seeds)],
        "scales": scales,
        "primary": "crps_fit",
        "calibration_set": {name: summary(calib, c, sc) for name, c in {"unscaled": 1.0, **scales}.items()},
        "eval": {},
    }
    for split in sorted({r["split"] for r in evals}):
        recs = [r for r in evals if r["split"] == split]
        result["eval"][split] = {name: summary(recs, c, sc) for name, c in {"unscaled": 1.0, **scales}.items()}
    (out_dir / "posthoc_fit.json").write_text(json.dumps(result, indent=2))
    return result


def scaled_tape(eval_tape: Path, c: float, out_path: Path) -> None:
    """A derived JSONL (not a replayable tape): the eval records with σ scaled by c and the coverage / crps_log
    fields and the reward recomputed, so scripts/paired_tests.py can compare it with another model's tape."""
    sc = read_meta(eval_tape)["reward_config"]["scoring"]
    with out_path.open("w") as fh:
        for r in read_tape(eval_tape):
            r = copy.deepcopy(r)
            b = r["breakdown"]
            if b["reason"] == "ok" and b.get("sigma_si") is not None:
                o = scaled_outcomes([{"mu": b["mu_si"], "sigma": b["sigma_si"], "truth": b["truth_si"]}], c, sc)[0]
                b["sigma_si"] = c * b["sigma_si"]
                b["rel_sigma"] = c * b["rel_sigma"]
                b["within_1sigma"], b["within_2sigma"] = o["within_1sigma"], o["within_2sigma"]
                r["reward"] = r["reward"] - b["w_final"] * b["score_final"] + b["w_final"] * o["score"]
                b["scores"][RULE] = b["score_final"] = o["score"]
                b["reward"] = r["reward"]
            r["posthoc_sigma_scale"] = c
            fh.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["generate", "fit"])
    ap.add_argument("--eval", default=None, help="SFT-only eval tape scored by the same verifier version")
    args = ap.parse_args()
    if args.action == "generate":
        generate()
        return
    res = fit(REPO_ROOT / args.eval)
    out_dir = REPO_ROOT / _config()["output_dir"]
    for name, c in res["scales"].items():
        scaled_tape(REPO_ROOT / args.eval, c, out_dir / f"sft_eval_scaled_{name}.jsonl")
    print(json.dumps({k: res[k] for k in ("scales", "calibration_set", "eval")}, indent=2))


if __name__ == "__main__":
    main()

"""How much crps_log reward is available through σ alone? (phase 3.7, CPU, verifier 1.1.0 tapes)

    uv run python scripts/phase37_sigma_headroom.py

For each eval tape, keep every μ and compare the mean crps_log score with (a) the stated σ, (b) the best single
σ scale fitted on the training-seed calibration set (results/posthoc_calibration/posthoc_fit.json), and (c) an
oracle per-problem σ = |ln(μ/y)| / 0.833, the CRPS-optimal σ for that problem's own error, which gives
CRPS = 0.5949·|ln(μ/y)|. (c) − (b) is the reward a policy can only get if its σ tracks *which* answers are wrong.
Writes results/posthoc_calibration/sigma_headroom.json.
"""

import json
import math
from pathlib import Path

from physcal.verify.scoring import crps_standard

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/train/overnight-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-512/compare_verifier_1.1.0"
NEW = ROOT / "results/train/overnight2-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512/compare"
TAPES = {"sft": OLD / "sft/rollouts.jsonl", "rl_lastnight": OLD / "rl/rollouts.jsonl",
         "rl287": NEW / "rl287/rollouts.jsonl", "rlfinal": NEW / "rlfinal/rollouts.jsonl"}
FLOOR = -5.0
Z_OPT = min((z / 10000 for z in range(1000, 30000)), key=lambda z: crps_standard(z) / z)
K = crps_standard(Z_OPT) / Z_OPT


def main() -> None:
    fit = json.loads((ROOT / "results/posthoc_calibration/posthoc_fit.json").read_text())
    out = {"oracle_z": Z_OPT, "oracle_crps_per_abs_log_error": K, "tapes": {}}
    for name, path in TAPES.items():
        if not path.exists():
            continue
        recs = [json.loads(l) for l in path.open()]
        out["tapes"][name] = {}
        for split in sorted({r["split"] for r in recs}):
            cur, orc = [], []
            for r in (r for r in recs if r["split"] == split):
                b = r["breakdown"]
                if b["reason"] != "ok":
                    cur.append(FLOOR); orc.append(FLOOR); continue
                cur.append(b["scores"]["crps_log"])
                orc.append(max(-K * abs(math.log(b["mu_si"] / b["truth_si"])), FLOOR) if b["mu_si"] > 0 else FLOOR)
            out["tapes"][name][split] = {"stated_sigma": sum(cur) / len(cur), "oracle_sigma": sum(orc) / len(orc)}
    for split, v in fit["eval"].items():
        out["tapes"]["sft"][split]["global_scale_crps_fit"] = v["crps_fit"]["mean_crps_log"]
    (ROOT / "results/posthoc_calibration/sigma_headroom.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

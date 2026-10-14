"""Sanity check after the split-credit seed-0 run (tonight's queue): no OOM and eval parse failure < 5%.

    uv run python scripts/queue_sanity.py <run dir> <training exit code>

Writes <run dir>/sanity.json and prints the verdict. Thresholds are the brief's (PREREGISTRATION.md family S:
"no OOM, no parse collapse"; the brief fixed parse collapse as >= 5% parse failures on the eval).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_PARSE_FAIL = 0.05
STEP = 287


def check(run: Path, train_rc: int) -> dict:
    out = {"training_exit_code": train_rc, "checks": {}}
    summary = run / "summary.json"
    s = json.loads(summary.read_text()) if summary.exists() else {}
    ooms = s.get("peak_vram", {}).get("num_ooms")
    mem = run / "memory.jsonl"
    if mem.exists():
        rows = [json.loads(l) for l in mem.open()]
        ooms = max([ooms or 0] + [r.get("num_ooms", 0) for r in rows])
        out["peak_reserved_gib"] = max((r["max_reserved_gib"] for r in rows), default=None)
    out["checks"]["no_oom"] = ooms == 0 and train_rc == 0
    out["num_ooms"] = ooms
    out["checks"]["reached_step"] = (run / "permanent_checkpoints" / f"checkpoint-{STEP}").is_dir()
    metrics = run / "compare" / "rl287" / "metrics.json"
    if metrics.exists():
        m = json.loads(metrics.read_text())["splits"]
        n = sum(v["n"] for v in m.values())
        pf = sum(v["parse_fail_rate"] * v["n"] for v in m.values()) / n
        out["eval_parse_fail_rate"] = pf
        out["checks"]["parse_fail_below_5pct"] = pf < MAX_PARSE_FAIL
    else:
        out["checks"]["parse_fail_below_5pct"] = False
    out["sane"] = all(out["checks"].values())
    (run / "sanity.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run = Path(sys.argv[1]) if Path(sys.argv[1]).is_absolute() else ROOT / sys.argv[1]
    res = check(run, int(sys.argv[2]))
    print(f"SANITY {'PASS' if res['sane'] else 'FAIL'}: {json.dumps(res)}")

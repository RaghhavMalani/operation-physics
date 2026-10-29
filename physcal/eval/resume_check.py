"""D142's acceptance check for resumable evals, on development problems only (never the frozen set):

    uv run python -m physcal.eval.resume_check check qwen --out <scratch>     # and: llama

The 30 development problems of D137's equivalence check (A6c + A6b, 3 per depth per set) go through `run_eval.run` with
the model's real eval config three times:
  (a) once, uninterrupted, into <out>/single;
  (b) again into <out>/resumed, SIGKILLed as soon as at least half of its batches are on disk;
  (c) restarted on <out>/resumed, so it resumes from (b)'s partial file.
- **Accepted iff:**
  - every completion of (c) is byte-identical to (a)'s (text, token count, truncation) and scores identically;
  - (c) read at least one batch from disk and generated at least one itself.
- **Models:**
  - **Qwen:** stage R4's D-pos-v2 seed-0 adapter, through stage R4's eval config (batch 16).
  - **Llama:** the stage-XM D-pos-v2 adapter, through stage XM's eval config (batch 6), i.e. the configs of the evals
    that resume in the queue.
- **Each run** is a child process (`run`), so a kill is a real kill. The children share this process's group, so the
  temperature guard stops them with it. The result goes to `results/stage_xm/checks/resume_<model>.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from physcal.config import REPO_ROOT


def eval_cfg(which: str, out: Path) -> dict:
    """The model's eval config with the 30 development problems as its two splits (files written under `out`)."""
    from physcal.eval.r5_feasibility import launch_cap
    from physcal.eval.xm_checks import _config, problems
    from physcal.train import stage_r4, stage_xm

    cfg = _config()
    cap = launch_cap({"cap_max_gib": cfg["max_reserved_gib"], "cap_margin_gib": 0.5})
    ev = stage_r4.eval_config(cfg["checks"]["qwen_arm"], seed=0, cap_gib=cap) if which == "qwen" \
        else stage_xm.eval_config("dpos2", cap_gib=cap)
    probs = problems(cfg)  # stage A's development pair files (data/eval_sets/<version>/), never confirmatory/
    split_dir = out / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    ev["splits"] = {}
    for s in cfg["checks"]["sets"]:
        f = split_dir / f"{s}.jsonl"
        f.write_text("".join(json.dumps({k: v for k, v in p.items() if k != "_set"}) + "\n"
                             for p in probs if p["_set"] == s))
        ev["splits"][s] = {"file": str(f), "verify_regenerate": -1}
    ev["run_id"] = f"resume-check-{which}-development"
    return ev


def run_once(which: str, out: Path) -> None:
    from physcal.eval.run_eval import run

    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    m = run(eval_cfg(which, out), output_dir=out)
    print(json.dumps({"resume": m["resume"], "generation_seconds": m["generation_seconds"]}), flush=True)


def _child(which: str, out: Path) -> subprocess.Popen:
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "run.log").open("a")
    return subprocess.Popen([sys.executable, "-m", "physcal.eval.resume_check", "run", which, "--out", str(out)],
                            cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)


def _batches_on_disk(out: Path) -> int:
    p = out / "completions.partial.jsonl"
    return p.read_text().count('"kind": "batch"') if p.exists() else 0


def _tape(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "rollouts.jsonl").read_text().splitlines()]


def check(which: str, out: Path, poll_s: float = 0.5) -> dict:
    from physcal.eval.xm_checks import _config

    t = {}
    t0 = time.time()
    rc_a = _child(which, out / "single").wait()
    t["single_s"] = time.time() - t0
    if rc_a != 0:
        raise RuntimeError(f"the uninterrupted run failed (exit {rc_a}); see {out / 'single' / 'run.log'}")
    n_batches = json.loads((out / "single" / "metrics.json").read_text())["resume"]["batches_total"]
    kill_at = max(1, math.ceil(n_batches / 2))
    t0 = time.time()
    proc = _child(which, out / "resumed")
    while proc.poll() is None and _batches_on_disk(out / "resumed") < kill_at:
        time.sleep(poll_s)
    if proc.poll() is not None:
        raise RuntimeError(f"run (b) ended (exit {proc.returncode}) before {kill_at} batches were on disk")
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    killed_with = _batches_on_disk(out / "resumed")
    t["killed_after_s"] = time.time() - t0
    t0 = time.time()
    rc_c = _child(which, out / "resumed").wait()
    t["resumed_s"] = time.time() - t0
    if rc_c != 0:
        raise RuntimeError(f"the resumed run failed (exit {rc_c}); see {out / 'resumed' / 'run.log'}")
    a, c = _tape(out / "single"), _tape(out / "resumed")
    key = lambda r: (r["split"], r["seed"])  # noqa: E731
    a, c = {key(r): r for r in a}, {key(r): r for r in c}
    fields = ("completion", "n_tokens", "truncated", "reward")
    same = {k: all(a[k][f] == c[k][f] for f in fields) for k in a}
    res_c = json.loads((out / "resumed" / "metrics.json").read_text())["resume"]
    res = {"decision": "D142 (acceptance: byte-identical greedy completions after a kill and a resume)", "model": which,
           "n_problems": len(a), "same_keys": set(a) == set(c), "n_identical": sum(same.values()),
           "byte_identical": set(a) == set(c) and all(same.values()), "batches_total": n_batches,
           "killed_with_batches_on_disk": killed_with, "resumed_run": res_c,
           "accepted": set(a) == set(c) and all(same.values()) and res_c["batches_resumed"] >= 1
           and res_c["batches_generated_this_session"] >= 1,
           "mismatches": [list(k) for k, s in same.items() if not s], "seconds": t,
           "batch_size": json.loads((out / "single" / "rollouts.meta.json").read_text())["generation"]["batch_size"]}
    dest = REPO_ROOT / _config()["checks"]["output_dir"]
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"resume_{which}.json").write_text(json.dumps(res, indent=1))
    with open(dest / f"resume_{which}_completions.jsonl", "w") as fh:
        for k in sorted(a):
            fh.write(json.dumps({"set": k[0], "seed": k[1], "single": a[k]["completion"],
                                 "resumed": c[k]["completion"]}) + "\n")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["check", "run"])
    ap.add_argument("model", choices=["qwen", "llama"])
    ap.add_argument("--out", required=True, help="a scratch directory")
    a = ap.parse_args()
    out = Path(a.out)
    if a.action == "run":
        run_once(a.model, out)
    else:
        res = check(a.model, out)
        print(json.dumps({k: v for k, v in res.items() if k != "mismatches"}, indent=1))
        sys.exit(0 if res["accepted"] else 1)


if __name__ == "__main__":
    main()

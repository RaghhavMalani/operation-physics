"""D66 GPU resume check (your queue, 2026-10-18): Unsloth's patched trainer + resuming a 4-bit PEFT checkpoint.

    uv run python scripts/resume_gpu_check.py

Config: configs/grpo_resume_smoke.yaml (grpo_split_credit with 4 steps, checkpoints at 2 and 4).
- Run A: 4 steps, uninterrupted.
- Run B: the same, killed (SIGKILL to its process group) as soon as step 3 has started (its step-3 metrics row is
  written at reward time), then `--resume latest`, which resumes from checkpoint-2.
PASS criteria, fixed before the run:
1. the resumed run finishes with 4 steps; its tape has 32 rows for each of steps 1-4 and replays exactly; its other
   per-step logs have one row per step; events.jsonl records the resume;
2. B's curriculum cells for steps 1-4 equal A's (chunk 3 is drawn before step 2 is observed and chunk 4 before step 3,
   so they depend only on steps 1-2, which reruns reproduce bit for bit, D63). Checked only if A and B agree on the
   step 1-2 outcomes; otherwise reported as "not comparable";
3. the 4-bit PEFT adapter was reloaded: B's step-4 LoRA weights are closer to A's step 4 than to A's step 2 (had the
   resume started from a fresh adapter, B would sit near A's step 2).
Writes results/resume_gpu_check/report.{json,md}. Run dirs: results/train/resume-smoke-{A,B}/ (checkpoints git-ignored).
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physcal.tape.tape import replay  # noqa: E402

CONFIG = "grpo_resume_smoke"
RUNS = ROOT / "results/train"
OUT = ROOT / "results/resume_gpu_check"
KILL_AT_STEP = 3
POLL_S = 2.0
STAGE_TIMEOUT_S = 1200
ROWS_PER_STEP = 32  # 4 prompts x 8 completions


def train(run_id: str, *extra: str, log: Path) -> subprocess.Popen:
    cmd = ["uv", "run", "python", "-m", "physcal.train.grpo", "--config", CONFIG, "--run-id", run_id, *extra]
    return subprocess.Popen(cmd, cwd=ROOT, stdout=log.open("w"), stderr=subprocess.STDOUT, start_new_session=True)


def lines(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def lora(ckpt: Path) -> dict:
    from safetensors.torch import load_file

    return {k: v.float() for k, v in load_file(str(ckpt / "adapter_model.safetensors")).items()}


def dist(a: dict, b: dict) -> float:
    return sum(float(((a[k] - b[k]) ** 2).sum()) for k in a) ** 0.5


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    a_dir, b_dir = RUNS / "resume-smoke-A", RUNS / "resume-smoke-B"
    for d in (a_dir, b_dir):
        shutil.rmtree(d, ignore_errors=True)
    t0 = time.time()
    a = train("resume-smoke-A", log=OUT / "run_A.log")
    rc_a = a.wait(timeout=STAGE_TIMEOUT_S)
    b = train("resume-smoke-B", log=OUT / "run_B_first.log")
    deadline = time.time() + STAGE_TIMEOUT_S
    while b.poll() is None and time.time() < deadline:
        if any(r["step"] >= KILL_AT_STEP for r in lines(b_dir / "train_metrics.jsonl")):
            os.killpg(b.pid, signal.SIGKILL)
            break
        time.sleep(POLL_S)
    b.wait(timeout=60)
    killed_after = max((r["step"] for r in lines(b_dir / "train_metrics.jsonl")), default=0)
    ckpts_at_kill = sorted(p.name for p in (b_dir / "checkpoints").glob("checkpoint-*"))
    r = train("resume-smoke-B", "--resume", "latest", log=OUT / "run_B_resume.log")
    rc_r = r.wait(timeout=STAGE_TIMEOUT_S)

    res = {"run_A_exit": rc_a, "run_B_killed_during_step": killed_after, "run_B_checkpoints_at_kill": ckpts_at_kill,
           "resume_exit": rc_r, "seconds": time.time() - t0}
    summary = json.loads((b_dir / "summary.json").read_text()) if (b_dir / "summary.json").exists() else {}
    tape = lines(b_dir / "rollouts.jsonl")
    counts = {s: sum(x["step"] == s for x in tape) for s in range(1, 5)}
    per_step = {n: [x["step"] for x in lines(b_dir / n)] for n in ("train_metrics.jsonl", "curriculum.jsonl",
                                                                       "split_credit.jsonl", "memory.jsonl")}
    events = lines(b_dir / "events.jsonl")
    c1 = {"steps_completed": summary.get("steps_completed"), "resumed_at_step": summary.get("resumed_at_step"),
          "tape_rows_per_step": counts, "tape_replays": bool(tape) and replay(b_dir / "rollouts.jsonl").ok,
          "per_step_logs": per_step, "resume_event": any(e.get("event", "").startswith("resumed from") for e in events)}
    c1["pass"] = (rc_r == 0 and c1["steps_completed"] == 4 and c1["resumed_at_step"] == 2
                  and all(v == ROWS_PER_STEP for v in counts.values()) and c1["tape_replays"]
                  and all(v == [1, 2, 3, 4] for v in per_step.values()) and c1["resume_event"])
    ca, cb = lines(a_dir / "curriculum.jsonl"), lines(b_dir / "curriculum.jsonl")
    outcomes = lambda d: [[x["by_rule"]["binary"]["score_final"] for x in lines(d / "rollouts.jsonl") if x["step"] == s]
                          for s in (1, 2)]
    same_12 = outcomes(a_dir) == outcomes(b_dir)
    c2 = {"A_cells": [x["cells"] for x in ca], "B_cells": [x["cells"] for x in cb], "step_1_2_outcomes_equal": same_12}
    c2["pass"] = (c2["A_cells"] == c2["B_cells"]) if same_12 else "not comparable"
    ck = lambda d, s: d / "checkpoints" / f"checkpoint-{s}"
    if all((ck(d, s) / "adapter_model.safetensors").exists() for d, s in ((a_dir, 2), (a_dir, 4), (b_dir, 4))):
        a2, a4, b4 = lora(ck(a_dir, 2)), lora(ck(a_dir, 4)), lora(ck(b_dir, 4))
        c3 = {"d_B4_A4": dist(b4, a4), "d_B4_A2": dist(b4, a2), "d_A4_A2": dist(a4, a2)}
        c3["pass"] = c3["d_B4_A4"] < c3["d_B4_A2"]
    else:
        c3 = {"pass": False, "missing": "a checkpoint adapter"}
    state_a, state_b = (json.loads((ck(d, 4) / "physcal_state.json").read_text()) if (ck(d, 4) / "physcal_state.json").exists()
                        else None for d in (a_dir, b_dir))
    res.update(criterion_1_logs=c1, criterion_2_curriculum=c2, criterion_3_adapter=c3,
               repeat_sampler_chunks_equal_at_step_4=(state_a or {}).get("repeat_sampler") == (state_b or {}).get("repeat_sampler"),
               verdict="PASS" if c1["pass"] and c2["pass"] in (True, "not comparable") and c3["pass"] else "FAIL")
    (OUT / "report.json").write_text(json.dumps(res, indent=2))
    md = ["# D66 GPU resume check", "", f"**Verdict: {res['verdict']}**", "",
          f"- Run A exit {rc_a}; run B killed during step {killed_after} (checkpoints then: {ckpts_at_kill}); "
          f"resume exit {rc_r}; {res['seconds']:.0f} s in all.",
          f"- 1. logs: steps completed {c1['steps_completed']}, resumed at {c1['resumed_at_step']}, tape rows per step "
          f"{counts}, replays {c1['tape_replays']}, resume event {c1['resume_event']} → {c1['pass']}",
          f"- 2. curriculum cells A vs B, steps 1–4 (step 1–2 outcomes equal: {same_12}) → {c2['pass']}",
          f"  - A: {c2['A_cells']}", f"  - B: {c2['B_cells']}",
          f"- 3. adapter reloaded: ‖B4 − A4‖ = {c3.get('d_B4_A4', float('nan')):.4g}, ‖B4 − A2‖ = "
          f"{c3.get('d_B4_A2', float('nan')):.4g} (‖A4 − A2‖ = {c3.get('d_A4_A2', float('nan')):.4g}) → {c3['pass']}",
          f"- repeat-sampler chunks equal at checkpoint-4: {res['repeat_sampler_chunks_equal_at_step_4']}"]
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return res


if __name__ == "__main__":
    sys.exit(0 if main()["verdict"] == "PASS" else 1)

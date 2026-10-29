"""D144: tonight's XM queue (scripts/queue_xm2.sh): its order, what it skips, the partial-file count it logs, and B-v2's
projection check (a run that cannot finish by the deadline is stopped; one that can is left alone). CPU only."""

import json
import os
import subprocess
import threading
import time

from physcal.config import REPO_ROOT

SCRIPTS = REPO_ROOT / "scripts"


def _dry(tmp_path, **env):
    e = {**os.environ, "DRY_RUN": "1", "DISK_GUARD": "off", "QUEUE_LOG": str(tmp_path / "q"),
         "XM_OUT": str(tmp_path / "out"), "MCB_OUT": str(tmp_path / "mcb"), "GPU_DEADLINE": "13:00", **env}
    r = subprocess.run(["bash", str(SCRIPTS / "queue_xm2.sh")], capture_output=True, text=True, env=e, timeout=600)
    assert r.returncode == 0, r.stderr
    log = (tmp_path / "q" / "chain.log").read_text()
    return [ln.split("DRY ", 1)[1].split(" ", 1)[0] for ln in log.splitlines()
            if "] DRY " in ln and "commit" not in ln and "disk" not in ln], log


def test_tonights_order_eval_go_rule_mc_b_then_b2_sft_and_nothing_else(tmp_path):
    names, log = _dry(tmp_path)
    assert names == ["xm_eval_dpos2_a1", "xm_go", "mc_b_gen_a6b_pairs", "mc_b_exec_a6b_pairs", "mc_b_analyze_a6b_pairs",
                     "xm_train_b2"]
    assert "PHYSCAL_CONFIRMATORY=1 uv run python -m physcal.train.stage_xm eval --arm dpos2 --max-reserved-gib 5.9" in log
    assert "physcal.eval.xm_go analyze" in log and "nice -n 19" in log
    assert "physcal.train.stage_xm train --arm b2 --stop-at" in log
    assert "stage_xm_seed" not in log and "eval --arm b2" not in log and "xm_pilot" not in log  # no seed 1, no B-v2 eval


def test_what_is_done_is_skipped(tmp_path):
    ev = tmp_path / "out" / "dpos2" / "seed0" / "confirmatory"
    ev.mkdir(parents=True)
    (ev / "metrics.json").write_text("{}")
    (tmp_path / "out" / "go_dpos2").mkdir()
    (tmp_path / "out" / "go_dpos2" / "go.json").write_text("{}")
    (tmp_path / "out" / "b2" / "seed0" / "final_adapter").mkdir(parents=True)
    m = tmp_path / "mcb" / "a6b_pairs"
    m.mkdir(parents=True)
    (m / "generations.jsonl").write_text("{}\n")
    (m / "meta.json").write_text(json.dumps({"n_problems": 1, "finished": "x"}))
    (m / "results.json").write_text("{}")
    names, log = _dry(tmp_path)
    assert names == []
    for s in ("SKIP eval xm dpos2: already evaluated", "SKIP xm_go", "SKIP mc_b generate a6b_pairs",
              "SKIP mc_b analysis a6b_pairs", "SKIP train xm b2"):
        assert s in log


def _lib(tmp_path, script: str, **env) -> subprocess.CompletedProcess:
    e = {**os.environ, "XM2_LIB_ONLY": "1", "QUEUE_LOG": str(tmp_path / "q"), "DISK_GUARD": "off", **env}
    return subprocess.run(["bash", "-c", f". ./queue_xm2.sh && {script}"], cwd=SCRIPTS, capture_output=True, text=True,
                          env=e, timeout=120)


def test_partial_batches_counts_the_batch_lines_of_the_partial_file(tmp_path):
    d = tmp_path / "ev"
    d.mkdir()
    assert _lib(tmp_path, f"partial_batches {d}").stdout.strip() == "0"
    lines = [{"kind": "header"}, {"kind": "session"}, {"kind": "batch", "b": 0}, {"kind": "batch", "b": 1}]
    (d / "completions.partial.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines[:2]))
    assert _lib(tmp_path, f"partial_batches {d}").stdout.strip() == "0"
    (d / "completions.partial.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert _lib(tmp_path, f"partial_batches {d}").stdout.strip() == "2"


def _projection(tmp_path, deadline_in_s: int):
    """A fake B-v2 stage (its own process group, like timeout) whose log shows 40 / 750 steps after 10 min."""
    q = tmp_path / "q"
    q.mkdir()
    stage = subprocess.Popen(["sleep", "60"], start_new_session=True)
    (q / "xm_train_b2.pid").write_text(str(stage.pid))
    (q / "xm_train_b2.log").write_text("Unsloth\r  5%|▌  | 39/750 [09:45<1:50:00, 9.3s/it]\r  5%|▌  | 40/750 [10:00<1:50:00]")
    t0 = int(time.time()) - 1
    r = _lib(tmp_path, f"DEADLINE=$(( $(date +%s) + {deadline_in_s} )); b2_projection {t0}", PROJ_FIRST_S="0",
             PROJ_EVERY_S="1", PROJ_GRACE_S="1", SLACK_S="0", B2_TAIL_S="600")
    return stage, r, (q / "chain.log").read_text() if (q / "chain.log").exists() else ""


def test_the_projection_stops_a_b2_run_that_cannot_finish(tmp_path):
    # 40 steps in 600 s -> 710 more need 10,650 s + 600 s tail: past a deadline 3 h away
    stage, r, log = _projection(tmp_path, 3 * 3600)
    assert stage.wait(timeout=30) != 0
    assert "B-v2 cannot finish by" in log and "40 / 750 steps in 10:00" in log and "stopping it (D144)" in log


def test_the_projection_leaves_a_b2_run_that_can_finish(tmp_path):
    q = tmp_path / "q"
    q.mkdir()
    stage = subprocess.Popen(["sleep", "3"], start_new_session=True)
    reaper = threading.Thread(target=stage.wait)  # as run_stage's `wait` does: no zombie answering kill -0
    reaper.start()
    (q / "xm_train_b2.pid").write_text(str(stage.pid))
    (q / "xm_train_b2.log").write_text("\r  5%|▌  | 40/750 [10:00<1:50:00]")
    _lib(tmp_path, f"DEADLINE=$(( $(date +%s) + 5 * 3600 )); b2_projection {int(time.time()) - 1}",
         PROJ_FIRST_S="0", PROJ_EVERY_S="1", PROJ_GRACE_S="1", SLACK_S="0", B2_TAIL_S="600")
    reaper.join(timeout=30)
    assert stage.returncode == 0  # ran to its end
    log = (q / "chain.log").read_text()
    assert "B-v2 projection: 40 / 750 steps in 10:00" in log and "cannot finish" not in log


def test_the_queue_is_executable():
    assert os.access(SCRIPTS / "queue_xm2.sh", os.X_OK)

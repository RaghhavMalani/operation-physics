"""D147: the R5-6 J queue (scripts/queue_r5_6.sh): six resumable evals in D118's order on the frozen J files, X1 once,
then the analysis once; what is done is skipped. CPU only."""

import os
import subprocess

from physcal.config import REPO_ROOT

SCRIPTS = REPO_ROOT / "scripts"
ORDER = [("dpos2", 0), ("b2", 0), ("dpos2", 1), ("b2", 1), ("dpos2", 2), ("b2", 2)]


def _dry(tmp_path, **env):
    e = {**os.environ, "DRY_RUN": "1", "DISK_GUARD": "off", "QUEUE_LOG": str(tmp_path / "q"),
         "QUEUE_OUT": str(tmp_path / "out"), "R56_OUT": str(tmp_path / "r56"), "X1_OUT": str(tmp_path / "x1"),
         "GPU_DEADLINE": "10:00", **env}
    r = subprocess.run(["bash", str(SCRIPTS / "queue_r5_6.sh")], capture_output=True, text=True, env=e, timeout=600)
    assert r.returncode == 0, r.stderr
    log = (tmp_path / "q" / "chain.log").read_text()
    return [ln.split("DRY ", 1)[1].split(" ", 1)[0] for ln in log.splitlines()
            if "] DRY " in ln and "commit" not in ln and "disk" not in ln], log


def test_six_evals_in_d118s_order_then_x1_then_the_analysis(tmp_path):
    names, log = _dry(tmp_path)
    assert names == [f"r56_eval_{a}_s{s}_a1" for a, s in ORDER] + ["x1_r5_6", "r5_6_analysis"]
    for a, s in ORDER:
        assert (f"PHYSCAL_CONFIRMATORY=1 uv run python -m physcal.train.stage_r4 eval --arm {a} --seed {s} --r5-6 "
                "--max-reserved-gib 5.9") in log
    assert "PHYSCAL_CONFIRMATORY=1 uv run python -m physcal.eval.r5_6 analyze" in log
    assert "PHYSCAL_CONFIRMATORY=1 uv run python -m physcal.eval.x1_a6c r5_6" in log
    assert log.count("OMP_NUM_THREADS=1") >= 2 and "CUDA_VISIBLE_DEVICES= " in log and "nice -n 19" in log
    assert "--confirmatory" not in log and " train " not in log and "mc_b" not in log


def test_what_is_done_is_skipped(tmp_path):
    for a, s in ORDER:
        ev = tmp_path / "out" / a / f"seed{s}" / "r5_6"
        ev.mkdir(parents=True)
        (ev / "metrics.json").write_text("{}")
    for d in ("r56", "x1"):
        (tmp_path / d).mkdir()
    (tmp_path / "r56" / "results.json").write_text("{}")
    (tmp_path / "x1" / "predictions.jsonl").write_text("")
    names, log = _dry(tmp_path)
    assert names == []
    assert log.count("SKIP eval r5_6") == 6 and "SKIP x1_r5_6" in log and "SKIP r5_6 analysis" in log


def test_the_queue_is_executable():
    assert os.access(SCRIPTS / "queue_r5_6.sh", os.X_OK)

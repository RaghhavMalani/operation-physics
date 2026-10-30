"""D146: the day queue (scripts/queue_xm3.sh): the B-v2 Llama eval through queue_xm2.sh's eval_xm, then the pilot
analysis once; what is done is skipped. CPU only."""

import os
import subprocess

from physcal.config import REPO_ROOT

SCRIPTS = REPO_ROOT / "scripts"


def _dry(tmp_path, **env):
    e = {**os.environ, "DRY_RUN": "1", "DISK_GUARD": "off", "QUEUE_LOG": str(tmp_path / "q"),
         "XM_OUT": str(tmp_path / "out"), "GPU_DEADLINE": "13:00", **env}
    r = subprocess.run(["bash", str(SCRIPTS / "queue_xm3.sh")], capture_output=True, text=True, env=e, timeout=600)
    assert r.returncode == 0, r.stderr
    log = (tmp_path / "q" / "chain.log").read_text()
    return [ln.split("DRY ", 1)[1].split(" ", 1)[0] for ln in log.splitlines()
            if "] DRY " in ln and "commit" not in ln and "disk" not in ln], log


def test_the_b2_eval_then_the_pilot_analysis_and_nothing_else(tmp_path):
    names, log = _dry(tmp_path)
    assert names == ["xm_eval_b2_a1", "xm_pilot"]
    assert "PHYSCAL_CONFIRMATORY=1 uv run python -m physcal.train.stage_xm eval --arm b2 --max-reserved-gib 5.9" in log
    assert "PHYSCAL_CONFIRMATORY=1 uv run python -m physcal.eval.xm_pilot analyze" in log and "nice -n 19" in log
    assert "OMP_NUM_THREADS=1" in log and "CUDA_VISIBLE_DEVICES= " in log  # the analysis is CPU, 1 thread
    assert "--arm dpos2" not in log and "stage_xm train" not in log and "mc_b" not in log and "xm_go" not in log


def test_what_is_done_is_skipped(tmp_path):
    ev = tmp_path / "out" / "b2" / "seed0" / "confirmatory"
    ev.mkdir(parents=True)
    (ev / "metrics.json").write_text("{}")
    (tmp_path / "out" / "pilot").mkdir()
    (tmp_path / "out" / "pilot" / "results.json").write_text("{}")
    names, log = _dry(tmp_path)
    assert names == []
    assert "SKIP eval xm b2: already evaluated" in log and "SKIP xm_pilot: the one-time analysis exists" in log


def test_the_queue_is_executable():
    assert os.access(SCRIPTS / "queue_xm3.sh", os.X_OK)

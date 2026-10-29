"""D143: the temperature guard's hysteresis. Above 87 C the stage is stopped as before; it continues at the first reading
at or below 75 C (read every CHECK_S seconds), or after PAUSE_S seconds at most; an unreadable reading never ends a pause.
Each pause is logged with its actual length. The between-stage cool() follows the same rule. CPU only, fake readings."""

import os
import re
import subprocess
import threading
import time

from physcal.config import REPO_ROOT

SCRIPTS = REPO_ROOT / "scripts"


def _state(pid: int) -> str:
    return subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()


def _guard(tmp_path, readings: list[tuple[float, str]], pause_s: int = 4, stage_s: int = 8):
    """Run the guard on a `sleep` stage; `readings` = [(seconds after start, reading)], written to the file TEMP_CMD
    reads. Returns the guard log and the stage's states sampled every 0.1 s."""
    temp, log = tmp_path / "temp", tmp_path / "chain.log"
    temp.write_text(readings[0][1])
    stage = subprocess.Popen(["sleep", str(stage_s)], start_new_session=True)  # leads its own group, like timeout
    guard = subprocess.Popen([str(SCRIPTS / "gpu_temp_guard.sh"), str(stage.pid), str(log)],
                             env={**os.environ, "TEMP_CMD": f"cat {temp}", "POLL_S": "1", "CHECK_S": "1",
                                  "PAUSE_S": str(pause_s)})
    t0, states = time.time(), []

    def feed():
        for at, value in readings[1:]:
            time.sleep(max(0.0, at - (time.time() - t0)))
            temp.write_text(value)

    th = threading.Thread(target=feed)
    th.start()
    while stage.poll() is None:
        states.append((time.time() - t0, _state(stage.pid)))
        time.sleep(0.1)
    th.join()
    guard.wait(timeout=60)
    return (log.read_text() if log.exists() else ""), states


def _paused_s(log: str) -> list[int]:
    return [int(x) for x in re.findall(r"stage continued after (\d+)s paused", log)]


def test_a_pause_ends_at_the_first_reading_at_or_below_75_c(tmp_path):
    log, states = _guard(tmp_path, [(0, "88.10"), (2.5, "80.00"), (3.5, "75.00")], pause_s=30)
    assert "GPU at 88.10C > 87C: stage stopped (continues at <= 75C, after 30s at most)" in log
    assert "GPU at 75.00C <= 75C" in log
    (paused,) = _paused_s(log)
    assert 1 <= paused <= 5  # not 30: 75.00 counts as cool
    assert any(s.startswith("T") for _, s in states)  # the group really was stopped


def test_a_pause_that_never_cools_lasts_the_maximum(tmp_path):
    log, _ = _guard(tmp_path, [(0, "88.10"), (2.0, "80.00")], pause_s=3, stage_s=6)
    assert "the 3s maximum" in log
    assert _paused_s(log)[0] == 3


def test_an_unreadable_temperature_never_ends_a_pause_early(tmp_path):
    log, _ = _guard(tmp_path, [(0, "88.10"), (1.5, "")], pause_s=3, stage_s=6)
    assert _paused_s(log)[0] == 3 and "the 3s maximum" in log


def test_a_reading_just_above_75_c_keeps_the_pause(tmp_path):
    log, _ = _guard(tmp_path, [(0, "88.10"), (1.5, "75.10")], pause_s=3, stage_s=6)
    assert _paused_s(log)[0] == 3


def test_no_pause_at_or_below_87_c(tmp_path):
    log, states = _guard(tmp_path, [(0, "87.00")], stage_s=3)
    assert log == "" and not any(s.startswith("T") for _, s in states)


def test_cool_between_stages_waits_for_75_c_then_logs_the_pause(tmp_path):
    d = tmp_path / "bin"
    d.mkdir()
    n = tmp_path / "n"
    n.write_text("0")
    smi = d / "nvidia-smi"  # 88.2, 81.0, then 74.5: the second reading during the pause is cool enough
    smi.write_text(f"""#!/usr/bin/env bash
case "$*" in
  *temperature.gpu*) i=$(cat {n}); echo $((i + 1)) > {n}
                     case $i in 0) echo 88.20;; 1) echo 81.00;; *) echo 74.50;; esac;;
  *memory.used*) echo 900;;
esac
""")
    smi.chmod(0o755)
    e = {**os.environ, "QUEUE_LIB_ONLY": "1", "QUEUE_LOG": str(tmp_path / "log"), "PATH": f"{d}:{os.environ['PATH']}",
         "COOL_CHECK_S": "1", "COOL_MAX_S": "30"}
    t0 = time.time()
    r = subprocess.run(["bash", "-c", ". ./queue_r4_seeds.sh && cool"], cwd=SCRIPTS, capture_output=True, text=True,
                       env=e, timeout=60)
    assert r.returncode == 0 and time.time() - t0 < 10
    assert "GPU at 88.20C > 87C: pausing until <= 75C (30s at most)" in r.stdout
    assert re.search(r"pause ended after [12]s \(GPU at 74.50C\)", r.stdout)
    assert "GPU at 74.50C, memory.used 900 MiB" in r.stdout

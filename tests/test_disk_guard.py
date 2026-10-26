"""D111: the free-space guard (scripts/disk_guard.sh) with a fake powershell.exe; every queue script sources it and
guards every stage launcher; a real queue stops cleanly (BLOCKED, exit 75) before its first stage when space is low."""

import re
import subprocess
import time
from pathlib import Path

import pytest

from physcal.config import REPO_ROOT

SCRIPTS = REPO_ROOT / "scripts"
GIB = 1024**3


def _fake_ps(tmp_path: Path, out: str | None, rc: int = 0) -> Path:
    p = tmp_path / "powershell.exe"
    p.write_text("#!/usr/bin/env bash\n" + (f"printf '{out}\\r\\n'\n" if out is not None else "") + f"exit {rc}\n")
    p.chmod(0o755)
    return p


def _guard(tmp_path, out, rc=0, **env):
    ps = _fake_ps(tmp_path, out, rc)
    e = {"PATH": "/usr/bin:/bin", "PS_EXE": str(ps), "DISK_GUARD_RETRY_S": "0", **{k: str(v) for k, v in env.items()}}
    cmd = f'log() {{ echo "LOG $*"; }}; . "{SCRIPTS / "disk_guard.sh"}"; disk_guard stage_x; echo "rc=$?"'
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, env=e, timeout=60)
    return r.stdout


def test_enough_space_passes(tmp_path):
    out = _guard(tmp_path, str(40 * GIB))
    assert "LOG disk: 40 GiB free on MLStorage" in out and "rc=0" in out


def test_low_space_blocks(tmp_path):
    out = _guard(tmp_path, str(14 * GIB + GIB // 2))  # 14.5 GiB: below 15
    assert "LOG BLOCKED stage_x: 14 GiB free on MLStorage < 15 GiB; queue stopped" in out and "rc=1" in out
    assert "rc=0" in _guard(tmp_path, str(15 * GIB))  # exactly the minimum passes


def test_unreadable_fails_closed_after_the_retries(tmp_path):
    out = _guard(tmp_path, None, rc=1, DISK_GUARD_TRIES=2)
    assert "BLOCKED stage_x: free space on MLStorage could not be read" in out and "rc=1" in out


def test_dry_run_and_bypass_never_block(tmp_path):
    assert "rc=0" in _guard(tmp_path, str(GIB), DRY_RUN=1) and "DRY disk guard: 1 GiB" in _guard(tmp_path, str(GIB), DRY_RUN=1)
    assert "rc=0" in _guard(tmp_path, None, rc=1, DRY_RUN=1, DISK_GUARD_TRIES=1)
    out = _guard(tmp_path, str(GIB), DISK_GUARD="off")
    assert "disk guard OFF" in out and "rc=0" in out


def test_threshold_and_label_are_settable(tmp_path):
    assert "rc=1" in _guard(tmp_path, str(40 * GIB), MIN_FREE_GB=50)
    assert "on Other" in _guard(tmp_path, str(40 * GIB), DISK_GUARD_LABEL="Other")


@pytest.mark.parametrize("script", sorted(p.name for p in SCRIPTS.glob("queue_*.sh")))
def test_every_queue_script_guards_every_stage_launcher(script):
    lines = (SCRIPTS / script).read_text().splitlines()
    cd = lines.index('cd "$(dirname "$0")/.."')
    assert lines[cd + 1].startswith(". scripts/disk_guard.sh"), "the helper is sourced right after the cd"
    launchers = [i for i, line in enumerate(lines) if re.match(r"^(run_stage|run_gpu|run_cpu)\(\) \{", line)]
    assert launchers, "a queue script without a stage launcher"
    for i in launchers:
        j = next(k for k in range(i + 1, len(lines)) if lines[k].startswith("  local name="))
        assert j == i + 1 and lines[j + 1].startswith('  disk_guard "$name" || exit 75'), lines[i]


def test_a_real_queue_stops_before_its_first_stage(tmp_path):
    ps = _fake_ps(tmp_path, str(GIB))  # 1 GiB free
    log = tmp_path / "log"
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PS_EXE": str(ps), "QUEUE_LOG": str(log),
           "GIT_LOCK": str(tmp_path / "lock"),  # six hours ahead (IST), so the arm's time gate never skips it first
           "GPU_DEADLINE": time.strftime("%H:%M", time.gmtime(time.time() + 5.5 * 3600 + 6 * 3600))}
    r = subprocess.run(["bash", str(SCRIPTS / "queue_stage_d.sh")], capture_output=True, text=True, env=env, timeout=120)
    chain = (log / "chain.log").read_text()
    assert r.returncode == 75, r.stdout + r.stderr
    assert "BLOCKED stage_d_train_pos: 1 GiB free on MLStorage < 15 GiB; queue stopped" in chain
    assert "start stage_d_train_pos" not in chain and not (log / "stage_d_train_pos.log").exists()

"""D171's stage runner: atomic chunks, resume, the deadline, the pins, restore and the zip. CPU only."""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import zipfile

import pytest

from physcal.eval import kaggle_stages as S


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _run(root, pins=None, clock=None, hours=1.0, worker="main"):
    return S.Run(root, pins or {"commit": "a"}, 1000.0, hours, 1.25, worker, clock or Clock())


def _writer(calls, clock=None, dt=0.0):
    def write(key, tmp):
        calls.append(key)
        if clock:
            clock.t += dt
        tmp.write_text(key)
    return write


def test_a_rerun_skips_what_exists_and_finish_records_every_chunk(tmp_path):
    keys, calls = ["a", "b", "c"], []
    r = _run(tmp_path / "w")
    assert r.chunks("s1", keys[:2], _writer(calls), 1.0, log=lambda m: None)
    assert _run(tmp_path / "w").chunks("s1", keys, _writer(calls), 1.0, log=lambda m: None)
    assert calls == ["a", "b", "c"]
    res = r.finish("s1", keys)
    assert set(res["chunks"]) == set(keys) and r.done("s1")
    assert r.verify()["chunks_checked"] == 3


def test_a_failed_chunk_leaves_nothing_under_its_name_and_is_redone(tmp_path):
    r = _run(tmp_path / "w")

    def bad(key, tmp):
        tmp.write_text("half")
        raise RuntimeError("killed")
    with pytest.raises(RuntimeError):
        r.chunks("s1", ["a"], bad, 1.0, log=lambda m: None)
    assert not r.has("s1", "a")
    calls = []
    assert r.chunks("s1", ["a"], _writer(calls), 1.0, log=lambda m: None) and calls == ["a"]
    assert r.path("s1", "a").read_text() == "a"


def test_no_chunk_starts_after_the_deadline_and_the_projection_uses_the_margin(tmp_path):
    clock, calls = Clock(), []
    r = S.Run(tmp_path / "w", {"c": 1}, 1000.0, 100 / 3600, 1.25, now=clock)
    # the deadline is t = 1100: 40 s chunks, projected at 50 s
    assert not r.chunks("s", ["a", "b", "c", "d"], _writer(calls, clock, 40.0), 40.0, log=lambda m: None)
    assert calls == ["a", "b"] and clock.t == 1080.0


def test_the_first_chunk_is_projected_from_the_estimate(tmp_path):
    r = S.Run(tmp_path / "w", {"c": 1}, 1000.0, 100 / 3600, 1.25, now=Clock())
    assert r.may_start("s", 80.0) and not r.may_start("s", 81.0)


def test_other_pins_are_refused(tmp_path):
    _run(tmp_path / "w", {"commit": "a", "config": "x"})
    with pytest.raises(SystemExit, match="commit"):
        _run(tmp_path / "w", {"commit": "b", "config": "x"})


def test_finish_refuses_a_missing_or_changed_chunk(tmp_path):
    r = _run(tmp_path / "w")
    r.chunks("s", ["a"], _writer([]), 1.0, log=lambda m: None)
    with pytest.raises(SystemExit, match="missing"):
        r.finish("s", ["a", "b"])
    r.path("s", "a").write_text("changed")
    with pytest.raises(SystemExit, match="ledger"):
        r.finish("s", ["a"])
    with pytest.raises(SystemExit, match="differs"):
        r.verify()


def test_two_workers_share_a_stage_through_their_own_ledgers(tmp_path):
    a, b = _run(tmp_path / "w", worker="g0"), _run(tmp_path / "w", worker="g1")
    a.chunks("s", ["x"], _writer([]), 1.0, log=lambda m: None)
    b.chunks("s", ["y"], _writer([]), 1.0, log=lambda m: None)
    assert set(a.ledger("s")) == {"x", "y"}
    assert set(a.finish("s", ["x", "y"])["chunks"]) == {"x", "y"}


def test_two_processes_share_a_queue_and_take_each_chunk_once(tmp_path):
    """D180: two real worker processes on one ordered list; every chunk is written once, by whichever worker was free."""
    root = tmp_path / "w"
    code = textwrap.dedent(f"""
        import sys, time
        from pathlib import Path
        from physcal.eval import kaggle_stages as S
        r = S.Run(Path({str(root)!r}), {{"commit": "a"}}, time.time(), 1.0, 1.25, sys.argv[1])
        def write(key, tmp):
            time.sleep(0.15)
            tmp.write_text(key + sys.argv[1])
        assert r.queue("s", [f"k{{i}}" for i in range(12)], write, 1.0, log=lambda m: None) in (True, False)
    """)
    env = {**os.environ, "PYTHONPATH": os.getcwd()}
    ps = [subprocess.Popen([sys.executable, "-c", code, w], env=env) for w in ("g0", "g1")]
    assert [p.wait() for p in ps] == [0, 0]
    keys = [f"k{i}" for i in range(12)]
    per = {w: [json.loads(x)["key"] for x in (root / "s" / f"ledger_{w}.jsonl").read_text().splitlines()] for w in ("g0", "g1")}
    assert sorted(per["g0"] + per["g1"]) == sorted(keys) and per["g0"] and per["g1"]
    r = _run(root)
    assert set(r.finish("s", keys)["chunks"]) == set(keys) and not list((root / "s").glob("*.claim"))


def test_the_queue_waits_while_a_key_may_become_ready_and_stops_when_it_cannot(tmp_path):
    r, calls, ready = _run(tmp_path / "w"), [], {"b": False}

    def sleep(s):                                       # the earlier stage finishes during the third poll
        calls.append("poll")
        if calls.count("poll") == 3:
            ready["b"] = True
    assert r.queue("s", ["a", "b"], _writer(calls), 1.0, ready=lambda k: k == "a" or ready["b"], waiting=lambda: True,
                   poll_s=0, log=lambda m: None, sleep=sleep)
    assert calls == ["a", "poll", "poll", "poll", "b"]
    r2, calls2 = _run(tmp_path / "w2"), []
    assert not r2.queue("s", ["a", "b"], _writer(calls2), 1.0, ready=lambda k: k == "a", waiting=lambda: False,
                        log=lambda m: None, sleep=lambda s: calls2.append("poll"))
    assert calls2 == ["a"]
    clock = Clock()
    r3, calls3 = _run(tmp_path / "w3", clock=clock), []
    assert not r3.queue("s", ["a", "b"], _writer(calls3, clock, 2000.0), 1000.0, ready=lambda k: k == "a",
                        waiting=lambda: True, log=lambda m: None, sleep=lambda s: calls3.append("poll"))
    assert calls3 == ["a"]                              # waiting is pointless once no chunk could end before the deadline


def test_a_claim_holds_its_chunk_until_cleared_and_restore_leaves_claims_behind(tmp_path):
    r, calls = _run(tmp_path / "w"), []
    assert r.claim("s", "a") and not r.claim("s", "a")
    assert not r.queue("s", ["a", "b"], _writer(calls), 1.0, log=lambda m: None) and calls == ["b"]
    src = tmp_path / "input" / "v1"
    shutil.copytree(tmp_path / "w", src / "w")
    assert S.restore(tmp_path / "w2" / "w", src, log=lambda m: None)["restored"] >= 1
    assert not list((tmp_path / "w2").rglob("*.claim"))
    assert r.clear_claims("s") == 2 and r.queue("s", ["a", "b"], _writer(calls), 1.0, log=lambda m: None)
    assert calls == ["b", "a"]


def test_a_hard_kill_mid_chunk_loses_only_that_chunk(tmp_path):
    """SIGKILL inside the chunk writer of a real process: the finished chunk stays and the re-run redoes the other."""
    root = tmp_path / "w"
    code = textwrap.dedent(f"""
        import os, signal
        from pathlib import Path
        from physcal.eval import kaggle_stages as S
        r = S.Run(Path({str(root)!r}), {{"commit": "a"}}, __import__("time").time(), 1.0, 1.25)
        def write(key, tmp):
            tmp.write_text("part")
            if key == "b":
                os.kill(os.getpid(), signal.SIGKILL)
            tmp.write_text(key)
        r.chunks("s", ["a", "b", "c"], write, 1.0)
    """)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert p.returncode == -signal.SIGKILL
    r = _run(root, clock=Clock(0.0))
    assert r.has("s", "a") and not r.has("s", "b")
    assert sorted(f.name for f in (root / "s").iterdir() if f.name.endswith(".tmp")) == ["b.tmp"]
    calls = []
    assert r.chunks("s", ["a", "b", "c"], _writer(calls), 1.0, log=lambda m: None) and calls == ["b", "c"]
    r.finish("s", ["a", "b", "c"])
    assert r.verify()["chunks_checked"] == 3


def test_restore_copies_a_previous_version_and_a_bad_file_is_caught(tmp_path):
    src = tmp_path / "input" / "some-notebook" / "w"
    r = S.Run(src, {"commit": "a"}, 1000.0, 1.0, 1.25, now=Clock())
    r.chunks("s", ["a", "b"], _writer([]), 1.0, log=lambda m: None)
    (src / "s" / "c.tmp").write_text("junk")
    dest = tmp_path / "work" / "w"
    assert S.restore(dest, tmp_path / "input", log=lambda m: None)["restored"] == 4   # pins, 2 chunks, ledger
    assert not (dest / "s" / "c.tmp").exists()
    r2 = _run(dest)
    calls = []
    assert r2.verify()["chunks_checked"] == 2
    assert r2.chunks("s", ["a", "b"], _writer(calls), 1.0, log=lambda m: None) and calls == []
    (dest / "s" / "a").write_text("flipped")
    with pytest.raises(SystemExit):
        r2.verify()


def test_restore_finds_nothing_on_a_first_session_and_refuses_two_runs(tmp_path):
    assert S.restore(tmp_path / "work" / "w", tmp_path, log=lambda m: None) == {"restored": 0}
    assert S.restore(tmp_path / "work" / "w", tmp_path / "no_such_input", log=lambda m: None) == {"restored": 0}
    for n in ("a", "b"):
        _run(tmp_path / "input" / n / "w")
    with pytest.raises(SystemExit, match="attach one"):
        S.restore(tmp_path / "work" / "w", tmp_path / "input", log=lambda m: None)


def test_the_zip_round_trips_and_its_checks_refuse(tmp_path):
    z, allow = tmp_path / "r.zip", re.compile(r"results\.json|logs/[a-z]+\.log")
    assert S.write_zip(z, {"results.json": b"{}", "logs/a.log": b"x"}, 1.0)["files"] == 2
    assert S.read_zip(z, allow, 1.0) == {"results.json": b"{}", "logs/a.log": b"x"}
    with pytest.raises(SystemExit, match="allow-list"):
        S.read_zip(z, re.compile(r"results\.json"), 1.0)
    with pytest.raises(SystemExit, match="above"):
        S.read_zip(z, allow, 0.0)
    with pytest.raises(SystemExit, match="limit"):
        S.write_zip(tmp_path / "big.zip", {"a": os.urandom(2_000_000)}, 1.0)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("results.json", b"{}")
        zf.writestr(S.MANIFEST, json.dumps({"results.json": "0" * 64}))
    with pytest.raises(SystemExit, match="manifest"):
        S.read_zip(bad, allow, 1.0)

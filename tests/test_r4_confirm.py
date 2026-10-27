"""D118 (the R4-3 confirmatory run): stage R4's seed and confirmatory options, tonight's queue (dry run, the launch
cap with a fake nvidia-smi) and the 23:00 waiter (scripts/run_at.sh). CPU only."""

import copy
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from physcal.config import REPO_ROOT, load_config

SCRIPTS = REPO_ROOT / "scripts"


# ------------------------------------------------------------------------------------------- stage R4 options
def test_seed_and_confirmatory_eval_configs():
    from physcal.train import stage_r4

    cfg = load_config("stage_r4")
    assert cfg["confirmatory"]["seeds"] == [0, 1, 2]
    assert stage_r4.arm_dir("dpos2", seed=1) == REPO_ROOT / "results/stage_r4/dpos2/seed1"
    assert stage_r4.arm_dir("b2") == REPO_ROOT / "results/stage_r4/b2/seed0"  # the default is unchanged
    assert stage_r4.arm_config("b2", seed=2)["seeds"] == [2]
    dev = stage_r4.eval_config("b2", seed=2)
    conf = stage_r4.eval_config("b2", seed=2, confirmatory=True)
    assert conf["run_id"] == "stage-r4-b2-seed2-confirmatory"
    assert conf["output_dir"].endswith("results/stage_r4/b2/seed2/confirmatory")
    assert conf["model"]["lora"].endswith("results/stage_r4/b2/seed2/final_adapter")
    assert sorted(conf["splits"]) == ["a6_pairs", "a6b_pairs", "a6c_pairs"]
    for name, sp in conf["splits"].items():
        assert sp["file"] == f"data/eval_sets/2.0.0+bdbb2279/confirmatory/{name}.jsonl" and sp["verify_regenerate"] == -1
    for k in ("run_id", "output_dir", "splits"):  # everything else is the development eval's (D118 §2)
        dev.pop(k), conf.pop(k)
    assert conf == dev
    assert dev["generation"]["max_new_tokens"] == 1280 and dev["generation"]["batch_size"] == 16
    with pytest.raises(ValueError):
        stage_r4.eval_config("dpos2", seed=3, confirmatory=True)
    assert stage_r4.eval_config("dpos2", seed=1, cap_gib=5.2)["model"]["max_reserved_gib"] == 5.2


def test_the_cap_is_never_raised():
    r = subprocess.run(["uv", "run", "python", "-m", "physcal.train.stage_r4", "eval", "--arm", "b2", "--seed", "1",
                        "--max-reserved-gib", "7.1"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "never raised" in r.stderr


# ------------------------------------------------------------------------------------------- the queue
def _fake_smi(tmp_path: Path, used: int, total: int = 8188, temp: int = 60) -> Path:
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    p = d / "nvidia-smi"
    p.write_text("#!/usr/bin/env bash\ncase \"$*\" in\n"
                 f"  *memory.used*) echo {used};;\n  *memory.total*) echo {total};;\n  *temperature*) echo {temp};;\nesac\n")
    p.chmod(0o755)
    return d


def _cap(tmp_path, used, **env):
    d = _fake_smi(tmp_path, used)
    e = {**os.environ, "PATH": f"{d}:{os.environ['PATH']}", "QUEUE_LIB_ONLY": "1", "QUEUE_LOG": str(tmp_path / "log"),
         "HOST_READ_S": "0", "HOST_RETRY_S": "0", "HOST_TRIES": "2", **env}
    r = subprocess.run(["bash", "-c", '. ./queue_r4_seeds.sh && launch_cap'], cwd=SCRIPTS, capture_output=True,
                       text=True, env=e, timeout=60)  # cwd: the script's `cd "$(dirname "$0")/.."` with $0 = bash
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_launch_cap_keeps_parity_and_is_only_lowered(tmp_path):
    assert _cap(tmp_path, 1500) == "5.9"  # (8188 - 1500) / 1024 - 0.5 = 6.03 -> the parity cap
    assert _cap(tmp_path, 1625) == "5.9"  # seed 0's launch reading (D115)
    assert _cap(tmp_path, 1640) == "5.8"
    assert _cap(tmp_path, 2300) == "5.2"  # 5.25, floored
    log = (tmp_path / "log" / "chain.log").read_text()
    assert "re-reading" in log and "host VRAM readings (MiB): 2300" in log


def test_decimal_temperatures_are_compared_as_numbers(tmp_path):
    """D129: the driver reports temperature.gpu as 55.70; keeping only its digits read 5570 C and paused forever."""
    e = {**os.environ, "QUEUE_LIB_ONLY": "1", "QUEUE_LOG": str(tmp_path / "log")}

    def hot(t: str) -> bool:
        return subprocess.run(["bash", "-c", f'. ./queue_r4_seeds.sh && hot "{t}"'], cwd=SCRIPTS, env=e,
                              timeout=60).returncode == 0

    assert [hot(t) for t in ("55.70", "87", "87.00", "87.5", "88", "")] == [False, False, False, True, True, False]
    d = _fake_smi(tmp_path, 1500, temp="55.70")
    r = subprocess.run(["bash", "-c", ". ./queue_r4_seeds.sh && cool"], cwd=SCRIPTS, capture_output=True, text=True,
                       env={**e, "PATH": f"{d}:{os.environ['PATH']}"}, timeout=60)
    assert r.returncode == 0 and "GPU at 55.70C, memory.used 1500 MiB" in r.stdout


def test_temp_guard_pauses_only_above_87_c_with_decimal_readings(tmp_path):
    """D129: the in-stage guard (scripts/gpu_temp_guard.sh) on the driver's decimal readings."""
    def guard_log(reading: str) -> str:
        log = tmp_path / f"guard_{reading}.log"
        stage = subprocess.Popen(["sleep", "3"], start_new_session=True)  # leads its own process group, like timeout
        guard = subprocess.Popen([str(SCRIPTS / "gpu_temp_guard.sh"), str(stage.pid), str(log)],
                                 env={**os.environ, "TEMP_CMD": f"echo {reading}", "POLL_S": "0.5", "PAUSE_S": "1"})
        stage.wait(timeout=60)
        guard.wait(timeout=60)
        return log.read_text() if log.exists() else ""

    assert guard_log("55.70") == ""
    assert "GPU at 88.10C > 87C: stage stopped" in guard_log("88.10")


def _dry(tmp_path, **env):
    d = _fake_smi(tmp_path, 1500)
    e = {**os.environ, "PATH": f"{d}:{os.environ['PATH']}", "DRY_RUN": "1", "QUEUE_LOG": str(tmp_path / "q"),
         "DISK_GUARD": "off", "GPU_DEADLINE": "09:00", "QUEUE_OUT": str(tmp_path / "out"), **env}
    r = subprocess.run(["bash", str(SCRIPTS / "queue_r4_seeds.sh")], capture_output=True, text=True, env=e, timeout=600)
    assert r.returncode == 0, r.stderr
    return [ln.split("DRY ", 1)[1] for ln in (tmp_path / "q" / "chain.log").read_text().splitlines()
            if "] DRY " in ln and "commit" not in ln and "disk" not in ln]


def test_dry_run_order_and_commands(tmp_path):
    dry = _dry(tmp_path)
    names = [ln.split(" ", 1)[0] for ln in dry]
    assert names == ["train_dpos2_s1", "train_dpos2_s2", "train_b2_s1", "train_b2_s2", "eval_dpos2_s0", "eval_b2_s0",
                     "eval_dpos2_s1", "eval_b2_s1", "eval_dpos2_s2", "eval_b2_s2", "r4_confirm"]
    assert "nice -n 19 env PHYSCAL_CONFIRMATORY=1 uv run python -m physcal.eval.r4_confirm analyze" in dry[-1]
    dry = dry[:-1]
    for ln in dry[:4]:
        assert "physcal.train.stage_r4 train" in ln and "--max-reserved-gib 5.9" in ln and "--stop-at" in ln
    for ln in dry[4:]:
        assert ln.split(": ", 1)[1].startswith("env PHYSCAL_CONFIRMATORY=1 uv run python -m physcal.train.stage_r4 eval")
        assert "--confirmatory" in ln and "--max-reserved-gib 5.9" in ln
    assert "launch cap 5.9 GiB" in (tmp_path / "q" / "chain.log").read_text()


def test_second_copy_exits_0_while_the_lock_is_held(tmp_path):
    """D128: the waiter and the Windows scheduled task may both start the queue; only one chain runs."""
    import fcntl

    log = tmp_path / "q"
    log.mkdir()
    with open(log / "queue.lock", "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)  # flock(2), as util-linux flock: a running chain
        d = _fake_smi(tmp_path, 1500)
        e = {**os.environ, "PATH": f"{d}:{os.environ['PATH']}", "DRY_RUN": "1", "QUEUE_LOG": str(log),
             "DISK_GUARD": "off", "QUEUE_OUT": str(tmp_path / "out")}
        r = subprocess.run(["bash", str(SCRIPTS / "queue_r4_seeds.sh")], capture_output=True, text=True, env=e,
                           timeout=600)
        assert r.returncode == 0, r.stderr
        chain = (log / "chain.log").read_text()
        assert "already running" in chain and "chain start" not in chain
        assert not (log / "queue.pid").exists()
    lines = _dry(tmp_path)  # the lock released: the same log dir runs the chain
    assert lines[0].split(" ", 1)[0] == "train_dpos2_s1"


def test_mc_b_analysis_waits_for_a_finished_generation(tmp_path):
    """D127: a generation stage stopped by its timeout leaves a partial generations.jsonl; it is not analyzed."""
    d = tmp_path / "a6c_pairs"
    d.mkdir()

    def generated(n_lines: int, finished: bool) -> int:
        (d / "generations.jsonl").write_text("{}\n" * n_lines)
        (d / "meta.json").write_text(json.dumps({"n_problems": 3, **({"finished": "x"} if finished else {})}))
        e = {**os.environ, "QUEUE_LIB_ONLY": "1", "QUEUE_LOG": str(tmp_path / "log")}
        return subprocess.run(["bash", "-c", f'. ./queue_r4_seeds.sh && mcb_generated "{d}"'], cwd=SCRIPTS,
                              capture_output=True, text=True, env=e, timeout=120).returncode

    assert generated(3, True) == 0
    assert generated(2, False) != 0  # stopped mid-set: meta.json was written at the start, without "finished"
    assert generated(2, True) != 0


def test_dry_run_with_mc_b_adds_its_stages_after_the_evals(tmp_path):
    names = [ln.split(" ", 1)[0] for ln in _dry(tmp_path, MC_B="1")]
    assert names[10:] == ["mc_b_gen_a6c_pairs", "mc_b_gen_a6_pairs", "mc_b_gen_a6b_pairs", "r4_confirm"]
    log = (tmp_path / "q" / "chain.log").read_text()
    assert log.count("BLOCKED mc_b analysis") == 3  # a dry run generates nothing, so execution is skipped and logged
    assert "physcal.eval.mc_b generate --set a6c_pairs --max-reserved-gib 5.9" in log


# ------------------------------------------------------------------------------------------- the waiter
def test_run_at_past_start_execs_at_once_under_its_own_pid(tmp_path):
    pid_file, out = tmp_path / "w.pid", tmp_path / "out"
    past = time.strftime("%H:%M", time.gmtime(time.time() + 5.5 * 3600 - 120))  # two minutes ago, IST
    e = {**os.environ, "START": past, "PID_FILE": str(pid_file), "POLL_S": "1"}
    r = subprocess.run(["bash", str(SCRIPTS / "run_at.sh"), "bash", "-c", f'echo $$ > "{out}"'], env=e,
                       capture_output=True, text=True, timeout=30)
    if time.strftime("%H:%M", time.gmtime(time.time() + 5.5 * 3600)) < past:  # the test straddled IST midnight
        pytest.skip("IST midnight")
    assert r.returncode == 0 and "starting" in r.stdout
    assert pid_file.read_text().strip() == out.read_text().strip()  # exec: the waiter's PID is the command's


def test_run_at_future_start_waits_and_can_be_cancelled(tmp_path):
    pid_file, out = tmp_path / "w.pid", tmp_path / "ran"
    later = time.strftime("%H:%M", time.gmtime(time.time() + 5.5 * 3600 + 3600))  # an hour from now, IST
    if later < time.strftime("%H:%M", time.gmtime(time.time() + 5.5 * 3600)):
        pytest.skip("IST midnight")
    e = {**os.environ, "START": later, "PID_FILE": str(pid_file), "POLL_S": "1"}
    p = subprocess.Popen(["bash", str(SCRIPTS / "run_at.sh"), "touch", str(out)], env=e, stdout=subprocess.PIPE, text=True)
    for _ in range(50):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        time.sleep(0.1)
    assert int(pid_file.read_text()) == p.pid
    time.sleep(1.5)
    assert p.poll() is None  # still waiting
    os.kill(p.pid, 15)
    p.wait(timeout=10)
    assert not out.exists() and "armed for" in p.stdout.read()


def test_queue_logs_are_ignored_by_nothing_and_the_scripts_are_executable():
    for name in ("queue_r4_seeds.sh", "run_at.sh"):
        assert os.access(SCRIPTS / name, os.X_OK)
    r = subprocess.run(["git", "check-ignore", "-q", "results/queue_r4_seeds/chain.log"], cwd=REPO_ROOT)
    assert r.returncode == 1  # the chain log is committed after every stage


# ------------------------------------------------------------------------------------------- the analysis (D118)
DEV_EVAL = {arm: REPO_ROOT / "results" / "stage_r4" / arm / "seed0" / "eval" for arm in ("dpos2", "b2")}


def test_the_analysis_refuses_without_the_guard_and_before_all_six_evals(monkeypatch, tmp_path):
    from physcal.eval import r4_confirm
    from physcal.eval.run_eval import CONFIRMATORY_ENV

    monkeypatch.delenv(CONFIRMATORY_ENV, raising=False)
    with pytest.raises(PermissionError):
        r4_confirm.analyze(root=tmp_path)
    monkeypatch.setenv(CONFIRMATORY_ENV, "1")
    with pytest.raises(RuntimeError, match="all six evals"):
        r4_confirm.analyze(root=tmp_path)


def test_confirmatory_stage_a_config_points_at_the_frozen_files():
    from physcal.eval import r4_confirm

    sa = r4_confirm.stage_a_config()
    assert sa["generator_version"] == "2.0.0+bdbb2279/confirmatory"
    assert sa["alternatives"]["x1_predictions"] == "results/x1_text_sigma/confirmatory/predictions.jsonl"
    assert load_config("stage_a")["generator_version"] == "2.0.0+bdbb2279"  # the development config is untouched


@pytest.mark.skipif(not all((d / "rollouts.jsonl").exists() or (d / "rollouts.jsonl.gz").exists()
                            for d in DEV_EVAL.values()), reason="seed 0's development evals not on disk")
def test_three_identical_seeds_reproduce_the_one_seed_development_analysis(monkeypatch, tmp_path):
    """The pipeline on the development sets with seed 0's tapes copied into three seeds: every seed-mean point
    estimate equals D113's one-seed value (R1-R8, results/stage_r4/results.json), and the seed SD is 0."""
    import shutil

    from physcal.eval import r4_confirm
    from physcal.eval.run_eval import CONFIRMATORY_ENV

    monkeypatch.setenv(CONFIRMATORY_ENV, "1")
    cfg = copy.deepcopy(load_config("stage_r4"))
    cc = cfg["confirmatory"]
    cc.update(set_subdir=".", x1_predictions="results/x1_text_sigma/a6c/predictions.jsonl",
              x1_reproduction="results/x1_text_sigma/a6c/reproduction.json",
              analysis_output=str(tmp_path / "out"))
    cfg["analysis"]["bootstrap"] = 200
    for arm, d in DEV_EVAL.items():
        for s in cc["seeds"]:
            dst = tmp_path / "root" / arm / f"seed{s}" / cc["output_subdir"]
            dst.mkdir(parents=True)
            src = d / "rollouts.jsonl"
            shutil.copy(src if src.exists() else d / "rollouts.jsonl.gz", dst)
    res = r4_confirm.analyze(cfg, root=tmp_path / "root")
    dev = json.loads((REPO_ROOT / "results" / "stage_r4" / "results.json").read_text())["tests"]
    for k, r in zip(r4_confirm.TESTS, [f"R{i}" for i in range(1, 9)]):
        assert res["main"]["tests"][k]["value"] == pytest.approx(dev[r]["value"], abs=1e-12), k
    for k in ("K1", "K2", "K3", "K4"):
        assert res["main"]["tests"][k]["seed_sd"] == pytest.approx(0.0, abs=1e-12)
    assert res["complete"] and res["seeds"] == {"dpos2": [0, 1, 2], "b2": [0, 1, 2]}
    c3 = res["main"]["criteria"]["dpos2_a"]["C3"]
    assert res["primary"]["criteria"] == ["K1", "C2", "C3"]  # D125
    assert res["primary"]["confirmed"] == (res["main"]["tests"]["K1"]["holm"]["rejected"] and dev["R1"]["value"] >= 0.5
                                           and c3)
    res_no_c3 = {**res["primary"], "C3": False}
    assert not all(res_no_c3[c] for c in res["primary"]["criteria"])  # invariance is required
    n = {name: s["n_problems"] for name, s in res["subsets"].items()}
    assert n["depths_1_4"] + n["depth_5"] == res["n_problems"] == 3060
    assert set(res["by_depth"]) == {"1", "2", "3", "4", "5"}
    assert "error" not in res["stage_b_analysis"]["dpos2"]
    assert (tmp_path / "out" / "summary.md").read_text().startswith("# R4-3 confirmatory run")

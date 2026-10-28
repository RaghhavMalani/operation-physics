"""The conditional addition to the XM chain (DECISIONS.md D139): the early go rule on D-pos-v2 (xm_go), the seed
override (stage_xm_seed) and the follow-up queue's dry run. CPU only."""

import json
import os
import subprocess

import pytest

from physcal.config import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "queue_xm_followup.sh"


def test_the_early_go_rule_reproduces_r4_confirms_seed_0_values_on_a_stand_in(monkeypatch):
    from physcal.eval import xm_go, xm_pilot as P

    monkeypatch.setenv("PHYSCAL_CONFIRMATORY", "1")
    cfg = P._config()
    g = xm_go.go_dpos2(P.models(cfg)["qwen_seed0"], cfg)
    crit = json.loads((REPO_ROOT / "results/stage_r4/confirmatory/results.json").read_text())["main"]["criteria"]["dpos2_a"]
    assert g["evaluable"] and g["n_problems"] == 2040
    assert g["a6c_slope"] == pytest.approx(crit["a6c"]["per_seed"]["0"]["value"], rel=1e-12)
    assert g["a6b_slope"] == pytest.approx(crit["a6b"]["per_seed"]["0"]["value"], rel=1e-12)
    assert g["a6c_n_pairs"] == crit["a6c"]["per_seed"]["0"]["n_pairs"]
    assert set(g) >= {"K1", "C1", "C2", "C3", "go"}


def test_an_unfinished_eval_is_not_evaluable(tmp_path):
    from physcal.eval import xm_go

    assert "no metrics.json" in xm_go.complete_eval(tmp_path, 2040)
    (tmp_path / "metrics.json").write_text("{}")
    (tmp_path / "rollouts.jsonl").write_text(json.dumps({"split": "a6c_pairs"}) + "\n")
    assert "1 records" in xm_go.complete_eval(tmp_path, 2040)


def test_the_seed_override_changes_only_the_seed_and_is_undone():
    from physcal.train import stage_xm, stage_xm_seed

    base = stage_xm._config()
    with stage_xm_seed.seed(1):
        assert stage_xm.arm_dir("dpos2").parts[-2:] == ("dpos2", "seed1")
        assert stage_xm.r4_config()["seed"] == 1
        assert stage_xm.eval_config("dpos2")["run_id"] == "stage-xm-dpos2-seed1-confirmatory"
        assert {k for k in base if stage_xm._config()[k] != base[k]} == {"seed"}
    assert stage_xm._config() == base and stage_xm.arm_dir("dpos2").name == "seed0"


def _followup(tmp_path, go):
    q, out = tmp_path / "q", tmp_path / "out"
    q.mkdir()
    (q / "chain.log").write_text("[01:00:00] start xm_eval_dpos2\n")
    if go is not None:
        (out / "go_dpos2").mkdir(parents=True)
        (out / "go_dpos2" / "go.json").write_text(json.dumps({"go": go}))
    dead = subprocess.Popen(["true"])
    dead.wait()  # an exited PID: the chain is over
    e = {**os.environ, "DRY_RUN": "1", "DISK_GUARD": "off", "CHAIN_LOG_DIR": str(q), "XM_OUT": str(out),
         "CHAIN_PID": str(dead.pid), "POLL_S": "0.1", "LOCK_WAIT_S": "5", "GPU_DEADLINE": "10:00"}
    r = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=e, timeout=300)
    assert r.returncode == 0, r.stderr
    log = (q / "followup" / "chain.log").read_text()
    return [ln.split("DRY ", 1)[1].split(" ", 1)[0] for ln in log.splitlines()
            if "] DRY " in ln and "commit" not in ln], log


def test_the_follow_up_adds_seed_1_only_on_go(tmp_path):
    names, log = _followup(tmp_path, True)
    assert names == ["xm_train_dpos2_seed1"] and "SKIP xm_go" in log
    assert "physcal.train.stage_xm_seed train --arm dpos2 --seed 1" in log and "--stop-at" in log


@pytest.mark.parametrize("go", [False, None])
def test_the_follow_up_adds_nothing_on_no_go_or_without_a_rule(tmp_path, go):
    names, log = _followup(tmp_path, go)
    assert names == ([] if go is False else ["xm_go"])
    assert "no seed 1 stage" in log


def test_the_follow_up_is_executable():
    assert os.access(SCRIPT, os.X_OK)

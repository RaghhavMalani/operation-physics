"""E1c / E1d, the R3-3 optimization pilots (D94): each config changes one variable from E1a; the trainer arguments
carry it (and leave E1a / E1b as they were); the ρ test, the collapse check and the decision rule."""

import copy
import json
import shutil

import numpy as np
import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.train import e1, e1_compare as C

E1A_EVAL = REPO_ROOT / "results" / "train" / "e1a-sigma-only-free-seed0" / "e1_eval" / "final_4bit"


@pytest.mark.parametrize("name, key, value", [("grpo_e1c", "scale_rewards", "batch"), ("grpo_e1d", "learning_rate", 5e-5)])
def test_each_pilot_changes_one_variable_from_e1a(name, key, value):
    a, b = load_config("grpo_e1a"), load_config(name)
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {"run_id", "grpo"} and b["run_id"] != a["run_id"]
    assert {k for k in set(a["grpo"]) | set(b["grpo"]) if a["grpo"].get(k) != b["grpo"].get(k)} == {key}
    assert b["grpo"][key] == value


def test_grpo_args_pass_the_pilot_setting_and_keep_e1a(tmp_path):
    from trl import GRPOConfig

    fields = set(GRPOConfig.__dataclass_fields__)
    built = {}
    for name in ("grpo_e1a", "grpo_e1b", "grpo_e1c", "grpo_e1d"):
        cfg = load_config(name)
        args = e1.grpo_args(cfg, tmp_path, cfg["run_id"], 300)
        dropped = sorted(k for k in args if k not in fields)
        assert dropped == json.loads((REPO_ROOT / "results" / "train" / "e1a-sigma-only-free-seed0" / "summary.json")
                                     .read_text())["grpo_config_dropped_keys"]  # as E1a's run; scale_rewards is kept
        built[name] = GRPOConfig(**{k: v for k, v in args.items() if v is not None and k in fields}
                                 | {"bf16": False, "use_cpu": True})  # as e1.train builds it, on CPU
    assert e1.grpo_args(load_config("grpo_e1a"), tmp_path, "x", 300)["scale_rewards"] is None  # E1a: TRL's default
    assert built["grpo_e1a"].scale_rewards == built["grpo_e1b"].scale_rewards == "group"
    assert built["grpo_e1c"].scale_rewards == "batch" and built["grpo_e1c"].learning_rate == 5e-6
    assert built["grpo_e1d"].scale_rewards == "group" and built["grpo_e1d"].learning_rate == 5e-5
    assert {b.loss_type for b in built.values()} == {"dapo"}  # Unsloth overrides scale_rewards only for dr_grpo


def _arm(rho_sign, n=200, seed=0):
    rng = np.random.default_rng(seed)
    err = rng.lognormal(size=n)
    sig = err * np.exp(rng.normal(scale=0.3, size=n)) if rho_sign > 0 else rng.lognormal(size=n)
    keys = [("energy", 1, i) for i in range(n)]
    return {k: {"fail": False, "rel_sigma": float(s), "abs_log_error": float(e)} for k, s, e in zip(keys, sig, err)}, keys


def test_rho_bootstrap_is_paired_over_problems():
    good, keys = _arm(+1)
    flat, _ = _arm(0)
    r = C.rho_bootstrap(good, flat, keys, 300, 0)
    assert r["rho"] > 0.8 and abs(r["rho_reference"]) < 0.2 and r["p_one_sided"] == 0.0
    assert C.rho_bootstrap(flat, flat, keys, 100, 0)["p_one_sided"] == 1.0  # the same arm never moves ρ
    assert C.rho_of({keys[0]: {"fail": True, "rel_sigma": None, "abs_log_error": None}}, keys[:1]) is None


def test_collapse_reads_the_eval_suffixes(tmp_path):
    t = tmp_path / "rollouts.jsonl"
    t.write_text("".join(json.dumps({"suffix": s}) + "\n" for s in [" 0 m"] * 19 + ["0.1 m"]))
    from physcal.tape import tape as T

    orig = T.read_tape
    C.read_tape = lambda p: [json.loads(l) for l in open(p)]
    try:
        c = C.collapse(t, 0.9)
    finally:
        C.read_tape = orig
    assert c == {"top_suffix": "0 m", "top_share": 0.95, "collapsed": True}


@pytest.mark.skipif(not (E1A_EVAL / "rows.jsonl").exists(), reason="E1a's corrected eval rows missing")
def test_compare_pilots_decision_rule(tmp_path, monkeypatch):
    e = copy.deepcopy(load_config("e1"))
    e["compare"]["bootstrap"] = 100
    pc = e["compare"]["pilots"]
    pc["output_dir"] = str(tmp_path / "out")
    pc["runs"] = {"E1c": str(tmp_path / "e1c"), "E1d": str(tmp_path / "e1d")}
    monkeypatch.setattr(C, "load_config", lambda name: e if name == "e1" else load_config(name))
    out = C.compare_pilots()
    assert out["tests"] == {"E1c": "not run", "E1d": "not run"} and out["decision_rule"].startswith("neither")
    d = tmp_path / "e1c" / "e1_eval" / "final_4bit"
    d.mkdir(parents=True)
    shutil.copy(E1A_EVAL / "rows.jsonl", d / "rows.jsonl")  # a pilot identical to E1a does not move ρ
    shutil.copy(E1A_EVAL / "rollouts.jsonl", d / "rollouts.jsonl")
    out = C.compare_pilots()
    t = out["tests"]["E1c"]
    assert t["diff"] == 0.0 and t["p_one_sided"] == 1.0 and not out["holm"]["E1c"]["rejected"]
    assert out["describe"]["E1c"]["spearman_sigma_error"] == pytest.approx(0.012674069264585797)  # E1a's, D67
    assert out["decision_rule"].startswith("neither") and not out["collapse"]["E1c"]["collapsed"]
    assert "Decision rule" in (tmp_path / "out" / "pilots.md").read_text()

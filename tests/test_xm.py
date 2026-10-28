"""The cross-model pilot (DECISIONS.md D137): the stage's config and tokenization, the go rule, the analysis on a
stand-in (Qwen seed 0's frozen logs reproduce r4_confirm's per-seed values), and the queue's dry run. CPU only."""

import json
import os
import subprocess

import pytest

from physcal.config import REPO_ROOT, load_config

SCRIPTS = REPO_ROOT / "scripts"


def _flat(d: dict, p: str = "") -> dict:
    out = {}
    for k, v in d.items():
        out.update(_flat(v, p + k + ".") if isinstance(v, dict) and k != "splits" else {p + k: v})
    return out


def test_the_llama_eval_differs_from_stage_r4s_confirmatory_eval_only_where_d137_says():
    from physcal.train import stage_r4, stage_xm

    q, x = _flat(stage_r4.eval_config("dpos2", seed=0, confirmatory=True)), _flat(stage_xm.eval_config("dpos2"))
    diff = {k for k in set(q) | set(x) if q.get(k) != x.get(k)}
    assert diff == {"generation.batch_size", "generation.last_logits_only", "model.chat_template_kwargs.date_string",
                    "model.id", "model.lora", "model.pinned_base", "model.prompt_add_special_tokens", "output_dir",
                    "run_id", "splits"}
    assert set(stage_xm.eval_config("b2")["splits"]) == {"a6c_pairs", "a6b_pairs"}
    assert all("/confirmatory/" in s["file"] for s in stage_xm.eval_config("b2")["splits"].values())
    r4, xm = load_config("stage_r4"), stage_xm.r4_config()
    assert all(xm[k] == r4[k] for k in ("arms", "lora", "train", "guards", "sft_phase12_config"))


def test_llama_rows_have_one_bos_and_close_with_eot_and_qwens_default_is_unchanged():
    from transformers import AutoTokenizer

    from physcal.base import base_config, pinned_base_dir
    from physcal.train import sft_phase12, stage_b, stage_r4, stage_xm

    acfg = stage_r4.arm_config("dpos2", stage_xm.r4_config(), 0)
    pc = load_config(acfg["sft_phase12_config"])
    arm_file, prob_file, _ = stage_b.check_data(acfg)
    row = json.loads(arm_file.open().readline())
    text = {json.loads(line)["seed"]: json.loads(line)["text"] for line in prob_file.open()}[row["seed"]]
    pin = base_config("base_llama32_3b_4bit")
    tok = AutoTokenizer.from_pretrained(pinned_base_dir(pin))
    t = sft_phase12.tokenize(row, text, tok, pc, pin["model"]["chat_template_kwargs"], pin["model"]["end_of_turn"])
    assert t["input_ids"].count(tok.bos_token_id) == 1 and t["input_ids"][-1] == tok.convert_tokens_to_ids("<|eot_id|>")
    assert t["labels"][-1] == t["input_ids"][-1]
    assert "26 Jul 2024" in tok.decode(t["input_ids"])  # the template's date is fixed
    qtok = AutoTokenizer.from_pretrained(pinned_base_dir(base_config("base_4bit")))
    assert sft_phase12.tokenize(row, text, qtok, pc) == sft_phase12.tokenize(row, text, qtok, pc, None, "<|im_end|>")


def test_the_go_rule_needs_k1_c2_and_c3():
    from physcal.eval import xm_pilot as P

    ac = {"alpha": 0.05, "practical_slope": 0.5}
    c = {"slope": 0.7, "ci": [0.4, 1.0], "p_one_sided": 0.001, "n_pairs": 500}
    b = {"slope": 0.0, "ci": [-0.1, 0.2], "n_pairs": 20}
    assert P.go_rule(c, b, ac, 0.3)["go"]
    assert not P.go_rule({**c, "slope": 0.45, "ci": [0.1, 0.8]}, b, ac, 0.3)["go"]  # C2
    assert not P.go_rule(c, {**b, "ci": [-0.4, 0.2]}, ac, 0.3)["go"]  # C3
    assert not P.go_rule(c, {"slope": None, "ci": None, "n_pairs": 2}, ac, 0.3)["go"]  # C3 not evaluable: not met
    assert not P.go_rule({**c, "p_one_sided": 0.08, "ci": [-0.05, 1.0]}, b, ac, 0.3)["go"]  # K1


def test_the_pilot_analysis_reproduces_r4_confirms_seed_0_values_on_a_stand_in(monkeypatch):
    from physcal.eval import xm_pilot as P

    monkeypatch.setenv("PHYSCAL_CONFIRMATORY", "1")
    cfg = P._config()
    ctx = P.context(cfg)
    m = P.analyze_model("qwen_seed0", P.models(cfg)["qwen_seed0"], ctx)
    crit = json.loads((REPO_ROOT / "results/stage_r4/confirmatory/results.json").read_text())["main"]["criteria"]["dpos2_a"]
    a = m["arms"]["dpos2"]["sources"]["a"]
    assert a["a6c_slope"] == pytest.approx(crit["a6c"]["per_seed"]["0"]["value"], rel=1e-12)
    assert a["a6b_slope"] == pytest.approx(crit["a6b"]["per_seed"]["0"]["value"], rel=1e-12)
    assert a["a6c_n_pairs"] == crit["a6c"]["per_seed"]["0"]["n_pairs"]
    assert set(m["arms"]["dpos2"]["sources"]) == {"a", "b", "c", "c_S"} and set(m["arms"]["b2"]["sources"]) == {"a"}


def _dry(tmp_path, **env):
    e = {**os.environ, "DRY_RUN": "1", "DISK_GUARD": "off", "QUEUE_LOG": str(tmp_path / "q"),
         "XM_OUT": str(tmp_path / "out"), "MCB_OUT": str(tmp_path / "mcb"), "GPU_DEADLINE": "10:00", **env}
    r = subprocess.run(["bash", str(SCRIPTS / "queue_xm.sh")], capture_output=True, text=True, env=e, timeout=600)
    assert r.returncode == 0, r.stderr
    log = (tmp_path / "q" / "chain.log").read_text()
    return [ln.split("DRY ", 1)[1].split(" ", 1)[0] for ln in log.splitlines()
            if "] DRY " in ln and "commit" not in ln], log


def test_the_xm_queue_runs_d137s_order(tmp_path):
    names, log = _dry(tmp_path)
    assert names == ["xm_train_dpos2", "xm_eval_dpos2", "xm_train_b2", "xm_eval_b2", "mc_b_gen_a6b_pairs", "xm_pilot"]
    assert "physcal.train.stage_xm eval --arm dpos2" in log and "PHYSCAL_CONFIRMATORY=1" in log
    assert "--stop-at" in log and "--max-reserved-gib 5.9" in log


def test_the_xm_queue_skips_what_is_done(tmp_path):
    for arm in ("dpos2", "b2"):
        d = tmp_path / "out" / arm / "seed0"
        (d / "final_adapter").mkdir(parents=True)
        (d / "confirmatory").mkdir()
        (d / "confirmatory" / "rollouts.jsonl").write_text("{}\n")
    (tmp_path / "out" / "pilot").mkdir()
    (tmp_path / "out" / "pilot" / "results.json").write_text("{}")
    m = tmp_path / "mcb" / "a6b_pairs"
    m.mkdir(parents=True)
    (m / "generations.jsonl").write_text("{}\n")
    (m / "meta.json").write_text(json.dumps({"n_problems": 1, "finished": "x"}))
    (m / "results.json").write_text("{}")
    names, log = _dry(tmp_path)
    assert names == []
    for s in ("SKIP train xm dpos2", "SKIP eval xm b2: already evaluated", "SKIP mc_b generate a6b_pairs",
              "SKIP mc_b analysis a6b_pairs", "SKIP xm_pilot"):
        assert s in log


def test_the_xm_queue_is_executable():
    assert os.access(SCRIPTS / "queue_xm.sh", os.X_OK)

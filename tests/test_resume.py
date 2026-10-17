"""D66: mid-run resume of curriculum GRPO runs. The curriculum (EMA sampler + its RNG), the repeat sampler's drawn
chunks and the σ = 0 monitor are checkpointed with every adapter checkpoint; resuming restores them, cuts the per-step
logs back, and continues exactly as an uninterrupted run would. CPU only."""

import gzip
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict

import numpy as np
import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.env.sampler import EMASampler, UniformSampler
from physcal.tape.tape import replay
from physcal.train.grpo import (CHECKPOINT_FILES, RESUME_STATE, CurriculumRepeatSampler, SigmaZeroMonitor,
                                curriculum_pool, find_resume_checkpoint, load_resume, truncate_step_logs)

CELLS = [(f, d) for f in ("dynamics", "energy", "kinematics") for d in (1, 2, 3, 4)]
CFG = {"alpha": 0.3, "weight_floor": 0.02, "prior_pass_rate": 0.5, "start_depths": [1, 2],
       "unlock_pass_rate": 0.5, "min_observations": 2}
SEED0 = REPO_ROOT / "results/train/split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed0"
TOKENIZER_DIR = REPO_ROOT / "results/sft/qwen2.5-math-1.5b-instruct/merged"
CELL = lambda key: (key.rsplit("/", 1)[0], int(key.rsplit("/", 1)[1]))


def _results(rng, cells):
    return [(c, list(rng.random(8) < 0.4)) for c in cells]


@pytest.mark.parametrize("make", [lambda: EMASampler(CELLS, CFG, seed=3), lambda: UniformSampler(CELLS, seed=3)])
def test_sampler_checkpoint_round_trips_through_json_and_continues_identically(make):
    a, rng = make(), np.random.default_rng(0)
    for _ in range(20):
        a.observe(_results(rng, a.next_batch(4)))
    b = make()
    b.restore(json.loads(json.dumps(a.checkpoint())))
    assert b.checkpoint() == a.checkpoint() and b.state() == a.state()
    for _ in range(30):
        cells = a.next_batch(4)
        assert b.next_batch(4) == cells
        res = _results(rng, cells)
        a.observe(res)
        b.observe(res)
    assert b.checkpoint() == a.checkpoint()


def test_restore_refuses_another_samplers_checkpoint():
    with pytest.raises(ValueError):
        EMASampler(CELLS[:6], CFG, seed=0).restore(EMASampler(CELLS, CFG, seed=0).checkpoint())
    with pytest.raises(ValueError):
        EMASampler(CELLS, CFG, seed=0).restore(UniformSampler(CELLS, seed=0).checkpoint())


def _seed0_logs():
    if not ((SEED0 / "curriculum.jsonl").exists() and (SEED0 / "rollouts.jsonl.gz").exists()):
        pytest.skip("split-credit seed-0 logs not available")
    cur = [json.loads(x) for x in (SEED0 / "curriculum.jsonl").read_text().splitlines()]
    outcomes = defaultdict(list)
    with gzip.open(SEED0 / "rollouts.jsonl.gz", "rt") as fh:
        for line in fh:
            r = json.loads(line)
            outcomes[r["step"]].append(r["by_rule"]["binary"]["score_final"] == 1.0)
    return cur, outcomes


def test_logged_curriculum_is_reproduced_with_draws_one_step_ahead_and_survives_a_restore():
    """The real split-credit seed-0 run (287 steps): an EMA sampler fed the tape's group outcomes reproduces every
    logged chunk iff chunk k + 1 is drawn before step k is observed (the trainer's prefetch). A sampler checkpointed
    after step 150 (chunk 151 drawn), saved as JSON and restored, continues to reproduce the log to step 287."""
    cur, outcomes = _seed0_logs()
    cfg = load_config("grpo_split_credit")
    cells = sorted(CELL(c) for c in cur[0]["p"])
    g, b = cfg["grpo"]["num_generations"], len(cur[0]["cells"])
    observe = lambda s, row: s.observe([(CELL(c), outcomes[row["step"]][i * g:(i + 1) * g])
                                        for i, c in enumerate(row["cells"])])
    draw = lambda s: [f"{f}/{d}" for f, d in s.next_batch(b)]

    s = EMASampler(cells, cfg["curriculum"], cfg["seed"])
    drawn, saved = [draw(s)], None
    for k, row in enumerate(cur):
        drawn.append(draw(s))  # chunk k + 2 is drawn before step k + 1 is observed
        assert drawn[k] == row["cells"], f"step {row['step']}"
        observe(s, row)
        assert s.state()["p"] == row["p"]
        if row["step"] == 150:
            saved = json.dumps(s.checkpoint())
    restored = EMASampler(cells, cfg["curriculum"], cfg["seed"])
    restored.restore(json.loads(saved))
    pending = drawn[150]  # chunk 151, drawn before the checkpoint
    for row in cur[150:]:
        assert pending == row["cells"], f"after restore, step {row['step']}"
        pending = draw(restored)
        observe(restored, row)
    assert restored.checkpoint() == s.checkpoint()


def test_repeat_sampler_restore_replays_the_drawn_chunks_first():
    _, cell_indices = curriculum_pool(3, 0, load_config("env"))
    mk = lambda cur: CurriculumRepeatSampler(cell_indices, cur, mini_repeat_count=2, batch_size=3, repeat_count=2,
                                             total_chunks=10)
    per_chunk = 3 * 2 * 2
    a = mk(EMASampler(list(cell_indices), CFG, seed=1))
    it = iter(a)
    stream = [next(it) for _ in range(per_chunk * 4 + 1)]  # 4 steps trained, chunk 5 already started (prefetch)
    assert len(a.chunks) == 5
    ckpt = json.loads(json.dumps({"rs": a.checkpoint(), "cur": a.curriculum.checkpoint()}))
    rest = list(it)
    b = mk(EMASampler(list(cell_indices), CFG, seed=1))
    b.curriculum.restore(ckpt["cur"])
    b.restore(ckpt["rs"], trained_steps=4)
    assert list(b) == stream[per_chunk * 4:] + rest
    assert b.checkpoint() == a.checkpoint() and len(b) == len(a)


def test_sigma_zero_monitor_checkpoint():
    a = SigmaZeroMonitor(threshold=0.5, window_steps=3)
    for n, z in [(10, 9), (10, 9), (10, 2)]:
        a.update(n, z)
    b = SigmaZeroMonitor(threshold=0.5, window_steps=3)
    b.restore(json.loads(json.dumps(a.checkpoint())))
    assert [b.update(10, 0) for _ in range(3)] == [a.update(10, 0) for _ in range(3)]


def _log(path, steps, per_step=1, partial=False):
    path.write_text("".join(json.dumps({"step": s, "i": i}) + "\n" for s in steps for i in range(per_step))
                    + ('{"step": 9, "i"' if partial else ""))


def test_truncate_step_logs_cuts_back_keeps_the_cut_and_checks_completeness(tmp_path):
    _log(tmp_path / "rollouts.jsonl", range(1, 7), per_step=4, partial=True)  # a half-written line from the crash
    _log(tmp_path / "train_metrics.jsonl", range(1, 7))
    rep = truncate_step_logs(tmp_path, 4, {"rollouts.jsonl": 4, "train_metrics.jsonl": 1})
    assert rep == {"rollouts.jsonl": {"kept": 16, "cut": 9}, "train_metrics.jsonl": {"kept": 4, "cut": 2}}
    assert [json.loads(x)["step"] for x in (tmp_path / "train_metrics.jsonl").read_text().splitlines()] == [1, 2, 3, 4]
    cut = sorted((tmp_path / "resume_discarded").iterdir())
    assert [p.name.split(".")[0] for p in cut] == ["rollouts", "train_metrics"]
    _log(tmp_path / "train_metrics.jsonl", [1, 2, 4, 5])  # step 3 missing
    with pytest.raises(ValueError, match="not all present"):
        truncate_step_logs(tmp_path, 4, {"train_metrics.jsonl": 1})
    _log(tmp_path / "train_metrics.jsonl", [1, 2, 3, 2])
    with pytest.raises(ValueError, match="backwards"):
        truncate_step_logs(tmp_path, 4, {"train_metrics.jsonl": 1})
    (tmp_path / "train_metrics.jsonl").write_text('{"step": 1}\nnot json\n{"step": 2}\n')
    with pytest.raises(ValueError, match="unreadable"):
        truncate_step_logs(tmp_path, 2, {"train_metrics.jsonl": 1})


def _checkpoint(d, step, with_state=True, max_steps=10, digest="x"):
    d.mkdir(parents=True)
    for f in CHECKPOINT_FILES:
        (d / f).write_text("{}")
    (d / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    if with_state:
        (d / RESUME_STATE).write_text(json.dumps({"global_step": step, "max_steps": max_steps, "meta_digest": digest}))
    else:
        (d / RESUME_STATE).unlink()


def test_find_resume_checkpoint_takes_the_newest_complete_one_and_refuses_pre_d66(tmp_path):
    _checkpoint(tmp_path / "checkpoints/checkpoint-50", 50)
    _checkpoint(tmp_path / "checkpoints/checkpoint-100", 100)
    _checkpoint(tmp_path / "checkpoints/checkpoint-150", 150, with_state=False)  # crashed before the state was written
    assert find_resume_checkpoint(tmp_path, "latest").name == "checkpoint-100"
    with pytest.raises(ValueError, match="not a complete"):
        find_resume_checkpoint(tmp_path, "checkpoints/checkpoint-150")
    shutil.rmtree(tmp_path / "checkpoints/checkpoint-50")
    shutil.rmtree(tmp_path / "checkpoints/checkpoint-100")
    with pytest.raises(ValueError, match="before D66"):
        find_resume_checkpoint(tmp_path, "latest")


def test_load_resume_refuses_a_changed_run(tmp_path):
    from physcal.train.grpo import meta_digest

    meta = {"train_config": {"seed": 0}, "model_id": "m"}
    (tmp_path / "rollouts.meta.json").write_text(json.dumps({"run_id": "r", **meta}))
    _checkpoint(tmp_path / "checkpoints/checkpoint-4", 4, digest=meta_digest(meta))
    ckpt = tmp_path / "checkpoints/checkpoint-4"
    assert load_resume(tmp_path, ckpt, 10, meta)["global_step"] == 4
    with pytest.raises(ValueError, match="max_steps"):
        load_resume(tmp_path, ckpt, 12, meta)
    with pytest.raises(ValueError, match="configs differ"):
        load_resume(tmp_path, ckpt, 10, {**meta, "model_id": "other"})


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        pytest.skip("local tokenizer not available")
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    d = tmp_path_factory.mktemp("tiny_qwen_resume")
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      tie_word_embeddings=True, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    Qwen2ForCausalLM(cfg).save_pretrained(str(d))
    tok.save_pretrained(str(d))
    return d


def _run(tiny_model, *args):
    subprocess.run([sys.executable, str(REPO_ROOT / "tests" / "_resume_cpu_run.py"), str(tiny_model), *map(str, args)],
                   cwd=REPO_ROOT, check=True, capture_output=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})


def test_resumed_cpu_run_equals_the_uninterrupted_run_and_a_naive_resume_does_not(tiny_model, tmp_path):
    """Real TRL GRPOTrainer + the trainer's own checkpoint resume (optimizer, scheduler, RNG, LoRA), wired by
    `setup_trainer`: 8 steps uninterrupted vs a crash after checkpoint-4 (logs already past step 4) resumed with
    D66, vs a naive resume (fresh curriculum, the trainer's data skipping: D63's failure)."""
    full, resumed, naive = tmp_path / "full", tmp_path / "resumed", tmp_path / "naive"
    _run(tiny_model, full, 8, "full")
    states = {k: json.loads((full / f"checkpoints/checkpoint-{k}" / RESUME_STATE).read_text()) for k in (2, 4, 6, 8)}
    assert {k: len(s["repeat_sampler"]["chunks"]) for k, s in states.items()} == {2: 3, 4: 5, 6: 7, 8: 9}  # prefetch
    shutil.copytree(full, resumed)
    for k in (6, 8):  # the "crash": steps 5-8 are in the logs, but only checkpoint-4 survived
        shutil.rmtree(resumed / f"checkpoints/checkpoint-{k}")
    _run(tiny_model, resumed, 8, "resume", "checkpoints/checkpoint-4")
    _run(tiny_model, naive, 8, "naive", full / "checkpoints/checkpoint-4")

    lines = lambda p: [json.loads(x) for x in p.read_text().splitlines()]
    for name in ("curriculum.jsonl", "train_metrics.jsonl", "rollouts.jsonl"):
        a, b = lines(full / name), lines(resumed / name)
        assert [r["step"] for r in a] == [r["step"] for r in b]
    assert lines(full / "curriculum.jsonl") == lines(resumed / "curriculum.jsonl")
    assert [(r["step"], r["seed"]) for r in lines(full / "rollouts.jsonl")] == \
           [(r["step"], r["seed"]) for r in lines(resumed / "rollouts.jsonl")]
    after = json.loads((resumed / "checkpoints/checkpoint-8" / RESUME_STATE).read_text())
    assert after == states[8]
    assert replay(resumed / "rollouts.jsonl").ok
    events = lines(resumed / "events.jsonl")
    assert events[-1]["step"] == 4 and events[-1]["event"].startswith("resumed from")
    assert len(list((resumed / "resume_discarded").iterdir())) == 3

    full_cur, naive_cur = lines(full / "curriculum.jsonl")[4:], lines(naive / "curriculum.jsonl")
    assert [r["step"] for r in naive_cur] == [5, 6, 7, 8]
    assert [r["cells"] for r in naive_cur] != [r["cells"] for r in full_cur]
    assert naive_cur[-1]["unlocked"] != full_cur[-1]["unlocked"]


def test_resume_smoke_config_differs_from_split_credit_only_in_length_and_checkpointing():
    a, b = load_config("grpo_split_credit"), load_config("grpo_resume_smoke")
    assert {k for k in a if a[k] != b.get(k)} == {"run_id", "grpo"} and set(a) == set(b)
    assert {k for k in a["grpo"] if a["grpo"][k] != b["grpo"][k]} == {"max_steps", "save_steps", "permanent_save_steps"}
    assert b["grpo"]["max_steps"] == 4 and b["grpo"]["save_steps"] == 2

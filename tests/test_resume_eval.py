"""D142: resumable greedy evals. Every finished batch goes to `completions.partial.jsonl`; a restarted eval generates
only the missing batches, with the same partition, so its completions equal a single run's. CPU only, fake models."""

import copy
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from physcal.config import REPO_ROOT, load_config
from physcal.eval.run_eval import (PARTIAL_NAME, Generation, HFBackend, PartialCompletions, build_split,
                                   generate_batched, run)
from physcal.env.generator import env_config

K = 1


def _batches(n: int, per: int) -> list[list[int]]:
    order = list(range(n))[::-1]
    return [order[s:s + per] for s in range(0, len(order), per)]


class Gen:
    """Deterministic fake generation whose text depends on the batch's composition (as left padding can), and that
    can fail after a number of batches (a kill)."""

    def __init__(self, fail_after: int | None = None):
        self.calls: list[tuple[str, ...]] = []
        self.fail_after = fail_after

    def __call__(self, prompts: list[str], k: int) -> list[list[Generation]]:
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("killed")
        self.calls.append(tuple(prompts))
        return [[Generation(f"{p}|with:{'+'.join(prompts)}", len(p), False) for _ in range(k)] for p in prompts]


PROMPTS = [f"p{i}" for i in range(11)]
BATCHES = _batches(len(PROMPTS), 3)
IDENT = {"model": "fake", "n": len(PROMPTS)}


def _texts(res):
    return [[g.text for g in per] for per in res]


def test_a_cut_off_run_resumes_to_the_same_completions_with_the_same_batches(tmp_path):
    full = Gen()
    want = generate_batched(PROMPTS, BATCHES, K, full)
    part = PartialCompletions(tmp_path / PARTIAL_NAME)
    with pytest.raises(RuntimeError, match="killed"):
        generate_batched(PROMPTS, BATCHES, K, Gen(fail_after=2), part, IDENT)
    again = Gen()
    part2 = PartialCompletions(tmp_path / PARTIAL_NAME)
    got = generate_batched(PROMPTS, BATCHES, K, again, part2, IDENT)
    assert _texts(got) == _texts(want)
    assert again.calls == full.calls[2:]  # only the missing batches, each with its single-run composition
    assert part2.summary() | {"moved_aside": None} == {
        "partial_file": PARTIAL_NAME, "sessions": 2, "batches_total": 4, "batches_resumed": 2,
        "batches_generated_this_session": 2, "moved_aside": None}


def test_a_torn_last_line_is_cut_off_and_that_batch_generated_again(tmp_path):
    path = tmp_path / PARTIAL_NAME
    with pytest.raises(RuntimeError):
        generate_batched(PROMPTS, BATCHES, K, Gen(fail_after=3), PartialCompletions(path), IDENT)
    raw = path.read_bytes()
    last = raw.rstrip(b"\n").rsplit(b"\n", 1)[1]
    path.write_bytes(raw[:len(raw) - len(last) - 1] + last[:25])  # the third batch's line, torn mid-write
    again = Gen()
    got = generate_batched(PROMPTS, BATCHES, K, again, PartialCompletions(path), IDENT)
    assert _texts(got) == _texts(generate_batched(PROMPTS, BATCHES, K, Gen()))
    assert len(again.calls) == 2  # batches 2 and 3
    assert all(json.loads(line) for line in path.read_text().splitlines())


def test_another_identity_is_moved_aside_never_mixed(tmp_path):
    path = tmp_path / PARTIAL_NAME
    with pytest.raises(RuntimeError):
        generate_batched(PROMPTS, BATCHES, K, Gen(fail_after=2), PartialCompletions(path), IDENT)
    again = Gen()
    part = PartialCompletions(path)
    generate_batched(PROMPTS, BATCHES, K, again, part, {**IDENT, "model": "other"})
    assert len(again.calls) == 4 and part.resumed == 0 and "identity" in part.moved_aside
    stale = [p for p in tmp_path.iterdir() if ".stale-" in p.name]
    assert len(stale) == 1 and len(stale[0].read_text().splitlines()) == 4  # header, session, 2 batches: kept


def test_damage_before_the_last_line_or_a_foreign_partition_moves_the_file_aside(tmp_path):
    path = tmp_path / PARTIAL_NAME
    with pytest.raises(RuntimeError):
        generate_batched(PROMPTS, BATCHES, K, Gen(fail_after=3), PartialCompletions(path), IDENT)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:2] + ["{not json"] + lines[3:]) + "\n")
    part = PartialCompletions(path)
    generate_batched(PROMPTS, BATCHES, K, Gen(), part, IDENT)
    assert part.resumed == 0 and "does not parse" in part.moved_aside
    path.unlink()
    with pytest.raises(RuntimeError):
        generate_batched(PROMPTS, BATCHES, K, Gen(fail_after=2), PartialCompletions(path), IDENT)
    part = PartialCompletions(path)
    generate_batched(PROMPTS, _batches(len(PROMPTS), 4), K, Gen(), part, IDENT)  # another batch size
    assert part.resumed == 0 and "partition" in part.moved_aside


SLOW = """
import sys, time
from physcal.eval.run_eval import Generation, PartialCompletions, generate_batched
prompts = [f"p{i}" for i in range(11)]
order = list(range(11))[::-1]
batches = [order[s:s + 3] for s in range(0, 11, 3)]
def gen(ps, k):
    time.sleep(0.4)
    return [[Generation(f"{p}|with:{'+'.join(ps)}", len(p), False)] for p in ps]
generate_batched(prompts, batches, 1, gen, PartialCompletions(sys.argv[1]), {"model": "fake", "n": 11})
"""


def test_a_sigkill_mid_run_loses_at_most_the_batch_in_flight(tmp_path):
    path = tmp_path / PARTIAL_NAME
    proc = subprocess.Popen([sys.executable, "-c", SLOW, str(path)], cwd=REPO_ROOT, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 60:
        if path.exists() and path.read_text().count('"kind": "batch"') >= 2:
            break
        time.sleep(0.05)
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=30)
    again = Gen()
    part = PartialCompletions(path)
    got = generate_batched(PROMPTS, BATCHES, K, again, part, IDENT)
    assert _texts(got) == _texts(generate_batched(PROMPTS, BATCHES, K, Gen()))
    assert 2 <= part.resumed < 4 and len(again.calls) == 4 - part.resumed


# ------------------------------------------------------------------------------------------- HFBackend and run()
def _fake_hf_backend(fail_after: int | None = None):
    """An HFBackend without a model: _generate is a Gen, the chat template is the user message."""
    import torch

    class Tok:
        def apply_chat_template(self, conv, tokenize, add_generation_prompt, **_):
            return conv[-1]["content"]

    b = object.__new__(HFBackend)
    b.torch, b.tok, b.chat_template_kwargs, b.add_special_tokens = torch, Tok(), {}, True
    b.gen_cfg = {"num_samples": 1, "batch_size": 3, "do_sample": False, "seed": 0}
    b.model_id, b.revision, b.lora_hash, b.loader = "fake", None, "abc", "unsloth_4bit"
    gen = Gen(fail_after)
    b._generate = gen
    return b, gen


def test_hf_backend_resumes_with_its_own_length_sorted_partition(tmp_path):
    convs = [[{"role": "user", "content": "x" * (i % 5 + 1) + str(i)}] for i in range(10)]
    single, g1 = _fake_hf_backend()
    want = [x.text for x in single(convs)]
    cut, _ = _fake_hf_backend(fail_after=2)
    with pytest.raises(RuntimeError):
        cut(convs, partial=PartialCompletions(tmp_path / PARTIAL_NAME))
    resumed, g3 = _fake_hf_backend()
    assert [x.text for x in resumed(convs, partial=PartialCompletions(tmp_path / PARTIAL_NAME))] == want
    assert g3.calls == g1.calls[2:]
    other, g4 = _fake_hf_backend()
    other.lora_hash = "def"  # another adapter: nothing is reused
    other(convs, partial=PartialCompletions(tmp_path / PARTIAL_NAME))
    assert g4.calls == g1.calls


_ORACLE: dict = {}


class ResumableOracle:
    """A run() backend that supports resume: oracle answers whose step line names the batch's members."""

    supports_resume = True

    def __init__(self, fail_after: int | None = None):
        self.gen = Gen(fail_after)

    def __call__(self, conversations, partial=None):
        texts = [c[-1]["content"] for c in conversations]

        def generate(prompts, k):
            self.gen(prompts, k)
            out = []
            for t in prompts:
                p = _ORACLE[t]
                steps = "\n".join(f"{s} = {v:.6g} {p.intermediate_si_units[s]}" for s, v in p.intermediates.items())
                ans = f"{p.target_value:.6g} ± {0.005 * p.target_value:.3g} {p.target_unit}"
                tag = "+".join(str(_ORACLE[q].seed) for q in prompts)
                out.append([Generation(f"batch {tag}\n<steps>\n{steps}\n</steps>\n<answer>{ans}</answer>", 50, False)])
            return out

        res = generate_batched(texts, _batches(len(texts), 5), 1, generate, partial, {"model": "oracle"})
        return [g for per in res for g in per]


def _tape(path):
    return [(r["seed"], r["split"], r["completion"], r["reward"]) for r in map(json.loads, path.read_text().splitlines())]


def test_run_resumes_a_cut_off_greedy_eval_to_the_same_tape_and_then_deletes_the_partial_file(tmp_path):
    env_cfg, eval_cfg = env_config(), load_config("eval")
    for i, (_, scfg) in enumerate(eval_cfg["splits"].items()):
        for p in build_split(i, scfg, eval_cfg, env_cfg, limit=8):
            _ORACLE[p.text] = p
    single = tmp_path / "single"
    m1 = run(eval_cfg, backend_factory=lambda _: ResumableOracle(), limit=8, output_dir=single)
    assert m1["resume"]["sessions"] == 1 and m1["resume"]["batches_resumed"] == 0
    assert not (single / PARTIAL_NAME).exists()
    cut = tmp_path / "cut"
    with pytest.raises(RuntimeError, match="killed"):
        run(eval_cfg, backend_factory=lambda _: ResumableOracle(fail_after=2), limit=8, output_dir=cut)
    assert (cut / PARTIAL_NAME).exists() and not (cut / "metrics.json").exists()
    backend = ResumableOracle()
    m2 = run(eval_cfg, backend_factory=lambda _: backend, limit=8, output_dir=cut)
    assert _tape(cut / "rollouts.jsonl") == _tape(single / "rollouts.jsonl")
    assert len(backend.gen.calls) == m1["resume"]["batches_total"] - 2
    assert m2["resume"] | {"moved_aside": None} == {
        "partial_file": PARTIAL_NAME, "sessions": 2, "batches_total": m1["resume"]["batches_total"],
        "batches_resumed": 2, "batches_generated_this_session": m1["resume"]["batches_total"] - 2, "moved_aside": None}
    assert not (cut / PARTIAL_NAME).exists()
    assert "resumed (D142): session 2" in (cut / "summary.md").read_text()
    for k in ("accuracy", "coverage_1sigma", "mean_reward"):
        assert [s[k] for s in m2["splits"].values()] == [s[k] for s in m1["splits"].values()]


def test_sampled_evals_and_plain_backends_never_write_a_partial_file(tmp_path):
    cfg = copy.deepcopy(load_config("eval"))
    cfg["generation"].update(do_sample=True)
    backend = ResumableOracle()
    env_cfg = env_config()
    for i, (_, scfg) in enumerate(cfg["splits"].items()):
        for p in build_split(i, scfg, cfg, env_cfg, limit=2):
            _ORACLE[p.text] = p
    m = run(cfg, backend_factory=lambda _: backend, limit=2, output_dir=tmp_path / "sampled")
    assert m["resume"] is None and not (tmp_path / "sampled" / PARTIAL_NAME).exists()
    m = run(load_config("eval"), backend_factory=lambda _: (lambda convs: [Generation("x", 1, False) for _ in convs]),
            limit=2, output_dir=tmp_path / "plain")
    assert m["resume"] is None and not (tmp_path / "plain" / PARTIAL_NAME).exists()

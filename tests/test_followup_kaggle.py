"""D157 / D160 (exploratory follow-up track): item 3's Kaggle sampler (chunks, timing, health gate, resume), the zip,
its local ingest, the per-model settings, and the notebooks' pinned commits. A fake engine, clock and tokenizer stand
in for vLLM; CPU only."""

import copy
import json
import re
import zipfile

import pytest

from physcal.eval import followup_kaggle as K
from physcal.eval import followup_scale as F
from physcal.tape.tape import TAPE_FIELDS

CS, K8, DEV = 4, 8, 12   # a 12-candidate prefix in chunks of 4


@pytest.fixture()
def cfg():
    c = copy.deepcopy(K._config())
    c["scale"]["chunk_size"] = CS
    c["scale"]["dev_indices"] = [DEV, DEV + 2]
    c["scale"]["kaggle"]["throughput_n"] = 3
    return c


class Tok:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "|".join(m["role"] + ":" + m["content"] for m in messages) + "|assistant:"

    def __call__(self, s, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": list(range(len(s) // 10))}


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class Engine:
    """Even samples box the target (correct), odd ones 10% off; `boxed=False`: no box at all. Each call advances the
    clock by `per_request` seconds per request."""

    def __init__(self, cfg, clock, per_request=1.0, boxed=True):
        self.pool, self.cfg, self.clock, self.per, self.boxed = F.pool(cfg), cfg, clock, per_request, boxed
        self.calls = []

    def __call__(self, ids, seeds):
        self.calls.append(len(seeds))
        self.clock.t += self.per * len(seeds)
        out = []
        for s in seeds:
            i, j = divmod(s - self.cfg["scale"]["kaggle"]["seed"], K8)
            v = self.pool[i]["target_value"] * (1.0 if j % 2 == 0 else 1.1)
            out.append((f"so \\boxed{{{v!r}}}" if self.boxed else "I am not sure", 100 + j, "stop"))
        return out


def run(cfg, tmp_path, clock, engine, start_offset=0.0, model=K.MODEL):
    made = []

    def factory():
        made.append(1)
        return engine

    rc = K.sample(factory, Tok(), tmp_path, clock.t - start_offset, cfg, now=clock, log=lambda m: None, model=model)
    return rc, len(made)


def test_sample_writes_the_prefix_in_chunks_and_a_rerun_resumes_nothing(cfg, tmp_path):
    clock = Clock()
    eng = Engine(cfg, clock)
    rc, made = run(cfg, tmp_path, clock, eng)
    assert rc == 0 and made == 1
    assert eng.calls == [3 * K8, 1 * K8, CS * K8, CS * K8]   # the timed 3, the rest of chunk 0, chunks 1 and 2
    rows = F.sampled_prefix(K.MODEL, cfg, K.samples_dir(tmp_path))
    assert len(rows) == DEV * K8
    r = rows[0]
    assert set(TAPE_FIELDS) <= set(r) and r["reward"] is None and r["lora_hash"] is None
    assert {"index", "key", "sample", "completion", "n_tokens", "truncated", "value", "correct", "brace"} <= set(r)
    assert len({x["sample_seed"] for x in rows}) == len(rows)
    assert [x["correct"] for x in rows[:K8]] == [True, False] * 4
    tp = json.loads((K.scale_dir(tmp_path) / "throughput.json").read_text())
    assert tp["health_ok"] and tp["n_candidates"] == 3 and tp["s_per_problem"] == pytest.approx(K8 * 1.0)
    meta = json.loads((K.samples_dir(tmp_path) / "meta.json").read_text())
    assert meta["segments"][0]["decoding"] == K.decoding(cfg) and meta["segments"][0]["last_chunk_end"] == DEV
    assert run(cfg, tmp_path, clock, Engine(cfg, clock)) == (0, 0)   # complete: no engine is built


def test_sample_stops_before_the_sampling_deadline_and_a_new_session_continues(cfg, tmp_path):
    clock = Clock()
    # chunk 0 takes 4 x 8 = 32 s; the session has 0.02 h = 72 s: 72 - 32 = 40 s left, one more chunk at 1.25 x 32 s
    cfg["scale"]["kaggle"]["sampling_hours"] = 0.02
    rc, _ = run(cfg, tmp_path, clock, Engine(cfg, clock))
    assert rc == 0
    plan = json.loads((K.scale_dir(tmp_path) / "plan.json").read_text())
    assert plan["n_planned"] == 2 * CS
    assert len(F.sampled_prefix(K.MODEL, cfg, K.samples_dir(tmp_path))) == 2 * CS * K8
    eng = Engine(cfg, clock)
    rc, _ = run(cfg, tmp_path, clock, eng)                          # a new session, a new deadline
    assert rc == 0 and eng.calls == [CS * K8]                       # no new timing; the last chunk
    assert len(F.sampled_prefix(K.MODEL, cfg, K.samples_dir(tmp_path))) == DEV * K8
    assert len(json.loads((K.samples_dir(tmp_path) / "meta.json").read_text())["segments"]) == 2


def test_the_health_gate_stops_the_run_and_stays_stopped(cfg, tmp_path):
    clock = Clock()
    rc, _ = run(cfg, tmp_path, clock, Engine(cfg, clock, boxed=False))
    assert rc == K.EXIT_HEALTH
    assert not list(K.samples_dir(tmp_path).glob("chunk_*.jsonl"))
    assert not json.loads((K.scale_dir(tmp_path) / "throughput.json").read_text())["health_ok"]
    assert run(cfg, tmp_path, clock, Engine(cfg, clock)) == (K.EXIT_HEALTH, 0)


def test_plan_total_and_may_start(cfg):
    m = cfg["scale"]["kaggle"]["rate_margin"]
    assert K.plan_total(4, 0.0, 10 * m * 2.0 * CS, 2.0, cfg) == DEV          # capped at the prefix
    assert K.plan_total(0, 0.0, 2.5 * m * 2.0 * CS, 2.0, cfg) == 2 * CS     # whole chunks only
    assert K.plan_total(4, 100.0, 50.0, 2.0, cfg) == 4                       # past the end: nothing more
    assert K.may_start(0.0, m * 10.0, [8.0, 12.0], 99.0, cfg)               # the session's mean chunk time
    assert not K.may_start(0.0, m * 10.0 - 1e-6, [8.0, 12.0], 99.0, cfg)
    assert K.may_start(0.0, m * 2.0 * CS, [], 2.0, cfg) and not K.may_start(1.0, m * 2.0 * CS, [], 2.0, cfg)
    full = K._config()
    assert K.max_candidates(full) == 1176


@pytest.mark.parametrize("model", ["7b", "1.5b"])
def test_pack_ingest_roundtrip_regrades_and_is_idempotent(cfg, tmp_path, model):
    clock = Clock()
    work = tmp_path / "working"
    run(cfg, work, clock, Engine(cfg, clock), model=model)
    z = K.pack(work, cfg, model)
    assert z.name == K.kaggle_cfg(cfg, model)["zip_name"]
    with zipfile.ZipFile(z) as zf:
        assert all(n.startswith(f"scale/{model}/") for n in zf.namelist())
    dest = tmp_path / "repo" / model
    res = K.ingest(z, cfg, tok=Tok(), staging=tmp_path / "staging", dest=dest, model=model)
    assert res["n_candidates"] == DEV and res["chunks_written"] == DEV // CS and res["regraded_changed_rows"] == 0
    assert res["prompts_checked"] == DEV and res["accuracy_per_sample"] == pytest.approx(0.5)
    rows = F.sampled_prefix(model, cfg, dest / "samples")
    kc = K.kaggle_cfg(cfg, model)
    assert len(rows) == DEV * K8 and rows[0]["model_id"] == f"{kc['repo_id']}@{kc['revision']}"
    assert (dest / "kaggle" / "throughput.json").exists() and (dest / "samples" / "meta.json").exists()
    again = K.ingest(z, cfg, tok=Tok(), staging=tmp_path / "staging2", dest=dest, model=model)
    assert again["chunks_written"] == 0
    other = "1.5b" if model == "7b" else "7b"   # another model's zip is refused before anything is read
    with pytest.raises(SystemExit, match="unexpected"):
        K.ingest(z, cfg, tok=Tok(), staging=tmp_path / "staging3", dest=tmp_path / "other", model=other)


def test_kaggle_settings_overlay_per_model(cfg):
    """D160: the 7B's settings are `scale.kaggle` itself; the 1.5B changes only its weights, zip and disk check."""
    assert K.kaggle_cfg(cfg, "7b") == cfg["scale"]["kaggle"]
    k15 = K.kaggle_cfg(cfg, "1.5b")
    assert set(k15) == set(cfg["scale"]["kaggle"])
    changed = {key for key in k15 if k15[key] != cfg["scale"]["kaggle"][key]}
    assert changed == {"repo_id", "revision", "weights_sha256", "zip_name", "min_free_gb"}
    assert k15["repo_id"] == "Qwen/Qwen2.5-Math-1.5B-Instruct"
    with pytest.raises(SystemExit, match="unknown model"):
        K.kaggle_cfg(cfg, "3b")


def test_ingest_refuses_another_tokenizer_rendering(cfg, tmp_path):
    clock = Clock()
    run(cfg, tmp_path / "w", clock, Engine(cfg, clock))
    z = K.pack(tmp_path / "w", cfg)

    class Other(Tok):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "other " + super().apply_chat_template(messages, tokenize, add_generation_prompt)

    with pytest.raises(SystemExit, match="another prompt"):
        K.ingest(z, cfg, tok=Other(), staging=tmp_path / "s", dest=tmp_path / "d")


@pytest.mark.parametrize("name, why", [("../evil.jsonl", "unexpected"), ("scale/7b/run.py", "unexpected"),
                                        ("/etc/passwd", "unexpected")])
def test_safe_extract_refuses_unexpected_names(cfg, tmp_path, name, why):
    z = tmp_path / "x.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(name, "x")
        zf.writestr("scale/7b/MANIFEST.json", json.dumps({name: K._sha256(b"x")}))
    with pytest.raises(SystemExit, match=why):
        K.safe_extract(z, tmp_path / "out", cfg)
    assert not any((tmp_path / "out").iterdir())


def test_safe_extract_checks_the_manifest(cfg, tmp_path):
    z = tmp_path / "x.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("scale/7b/plan.json", "{}")
        zf.writestr("scale/7b/MANIFEST.json", json.dumps({"scale/7b/plan.json": K._sha256(b"[]")}))
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        K.safe_extract(z, tmp_path / "out", cfg)


def test_restore_copies_the_prefix_and_refuses_other_settings(cfg, tmp_path):
    clock = Clock()
    prev = tmp_path / "input" / "prev-version"
    cfg2 = copy.deepcopy(cfg)
    cfg2["scale"]["kaggle"]["sampling_hours"] = 0.02
    run(cfg2, prev, clock, Engine(cfg2, clock))                       # two chunks
    res = K.restore(tmp_path / "w", tmp_path / "input", cfg)
    assert res["restored_candidates"] == 2 * CS
    assert len(F.sampled_prefix(K.MODEL, cfg, K.samples_dir(tmp_path / "w"))) == 2 * CS * K8
    assert (K.scale_dir(tmp_path / "w") / "throughput.json").exists()
    other = copy.deepcopy(cfg)
    other["scale"]["kaggle"]["max_new_tokens"] = 2048
    with pytest.raises(SystemExit, match="other settings"):
        K.restore(tmp_path / "w2", tmp_path / "input", other)
    assert K.restore(tmp_path / "w3", tmp_path / "nothing-here", cfg) == {"restored_rows": 0}


# ------------------------------------------------------------------------------------------- the notebooks
SAMPLING_KEYS = ("generation", "system_prompt", "tolerance", "chunk_size", "dev_indices", "pool_config")


def _notebook(name):
    import ast
    import re
    import subprocess

    from physcal.config import REPO_ROOT

    nb = json.loads((REPO_ROOT / "notebooks" / name).read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for src in code:
        ast.parse(src)
    commit = re.search(r'^COMMIT = "([0-9a-f]{40})"$', code[0], flags=re.M).group(1)

    def pinned(path):
        return subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{commit}:{path}"], capture_output=True,
                              text=True, check=True).stdout

    return code, commit, pinned


def test_the_7b_notebook_is_the_commit_that_sampled_the_ingested_7b_rows():
    """D160: the 7B's Kaggle run is done, so its notebook's pin is a provenance check: its commit is the one every
    ingested 7B segment recorded, and that commit's sampling settings are today's 7B settings (which the local ingest
    and analysis read)."""
    import yaml

    code, commit, pinned = _notebook("kaggle_7b_scale.ipynb")
    meta = json.loads((F.out_dir("7b") / "samples" / "meta.json").read_text())
    assert {seg["commit"] for seg in meta["segments"]} == {commit}
    old, cfg = yaml.safe_load(pinned("configs/followup.yaml"))["scale"], K._config()
    assert old["kaggle"] == K.kaggle_cfg(cfg, "7b")
    for key in SAMPLING_KEYS:
        assert old[key] == cfg["scale"][key], f"scale.{key} changed since the 7B's commit {commit[:12]}"


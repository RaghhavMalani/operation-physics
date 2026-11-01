"""D157 item 3 on Kaggle (EXPLORATORY, follow-up project; not the paper): the 7B's k = 8 samples with vLLM on
"GPU T4 x2" (the official Qwen/Qwen2.5-Math-7B-Instruct, fp16, tensor-parallel 2), in followup_scale's chunk format
plus the tape fields, so a session cutoff loses at most one chunk and a re-run resumes; and, locally, the ingest of
the downloaded zip.

On Kaggle (notebooks/kaggle_7b_scale.ipynb runs these with its venv's python, the repo on PYTHONPATH):
    python -m physcal.eval.followup_kaggle download --model-dir D        # the pinned revision, sha256-checked
    python -m physcal.eval.followup_kaggle restore --out W --src /kaggle/input
    python -m physcal.eval.followup_kaggle sample --model-dir D --out W --session-start T [--enforce-eager]
    python -m physcal.eval.followup_kaggle pack --out W
Locally (CPU), before scripts/queue_followup_kaggle.sh's readouts and analysis:
    uv run python -m physcal.eval.followup_kaggle ingest --zip Z

`sample` exits 0 when done, EXIT_ENGINE if the engine cannot start, EXIT_HEALTH if the health gate fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config
from physcal.eval import followup_scale as F

MODEL = "7b"
EXIT_ENGINE, EXIT_HEALTH = 2, 3
_ALLOWED = re.compile(r"scale/7b/(samples/chunk_\d{5}\.jsonl|samples/meta\.json|throughput\.json|plan\.json"
                      r"|logs/[A-Za-z0-9_.-]+\.log)")


def _config() -> dict:
    return load_config("followup")


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def scale_dir(out: Path) -> Path:
    return Path(out) / "scale" / MODEL


def samples_dir(out: Path) -> Path:
    return scale_dir(out) / "samples"


def max_candidates(cfg: dict) -> int:
    """The full chunks below the development indices (never analyzed, D151): 1,176 of X2's 1,200."""
    sc = cfg["scale"]
    return sc["dev_indices"][0] // sc["chunk_size"] * sc["chunk_size"]


def decoding(cfg: dict) -> dict:
    """What every segment must share: D152's sampling, k and the Kaggle budget (the local generation block's
    max_new_tokens, batch size and seed are not used on Kaggle)."""
    g = cfg["scale"]["generation"]
    return {key: g[key] for key in ("k", "temperature", "top_p", "top_k", "repetition_penalty")} | {
        "max_new_tokens": cfg["scale"]["kaggle"]["max_new_tokens"], "seed": cfg["scale"]["kaggle"]["seed"]}


def sample_seed(index: int, sample: int, cfg: dict) -> int:
    return cfg["scale"]["kaggle"]["seed"] + cfg["scale"]["generation"]["k"] * index + sample


def render_prompt(tok, text: str, cfg: dict) -> tuple[str, list[int]]:
    """D151's native prompt through the tokenizer's chat template; the ids are what the sampler is given."""
    s = tok.apply_chat_template(F.messages(text, cfg), tokenize=False, add_generation_prompt=True)
    return s, tok(s, add_special_tokens=False)["input_ids"]


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.rename(path)


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


# ------------------------------------------------------------------------------------------- rows
def rows_for(probs: list[dict], prompts: dict, gens: list[tuple[str, int, str]], cfg: dict, run_id: str) -> list[dict]:
    """One row per completion, problem-major (k per problem): the tape fields (reward null, since the native
    \\boxed format has no verifier reward; breakdown = the D151 / D154 grade), item 3's chunk fields
    (followup_scale.generate_problems) and the sampling provenance."""
    sc = cfg["scale"]
    k = sc["generation"]["k"]
    rows = []
    for i, p in enumerate(probs):
        problem = {key: v for key, v in p.items() if key != "_index"}
        text_sha, n_prompt = prompts[p["_index"]]
        for s in range(k):
            text, n_tokens, finish = gens[i * k + s]
            truncated = finish == "length"
            gr = F.grade(problem, text, truncated, sc["tolerance"])
            rows.append({"run_id": run_id, "step": 0, "seed": p["seed"], "generator_version": p["generator_version"],
                         "model_id": f"{sc['kaggle']['repo_id']}@{sc['kaggle']['revision']}", "lora_hash": None,
                         "problem": problem, "completion": text, "reward": None,
                         "breakdown": {"grader": "followup_scale.grade (D151, D154)", "reason": gr["reason"],
                                       "correct": gr["correct"]},
                         "index": p["_index"], "key": [p["family"], p["depth"], p["seed"]], "sample": s,
                         "n_tokens": n_tokens, "truncated": truncated, **gr,
                         "finish_reason": finish, "sample_seed": sample_seed(p["_index"], s, cfg),
                         "prompt_sha256": text_sha, "prompt_tokens": n_prompt})
    return rows


def plan_total(start: int, now: float, t_end: float, s_per_problem: float, cfg: dict) -> int:
    """Candidates planned in all: the done prefix plus the whole chunks that fit before t_end at rate_margin x the
    measured time per problem, capped at max_candidates."""
    cs, m = cfg["scale"]["chunk_size"], cfg["scale"]["kaggle"]["rate_margin"]
    fit = int(max(0.0, t_end - now) // (m * s_per_problem * cs)) if s_per_problem > 0 else 0
    return min(max_candidates(cfg), start + fit * cs)


def may_start(now: float, t_end: float, chunk_times: list[float], s_per_problem: float, cfg: dict) -> bool:
    """A chunk starts only if its projected end (rate_margin x the session's mean chunk time; before the first,
    x the measured time per problem x chunk size) is before t_end."""
    cs, m = cfg["scale"]["chunk_size"], cfg["scale"]["kaggle"]["rate_margin"]
    proj = m * (float(np.mean(chunk_times)) if chunk_times else s_per_problem * cs)
    return now + proj <= t_end


# ------------------------------------------------------------------------------------------- Kaggle: the engine
class VLLMEngine:
    """vLLM on the pinned weights: one request per sample (n = 1) with its own seed; D152's decoding."""

    def __init__(self, model_dir: Path, cfg: dict, enforce_eager: bool):
        from vllm import LLM, SamplingParams

        kc, self.g = cfg["scale"]["kaggle"], cfg["scale"]["generation"]
        self.max_tokens = kc["max_new_tokens"]
        self.engine_args = {"model": str(model_dir), "tokenizer": str(model_dir), "dtype": kc["dtype"],
                            "tensor_parallel_size": kc["tensor_parallel_size"], "max_model_len": kc["max_model_len"],
                            "gpu_memory_utilization": kc["gpu_memory_utilization"],
                            "disable_custom_all_reduce": kc["disable_custom_all_reduce"],
                            "attention_backend": kc["attention_backend"], "enforce_eager": enforce_eager,
                            "seed": kc["seed"]}
        eos = json.loads((Path(model_dir) / "generation_config.json").read_text())["eos_token_id"]
        self.stop_ids = sorted(eos if isinstance(eos, list) else [eos])
        self.SamplingParams = SamplingParams
        self.llm = LLM(**self.engine_args)

    def __call__(self, prompt_ids: list[list[int]], seeds: list[int]) -> list[tuple[str, int, str]]:
        g = self.g
        params = [self.SamplingParams(n=1, temperature=g["temperature"], top_p=g["top_p"], top_k=g["top_k"],
                                      repetition_penalty=g["repetition_penalty"], max_tokens=self.max_tokens,
                                      seed=s, stop_token_ids=self.stop_ids, skip_special_tokens=True) for s in seeds]
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompt_ids], sampling_params=params, use_tqdm=False)
        return [(o.outputs[0].text, len(o.outputs[0].token_ids), o.outputs[0].finish_reason) for o in outs]


def _environment(model_dir: Path) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    def v(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as exc:  # recorded, never fatal
            return f"unavailable: {exc}"

    tc = json.loads((Path(model_dir) / "tokenizer_config.json").read_text())
    return {"python": sys.version.split()[0], "packages": {n: v(n) for n in ("vllm", "torch", "transformers", "pint")},
            "gpus": run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
            "commit": run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
            "config_sha256": _file_sha256(REPO_ROOT / "configs" / "followup.yaml"),
            "chat_template_sha256": _sha256(tc["chat_template"].encode()),
            "tokenizer_sha256": _file_sha256(Path(model_dir) / "tokenizer.json")}


# ------------------------------------------------------------------------------------------- Kaggle: the stages
def download(model_dir: Path, cfg: dict | None = None) -> dict:
    """The pinned revision of the official weights; every safetensors file must match HF's LFS sha256."""
    from huggingface_hub import snapshot_download

    cfg = cfg or _config()
    kc = cfg["scale"]["kaggle"]
    snapshot_download(kc["repo_id"], revision=kc["revision"], local_dir=str(model_dir),
                      allow_patterns=kc["download_patterns"])
    got = {name: _file_sha256(Path(model_dir) / name) for name in kc["weights_sha256"]}
    bad = {n: h for n, h in got.items() if h != kc["weights_sha256"][n]}
    if bad:
        raise SystemExit(f"weights do not match the pinned sha256: {bad}")
    res = {"repo_id": kc["repo_id"], "revision": kc["revision"], "weights_sha256": got}
    (Path(model_dir) / "_verified.json").write_text(json.dumps(res, indent=1))
    return res


def restore(out: Path, src: Path, cfg: dict | None = None) -> dict:
    """A previous session's output (attached as a Kaggle input): the longest valid chunk prefix under `src`, with its
    throughput.json and meta.json, copied into `out`. Chunks of another revision or decoding are refused."""
    cfg = cfg or _config()
    kc = cfg["scale"]["kaggle"]
    found = []
    for d in sorted({p.parent for p in Path(src).rglob("chunk_*.jsonl") if p.parent.name == "samples"}):
        try:
            found.append((len(F.sampled_prefix(MODEL, cfg, d)), d))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"restore: skipping {d}: {exc}", flush=True)
    found = [f for f in found if f[0]]
    if not found:
        return {"restored_rows": 0}
    n, d = max(found)
    meta = json.loads((d / "meta.json").read_text())
    for seg in meta["segments"]:
        if seg["revision"] != kc["revision"] or seg["decoding"] != decoding(cfg):
            raise SystemExit(f"restore: {d} was sampled with other settings: {seg}")
    dst = samples_dir(out)
    dst.mkdir(parents=True, exist_ok=True)
    k, cs = cfg["scale"]["generation"]["k"], cfg["scale"]["chunk_size"]
    for start in range(0, n // k, cs):
        f = F.chunk_path(MODEL, start, cfg, d)
        if not (dst / f.name).exists():
            shutil.copyfile(f, dst / f.name)
    shutil.copyfile(d / "meta.json", dst / "meta.json")
    for name in ("throughput.json", "plan.json"):
        if (d.parent / name).exists():
            shutil.copyfile(d.parent / name, scale_dir(out) / name)
    if (d.parent / "logs").is_dir():
        shutil.copytree(d.parent / "logs", scale_dir(out) / "logs", dirs_exist_ok=True)
    res = {"restored_rows": n, "restored_candidates": n // k, "from": str(d)}
    print(f"restore: {res}", flush=True)
    return res


def sample(engine_factory, tok, out: Path, session_start: float, cfg: dict | None = None, now=time.time,
           log=print, settings: dict | None = None) -> int:
    """D157: from the first missing chunk, in pool order. If the run has no throughput.json yet, the first
    throughput_n candidates of that chunk are generated alone and timed (the health gate reads them), and the chunk is
    finished. Then N is planned (plan_total) and chunks run while may_start. Returns 0 or EXIT_HEALTH."""
    cfg = cfg or _config()
    sc, kc = cfg["scale"], cfg["scale"]["kaggle"]
    cs, k = sc["chunk_size"], sc["generation"]["k"]
    sd = samples_dir(out)
    sd.mkdir(parents=True, exist_ok=True)
    probs = [{**p, "_index": i} for i, p in enumerate(F.pool(cfg))]
    t_end = session_start + kc["sampling_hours"] * 3600
    start = len(F.sampled_prefix(MODEL, cfg, sd)) // k
    tp_path, meta_path, plan_path = scale_dir(out) / "throughput.json", sd / "meta.json", scale_dir(out) / "plan.json"
    if start + cs > max_candidates(cfg):
        log(f"the prefix is complete ({start} candidates); nothing to sample")
        return 0
    tp = json.loads(tp_path.read_text()) if tp_path.exists() else None
    if tp is not None and not tp["health_ok"]:
        log(f"the health gate failed in an earlier session ({tp['answered_share']:.3f} answered); nothing is sampled")
        return EXIT_HEALTH
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "decision": "D157: item 3's 7B samples on Kaggle (exploratory, follow-up project; not the paper)",
        "run_id": f"followup-scale-7b-kaggle-{kc['revision'][:8]}", "segments": []}
    seg = {"session_start": session_start, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "first_chunk": start,
           "repo_id": kc["repo_id"], "revision": kc["revision"], "decoding": decoding(cfg), **(settings or {})}
    meta["segments"].append(seg)
    engine = engine_factory()
    seg["engine_args"] = getattr(engine, "engine_args", None)
    prompts: dict = {}

    def run(ps: list[dict]) -> tuple[list[dict], float]:
        if not ps:
            return [], 0.0
        ids = []
        for p in ps:
            text, pid = render_prompt(tok, p["text"], cfg)
            prompts[p["_index"]] = (_sha256(text.encode()), len(pid))
            ids += [pid] * k
        seeds = [sample_seed(p["_index"], s, cfg) for p in ps for s in range(k)]
        t0 = now()
        gens = engine(ids, seeds)
        return rows_for(ps, prompts, gens, cfg, meta["run_id"]), now() - t0

    def save_meta(**kw):
        seg.update(kw)
        _write_atomic(meta_path, json.dumps(meta, indent=1, default=str))

    times: list[float] = []
    if tp is None:
        n = kc["throughput_n"]
        rows, dt = run(probs[start:start + n])
        answered = float(np.mean([r["value"] is not None for r in rows]))
        tp = {"decision": "D157", "n_candidates": n, "first_index": start, "seconds": dt, "s_per_problem": dt / n,
              "completion_tokens": int(sum(r["n_tokens"] for r in rows)),
              "tokens_per_s": sum(r["n_tokens"] for r in rows) / dt if dt > 0 else None,
              "answered_share": answered, "truncated_share": float(np.mean([r["truncated"] for r in rows])),
              "distinct_completions_per_problem": float(np.mean(
                  [len({r["completion"] for r in rows[i:i + k]}) for i in range(0, len(rows), k)])),
              "min_answered_share": kc["min_answered_share"], "health_ok": answered >= kc["min_answered_share"]}
        _write_atomic(tp_path, json.dumps(tp, indent=1))
        log(f"throughput on {n} candidates: {dt:.0f} s ({tp['s_per_problem']:.1f} s / problem, "
            f"{tp['tokens_per_s'] or 0:.0f} tokens / s); answered {answered:.3f}, truncated {tp['truncated_share']:.3f}")
        if not tp["health_ok"]:
            log(f"STOP: the health gate failed ({answered:.3f} answered < {kc['min_answered_share']}); no more chunks")
            save_meta(chunks=0, stop="health gate")
            return EXIT_HEALTH
        rest, dt2 = run(probs[start + n:start + cs])
        _write_atomic(F.chunk_path(MODEL, start, cfg, sd), _jsonl(rows + rest))
        times.append(dt + dt2)
        start += cs
    n_planned = plan_total(start, now(), t_end, tp["s_per_problem"], cfg)
    plan = {"session_start": session_start, "t_end": t_end, "planned_at": now(), "done_at_plan": start,
            "s_per_problem": tp["s_per_problem"], "rate_margin": kc["rate_margin"], "n_planned": n_planned}
    _write_atomic(plan_path, json.dumps(plan, indent=1))
    log(f"plan: {n_planned} candidates in all ({start} done); sampling ends by "
        f"{time.strftime('%H:%M', time.gmtime(t_end))} UTC")
    save_meta(plan=plan)
    stop = "planned candidates done"
    while start + cs <= n_planned:
        if not may_start(now(), t_end, times, tp["s_per_problem"], cfg):
            stop = f"chunk {start} would end after the sampling deadline"
            break
        rows, dt = run(probs[start:start + cs])
        _write_atomic(F.chunk_path(MODEL, start, cfg, sd), _jsonl(rows))
        times.append(dt)
        acc = np.mean([bool(r["correct"]) for r in rows])
        start += cs
        log(f"  chunk {start - cs}: {dt:.0f} s ({dt / cs:.1f} s / problem), per-sample accuracy {acc:.3f}; "
            f"{start} / {n_planned} candidates")
        save_meta(chunks=len(times), seconds=float(sum(times)), last_chunk_end=start)
    log(f"stop: {stop}; {start} candidates on disk")
    save_meta(chunks=len(times), seconds=float(sum(times)), last_chunk_end=start, stop=stop,
              ended=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    return 0


def pack(out: Path, cfg: dict | None = None) -> Path:
    """scale/7b/ (chunks, meta, throughput, plan, logs) and MANIFEST.json (sha256 per file) in one zip."""
    cfg = cfg or _config()
    root = Path(out)
    files = sorted(p for p in scale_dir(out).rglob("*") if p.is_file() and not p.name.endswith(".tmp")
                   and p.name != "MANIFEST.json")
    data = {p.relative_to(root).as_posix(): p.read_bytes() for p in files}   # read once: the manifest is the zip's
    manifest = {name: _sha256(b) for name, b in data.items()}
    data["scale/7b/MANIFEST.json"] = json.dumps(manifest, indent=1).encode()
    (scale_dir(out) / "MANIFEST.json").write_bytes(data["scale/7b/MANIFEST.json"])
    z = root / cfg["scale"]["kaggle"]["zip_name"]
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, b in data.items():
            zf.writestr(name, b)
    print(f"pack: {z} ({z.stat().st_size / 1e6:.1f} MB, {len(files)} files), sha256 {_file_sha256(z)}", flush=True)
    return z


# ------------------------------------------------------------------------------------------- local: the ingest
def safe_extract(zip_path: Path, dest: Path, cfg: dict) -> dict:
    """The zip is untrusted data: only _ALLOWED regular files no larger than max_file_mb, written by name into an
    empty `dest`; MANIFEST.json must list exactly those files with matching sha256. Returns the manifest."""
    limit = cfg["scale"]["kaggle"]["max_file_mb"] * 1e6
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise SystemExit(f"{dest} is not empty")
    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        names = [i.filename for i in infos]
        if len(set(names)) != len(names):
            raise SystemExit("duplicate names in the zip")
        for i in infos:
            if i.filename != "scale/7b/MANIFEST.json" and not _ALLOWED.fullmatch(i.filename):
                raise SystemExit(f"unexpected file in the zip: {i.filename!r}")
            if (i.external_attr >> 16) & 0o170000 not in (0, 0o100000):
                raise SystemExit(f"not a regular file: {i.filename!r}")
            if i.file_size > limit:
                raise SystemExit(f"{i.filename!r} is larger than {limit / 1e6:.0f} MB")
        data = {i.filename: zf.read(i) for i in infos}
    manifest = json.loads(data.pop("scale/7b/MANIFEST.json", b"null"))
    if not isinstance(manifest, dict) or set(manifest) != set(data):
        raise SystemExit("MANIFEST.json does not list exactly the zip's files")
    for name, b in data.items():
        if _sha256(b) != manifest[name]:
            raise SystemExit(f"sha256 mismatch: {name}")
        p = dest / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    return manifest


def ingest(zip_path: Path, cfg: dict | None = None, tok=None, staging: Path | None = None,
           dest: Path | None = None) -> dict:
    """The downloaded zip into results/followup/scale/7b/: safe extraction and the manifest; the chunk prefix
    (followup_scale.sampled_prefix); every segment's revision and decoding; every row's problem equal to the frozen
    pool's; every prompt re-rendered with the local tokenizer (the 4-bit copy's, which the readouts use) to the same
    sha256 and length; every row re-graded with the local parser (authoritative, changes counted). Idempotent."""
    cfg = cfg or _config()
    sc, kc = cfg["scale"], cfg["scale"]["kaggle"]
    k = sc["generation"]["k"]
    zsha = _file_sha256(Path(zip_path))
    staging = staging or REPO_ROOT / ".cache" / "kaggle_import" / zsha[:16]
    if staging.exists():
        shutil.rmtree(staging)
    safe_extract(Path(zip_path), staging, cfg)
    sd = samples_dir(staging)
    rows = F.sampled_prefix(MODEL, cfg, sd)
    if not rows:
        raise SystemExit("the zip holds no complete chunk")
    meta = json.loads((sd / "meta.json").read_text())
    for seg in meta["segments"]:
        if seg["repo_id"] != kc["repo_id"] or seg["revision"] != kc["revision"] or seg["decoding"] != decoding(cfg):
            raise SystemExit(f"a segment was sampled with other settings: {seg}")
    pool = F.pool(cfg)
    if tok is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(REPO_ROOT / sc["models"][MODEL]["dir"]))
    checked: dict = {}
    changed = 0
    for r in rows:
        i = r["index"]
        if r["problem"] != pool[i] or r["key"] != [pool[i]["family"], pool[i]["depth"], pool[i]["seed"]]:
            raise SystemExit(f"row of candidate {i}: the problem differs from the frozen pool")
        if r["sample_seed"] != sample_seed(i, r["sample"], cfg):
            raise SystemExit(f"row of candidate {i}, sample {r['sample']}: unexpected seed")
        if i not in checked:
            text, ids = render_prompt(tok, pool[i]["text"], cfg)
            checked[i] = (_sha256(text.encode()), len(ids))
        if (r["prompt_sha256"], r["prompt_tokens"]) != checked[i]:
            raise SystemExit(f"candidate {i}: the local tokenizer renders another prompt")
        g = F.grade(pool[i], r["completion"], r["truncated"], sc["tolerance"])
        old = {key: r.get(key) for key in g}
        for key in ("value", "reason", "correct", "rel_error", "brace", "unit_text"):
            r.pop(key, None)
        r.update(g)
        r["breakdown"] = {**r["breakdown"], "reason": g["reason"], "correct": g["correct"]}
        changed += old != g
    by_index: dict = {}
    for r in rows:
        by_index.setdefault(r["index"], set()).add(r["sample"])
    if any(v != set(range(k)) for v in by_index.values()):
        raise SystemExit("a candidate lacks some of its k samples")
    dest = dest or F.out_dir(MODEL, cfg)
    dsd = dest / "samples"
    dsd.mkdir(parents=True, exist_ok=True)
    cs = sc["chunk_size"]
    written = 0
    for start in range(0, len(rows) // k, cs):
        text = _jsonl(rows[start * k:(start + cs) * k])
        f = F.chunk_path(MODEL, start, cfg, dsd)
        if f.exists():
            if f.read_text() != text:
                raise SystemExit(f"{f} exists with other content; move it away to re-ingest")
            continue
        _write_atomic(f, text)
        written += 1
    F.sampled_prefix(MODEL, cfg, dsd)  # the destination is still one prefix
    shutil.copyfile(sd / "meta.json", dsd / "meta.json")
    kd = dest / "kaggle"
    kd.mkdir(exist_ok=True)
    for name in ("throughput.json", "plan.json"):
        if (scale_dir(staging) / name).exists():
            shutil.copyfile(scale_dir(staging) / name, kd / name)
    if (scale_dir(staging) / "logs").is_dir():
        shutil.copytree(scale_dir(staging) / "logs", kd / "logs", dirs_exist_ok=True)
    res = {"decision": "D157: the Kaggle samples ingested (exploratory)", "zip_sha256": zsha,
           "n_candidates": len(rows) // k, "n_rows": len(rows), "chunks_written": written,
           "prompts_checked": len(checked), "regraded_changed_rows": changed,
           "answered_share": float(np.mean([r["value"] is not None for r in rows])),
           "accuracy_per_sample": float(np.mean([bool(r["correct"]) for r in rows])),
           "segments": [{key: s.get(key) for key in ("session_start", "commit", "packages", "gpus", "enforce_eager",
                                                      "env", "chunks", "stop")} for s in meta["segments"]]}
    (kd / "ingest.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["download", "restore", "sample", "pack", "ingest"])
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--out", type=Path, help="Kaggle: the output root (/kaggle/working)")
    ap.add_argument("--src", type=Path, help="restore: where a previous session's output is mounted")
    ap.add_argument("--session-start", type=float, help="sample: epoch seconds when the session started")
    ap.add_argument("--enforce-eager", action="store_true", help="sample: no CUDA graphs (the notebook's retry)")
    ap.add_argument("--zip", type=Path, help="ingest: the downloaded zip")
    a = ap.parse_args()
    cfg = _config()
    if a.action == "download":
        print(json.dumps(download(a.model_dir, cfg), indent=1))
    elif a.action == "restore":
        restore(a.out, a.src, cfg)
    elif a.action == "pack":
        pack(a.out, cfg)
    elif a.action == "ingest":
        ingest(a.zip, cfg)
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(a.model_dir))
        env = _environment(a.model_dir)
        engine_error: list[BaseException] = []

        def factory():
            try:
                return VLLMEngine(a.model_dir, cfg, a.enforce_eager)
            except Exception as exc:
                engine_error.append(exc)
                raise

        settings = {**env, "enforce_eager": a.enforce_eager, "env": {name: os.environ.get(name) for name in (
            "NCCL_P2P_DISABLE", "VLLM_USE_FLASHINFER_SAMPLER")}}
        try:
            rc = sample(factory, tok, a.out, a.session_start, cfg, log=lambda m: print(m, flush=True),
                        settings=settings)
        except Exception:
            if engine_error:
                import traceback

                traceback.print_exc()
                sys.exit(EXIT_ENGINE)
            raise
        sys.exit(rc)


if __name__ == "__main__":
    main()

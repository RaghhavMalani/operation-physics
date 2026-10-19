"""D78: the pinned 4-bit base for all Phase 12 work (SFT, RL, eval), resolved to one exact snapshot.

A hub name is never passed to a loader. Unsloth drops the revision of a hub load when it remaps the name to its own
mirror, and loads an adapter's base with revision=None, i.e. whatever `main` is (D70). So the base is the local
snapshot dir of the pinned commit, named repo-relative (as every adapter's `base_model_name_or_path` already is):
- `pinned_base_dir` resolves (repo_id, revision) to that dir in the repo-local HF cache (offline unless asked);
- `verify_base_dir` checks the dir's commit and the content hash of its weights before any load;
- `assert_loaded_base` checks after a load that the model (or a PEFT model's base) came from that dir;
- `check_model_source` applies both to a training/eval `model.id`, which is the base dir or an adapter dir.
The content hash is `fingerprint._dir_content_hash`, so it equals the `base_identity` in every eval tape (R2-1).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from physcal.config import REPO_ROOT, load_config

HF_HUB = REPO_ROOT / ".cache" / "huggingface" / "hub"


class BaseNotPinned(RuntimeError):
    """The base that would be (or was) loaded is not the pinned snapshot."""


def base_config(name: str = "base_4bit") -> dict:
    return load_config(name)


def _rel(p: Path) -> str:
    p = Path(p).absolute()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def pinned_base_dir(cfg: dict | None = None, download: bool = False, cache_dir: Path | None = None) -> str:
    """The pinned commit's snapshot dir, repo-relative. Offline by default: a missing snapshot raises unless
    `download` is set (then exactly that commit is fetched, never `main`)."""
    from huggingface_hub import snapshot_download

    cfg = cfg or base_config()
    path = snapshot_download(repo_id=cfg["repo_id"], revision=cfg["revision"], cache_dir=str(cache_dir or HF_HUB),
                             local_files_only=not download)
    verify_base_dir(path, cfg)
    return _rel(Path(path))


def verify_base_dir(path: str | Path, cfg: dict | None = None) -> dict:
    """Raise BaseNotPinned unless `path` is the pinned commit's snapshot with the pinned weights."""
    from physcal.eval.fingerprint import _dir_content_hash

    cfg = cfg or base_config()
    path = Path(path)
    if path.name != cfg["revision"]:
        raise BaseNotPinned(f"{path} is not the snapshot of the pinned commit {cfg['revision']}")
    h = _dir_content_hash(path)
    if h != cfg["content_sha256"]:
        raise BaseNotPinned(f"{path}: weights hash {h}, pinned {cfg['content_sha256']}")
    return {"dir": _rel(path), "repo_id": cfg["repo_id"], "revision": cfg["revision"], "content_sha256": h}


def adapter_base(adapter_dir: str | Path) -> str | None:
    """The base an adapter dir names, or None if `adapter_dir` is not an adapter."""
    f = Path(adapter_dir) / "adapter_config.json"
    return json.loads(f.read_text())["base_model_name_or_path"] if f.exists() else None


def _same_dir(a: str | Path, b: str | Path) -> bool:
    return Path(a).resolve() == Path(b).resolve()  # relative names resolve against the cwd, as Unsloth's do


def check_model_source(model_id: str, cfg: dict | None = None) -> dict:
    """Before loading `model_id` (the base dir itself, or an adapter dir trained on it): the base it resolves to
    must be the pinned snapshot. Returns the verified base record (plus the adapter dir, if any)."""
    cfg = cfg or base_config()
    pinned = pinned_base_dir(cfg)
    base = adapter_base(model_id)
    named = base if base is not None else model_id
    if not _same_dir(named, pinned):
        what = f"adapter {model_id} names base {named!r}" if base is not None else f"model {model_id!r}"
        raise BaseNotPinned(f"{what} (cwd {os.getcwd()}); the pinned base is {pinned} (D78)")
    return {**verify_base_dir(pinned, cfg), "adapter": str(model_id) if base is not None else None}


def assert_loaded_base(model, cfg: dict | None = None) -> str:
    """After a load: the model's (or a PEFT model's base's) `_name_or_path` is the pinned snapshot dir."""
    cfg = cfg or base_config()
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    loaded = str(getattr(base.config, "_name_or_path", ""))
    pinned = pinned_base_dir(cfg)
    if not loaded or not _same_dir(loaded, pinned):
        raise BaseNotPinned(f"the loaded base is {loaded!r}, not the pinned {pinned} (D78)")
    return loaded


def resolve_model(m: dict) -> tuple[str, str | None, dict | None]:
    """A trainer's `model` entry -> (model_name, revision, pin) for `FastLanguageModel.from_pretrained`.
    With `pinned_base: <config>` (D78): `id` is an adapter dir trained on the pinned base, or absent for the base
    itself; `revision` must be absent (the commit is the config's); the source is checked before loading and `pin`
    is its verified record. Without it, the legacy pair (id, revision) is returned unchanged and pin is None."""
    if not m.get("pinned_base"):
        return m["id"], m.get("revision"), None
    if m.get("revision"):
        raise ValueError("a pinned_base model takes no revision: the pinned config fixes the commit (D78)")
    cfg = base_config(m["pinned_base"])
    model_id = m.get("id") or pinned_base_dir(cfg)
    return model_id, None, {**check_model_source(model_id, cfg), "config": m["pinned_base"]}

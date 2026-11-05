"""D171's shared stage runner for Kaggle-first experiments (EXPLORATORY, follow-up project; not the paper).

A run lives in one directory (`Run.root`), which is the notebook's resume tier. Every stage writes atomic chunks, each
logged with its time and sha256 in a per-worker ledger, then a DONE.json. A re-run skips what exists and refuses another
commit or config (`pins`). No chunk starts unless its projected end is before session start + `hours` (D157's rule).
`restore` brings a previous version's output back, checking every hash; `write_zip` / `read_zip` make and check the
results zip (a sha256 manifest and an allow-list). Nothing here knows about any one experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import zipfile
from pathlib import Path

PINS, DONE, MANIFEST = "run.json", "DONE.json", "MANIFEST.json"


class StageStop(RuntimeError):
    """A stop rule fired (non-finite value, format): the stage stops, and what is saved stays."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class Run:
    def __init__(self, root: Path, pins: dict, session_start: float, hours: float, margin: float, worker: str = "main",
                 now=time.time):
        self.root, self.pins, self.worker, self.now, self.margin = Path(root), pins, worker, now, margin
        self.t_end = session_start + hours * 3600
        self.times: dict[str, list[float]] = {}
        self.root.mkdir(parents=True, exist_ok=True)
        f = self.root / PINS
        if f.exists():
            old = json.loads(f.read_text())
            if old != pins:
                diff = sorted(k for k in set(old) | set(pins) if old.get(k) != pins.get(k))
                raise SystemExit(f"{f}: this run was started with other pins ({diff}); refusing to mix them")
        else:
            write_atomic(f, json.dumps(pins, indent=1, sort_keys=True))

    def dir(self, stage: str) -> Path:
        d = self.root / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def path(self, stage: str, key: str) -> Path:
        return self.dir(stage) / key

    def has(self, stage: str, key: str) -> bool:
        return self.path(stage, key).exists()

    def done(self, stage: str) -> bool:
        return (self.root / stage / DONE).exists()

    def may_start(self, stage: str, est_s: float) -> bool:
        """The chunk's projected end (margin x this session's mean chunk time, or `est_s` before the first)."""
        t = self.times.get(stage)
        return self.now() + self.margin * (sum(t) / len(t) if t else est_s) <= self.t_end

    def save(self, stage: str, key: str, write) -> None:
        """write(tmp_path) makes the chunk; it appears under its name whole, or not at all."""
        t0, path = self.now(), self.path(stage, key)
        tmp = path.with_name(path.name + ".tmp")
        write(tmp)
        os.replace(tmp, path)
        dt = self.now() - t0
        self.times.setdefault(stage, []).append(dt)
        with (self.dir(stage) / f"ledger_{self.worker}.jsonl").open("a") as fh:
            fh.write(json.dumps({"key": key, "seconds": dt, "sha256": file_sha256(path), "at": self.now()}) + "\n")

    def chunks(self, stage: str, keys: list[str], write_one, est_s: float, log=print) -> bool:
        """write_one(key, tmp_path) for each missing key while the deadline allows. True if none is left to do."""
        for key in keys:
            if self.has(stage, key):
                continue
            if not self.may_start(stage, est_s):
                log(f"{stage}: {key} would end after the deadline; stopping here")
                return False
            self.save(stage, key, lambda tmp, k=key: write_one(k, tmp))
            log(f"{stage}: {key} in {self.times[stage][-1]:.0f} s")
        return True

    def ledger(self, stage: str) -> dict[str, dict]:
        """Every chunk the stage's ledgers record (the last line per key)."""
        out: dict = {}
        for f in sorted((self.root / stage).glob("ledger_*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    out[r["key"]] = r
        return out

    def finish(self, stage: str, keys: list[str]) -> dict:
        """The stage's done marker. Every key must exist and match its ledger."""
        led = self.ledger(stage)
        missing = [k for k in keys if not self.has(stage, k)]
        if missing:
            raise SystemExit(f"{stage}: cannot finish, {len(missing)} chunks missing: {missing[:3]}")
        bad = [k for k in keys if k in led and file_sha256(self.path(stage, k)) != led[k]["sha256"]]
        if bad:
            raise SystemExit(f"{stage}: chunks differ from their ledger: {bad[:3]}")
        res = {"stage": stage, "chunks": {k: {"sha256": file_sha256(self.path(stage, k)),
                                              "seconds": led.get(k, {}).get("seconds")} for k in keys},
               "seconds": sum(led[k]["seconds"] for k in keys if k in led)}
        write_atomic(self.path(stage, DONE), json.dumps(res, indent=1))
        return res

    def verify(self) -> dict:
        """Every chunk any ledger or DONE.json records against its file; raises on a difference."""
        n = 0
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            want = {k: r["sha256"] for k, r in self.ledger(d.name).items()}
            if (d / DONE).exists():
                want |= {k: r["sha256"] for k, r in json.loads((d / DONE).read_text())["chunks"].items()}
            for k, h in want.items():
                f = d / k
                if not f.exists() or file_sha256(f) != h:
                    raise SystemExit(f"{f}: missing or differs from its record")
                n += 1
        return {"chunks_checked": n}


def restore(root: Path, src: Path, log=print) -> dict:
    """A previous version's output (mounted anywhere under `src`) copied into `root`; .tmp files are left behind. Hashes
    are checked by the next Run.verify(). Refuses a second, different source."""
    root, src = Path(root), Path(src)
    pins = [p for p in src.rglob(PINS) if p.parent.name == root.name] if src.exists() else []
    if not pins:
        log(f"restore: no previous run under {src}")
        return {"restored": 0}
    if len(pins) > 1:
        raise SystemExit(f"restore: {len(pins)} runs under {src}; attach one version only: {pins}")
    base, n = pins[0].parent, 0
    for f in sorted(base.rglob("*")):
        if f.is_file() and not f.name.endswith(".tmp"):
            dest = root / f.relative_to(base)
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                n += 1
    log(f"restore: {n} files from {base}")
    return {"restored": n, "from": str(base)}


def write_zip(zip_path: Path, files: dict[str, bytes], max_mb: float) -> dict:
    """`files` (name -> bytes) and a MANIFEST.json of their sha256s in one zip, refused above max_mb."""
    manifest = {name: hashlib.sha256(b).hexdigest() for name, b in sorted(files.items())}
    tmp = Path(zip_path).with_name(Path(zip_path).name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, b in sorted(files.items()):
            zf.writestr(name, b)
        zf.writestr(MANIFEST, json.dumps(manifest, indent=1))
    mb = tmp.stat().st_size / 1e6
    if mb > max_mb:
        tmp.unlink()
        raise SystemExit(f"{zip_path}: {mb:.1f} MB is above the {max_mb} MB limit")
    os.replace(tmp, zip_path)
    return {"mb": mb, "files": len(files), "sha256": file_sha256(Path(zip_path))}


def read_zip(zip_path: Path, allow: re.Pattern, max_file_mb: float) -> dict[str, bytes]:
    """The zip's files after its checks: only allow-listed names, none oversized, every hash as its manifest says."""
    out: dict[str, bytes] = {}
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if MANIFEST not in names:
            raise SystemExit(f"{zip_path}: no {MANIFEST}")
        manifest = json.loads(zf.read(MANIFEST))
        for info in zf.infolist():
            if info.filename == MANIFEST:
                continue
            if not allow.fullmatch(info.filename):
                raise SystemExit(f"{zip_path}: {info.filename!r} is not on the allow-list")
            if info.file_size > max_file_mb * 1e6:
                raise SystemExit(f"{zip_path}: {info.filename} is above {max_file_mb} MB")
            b = zf.read(info.filename)
            if hashlib.sha256(b).hexdigest() != manifest.get(info.filename):
                raise SystemExit(f"{zip_path}: {info.filename} does not match the manifest")
            out[info.filename] = b
    if set(out) != set(manifest):
        raise SystemExit(f"{zip_path}: the manifest and the zip list different files")
    return out

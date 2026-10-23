"""Read-only integrity checks on files written around a crash (DECISIONS.md D111; first used after the 2026-10-23
disk-full crash): NUL runs and truncation in text / logs, every JSONL line parses, gzip CRCs (and equality with an
uncompressed sibling), safetensors headers / offsets / all-zero or non-finite tensors, torch zip CRCs, PNG chunk CRCs,
and tracked files that git calls unchanged against their HEAD blobs. Prints a JSON report; exit 1 if any problem.

    find . \( -path ./.venv -o -path ./.git -o -path ./.cache \) -prune -o -type f -newermt 2026-10-22T17:50:00Z \
        -printf '%s\t%p\n' > files.tsv
    .venv/bin/python -I scripts/integrity_check.py "$PWD" files.tsv [EXTRA_FILE ...]
"""
import gzip
import json
import struct
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np

TEXT = {".md", ".log", ".json", ".jsonl", ".py", ".yaml", ".yml", ".sh", ".txt", ".csv", ".jinja", ".pid", ".tsv"}
BLOCK = 4096
repo = Path(sys.argv[1])
files = [repo / line.split("\t", 1)[1].strip() for line in open(sys.argv[2]) if line.strip()]
files += [Path(p) for p in sys.argv[3:]]
problems, notes = [], []


def zero_blocks(b: bytes) -> int:
    n = len(b) // BLOCK
    z = bytes(BLOCK)
    return sum(1 for i in range(n) if b[i * BLOCK:(i + 1) * BLOCK] == z)


def longest_nul_run(b: bytes) -> int:
    best = cur = 0
    for ch in b:
        if ch == 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def check_jsonl_lines(text: str, name: str) -> int:
    bad, n = [], 0
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        n += 1
        try:
            json.loads(line)
        except Exception as e:
            bad.append((i, str(e)[:60]))
    if bad:
        problems.append(f"{name}: {len(bad)} unparseable JSONL lines, first {bad[:3]}")
    return n


def check_text(p: Path, b: bytes):
    nul = b.count(0)
    if nul:
        problems.append(f"{p}: {nul} NUL bytes (longest run {longest_nul_run(b)})")
    try:
        text = b.decode("utf-8")
    except UnicodeDecodeError as e:
        problems.append(f"{p}: not valid UTF-8 ({e})")
        return
    if p.suffix in (".jsonl", ".log") and b and not b.endswith(b"\n"):
        notes.append(f"{p}: last line has no newline (possible truncation)")
    if p.suffix == ".json":
        try:
            json.loads(text)
        except Exception as e:
            problems.append(f"{p}: JSON does not parse ({str(e)[:80]})")
    if p.suffix == ".jsonl":
        n = check_jsonl_lines(text, str(p))
        notes.append(f"{p}: {n} JSONL lines parse")


def check_gz(p: Path):
    try:
        raw = gzip.decompress(p.read_bytes())
    except Exception as e:
        problems.append(f"{p}: gzip fails ({e})")
        return
    if p.name.endswith(".jsonl.gz"):
        n = check_jsonl_lines(raw.decode("utf-8"), str(p))
        sib = p.with_suffix("")
        same = "" if not sib.exists() else (" = uncompressed sibling" if sib.read_bytes() == raw else
                                            " DIFFERS from the uncompressed sibling")
        (notes if "DIFFERS" not in same else problems).append(f"{p}: gzip CRC ok, {n} JSONL lines parse{same}")


DT = {"F32": (np.float32, 4), "F16": (np.float16, 2), "BF16": (np.uint16, 2), "I64": (np.int64, 8), "I32": (np.int32, 4),
      "U8": (np.uint8, 1), "I8": (np.int8, 1), "BOOL": (np.uint8, 1), "F64": (np.float64, 8)}


def check_safetensors(p: Path, b: bytes):
    (hl,) = struct.unpack("<Q", b[:8])
    header = json.loads(b[8:8 + hl])
    data = b[8 + hl:]
    ends, zero_t, nonfinite = [], [], []
    for name, t in header.items():
        if name == "__metadata__":
            continue
        s, e = t["data_offsets"]
        ends.append((s, e))
        dt, size = DT[t["dtype"]]
        n = int(np.prod(t["shape"])) if t["shape"] else 1
        if e - s != n * size:
            problems.append(f"{p}: tensor {name} has {e - s} bytes for {n} x {size}")
            continue
        arr = np.frombuffer(data[s:e], dtype=dt)
        if not arr.any():
            zero_t.append(name)
        if t["dtype"] == "BF16":
            if np.any((arr & 0x7F80) == 0x7F80):
                nonfinite.append(name)
        elif arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
            nonfinite.append(name)
    ends.sort()
    contiguous = all(a[1] == b_[0] for a, b_ in zip(ends, ends[1:])) and ends[0][0] == 0
    if not contiguous or ends[-1][1] != len(data):
        problems.append(f"{p}: data offsets not contiguous / end {ends[-1][1]} != data {len(data)}")
    if zero_t:
        problems.append(f"{p}: {len(zero_t)} all-zero tensors, e.g. {zero_t[:3]}")
    if nonfinite:
        problems.append(f"{p}: {len(nonfinite)} tensors with NaN/Inf, e.g. {nonfinite[:3]}")
    notes.append(f"{p}: safetensors ok structurally, {len(ends)} tensors, {len(zero_t)} all-zero, "
                 f"{zero_blocks(b)} zero 4 KiB blocks")


def check_png(p: Path, b: bytes):
    if b[:8] != b"\x89PNG\r\n\x1a\n":
        problems.append(f"{p}: bad PNG signature")
        return
    i, chunks, seen_end = 8, 0, False
    while i < len(b):
        if i + 8 > len(b):
            problems.append(f"{p}: truncated chunk header at {i}")
            return
        (ln,), typ = struct.unpack(">I", b[i:i + 4]), b[i + 4:i + 8]
        body = b[i + 8:i + 8 + ln]
        (crc,) = struct.unpack(">I", b[i + 8 + ln:i + 12 + ln]) if i + 12 + ln <= len(b) else (None,)
        if crc is None or zlib.crc32(typ + body) & 0xFFFFFFFF != crc:
            problems.append(f"{p}: CRC mismatch in chunk {typ!r} at {i}")
            return
        chunks += 1
        i += 12 + ln
        if typ == b"IEND":
            seen_end = True
            break
    if not seen_end:
        problems.append(f"{p}: no IEND")
    elif i != len(b):
        problems.append(f"{p}: {len(b) - i} bytes after IEND")
    else:
        notes.append(f"{p}: PNG ok, {chunks} chunks, all CRCs match")


def check_zip(p: Path, b: bytes):
    try:
        with zipfile.ZipFile(p) as z:
            bad = z.testzip()
        if bad:
            problems.append(f"{p}: zip member {bad} fails its CRC")
        else:
            notes.append(f"{p}: zip (torch) CRCs ok, {zero_blocks(b)} zero 4 KiB blocks")
    except zipfile.BadZipFile as e:
        problems.append(f"{p}: not a valid zip ({e})")


def tracked_blob_check(paths):
    """Tracked files that git status calls unchanged must hash to HEAD's blob (catches content damage behind an
    unchanged stat)."""
    modified = set(subprocess.run(["git", "-C", str(repo), "diff", "--name-only", "HEAD"], capture_output=True,
                                  text=True).stdout.split())
    n = 0
    for p in paths:
        rel = str(p.relative_to(repo)) if p.is_relative_to(repo) else None
        if rel is None or rel in modified:
            continue
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "-q", "--verify", f"HEAD:{rel}"], capture_output=True,
                              text=True).stdout.strip()
        if not head:
            continue
        now = subprocess.run(["git", "-C", str(repo), "hash-object", rel], capture_output=True, text=True).stdout.strip()
        n += 1
        if now != head:
            problems.append(f"{rel}: content differs from HEAD although git status shows it unchanged")
    notes.append(f"tracked, unchanged files checked against HEAD blobs: {n}")


for p in files:
    if not p.exists():
        problems.append(f"{p}: listed but missing")
        continue
    b = p.read_bytes()
    if not b:
        continue
    suf = p.suffix
    if p.name.endswith(".gz"):
        check_gz(p)
    elif suf in TEXT or p.name in ("README.md",):
        check_text(p, b)
    elif suf == ".safetensors":
        check_safetensors(p, b)
    elif suf == ".png":
        check_png(p, b)
    elif suf in (".pt", ".pth", ".bin"):
        check_zip(p, b)
    else:
        notes.append(f"{p}: {suf or 'no suffix'} not checked ({zero_blocks(b)} zero 4 KiB blocks)")
tracked_blob_check(files)
print(json.dumps({"n_files": len(files), "problems": problems, "notes": notes}, indent=1))
sys.exit(1 if problems else 0)

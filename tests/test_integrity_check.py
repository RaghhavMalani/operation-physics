"""D111: scripts/integrity_check.py finds each kind of damage it checks for, and passes intact files."""

import gzip
import json
import struct
import subprocess
import sys
import zlib

from physcal.config import REPO_ROOT


def _png(corrupt: bool = False) -> bytes:
    def chunk(typ, body):
        crc = zlib.crc32(typ + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + typ + body + struct.pack(">I", crc ^ (1 if corrupt and typ == b"IDAT" else 0))
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\x00\x7f")) + chunk(b"IEND", b"")


def _safetensors(values: list[float]) -> bytes:
    data = struct.pack(f"<{len(values)}f", *values)
    header = json.dumps({"w": {"dtype": "F32", "shape": [len(values)], "data_offsets": [0, len(data)]}}).encode()
    return struct.pack("<Q", len(header)) + header + data


def test_finds_damage_and_passes_intact_files(tmp_path):
    files = {
        "good.jsonl": b'{"a": 1}\n{"b": 2}\n', "bad.jsonl": b'{"a": 1}\n{"b": \n',
        "good.log": b"line\n", "nul.log": b"line\n" + bytes(64) + b"more\n",
        "good.png": _png(), "bad.png": _png(corrupt=True),
        "good.safetensors": _safetensors([0.5, -1.0]), "zero.safetensors": _safetensors([0.0, 0.0]),
        "same.jsonl": b'{"x": 1}\n', "same.jsonl.gz": gzip.compress(b'{"x": 1}\n'),
        "diff.jsonl": b'{"x": 1}\n', "diff.jsonl.gz": gzip.compress(b'{"x": 2}\n'),
        "good.json": b'{"k": [1, 2]}', "bad.json": b'{"k": [1, 2',
    }
    for name, blob in files.items():
        (tmp_path / name).write_bytes(blob)
    (tmp_path / "list.tsv").write_text("".join(f"{len(b)}\t./{n}\n" for n, b in files.items()))
    r = subprocess.run([sys.executable, "-I", str(REPO_ROOT / "scripts" / "integrity_check.py"), str(tmp_path),
                        str(tmp_path / "list.tsv")], capture_output=True, text=True, timeout=120)
    rep = json.loads(r.stdout)
    assert r.returncode == 1 and rep["n_files"] == len(files)
    flagged = {name for name in files if any(f"/{name}:" in p for p in rep["problems"])}
    assert flagged == {"bad.jsonl", "nul.log", "bad.png", "zero.safetensors", "diff.jsonl.gz", "bad.json"}, rep["problems"]

"""YAML config loading. All thresholds and weights live in `configs/*.yaml`."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


@lru_cache(maxsize=None)
def _load(path: str) -> str:
    return Path(path).read_text()


def load_config(name_or_path: str | Path) -> dict[str, Any]:
    """Load `configs/<name>.yaml`, or an explicit path. Returns a fresh dict each call."""
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    return yaml.safe_load(_load(str(path)))


def stable_hash(obj: Any, length: int = 8) -> str:
    """Process-independent short hash of a JSON-serialisable object."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:length]

"""Freeze the eval problem sets of the current generator version to disk.

    uv run python -m physcal.eval.export_sets [--config eval]

Writes data/eval_sets/<generator_version>/<split>.jsonl plus manifest.json. A test regenerates the sets and
compares them with these files, so any change to generator output without a version bump fails loudly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from physcal.config import REPO_ROOT, load_config
from physcal.env.generator import env_config, generator_version
from physcal.eval.run_eval import build_split

EVAL_SETS_DIR = REPO_ROOT / "data" / "eval_sets"


def serialize(problems) -> str:
    return "".join(json.dumps(p.to_dict(), sort_keys=True) + "\n" for p in problems)


def export(eval_cfg: dict, env_cfg: dict, root: Path = EVAL_SETS_DIR) -> dict:
    version = generator_version(env_cfg)
    out = root / version
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"generator_version": version, "splits": {}}
    for index, (name, split_cfg) in enumerate(eval_cfg["splits"].items()):
        problems = build_split(index, split_cfg, eval_cfg, env_cfg)
        if not problems:
            manifest["splits"][name] = {"skipped": f"no implemented families for '{split_cfg['families']}'"}
            continue
        blob = serialize(problems)
        (out / f"{name}.jsonl").write_text(blob)
        manifest["splits"][name] = {
            "n": len(problems),
            "sha256": hashlib.sha256(blob.encode()).hexdigest(),
            "seeds": [problems[0].seed, problems[-1].seed],
            "cells": sorted({f"{p.family}/{p.depth}" for p in problems}),
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="eval")
    args = ap.parse_args()
    manifest = export(load_config(args.config), env_config())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

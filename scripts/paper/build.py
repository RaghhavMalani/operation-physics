"""Build every generated input of the paper (paper/generated/): the prose values (macros.tex), the tables and the
figures, from committed results only, and record what each was built from (sources.json, SOURCES.md).

    uv run python scripts/paper/build.py          # then: cd paper && tectonic main.tex  (or latexmk -pdf main.tex)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figures  # noqa: E402
import macros  # noqa: E402
import tables  # noqa: E402
from paperlib import OUT, Sources, _git, ascii_only  # noqa: E402


def main() -> None:
    S = Sources()
    OUT.mkdir(parents=True, exist_ok=True)
    M = macros.build(S)
    (OUT / "macros.tex").write_text(ascii_only("macros", M.tex(S.header(macros.A, "scripts/paper/macros.py"))))
    tabs = tables.build(S)
    figs = figures.build(S)
    rec = {"commit": _git("rev-parse", "HEAD"), "artifacts": S.by_artifact, "tables": tabs, "figures": figs,
           "n_macros": len(M.values)}
    (OUT / "sources.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    lines = ["# Sources of the paper's generated values, tables and figures", "",
             f"Built from commit `{rec['commit'][:12]}` by `scripts/paper/build.py`; every source is tracked by git and "
             "had no uncommitted change. Blob hashes identify the exact file version.", ""]
    for art, srcs in sorted(S.by_artifact.items()):
        lines += [f"## {art}", ""] + [f"- `{p}` @ `{b}`" for p, b in sorted(srcs.items())] + [""]
    (OUT / "SOURCES.md").write_text("\n".join(lines))
    print(f"{len(M.values)} macros, {len(tabs)} tables, {len(figs)} figure files -> {OUT}")


if __name__ == "__main__":
    main()

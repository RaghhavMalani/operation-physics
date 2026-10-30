"""The paper's figures (paper/generated/figures/), from committed results only.

The headline figure is scripts/figures/headline.py's, unchanged: its inputs (headline.SRC) are checked and recorded
through paperlib.Sources first, then it is drawn into the paper's directory with its sidecar JSON.
"""

from __future__ import annotations

import sys

from paperlib import OUT, ROOT, Sources

sys.path.insert(0, str(ROOT / "scripts" / "figures"))


def build(S: Sources) -> list[str]:
    import os

    os.environ["SOURCE_DATE_EPOCH"] = "0"  # matplotlib then writes no build date: the PDF is reproducible
    import headline

    S.path("fig_headline", "scripts/figures/headline.py")
    for rel in headline.SRC.values():
        S.path("fig_headline", rel)
    paths = headline.draw(headline.data(), OUT / "figures")
    return [p.name for p in paths]

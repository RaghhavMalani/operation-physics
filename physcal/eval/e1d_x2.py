"""E1d on the X2 pairs (DECISIONS.md D102): greedy σ suffixes on X2's kept answers, μ fixed (GPU).

    uv run python -m physcal.eval.e1d_x2 suffix --which e1d     # E1d's final adapter (4-bit base, unmerged)
    uv run python -m physcal.eval.e1d_x2 suffix --which sft     # the SFT model's own suffix, no adapter (control)
    uv run python -m physcal.eval.e1d_x2_analysis analyze       # CPU: D102's tests

Each of X2's 4,116 kept answers is truncated after "±" by E1's `rows_from_records` (as X2's method 2 and X3's X2
eval) and the model writes a greedy σ suffix through E1's eval machinery, with E1d's own loader settings, D87's batch
and allocator cap, and Unsloth imported before transformers. Rows carry (key, source_sample).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from physcal.config import REPO_ROOT, load_config

OUT_NAMES = {"e1d": "e1d_final", "sft": "sft_suffix"}


def _config() -> dict:
    return load_config("e1d_x2")


def prefix_rows(tok, cfg: dict | None = None) -> tuple[list[dict], dict, str]:
    """X2's kept answers as E1 prefix rows (prompt v4), the excluded counts, and the source tapes' description."""
    from physcal.eval import x2_pairs
    from physcal.train import e1

    cfg = cfg or _config()
    e1cfg = e1._config()
    recs, source = x2_pairs.kept_records(load_config(cfg["x2_config"]))
    rows, excluded = e1.rows_from_records(recs, tok, e1cfg["prompt"], e1cfg.get("chat_template_kwargs", {}))
    return rows, excluded, source


def suffix(which: str, backend=None, out: Path | None = None) -> dict:
    if backend is None:
        import unsloth  # noqa: F401  (before transformers, as in training; D87)
    from transformers import AutoTokenizer

    from physcal.tape.tape import read_meta
    from physcal.train import e1

    cfg = _config()
    run = REPO_ROOT / cfg["run"]
    tok = AutoTokenizer.from_pretrained(str(REPO_ROOT / e1._config()["model_id"]))
    rows, excluded, source = prefix_rows(tok, cfg)
    model = read_meta(run / "rollouts.jsonl")["train_config"]["model"]   # E1d's loader settings, for both arms
    lora = str(run / cfg["adapter"]) if which == "e1d" else None
    out = out or REPO_ROOT / cfg["output_dir"] / OUT_NAMES[which]
    sc = cfg["suffix"]
    res = e1._evaluate(out, f"e1d-x2-{which}-suffix", "free", lora, model, backend, 0, "unsloth_4bit", rows=rows,
                       source_tapes=[source], batch_size=sc["batch_size"],
                       max_reserved_gib=sc["max_reserved_gib"] if backend is None else None)
    (out / "excluded.json").write_text(json.dumps(excluded))
    return {**res, "excluded": excluded}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["suffix", "rows"])
    ap.add_argument("--which", choices=sorted(OUT_NAMES), default="e1d")
    args = ap.parse_args()
    if args.action == "rows":  # CPU check: the prefix rows build
        from transformers import AutoTokenizer

        from physcal.train import e1

        rows, excluded, source = prefix_rows(AutoTokenizer.from_pretrained(str(REPO_ROOT / e1._config()["model_id"])))
        print(json.dumps({"n_rows": len(rows), "excluded": excluded, "source": source}))
        return
    print(json.dumps(suffix(args.which), indent=1))


if __name__ == "__main__":
    main()

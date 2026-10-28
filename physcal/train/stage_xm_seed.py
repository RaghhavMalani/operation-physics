"""Stage XM for another seed (DECISIONS.md D139; D137: 3 seeds only if the go rule passes). stage_xm.py is not edited,
because the running XM chain's stages import it: this module runs its functions with configs/stage_xm.yaml's seed
replaced, so the arm config, LoRA init, data order, output dir (<arm>/seed<s>) and run id follow the seed, as stage
R4's seeds did. Nothing else changes.

    uv run python -m physcal.train.stage_xm_seed train --arm dpos2 --seed 1 [--stop-at HH:MM] [--max-reserved-gib G]
"""

from __future__ import annotations

import argparse
import contextlib
import json

from physcal.train import stage_xm


@contextlib.contextmanager
def seed(s: int):
    """stage_xm's functions, with the config's seed set to `s`, inside the block."""
    base = stage_xm._config
    stage_xm._config = lambda: {**base(), "seed": s}
    try:
        yield
    finally:
        stage_xm._config = base


def train(arm: str, s: int, deadline: float | None = None, cap_gib: float | None = None) -> dict:
    with seed(s):
        return stage_xm.train(arm, deadline, None, cap_gib)


def main() -> None:
    from physcal.train.grpo import parse_deadline

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["train"])
    ap.add_argument("--arm", choices=stage_xm.ARMS, default="dpos2")
    ap.add_argument("--seed", type=int, required=True, help="a new seed (seed 0 is configs/stage_xm.yaml's own)")
    ap.add_argument("--stop-at", default=None, help="stop cleanly at this time (ISO or 'HH:MM Asia/Kolkata')")
    ap.add_argument("--max-reserved-gib", type=float, default=None, help="the launch cap, if lower than the config's")
    args = ap.parse_args()
    cfg = stage_xm._config()
    if args.seed == cfg["seed"]:
        raise ValueError(f"seed {args.seed} is configs/stage_xm.yaml's own; train it with stage_xm")
    if args.max_reserved_gib is not None and args.max_reserved_gib > cfg["max_reserved_gib"]:
        raise ValueError(f"--max-reserved-gib {args.max_reserved_gib} is above the config's {cfg['max_reserved_gib']}")
    s = train(args.arm, args.seed, parse_deadline(args.stop_at) if args.stop_at else None, args.max_reserved_gib)
    print(json.dumps({k: s.get(k) for k in ("steps", "train_loss", "seconds", "peak_vram")}, indent=1, default=str))


if __name__ == "__main__":
    main()

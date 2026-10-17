"""Adapter provenance and the merge guard (D67, D68): RL adapters are never merged into a base model.

Merging an RL LoRA into the bf16 SFT model rounds away 91–96% of its update (FINDINGS #15), so every RL adapter is
evaluated unmerged on its 4-bit training base. `check_merge_allowed` is called before any merge: it refuses an
adapter with any sign of RL training, and it refuses every other adapter unless the merge was asked for explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

# Adapter dirs sit at <run>/final_adapter or <run>/{checkpoints,permanent_checkpoints}/checkpoint-N, so the run dir
# (with its training tape's meta) is at most this many levels above the adapter dir.
PROVENANCE_LEVELS = 2


class MergeRefused(ValueError):
    """Raised instead of merging an adapter that must stay unmerged."""


def rl_adapter_evidence(adapter: str | Path) -> list[str]:
    """Reasons to treat `adapter` as an RL adapter (empty if none): the nearest run dir's training tape has a GRPO
    config, or the adapter's model card names GRPO (TRL writes it)."""
    adapter = Path(adapter)
    evidence = []
    for d in [adapter, *list(adapter.parents)[:PROVENANCE_LEVELS]]:
        meta = d / "rollouts.meta.json"
        if meta.exists():
            if "grpo" in json.loads(meta.read_text()).get("train_config", {}):
                evidence.append(f"{meta} holds a GRPO training config")
            break
    card = adapter / "README.md"
    if card.exists() and "grpo" in card.read_text(errors="replace").lower():
        evidence.append(f"{card} names GRPO")
    return evidence


def check_merge_allowed(adapter: str | Path, requested: bool) -> None:
    """Raise MergeRefused unless merging `adapter` is allowed: never for an RL adapter, and otherwise only when the
    merge was requested explicitly (`merge_adapter: true` in an eval's model config)."""
    evidence = rl_adapter_evidence(adapter)
    if evidence:
        raise MergeRefused(f"refusing to merge RL adapter {adapter}: {'; '.join(evidence)}. RL adapters are evaluated "
                           "unmerged on their 4-bit training base (loader unsloth_4bit; D67, D68, FINDINGS #15)")
    if not requested:
        raise MergeRefused(f"merging adapter {adapter} was not requested: set merge_adapter: true for a non-RL adapter, "
                           "or evaluate it unmerged (loader unsloth_4bit; D68)")

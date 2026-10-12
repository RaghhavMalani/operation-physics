"""Score one completion under every scoring rule, so a single generation pass answers questions about all of
them (e.g. "would a binary-reward GRPO group have any variance?").

The binary rule doesn't need a ± uncertainty, so its outcome is also the "σ-free" view of a completion.
"""

from __future__ import annotations

import copy
from typing import Any

from physcal.verify.reward import reward
from physcal.verify.scoring import RULES


def rule_configs(reward_cfg: dict) -> dict[str, dict]:
    """One reward config per scoring rule, otherwise identical to `reward_cfg`."""
    out = {}
    for rule in RULES:
        cfg = copy.deepcopy(reward_cfg)
        cfg["scoring"]["rule"] = rule
        out[rule] = cfg
    return out


def rule_outcomes(problem: Any, completion: str, configs: dict[str, dict]) -> dict[str, dict]:
    """{rule: {reason, score_final, reward, abs_rel_error}}. `reward` includes the step-credit term."""
    out = {}
    for rule, cfg in configs.items():
        r, bd = reward(problem, completion, cfg)
        out[rule] = {
            "reason": bd["reason"],
            "score_final": bd["score_final"],
            "reward": r,
            "abs_rel_error": bd.get("abs_rel_error") if bd["reason"] == "ok" else None,
        }
    return out

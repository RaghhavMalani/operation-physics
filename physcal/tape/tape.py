"""Rollout tape: one JSONL line per completion, plus a meta.json with the configs used.

`replay(path)` regenerates every problem from (generator_version, family, depth, seed), checks it matches the
logged problem, re-scores the logged completion with the logged reward config, and checks that the reward
matches exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from physcal.env.generator import Problem, generate_problem, generator_version, regenerate
from physcal.verify.reward import reward

TAPE_FIELDS = (
    "run_id", "step", "seed", "generator_version", "model_id", "lora_hash",
    "problem", "completion", "reward", "breakdown",
)


def meta_path(tape_path: Path) -> Path:
    return tape_path.with_suffix(".meta.json")


class TapeWriter:
    """Append-only JSONL writer. `meta` (configs, model info) is written once, next to the tape."""

    def __init__(self, path: str | Path, run_id: str, meta: dict[str, Any]):
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        meta_path(self.path).write_text(json.dumps({"run_id": run_id, **meta}, indent=2, sort_keys=True))
        self._fh = self.path.open("a")

    def log(
        self,
        step: int,
        problem: Problem,
        completion: str,
        reward_value: float,
        breakdown: dict,
        model_id: str,
        lora_hash: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "run_id": self.run_id,
            "step": step,
            "seed": problem.seed,
            "generator_version": problem.generator_version,
            "model_id": model_id,
            "lora_hash": lora_hash,
            "problem": problem.to_dict(),
            "completion": completion,
            "reward": reward_value,
            "breakdown": breakdown,
            **(extra or {}),
        }
        self._fh.write(json.dumps(record, allow_nan=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "TapeWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_tape(path: str | Path) -> Iterator[dict]:
    with Path(path).open() as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def read_meta(path: str | Path) -> dict:
    return json.loads(meta_path(Path(path)).read_text())


@dataclass
class ReplayReport:
    n: int
    reward_mismatches: list[int]
    problem_mismatches: list[int]

    @property
    def ok(self) -> bool:
        return self.n > 0 and not self.reward_mismatches and not self.problem_mismatches


def replay(path: str | Path) -> ReplayReport:
    """Regenerate and re-score every logged rollout; rewards must match exactly."""
    meta = read_meta(path)
    env_cfg, reward_cfg = meta["env_config"], meta["reward_config"]
    expected_version = generator_version(env_cfg)
    reward_bad, problem_bad, n = [], [], 0
    for i, rec in enumerate(read_tape(path)):
        n += 1
        if rec["generator_version"] != expected_version:
            raise ValueError(f"line {i}: generator {rec['generator_version']} != config {expected_version}")
        logged = rec["problem"]
        problem = regenerate(logged, env_cfg)
        if problem.to_dict() != logged:
            problem_bad.append(i)
        r, _ = reward(problem, rec["completion"], reward_cfg)
        if r != rec["reward"]:
            reward_bad.append(i)
    return ReplayReport(n=n, reward_mismatches=reward_bad, problem_mismatches=problem_bad)

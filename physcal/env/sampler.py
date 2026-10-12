"""Curriculum samplers over (family, depth) cells.

Interface (spec §5): `next_batch(n) -> list[cell]` and `observe(results)`, where results are
`(cell, [binary correctness of each completion in the group])`. The HMM sampler (phase 8) will implement the
same interface. EMA is the baseline: weight w = p·(1 − p) + floor, which peaks at p = 0.5, where GRPO groups
have reward variance.
"""

from __future__ import annotations

from typing import Iterable, Protocol

import numpy as np

Cell = tuple[str, int]


class Sampler(Protocol):
    def next_batch(self, n: int) -> list[Cell]: ...

    def observe(self, results: Iterable[tuple[Cell, list[bool]]]) -> None: ...

    def state(self) -> dict: ...


class UniformSampler:
    def __init__(self, cells: list[Cell], seed: int):
        self.cells = list(cells)
        self.rng = np.random.default_rng(seed)

    def next_batch(self, n: int) -> list[Cell]:
        return [self.cells[i] for i in self.rng.integers(len(self.cells), size=n)]

    def observe(self, results) -> None:
        pass

    def state(self) -> dict:
        return {"kind": "uniform"}


class EMASampler:
    """EMA pass rate per cell, sampling weight p(1-p) + floor over unlocked cells.

    Curriculum gate (config): only `start_depths` are open at first; depth d+1 of a family opens once its depth d
    has an EMA pass rate >= `unlock_pass_rate` after at least `min_observations` groups.
    """

    def __init__(self, cells: list[Cell], cfg: dict, seed: int):
        self.cells = sorted(cells)
        self.alpha = cfg["alpha"]
        self.floor = cfg["weight_floor"]
        self.unlock_pass_rate = cfg["unlock_pass_rate"]
        self.min_observations = cfg["min_observations"]
        self.p = {c: cfg["prior_pass_rate"] for c in self.cells}
        self.n = {c: 0 for c in self.cells}
        self.unlocked = {c for c in self.cells if c[1] in cfg["start_depths"]}
        self.unlock_log: list[tuple[int, Cell]] = []
        self.observations = 0
        self.rng = np.random.default_rng(seed)

    def weights(self) -> dict[Cell, float]:
        return {c: self.p[c] * (1.0 - self.p[c]) + self.floor for c in self.cells if c in self.unlocked}

    def next_batch(self, n: int) -> list[Cell]:
        w = self.weights()
        cells = sorted(w)
        probs = np.array([w[c] for c in cells])
        return [cells[i] for i in self.rng.choice(len(cells), size=n, p=probs / probs.sum())]

    def observe(self, results) -> None:
        for cell, correct in results:
            if not correct:
                continue
            self.p[cell] = (1.0 - self.alpha) * self.p[cell] + self.alpha * (sum(correct) / len(correct))
            self.n[cell] += 1
            self.observations += 1
        for family, depth in sorted(self.unlocked):
            nxt = (family, depth + 1)
            if nxt in self.p and nxt not in self.unlocked and self.n[(family, depth)] >= self.min_observations \
                    and self.p[(family, depth)] >= self.unlock_pass_rate:
                self.unlocked.add(nxt)
                self.unlock_log.append((self.observations, nxt))

    def state(self) -> dict:
        return {
            "kind": "ema",
            "p": {f"{f}/{d}": round(v, 4) for (f, d), v in self.p.items()},
            "n": {f"{f}/{d}": v for (f, d), v in self.n.items()},
            "unlocked": sorted(f"{f}/{d}" for f, d in self.unlocked),
            "unlock_log": [(obs, f"{f}/{d}") for obs, (f, d) in self.unlock_log],
        }

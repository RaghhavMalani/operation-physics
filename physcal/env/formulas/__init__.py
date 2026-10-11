"""Registry of formula families. Each family module exposes `build() -> Family`."""

from __future__ import annotations

from functools import lru_cache

from physcal.env.formulas import dynamics, energy, kinematics
from physcal.env.graph import Family

_BUILDERS = {
    "kinematics": kinematics.build,
    "dynamics": dynamics.build,
    "energy": energy.build,
}

FAMILY_NAMES: tuple[str, ...] = tuple(_BUILDERS)


@lru_cache(maxsize=None)
def get_family(name: str) -> Family:
    if name not in _BUILDERS:
        raise KeyError(f"unknown family {name!r}; registered: {FAMILY_NAMES}")
    return _BUILDERS[name]()

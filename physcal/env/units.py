"""Shared pint registry and unit helpers. Every module must use this single registry."""

from __future__ import annotations

import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

DIMENSIONLESS = ""

# Display-unit menus, keyed by SI unit. Every string must parse in pint.
UNIT_MENUS: dict[str, tuple[str, ...]] = {
    "m": ("m", "km", "cm", "mm"),
    "s": ("s", "ms", "min", "h"),
    "kg": ("kg", "g"),
    "m/s": ("m/s", "km/h", "cm/s"),
    "m/s^2": ("m/s^2", "cm/s^2"),
    "N": ("N", "kN", "mN"),
    "J": ("J", "kJ", "mJ", "MJ"),
    "kg m/s": ("kg m/s", "N s"),
    "N/m": ("N/m", "kN/m", "N/cm"),
    "Pa": ("Pa", "kPa"),
    DIMENSIONLESS: (DIMENSIONLESS,),
}


def quantity(value: float, unit: str) -> pint.Quantity:
    return Q_(value, unit or "dimensionless")


def to_si(value: float, unit: str) -> float:
    """Magnitude of `value [unit]` in SI base units."""
    return float(quantity(value, unit).to_base_units().magnitude)


def from_si(value_si: float, si_unit: str, unit: str) -> float:
    """Convert an SI magnitude (whose SI unit is `si_unit`) to `unit`."""
    return float(quantity(value_si, si_unit).to(unit or "dimensionless").magnitude)


def dimensionality(unit: str):
    return quantity(1.0, unit).dimensionality


def same_dimension(unit_a: str, unit_b: str) -> bool:
    return dimensionality(unit_a) == dimensionality(unit_b)

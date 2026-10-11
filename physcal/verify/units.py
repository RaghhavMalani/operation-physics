"""Pint units gate: convert a reported value to SI, or fail with `unit_mismatch`."""

from __future__ import annotations

from physcal.env.units import quantity, ureg


class UnitMismatch(ValueError):
    """Wrong dimensions, or a unit pint cannot parse."""


def parse_unit(unit: str):
    if not unit:
        return ureg.dimensionless
    try:
        return ureg.parse_units(unit)
    except Exception as exc:  # pint raises several exception types for bad input
        raise UnitMismatch(f"unparseable unit {unit!r}") from exc


def to_si_checked(value: float, unit: str, expected_si_unit: str) -> float:
    """SI magnitude of `value [unit]`, provided the dimensions match `expected_si_unit`."""
    u = parse_unit(unit)
    expected = quantity(1.0, expected_si_unit).dimensionality
    if (1.0 * u).dimensionality != expected:
        raise UnitMismatch(f"unit {unit!r} has dimensions {(1.0 * u).dimensionality}, expected {expected}")
    return float((value * u).to_base_units().magnitude)

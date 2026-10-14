"""Problem text: number formatting, unit randomization, phrasing templates, distractors."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from physcal.env.graph import Family, Quantity
from physcal.env.units import from_si, to_si

GIVEN_TEMPLATES = (
    "The {desc} is {sym} = {val}.",
    "{sym} = {val} (the {desc}).",
    "The {desc}, {sym}, equals {val}.",
    "It is known that {sym} = {val}, where {sym} is the {desc}.",
)

QUESTION_TEMPLATES = (
    "Find the {desc}, {sym}, in {unit}.",
    "What is the {desc}, {sym}? Give your answer in {unit}.",
    "Determine {sym}, the {desc}. Express the result in {unit}.",
)

QUESTION_TEMPLATES_DIMENSIONLESS = (
    "Find the {desc}, {sym}.",
    "What is the {desc}, {sym}? Give your answer as a pure number.",
    "Determine {sym}, the {desc}.",
)

STEPS_TEMPLATES = (
    "In <steps>, report the intermediate values {syms}.",
    "Report these intermediate quantities in <steps>: {syms}.",
    "Your <steps> must include the intermediate values {syms}.",
)

# Depth-1 sentence, keyed by generator version so that old tapes stay replayable. The 1.0.0 wording made the
# model skip its calculation entirely (96% answer-only at depth 1 vs 0% at depths 2-5, PROGRESS phase 3).
NO_STEPS_TEMPLATES = {
    "1.0.0": "No intermediate values need to be reported, so <steps> may be empty.",
    "1.1.0": "Show your calculation first; no intermediate values need to be listed in <steps>.",
    "2.0.0": "Show your calculation first; no intermediate values need to be listed in <steps>.",
}

# Measurement-noise mode (generator 2.0.0): what "± p%" on a given means. Only rendered when a given is noisy.
NOISE_NOTES = {
    "2.0.0": "Values marked ± p% are measurements: p% is the relative standard uncertainty (one standard deviation) "
             "of that measurement. Values without ± are exact.",
}


def no_steps_template(version: str) -> str:
    if version not in NO_STEPS_TEMPLATES:
        raise ValueError(f"generator version {version!r} has no depth-1 wording; known: {sorted(NO_STEPS_TEMPLATES)}")
    return NO_STEPS_TEMPLATES[version]


def noise_note(version: str) -> str:
    if version not in NOISE_NOTES:
        raise ValueError(f"generator version {version!r} has no noise mode; known: {sorted(NOISE_NOTES)}")
    return NOISE_NOTES[version]

CONSTANT_TEMPLATE = "Use {sym} = {val}."

GIVEN_HEADERS = ("Given:", "Known values:", "Data:")


@dataclass(frozen=True)
class ShownValue:
    """A number as it appears in the text, and its exact SI value."""

    symbol: str
    desc: str
    display: str
    unit: str
    value_si: float
    noise_rel: float = 0.0  # stated relative standard uncertainty of a measured value (0 = exact)

    @property
    def text(self) -> str:
        base = f"{self.display} {self.unit}".strip()
        return f"{base} ± {100 * self.noise_rel:g}%" if self.noise_rel else base

    def to_dict(self) -> dict:
        d = asdict(self)
        if not self.noise_rel:  # exact values keep the generator-1.x dict shape (frozen sets, tape replay)
            del d["noise_rel"]
        return d


def format_number(x: float, sig_figs: int, fixed_exponents: tuple[int, int]) -> str:
    """Round to `sig_figs` significant figures. Fixed notation inside `fixed_exponents`, else e-notation."""
    if x == 0:
        return "0"
    rounded = float(f"{x:.{sig_figs - 1}e}")
    exp = math.floor(math.log10(abs(rounded)))
    lo, hi = fixed_exponents
    if lo <= exp <= hi:
        decimals = max(sig_figs - 1 - exp, 0)
        return f"{rounded:.{decimals}f}"
    mantissa = rounded / 10**exp
    return f"{mantissa:.{sig_figs - 1}f}e{exp}"


def choose_unit(q: Quantity, value_si: float, rng: np.random.Generator, magnitude_range: tuple[float, float]) -> str:
    """Random unit from the quantity's menu whose magnitude lands in `magnitude_range`; else the SI unit."""
    lo, hi = magnitude_range
    options = [u for u in q.units if lo <= abs(from_si(value_si, q.si_unit, u)) < hi]
    if not options:
        return q.si_unit
    return options[rng.integers(len(options))]


def show_value(q: Quantity, value_si: float, rng: np.random.Generator, display_cfg: dict) -> ShownValue:
    """Pick a display unit, round, and return the SI value of exactly what is displayed."""
    unit = choose_unit(q, value_si, rng, tuple(display_cfg["magnitude_range"]))
    shown = format_number(from_si(value_si, q.si_unit, unit), display_cfg["sig_figs"], tuple(display_cfg["fixed_exponent_range"]))
    return ShownValue(q.symbol, q.desc, shown, unit, to_si(float(shown), unit))


def show_measured(q: Quantity, true_si: float, e: float, noise_rel: float, rng: np.random.Generator,
                  display_cfg: dict, sig_figs: int) -> ShownValue:
    """A measured value true·(1 + e), shown with `sig_figs` significant figures and its stated uncertainty. The
    unit is picked from the true value with the same random draws as `show_value`, so problems that differ only
    in their noise level share units."""
    unit = choose_unit(q, true_si, rng, tuple(display_cfg["magnitude_range"]))
    return measured_in_unit(q, true_si, e, noise_rel, unit, display_cfg, sig_figs)


def measured_in_unit(q: Quantity, true_si: float, e: float, noise_rel: float, unit: str, display_cfg: dict,
                     sig_figs: int) -> ShownValue:
    """A measured value true·(1 + e) in a given display unit (A6 pairs share units across both members)."""
    shown = format_number(from_si(true_si * (1.0 + e), q.si_unit, unit), sig_figs,
                          tuple(display_cfg["fixed_exponent_range"]))
    return ShownValue(q.symbol, q.desc, shown, unit, to_si(float(shown), unit), noise_rel)


def _pick(rng: np.random.Generator, options: tuple[str, ...]) -> str:
    return options[rng.integers(len(options))]


def render_text(
    family: Family,
    target: Quantity,
    target_unit: str,
    shown: list[ShownValue],
    intermediates: tuple[str, ...],
    rng: np.random.Generator,
    display_cfg: dict,
    no_steps_text: str,
    noise_text: str | None = None,
) -> str:
    """Assemble intro, givens (in random order), constants, question and the steps instruction. `noise_text`
    (noise mode) follows the givens; it draws nothing from `rng`."""
    intro = _pick(rng, family.intros)
    order = rng.permutation(len(shown))
    lines = [
        _pick(rng, GIVEN_TEMPLATES).format(desc=shown[i].desc, sym=shown[i].symbol, val=shown[i].text)
        for i in order
    ]
    for sym, (value, unit) in family.constants.items():
        val = f"{format_number(value, display_cfg['sig_figs'], tuple(display_cfg['fixed_exponent_range']))} {unit}"
        lines.append(CONSTANT_TEMPLATE.format(sym=sym, val=val))

    if target_unit:
        question = _pick(rng, QUESTION_TEMPLATES).format(desc=target.desc, sym=target.symbol, unit=target_unit)
    else:
        question = _pick(rng, QUESTION_TEMPLATES_DIMENSIONLESS).format(desc=target.desc, sym=target.symbol)
    if intermediates:
        steps = _pick(rng, STEPS_TEMPLATES).format(syms=", ".join(intermediates))
    else:
        steps = no_steps_text

    if rng.integers(2):
        givens = _pick(rng, GIVEN_HEADERS) + "\n" + "\n".join(f"- {line}" for line in lines)
    else:
        givens = " ".join(lines)
    if noise_text:
        givens = f"{givens}\n\n{noise_text}"
    return f"{intro}\n\n{givens}\n\n{question} {steps}"

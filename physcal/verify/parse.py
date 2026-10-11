"""Extract reported steps and the final answer from a completion.

Accepted answer forms (after normalization), all optionally prefixed by `symbol =`:
    12.4 ± 0.8 m/s          12.4 +/- 0.8 m/s         (12.4 ± 0.8) m/s
    (1.24 ± 0.08) × 10^1 m/s                         1.24e1 ± 8e-1 m/s
    12.4 m/s ± 0.8 m/s      12.4 m/s ± 5%            12.4 m/s        (no uncertainty)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
STEPS_RE = re.compile(r"<steps>(.*?)</steps>", re.S | re.I)

NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
SCI = r"(?:\s*(?:[x×*·])\s*10\s*(?:\^|\*\*)\s*\(?\s*([-+]?\d+)\s*\)?)?"
NUMBER_RE = re.compile(rf"^({NUM}){SCI}")
PAREN_RE = re.compile(rf"^\(\s*({NUM}){SCI}\s*±\s*({NUM}){SCI}\s*\){SCI}\s*(.*)$")

SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
SUPERSCRIPT_RUN = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+")
LATEX_WRAPPERS = re.compile(r"\\(?:text|mathrm|mathit|operatorname|textrm|boxed|mathbf|textbf)\s*\{([^{}]*)\}")
THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
DIMENSIONLESS_WORDS = re.compile(r"^\(?\s*(?:dimensionless|unitless|no units?|none)\s*\)?$", re.I)


class ParseError(ValueError):
    """The completion does not follow the required format."""


@dataclass(frozen=True)
class Reported:
    """A reported number with its unit(s), before unit conversion."""

    value: float
    unit: str
    sigma: float | None = None
    sigma_unit: str = ""
    sigma_is_percent: bool = False


@dataclass
class ParsedCompletion:
    answer: Reported
    steps: dict[str, Reported] = field(default_factory=dict)
    steps_block_found: bool = False


def normalize(text: str) -> str:
    """Map unicode and LaTeX spellings onto one plain-text dialect."""
    s = text.replace("−", "-").replace("–", "-").replace("\u00a0", " ")
    s = SUPERSCRIPT_RUN.sub(lambda m: "^" + m.group(0).translate(SUPERSCRIPTS), s)
    s = s.replace("$", "")
    s = re.sub(r"\^\s*\{\s*([^{}]*)\s*\}", r"^(\1)", s)
    for _ in range(3):  # nested wrappers
        s = LATEX_WRAPPERS.sub(r"\1", s)
    s = s.replace("\\pm", "±").replace("+/-", "±").replace("+-", "±")
    s = s.replace("\\times", "×").replace("\\cdot", "·")
    s = re.sub(r"\\[,;:! ]", " ", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    s = THOUSANDS.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def canonical_symbol(sym: str) -> str:
    """`t_1`, `t_{1}`, ` t1 ` -> `t1`; case is preserved (F and f are different quantities)."""
    return re.sub(r"[\s_{}\\$*]", "", normalize(sym))


def clean_unit(unit: str) -> str:
    u = unit.strip().rstrip(".,;:").strip()
    if u.startswith("[") and u.endswith("]"):
        u = u[1:-1].strip()
    if DIMENSIONLESS_WORDS.match(u):
        return ""
    return u


def _take_number(s: str) -> tuple[float, str]:
    m = NUMBER_RE.match(s.strip())
    if not m:
        raise ParseError(f"expected a number at {s.strip()[:30]!r}")
    value = float(m.group(1))
    if m.group(2):
        value *= 10.0 ** int(m.group(2))
    return value, s.strip()[m.end():].strip()


def _scaled(value: str, exp: str | None) -> float:
    return float(value) * (10.0 ** int(exp) if exp else 1.0)


def parse_value(text: str) -> Reported:
    """Parse `[symbol =] value [unit] [± sigma [unit|%]]` (see module docstring)."""
    s = normalize(text).strip("*`_ ").rstrip(".;,").strip()
    if "=" in s:
        s = s.rsplit("=", 1)[1].strip()
    if not s:
        raise ParseError("empty value")
    if s.count("±") > 1:
        raise ParseError("more than one ±")

    m = PAREN_RE.match(s)
    if m:
        mu_v, mu_e, sd_v, sd_e, shared_e, unit = m.groups()
        scale = 10.0 ** int(shared_e) if shared_e else 1.0
        u = clean_unit(unit)
        return Reported(_scaled(mu_v, mu_e) * scale, u, abs(_scaled(sd_v, sd_e) * scale), u)

    if "±" not in s:
        value, unit = _take_number(s)
        return Reported(value, clean_unit(unit))

    left, right = s.split("±")
    value, unit1 = _take_number(left)
    sigma, rest = _take_number(right)
    if rest.startswith("%"):
        return Reported(value, clean_unit(unit1 or rest[1:]), abs(sigma), "", sigma_is_percent=True)
    unit1, unit2 = clean_unit(unit1), clean_unit(rest)
    if unit1 and unit2:
        return Reported(value, unit1, abs(sigma), unit2)
    unit = unit1 or unit2
    return Reported(value, unit, abs(sigma), unit)


def parse_steps(block: str) -> dict[str, Reported]:
    """Lines `symbol = [expression =] value unit`. Unparseable lines are skipped; later lines win."""
    steps: dict[str, Reported] = {}
    for raw in block.splitlines():
        line = normalize(raw).replace("≈", "=")
        line = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", line)
        if "=" not in line:
            continue
        sym = canonical_symbol(line.split("=", 1)[0])
        if not sym:
            continue
        try:
            steps[sym] = parse_value(line)
        except ParseError:
            continue
    return steps


def parse_completion(text: str) -> ParsedCompletion:
    answers = ANSWER_RE.findall(text)
    if not answers:
        raise ParseError("no <answer>...</answer> block")
    if len(answers) > 1:
        raise ParseError("more than one <answer> block")
    answer = parse_value(answers[0])
    blocks = STEPS_RE.findall(text)
    steps = parse_steps(blocks[-1]) if blocks else {}
    return ParsedCompletion(answer=answer, steps=steps, steps_block_found=bool(blocks))

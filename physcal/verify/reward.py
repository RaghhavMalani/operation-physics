"""Reward = pure verifier output: parse -> units gate -> scoring rule -> step credit.

`reward(problem, completion)` returns `(float, breakdown)`. Failures (parse_fail, unit_mismatch) get the
rule's floor, i.e. the worst score a valid answer can get, so a malformed answer never beats a real attempt.
"""

from __future__ import annotations

import math
from typing import Any

from physcal.config import load_config
from physcal.env.units import same_dimension
from physcal.verify.parse import ParseError, Reported, canonical_symbol, parse_completion
from physcal.verify.scoring import (
    RULES,
    RULES_NEEDING_SIGMA,
    ceiling,
    floor,
    norm_cdf,
    relative_terms,
    score,
)
from physcal.verify.units import UnitMismatch, to_si_checked

OK, PARSE_FAIL, UNIT_MISMATCH = "ok", "parse_fail", "unit_mismatch"


def reward_config() -> dict:
    return load_config("reward")


def validate_reward_config(cfg: dict) -> None:
    sc, w = cfg["scoring"], cfg["weights"]
    if sc["rule"] not in RULES:
        raise ValueError(f"unknown scoring rule {sc['rule']!r}; expected one of {RULES}")
    if w["final"] <= 0 or w["steps"] < 0:
        raise ValueError("weights must satisfy final > 0 and steps >= 0")
    # A perfect final answer with no steps must beat a failed final answer with every step right.
    span = ceiling(sc["rule"], sc) - floor(sc["rule"], sc)
    if not w["steps"] < w["final"] * span:
        raise ValueError(f"step weight {w['steps']} can outweigh the final answer (final span {w['final'] * span})")


def _as_dict(problem: Any) -> dict:
    return problem if isinstance(problem, dict) else problem.to_dict()


def _finite_or_none(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _finite_or_none(v) for k, v in x.items()}
    return x


def score_steps(problem: dict, reported: dict[str, Reported], cfg: dict) -> tuple[float, dict]:
    requested = problem["intermediates"]
    if not requested:
        return cfg["steps"]["empty_score"], {}
    by_symbol = {canonical_symbol(k): v for k, v in reported.items()}
    detail: dict[str, dict] = {}
    for sym, truth in requested.items():
        r = by_symbol.get(canonical_symbol(sym))
        if r is None:
            detail[sym] = {"credit": 0.0, "error": "missing"}
            continue
        try:
            value = to_si_checked(r.value, r.unit, problem["intermediate_si_units"][sym])
        except UnitMismatch:
            detail[sym] = {"credit": 0.0, "error": UNIT_MISMATCH}
            continue
        rel = abs(value - truth) / abs(truth)
        detail[sym] = {"credit": float(rel <= cfg["steps"]["tolerance"]), "reported_si": value, "rel_error": rel}
    return sum(d["credit"] for d in detail.values()) / len(requested), detail


def _answer_si(answer: Reported, si_unit: str) -> tuple[float, float | None]:
    mu = to_si_checked(answer.value, answer.unit, si_unit)
    if answer.sigma is None:
        return mu, None
    if answer.sigma_is_percent:
        return mu, abs(mu) * answer.sigma / 100.0
    return mu, abs(to_si_checked(answer.sigma, answer.sigma_unit, si_unit))


def _copied_given(problem: dict, answer: Reported, mu_si: float, tol: float) -> bool:
    """The answer repeats a given number, either as written or as the same physical value."""
    close = lambda a, b: b != 0 and abs(a - b) <= tol * abs(b)
    for g in problem["givens"] + problem["distractors"]:
        if close(answer.value, float(g["display"])):
            return True
        if same_dimension(g["unit"], problem["target_si_unit"]) and close(mu_si, g["value_si"]):
            return True
    return False


def _calibration_stats(mu: float, sigma: float | None, y: float, sc: dict) -> dict:
    e, s = relative_terms(mu, sigma, y, sc["sigma_floor"])
    stats: dict[str, Any] = {
        "mu_si": mu,
        "sigma_si": sigma,
        "rel_error": e,
        "abs_rel_error": abs(e),
        "correct": abs(e) <= sc["binary_tolerance"],
        "rel_sigma": None if sigma is None else sigma / abs(y),
        "rel_sigma_floored": s,
        "z": None if s is None else e / s,
        "within_1sigma": None if sigma is None else abs(mu - y) <= sigma,
        "within_2sigma": None if sigma is None else abs(mu - y) <= 2 * sigma,
        "pit": None,
    }
    if sigma is not None:
        stats["pit"] = norm_cdf((y - mu) / sigma) if sigma > 0 else (0.5 if y == mu else float(y > mu))
    stats["scores"] = {
        r: (None if (s is None and r in RULES_NEEDING_SIGMA) else score(r, e, s, sc)) for r in RULES
    }
    return stats


def reward(problem: Any, completion: str, cfg: dict | None = None) -> tuple[float, dict]:
    cfg = cfg or reward_config()
    validate_reward_config(cfg)
    p = _as_dict(problem)
    sc, w = cfg["scoring"], cfg["weights"]
    rule = sc["rule"]
    worst = floor(rule, sc)
    bd: dict[str, Any] = {
        "rule": rule,
        "reason": OK,
        "detail": "",
        "truth_si": p["target_value_si"],
        "w_final": w["final"],
        "w_steps": w["steps"],
        "n_steps_requested": len(p["intermediates"]),
    }

    try:
        parsed = parse_completion(completion)
        if parsed.answer.sigma is None and rule in RULES_NEEDING_SIGMA:
            raise ParseError("answer has no ± uncertainty")
        if not all(math.isfinite(x) for x in (parsed.answer.value, parsed.answer.sigma or 0.0)):
            raise ParseError("non-finite number in answer")
    except ParseError as exc:
        bd.update(reason=PARSE_FAIL, detail=str(exc), score_final=worst, score_steps=0.0, steps={})
        bd["reward"] = w["final"] * worst
        return bd["reward"], _finite_or_none(bd)

    bd["steps_block_found"] = parsed.steps_block_found
    bd["answer_unit"] = parsed.answer.unit
    bd["score_steps"], bd["steps"] = score_steps(p, parsed.steps, cfg)
    bd["n_steps_correct"] = int(sum(d["credit"] for d in bd["steps"].values()))

    try:
        mu, sigma = _answer_si(parsed.answer, p["target_si_unit"])
    except UnitMismatch as exc:
        bd.update(reason=UNIT_MISMATCH, detail=str(exc), score_final=worst)
    else:
        stats = _calibration_stats(mu, sigma, p["target_value_si"], sc)
        bd.update(stats)
        bd["score_final"] = stats["scores"][rule]
        bd["copied_given"] = _copied_given(p, parsed.answer, mu, cfg["checks"]["copy_tolerance"])

    bd["reward"] = w["final"] * bd["score_final"] + w["steps"] * bd["score_steps"]
    return bd["reward"], _finite_or_none(bd)

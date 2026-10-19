"""Reward = pure verifier output: parse -> units gate -> scoring rule -> step credit (-> confidence, >= 1.2.0).

`reward(problem, completion)` returns `(float, breakdown)`. Failures (parse_fail, unit_mismatch) get the
rule's floor, i.e. the worst score a valid answer can get, so a malformed answer never beats a real attempt.

Verifier 1.2.0 (D83) adds the epistemic channel of the two-channel format: a `<confidence>c%</confidence>` block
after `</answer>`, c one of the configured bins, scored by the Brier score of c against the computation-correct event
of D50/A1, |ln μ − ln f(x_meas)| <= τ, where f(x_meas) is the generator's point estimate from the displayed values
(the truth itself for problems without measurement noise). score_confidence = −(c − 1{event})²; a missing or
malformed block, or an answer that does not parse, gets the worst Brier score, −1. It enters the reward with
weights.confidence. Only configs that carry a `confidence` section (version >= 1.2.0) do any of this, so older
configs, and every tape logged with one, score exactly as before.
"""

from __future__ import annotations

import math
from typing import Any

from physcal.config import load_config
from physcal.env.units import same_dimension
from physcal.verify.parse import ParseError, Reported, canonical_symbol, parse_completion, parse_confidence
from physcal.verify.scoring import (
    RULES,
    RULES_NEEDING_SIGMA,
    ceiling,
    floor,
    norm_cdf,
    relative_terms,
    score,
    sigma_floor,
)
from physcal.verify.units import UnitMismatch, to_si_checked

OK, PARSE_FAIL, UNIT_MISMATCH = "ok", "parse_fail", "unit_mismatch"
CONFIDENCE_MISSING, ANSWER_UNPARSED = "confidence_missing", "answer_parse_fail"
LEGACY_VERIFIER_VERSION = "1.0.0"  # reward configs logged before the version key existed
CONFIDENCE_VERSION = (1, 2, 0)     # the first verifier with the confidence channel (D83)
BRIER_FLOOR = -1.0                 # the worst Brier score: given to a missing block or an unparsed answer


def reward_config(name: str = "reward") -> dict:
    """`reward` is the default (1.1.0); `reward_v12` is verifier 1.2.0 with the confidence channel (D83)."""
    return load_config(name)


def verifier_version(cfg: dict) -> str:
    return cfg.get("version", LEGACY_VERIFIER_VERSION)


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def validate_reward_config(cfg: dict) -> None:
    sc, w = cfg["scoring"], cfg["weights"]
    # 1.0.0 floored crps_log at sigma_floor; later versions must say which floor crps_log uses.
    if (verifier_version(cfg) != LEGACY_VERIFIER_VERSION) != ("crps_log_sigma_floor" in sc):
        raise ValueError(f"verifier {verifier_version(cfg)}: crps_log_sigma_floor must be set iff version > 1.0.0")
    # 1.2.0 added the confidence channel: its section and weight exist iff the version has it.
    has_conf = _version_tuple(verifier_version(cfg)) >= CONFIDENCE_VERSION
    if has_conf != ("confidence" in cfg) or has_conf != ("confidence" in w):
        raise ValueError(f"verifier {verifier_version(cfg)}: the confidence section and weights.confidence must be "
                         f"set iff version >= {'.'.join(map(str, CONFIDENCE_VERSION))}")
    if sc["rule"] not in RULES:
        raise ValueError(f"unknown scoring rule {sc['rule']!r}; expected one of {RULES}")
    if w["final"] <= 0 or w["steps"] < 0 or w.get("confidence", 0.0) < 0:
        raise ValueError("weights must satisfy final > 0, steps >= 0 and confidence >= 0")
    if has_conf:
        cc = cfg["confidence"]
        if not cc["bins"] or any(not 0 < b < 100 for b in cc["bins"]) or cc["tau_log"] <= 0:
            raise ValueError("confidence bins must lie strictly between 0 and 100 percent, and tau_log > 0")
    # A perfect final answer with no steps (and the worst confidence) must beat a failed final answer with every step
    # right (and the best confidence).
    span = ceiling(sc["rule"], sc) - floor(sc["rule"], sc)
    side = w["steps"] + w.get("confidence", 0.0) * (0.0 - BRIER_FLOOR)
    if not side < w["final"] * span:
        raise ValueError(f"step + confidence weights {side} can outweigh the final answer (final span "
                         f"{w['final'] * span})")


def computation_reference_si(problem: dict) -> float:
    """f(x_meas): the generator's point estimate from the displayed values (>= 2.0.0); the truth when the givens are
    exact (generator 1.x, and noise 0 in 2.0.0, where the two coincide)."""
    return (problem.get("noise") or {}).get("point_estimate_si", problem["target_value_si"])


def confidence_terms(problem: dict, completion: str, mu_si: float | None, answer_ok: bool, cc: dict) -> dict:
    """The epistemic channel (D83): c, the computation-correct event and −Brier."""
    c = parse_confidence(completion, cc["bins"])
    ref = computation_reference_si(problem)
    event = bool(answer_ok and mu_si is not None and mu_si / ref > 0
                 and abs(math.log(mu_si / ref)) <= cc["tau_log"])
    out: dict[str, Any] = {"confidence": c, "computation_correct": event,
                           "computation_reference_si": ref, "confidence_reason": OK}
    if c is None:
        out.update(score_confidence=BRIER_FLOOR, confidence_reason=CONFIDENCE_MISSING)
    else:
        out["score_confidence"] = -((c / 100.0 - float(event)) ** 2)
    return out


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
    # z / rel_sigma_floored above use the shared sigma_floor; each rule scores with its own floor.
    stats["scores"] = {
        r: (None if (s is None and r in RULES_NEEDING_SIGMA)
            else score(r, e, relative_terms(mu, sigma, y, sigma_floor(r, sc))[1], sc)) for r in RULES
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
        "verifier_version": verifier_version(cfg),
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
        if "confidence" in cfg:  # >= 1.2.0: an answer that does not parse leaves nothing to be confident about
            bd.update(confidence=parse_confidence(completion, cfg["confidence"]["bins"]), computation_correct=False,
                      computation_reference_si=computation_reference_si(p), confidence_reason=ANSWER_UNPARSED,
                      score_confidence=BRIER_FLOOR, w_confidence=w["confidence"])
            bd["reward"] += w["confidence"] * BRIER_FLOOR
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
    if "confidence" in cfg:  # >= 1.2.0 (D83)
        bd.update(confidence_terms(p, completion, bd.get("mu_si"), bd["reason"] == OK, cfg["confidence"]),
                  w_confidence=w["confidence"])
        bd["reward"] += w["confidence"] * bd["score_confidence"]
    return bd["reward"], _finite_or_none(bd)

"""System prompt that fixes the output format, and chat-message construction."""

from __future__ import annotations

SYSTEM_PROMPT = """You are a careful physics problem solver.

Solve the problem. You may think step by step first. Then finish with exactly this format:

<steps>
symbol = value unit
</steps>
<answer>value ± uncertainty unit</answer>

Rules:
- In <steps>, write one line per intermediate quantity the problem asks you to report, using the problem's symbol names, each with a number and a unit.
- In <answer>, give the final value in the unit the problem asks for, then ± and your honest one-standard-deviation uncertainty in the same unit. The true value should lie within ± one uncertainty of your value about 68% of the time: say a large uncertainty when unsure and a small one when confident.
- Numbers may use scientific notation such as 1.23e-4.
- Write exactly one <answer> block, and nothing after it.

Example of the format:
<steps>
a = 2.45 m/s^2
t = 4.1 s
</steps>
<answer>12.4 ± 0.8 m/s</answer>"""


# Diagnostic variant (phase 3): same format and rules, but it explicitly asks for worked reasoning first.
# The v1 baseline showed the model often answering with no calculation at all (e.g. at depth 1, where
# "<steps> may be empty").
SYSTEM_PROMPT_REASON_FIRST = """You are a careful physics problem solver.

Always work the problem out in plain text before giving any result: convert every given value to SI units, write down each formula you use, substitute the numbers, and compute every intermediate quantity. Do this even when the problem asks for no intermediate values. Only after that calculation, finish with exactly this format:

<steps>
symbol = value unit
</steps>
<answer>value ± uncertainty unit</answer>

Rules:
- In <steps>, write one line per intermediate quantity the problem asks you to report, using the problem's symbol names, each with a number and a unit.
- In <answer>, give the final value in the unit the problem asks for, then ± and your honest one-standard-deviation uncertainty in the same unit. The true value should lie within ± one uncertainty of your value about 68% of the time: say a large uncertainty when unsure and a small one when confident.
- Numbers may use scientific notation such as 1.23e-4.
- Write exactly one <answer> block, and nothing after it.

Example of the format (after your calculation):
<steps>
a = 2.45 m/s^2
t = 4.1 s
</steps>
<answer>12.4 ± 0.8 m/s</answer>"""

# Diagnostic variant (phase 3): v1's rules plus one fully worked example (a training-seed problem:
# dynamics, depth 2, seed 424242, generator 1.0.0), embedded verbatim so the prompt never drifts with the
# generator. Caveat for calibration studies: the example's "about 1%" uncertainty is itself a cue for σ.
WORKED_EXAMPLE = """Example problem:
A crate of mass m rests on a level floor, and the coefficient of kinetic friction between the crate and the floor is mu. Starting from rest, the crate is pulled by a constant horizontal force F for a time t1. While it slides, the floor exerts a normal force N and a kinetic friction force f_k on it; the net horizontal force on the crate is F_net and its acceleration is a1. When the pull stops, the crate has speed v and momentum p, and it has moved a distance s1. Afterwards it slides freely, slowing down under friction alone with a deceleration of magnitude a2, and stops after a further time t2 and distance s2.

Known values:
- len_rope = 7110 mm (the length of the rope used to pull the crate).
- The mass of the crate, m, equals 2190 g.
- P_air = 102 kPa (the air pressure in the room).
- F_net = 5370 mN (the net horizontal force on the crate during the pull).
- The speed of the crate when the pull stops is v = 42.8 km/h.
- Use g = 9.81 m/s^2.

Determine s1, the distance the crate moves during the pull. Express the result in km. Report these intermediate quantities in <steps>: a1.

Example solution:
Convert the needed values to SI: F_net = 5370 mN = 5.37 N, m = 2190 g = 2.19 kg, v = 42.8 km/h = 42.8 / 3.6 m/s = 11.889 m/s. The rope length and the air pressure are not needed.
Newton's second law: a1 = F_net / m = 5.37 / 2.19 = 2.452 m/s^2.
The crate starts from rest with constant acceleration, so v^2 = 2 a1 s1 and s1 = v^2 / (2 a1) = 11.889^2 / (2 * 2.452) = 141.35 / 4.904 = 28.82 m.
In the requested unit: s1 = 28.82 m = 0.02882 km. The inputs have 3 significant figures, so I estimate a relative uncertainty of about 1%, i.e. ± 0.0003 km.
<steps>
a1 = 2.452 m/s^2
</steps>
<answer>0.02882 ± 0.0003 km</answer>"""

SYSTEM_PROMPT_WORKED_EXAMPLE = SYSTEM_PROMPT[: SYSTEM_PROMPT.index("Example of the format:")] + WORKED_EXAMPLE

# Phase 3.5: the worked example without the uncertainty heuristic. The example's answer line must still show
# a ± value (that is the format); its solution text no longer explains how the σ was chosen.
SIGMA_HINT = (" The inputs have 3 significant figures, so I estimate a relative uncertainty of about 1%,"
              " i.e. ± 0.0003 km.")
assert WORKED_EXAMPLE.count(SIGMA_HINT) == 1
SYSTEM_PROMPT_WORKED_EXAMPLE_NO_SIGMA_HINT = SYSTEM_PROMPT_WORKED_EXAMPLE.replace(SIGMA_HINT, "")

SYSTEM_PROMPTS = {
    "v1": SYSTEM_PROMPT,
    "v2_reason_first": SYSTEM_PROMPT_REASON_FIRST,
    "v3_worked_example": SYSTEM_PROMPT_WORKED_EXAMPLE,
    "v4_worked_example": SYSTEM_PROMPT_WORKED_EXAMPLE_NO_SIGMA_HINT,
}


def build_messages(problem_text: str, prompt_id: str = "v1") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[prompt_id]},
        {"role": "user", "content": problem_text},
    ]

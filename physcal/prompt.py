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


def build_messages(problem_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem_text},
    ]

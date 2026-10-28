"""R5-5 feasibility (DECISIONS.md D133): the per-model chat handling, CPU only."""

from physcal.eval import r5_feasibility as F
from physcal.prompt import SYSTEM_PROMPTS


def test_a_template_without_a_system_role_gets_the_system_prompt_atop_the_user_turn():
    cfg = F._config()
    gemma, llama = cfg["candidates"]["gemma2_2b"], cfg["candidates"]["llama32_3b"]
    sys = SYSTEM_PROMPTS[cfg["baseline"]["prompt"]]
    m = F.messages("PROBLEM", gemma, cfg["baseline"]["prompt"])
    assert m == [{"role": "user", "content": f"{sys}\n\nPROBLEM"}]
    m = F.messages("PROBLEM", llama, cfg["baseline"]["prompt"])
    assert [x["role"] for x in m] == ["system", "user"] and m[1]["content"] == "PROBLEM"


def test_every_candidate_names_its_end_of_turn_and_the_reference_is_the_pinned_base():
    cands = F._config()["candidates"]
    assert all(c["end_of_turn"] for c in cands.values())
    assert cands["qwen25_math_1p5b"]["pinned_base"] == "base_4bit"
    assert all("repo" in c for n, c in cands.items() if n != "qwen25_math_1p5b")


def test_lenient_number_reads_boxed_then_answer_then_the_last_number():
    assert F.lenient_number("so v = 3 m/s and \\boxed{16.94} \\text{m/s}") == 16.94
    assert F.lenient_number("<steps>a = 2</steps>\n<answer>11.66 ± 0.01 m/s</answer>") == 11.66
    assert F.lenient_number("the speed is about 4.2e1 m/s") == 42.0
    assert F.lenient_number("no number here") is None

"""D173's two controls: the same-label texts and the quantities that read them (synthetic; no model)."""

import json

import numpy as np
import pytest

from physcal.eval import followup_r6 as R6
from physcal.eval import followup_r6_controls as C


def _item(pid, sample, correct, box, reasoning=None):
    comp = f"{reasoning or f'reasoning {pid} {sample}'} so \\boxed{{{box}}} done"
    return {"id": f"m|f|1|{pid}|{sample}", "key": ["f", 1, pid], "sample": sample, "correct": correct, "completion": comp,
            "text": "q", "frame": "native", "n_tokens": 10, "truncated": False, "source": "m"}


def _problem(pid=1):
    """Two correct answers (40), three wrong: two at 7 (one within 1%), one at 9."""
    return [_item(pid, 0, False, "7"), _item(pid, 1, True, "40"), _item(pid, 2, False, "9"), _item(pid, 3, True, "40.1"),
            _item(pid, 4, False, "7.01")]


def _values(items):
    return {it["id"]: float(F_box(it)) for it in items}


def F_box(it):
    return C.F.last_boxed(it["completion"])[0]


def test_each_answer_gets_the_lowest_sample_donor_of_each_kind_and_never_itself():
    items = _problem()
    texts = {(t["sample"], t["kind"]): t for t in C.same_label_set(items, _values(items), 0.01)}
    assert set(texts) == {(1, "correct"), (3, "correct"), (0, "same_value"), (0, "diff_value"), (4, "same_value"),
                          (4, "diff_value"), (2, "diff_value")}
    assert texts[(1, "correct")]["donor_sample"] == 3 and texts[(3, "correct")]["donor_sample"] == 1
    assert texts[(0, "same_value")]["donor_sample"] == 4 and texts[(0, "diff_value")]["donor_sample"] == 2
    assert texts[(4, "same_value")]["donor_sample"] == 0 and texts[(2, "diff_value")]["donor_sample"] == 0


def test_a_text_is_the_donors_reasoning_with_the_answers_box_and_a_position_on_its_closing_brace():
    items = _problem()
    t = next(x for x in C.same_label_set(items, _values(items), 0.01) if x["sample"] == 2)
    assert t["completion"] == "reasoning 1 0 so \\boxed{9}" and t["orig_id"] == items[2]["id"]
    assert t["donor_id"] == items[0]["id"] and t["correct"] is False
    assert t["completion"][t["p2"]] == "}" or t["completion"][t["p2"] - 1] == "}"
    assert (t["p1"], t["p2"]) == R6.native_positions(t["completion"])


def test_an_answer_without_a_donor_of_a_kind_has_no_text_of_it():
    items = [_item(1, 0, True, "40"), _item(1, 1, False, "7"), _item(1, 2, False, "9")]
    texts = C.same_label_set(items, _values(items), 0.01)
    assert {(t["sample"], t["kind"]) for t in texts} == {(1, "diff_value"), (2, "diff_value")}


def _scored(items, box_only=None, reasoning=None, penalty=0.0):
    """Texts scored by a fake probe: p = the reasoning's own level, minus `penalty` when the box disagrees with it."""
    texts = C.same_label_set(items, _values(items), 0.01)
    level = {it["id"]: (0.8 if it["correct"] else 0.3) + 0.01 * it["sample"] for it in items}
    p = {t["id"]: level[t["donor_id"]] - (penalty if t["kind"] == "diff_value" else 0.0) for t in texts}
    return texts, p, level


def test_the_mismatch_cost_is_zero_for_a_probe_that_reads_the_reasoning_alone():
    items = _problem() + _problem(2)
    texts, p, level = _scored(items)
    keys, m = C.mismatch_per_problem(texts, C.box_effects(texts, p, level))
    assert len(keys) == 2 and np.allclose(m, 0.0)


def test_the_mismatch_cost_recovers_a_known_penalty_and_uses_only_answers_with_both_kinds():
    items = _problem() + _problem(2)
    texts, p, level = _scored(items, penalty=0.2)
    e = C.box_effects(texts, p, level)
    _, m = C.mismatch_per_problem(texts, e)
    assert np.allclose(m, -0.2)
    by = C.mean_box_effects(texts, e)
    assert by["correct"]["mean"] == pytest.approx(0) and by["same_value"]["mean"] == pytest.approx(0)
    assert by["diff_value"]["mean"] == pytest.approx(-0.2) and by["diff_value"]["n"] == 6


def test_the_fixed_reasoning_pairs_share_one_donor_and_score_ties_at_a_half():
    items = _problem()
    texts = C.same_label_set(items, _values(items), 0.01)
    opp = R6.swap_set("m", items)
    p_opp = {t["id"]: 0.6 for t in opp}
    p_same = {t["id"]: (0.5 if t["sample"] == 2 else 0.6) for t in texts}
    keys, f = C.fixed_reasoning_per_problem(opp, texts, p_opp, p_same)
    # d0 = sample 0: correct boxes (samples 1, 3) after it against the wrong box 9 (sample 2) after it, and nothing else
    assert keys == [("f", 1, 1)] and f.tolist() == [1.0]
    _, tie = C.fixed_reasoning_per_problem(opp, texts, p_opp, {t["id"]: 0.6 for t in texts})
    assert tie.tolist() == [0.5]


def test_a_problem_whose_wrong_answers_all_agree_has_no_fixed_reasoning_pair():
    items = [_item(1, 0, False, "7"), _item(1, 1, True, "40"), _item(1, 2, False, "7")]
    texts = C.same_label_set(items, _values(items), 0.01)
    opp = R6.swap_set("m", items)
    keys, f = C.fixed_reasoning_per_problem(opp, texts, {t["id"]: 0.5 for t in opp}, {t["id"]: 0.5 for t in texts})
    assert keys == [] and len(f) == 0


def test_the_extra_manifest_serves_its_set_and_d167s_manifest_keeps_priority(tmp_path):
    cfg = {"output_dir": str(tmp_path), "r6": {"output_dir": "r6"}}
    d = R6.r6_dir(cfg) / "items"
    d.mkdir(parents=True)
    blob = json.dumps({"id": "x"}) + "\n"
    (d / "a.jsonl").write_text(blob)
    (d / "b.jsonl").write_text(blob)
    (d / "manifest.json").write_text(json.dumps({"sets": {"a": {"sha256": R6._sha(blob)}}}))
    (d / R6.EXTRA_MANIFEST).write_text(json.dumps({"sets": {"a": {"sha256": "no"}, "b": {"sha256": R6._sha(blob)}}}))
    assert R6.load_set("a", cfg) == [{"id": "x"}] and R6.load_set("b", cfg) == [{"id": "x"}]
    (d / "b.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit):
        R6.load_set("b", cfg)


def test_the_stages_are_d173s_two():
    assert C.GPU_STAGES == (("states", "swap_same_1.5b", "1.5b"), ("states", "answer_only_1.5b", "random_1.5b"))

"""D167 (exploratory follow-up track, paper #2): R6-3 / R6-4 / R6-5's item sets, pairs, factorial reading, text
features, fold-probe refits, and the GPU readout path on a tiny Qwen2 on CPU. Nothing here touches a GPU or reads a
real readout."""

import copy
import json
import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from physcal.config import REPO_ROOT
from physcal.eval import followup_r6 as R6
from physcal.eval import followup_scale as F

SC = {"bootstrap": 200, "seed": 0, "generation": {"k": 8}}


def _row(key, sample, correct, value=1.0, truncated=False, n_tokens=100, completion=None):
    comp = completion or (f"work {sample}\nso \\boxed{{{value} m}}" if not truncated else f"work {sample} and on")
    return {"key": list(key), "sample": sample, "value": None if truncated or correct is None else value,
            "correct": None if truncated else correct, "truncated": truncated, "n_tokens": n_tokens,
            "completion": comp, "text": "A problem. What is x in m?", "index": key[2]}


# ------------------------------------------------------------------------------------------- positions and item sets
def test_native_positions_are_the_answers_last_character_and_the_closing_brace():
    c = "so \\boxed{12.4 \\text{ m} } done"
    p1, p2 = R6.native_positions(c)
    assert c[p2] == "}" and c[p2 - 1] == " " and c[p1] == "}" and c[:p1 + 1].endswith("\\text{ m}")
    assert R6.native_positions("no box") is None and R6.native_positions("\\boxed{ }") is None


def test_terse_positions_need_one_answer_block():
    c = "SI values: ...\n<steps>\n</steps>\n<answer>54.74 ± 0 cm/s^2</answer>"
    p1, p2 = R6.terse_positions(c)
    assert c[p1] == "2" and c[p2:].startswith("</answer>")
    assert R6.terse_positions("<answer>1</answer><answer>2</answer>") is None
    assert R6.terse_positions("<answer></answer>") is None


def test_bank_sets_take_the_problems_both_models_keep():
    a, b, c = ("f", 1, 1), ("f", 1, 2), ("f", 2, 3)
    rows = {"7b": [_row(a, 0, True), _row(a, 1, False), _row(a, 2, None, truncated=True),
                   _row(b, 0, True), _row(b, 1, False), _row(c, 0, True), _row(c, 1, True)],
            "1.5b": [_row(b, 0, False), _row(b, 1, True), _row(a, 0, True), _row(a, 1, False), _row(c, 0, False),
                     _row(c, 1, False)]}
    s = R6.bank_sets(rows, spot_n=1)
    assert [tuple(it["key"]) for it in s["bank_7b"]] == [a, a, b, b]          # the 7B's kept order; c is not mixed
    assert [it["sample"] for it in s["bank_1.5b"]] == [0, 1, 0, 1] and s["bank_1.5b"][0]["key"] == list(a)
    assert [it["id"] for it in s["truncated"]] == ["7b|f|1|1|2"] and s["truncated"][0]["p2"] is None
    assert s["spot_7b"] == s["bank_7b"][:1]
    it = s["bank_7b"][0]
    assert it["frame"] == "native" and it["completion"][it["p2"]] == "}" and it["correct"] is True


def test_match_candidates_pairs_correct_by_order_and_incorrect_by_value():
    nat = [{"id": f"n{i}", "correct": c, "ratio": r} for i, (c, r) in enumerate([(True, 1.0), (False, 2.0), (False, 3.0),
                                                                                  (True, 1.004), (False, 9.0)])]
    ter = [{"id": f"t{i}", "correct": c, "ratio": r} for i, (c, r) in enumerate([(False, 3.02), (True, 0.999),
                                                                                  (False, 2.01), (False, 2.0)])]
    out = R6.match_candidates(nat, ter, 0.01)
    assert out == [{"native_id": "n0", "terse_id": "t1", "correct": True},      # one correct pair: min(2, 1)
                   {"native_id": "n1", "terse_id": "t2", "correct": False},     # 2.01: the first agreeing, not 2.0
                   {"native_id": "n2", "terse_id": "t0", "correct": False}]     # 3.02 is 0.66% off; 9.0 has no match


def test_factorial_sets_need_both_models_mixed_and_keep_identical_candidates():
    a, b = ("f", 1, 10), ("f", 1, 11)
    native = [_row(a, 0, True, 5.0), _row(a, 1, False, 9.0), _row(b, 0, True, 5.0), _row(b, 1, False, 7.0)]

    def rec(key, s, rel, err):
        comp = f"x = {rel}\n<steps>\n</steps>\n<answer>{rel} ± 0 m</answer>"
        return {"problem": {"family": key[0], "depth": key[1], "seed": key[2], "text": "A problem."}, "sample": s,
                "completion": comp, "n_tokens": 50, "breakdown": {"reason": "ok", "mu_si": rel, "truth_si": 1.0,
                                                                  "abs_rel_error": err}}
    sft = [rec(a, 0, 1.0, 0.0), rec(a, 1, 9 / 5, 0.8), rec(b, 0, 1.0, 0.0), rec(b, 1, 3.0, 2.0)]
    fn, ft, cands = R6.factorial_sets(native, sft, {10: 5.0, 11: 5.0}, 0.01, {10, 11})
    assert len(fn) == 4 and len(ft) == 4 and all(t["frame"] == "terse" for t in ft)
    assert {tuple(c["key"]) for c in cands} == {a}                    # b's wrong values differ: no incorrect candidate
    assert {c["native_id"] for c in cands} == {"1.5b|f|1|10|0", "1.5b|f|1|10|1"}


def test_answer_only_and_swap_texts():
    a = ("f", 1, 1)
    items = [R6.native_item("7b", _row(a, s, s < 2, value=1.0 + s)) for s in range(4)]
    ao = R6.answer_only_set("7b", items)
    assert ao[0]["completion"] == "\\boxed{1.0 m}" and ao[0]["completion"][ao[0]["p2"]] == "}"
    sw = R6.swap_set("7b", items)
    assert sw[0]["donor_sample"] == 2 and sw[3]["donor_sample"] == 0          # the lowest opposite-label sample
    assert sw[0]["completion"] == "work 2\nso \\boxed{1.0 m}" and sw[0]["orig_id"] == items[0]["id"]
    assert sw[0]["completion"][sw[0]["p2"]] == "}" and sw[0]["correct"] is True


def test_every_gpu_stage_names_a_built_set_and_an_experiment():
    names = {"bank_7b", "bank_1.5b", "truncated", "spot_7b", "spot_1.5b", "fact_native", "fact_terse",
             "answer_only_7b", "answer_only_1.5b", "swap_7b", "swap_1.5b"}
    for kind, s, reader in R6.GPU_STAGES:
        assert s in names and reader in R6.READERS and kind in ("states", "yes")
        assert R6.EXPERIMENT[s.split("_")[0]] in ("R6-3", "R6-4", "R6-5")
    assert all(k == "yes" for k, s, _ in R6.GPU_STAGES if s == "truncated")
    assert {r for k, s, r in R6.GPU_STAGES if s.startswith("fact")} == {"1.5b", "sft", "random_1.5b"}


# ------------------------------------------------------------------------------------------- pairs and effects
def _items(src, key, labels, ntok=None):
    return [{"id": f"{src}|{key}|{i}", "key": list(key), "source": src, "correct": c,
             "n_tokens": (ntok or [100] * len(labels))[i]} for i, c in enumerate(labels)]


def test_pair_filters_and_per_problem():
    k = ("f", 1, 1)
    items = _items("7b", k, [True, False]) + _items("1.5b", k, [True, False], ntok=[100, 300])
    g = R6.by_problem(items)
    conf = {it["id"]: (1.0 if it["correct"] else 0.0) for it in items}
    conf["1.5b|('f', 1, 1)|1"] = 2.0                                     # the 1.5B's wrong answer ranked highest
    filt = R6.pair_filters(0.1)
    assert R6.n_pairs(g, [k], filt["within_source"])[0] == 2 and R6.n_pairs(g, [k], filt["cross_source"])[0] == 2
    assert R6.per_problem(g, [k], conf, filt["source_7b"])[0] == 1.0
    assert R6.per_problem(g, [k], conf, filt["source_1.5b"])[0] == 0.0
    assert R6.per_problem(g, [k], conf, filt["within_source"])[0] == 0.5
    assert R6.n_pairs(g, [k], filt["within_source_length_matched"])[0] == 1
    assert np.isnan(R6.per_problem(g, [k], conf, lambda a, b: False)[0])


def test_factorial_effects_read_the_dominant_factor():
    rng = np.random.default_rng(0)
    n = 60
    base = 0.55 + 0.02 * rng.standard_normal(n)
    pa = {("1.5b", "native"): base + 0.2, ("1.5b", "terse"): base + 0.2,     # weights matter, presentation does not
          ("sft", "native"): base, ("sft", "terse"): base + 0.002 * rng.standard_normal(n),
          ("random_1.5b", "native"): base, ("random_1.5b", "terse"): base}
    out = R6.factorial_effects(pa, SC)
    assert out["reading"] == "the weights factor dominates: fine-tuning is implicated"
    assert out["W"]["value"] == pytest.approx(-0.2, abs=0.01) and not out["the factors interact"]
    pa2 = {("1.5b", "native"): base + 0.2, ("1.5b", "terse"): base, ("sft", "native"): base + 0.2,
           ("sft", "terse"): base + 0.001 * rng.standard_normal(n)}
    out2 = R6.factorial_effects(pa2, SC)
    assert out2["reading"] == "the presentation factor dominates: format / readout"
    assert "surface (random/terse - random/native)" not in out2


def test_candidate_pa_scores_each_candidate_by_its_trace_in_the_frame():
    cands = [{"key": ["f", 1, 1], "native_id": "n0", "terse_id": "t0", "correct": True},
             {"key": ["f", 1, 1], "native_id": "n1", "terse_id": "t1", "correct": False}]
    conf = {"n0": 0.9, "n1": 0.1, "t0": 0.2, "t1": 0.8}
    assert R6.candidate_pa(cands, conf, "native")[1][0] == 1.0
    assert R6.candidate_pa(cands, conf, "terse")[1][0] == 0.0


# ------------------------------------------------------------------------------------------- R6-5 pieces
RT = {"ngram_range": [1, 2], "hesitation": ["wait", "let me"], "operators": ["\\frac", "\\sqrt"], "log10_zero": -12.0}


def test_hand_features():
    comp = "Wait, let me see: x = \\frac{1}{2} = 0.5, so \\boxed{3} no; \\text{ok} = 7 \\boxed{12.5 \\text{ m}}"
    from physcal.eval import followup_audit as AU

    rec = AU.text_fields("A problem.", comp)
    h = R6.hand_features(rec, comp, 12.5, 3, 40, RT)
    assert len(h) == 15
    assert h[0] == 40 and h[5] == 3 and h[6] == pytest.approx(np.log10(12.5)) and h[7] == 1.0
    assert h[8] == 1 and h[9] == 3 and h[10] == 1 and h[11] == 2 and h[12] == 2 and h[14] == 3
    assert R6.hand_features(rec, comp, 0.0, 1, 1, RT)[6] == -12.0


def test_crossfit_rich_reads_text_and_hand_features():
    rng = np.random.default_rng(0)
    recs, H, y, groups = [], [], [], []
    for p in range(25):
        for s in range(6):
            ok = s % 2 == 0
            words = " ".join(rng.choice(["alpha", "beta", "gamma", "delta"], 6))
            recs.append({"problem": f"problem {p % 4} text", "answer": f"{s} m",
                         "reasoning": words + (" verified twice" if ok else " hmm wait")})
            H.append([float(ok) + rng.normal(0, 0.1), rng.normal()])
            y.append(int(ok))
            groups.append(f"p{p}")
    a5 = {"text": {"token_pattern": "\\S+", "ngram_range": [1, 1], "min_df": 3, "sublinear_tf": True,
                   "gbt_max_features": 50}, "grids": {"logistic_C": [1.0], "gbt": [{"max_iter": 20, "max_depth": 2,
                                                                                     "learning_rate": 0.1,
                                                                                     "min_samples_leaf": 5}]},
          "seed": 0}
    p, chosen = R6.crossfit_rich(recs, np.array(H), np.array(y), groups, 5, 3, a5, RT)
    assert not np.isnan(p).any() and len(chosen) == 5
    assert p[np.array(y) == 1].mean() > p[np.array(y) == 0].mean() + 0.3


def test_refit_fold_probes_reproduce_the_crossfit():
    from physcal.eval import x2_analysis as XA

    rng = np.random.default_rng(1)
    index = [{"key": ["f", 1, p], "sample": s, "correct": bool((s + p) % 2)} for p in range(30) for s in range(4)]
    y = np.array([r["correct"] for r in index], dtype=float)
    layers = [rng.standard_normal((len(index), 8)) + k * 0.5 * y[:, None] for k in range(3)]
    pc = {"C": [0.01, 1.0], "folds": 5, "max_iter": 2000, "n_jobs": 1}
    conf, info = XA.crossfit_confidence(layers, index, pc)
    cfg = {"scale": {"probe": pc}}
    got = np.full(len(index), np.nan)
    for te, layer, f in R6.refit_fold_probes(layers, index, info["folds"], cfg):
        got[te] = f(layers[layer][te])
    want = np.array([conf[(tuple(r["key"]), r["sample"])] for r in index])
    assert np.abs(got - want).max() < 1e-9


def test_spot_check():
    ref = [np.ones((3, 4), dtype=np.float16)]
    assert R6.spot_check([ref[0] + np.float16(0.001)], ref, 1e-2)["passes"]
    assert not R6.spot_check([ref[0] * 2], ref, 1e-2)["passes"]


# ------------------------------------------------------------------------------------------- the GPU path, on CPU
@pytest.fixture(scope="module")
def tiny_backend():
    import torch
    import transformers

    tok_dir = REPO_ROOT / F._config()["scale"]["models"]["1.5b"]["dir"]
    if not (tok_dir / "tokenizer.json").exists():
        pytest.skip("no local tokenizer")
    tok = transformers.AutoTokenizer.from_pretrained(str(tok_dir))
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    model = transformers.Qwen2ForCausalLM(cfg).eval()
    return SimpleNamespace(tok=tok, model=model, device="cpu", pad_id=tok.pad_token_id)


def _cfg(tmp_path):
    cfg = copy.deepcopy(F._config())
    cfg["output_dir"] = str(tmp_path / "out")
    cfg["scale"]["hidden_batch_size"] = 2
    return cfg


def _two_items():
    nat = R6.native_item("1.5b", _row(("f", 1, 1), 0, True, completion="Step one. So \\boxed{12.4 \\text{ m}}."))
    comp = "SI values: x = 1 m.\n<steps>\n</steps>\n<answer>12.4 ± 0 m</answer>"
    p1, p2 = R6.terse_positions(comp)
    ter = nat | {"id": "sft|f|1|1|0", "frame": "terse", "completion": comp, "p1": p1, "p2": p2, "source": "sft"}
    return [nat, ter]


def test_item_states_read_the_named_tokens_and_match_an_unpadded_pass(tmp_path, tiny_backend):
    import torch

    from physcal.eval import x2_hidden as XH

    cfg = _cfg(tmp_path)
    items = _two_items()
    st, check, shas, same = R6.item_states(tiny_backend, items, cfg)
    assert XH._passes(check, cfg["scale"]["logit_check_rtol"]) and len(shas) == 2 and len(same) == 2
    assert len(st["p2"]) == 3 and st["p2"][0].shape == (2, 32) and st["p2"][0].dtype == np.float16
    tok = tiny_backend.tok
    for j, it in enumerate(items):
        prompt = R6.prompt_text(it, tok, cfg)
        enc = tok([prompt + it["completion"]], return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0].tolist()
        t2 = XH.token_at(offs, len(prompt) + it["p2"])
        piece = tok.decode(enc["input_ids"][0, t2])
        assert ("}" in piece) if it["frame"] == "native" else ("<" in piece)
        with torch.inference_mode():
            hs = tiny_backend.model.model(**enc, output_hidden_states=True).hidden_states
        assert np.allclose(st["p2"][-1][j], hs[-1][0, t2].numpy().astype(np.float16), atol=1e-2)
    assert "Please reason step by step" in R6.prompt_text(items[0], tok, cfg)
    assert "Please reason step by step" not in R6.prompt_text(items[1], tok, cfg)


def test_run_states_freezes_resumes_and_loads(tmp_path, tiny_backend, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg["r6"]["part_size"] = 1
    cfg["r6"]["p1_sets"] = ["toy"]                                          # this set keeps p1 (as R6-4's do)
    items = _two_items()
    d = R6.r6_dir(cfg) / "items"
    d.mkdir(parents=True)
    blob = "".join(json.dumps(it, sort_keys=True) + "\n" for it in items)
    (d / "toy.jsonl").write_text(blob)
    (d / "manifest.json").write_text(json.dumps({"sets": {"toy": {"n": 2, "sha256": R6._sha(blob)}}}))
    calls = []
    monkeypatch.setattr(R6, "reader_backend", lambda *a: calls.append(a) or tiny_backend)
    monkeypatch.setattr(R6, "reader_dir", lambda *a: tmp_path)
    info = R6.run_states("toy", "1.5b", 1.0, cfg)
    assert info["parts"] == 2 and len(calls) == 1
    R6.run_states("toy", "1.5b", 1.0, cfg)
    assert len(calls) == 1                                                  # resumed: nothing read again
    layers, ids, shas = R6.load_states("toy", "1.5b", "p1", cfg)
    assert ids == [it["id"] for it in items] and layers[0].shape == (2, 32) and len(set(shas)) == 2
    assert 0.0 <= R6.p1_is_p2_share("toy", "1.5b", cfg) <= 1.0
    cfg["r6"]["p1_sets"] = []                                               # another set: p2 alone is stored
    (d / "manifest.json").write_text(json.dumps({"sets": {k: {"n": 2, "sha256": R6._sha(blob)} for k in ("toy", "toy2")}}))
    (d / "toy2.jsonl").write_text(blob)
    R6.run_states("toy2", "1.5b", 1.0, cfg)
    z = np.load(sorted(R6.states_dir("toy2", "1.5b", cfg).glob("part_*.npz"))[0])
    assert any(k.startswith("p2_layer_") for k in z.files) and not any(k.startswith("p1_layer_") for k in z.files)
    (d / "toy.jsonl").write_text(blob.replace("12.4", "12.5"))
    with pytest.raises(SystemExit):
        R6.load_set("toy", cfg)                                             # a changed freeze is refused


def test_the_queue_dry_run_lists_every_stage_then_the_analysis(tmp_path):
    out = tmp_path / "r6"
    (out / "items").mkdir(parents=True)
    (out / "items" / "manifest.json").write_text("{}")
    e = {**os.environ, "DRY_RUN": "1", "DISK_GUARD": "off", "QUEUE_LOG": str(tmp_path / "q"), "R6_OUT": str(out),
         "GPU_DEADLINE": "13:00", "MIN_STAGE_S": "0"}
    r = subprocess.run(["bash", str(REPO_ROOT / "scripts" / "queue_followup_r6.sh")], capture_output=True, text=True,
                       env=e, timeout=600)
    assert r.returncode == 0, r.stderr
    log = (tmp_path / "q" / "chain.log").read_text()
    ran = [ln.split("DRY ", 1)[1].split(" ", 1)[0] for ln in log.splitlines()
           if "] DRY " in ln and "commit" not in ln and "disk" not in ln]
    assert ran == [f"r6_{k}_{s}_{r}" for k, s, r in R6.GPU_STAGES] + ["r6_analyze"]
    assert os.access(REPO_ROOT / "scripts" / "queue_followup_r6.sh", os.X_OK)


def test_the_analysis_cache_refits_only_when_its_inputs_change(tmp_path):
    cfg = _cfg(tmp_path)
    calls = []

    def fit():
        calls.append(1)
        return {"a": 0.25}, {"folds": [1]}
    x = np.arange(6, dtype=np.float16)
    assert R6.cached(cfg, "toy", ["toy", x], fit) == ({"a": 0.25}, {"folds": [1]})
    assert R6.cached(cfg, "toy", ["toy", x], fit) == ({"a": 0.25}, {"folds": [1]}) and len(calls) == 1
    R6.cached(cfg, "toy", ["toy", x + 1], fit)
    assert len(calls) == 2


def test_no_analysis_function_shadows_the_cache_helper():
    """D170: analyze_controls bound a local `cached` (the probe's JSON) and then called `cached(...)`, so R6-5 would have
    died at its first fit. Every function that calls the helper must see the module's."""
    import symtable

    mod = symtable.symtable(open(R6.__file__).read(), R6.__file__, "exec")
    callers = [f for f in mod.get_children() if f.get_type() == "function" and "cached" in f.get_identifiers()]
    assert {f.get_name() for f in callers} >= {"analyze_bank", "analyze_factorial", "analyze_controls"}
    assert not [f.get_name() for f in callers if f.lookup("cached").is_local()]

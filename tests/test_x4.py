"""X4 (D89): self-verification on the X2 pairs. The pre-registered prompt; the canonical reading position; the yes/no
tokens and the confidence; the donor derangement; the real X2 answers; the whole run on a tiny CPU model (resume,
the shuffled arm, the float32 head against the model's own logits); the analysis and D89's reading rules."""

import copy
import hashlib
import json
import math

import numpy as np
import pytest

from physcal.config import REPO_ROOT
from physcal.eval import x2_pairs as X
from physcal.eval import x4_analysis as A
from physcal.eval import x4_verify as V

TOKENIZER_DIR = REPO_ROOT / "results" / "sft" / "qwen2.5-math-1.5b-instruct" / "merged"
K = 4
needs_tokenizer = pytest.mark.skipif(not (TOKENIZER_DIR / "tokenizer.json").exists(), reason="local tokenizer missing")


# ------------------------------------------------------------------------------------------- fixtures
def _pool(n):
    from physcal.env.generator import problem_from_dict

    lines = (REPO_ROOT / X._config()["candidates"]["file"]).read_text().splitlines()[:n]
    return [problem_from_dict(json.loads(l)) for l in lines]


def _write_chunk(root, problems):
    """K answers per problem, two correct then two wrong (every problem is kept)."""
    from physcal.tape.tape import TapeWriter

    d = root / "chunk_00000"
    d.mkdir(parents=True)
    with TapeWriter(d / "rollouts.jsonl", "fake", {"generation": {"num_samples": K},
                                                   "prompt_id": "v4_worked_example"}) as t:
        for p in problems:
            for j in range(K):
                err = 0.2 if j >= 2 else 0.001
                mu = p.target_value * (1 + err)
                comp = f"Work {j}.\n<steps>\n</steps>\n<answer>{mu:.4g} ± {0.01 * mu:.2g} {p.target_unit}</answer>"
                bd = {"reason": "ok", "mu_si": p.target_value_si * (1 + err), "sigma_si": 0.01 * p.target_value_si,
                      "abs_rel_error": err, "truth_si": p.target_value_si}
                t.log(0, p, comp, 0.0, bd, "fake", None, {"split": "x2", "sample": j})


@pytest.fixture
def cfg(tmp_path):
    x2 = copy.deepcopy(X._config())
    x2["sampling"]["output_dir"] = str(tmp_path / "sft_k8")
    x2["output_dir"] = str(tmp_path / "x2")
    _write_chunk(tmp_path / "sft_k8", _pool(24))  # 2 problems per training cell (round-robin pool order)
    c = copy.deepcopy(V._config())
    c["x2_config"] = x2
    c["output_dir"] = str(tmp_path / "x4")
    c["generation"].update(max_new_tokens=6, batch_size=4)
    c["analysis"].update(bootstrap=200)
    return c


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A tiny random Qwen2 with the real tokenizer, saved, and loaded through the eval backend on CPU."""
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        pytest.skip("local tokenizer missing")
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    from physcal.eval.run_eval import HFBackend

    d = tmp_path_factory.mktemp("tiny")
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    torch.manual_seed(0)
    Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                 tie_word_embeddings=True)).save_pretrained(str(d))
    tok.save_pretrained(str(d))
    gen = {"do_sample": False, "num_samples": 1, "max_new_tokens": 6, "batch_size": 4, "repetition_penalty": 1.0,
           "seed": 0, "temperature": None, "top_p": None, "top_k": None}
    return HFBackend(str(d), {"loader": "transformers", "device": "cpu", "dtype": "float32", "model_class": "causal_lm",
                              "lora": None}, gen)


# ------------------------------------------------------------------------------------------- unit tests
def test_the_prompt_is_the_preregistered_one():
    cfg = V._config()
    text = V.check_prompt(cfg)
    assert hashlib.sha256(text.encode()).hexdigest().startswith("5f5fa39d")  # D89
    assert '"Verdict: yes"' in text and '"Verdict: no"' in text and "within 1%" in text and "redo" in text
    bad = copy.deepcopy(cfg)
    bad["verify_prompt_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        V.check_prompt(bad)


def test_reading_text_cuts_before_the_last_verdict():
    rc = V._config()["reading"]
    assert V.reading_text("calc\n<answer>1 ± 2 m</answer>\nVerdict: yes", rc) == \
        ("calc\n<answer>1 ± 2 m</answer>\nVerdict:", "yes")
    assert V.reading_text("calc\n**Verdict:** No", rc) == ("calc\nVerdict:", "no")  # markdown dropped
    assert V.reading_text("calc 3.2", rc) == ("calc 3.2\nVerdict:", None)  # no verdict: appended after all the text
    assert V.reading_text("", rc) == ("Verdict:", None)
    assert V.reading_text("Verdict: yes", rc) == ("Verdict:", "yes")
    echo = 'I end with "Verdict: yes" or "Verdict: no".\nRedo: 4.1 s\nverdict - NO'
    assert V.reading_text(echo, rc) == ('I end with "Verdict: yes" or "Verdict: no".\nRedo: 4.1 s\nVerdict:', "no")
    assert V.reading_text("x\nVerdict:", rc) == ("x\nVerdict:", "")  # a verdict line without a word


@needs_tokenizer
def test_yes_and_no_are_single_distinct_tokens():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    rc = V._config()["reading"]
    yes, no = V.token_ids(tok, rc["yes_tokens"]), V.token_ids(tok, rc["no_tokens"])
    assert len(set(yes + no)) == 12
    assert tok.encode("Verdict: yes", add_special_tokens=False)[-1] == yes[1]  # " yes" follows the marker
    with pytest.raises(ValueError):
        V.token_ids(tok, ["Verdict"])


def test_confidence_is_the_yes_no_log_odds():
    import torch

    x = torch.tensor([0.0, 1.0, 2.0, 3.0, -1.0, 0.5])
    r = V.confidence(x, [1, 3], [2, 5])
    ly, ln = math.log(math.exp(1) + math.exp(3)), math.log(math.exp(2) + math.exp(0.5))
    z = sum(math.exp(v) for v in x.tolist())
    assert r["log_odds"] == pytest.approx(ly - ln) and r["p_yes"] == pytest.approx(1 / (1 + math.exp(ln - ly)))
    assert r["p_yes_raw"] == pytest.approx(math.exp(ly) / z) and r["p_no_raw"] == pytest.approx(math.exp(ln) / z)
    assert r["top_token"] == 3
    far = V.confidence(torch.tensor([0.0, 900.0, -900.0]), [1], [2])  # no overflow, no saturation of the log-odds
    assert far["log_odds"] == pytest.approx(1800.0) and far["p_yes"] == 1.0
    assert V.confidence(torch.tensor([0.0, -900.0, 900.0]), [1], [2])["p_yes"] == 0.0


def test_donors_are_a_same_cell_derangement_of_the_real_x2_problems():
    keys = [tuple(k) for k in json.loads((REPO_ROOT / "results" / "x2" / "selected.json").read_text())["keys"]]
    d = V.donors(keys)
    assert len(keys) == 516 and set(d) == set(keys) and sorted(d.values()) == sorted(keys)  # a bijection
    assert all(d[k] != k and d[k][:2] == k[:2] for k in keys)  # never its own; the same (family, depth) cell
    cell = [k for k in keys if k[:2] == keys[0][:2]]
    assert d[cell[0]] == cell[1] and d[cell[-1]] == cell[0]  # the next one, cyclically
    with pytest.raises(ValueError):
        V.donors([("energy", 1, 1), ("energy", 2, 2)])


def test_the_answers_are_exactly_x2s():
    its, keys = V.items(V._config())
    g, kk = X.kept_groups()
    assert len(keys) == 516 and len(its) == sum(len(g[k]) for k in kk) == 4116
    rank = {k: n for n, k in enumerate(keys)}
    order = [(rank[tuple(i["key"])], i["sample"]) for i in its]
    assert order == sorted(order)  # the kept order, then the sample index
    ans = {(tuple(k), a["sample"]): a["correct"] for k in kk for a in g[k]}
    assert all(ans[(tuple(i["key"]), i["sample"])] == i["correct"] for i in its)
    assert sum(i["correct"] for i in its) == sum(ans.values())


def test_cap_follows_the_wsl_spill_rule():
    c = V._config()
    assert V.cap_gib(c, used_mib=171, total_mib=8188) == round((8188 - 171) / 1024 - 0.5, 2)
    assert V.cap_gib(c, used_mib=1700, total_mib=8188) == round((8188 - 1700) / 1024 - 0.5, 2)
    assert V.cap_gib(c, used_mib=5000, total_mib=8188) == 4.0
    assert V.cap_gib(c, used_mib=0, total_mib=16384) == 7.5


# ------------------------------------------------------------------------------------------- the run, tiny model
def _context_reading(backend, cfg, shown_text, completion, generation):
    """D89's reading recomputed independently: the model's own logits after the canonical context."""
    tok = backend.tok
    prompt = tok.apply_chat_template(V.conversation(shown_text, completion, cfg["prompt"], V.check_prompt(cfg)),
                                      tokenize=False, add_generation_prompt=True)
    text, _ = V.reading_text(generation, cfg["reading"])
    ids = tok(prompt + text)["input_ids"]
    rc = cfg["reading"]
    return V.confidence(V.own_logits(backend.model, ids, "cpu"), V.token_ids(tok, rc["yes_tokens"]),
                        V.token_ids(tok, rc["no_tokens"]))


def test_run_on_a_tiny_cpu_model_resumes_and_matches_an_independent_reading(cfg, tiny):
    first = V.run("sft", cfg, backend=tiny, limit=10)
    assert first == {"arm": "sft", "n_rows": 10, "complete": False, "seconds": first["seconds"], "path": "float32_head"}
    full = V.run("sft", cfg, backend=tiny)
    rows = V.read_rows("sft", cfg)
    assert full["n_rows"] == len(rows) == 96 and full["complete"]
    assert len({(tuple(r["key"]), r["sample"]) for r in rows}) == 96  # resumed, no duplicates
    meta = json.loads((V.arm_dir("sft", cfg) / "meta.json").read_text())
    assert [s["n_done_before"] for s in meta["segments"]] == [0, 10] and meta["complete"]
    assert meta["segments"][0]["logit_check"]["relative_gap"] < 1e-5
    assert V.run("sft", cfg, backend=tiny)["n_rows"] == 96  # nothing left to do
    path = V.arm_dir("sft", cfg) / "rows.jsonl"
    whole = path.read_text()
    path.write_text(whole + whole.splitlines()[0][:40])  # a kill mid-write leaves a partial line
    assert len(V.read_rows("sft", cfg)) == 96 and path.read_text() != whole
    assert V.run("sft", cfg, backend=tiny)["n_rows"] == 96 and path.read_text() == whole  # resume repairs it
    its, _ = V.items(cfg)
    assert [(r["key"], r["sample"], r["correct"]) for r in rows] == [(i["key"], i["sample"], i["correct"]) for i in its]
    for r, it in list(zip(rows, its))[::23]:
        assert r["shown_key"] == r["key"] and r["gen_tokens"] <= 6
        ind = _context_reading(tiny, cfg, it["problem"]["text"], it["completion"], r["generation"])
        assert r["log_odds"] == pytest.approx(ind["log_odds"], abs=1e-4)
        assert r["p_yes_raw"] == pytest.approx(ind["p_yes_raw"], rel=1e-3)


def test_shuffled_arm_shows_the_donor_problem(cfg, tiny):
    V.run("shuffled", cfg, backend=tiny, limit=8)
    rows = V.read_rows("shuffled", cfg)
    its, keys = V.items(cfg)
    d = V.donors(keys)
    probs = {tuple(i["key"]): i["problem"] for i in its}
    assert len(rows) == 8
    for r, it in zip(rows, its):
        assert tuple(r["shown_key"]) == d[tuple(r["key"])] != tuple(r["key"])
        ind = _context_reading(tiny, cfg, probs[tuple(r["shown_key"])]["text"], it["completion"], r["generation"])
        assert r["log_odds"] == pytest.approx(ind["log_odds"], abs=1e-4)


def test_an_oom_backoff_keeps_the_halved_batch(cfg, tiny, monkeypatch):
    sizes = []
    real = tiny._generate

    def gen(prompts, k):
        sizes.append(len(prompts))
        if len(sizes) == 1:  # the eval backoff: an OOM on the first call is split in two
            tiny.oom_backoffs += 1
        return real(prompts, k)

    monkeypatch.setattr(tiny, "_generate", gen)
    monkeypatch.setattr(tiny, "oom_backoffs", 0)
    out = V.run("sft", cfg, backend=tiny, limit=12)
    assert out["n_rows"] == 12 and sizes == [4, 2, 2, 2, 2]
    meta = json.loads((V.arm_dir("sft", cfg) / "meta.json").read_text())
    assert meta["segments"][0]["final_batch_size"] == 2 and meta["segments"][0]["oom_backoffs"] == 1


def test_a_failed_logit_check_falls_back_to_the_models_own_logits(cfg, tiny, monkeypatch):
    import torch

    real = V.head_logits
    monkeypatch.setattr(V, "head_logits", lambda m, h, c: (lambda x: x + 50.0 * torch.linspace(0, 1, x.shape[-1]))(
        real(m, h, c)))  # a head that disagrees with the model's own logits
    out = V.run("random_init", cfg, backend=tiny, limit=4)
    assert out["path"] == "model_logits"
    rows = V.read_rows("random_init", cfg)
    its, _ = V.items(cfg)
    ind = _context_reading(tiny, cfg, its[0]["problem"]["text"], its[0]["completion"], rows[0]["generation"])
    assert rows[0]["path"] == "model_logits" and rows[0]["log_odds"] == pytest.approx(ind["log_odds"], abs=1e-6)


@pytest.mark.parametrize("answer, edited", [
    ("54.74 ± 0 cm/s^2", "68.42 ± 0 cm/s^2"),  # 68.425: the original's 4 significant digits
    ("8541 ± 10 mm", "10676 ± 10 mm"),
    ("v = 12.4 ± 0.8 m/s", "v = 15.5 ± 0.8 m/s"),  # after the last "="
    ("2 ± 1 s", "2.50 ± 1 s"),  # at least 3 significant digits
    ("0.0235 ± 0.0003 m", "0.0294 ± 0.0003 m"),
    ("-3.2 ± 0.1 m", "-4.00 ± 0.1 m"),
    ("1.2e-3 ± 1e-4 A", "1.50e-3 ± 1e-4 A"),  # only the mantissa of an e exponent
    ("3.2 × 10^4 J ± 2%", "4.00 × 10^4 J ± 2%"),  # a following × 10^k is kept; a percent σ stays as written
    ("(1.2 ± 0.1) × 10^3 m", "(1.50 ± 0.1) × 10^3 m"),
])
def test_perturb_scales_only_the_final_mu(answer, edited):
    comp = f"Work: v = 12.4 m/s\n<steps>\nv = 12.4 m/s\n</steps>\n<answer>{answer}</answer>"
    text, info = V.perturb(comp, 1.25, 3)
    assert text == comp.replace(f"<answer>{answer}</answer>", f"<answer>{edited}</answer>")  # steps unchanged
    a0, a1 = V.parse_completion(comp).answer, V.parse_completion(text).answer
    assert a1.value == pytest.approx(1.25 * a0.value, rel=5e-3) and a1.sigma == a0.sigma and a1.unit == a0.unit


def test_perturb_excludes_what_it_cannot_scale():
    assert V.perturb("<answer>0 ± 0 m</answer>", 1.25, 3) == (None, {"reason": "mu_zero", "old": "0"})
    assert V.perturb("no answer", 1.25, 3)[1]["reason"] == "answer_blocks"
    assert V.perturb("<answer>1 m</answer><answer>2 m</answer>", 1.25, 3)[1]["reason"] == "answer_blocks"
    assert V.perturb("<answer>about ten m</answer>", 1.25, 3)[1]["reason"] == "no_number"
    assert V.perturb("<answer>1,250 ± 3 m</answer>", 1.25, 3)[1]["reason"] == "reparse_mismatch"  # thousands sep.


def test_perturb_on_the_real_x2_answers():
    its, _ = V.items(V._config())
    out = [(it, *V.perturb(it["completion"], 1.25, 3)) for it in its]
    bad = [info["reason"] for _, t, info in out if t is None]
    assert bad == ["mu_zero"]  # D91: excluded and counted
    for it, t, info in out[::37]:
        if t is not None:
            i = it["completion"].index("<answer>")
            assert t[:i] == it["completion"][:i] and t != it["completion"]  # only the answer block changes


def test_direct_arm_reads_right_after_the_generation_prompt(cfg, tiny, monkeypatch):
    monkeypatch.setattr(tiny, "_generate", lambda *a: pytest.fail("direct mode must not generate"))
    out = V.run("direct", cfg, backend=tiny, limit=8)
    rows = V.read_rows("direct", cfg)
    its, _ = V.items(cfg)
    assert out["n_rows"] == 8 and len(rows) == 8
    for r, it in zip(rows, its):
        assert r["gen_tokens"] == 0 and r["generation"] == "" and r["mode"] == "direct" and not r["truncated"]
        assert r["context_tokens"] == r["prompt_tokens"] + len(tiny.tok(cfg["reading"]["marker"],
                                                                         add_special_tokens=False)["input_ids"])
        ind = _context_reading(tiny, cfg, it["problem"]["text"], it["completion"], "")
        assert r["log_odds"] == pytest.approx(ind["log_odds"], abs=1e-4)
    meta = json.loads((V.arm_dir("direct", cfg) / "meta.json").read_text())
    assert meta["mode"] == "direct" and meta["perturb_factor"] is None


def test_perturbed_arm_shows_the_scaled_answer_and_keeps_the_label(cfg, tiny):
    V.run("perturbed", cfg, backend=tiny, limit=8)
    rows = V.read_rows("perturbed", cfg)
    its, _ = V.items(cfg)
    assert len(rows) == 8
    for r, it in zip(rows, its):
        text, info = V.perturb(it["completion"], 1.25, cfg["perturb"]["min_sig_digits"])
        assert r["correct"] == it["correct"] and r["mu_text"] == [info["old"], info["new"]] and "mode" not in r
        assert r["shown_key"] == r["key"]
        ind = _context_reading(tiny, cfg, it["problem"]["text"], text, r["generation"])
        assert r["log_odds"] == pytest.approx(ind["log_odds"], abs=1e-4)
    meta = json.loads((V.arm_dir("perturbed", cfg) / "meta.json").read_text())
    assert meta["perturb_factor"] == 1.25 and meta["excluded"] == [] and meta["mode"] == "recompute"


def test_the_sft_rows_are_unchanged_by_d91(cfg, tiny):
    """D91 adds arms; an sft row has exactly D89's fields (the main stage ran on the D89 code)."""
    V.run("sft", cfg, backend=tiny, limit=4)
    r = V.read_rows("sft", cfg)[0]
    assert set(r) == {"arm", "key", "sample", "shown_key", "correct", "log_odds", "p_yes", "p_yes_raw", "p_no_raw",
                      "top_token", "path", "natural_verdict", "prompt_tokens", "gen_tokens", "truncated",
                      "context_tokens", "generation"}
    assert "mode" not in json.loads((V.arm_dir("sft", cfg) / "meta.json").read_text())


# ------------------------------------------------------------------------------------------- the analysis
def _write_rows(cfg, arm, conf, keys=None):
    """Synthetic rows for an arm: conf(item) gives the log-odds."""
    its, _ = V.items(cfg)
    d = V.arm_dir(arm, cfg)
    d.mkdir(parents=True, exist_ok=True)
    with (d / "rows.jsonl").open("w") as f:
        for it in its:
            if keys is not None and tuple(it["key"]) not in keys:
                continue
            lo = conf(it)
            f.write(json.dumps({"arm": arm, "key": it["key"], "sample": it["sample"], "shown_key": it["key"],
                                "correct": it["correct"], "log_odds": lo, "p_yes": 1 / (1 + math.exp(-lo)),
                                "p_yes_raw": 0.5 / (1 + math.exp(-lo)), "p_no_raw": 0.5 / (1 + math.exp(lo)),
                                "top_token": 0, "path": "float32_head",
                                "natural_verdict": "yes" if lo > 0 else None, "prompt_tokens": 10, "gen_tokens": 5,
                                "truncated": False, "context_tokens": 20, "generation": "no answer here"}) + "\n")


def _write_reference(cfg, probe):
    """A fake X2 reference and the X2 results.json it must reproduce."""
    g, keys = X.kept_groups(cfg["x2_config"])
    conf = {(tuple(k), a["sample"]): probe(a) for k in keys for a in g[k]}
    sig = {(tuple(k), a["sample"]): X.stated_sigma_conf(k, a) for k in keys for a in g[k]}
    out = REPO_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "x2_reference.json").write_text(json.dumps({
        "probe": [[list(k), s, v] for (k, s), v in conf.items()],
        "sft_sigma": [[list(k), s, v] for (k, s), v in sig.items()]}))
    pa = float(A.per_problem(g, keys, conf).mean())
    x2out = REPO_ROOT / cfg["x2_config"]["output_dir"]
    x2out.mkdir(parents=True, exist_ok=True)
    (x2out / "results.json").write_text(json.dumps({"methods": {"probe": {"pairwise_accuracy": pa}}}))
    return pa


def test_reading_rules_on_synthetic_rows(cfg):
    rng = np.random.default_rng(0)
    _write_reference(cfg, lambda a: 0.5)
    _write_rows(cfg, "sft", lambda it: 2.0 if it["correct"] else -2.0)  # perfect within problems
    _write_rows(cfg, "shuffled", lambda it: 0.0)  # all ties
    _write_rows(cfg, "random_init", lambda it: float(rng.normal()))
    _write_rows(cfg, "direct", lambda it: 1.0 if it["correct"] else -1.0)  # D91: as good as recompute
    _write_rows(cfg, "perturbed", lambda it: -3.0)  # D91: every perturbed answer rejected alike
    res = A.analyze(cfg)
    assert res["methods"]["sft"]["pairwise_accuracy"] == 1.0 and res["methods"]["shuffled"]["pairwise_accuracy"] == 0.5
    assert res["methods"]["sft"]["n_pairs"] == 24 * 4 and res["analyzed_problems"] == {a: 24 for a in A.ARMS}
    assert res["methods"]["sft"]["auroc_across_problems"]["auroc"] == 1.0
    assert res["holm"]["sft"]["rejected"] and not res["holm"]["shuffled"]["rejected"]
    assert res["paired"]["sft - x2_probe"]["mean_diff"] == pytest.approx(0.5)
    assert res["reading"]["outcome"].startswith("succeeds, problem-specific, above the probe")
    assert res["diagnostics"]["sft"]["natural_yes_rate_correct"] == 1.0
    assert res["secondary"]["sft recomputation agreement (exploratory)"]["share_answers_with_parsed_recomputation"] == 0
    assert "Reading (D89)" in (REPO_ROOT / cfg["output_dir"] / "summary.md").read_text()
    # D91: five arms in Holm; the added flags; detection; the perturbed answers' own correctness; tokens; CIs
    assert set(res["holm"]) == set(A.ARMS) and res["holm"]["direct"]["rejected"]
    assert not res["holm"]["perturbed"]["rejected"] and res["methods"]["perturbed"]["pairwise_accuracy"] == 0.5
    rd = res["reading"]
    assert rd["final_value_sensitive"] and rd["direct_succeeds"] and not rd["recomputation_helps"]
    assert res["paired"]["sft - perturbed"]["mean_diff"] == pytest.approx(0.5)
    assert res["paired"]["sft - direct"]["ci95"] == [0.0, 0.0] and res["paired"]["direct - x2_probe"]["n_problems"] == 24
    assert res["secondary"]["perturbed detection (D91)"]["pairwise_accuracy"] == 1.0
    assert res["perturbed_correctness"] == {"correct_to_correct": 0, "correct_to_incorrect": 48,
                                            "incorrect_to_correct": 0, "incorrect_to_incorrect": 48, "excluded": 0}
    tk = res["tokens"]["sft"]
    assert tk["generated"]["mean"] == 5.0 and tk["generated"]["ci95"] == [5.0, 5.0]
    if tk.get("overhead"):
        assert tk["total_extra"]["mean"] == 5.0 + tk["overhead"]["verification_turn"] + tk["overhead"]["marker"]
    ci = res["diagnostics_ci"]["sft"]
    assert ci["share_truncated"]["ci95"] == [0.0, 0.0] and ci["mean_p_yes_correct"]["mean"] == pytest.approx(
        res["diagnostics"]["sft"]["mean_p_yes_correct"])
    assert "D91 flags" in (REPO_ROOT / cfg["output_dir"] / "summary.md").read_text()

    _write_rows(cfg, "shuffled", lambda it: 2.0 if it["correct"] else -2.0)  # the swap changes nothing
    assert A.analyze(cfg)["reading"]["outcome"].startswith("succeeds but not problem-specific")
    _write_rows(cfg, "sft", lambda it: 0.0)
    assert A.analyze(cfg)["reading"]["outcome"].startswith("sft not Holm-supported")


def test_partial_and_missing_arms(cfg):
    _write_reference(cfg, lambda a: 0.5)
    g, keys = X.kept_groups(cfg["x2_config"])
    _write_rows(cfg, "sft", lambda it: 2.0 if it["correct"] else -2.0)
    _write_rows(cfg, "shuffled", lambda it: 0.0, keys=set(keys[:10]))  # a night that ended early
    rows = V.read_rows("shuffled", cfg)
    (V.arm_dir("shuffled", cfg) / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]))
    res = A.analyze(cfg)
    assert res["analyzed_problems"] == {"sft": 24, "direct": 0, "perturbed": 0, "shuffled": 9,
                                        "random_init": 0}  # only complete problems
    assert res["paired"]["sft - shuffled"]["n_problems"] == 9 and res["paired"]["sft - random_init"] == {"missing": True}
    assert res["methods"]["random_init"] == {"missing": True} and res["holm"]["random_init"]["p"] == 1.0
    assert res["reading"]["outcome"].startswith("another combination")  # supported, but no random-init comparison


def test_the_reference_must_reproduce_x2(cfg):
    _write_reference(cfg, lambda a: 1.0 if a["correct"] else 0.0)
    x2out = REPO_ROOT / cfg["x2_config"]["output_dir"] / "results.json"
    x2out.write_text(json.dumps({"methods": {"probe": {"pairwise_accuracy": 0.9}}}))
    _write_rows(cfg, "sft", lambda it: 0.0)
    with pytest.raises(RuntimeError):
        A.analyze(cfg)


def test_cluster_bootstrap_resamples_problems_not_answers():
    """D91: two problems of ten answers each, all right and all wrong: resampling the two problems gives [0, 1];
    resampling the twenty answers would give a narrow interval around 0.5."""
    r = A.cluster_mean([[1.0] * 10, [0.0] * 10], 4000, 0)
    assert r["mean"] == 0.5 and r["ci95"] == [0.0, 1.0] and r["n_problems"] == 2
    assert A.cluster_mean([[], [2.0, 4.0]], 100, 0) == {"mean": 3.0, "ci95": [3.0, 3.0], "n_problems": 1}
    assert A.cluster_mean([[]], 100, 0)["mean"] is None


def test_an_excluded_perturbed_answer_leaves_its_problem_complete(cfg):
    _write_reference(cfg, lambda a: 0.5)
    g, keys = X.kept_groups(cfg["x2_config"])
    _write_rows(cfg, "sft", lambda it: 2.0 if it["correct"] else -2.0)
    _write_rows(cfg, "perturbed", lambda it: 2.0 if it["correct"] else -2.0)
    rows = V.read_rows("perturbed", cfg)
    gone = rows[1]  # its pairs count 0.5 (D89's missing rule)
    (V.arm_dir("perturbed", cfg) / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if r is not gone))
    (V.arm_dir("perturbed", cfg) / "meta.json").write_text(json.dumps(
        {"excluded": [[gone["key"], gone["sample"], "mu_zero"]], "perturb_factor": 1.25, "mode": "recompute"}))
    res = A.analyze(cfg)
    assert res["analyzed_problems"]["perturbed"] == 24 and res["excluded_answers"] == {"perturbed": 1}
    first = [a for a in g[keys[0]]]
    n_c, n_w = sum(a["correct"] for a in first), sum(not a["correct"] for a in first)
    lost = n_w if gone["correct"] else n_c  # the pairs of the excluded answer
    expect = (23 + (n_c * n_w - lost + 0.5 * lost) / (n_c * n_w)) / 24
    assert res["methods"]["perturbed"]["pairwise_accuracy"] == pytest.approx(expect)
    assert res["meta"]["perturbed"]["n_excluded"] == 1


@needs_tokenizer
def test_token_overhead_is_the_verification_turn_and_the_marker():
    cfg = V._config()
    o = A.token_overhead(cfg)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    n_verify = len(tok(V.check_prompt(cfg), add_special_tokens=False)["input_ids"])
    assert n_verify < o["verification_turn"] <= n_verify + 12  # + the turn's chat markup
    assert o["marker"] == len(tok("Verdict:", add_special_tokens=False)["input_ids"]) >= 1

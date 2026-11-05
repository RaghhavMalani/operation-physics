"""D175's R6-6a: the frozen pools and splits, the reading, the training, the analysis, and a tiny Qwen through every stage
on CPU, with a kill and a resume (D171 rule 9). Nothing here touches a GPU or reads a real sample."""

import copy
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pytest

from physcal.config import REPO_ROOT
from physcal.eval import followup_kaggle as K
from physcal.eval.followup_audit import text_grid as _ORIG_GRID
from physcal.eval import followup_r6 as R6
from physcal.eval import followup_scale as F
from physcal.eval import kaggle_stages as S
from physcal.eval import r6_6a as M

TOK_DIR = REPO_ROOT / F._config()["scale"]["models"]["1.5b"]["dir"]


# ------------------------------------------------------------------------------------------- the pools
@pytest.mark.parametrize("mode,n", [("full", 2400), ("smoke", 48)])
def test_the_frozen_pool_matches_its_manifest_and_its_splits_partition_it(mode, n):
    rc = M.mode_config(mode)
    blob = (REPO_ROOT / rc["candidates"]["file"]).read_bytes()
    man = json.loads(M.manifest_path(rc).read_text())
    assert S.hashlib.sha256(blob).hexdigest() == man["sha256"] and man["n_problems"] == n
    sp = man["splits"]
    assert sorted(i for v in sp.values() for i in v) == list(range(n))
    assert len(sp["train"]) == rc["pool"]["n_train"] and len(sp["validation"]) == rc["pool"]["n_validation"]
    probs = [json.loads(x) for x in blob.decode().splitlines()]
    assert [p["seed"] for p in probs] == [rc["pool"]["seed_offset"] + i for i in range(n)]
    assert {M.question_kind(p["text"]) for p in probs} == {"Find", "What", "Determine"}
    assert {M.question_kind(probs[i]["text"]) for i in sp["held_out"]} == {"Determine"}
    rest = sp["train"] + sp["validation"] + sp["id_test"]
    assert "Determine" not in {M.question_kind(probs[i]["text"]) for i in rest}


def test_no_pool_shares_a_seed_or_text_with_a_frozen_set_or_with_the_other_pool():
    seeds, texts = M._frozen_problems()
    full, smoke = ([json.loads(x) for x in (REPO_ROOT / M.mode_config(m)["candidates"]["file"]).open()] for m in ("full", "smoke"))
    assert not ({p["seed"] for p in full + smoke} & seeds or {p["text"] for p in full + smoke} & texts)
    assert not {p["seed"] for p in full} & {p["seed"] for p in smoke}
    assert not {p["text"] for p in full} & {p["text"] for p in smoke}


def test_the_splits_depend_only_on_the_texts_and_the_seed():
    rc = M.mode_config("smoke")
    texts = [json.loads(x)["text"] for x in (REPO_ROOT / rc["candidates"]["file"]).open()]
    assert M.assign_splits(texts, rc["pool"]) == json.loads(M.manifest_path(rc).read_text())["splits"]


def test_the_overlay_points_the_sampler_at_this_pool_under_its_own_name():
    rc = M.mode_config("smoke")
    c = M.scale_config(rc)
    assert len(F.pool(c)) == 48 and K.max_candidates(c) == 48
    kc = K.kaggle_cfg(c, rc["name"])
    assert kc["repo_id"] == "Qwen/Qwen2.5-Math-1.5B-Instruct" and kc["revision"].startswith("aafeb0fc")
    assert kc["sampling_hours"] == rc["session"]["hours"]


def test_the_smoke_changes_only_what_d175_lets_it():
    full, smoke = M.mode_config("full"), M.mode_config("smoke")
    for key in ("train", "teachers", "analysis", "verify_w1", "readout", "packages", "model_from", "secondary_models"):
        assert full[key] == smoke[key], key


# ------------------------------------------------------------------------------------------- the small pieces
def test_batches_keep_the_token_budget_and_every_index_once():
    lengths = [100, 700, 300, 300, 1000, 50]
    bs = M.batches(lengths, 1000)
    assert sorted(i for b in bs for i in b) == list(range(6))
    assert all(len(b) * max(lengths[i] for i in b) <= 1000 for b in bs)
    assert M.batches([5000], 1000) == [[0]]


def test_the_learning_rate_warms_up_then_decays_to_zero():
    v = [M.lr_scale(s, 100, 5) for s in range(100)]
    assert v[0] == 0.2 and v[4] == 1.0 and v[5] == pytest.approx(1.0) and v[99] == pytest.approx(1 / 95)
    assert all(a >= b for a, b in zip(v[4:], v[5:]))


def test_aurc_and_brier_by_hand():
    c = np.array([True, True, False, False])
    assert M.aurc(np.array([0.9, 0.8, 0.2, 0.1]), c) == pytest.approx((0 + 0 + 1 / 3 + 2 / 4) / 4)
    assert M.aurc(np.array([0.1, 0.2, 0.8, 0.9]), c) == pytest.approx((1 + 1 + 2 / 3 + 2 / 4) / 4)
    assert M.brier(np.array([1.0, 0.5, 0.5, 0.0]), c) == pytest.approx(0.125)


def test_arm_e_shuffles_within_problems_only():
    its = [{"key": ["f", 1, i // 4]} for i in range(12)]
    t = np.arange(12.0)
    s = M.shuffled_within_problems(its, t, 0)
    assert not np.array_equal(s, t)
    for i in range(0, 12, 4):
        assert sorted(s[i:i + 4]) == sorted(t[i:i + 4])


def test_every_contrast_has_three_readings_and_the_ci_picks_one():
    assert M.reading(("C", "D"), {"ci95": [0.01, 0.1]}) == "it moves more than surface"
    assert M.reading(("C", "D"), {"ci95": [-0.01, 0.1]}) == "distillation is surface transfer"
    assert M.reading(("C", "D"), {"ci95": [-0.1, -0.01]}) == "the text target moves more"
    assert "leaks" in M.reading(("E", "A"), {"ci95": [0.01, 0.1]})
    assert set(M.READINGS) == set(M.CONTRASTS)


def _rows(n_problems=40, seed=0):
    """Eight answers per problem: arm C ranks every pair right, A is noise, E is A."""
    rng = np.random.default_rng(seed)
    out = []
    for p in range(n_problems):
        for s in range(8):
            ok = s % 2 == 0
            a = float(rng.normal())
            out.append({"id": f"{p}|{s}", "key": ["f", 1, p], "sample": s, "split": "id_test", "correct": ok, "n_tokens": 10,
                        "conf": {"A": a, "B_0": a, "C_0": 1.0 if ok else -1.0, "D_0": float(rng.normal()), "E_0": a}})
    return out


def test_the_pairwise_block_averages_seeds_and_reads_its_contrasts():
    rows = _rows()
    for r in rows:
        r["conf"]["C_1"] = 0.0 if r["correct"] else 0.0            # all ties: 0.5
    an = {"bootstrap": 200, "seed": 0}
    b = M.pairwise_block(rows, {"A": ["A"], "C": ["C_0", "C_1"], "D": ["D_0"], "E": ["E_0"]}, an)
    assert b["n_problems"] == 40 and b["columns"]["C_0"] == 1.0 and b["columns"]["C_1"] == 0.5
    assert b["arms"]["C"]["pairwise_accuracy"] == 0.75
    assert b["contrasts"]["C - A"]["reading"] == "distillation helps"
    assert b["contrasts"]["E - A"]["diff"] == 0 and b["contrasts"]["E - A"]["reading"] == "as expected"
    assert "reading" not in M.pairwise_block(rows, {"A": ["A"], "C": ["C_0"]}, an, readings=False)["contrasts"]["C - A"]


def test_item_metrics_skip_brier_for_length_and_give_a_ci_against_a():
    rows = _rows(12)
    for r in rows:
        r["conf"]["length"] = -float(r["n_tokens"])
    m = M.item_metrics(rows, {"A": ["A"], "C": ["C_0"], "length": ["length"]}, {"bootstrap": 100, "seed": 0})
    assert "brier" not in m["length"] and "aurc" in m["length"]
    assert m["C"]["aurc"]["value"] == pytest.approx(M.aurc(np.tile([1.0, -1.0], 48), np.tile([True, False], 48)))
    assert m["C"]["aurc - A"]["ci95"][1] < 0


def _swap_rows():
    rows = []
    for p in range(6):
        for s, ok in enumerate((True, True, False, False)):
            rows.append({"id": f"{p}|{s}|swap", "key": ["f", 1, p], "sample": s, "set": "sw_opposite", "correct": ok,
                         "orig_id": f"{p}|{s}", "donor_id": None, "conf": {"A": -1.0 if ok else 1.0}})
            rows.append({"id": f"{p}|{s}|cut", "key": ["f", 1, p], "sample": s, "set": "sw_own_cut", "correct": ok,
                         "orig_id": f"{p}|{s}", "donor_id": None, "conf": {"A": 0.0}})
        rows.append({"id": f"{p}|2|d", "key": ["f", 1, p], "sample": 2, "set": "sw_diff", "correct": False,
                     "orig_id": f"{p}|2", "donor_id": f"{p}|3", "conf": {"A": -1.0}})
    return rows


def test_the_swap_block_reads_the_box_label_and_compares_with_the_cut_text():
    r = M.swap_block(_swap_rows(), ["A"], {"bootstrap": 100, "seed": 0})["models"]["A"]
    assert r["opposite"]["pairwise_accuracy"] == 0.0 and r["reading"] == "tracks the reasoning"
    assert r["diff_value_box_effect"]["diff"] == pytest.approx(float(M.sigmoid(-1.0) - 0.5))


def test_swap_sets_build_d175s_three_texts():
    def it(p, s, ok, box, split="id_test"):
        comp = f"work {p} {s} so \\boxed{{{box}}}. tail"
        i = {"id": f"m|{p}|{s}", "key": ["f", 1, p], "sample": s, "correct": ok, "completion": comp, "text": "q", "split": split,
             "frame": "native", "n_tokens": 5, "truncated": False, "source": "m", "value": float(box)}
        p1, p2 = R6.native_positions(comp)
        return i | {"p1": p1, "p2": p2}
    its = [it(1, 0, True, 5), it(1, 1, False, 7), it(1, 2, False, 9), it(2, 0, True, 5), it(2, 1, True, 5), it(3, 0, True, 1),
           it(3, 1, False, 2), it(4, 0, True, 1, "train"), it(4, 1, False, 2, "train")]
    rc = {"swap_problems": 1, "name": "m"}
    sw = M.swap_sets(its, rc, 0.01)
    assert {t["key"][2] for v in sw.values() for t in v} == {1}             # the first ID-test problem with both classes
    assert len(sw["sw_opposite"]) == 3 and len(sw["sw_own_cut"]) == 3 and len(sw["sw_diff"]) == 2
    cut = sw["sw_own_cut"][0]
    assert cut["completion"] == "work 1 0 so \\boxed{5}" and cut["orig_id"] == "m|1|0"
    assert all(t["completion"].endswith("}") for t in sw["sw_diff"] + sw["sw_opposite"])


# ------------------------------------------------------------------------------------------- a tiny Qwen through it all
class Tiny:
    """A callable like Ctx.load: a fresh tiny fp32 Qwen2 and the real tokenizer, on CPU."""

    def __init__(self):
        import torch
        import transformers

        if not (TOK_DIR / "tokenizer.json").exists():
            pytest.skip("no local tokenizer")
        self.tok = transformers.AutoTokenizer.from_pretrained(str(TOK_DIR))
        torch.manual_seed(0)
        cfg = transformers.Qwen2Config(vocab_size=len(self.tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
        self.model = transformers.Qwen2ForCausalLM(cfg).eval()
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return copy.deepcopy(self.model)


@pytest.fixture(scope="module")
def tiny():
    return Tiny()


class Engine:
    """Fake vLLM: a sample's correctness is random per (problem, sample), so most problems have both classes."""

    def __init__(self, pool, k):
        self.pool, self.k = pool, k

    def __call__(self, prompt_ids, seeds):
        out = []
        for s in seeds:
            i, j = divmod(s, self.k)
            p = self.pool[i]
            r = np.random.default_rng(s)
            ok = r.random() < 0.5
            v = p["target_value"] * (1 if ok else 1.1 + 0.1 * (j % 3))
            word = "carefully" if (ok and r.random() < 0.8) else "quickly"
            out.append((f"We work {word} through step {j} of this.\nThe result is \\boxed{{{v:.6g}}}.", 30, "stop"))
        return out


def _rc():
    rc = copy.deepcopy(M.mode_config("smoke"))
    rc["analysis"]["bootstrap"] = 50
    rc["readout"].update(part_size=64, token_budget=4096)
    rc["train"].update(batch_size=16, token_budget=4096, log_every=5)
    rc["teachers"]["n_jobs"] = 1
    return rc


def _ctx(root, tiny, rc=None, worker="w0", clock=None, session_start=1000.0):
    rc = rc or _rc()
    sc = M.scale_config(rc)
    sc["scale"]["probe"].update(C=[0.1], folds=2)
    sc["audit"]["text"]["inner_folds"] = 2
    run = S.Run(root, {"commit": "t"}, session_start, rc["session"]["hours"], rc["session"]["rate_margin"], worker,
                clock or (lambda: 1000.0))
    return M.Ctx(Path(root), rc, sc, run, None, "cpu", tiny, lambda m: None)


def _sample(ctx):
    pool = F.pool(ctx.scfg)
    k = ctx.scfg["scale"]["generation"]["k"]
    ticks = itertools.count()
    assert K.sample(lambda: Engine(pool, k), ctx.load.tok, ctx.root, 1000.0, ctx.scfg, log=lambda m: None, model=ctx.rc["name"],
                    now=lambda: 1000.0 + next(ticks)) == 0


@pytest.fixture(scope="module")
def full_run(tmp_path_factory, tiny):
    """Every stage in order on the fake samples, through drive(), with a fake launch: the stages run in this process."""
    import physcal.eval.followup_audit as AU

    root = tmp_path_factory.mktemp("run") / "r6_6a_smoke"
    ctx = _ctx(root, tiny)
    mp = pytest.MonkeyPatch()
    mp.setattr(AU, "text_grid", lambda a5: _ORIG_GRID(a5)[:1])     # one candidate: the grid is not under test here
    calls = []

    def launch(stage, arms):
        calls.append((stage, arms))
        if stage == M.S1:
            _sample(ctx)
        elif stage == M.S2:
            M.stage_read(ctx)
        elif stage == M.S3:
            M.stage_teachers(ctx)
        elif stage == M.S4:
            M.stage_train(ctx, arms)
        else:
            M.stage_score(ctx)
        return [0]
    status = M.drive(ctx, launch, lambda m: None)
    mp.undo()
    return ctx, status, calls


def test_drive_runs_every_stage_once_in_order_and_finishes(full_run):
    ctx, status, calls = full_run
    assert status["stopped_at"] is None and status["done"] == [M.S2, M.S3, M.S4, M.S5]
    assert [c[0] for c in calls] == [M.S1, M.S2, M.S3, M.S4, M.S4, M.S5] or \
        [c[0] for c in calls] == [M.S1, M.S2, M.S4, M.S3, M.S4, M.S5]
    assert (M.S4, ("B",)) in calls and (M.S4, ("C", "D", "E")) in calls


def test_a_second_session_after_the_end_repeats_no_stage(full_run):
    ctx, _, _ = full_run
    calls = []
    st = M.drive(ctx, lambda s, a: calls.append(s) or [0], lambda m: None)
    assert calls == [] and st["stopped_at"] is None


def test_states_and_log_odds_are_saved_for_every_answer_and_are_finite(full_run):
    ctx, _, _ = full_run
    its = M.items(ctx)
    assert len(its) == 48 * 8 and set(M.read_table(ctx)) == {it["id"] for it in its}
    L = M.load_layers(ctx, [its[3]["id"], its[0]["id"]])
    assert len(L) == 3 and L[0].shape == (2, 32) and L[0].dtype == np.float16
    A = M.load_layers(ctx, [its[0]["id"]], "answer_only")
    assert np.isfinite(A[0]).all() and not np.array_equal(A[2][0], L[2][1])


def test_the_saved_state_is_the_brace_token_of_an_unpadded_pass_and_the_log_odds_match_x4(full_run, tiny):
    import torch

    from physcal.eval import x4_verify as X4

    ctx, _, _ = full_run
    it = M.items(ctx)[5]
    model = tiny()
    r = M.make_reader(tiny.tok, model, ctx.scfg, "cpu")
    ids, pos = M.context(it, r, ctx.scfg, F.verify_text(ctx.scfg))
    assert "}" in tiny.tok.decode(ids[pos])
    with torch.inference_mode():
        out = model.model(input_ids=torch.tensor([ids[:pos + 1]]), output_hidden_states=True)
        logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
    saved = M.load_layers(ctx, [it["id"]])
    assert np.allclose(saved[2][0].astype(float), out.hidden_states[2][0, -1].numpy(), atol=2e-2)
    want = X4.confidence(logits, r.yes, r.no)["log_odds"]
    assert M.read_table(ctx)[it["id"]] == pytest.approx(want, abs=1e-3)


def test_targets_follow_the_arms_and_test_problems_never_enter_them(full_run):
    ctx, _, _ = full_run
    its = M.train_items(ctx)
    assert {it["split"] for it in its} == {"train"}
    b, c, d, e = (M.targets(ctx, a, its) for a in "BCDE")
    assert set(b) <= {0.0, 1.0} and 0 < c.min() and c.max() < 1 and not np.array_equal(c, d)
    assert sorted(e) == sorted(c) and not np.array_equal(e, c)
    ids = M.teacher_values(ctx, M.FOLDS_PROBE, ctx.scfg["scale"]["probe"]["folds"])
    assert set(ids) == {it["id"] for it in M.in_split(M.items(ctx), "train")}
    fin = json.loads(ctx.run.path(M.S3, "final_probe.json").read_text())
    assert not set(fin["ids"]) & set(ids)


def test_the_teacher_folds_equal_x2s_crossfit(full_run):
    from physcal.eval import followup_audit as AU
    from physcal.eval import x2_analysis as XA

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    L = [rng.standard_normal((60, 4)).astype(np.float16) + y[:, None] for _ in range(3)]
    keys = [["f", 1, i // 6] for i in range(60)]
    index = [{"key": k, "sample": i, "correct": bool(v)} for i, (k, v) in enumerate(zip(keys, y))]
    pc = {"C": [0.1, 1.0], "folds": 3, "max_iter": 200, "n_jobs": 1}
    want, _ = XA.crossfit_confidence(L, index, pc)
    groups = ["|".join(map(str, k)) for k in keys]
    got = np.full(60, np.nan)
    for a, b in AU._folds(groups, 3):
        k, f, _ = M.probe_fit([x[a] for x in L], y[a], [groups[i] for i in a], pc)
        got[b] = f(L[k][b])
    assert np.allclose(got, [want[(tuple(keys[i]), i)] for i in range(60)])


def test_training_moves_the_log_odds_and_the_saved_adapter_reloads_exactly(full_run, tiny):
    import torch
    from peft import get_peft_model
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

    ctx, _, _ = full_run
    with ctx.run.path(M.S4, "B_0.pt").open("rb") as fh:
        r = torch.load(fh, weights_only=False)
    assert r["steps"] == -(-len(M.train_items(ctx)) // 16) and r["log"][-1]["step"] == r["steps"] - 1
    assert all(np.isfinite(x["loss"]) for x in r["log"])
    model = get_peft_model(tiny(), M.lora_config(ctx.rc)).eval()
    set_peft_model_state_dict(model, r["adapter"])
    back = get_peft_model_state_dict(model)
    assert set(back) == set(r["adapter"]) and all(torch.equal(back[k], r["adapter"][k]) for k in back)
    a = M.read_table(ctx)
    scored = json.loads(ctx.run.path(M.S5, "B_0__id_test.json").read_text())
    assert max(abs(a[i] - v) for i, v in scored.items()) > 1e-3


def test_arm_a_read_through_the_disabled_adapter_equals_a_plain_base_reading(full_run, tiny):
    ctx, _, _ = full_run
    got = json.loads(ctx.run.path(M.S5, "A__sw_own_cut.json").read_text())
    cut = M.swap_sets(M.items(ctx), ctx.rc, ctx.scfg["scale"]["tolerance"])["sw_own_cut"]
    model = tiny()
    r = M.make_reader(tiny.tok, model, ctx.scfg, "cpu")
    lo, _ = M.read_log_odds(model, cut, r, ctx.scfg, F.verify_text(ctx.scfg), 4096, "cpu")
    assert {t["id"] for t in cut} == set(got)
    assert np.allclose([got[t["id"]] for t in cut], lo, atol=1e-4)


def test_results_hold_every_registered_block_and_the_zip_round_trips_through_the_ingest(full_run, tmp_path):
    ctx, _, _ = full_run
    res = json.loads((ctx.root / "s6_analysis" / "results.json").read_text())
    for name in ("id_test", "held_out"):
        p = res[name]["pairwise"]
        assert set(p["arms"]) == {"A", "B", "C", "D", "E", *M.REFERENCES}
        assert set(p["contrasts"]) == {"B - A", "C - A", "C - B", "C - D", "E - A"}
        assert all("reading" in c for c in p["contrasts"].values())
        assert "brier" in res[name]["items"]["C"] and "aurc - A" in res[name]["items"]["C"]
    assert "reading" not in res["held_out_wording"]["contrasts"]["C - D"]
    assert set(res["swap"]["models"]) == {"A", "B_0", "C_0", "D_0"}
    assert res["counts"]["train"]["samples"] == 22 * 8
    assert set(res["validation_bce"]) == {"A", "B_0", "C_0", "D_0", "E_0"}
    z = M.pack(ctx, tmp_path)
    assert z["mb"] < 5
    path = tmp_path / ctx.rc["results"]["zip_name"]
    files = S.read_zip(path, re.compile(M.OK_NAMES), 40)
    again = json.loads(json.dumps(M.results_from_tables(M._gz_rows(files["scores.jsonl.gz"]), M._gz_rows(files["swap_scores.jsonl.gz"]),
                                                        res["counts"], ctx.rc), default=float))
    assert again == res


def test_the_ingest_refuses_results_that_do_not_follow_from_their_tables(full_run, tmp_path, monkeypatch):
    ctx, _, _ = full_run
    M.pack(ctx, tmp_path)
    src = tmp_path / ctx.rc["results"]["zip_name"]
    monkeypatch.setattr(M, "mode_config", lambda m: ctx.rc)
    files = S.read_zip(src, re.compile(M.OK_NAMES), 40)
    assert M.ingest(src, "smoke", tmp_path / "ok")["results_recomputed"]
    res = json.loads(files["results.json"])
    res["id_test"]["pairwise"]["arms"]["C"]["pairwise_accuracy"] += 0.01
    files["results.json"] = json.dumps(res).encode()
    bad = tmp_path / "bad.zip"
    S.write_zip(bad, files, 50)
    with pytest.raises(SystemExit, match="does not follow"):
        M.ingest(bad, "smoke", tmp_path / "no")


def test_a_deadline_stops_a_stage_with_a_status_and_the_next_session_resumes_it(tmp_path, tiny, monkeypatch):
    """A session that runs out of time in S2: drive() stops there; the next session's Run carries on from the saved parts."""
    root = tmp_path / "r6_6a_smoke"
    rc = _rc()
    clock = {"t": 1000.0}
    ctx = _ctx(root, tiny, rc, clock=lambda: clock["t"])
    _sample(ctx)
    ctx.run.t_end = 1000.0 + 300.0                         # the clock moves 200 s per part: room for two of six
    real = ctx.run.save

    def slow_save(stage, key, write):
        real(stage, key, write)
        clock["t"] += 200.0
    ctx.run.save = slow_save

    def launch(stage, arms):
        assert stage == M.S2
        M.stage_read(ctx)
        return [0]
    st = M.drive(ctx, launch, lambda m: None)
    assert st["stopped_at"] == M.S2 and not ctx.run.done(M.S2) and 0 < len(ctx.run.ledger(M.S2)) < 6
    done = set(ctx.run.ledger(M.S2))
    json.loads((root / "status.json").read_text())
    ctx2 = _ctx(root, tiny, rc, clock=lambda: 5000.0, session_start=5000.0)
    assert M.stage_read(ctx2) and done < set(ctx2.run.ledger(M.S2))
    keys = M.part_keys(len(M.items(ctx2)), rc["readout"]["part_size"])
    assert ctx2.run.finish(M.S2, keys)["chunks"].keys() == set(keys)


def test_a_non_finite_state_is_a_stop_rule():
    with pytest.raises(S.StageStop):
        M._finite(np.array([1.0, np.inf]), "state")


def test_the_status_of_a_stopped_rule_is_an_exit_code(tmp_path, monkeypatch):
    def boom(*a):
        raise S.StageStop("non-finite loss at step 3")
    monkeypatch.setattr(M, "stage_read", boom)
    monkeypatch.setattr(M, "make_ctx", lambda *a: M.Ctx(tmp_path, {}, {}, None))
    a = type("A", (), dict(stage=M.S2, mode="smoke", out=tmp_path, session_start=1.0, model_dir=None, worker=0, workers=1))
    assert M.worker_cli(a) == M.EXIT_STOP


def test_the_smoke_report_projects_the_full_run_from_measured_rates_and_lists_what_is_missing(full_run, tmp_path):
    ctx, _, _ = full_run
    sd = K.scale_dir(ctx.root, ctx.rc["name"])
    tp = json.loads((sd / "throughput.json").read_text())
    p = M.projection(ctx)
    f = ctx.rc["smoke"]["full"]
    assert p["missing"] == [] and p["hours"]["S1 sampling"] == pytest.approx(tp["s_per_problem"] * f["problems"] / 3600)
    assert p["hours"]["S4 training"] == pytest.approx(p["s_per_step_median"] * f["train_steps"] * f["runs"] / 2 / 3600)
    assert len(p["train"]) == 4 and p["train"][0]["adapter_mb"] > 0 and set(p["teacher_chunks_s"]) >= {"final_probe.json"}
    early = _ctx(tmp_path / "r6_6a_smoke", ctx.load)        # a session stopped before any stage ran
    assert M.projection(early)["missing"] == ["S1 sampling", "S2 reading", "S4 training", "S5 scoring"]


def test_a_second_smoke_session_keeps_the_first_machine_report_and_counts_what_survived(tmp_path, monkeypatch):
    rc = _rc()
    root = tmp_path / rc["name"]
    ctx = M.Ctx(root, rc, {}, S.Run(root, {"t": 1}, 0.0, 1.0, 1.25, now=lambda: 0.0))
    monkeypatch.setattr(M, "machine_report", lambda: {"cpu_count": 4})
    monkeypatch.setattr(M, "cpu_rate", lambda r: {"seconds": 1})
    monkeypatch.setattr(M, "projection", lambda c: {"hours": {}})
    M.smoke_report(ctx)
    M.smoke_report(ctx)
    assert (root / "smoke" / "machine.json").exists() and (root / "smoke" / "machine_second.json").exists()
    (root / "filecount" / "f_0000").unlink()
    r = M.survival_report(root, rc, {"restored": 3})
    assert r["filecount_files"] == rc["smoke"]["file_test_n"] - 1 and r["filecount_expected"] == rc["smoke"]["file_test_n"]
    assert re.fullmatch(M.OK_NAMES, "smoke/machine_second.json") and re.fullmatch(M.OK_NAMES, "smoke/survival.json")


def test_the_machine_report_and_cpu_rate_run_here():
    m = M.machine_report()
    assert m["cpu_affinity"] >= 1 and m["mem_total"]
    r = M.cpu_rate({"smoke": {"cpu_shape": [200, 16], "cpu_fits": 2}})
    assert r["fits"] == 2 and r["seconds"] > 0


def test_every_command_the_driver_and_the_notebook_build_parses():
    a = M.parser().parse_args(["run", "--mode", "smoke", "--out", "/w", "--model-dir", "/m", "--session-start", "12.5"])
    for stage, extra in ((M.S2, None), (M.S3, None), (M.S4, ["--arms", "B"]), (M.S5, None)):
        b = M.parser().parse_args(M.worker_cmd(stage, a, 1, 2, extra)[3:])
        assert b.action == "worker" and b.stage == stage and b.worker == 1 and b.workers == 2 and b.session_start == 12.5
    assert M.parser().parse_args(M.worker_cmd(M.S4, a, 0, 2, ["--arms", "C,D,E"])[3:]).arms == "C,D,E"
    for line in ("restore --out /w --src /kaggle/input --mode smoke", "download --model-dir /m --mode full", "pack --out /w --mode full"):
        M.parser().parse_args(line.split())


def test_spawn_returns_exit_codes_and_kills_a_silent_child(tmp_path):
    import sys

    log = tmp_path / "l.log"
    ok = [sys.executable, "-c", "print('hello', flush=True)"]
    bad = [sys.executable, "-c", "import sys; sys.exit(3)"]
    assert M.spawn([(ok, {}), (bad, {})], log, 60, lambda m: None) == [0, 3]
    assert "hello" in log.read_text()
    lines = []
    assert M.spawn([([sys.executable, "-c", "import time; time.sleep(120)"], {})], log, 1, lines.append) == [-9]
    assert any("STALL" in x for x in lines)


# ------------------------------------------------------------------------------------------- the notebooks
PINNED = ("physcal/eval/r6_6a.py", "physcal/eval/kaggle_stages.py", "physcal/eval/followup_kaggle.py",
          "physcal/eval/followup_scale.py", "physcal/eval/followup_r6.py", "physcal/eval/followup_r6_controls.py",
          "physcal/eval/followup_audit.py", "physcal/eval/x2_analysis.py", "physcal/eval/x2_hidden.py",
          "physcal/eval/x4_verify.py", "physcal/eval/nested_baselines.py", "configs/r6_6a.yaml", "configs/r6_6a_smoke.yaml",
          "configs/followup.yaml", "data/eval_sets/1.1.0+c735a0d4/r6_6a_pool.jsonl",
          "data/eval_sets/1.1.0+c735a0d4/r6_6a_pool_manifest.json", "data/eval_sets/1.1.0+c735a0d4/r6_6a_smoke_pool.jsonl",
          "data/eval_sets/1.1.0+c735a0d4/r6_6a_smoke_pool_manifest.json")


def _code(name):
    nb = json.loads((REPO_ROOT / "notebooks" / name).read_text())
    assert nb["metadata"]["kaggle"]["accelerator"] == "nvidiaTeslaT4" and nb["metadata"]["kaggle"]["isInternetEnabled"]
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.mark.parametrize("name,mode", [("kaggle_r6_6a.ipynb", "full"), ("kaggle_r6_6a_smoke.ipynb", "smoke")])
def test_the_notebook_compiles_and_its_pinned_commit_holds_this_code_and_d175(name, mode):
    import subprocess

    code = _code(name)
    for c in code:
        compile(c, name, "exec")
    commit = re.search(r'^COMMIT = "([0-9a-f]{40})"$', code[0], flags=re.M).group(1)
    assert f'MODE = "{mode}"' in code[0]

    def pinned(path):
        return subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{commit}:{path}"], capture_output=True, check=True).stdout
    for path in PINNED:
        assert pinned(path) == (REPO_ROOT / path).read_bytes(), f"{path} changed since the notebook's commit {commit[:12]}"
    assert b"### D175." in pinned("docs/exploratory/DECISIONS_FOLLOWUP.md")


def test_only_the_smoke_notebook_crashes_on_purpose():
    assert not any("SIGKILL" in c or "deliberate" in c for c in _code("kaggle_r6_6a.ipynb"))
    last = _code("kaggle_r6_6a_smoke.ipynb")[-1]
    assert "raise RuntimeError" in last and "signal.SIGKILL" in last and "survival.json" in last


def test_the_notebooks_sparse_checkout_holds_what_a_run_reads():
    repo_cell = _code("kaggle_r6_6a.ipynb")[1]
    for mode in ("full", "smoke"):
        rc = M.mode_config(mode)
        assert rc["candidates"]["file"] in repo_cell and M.manifest_path(rc).name in repo_cell


def test_the_packages_are_pinned_and_hold_what_a_run_imports():
    pk = M.mode_config("full")["packages"]
    assert all(re.fullmatch(r"[a-z-]+==[0-9.]+", p) for p in pk)
    assert {p.split("==")[0] for p in pk} >= {"vllm", "peft", "scikit-learn", "scipy", "joblib", "pint", "pyyaml", "sympy"}

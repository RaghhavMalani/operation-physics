"""X3 (D79, pre-registered): the distillation capacity pilot. Can the RL runs' own setup (the merged SFT model loaded
4-bit, plus a fresh LoRA r = 16) express a σ that ranks errors when SFT tells it the target?

    uv run python -m physcal.train.x3 data                               # CPU: rows, OOF A5 teacher σ, oracle σ
    uv run python -m physcal.train.x3 train --arm a [--minutes M] [--max-steps N]          # GPU
    uv run python -m physcal.train.x3 eval --arm a --set eval|x2         # GPU: greedy σ suffixes, μ fixed
    uv run python -m physcal.train.x3 analyze                            # CPU: D79's metrics and reading labels

Rows are D69's 4-bit SFT calibration completions. The context is the prompt plus the completion up to and including
"±" (no loss); the target is the rest of the answer with the σ number replaced (loss), as in E1. Student (a) learns
the strongest 4-bit A5 σ predicted out of fold (A5's own CV folds and scale); student (b) learns the σ that
minimizes each completion's own crps_log given its realized error, |ln(μ/truth)| / √(ln 2). Every target is read back
through the verifier: μ must be unchanged and σ/μ must equal the target σ_log up to the `.2g` rounding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np

from physcal.config import REPO_ROOT, load_config

ORACLE_Z = math.sqrt(math.log(2.0))  # argmin over σ of CRPS(N(0, σ²), e) is σ = |e| / √(ln 2)
SIGMA_NUM_RE = re.compile(r"^(\s*)([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")
CHECKED_KEYS = ("mean_crps_log", "spearman_sigma_error", "coverage_1sigma", "coverage_2sigma")


def _config() -> dict:
    return load_config("x3")


def oracle_sigma_log(mu: float, truth: float, floor: float) -> float:
    """Student (b)'s target: the σ_log minimizing crps_log of this one answer given its realized log error."""
    return max(abs(math.log(mu / truth)), floor) / ORACLE_Z


# ------------------------------------------------------------------------------------------- data
def teacher_oof(cfg: dict) -> tuple[list[dict], list[dict], np.ndarray, dict]:
    """The calibration tape rows, their A5 records, and the strongest A5 σ_log predicted out of fold with A5's own
    folds and scale c, for the answered (`ok`) records in A5's order. The check compares the result with A5's
    recorded CV metrics for that candidate (it must reproduce them)."""
    from sklearn.model_selection import GroupKFold

    from physcal.eval import nested_baselines as nb
    from physcal.tape.tape import read_meta, read_tape

    a5 = nb._config(cfg["a5_config"])
    sel = json.loads((REPO_ROOT / a5["output_dir"] / "selection.json").read_text())
    name = sel["strongest"]["sigma"]
    entry = sel["cv_table"]["sigma"][name]
    tapes = [REPO_ROOT / t for t in a5["fit_tapes"]]
    sc = read_meta(tapes[0])["reward_config"]["scoring"]
    tape_rows, recs = [], []
    for t in tapes:
        for r in read_tape(t):
            tape_rows.append(r)
            recs.append(nb.record(r["problem"], r["completion"], r["breakdown"], sc["binary_tolerance"],
                                  r.get("split", "all")))
    ok = [i for i, r in enumerate(recs) if r["ok"]]
    ok_recs = [recs[i] for i in ok]
    y = nb.log_error(ok_recs, a5["log_error_floor"])
    folds = list(GroupKFold(a5["cv"]["folds"]).split(ok_recs, y, [r["template"] for r in ok_recs]))
    fs, kind = name.split("/")
    oof = np.zeros(len(ok_recs))
    for tr, te in folds:
        cand = nb.Candidate(fs, kind, "sigma", entry["hp"], a5).fit([ok_recs[i] for i in tr], y[tr])
        oof[te] = cand.raw([ok_recs[i] for i in te])
    sigma_log = entry["c"] * np.exp(oof)
    got = nb.sigma_summary(ok_recs, sigma_log, sc)
    diffs = {k: abs(got[k] - entry["cv"][k]) for k in CHECKED_KEYS}
    check = {"candidate": name, "hp": entry["hp"], "c": entry["c"], "reproduced": got, "recorded": entry["cv"],
             "max_abs_diff": max(diffs.values()), "diffs": diffs}
    return [tape_rows[i] for i in ok], ok_recs, sigma_log, check


def replace_sigma(prefix: str, suffix: str, sigma_log: float, fmt: str) -> str | None:
    """The suffix after "±" with its σ number replaced by σ_log in the answer's own form: a percentage if the
    original σ was one, otherwise σ_log·|μ| in the answer's unit. None if the suffix has no leading number."""
    from physcal.verify.parse import ANSWER_RE, parse_value

    m = SIGMA_NUM_RE.match(suffix)
    if m is None:
        return None
    rest = suffix[m.end():]
    if rest.lstrip().startswith("%"):
        value = 100.0 * sigma_log
    else:
        block = ANSWER_RE.search(prefix + suffix)
        mu = parse_value(block.group(1)).value if block else None
        if not mu:
            return None
        value = sigma_log * abs(mu)
    return m.group(1) + format(value, fmt) + rest


def build_data() -> dict:
    """results/x3/data/rows.jsonl (both arms' targets per calibration completion) and data_report.json."""
    from scipy.stats import spearmanr

    from physcal.env.generator import problem_from_dict
    from physcal.train.e1 import truncate_after_pm
    from physcal.verify.reward import reward, reward_config

    cfg = _config()
    tape_rows, recs, sigma_a, check = teacher_oof(cfg)
    if check["max_abs_diff"] > 1e-6:
        raise RuntimeError(f"the out-of-fold teacher does not reproduce A5's CV metrics: {check['diffs']}")
    rw_cfg = reward_config()
    rw_cfg["scoring"]["rule"] = "crps_log"
    excluded = {"no_marker": 0, "no_sigma_number": 0, "readback_mu_changed": 0, "readback_sigma_off": 0}
    rows, readback = [], {"a": [], "b": []}
    for r, rec, s_a in zip(tape_rows, recs, sigma_a):
        prefix = truncate_after_pm(r["completion"])
        if prefix is None:
            excluded["no_marker"] += 1
            continue
        suffix = r["completion"][len(prefix):]
        b = r["breakdown"]
        targets = {"a": float(s_a), "b": oracle_sigma_log(b["mu_si"], b["truth_si"], cfg["oracle_floor"])}
        new = {arm: replace_sigma(prefix, suffix, s, cfg["sigma_format"]) for arm, s in targets.items()}
        if any(v is None for v in new.values()):
            excluded["no_sigma_number"] += 1
            continue
        problem = problem_from_dict(r["problem"])
        reasons = []
        for arm, sfx in new.items():
            _, bd = reward(problem, prefix + sfx, rw_cfg)
            if bd["reason"] != "ok" or abs(bd["mu_si"] / b["mu_si"] - 1.0) > 1e-9:
                reasons.append("readback_mu_changed")
                continue
            rel = abs(bd["sigma_si"] / abs(bd["mu_si"]) / targets[arm] - 1.0)
            readback[arm].append(rel)
            if rel > cfg["target_check_rtol"]:
                reasons.append("readback_sigma_off")
        if reasons:
            excluded[reasons[0]] += 1
            continue
        rows.append({"key": list(rec["key"]), "split": rec["split"], "template": rec["template"],
                     "problem_json": json.dumps(r["problem"]), "prefix": prefix, "suffix_orig": suffix,
                     "suffix_a": new["a"], "suffix_b": new["b"], "sigma_log_a": targets["a"],
                     "sigma_log_b": targets["b"], "abs_log_error": abs(math.log(b["mu_si"] / b["truth_si"])),
                     "correct": rec["correct"], "mu_si": b["mu_si"], "truth_si": b["truth_si"]})
    out = REPO_ROOT / cfg["data_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    err = [x["abs_log_error"] for x in rows]
    q = lambda v: {p: float(np.percentile(v, p)) for p in (5, 25, 50, 75, 95)}
    report = {"decision": "D79 (X3)", "n_calibration_rows": None, "n_answered": len(recs), "n_rows": len(rows),
              "excluded": excluded, "teacher_check": check,
              "readback_max_rel_diff": {a: float(max(v)) if v else None for a, v in readback.items()},
              "sigma_log_quantiles": {a: q([x[f"sigma_log_{a}"] for x in rows]) for a in ("a", "b")},
              "spearman_target_vs_error_on_rows": {
                  a: float(spearmanr([x[f"sigma_log_{a}"] for x in rows], err).statistic) for a in ("a", "b")},
              "share_correct": float(np.mean([x["correct"] for x in rows])),
              "seeds": [min(x["key"][2] for x in rows), max(x["key"][2] for x in rows)]}
    from physcal.eval import nested_baselines as nb

    report["n_calibration_rows"] = sum(1 for t in nb._config(cfg["a5_config"])["fit_tapes"]
                                       for _ in (REPO_ROOT / t).open())
    (out / "data_report.json").write_text(json.dumps(report, indent=1))
    return report


def read_data() -> list[dict]:
    return [json.loads(line) for line in (REPO_ROOT / _config()["data_dir"] / "rows.jsonl").open()]


def tokenize_row(row: dict, arm: str, tok, prompt_id: str, chat_kwargs: dict) -> dict:
    """Context = chat prompt + completion up to "±" (tokenized as one string, exactly as the eval feeds it; labels
    -100); target = the arm's suffix + end-of-turn (labels = ids)."""
    from physcal.prompt import build_messages

    problem = json.loads(row["problem_json"])
    prompt = tok.apply_chat_template(build_messages(problem["text"], prompt_id), tokenize=False,
                                     add_generation_prompt=True, **chat_kwargs)
    ctx = tok(prompt + row["prefix"], add_special_tokens=False)["input_ids"]
    tgt = tok(row[f"suffix_{arm}"], add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids("<|im_end|>")]
    return {"input_ids": ctx + tgt, "attention_mask": [1] * (len(ctx) + len(tgt)), "labels": [-100] * len(ctx) + tgt,
            "n_target": len(tgt)}


# ------------------------------------------------------------------------------------------- training
def fit(model, tok, rows: list[dict], arm: str, out: Path, cfg: dict, max_steps: int | None = None,
        minutes: float | None = None, bf16: bool = True, optim: str | None = None) -> dict:
    """SFT of `model` (a PEFT model) on the arm's targets; saves out/final_adapter. Device-agnostic (CPU tests pass
    a tiny model, bf16=False and a torch optimizer)."""
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    from physcal.train.grpo import make_stop_callback

    t, g = cfg["train"], cfg["guards"]
    data = [tokenize_row(r, arm, tok, cfg["prompt"], cfg["chat_template_kwargs"]) for r in rows]
    random.Random(t["seed"]).shuffle(data)
    ds = Dataset.from_list([{k: d[k] for k in ("input_ids", "attention_mask", "labels")} for d in data])
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"), per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"], learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"], warmup_steps=t["warmup_steps"], weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"], optim=optim or t["optim"], logging_steps=t["logging_steps"],
        num_train_epochs=t["epochs"], max_steps=max_steps if max_steps else -1, bf16=bf16, save_strategy="no",
        report_to="none", seed=t["seed"], remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    if minutes:
        trainer.add_callback(make_stop_callback(minutes * 60, None, g["max_gpu_temp_c"], g["pause_seconds"],
                                                g["check_every_steps"], out / "events.jsonl"))
    t0 = time.time()
    result = trainer.train()
    model.save_pretrained(str(out / "final_adapter"))
    tok.save_pretrained(str(out / "final_adapter"))
    steps = trainer.state.global_step
    per_epoch = math.ceil(len(ds) / (t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]))
    return {"arm": arm, "n_rows": len(ds), "target_tokens_mean": float(np.mean([d["n_target"] for d in data])),
            "steps": steps, "steps_per_epoch": per_epoch, "epochs_completed": steps / per_epoch,
            "max_steps_requested": max_steps, "minutes_cap": minutes, "seconds": time.time() - t0,
            "train_loss": getattr(result, "training_loss", None), "log_history": trainer.state.log_history}


def train(arm: str, minutes: float | None = None, max_steps: int | None = None) -> dict:
    """GPU: the RL runs' base and LoRA (loader settings of `loader_settings_from`), SFT on the arm's targets, the
    R2-1 fingerprint after the save."""
    import os

    import torch

    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    from physcal.eval.fingerprint import record_after_training
    from physcal.eval.run_eval import GpuMemoryPoller

    cfg = _config()
    ls = load_config(cfg["loader_settings_from"])
    m, lora = ls["model"], ls["lora"]
    if m["id"] != cfg["model_dir"]:
        raise ValueError(f"X3 students must start where the RL runs did: {m['id']} != {cfg['model_dir']}")
    out = REPO_ROOT / cfg["output_dir"] / arm
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    model, tok = FastLanguageModel.from_pretrained(model_name=cfg["model_dir"], max_seq_length=m["max_seq_length"],
                                                   load_in_4bit=True, fast_inference=False, dtype=None,
                                                   text_only=m.get("text_only", False))
    model = FastLanguageModel.get_peft_model(model, r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                             target_modules=lora["target_modules"],
                                             use_gradient_checkpointing="unsloth", random_state=ls["seed"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = read_data()
    with GpuMemoryPoller(interval_s=2.0) as poller:
        summary = fit(model, tok, rows, arm, out, cfg, max_steps, minutes)
    summary.update(model_dir=cfg["model_dir"], lora=lora, lora_seed=ls["seed"],
                   data_sha256=hashlib.sha256((REPO_ROOT / cfg["data_dir"] / "rows.jsonl").read_bytes()).hexdigest()[:16],
                   peak_vram={"torch_max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                              "nvidia_smi_peak_used_gib_device_wide": poller.peak_mib / 1024})
    summary["fingerprint"] = {k: v for k, v in record_after_training(
        model, tok, cfg["model_dir"], out / "final_adapter", cfg["chat_template_kwargs"],
        {"run_id": f"x3-{arm}", "step": summary["steps"]}, out / "events.jsonl",
        prepare=FastLanguageModel.for_inference).items() if k != "traceback"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


# ------------------------------------------------------------------------------------------- eval
def eval_rows(which: str, tok) -> tuple[list[dict], dict, list[str]]:
    """Prefix rows: "eval" = the 400 frozen-set 4-bit SFT answers; "x2" = every answer of the kept X2 problems."""
    from physcal.train import e1

    cfg = _config()
    if which == "eval":
        tapes = [cfg["eval"]["eval_tape"]]
        rows, excluded = e1.build_rows([REPO_ROOT / t for t in tapes], tok, cfg["prompt"], cfg["chat_template_kwargs"])
        return rows, excluded, tapes
    from physcal.eval import x2_pairs

    recs, tape = x2_pairs.kept_records()
    rows, excluded = e1.rows_from_records(recs, tok, cfg["prompt"], cfg["chat_template_kwargs"])
    return rows, excluded, [tape]


def evaluate(arm: str, which: str, backend=None, lora: str | None = None, out: Path | None = None) -> dict:
    """Greedy σ suffixes from a student (or `lora`) on fixed prefixes, via E1's eval machinery (4-bit, unmerged)."""
    from transformers import AutoTokenizer

    from physcal.train import e1

    cfg = _config()
    tok = AutoTokenizer.from_pretrained(str(REPO_ROOT / cfg["model_dir"]))
    rows, excluded, tapes = eval_rows(which, tok)
    lora = lora or str(REPO_ROOT / cfg["output_dir"] / arm / "final_adapter")
    out = out or REPO_ROOT / cfg["output_dir"] / arm / f"eval_{which}"
    res = e1._evaluate(out, f"x3-{arm}-{which}", "free", lora, load_config(cfg["loader_settings_from"])["model"],
                       backend, 0, "unsloth_4bit", rows=rows, source_tapes=tapes, batch_size=cfg["eval"]["batch_size"])
    res["excluded"] = excluded
    (out / "excluded.json").write_text(json.dumps(excluded))
    return res


# ------------------------------------------------------------------------------------------- analysis
def _rho_boot(arm: dict, keys: list, n_boot: int, seed: int, other: dict | None = None) -> dict:
    """ρ(σ, |log err|) over answered, μ-unchanged rows, with a bootstrap over problems (all keys resampled); with
    `other`, the paired difference ρ(arm) − ρ(other) on the same resamples."""
    from scipy.stats import spearmanr

    def rho(d, ks):
        v = [(d[k]["rel_sigma"], d[k]["abs_log_error"]) for k in ks if not d[k]["fail"]
             and d[k]["rel_sigma"] is not None and d[k]["abs_log_error"] is not None]
        return float(spearmanr(*zip(*v)).statistic) if len(v) > 2 else float("nan")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), (n_boot, len(keys)))
    boots = np.array([rho(arm, [keys[i] for i in row]) - (rho(other, [keys[i] for i in row]) if other else 0.0)
                      for row in idx])
    point = rho(arm, keys) - (rho(other, keys) if other else 0.0)
    return {"value": point, "ci95": np.nanpercentile(boots, [2.5, 97.5]).tolist()}


def label(rho_ci: dict, bar: float) -> str:
    """D79's reading rule for student (a)."""
    if rho_ci["ci95"][0] > bar:
        return "capacity shown"
    if rho_ci["value"] >= bar:
        return "capacity likely"
    return "not shown at the pilot budget"


def analyze() -> dict:
    from scipy.stats import spearmanr

    from physcal.train.e1_compare import describe, rows_arm, tape_arm

    cfg = _config()
    ac, out = cfg["analysis"], REPO_ROOT / cfg["output_dir"]
    e1cfg = load_config("e1")
    teacher = tape_arm(REPO_ROOT / cfg["eval"]["a5_sigma_tape"])
    sft = tape_arm(REPO_ROOT / cfg["eval"]["eval_tape"])
    arms = {name: rows_arm(out / name / "eval_eval" / "rows.jsonl") for name in cfg["arms"]
            if (out / name / "eval_eval" / "rows.jsonl").exists()}
    if not arms:
        raise FileNotFoundError("no X3 student eval exists")
    keys = sorted(next(iter(arms.values())))
    res: dict = {"decision": "D79 (X3 pilot)", "n_problems": len(keys), "arms": {}, "reference": {}}
    for name, d in {"A5 teacher σ (4-bit)": teacher, "SFT stated σ (4-bit)": sft}.items():
        res["reference"][name] = describe(d, keys, e1cfg["bins"], e1cfg["compare"]["coverage_target"])
    for name, arm in arms.items():
        if sorted(arm) != keys:
            raise ValueError(f"student {name} was evaluated on different prefixes")
        r = {"describe": describe(arm, keys, e1cfg["bins"], e1cfg["compare"]["coverage_target"]),
             "rho": _rho_boot(arm, keys, ac["bootstrap"], ac["seed"]),
             "rho_minus_teacher": _rho_boot(arm, keys, ac["bootstrap"], ac["seed"], teacher),
             "rho_minus_sft": _rho_boot(arm, keys, ac["bootstrap"], ac["seed"], sft)}
        both = [k for k in keys if not arm[k]["fail"] and arm[k]["rel_sigma"] is not None and not teacher[k]["fail"]]
        r["fidelity_spearman_vs_teacher"] = float(spearmanr([arm[k]["rel_sigma"] for k in both],
                                                            [teacher[k]["rel_sigma"] for k in both]).statistic)
        summ = out / name / "summary.json"
        r["training"] = {k: v for k, v in json.loads(summ.read_text()).items() if k != "log_history"} \
            if summ.exists() else None
        x2 = out / name / "eval_x2" / "rows.jsonl"
        if x2.exists():
            r["x2"] = x2_pairwise(x2)
        res["arms"][name] = r
    if "a" in res["arms"]:
        res["label_a"] = label(res["arms"]["a"]["rho"], ac["rho_bar"])
    if "a" in arms and "b" in arms:
        res["rho_b_minus_a"] = _rho_boot(arms["b"], keys, ac["bootstrap"], ac["seed"], arms["a"])
    (out / "pilot.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "pilot.md").write_text(render(res))
    return res


def x2_pairwise(rows_path: Path) -> dict:
    """Pairwise accuracy (D74's metric) of a σ-suffix eval on the X2 answers: confidence −ln σ_rel; a failed suffix
    (no parse, unit mismatch, μ changed) is a tie."""
    from physcal.eval import x2_pairs

    cfg = x2_pairs._config()
    floor = cfg["analysis"]["sigma_floor_rel"]
    conf = {}
    for line in rows_path.open():
        r = json.loads(line)
        ok = not r["fail"] and r["rel_sigma_used"] is not None
        conf[(tuple(r["key"]), r["source_sample"])] = -math.log(max(r["rel_sigma_used"], floor)) if ok else None
    groups, keys = x2_pairs.kept_groups()
    pa = x2_pairs.pairwise_accuracy(groups, keys, lambda k, a: conf.get((tuple(k), a["sample"])))
    return x2_pairs.summarize(pa, cfg["analysis"]["bootstrap"], cfg["analysis"]["seed"])


def render(res: dict) -> str:
    f = lambda x, pct=False: "–" if x is None else (f"{100 * x:.1f}%" if pct else f"{x:.3f}")
    ci = lambda d: f"{d['value']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
    L = [f"# X3 pilot (D79): distillation capacity, {res['n_problems']} frozen-set answers (μ fixed)", ""]
    if "label_a" in res:
        L += [f"**Student (a), A5 teacher: {res['label_a']}** (bar ρ = 0.5).", ""]
    L += ["| σ source | ρ(σ, err) [95% CI] | mean crps_log | 1σ / 2σ cov | median rel σ | σ = 0 | failed | μ changed "
          "| fidelity vs teacher | X2 pairwise acc. |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res["arms"].items():
        d = r["describe"]
        x2 = r.get("x2")
        L.append(f"| student ({name}) | {ci(r['rho'])} | {f(d['mean_crps_log'])} | {f(d['coverage_1sigma'], True)} / "
                 f"{f(d['coverage_2sigma'], True)} | {f(d['median_rel_sigma'], True)} | {f(d['sigma_zero_share'], True)} "
                 f"| {d['n_failed']} | {d['mu_changed']} | {f(r['fidelity_spearman_vs_teacher'])} | "
                 f"{'–' if not x2 else f(x2['pairwise_accuracy']) + ' [' + f(x2['ci95'][0]) + ', ' + f(x2['ci95'][1]) + ']'} |")
    for name, d in res["reference"].items():
        L.append(f"| {name} | {f(d['spearman_sigma_error'])} | {f(d['mean_crps_log'])} | {f(d['coverage_1sigma'], True)} / "
                 f"{f(d['coverage_2sigma'], True)} | {f(d['median_rel_sigma'], True)} | {f(d['sigma_zero_share'], True)} "
                 f"| {d['n_failed']} | – | – | – |")
    L.append("")
    for name, r in res["arms"].items():
        L.append(f"- student ({name}): ρ − teacher {ci(r['rho_minus_teacher'])}; ρ − SFT {ci(r['rho_minus_sft'])}")
        tr = r.get("training")
        if tr:
            L.append(f"  - training: {tr['steps']} steps ({tr['epochs_completed']:.2f} epochs), loss "
                     f"{f(tr['train_loss'])}, {tr['seconds'] / 60:.0f} min")
    if "rho_b_minus_a" in res:
        L.append(f"- ρ(b) − ρ(a): {ci(res['rho_b_minus_a'])}")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["data", "train", "eval", "analyze"])
    ap.add_argument("--arm", choices=["a", "b"])
    ap.add_argument("--set", choices=["eval", "x2"], default="eval")
    ap.add_argument("--minutes", type=float, default=None, help="train: wall-clock cap")
    ap.add_argument("--max-steps", type=int, default=None, help="train: exact step budget (student b = a's steps)")
    args = ap.parse_args()
    if args.action == "data":
        print(json.dumps(build_data(), indent=1))
    elif args.action == "train":
        print(json.dumps(train(args.arm, args.minutes, args.max_steps), indent=1, default=str)[:4000])
    elif args.action == "eval":
        print(json.dumps(evaluate(args.arm, args.set), indent=1))
    else:
        print(render(analyze()))


if __name__ == "__main__":
    main()

"""Trained-adapter load check (~3 min GPU, 2026-10-25): stage D's D-pos adapter loads on the pinned 4-bit base through
the eval backend, and its first eval batch reproduces the logged greedy completions. A capability check, not a result.

    HF_HUB_OFFLINE=1 uv run python scripts/adapter_load_check.py

The eval sorts all 3,060 prompts by length (longest first) and generates 32 per batch, so the 32 longest prompts were
one batch. Regenerating exactly that batch gives the same padding and batch composition, so greedy decoding should
reproduce the tape. Checks: the backend's adapter hash equals the tape's; the regenerated completions match the
logged ones (reported per completion) and re-score to the logged rewards.
Writes results/adapter_load_check/report.json and prints PASS / FAIL.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "adapter_load_check"
ARM = "pos"
MIN_EXACT_SHARE = 0.9  # greedy, same batch: a lower share means the loaded model differs from the evaluated one


def main() -> int:
    os.chdir(ROOT)
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    from physcal.env.generator import problem_from_dict
    from physcal.eval.run_eval import HFBackend
    from physcal.prompt import build_messages
    from physcal.train import stage_d
    from physcal.verify.reward import reward, reward_config

    ev = stage_d.eval_config(ARM)
    tape = Path(ev["output_dir"]) / "rollouts.jsonl"
    rows = [json.loads(line) for line in tape.open()]
    t0 = time.time()
    backend = HFBackend(ev["model"]["id"], ev["model"], ev["generation"])
    rep: dict = {"check": "trained adapter load", "arm": ARM, "adapter": ev["model"]["lora"],
                 "tape": str(tape.relative_to(ROOT)), "lora_hash": backend.lora_hash,
                 "tape_lora_hash": rows[0]["lora_hash"], "provenance": backend.provenance, "checks": {}}
    rep["checks"]["lora_hash_matches_tape"] = backend.lora_hash == rows[0]["lora_hash"]
    convs = [build_messages(r["problem"]["text"], ev["prompt"]) for r in rows]
    prompts = [backend.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True,
                                               **backend.chat_template_kwargs) for c in convs]
    first = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))[:ev["generation"]["batch_size"]]
    gens = backend([convs[i] for i in first])
    rw_cfg = reward_config()
    per = []
    for i, g in zip(first, gens):
        r, _ = reward(problem_from_dict(rows[i]["problem"]), g.text, rw_cfg)
        per.append({"tape_line": i, "split": rows[i]["split"], "exact": g.text == rows[i]["completion"],
                    "reward": r, "tape_reward": rows[i]["reward"]})
    rep["n"] = len(per)
    rep["exact_share"] = sum(p["exact"] for p in per) / len(per)
    rep["checks"]["completions_reproduce"] = rep["exact_share"] >= MIN_EXACT_SHARE
    rep["checks"]["exact_ones_rescore_to_tape"] = all(p["reward"] == p["tape_reward"] for p in per if p["exact"])
    rep["per_completion"] = per
    rep["seconds"] = time.time() - t0
    rep["status"] = "PASS" if all(rep["checks"].values()) else "FAIL"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "per_completion"}, indent=1))
    print(rep["status"])
    return 0 if rep["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

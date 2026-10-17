"""Post-hoc diagnostic for D65 (added after the 4-bit greedy results; exploratory, CPU only, no tests).

    CUDA_VISIBLE_DEVICES="" uv run python scripts/precision_merge_check.py [N]

Question: is the bf16 eval's narrow σ an artifact of merging the LoRA (merge_and_unload) rather than of the base
precision? On the first N E1a eval prefixes whose bf16 and 4-bit greedy σ suffixes differ, in bf16 on CPU:
- greedy σ suffix with the adapter attached unmerged (PEFT) and after merge_and_unload (the bf16 eval's path);
- the next-token top-5 right after "±" for the SFT base, the unmerged adapter, and the merged adapter.
If unmerged == merged (up to bf16 rounding) and both match the bf16 eval, the merge is not the cause and the
difference comes from the base the adapter is applied to. Writes results/precision_check/merge_check.{json,md}.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
E1A = ROOT / "results/train/e1a-sigma-only-free-seed0"
SFT = ROOT / "results/sft/qwen2.5-math-1.5b-instruct/merged"
OUT = ROOT / "results/precision_check"
MAX_NEW = 24       # e1.yaml eval budget
TOP_K = 5
THREADS = 6


def main(n: int) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from physcal.train import e1

    torch.set_num_threads(THREADS)
    rows = e1.read_rows("eval")
    bf = [json.loads(x) for x in (E1A / "e1_eval/final/rollouts.jsonl").read_text().splitlines()]
    q4 = [json.loads(x) for x in (E1A / "e1_eval/final_4bit/rollouts.jsonl").read_text().splitlines()]
    picked = [i for i, (x, y) in enumerate(zip(bf, q4)) if x["suffix"] != y["suffix"]][:n]
    tok = AutoTokenizer.from_pretrained(str(SFT))
    model = AutoModelForCausalLM.from_pretrained(str(SFT), dtype=torch.bfloat16).eval()
    eos = [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")]

    def top(m, ids):
        with torch.inference_mode():
            p = torch.softmax(m(ids).logits[0, -1].float(), -1)
        v, i = p.topk(TOP_K)
        return [[tok.decode([int(t)]), round(float(x), 4)] for x, t in zip(v, i)], p

    def greedy(m, ids):
        with torch.inference_mode():
            out = m.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, eos_token_id=eos, pad_token_id=tok.eos_token_id,
                             repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    enc = {i: tok(rows[i]["prompt"], return_tensors="pt").input_ids for i in picked}
    res = {i: {"key": rows[i]["key"], "bf16_eval": bf[i]["suffix"], "4bit_eval": q4[i]["suffix"]} for i in picked}
    for i in picked:
        res[i]["top_base"], _ = top(model, enc[i])
    model = PeftModel.from_pretrained(model, str(E1A / "final_adapter")).eval()
    probs = {}
    for i in picked:
        res[i]["top_unmerged"], probs[i] = top(model, enc[i])
        res[i]["greedy_unmerged"] = greedy(model, enc[i])
    model = model.merge_and_unload().eval()
    for i in picked:
        res[i]["top_merged"], p = top(model, enc[i])
        res[i]["max_abs_prob_diff_merged_vs_unmerged"] = float((p - probs[i]).abs().max())
        res[i]["greedy_merged"] = greedy(model, enc[i])
    out = {"n": len(picked), "rows": list(res.values()),
           "greedy_unmerged_equals_merged": sum(r["greedy_unmerged"] == r["greedy_merged"] for r in res.values()),
           "greedy_merged_equals_bf16_eval": sum(r["greedy_merged"] == r["bf16_eval"] for r in res.values()),
           "greedy_unmerged_equals_4bit_eval": sum(r["greedy_unmerged"] == r["4bit_eval"] for r in res.values())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "merge_check.json").write_text(json.dumps(out, indent=2))
    lines = ["# Merge check (post-hoc diagnostic for D65; CPU, bf16, E1a final adapter)", "",
             f"{out['n']} E1a eval prefixes whose bf16 and 4-bit greedy σ suffixes differ.", "",
             f"- greedy suffix, unmerged == merged: {out['greedy_unmerged_equals_merged']} / {out['n']}",
             f"- merged (CPU) == the bf16 eval's suffix: {out['greedy_merged_equals_bf16_eval']} / {out['n']}",
             f"- unmerged bf16 == the 4-bit eval's suffix: {out['greedy_unmerged_equals_4bit_eval']} / {out['n']}", "",
             "| key | bf16 eval | 4-bit eval | bf16 unmerged | bf16 merged | max |Δp| merged vs unmerged | top-5 after ± (base / unmerged) |",
             "|---|---|---|---|---|---|---|"]
    for r in res.values():
        t = lambda xs: ", ".join(f"{s!r} {p:.2f}" for s, p in xs)
        lines.append(f"| {r['key']} | {r['bf16_eval']!r} | {r['4bit_eval']!r} | {r['greedy_unmerged']!r} | "
                     f"{r['greedy_merged']!r} | {r['max_abs_prob_diff_merged_vs_unmerged']:.4f} | "
                     f"{t(r['top_base'])} / {t(r['top_unmerged'])} |")
    (OUT / "merge_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)

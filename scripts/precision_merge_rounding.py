"""Post-hoc diagnostic for D65 (exploratory, CPU, weights only, no tests): how much of a LoRA update survives
merge_and_unload into the bf16 base?

    uv run python scripts/precision_merge_rounding.py

Per LoRA target module: delta = scaling · B @ A in fp32 (the update the adapter applies at train time, unmerged).
PEFT's merge writes W ← bf16(W + bf16(delta)) (CPU path: delta in fp32, cast to the weight dtype, added in place).
The effective update is eff = merged − W (fp32). Reported over all modules, pooled:
- retained = ||eff||_F / ||delta||_F and rel_error = ||eff − delta||_F / ||delta||_F;
- swallowed = the share of entries with delta ≠ 0 but eff == 0 (rounded away entirely);
- the median |delta| / ulp_bf16(W) (below 0.5, an entry's update rounds to nothing unless W sits near a boundary).
Adapters: the RL runs (split credit seeds 0-2 @287, uniform credit @287, E1a, E1b) on the merged SFT model, and the
SFT adapter on its own base. Writes results/precision_check/merge_rounding.{json,md}.
"""

import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/precision_check"
SFT_MERGED = ROOT / "results/sft/qwen2.5-math-1.5b-instruct/merged"
RUNS = ROOT / "results/train"
ADAPTERS = {
    **{f"split credit seed {s} @287": RUNS / f"split-credit-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512-seed{s}"
       / "permanent_checkpoints/checkpoint-287" for s in (0, 1, 2)},
    "uniform credit @287": RUNS / "overnight2-grpo-qwen2.5-math-1.5b-instruct-sft-crps_log-v1.1.0-512/permanent_checkpoints/checkpoint-287",
    "E1a final": RUNS / "e1a-sigma-only-free-seed0/final_adapter",
    "E1b final": RUNS / "e1b-sigma-only-bins-seed0/final_adapter",
}
SFT_ADAPTER = ROOT / "results/sft/qwen2.5-math-1.5b-instruct/adapter"


def base_files(merged_dir: Path) -> list[Path]:
    return sorted(merged_dir.glob("*.safetensors"))


def sft_base_dir() -> Path | None:
    """The original base the SFT adapter was merged into (merge_info.json), from the local HF cache."""
    info = json.loads((SFT_MERGED / "merge_info.json").read_text())
    hub = Path(os.environ.get("HF_HOME", ROOT / ".cache" / "huggingface")) / "hub"
    snap = hub / f"models--{info['base'].replace('/', '--')}" / "snapshots" / info["revision"]
    return snap if snap.exists() else None


def bf16_ulp(w: torch.Tensor) -> torch.Tensor:
    """Spacing of bf16 numbers at |w| (8 significant bits): 2^(floor(log2|w|) - 7)."""
    a = w.float().abs().clamp_min(torch.finfo(torch.float32).tiny)
    return torch.exp2(torch.floor(torch.log2(a)) - 7)


def measure(adapter: Path, base: list[Path]) -> dict:
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    scaling = cfg["lora_alpha"] / cfg["r"]
    lora = safe_open(str(adapter / "adapter_model.safetensors"), "pt")
    keys = [k for k in lora.keys() if k.endswith("lora_A.weight")]
    handles = [safe_open(str(f), "pt") for f in base]
    where = {k: h for h in handles for k in h.keys()}
    sq_delta = sq_eff = sq_err = 0.0
    nonzero = swallowed = 0
    ratios = []
    for ka in keys:
        name = ka.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
        w = where[name].get_tensor(name)
        a, b = lora.get_tensor(ka).float(), lora.get_tensor(ka.replace("lora_A", "lora_B")).float()
        delta = scaling * (b @ a)
        merged = w + delta.to(w.dtype)  # PEFT, CPU: fp32 delta cast to the weight dtype, added in place
        eff = merged.float() - w.float()
        sq_delta += float((delta ** 2).sum())
        sq_eff += float((eff ** 2).sum())
        sq_err += float(((eff - delta) ** 2).sum())
        nz = delta != 0
        nonzero += int(nz.sum())
        swallowed += int((nz & (eff == 0)).sum())
        r = (delta.abs() / bf16_ulp(w))[nz]
        ratios.append(r[torch.randint(len(r), (min(len(r), 20000),), generator=torch.Generator().manual_seed(0))])
    ratios = torch.cat(ratios)
    return {"modules": len(keys), "retained_norm": (sq_eff / sq_delta) ** 0.5, "rel_error": (sq_err / sq_delta) ** 0.5,
            "swallowed_share": swallowed / nonzero, "median_abs_delta_over_ulp": float(ratios.median()),
            "share_abs_delta_below_half_ulp": float((ratios < 0.5).float().mean())}


def main() -> dict:
    out = {}
    for name, adapter in ADAPTERS.items():
        out[name] = measure(adapter, base_files(SFT_MERGED)) if adapter.exists() else {"missing": str(adapter)}
    sft_base = sft_base_dir()
    out["SFT adapter (on its original base)"] = measure(SFT_ADAPTER, base_files(sft_base)) if sft_base else \
        {"missing": "original base not in the local HF cache"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "merge_rounding.json").write_text(json.dumps(out, indent=2))
    lines = ["# How much of each LoRA update survives a bf16 merge (post-hoc diagnostic for D65; weights only)", "",
             "| adapter | modules | retained ‖eff‖/‖Δ‖ | rel. error ‖eff − Δ‖/‖Δ‖ | entries rounded to 0 | median |Δ| / ulp(W) | |Δ| < ½ ulp |",
             "|---|---|---|---|---|---|---|"]
    for name, r in out.items():
        if "missing" in r:
            lines.append(f"| {name} | missing: {r['missing']} | | | | | |")
            continue
        lines.append(f"| {name} | {r['modules']} | {r['retained_norm']:.3f} | {r['rel_error']:.3f} | "
                     f"{r['swallowed_share']:.1%} | {r['median_abs_delta_over_ulp']:.3f} | "
                     f"{r['share_abs_delta_below_half_ulp']:.1%} |")
    (OUT / "merge_rounding.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


if __name__ == "__main__":
    sys.exit(main() is None)

"""Phase 9.5 step 3: split credit for σ tokens (verifier unchanged; PREREGISTRATION.md family S).

The verifier's reward is split into two parts that add back up to it:
    r_answer = w_final · crps_log(μ, σ_ref) + w_steps · steps      σ_ref = `sigma_ref_rel` · μ (e.g. 1% of μ)
    r_sigma  = w_final · [crps_log(μ, σ) − crps_log(μ, σ_ref)]
Failures (parse failure, unit mismatch, no σ) keep their whole reward in r_answer and r_sigma = 0. Each part gets
TRL's group-normalized advantage ((r − group mean) / (group std + 1e-4)). Every completion token gets the r_answer
advantage, except the tokens of the stated σ number (located by parsing), which get the r_sigma advantage. The
result is a (batch, completion length) advantage tensor; TRL/Unsloth's GRPO loss uses advantages elementwise, so
nothing else in the loss changes, except that Unsloth's fused loss left-packs each micro-batch, so the per-token
advantages are moved into that layout first (`pack_like_unsloth`). The total reward that TRL logs is still the
verifier's.

`make_split_credit_trainer(base)` returns a subclass of `base` (a GRPOTrainer) that swaps the advantages in.
`VerifierReward.split_credit` (a `SplitCredit`) collects the per-completion parts during reward computation.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import torch

from physcal.verify.parse import NUM
from physcal.verify.scoring import score, sigma_floor

ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
PM_RE = re.compile(r"±|\+/-|\+-|\\pm")
SIGMA_NUMBER_RE = re.compile(rf"\s*\(?\s*({NUM})")


def sigma_char_span(completion: str) -> tuple[int, int] | None:
    """Character span of the stated σ number: the first number after the first ± marker in the single answer
    block. None if there is no single answer block, no ± marker or no number after it."""
    blocks = list(ANSWER_BLOCK_RE.finditer(completion))
    if len(blocks) != 1:
        return None
    start, body = blocks[0].start(1), blocks[0].group(1)
    pm = PM_RE.search(body)
    if pm is None:
        return None
    m = SIGMA_NUMBER_RE.match(body, pm.end())
    if m is None:
        return None
    return start + m.start(1), start + m.end(1)


def split_from_breakdown(r: float, bd: dict, completion: str, cfg: dict, sigma_ref_rel: float) -> dict[str, Any]:
    """{r_total, r_answer, r_sigma, span}. r_answer + r_sigma == r up to float rounding."""
    sc, w = cfg["scoring"], cfg["weights"]
    if sc["rule"] != "crps_log":
        raise ValueError("split credit is defined for the crps_log rule")
    if bd["reason"] != "ok" or bd.get("sigma_si") is None:
        return {"r_total": r, "r_answer": r, "r_sigma": 0.0, "span": None}
    s_ref = sigma_ref_rel * abs(bd["mu_si"]) / abs(bd["truth_si"])
    s_ref_score = score("crps_log", bd["rel_error"], max(s_ref, sigma_floor("crps_log", sc)), sc)
    r_answer = w["final"] * s_ref_score + w["steps"] * bd["score_steps"]
    return {"r_total": r, "r_answer": r_answer, "r_sigma": w["final"] * (bd["score_final"] - s_ref_score),
            "span": sigma_char_span(completion)}


def group_advantages(r: torch.Tensor, group_size: int, eps: float = 1e-4) -> torch.Tensor:
    """TRL's scale_rewards="group" advantage: (r − group mean) / (group nanstd + eps), NaN → 0."""
    from trl.trainer.utils import nanstd

    g = r.view(-1, group_size)
    mean = torch.nanmean(g, dim=1).repeat_interleave(group_size)
    std = nanstd(g, dim=1).repeat_interleave(group_size)
    return torch.nan_to_num((r - mean) / (std + eps), nan=0.0)


def token_char_ends(tokenizer, ids: list[int], text: str) -> list[int] | None:
    """End character (in `text`) of each token of `ids`, where `text` is their decode with special tokens skipped.
    Fast path: re-encode `text` with offsets and check it gives the same ids; otherwise decode every prefix."""
    special = set(tokenizer.all_special_ids)
    content = [i for i in ids if i not in special]
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if enc["input_ids"] == content:
        ends, it = [], iter(enc["offset_mapping"])
        last = 0
        for i in ids:
            if i not in special:
                last = next(it)[1]
            ends.append(last)
        return ends
    ends = [len(tokenizer.decode(ids[:k + 1], skip_special_tokens=True)) for k in range(len(ids))]
    return ends if ends and ends[-1] == len(text) else None


def sigma_token_mask(tokenizer, completion_ids: torch.Tensor, completion_mask: torch.Tensor, texts: list[str],
                     spans: list[tuple[int, int] | None]) -> tuple[torch.Tensor, int]:
    """(B, T) bool mask of the tokens overlapping each completion's σ span; also the number of rows whose token
    offsets could not be matched to the text (those rows get no σ tokens)."""
    mask = torch.zeros_like(completion_mask, dtype=torch.bool)
    unmatched = 0
    for b, (text, span) in enumerate(zip(texts, spans)):
        if span is None:
            continue
        n = int(completion_mask[b].sum())
        ids = completion_ids[b, :n].tolist()
        ends = token_char_ends(tokenizer, ids, text)
        if ends is None:
            unmatched += 1
            continue
        lo, hi = span
        prev = 0
        for k, end in enumerate(ends):
            if end > prev and prev < hi and end > lo:  # token covers [prev, end) and overlaps [lo, hi)
                mask[b, k] = True
            prev = end
    return mask, unmatched


def token_advantages(a_answer: torch.Tensor, a_sigma: torch.Tensor, sigma_mask: torch.Tensor) -> torch.Tensor:
    """(B, T): the answer advantage everywhere, the σ advantage on σ tokens."""
    a_answer, a_sigma = a_answer.unsqueeze(1), a_sigma.unsqueeze(1)
    return torch.where(sigma_mask, a_sigma.expand_as(sigma_mask), a_answer.expand_as(sigma_mask)).float()


def pack_like_unsloth(adv: torch.Tensor, prompt_ids: torch.Tensor, completion_ids: torch.Tensor, pad_id: int,
                      old_logps: torch.Tensor | None = None, ref_logps: torch.Tensor | None = None) -> torch.Tensor:
    """Re-lay (B, L) completion-token advantages the way Unsloth's fused GRPO loss lays out a micro-batch.

    `grpo_accumulated_loss` left-packs [prompt | completion] (every pad token moves to the right) and scores the last
    L + M columns, M = the largest prompt left-padding in the micro-batch (or the old/ref logp width − L). Completion
    token j of row i then sits at column M − leftpad_i + j. Columns that are not completion tokens are masked by
    Unsloth, so their value is irrelevant (0 here)."""
    L = completion_ids.size(1)
    left_pad = (prompt_ids == pad_id).sum(dim=1)
    if old_logps is not None:
        M = old_logps.size(1) - L
    elif ref_logps is not None:
        M = ref_logps.size(1) - L
    else:
        M = int(left_pad.max())
    out = adv.new_zeros(adv.size(0), L + M)
    cols = (M - left_pad).unsqueeze(1) + torch.arange(L, device=adv.device).unsqueeze(0)
    return out.scatter_(1, cols, adv)


class SplitCredit:
    """Per-batch store of reward parts (filled by VerifierReward, consumed by the trainer) plus a JSONL log."""

    def __init__(self, sigma_ref_rel: float, log_path: Path | None = None):
        self.sigma_ref_rel = sigma_ref_rel
        self.log_path = log_path
        self.parts: list[dict] = []
        self.texts: list[str] = []

    def add(self, r: float, bd: dict, completion: str, cfg: dict) -> None:
        self.parts.append(split_from_breakdown(r, bd, completion, cfg, self.sigma_ref_rel))
        self.texts.append(completion)

    def pop(self) -> tuple[list[dict], list[str]]:
        parts, texts = self.parts, self.texts
        self.parts, self.texts = [], []
        return parts, texts

    def log(self, row: dict) -> None:
        if self.log_path is not None:
            with self.log_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")


def make_split_credit_trainer(base, split: SplitCredit):
    """A subclass of the GRPO trainer class `base` whose advantages are per token (see module docstring)."""

    class SplitCreditGRPOTrainer(base):
        def _generate_and_score_completions(self, inputs):
            out = super()._generate_and_score_completions(inputs)
            parts, texts = split.pop()
            ids, cmask = out["completion_ids"], out["completion_mask"]
            if len(parts) != ids.size(0):
                raise RuntimeError(f"split credit: {len(parts)} reward parts for {ids.size(0)} completions")
            decoded = self.processing_class.batch_decode(ids, skip_special_tokens=True)
            if decoded != texts:
                raise RuntimeError("split credit: reward parts are not aligned with the completions")
            g = self.num_generations
            dev = out["advantages"].device
            r_ans = torch.tensor([p["r_answer"] for p in parts], dtype=torch.float32, device=dev)
            r_sig = torch.tensor([p["r_sigma"] for p in parts], dtype=torch.float32, device=dev)
            a_ans, a_sig = group_advantages(r_ans, g), group_advantages(r_sig, g)
            smask, unmatched = sigma_token_mask(self.processing_class, ids.cpu(), cmask.cpu(), texts,
                                                [p["span"] for p in parts])
            smask = smask.to(dev) & cmask.bool()
            out["advantages"] = token_advantages(a_ans, a_sig, smask)
            has_sigma = smask.any(dim=1)
            split.log({
                "step": self.state.global_step + 1, "time": time.time(), "n": len(parts),
                "share_with_sigma_tokens": float(has_sigma.float().mean()),
                "mean_sigma_tokens": float(smask.sum(1).float()[has_sigma].mean()) if has_sigma.any() else 0.0,
                "unmatched_offsets": unmatched,
                "r_sigma_mean": float(r_sig.mean()), "r_sigma_abs_max": float(r_sig.abs().max()),
                "r_sigma_zero_var_groups": float((r_sig.view(-1, g).std(1) == 0).float().mean()),
                "a_sigma_abs_mean_on_sigma_tokens": float(a_sig[has_sigma].abs().mean()) if has_sigma.any() else 0.0,
                "a_answer_abs_mean": float(a_ans.abs().mean()),
            })
            return out

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Unsloth's fused loss (and its clipping metrics) work on a left-packed, L + M wide layout; 1-D
            # advantages broadcast over it, ours must be moved into it. If Unsloth ever took its unpacked slow
            # path instead, the widths would disagree and the loss would raise, not silently misalign.
            adv = inputs["advantages"]
            if adv.dim() == 2:
                inputs = {**inputs, "advantages": pack_like_unsloth(
                    adv, inputs["prompt_ids"], inputs["completion_ids"], self.processing_class.pad_token_id,
                    inputs.get("old_per_token_logps"), inputs.get("ref_per_token_logps"))}
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)

    return SplitCreditGRPOTrainer

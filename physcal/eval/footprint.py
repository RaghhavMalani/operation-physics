"""Estimated peak VRAM of a GRPO step (4-bit base + LoRA) for a Hugging Face checkpoint, without loading it.

Shapes come from the checkpoint's config.json and its safetensors headers (no weight download). Every
assumption is in configs/footprint.yaml. This is an estimate: confirm it with the first phase-4 smoke step.
`estimate_inference_bf16` applies the same activation model to plain bf16 generation, so it can be checked
against peaks measured by run_eval.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from physcal.config import REPO_ROOT

GIB = 1024**3
EXCLUDED_PREFIXES = ("model.visual.", "visual.", "mtp.")  # vision tower and multi-token-prediction head


@dataclass(frozen=True)
class ModelShape:
    params_quantized: int  # 2-D weights inside decoder layers (bnb 4-bit Linear)
    params_embedding: int  # embed_tokens (+ lm_head when untied), kept in bf16
    params_other: int  # norms, biases, conv kernels, ... (bf16)
    params_excluded: int  # vision tower / MTP head: not needed for text-only training
    lora_params_per_rank: int  # sum of (in + out) over the quantized linears
    n_layers: int
    hidden: int
    intermediate: int
    vocab: int
    kv_bytes_per_token: int  # bf16 K and V of the full-attention layers
    recurrent_state_bytes: int  # fp32 state of the linear-attention layers, per sequence

    @property
    def params_text(self) -> int:
        return self.params_quantized + self.params_embedding + self.params_other


def _numel(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def model_shape(model_id: str, revision: str) -> ModelShape:
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    from huggingface_hub import get_safetensors_metadata, hf_hub_download

    cfg = json.loads(open(hf_hub_download(model_id, "config.json", revision=revision)).read())
    tensors = {
        name: info.shape
        for f in get_safetensors_metadata(model_id, revision=revision).files_metadata.values()
        for name, info in f.tensors.items()
    }
    return shape_from(cfg, tensors)


def shape_from(cfg: dict, tensors: dict[str, list[int]]) -> ModelShape:
    t = cfg.get("text_config", cfg)
    quantized = embedding = other = excluded = lora = 0
    for name, shape in tensors.items():
        n = _numel(shape)
        if name.startswith(EXCLUDED_PREFIXES):
            excluded += n
        elif "embed_tokens" in name or "lm_head" in name:
            embedding += n
        elif ".layers." in name and len(shape) == 2 and name.endswith(".weight") and "norm" not in name:
            quantized += n
            lora += shape[0] + shape[1]
        else:
            other += n
    n_layers = t["num_hidden_layers"]
    layer_types = t.get("layer_types") or ["full_attention"] * n_layers
    n_full = sum(lt == "full_attention" for lt in layer_types)
    n_linear = sum(lt == "linear_attention" for lt in layer_types)
    head_dim = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
    kv = 2 * n_full * t["num_key_value_heads"] * head_dim * 2
    state = n_linear * t.get("linear_num_value_heads", 0) * t.get("linear_key_head_dim", 0) \
        * t.get("linear_value_head_dim", 0) * 4
    return ModelShape(quantized, embedding, other, excluded, lora, n_layers, t["hidden_size"],
                      t["intermediate_size"], t["vocab_size"], kv, state)


def _merge(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) else v
    return out


def _layer_working_set(shape: ModelShape, sequences: int, tokens: int, act: dict) -> float:
    return sequences * tokens * (act["hidden_factor"] * shape.hidden + act["mlp_factor"] * shape.intermediate) * 2


def estimate_grpo(shape: ModelShape, prompt_tokens: int, cfg: dict, pessimistic: bool = False) -> dict[str, Any]:
    """Peak device memory (GiB) of one GRPO step: max of the rollout phase and the training phase."""
    c = _merge(cfg, cfg["pessimistic_overrides"]) if pessimistic else cfg
    g, q, act, lg, lo = c["grpo"], c["quantization"], c["activations"], c["logits"], c["lora"]
    seq = prompt_tokens + g["completion_tokens"]
    weights = (shape.params_quantized * q["bits"] / 8 * (1 + q["overhead"])
               + 2 * (shape.params_embedding + shape.params_other))
    lora = lo["rank"] * shape.lora_params_per_rank * lo["bytes_per_param"]
    rollout = (g["group_size"] * seq * shape.kv_bytes_per_token + g["group_size"] * shape.recurrent_state_bytes
               + _layer_working_set(shape, g["group_size"], prompt_tokens, act)
               + g["group_size"] * shape.vocab * 4)
    m = g["micro_batch"]
    checkpoints = 0 if act["checkpoint_offload"] else shape.n_layers * m * seq * shape.hidden * 2
    layer = _layer_working_set(shape, m, seq, act)
    logits = m * g["completion_tokens"] * shape.vocab * lg["bytes_per_entry"] * lg["live_copies"] / lg["chunks"]
    training = checkpoints + layer + logits
    frag, ctx = c["allocator"]["fragmentation"], c["allocator"]["cuda_context_gib"]
    peak = (weights + lora + max(rollout, training)) * (1 + frag) / GIB + ctx
    return {
        "prompt_tokens": prompt_tokens,
        "weights_gib": weights / GIB,
        "lora_and_optimizer_gib": lora / GIB,
        "rollout_kv_and_activations_gib": rollout / GIB,
        "training_activations_gib": (checkpoints + layer) / GIB,
        "training_logits_gib": logits / GIB,
        "phase_at_peak": "rollout" if rollout >= training else "training",
        "peak_device_gib": peak,
        "peak_device_gb": peak * GIB / 1e9,
    }


def estimate_inference_bf16(shape: ModelShape, sequences: int, prompt_tokens: int, new_tokens: int,
                            cfg: dict) -> float:
    """torch-allocated peak (GiB) of bf16 generation, the quantity run_eval measures (no CUDA context)."""
    weights = 2 * (shape.params_text + shape.params_excluded)  # the whole checkpoint is loaded for eval
    kv = sequences * (prompt_tokens + new_tokens) * shape.kv_bytes_per_token + sequences * shape.recurrent_state_bytes
    prefill = _layer_working_set(shape, sequences, prompt_tokens, cfg["activations"]) + sequences * shape.vocab * 4
    return (weights + kv + prefill) / GIB


def describe(shape: ModelShape) -> dict[str, Any]:
    return {**asdict(shape), "params_text": shape.params_text}

"""Stream reader for Kimi K2.6 safetensors shards.

Indexes all tensor keys across 64 shards and provides targeted reads of
individual tensors (one router / one expert / one attention proj at a
time) via safetensors' zero-copy memmap interface.

Key conventions in Kimi K2.6:
  language_model.model.embed_tokens.weight                  BF16 (V, H)
  language_model.model.layers.<L>.input_layernorm.weight    BF16 (H,)
  language_model.model.layers.<L>.post_attention_layernorm.weight  BF16 (H,)

  # MLA attention (layers 0..60; layer 0 is dense mlp, not dense-attn)
  language_model.model.layers.<L>.self_attn.q_a_proj.weight_packed
  language_model.model.layers.<L>.self_attn.q_a_proj.weight_scale
  language_model.model.layers.<L>.self_attn.q_a_proj.weight_shape
  language_model.model.layers.<L>.self_attn.q_a_layernorm.weight
  language_model.model.layers.<L>.self_attn.q_b_proj.weight_packed
  ...
  language_model.model.layers.<L>.self_attn.kv_a_proj_with_mqa.weight_packed
  language_model.model.layers.<L>.self_attn.kv_a_layernorm.weight
  language_model.model.layers.<L>.self_attn.kv_b_proj.weight_packed
  language_model.model.layers.<L>.self_attn.o_proj.weight_packed

  # Dense MLP (layer 0 only)
  language_model.model.layers.0.mlp.gate_proj.weight        BF16
  language_model.model.layers.0.mlp.up_proj.weight          BF16
  language_model.model.layers.0.mlp.down_proj.weight        BF16

  # MoE (layers 1..60)
  language_model.model.layers.<L>.mlp.gate.weight                     BF16 (E, H)  router
  language_model.model.layers.<L>.mlp.gate.e_score_correction_bias    F32  (E,)
  language_model.model.layers.<L>.mlp.experts.<E>.gate_proj.weight_packed
  language_model.model.layers.<L>.mlp.experts.<E>.up_proj.weight_packed
  language_model.model.layers.<L>.mlp.experts.<E>.down_proj.weight_packed
  language_model.model.layers.<L>.mlp.shared_experts.gate_proj.weight_packed
  ...

  # Final
  language_model.model.norm.weight
  language_model.lm_head.weight_packed
"""

from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .int4_codec import unpack_int4, bf16_u16_to_f32


@dataclass
class ShardIndex:
    """Map tensor_key -> (shard_path, dtype-string)."""
    key_to_shard: dict[str, Path]
    shards: list[Path]


def build_index(model_dir: Path) -> ShardIndex:
    """Scan all .safetensors shards, build tensor → shard map.

    Uses torch framework to read headers (avoids numpy's lack of bf16 support).
    """
    from safetensors import safe_open
    shards = sorted(model_dir.glob("model-*.safetensors"))
    key_to_shard: dict[str, Path] = {}
    for sp in shards:
        with safe_open(sp, framework="pt") as f:
            for k in f.keys():
                key_to_shard[k] = sp
    return ShardIndex(key_to_shard=key_to_shard, shards=shards)


def read_tensor(idx: ShardIndex, key: str) -> np.ndarray:
    """Read one tensor from whichever shard owns it.

    Reads via torch (handles bf16 natively), converts to numpy float32
    for bf16, or preserves native dtype otherwise.
    """
    import torch
    from safetensors import safe_open
    sp = idx.key_to_shard[key]
    with safe_open(sp, framework="pt") as f:
        t = f.get_tensor(key)
    if t.dtype == torch.bfloat16:
        return t.to(torch.float32).numpy()
    return t.numpy()


def _has(idx: ShardIndex, key: str) -> bool:
    return key in idx.key_to_shard


def read_int4_weight(idx: ShardIndex, base: str) -> np.ndarray:
    """Read + dequantize a compressed-tensors INT4 weight.

    `base` is the module path WITHOUT the `.weight_packed` suffix, e.g.
    "language_model.model.layers.5.self_attn.q_b_proj".
    Returns (out, in) float32.

    Falls back to reading `<base>.weight` as bf16 if the tensor isn't
    compressed (some attention norms etc. are stored bf16).
    """
    kp = base + ".weight_packed"
    if kp in idx.key_to_shard:
        ks = base + ".weight_scale"
        kh = base + ".weight_shape"
        packed = read_tensor(idx, kp)
        scale = read_tensor(idx, ks).astype(np.float32, copy=False)
        shape = read_tensor(idx, kh)
        return unpack_int4(packed, scale, shape, group_size=32)
    # Not compressed — plain bf16/f32 weight (read_tensor already promotes bf16 to f32).
    return read_tensor(idx, base + ".weight").astype(np.float32, copy=False)


def read_router(idx: ShardIndex, layer: int):
    """Return (router_weight, e_score_correction_bias).

    router_weight: (E, H) float32 from language_model.model.layers.L.mlp.gate.weight
    bias: (E,) float32 or None if absent
    """
    wk = f"language_model.model.layers.{layer}.mlp.gate.weight"
    bk = f"language_model.model.layers.{layer}.mlp.gate.e_score_correction_bias"
    w = read_tensor(idx, wk)
    if w.dtype == np.uint16:
        w = bf16_u16_to_f32(w)
    else:
        w = w.astype(np.float32)
    bias = None
    if bk in idx.key_to_shard:
        b = read_tensor(idx, bk)
        bias = b.astype(np.float32)
    return w, bias


def read_expert(idx: ShardIndex, layer: int, expert: int) -> dict[str, np.ndarray]:
    """Return {'gate_proj': (..., H), 'up_proj': (..., H), 'down_proj': (H, ...)} float32."""
    base = f"language_model.model.layers.{layer}.mlp.experts.{expert}"
    out = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        out[proj] = read_int4_weight(idx, f"{base}.{proj}")
    return out


def read_shared_expert(idx: ShardIndex, layer: int) -> dict[str, np.ndarray] | None:
    """Shared expert (always-on). Returns None if not present."""
    base = f"language_model.model.layers.{layer}.mlp.shared_experts"
    # Probe for existence.
    if not any(k.startswith(base + ".gate_proj") for k in idx.key_to_shard):
        return None
    return {
        "gate_proj": read_int4_weight(idx, f"{base}.gate_proj"),
        "up_proj":   read_int4_weight(idx, f"{base}.up_proj"),
        "down_proj": read_int4_weight(idx, f"{base}.down_proj"),
    }


def read_dense_mlp(idx: ShardIndex, layer: int) -> dict[str, np.ndarray]:
    """Dense MLP for layer 0 (first_k_dense_replace=1)."""
    base = f"language_model.model.layers.{layer}.mlp"
    return {
        "gate_proj": read_int4_weight(idx, f"{base}.gate_proj"),
        "up_proj":   read_int4_weight(idx, f"{base}.up_proj"),
        "down_proj": read_int4_weight(idx, f"{base}.down_proj"),
    }


def read_mla_attention(idx: ShardIndex, layer: int) -> dict[str, np.ndarray]:
    """MLA attention weights + layernorms for one layer."""
    base = f"language_model.model.layers.{layer}.self_attn"
    out = {
        "q_a_proj":            read_int4_weight(idx, f"{base}.q_a_proj"),
        "q_a_layernorm":       read_int4_weight(idx, f"{base}.q_a_layernorm"),
        "q_b_proj":            read_int4_weight(idx, f"{base}.q_b_proj"),
        "kv_a_proj_with_mqa":  read_int4_weight(idx, f"{base}.kv_a_proj_with_mqa"),
        "kv_a_layernorm":      read_int4_weight(idx, f"{base}.kv_a_layernorm"),
        "kv_b_proj":           read_int4_weight(idx, f"{base}.kv_b_proj"),
        "o_proj":              read_int4_weight(idx, f"{base}.o_proj"),
    }
    return out


def read_layernorms(idx: ShardIndex, layer: int) -> dict[str, np.ndarray]:
    base = f"language_model.model.layers.{layer}"
    return {
        "input_layernorm":          read_int4_weight(idx, f"{base}.input_layernorm"),
        "post_attention_layernorm": read_int4_weight(idx, f"{base}.post_attention_layernorm"),
    }


def read_embed(idx: ShardIndex) -> np.ndarray:
    """Token embedding table (V, H) float32."""
    return read_int4_weight(idx, "language_model.model.embed_tokens")


def read_final_norm(idx: ShardIndex) -> np.ndarray:
    return read_int4_weight(idx, "language_model.model.norm")

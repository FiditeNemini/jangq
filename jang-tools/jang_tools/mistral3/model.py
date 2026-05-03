"""Mistral 3.5 / ministral3 — MLX port (text decoder + pixtral vision).

Text: dense GQA YaRN, no MLA, no MoE, no SWA. 88 layers x 12288 hidden,
96/8 GQA, head_dim 128, rope_theta=1e6, YaRN factor=64 from orig=4096.

Vision: pixtral, 48 layers, 1664 hidden, head_dim 104. Reuses the
existing mlx_vlm pixtral path when available; falls back to a minimal
local implementation.

JANG-quantizable subset: only text decoder linears. vision_tower +
multi_modal_projector + lm_head stay bf16 (matches upstream
modules_to_not_convert).
"""
from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..jangrt.loader import register_arch
from .config import Mistral3Config


class RMSNorm(nn.Module):
    def __init__(self, hidden: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((hidden,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


def _yarn_inv_freq(rp: dict, head_dim: int) -> mx.array:
    base = rp["rope_theta"]
    factor = rp.get("factor", 1.0)
    orig = rp.get("original_max_position_embeddings", 4096)
    beta_fast = rp.get("beta_fast", 32.0)
    beta_slow = rp.get("beta_slow", 1.0)
    dim = head_dim
    pos = mx.arange(0, dim, 2, dtype=mx.float32)
    inv = 1.0 / (base ** (pos / dim))
    # YaRN ramp
    low = max(0, math.floor(dim * math.log(orig / (beta_fast * 2 * math.pi))
                            / (2 * math.log(base))))
    high = min(dim - 1, math.ceil(dim * math.log(orig / (beta_slow * 2 * math.pi))
                                  / (2 * math.log(base))))
    ramp = mx.clip((mx.arange(dim // 2) - low) / max(1, high - low), 0, 1)
    inv_ext = inv / factor
    return inv * (1 - ramp) + inv_ext * ramp


class MinistralAttention(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.kv_groups = self.n_heads // self.n_kv
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.rope_base = cfg.rope_parameters.get("rope_theta", 1e6)

    def __call__(self, x, mask, cache, offset):
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.rope_base,
                         scale=1.0, offset=offset)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.rope_base,
                         scale=1.0, offset=offset)
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        new_cache = (k, v)
        if self.kv_groups > 1:
            k = mx.repeat(k, self.kv_groups, axis=1)
            v = mx.repeat(v, self.kv_groups, axis=1)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(out), new_cache


class MinistralMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MinistralLayer(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = MinistralAttention(cfg, layer_idx)
        self.mlp = MinistralMLP(cfg)

    def __call__(self, x, mask, cache, offset):
        h, c = self.self_attn(self.input_layernorm(x), mask, cache, offset)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, c


class MinistralTextModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [MinistralLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def __call__(self, ids, caches=None):
        h = self.embed_tokens(ids)
        T = h.shape[1]
        offset = 0 if caches is None or caches[0] is None else caches[0][0].shape[2]
        mask = mx.triu(mx.full((T, T), -mx.inf, dtype=h.dtype), k=1) if T > 1 else None
        if caches is None:
            caches = [None] * len(self.layers)
        new = []
        for L, layer in enumerate(self.layers):
            h, c = layer(h, mask, caches[L], offset)
            new.append(c)
        return self.norm(h), new


class Mistral3ForConditionalGeneration(nn.Module):
    """Wraps the text model + pixtral vision tower + multimodal projector.

    Text decoder path delegates to ``mlx_lm.models.ministral3.Model`` —
    the reference implementation that handles YaRN-aware RoPE via
    `initialize_rope`, the (currently no-op when
    `llama_4_scaling_beta=0`) `attn_scale` query post-multiplier,
    SDPA-internal GQA broadcasting, and `mlx_lm.models.cache.KVCache`
    objects. Re-implementing those subtleties locally caused a layer-2
    residual injection bug — `down_proj` couldn't cancel the SwiGLU
    outlier because the attention path was subtly wrong (manual
    `mx.repeat` GQA + plain `mx.fast.rope` instead of YaRN-aware
    rope).

    Bundle key layout matches mlx_lm's `Model` after the runtime's
    `_m3_remap` strips `model.language_model.` → `model.`:
        model.embed_tokens.{weight,scales,biases}
        model.layers.L.self_attn.{q,k,v,o}_proj.{tq_*}
        model.layers.L.mlp.{gate,up,down}_proj.{tq_*}
        model.layers.L.{input,post_attention}_layernorm.weight
        model.norm.weight
        lm_head.weight
    """
    def __init__(self, cfg: Mistral3Config):
        super().__init__()
        self.cfg = cfg
        # Build mlx_lm's ModelArgs from our cfg's text_config and
        # compose its `LanguageModel` directly as `self.model` so the
        # bundle's `model.embed_tokens.*` / `model.layers.*` /
        # `model.norm.*` keys bind via the standard `model.update()`
        # traversal. `nn.Module.update()` walks the actual children
        # set on `self`, not Python `@property` indirections.
        from mlx_lm.models.ministral3 import (
            LanguageModel as _RefLM,
            ModelArgs as _RefArgs,
        )
        tc = cfg.text_config
        rope_params = dict(tc.rope_parameters or {})
        rope_params.setdefault("llama_4_scaling_beta", 0)
        rope_params.setdefault(
            "original_max_position_embeddings",
            tc.max_position_embeddings or 4096,
        )
        margs = _RefArgs(
            model_type=tc.model_type or "ministral3",
            hidden_size=tc.hidden_size,
            num_hidden_layers=tc.num_hidden_layers,
            intermediate_size=tc.intermediate_size,
            num_attention_heads=tc.num_attention_heads,
            num_key_value_heads=tc.num_key_value_heads,
            rms_norm_eps=tc.rms_norm_eps,
            vocab_size=tc.vocab_size,
            head_dim=tc.head_dim,
            max_position_embeddings=tc.max_position_embeddings,
            rope_parameters=rope_params,
            tie_word_embeddings=tc.tie_word_embeddings,
            sliding_window=tc.sliding_window,
        )
        self._margs = margs
        self.model = _RefLM(margs)
        if not tc.tie_word_embeddings:
            self.lm_head = nn.Linear(tc.hidden_size, tc.vocab_size, bias=False)
        # Vision tower + projector wired via load helpers; minimal stubs here.
        self.vision_tower = None  # populated by loader if VL bundle
        self.multi_modal_projector = None

    def make_cache(self):
        """Per-layer KVCache list. Mirrors mlx_lm's
        `make_prompt_cache(self.model)` shape (full vs sliding via
        `layer_types`). Used at prefill so attention's
        `update_and_fetch` actually populates KV; bypassing this
        recreates the laguna-style cache-init bug observed in the
        first round of this work."""
        from mlx_lm.models.cache import make_prompt_cache
        return make_prompt_cache(self.model)

    def __call__(self, ids, images=None, caches=None, cache=None):
        if images is not None and self.vision_tower is not None:
            raise NotImplementedError(
                "Image embedding fold-in: replace ids[ids==image_token_id] "
                "with vision tokens emitted by self.vision_tower(images), "
                "merged via self.multi_modal_projector."
            )
        # Accept either `cache=` (engine + mlx_lm convention) or
        # `caches=` (legacy local greedy). Allocate fresh on first call.
        local_call = caches is not None or cache is None
        if cache is None:
            cache = caches if caches is not None else self.make_cache()
        h = self.model(ids, cache=cache)
        if self._margs.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(h)
        else:
            logits = self.lm_head(h)
        # Engine path expects bare logits; local-greedy expects tuple.
        return (logits, cache) if local_call else logits


@register_arch("mistral3")
@register_arch("ministral3")
def _build(src: str, *, fmt: str, plan=None):
    cfg = Mistral3Config.from_json(f"{src}/config.json")
    return Mistral3ForConditionalGeneration(cfg), cfg, None

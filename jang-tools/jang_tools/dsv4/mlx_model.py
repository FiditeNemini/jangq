"""MLX model file for DeepSeek-V4 — the runtime mlx_lm plugs into.

Mirrors mlx_lm/models/deepseek_v32.py patterns with DSV4-specific changes:
- MLA with head_dim=512, o_lora_rank+o_groups grouped output projection
- mHC (Manifold-Constrained Hyper-Connections) wrapping attn + ffn
- sqrtsoftplus scoring + hash-routing for first N layers
- Full attention (no CSA/HCA yet — those are Phase 7.5B.2)
- No MTP head at inference (discarded per DSV convention)

This file is registered into mlx_lm.models at runtime via
`jang_tools.dsv4.mlx_register`, so `load_jangtq_model` works on
DSV4-Flash bundles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs, create_attention_mask, scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 2048
    num_hash_layers: int = 3
    num_nextn_predict_layers: int = 1
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    # mHC
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # RoPE
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    compress_rope_theta: float = 160000.0
    max_position_embeddings: int = 1048576
    sliding_window: int = 128
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    # Unused but present in config
    hc_mult_: int = 4
    compress_ratios: Optional[List[int]] = None
    # Indexer (for compress_ratio=4 layers)
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512


# ---------- Pure-MLX ops ----------

def _hc_split_sinkhorn_ops(
    mixes: mx.array, hc_scale: mx.array, hc_base: mx.array,
    hc_mult: int, iters: int = 20, eps: float = 1e-6,
):
    """Pure-MLX implementation matching mlx_lm PR #1192 deepseek_v4 reference.
    Fallback when fused Metal kernel is unavailable (CPU backend or no Metal).

    Splits mixes (shape (..., (2+mult)*mult)) into (pre, post, comb):
      pre:  (..., mult)         — sigmoid + eps, NO normalization
      post: (..., mult)         — 2 * sigmoid, NO eps (factor of 2 is critical)
      comb: (..., mult, mult)   — doubly-stochastic via Sinkhorn
                                  (softmax init + col-norm + (iters-1) row/col iterations)
    """
    mixes = mixes.astype(mx.float32)
    hc_scale = hc_scale.astype(mx.float32)
    hc_base = hc_base.astype(mx.float32)
    mh = hc_mult
    pre_scale, post_scale, comb_scale = hc_scale[0], hc_scale[1], hc_scale[2]

    pre = mx.sigmoid(mixes[..., :mh] * pre_scale + hc_base[:mh]) + eps
    post = 2 * mx.sigmoid(mixes[..., mh:2 * mh] * post_scale + hc_base[mh:2 * mh])
    comb = mx.reshape(
        mixes[..., 2 * mh:] * comb_scale,
        mixes.shape[:-1] + (mh, mh),
    ) + mx.reshape(hc_base[2 * mh:], (mh, mh))
    comb = mx.softmax(comb, axis=-1, precise=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(max(iters - 1, 0)):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def _make_hc_split_sinkhorn_kernel():
    """Fused Metal kernel for HC Sinkhorn. Ports mlx-lm PR #1192 latest optim
    (commit c0d9222d, 2026-04-24). Does the entire pre/post/comb compute in
    a SINGLE GPU kernel launch — avoids 40+ intermediate MLX op graphs per
    layer × 43 layers = 3000+ graph nodes saved per token.

    Returns None if Metal is unavailable (fallback to pure-ops path).
    """
    try:
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            return None
    except Exception:
        return None

    source = """
        uint idx = thread_position_in_grid.x;
        constexpr int MIX = (2 + HC) * HC;
        float epsv = static_cast<float>(eps[0]);

        auto mix = mixes + idx * MIX;
        auto pre_out = pre + idx * HC;
        auto post_out = post + idx * HC;
        auto comb_out = comb + idx * HC * HC;

        float pre_scale = static_cast<float>(scale[0]);
        float post_scale = static_cast<float>(scale[1]);
        float comb_scale = static_cast<float>(scale[2]);

        for (int i = 0; i < HC; ++i) {
            float z = static_cast<float>(mix[i]) * pre_scale
                + static_cast<float>(base[i]);
            pre_out[i] = 1.0f / (1.0f + metal::fast::exp(-z)) + epsv;
        }
        for (int i = 0; i < HC; ++i) {
            int off = HC + i;
            float z = static_cast<float>(mix[off]) * post_scale
                + static_cast<float>(base[off]);
            post_out[i] = 2.0f / (1.0f + metal::fast::exp(-z));
        }

        float c[HC * HC];
        for (int i = 0; i < HC; ++i) {
            float row_max = -INFINITY;
            for (int j = 0; j < HC; ++j) {
                int cidx = i * HC + j;
                int off = 2 * HC + cidx;
                float v = static_cast<float>(mix[off]) * comb_scale
                    + static_cast<float>(base[off]);
                c[cidx] = v;
                row_max = metal::max(row_max, v);
            }
            float row_sum = 0.0f;
            for (int j = 0; j < HC; ++j) {
                int cidx = i * HC + j;
                float v = metal::fast::exp(c[cidx] - row_max);
                c[cidx] = v;
                row_sum += v;
            }
            float inv_sum = 1.0f / row_sum;
            for (int j = 0; j < HC; ++j) {
                int cidx = i * HC + j;
                c[cidx] = c[cidx] * inv_sum + epsv;
            }
        }

        for (int j = 0; j < HC; ++j) {
            float col_sum = 0.0f;
            for (int i = 0; i < HC; ++i) {
                col_sum += c[i * HC + j];
            }
            float inv_denom = 1.0f / (col_sum + epsv);
            for (int i = 0; i < HC; ++i) {
                c[i * HC + j] *= inv_denom;
            }
        }

        for (int iter = 1; iter < ITERS; ++iter) {
            for (int i = 0; i < HC; ++i) {
                float row_sum = 0.0f;
                for (int j = 0; j < HC; ++j) {
                    row_sum += c[i * HC + j];
                }
                float inv_denom = 1.0f / (row_sum + epsv);
                for (int j = 0; j < HC; ++j) {
                    c[i * HC + j] *= inv_denom;
                }
            }
            for (int j = 0; j < HC; ++j) {
                float col_sum = 0.0f;
                for (int i = 0; i < HC; ++i) {
                    col_sum += c[i * HC + j];
                }
                float inv_denom = 1.0f / (col_sum + epsv);
                for (int i = 0; i < HC; ++i) {
                    c[i * HC + j] *= inv_denom;
                }
            }
        }

        for (int i = 0; i < HC * HC; ++i) {
            comb_out[i] = c[i];
        }
    """

    return mx.fast.metal_kernel(
        name="deepseek_v4_hc_split_sinkhorn",
        input_names=["mixes", "scale", "base", "eps"],
        output_names=["pre", "post", "comb"],
        source=source,
    )


_hc_split_sinkhorn_kernel = _make_hc_split_sinkhorn_kernel()
_hc_eps_array_cache = None


def hc_split_sinkhorn(
    mixes: mx.array, hc_scale: mx.array, hc_base: mx.array,
    hc_mult: int, iters: int = 20, eps: float = 1e-6,
):
    """Public API — dispatches to fused Metal kernel when available.
    Same output semantics as `_hc_split_sinkhorn_ops`.
    """
    if _hc_split_sinkhorn_kernel is None:
        return _hc_split_sinkhorn_ops(mixes, scale=hc_scale, base=hc_base,
                                       hc_mult=hc_mult, iters=iters, eps=eps)
    global _hc_eps_array_cache
    if _hc_eps_array_cache is None:
        _hc_eps_array_cache = mx.array([eps], dtype=mx.float32)
    return _hc_split_sinkhorn_kernel(
        inputs=[mixes, hc_scale, hc_base, _hc_eps_array_cache],
        template=[("HC", hc_mult), ("ITERS", iters)],
        grid=(mixes.size // ((2 + hc_mult) * hc_mult), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (*mixes.shape[:-1], hc_mult),
            (*mixes.shape[:-1], hc_mult),
            (*mixes.shape[:-1], hc_mult, hc_mult),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )


# ---------- Attention (simplified: full scaled_dot_product) ----------


def act_quant_sim(x: mx.array, block_size: int = 128) -> mx.array:
    """FP8 e4m3 fake-quant on activations (block_size=128 per last dim).

    Reference (`inference/kernel.py act_quant_kernel`) applies this before
    every Linear with FP4/FP8 weights during inference. The model was
    TRAINED expecting this activation noise — skipping it gives weights
    cleaner-than-trained inputs, which paradoxically hurts accuracy.

    e4m3fn has fp8_max=448. Per-block scale = amax/fp8_max rounded to
    power-of-2 (fast_round_scale). Quantize → 8-bit e4m3 levels → dequant.

    Enable with env DSV4_ACT_QUANT=1; default OFF because it adds overhead
    and most runtime queries work without it. For arithmetic-heavy
    reasoning, turn ON.
    """
    import os as _os
    if _os.environ.get("DSV4_ACT_QUANT", "0") != "1":
        return x
    orig_dtype = x.dtype
    shape = x.shape
    if shape[-1] % block_size != 0:
        return x
    x32 = x.astype(mx.float32)
    reshaped = x32.reshape(*shape[:-1], -1, block_size)
    fp8_max = 448.0
    amax = mx.max(mx.abs(reshaped), axis=-1, keepdims=True)
    # fast_round_scale: 2^ceil(log2(amax/fp8_max)), zero-safe
    raw_scale = amax / fp8_max
    # Handle zero blocks — leave them as 1.0 (no-op)
    safe_scale = mx.where(raw_scale > 1e-30, raw_scale, mx.ones_like(raw_scale))
    log2s = mx.ceil(mx.log2(safe_scale))
    scale = mx.power(mx.array(2.0, dtype=mx.float32), log2s)
    # e4m3fn has 3 mantissa bits → ~8 equal-ratio levels per binade.
    # Approximation: uniform 8-bit levels within ±fp8_max. Real e4m3 is
    # log-spaced per binade; this simplification is close enough for
    # the fake-quant effect the model was trained with.
    normalized = reshaped / scale
    # Round-trip through 8-bit signed range [-127, 127] scaled to [-448, 448]
    q = mx.round(normalized * (127.0 / fp8_max)) * (fp8_max / 127.0)
    q = mx.clip(q, -fp8_max, fp8_max)
    result = (q * scale).reshape(shape)
    return result.astype(orig_dtype)


class DeepseekV4RoPE(nn.Module):
    """Port of PR #1192 DeepseekV4RoPE — on-the-fly cos/sin for YaRN."""
    def __init__(self, dims, base, scaling_config=None, max_position_embeddings=1048576):
        super().__init__()
        self.dims = dims
        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        rope_type = None
        if scaling_config is not None:
            rope_type = scaling_config.get("type") or scaling_config.get("rope_type")
        if rope_type in ("yarn", "deepseek_yarn"):
            factor = scaling_config["factor"]
            orig = scaling_config["original_max_position_embeddings"]
            beta_fast = scaling_config.get("beta_fast", 32)
            beta_slow = scaling_config.get("beta_slow", 1)
            def correction_dim(n):
                return dims * math.log(orig / (n * 2 * math.pi)) / (2 * math.log(base))
            low = max(math.floor(correction_dim(beta_fast)), 0)
            high = min(math.ceil(correction_dim(beta_slow)), dims - 1)
            if low == high:
                high += 0.001
            ramp = (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low)
            smooth = 1 - mx.clip(ramp, 0, 1)
            inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth
        elif rope_type not in (None, "default"):
            raise ValueError(f"Unsupported DeepSeek-V4 RoPE type: {rope_type}")
        self._inv_freq = (inv_freq,)

    @property
    def inv_freq(self):
        return self._inv_freq[0]

    def __call__(self, x, offset=0, inverse=False, positions=None):
        # NOTE: mx.fast.rope was tried as a fast path here but produced
        # incoherent output (likely an inv_freq layout/scale convention
        # mismatch with YaRN-modified freqs). Reverted to manual cos/sin
        # path which is verified bit-exact against PR #1192 reference.
        # Future: investigate exact mx.fast.rope freqs format requirements.
        dtype = x.dtype
        L = x.shape[-2]
        pos = (
            mx.arange(offset, offset + L, dtype=mx.float32)
            if positions is None
            else positions.astype(mx.float32)
        )
        freqs = pos[:, None] * self.inv_freq[None, :]
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)
        if inverse:
            sin = -sin
        broadcast_shape = (1,) * (x.ndim - 2) + cos.shape
        cos = cos.reshape(broadcast_shape).astype(dtype)
        sin = sin.reshape(broadcast_shape).astype(dtype)
        x = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        x0, x1 = x[..., 0], x[..., 1]
        out = mx.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)
        return out.reshape(*out.shape[:-2], out.shape[-2] * 2)


def _apply_partial_rope(x, rope, offset=0, inverse=False, positions=None):
    rope_dim = rope.dims
    if x.shape[-1] == rope_dim:
        return rope(x, offset=offset, inverse=inverse, positions=positions)
    nope, pe = mx.split(x, [x.shape[-1] - rope_dim], axis=-1)
    pe = rope(pe, offset=offset, inverse=inverse, positions=positions)
    return mx.concatenate([nope, pe], axis=-1)


class DeepseekV4Cache:
    """Simplified cache for DSV4: wraps a plain KVCache with compressor/indexer
    state buffers (cross-window pooling). For short prompts (<sliding_window),
    the plain KVCache is equivalent to RotatingKVCache."""
    def __init__(self, sliding_window):
        from mlx_lm.models.cache import RotatingKVCache
        self.local = RotatingKVCache(max_size=sliding_window, keep=0)
        self.compressor_state = {"buffer_kv": None, "buffer_gate": None, "pooled": None}
        self.indexer_state = {"buffer_kv": None, "buffer_gate": None, "pooled": None}

    @property
    def offset(self):
        return self.local.offset

    @property
    def keys(self):
        return self.local.keys

    @keys.setter
    def keys(self, value):
        self.local.keys = value

    @property
    def state(self):
        """Cache state tuple — mlx_lm generate iterates this for pipelined evaluation."""
        local_state = None if self.local.empty() else self.local.state
        return (
            local_state,
            tuple(self.compressor_state[k] for k in ("buffer_kv", "buffer_gate", "pooled")),
            tuple(self.indexer_state[k] for k in ("buffer_kv", "buffer_gate", "pooled")),
        )

    @state.setter
    def state(self, value):
        local_state, compressor_state, indexer_state = value
        if local_state is None:
            self.local.keys = None
            self.local.values = None
        else:
            self.local.state = local_state
        self.compressor_state = dict(zip(("buffer_kv", "buffer_gate", "pooled"), compressor_state))
        self.indexer_state = dict(zip(("buffer_kv", "buffer_gate", "pooled"), indexer_state))

    @property
    def meta_state(self):
        return self.local.meta_state

    @meta_state.setter
    def meta_state(self, value):
        self.local.meta_state = value

    def update_and_fetch(self, keys, values):
        return self.local.update_and_fetch(keys, values)

    def make_mask(self, *a, **k):
        return self.local.make_mask(*a, **k)

    def is_trimmable(self):
        return self.local.is_trimmable()

    def trim(self, n):
        return self.local.trim(n)

    def size(self):
        return self.local.size()

    def empty(self):
        return self.local.empty()

    @property
    def nbytes(self):
        total = self.local.nbytes
        for state in (self.compressor_state, self.indexer_state):
            for value in state.values():
                if value is not None:
                    total += value.nbytes
        return total

    def _branch_state(self, key):
        return self.indexer_state if key == "indexer_state" else self.compressor_state

    def accumulate_windows(self, kv, gate, state_key, ratio, start_pos):
        state = self._branch_state(state_key)
        buf_kv, buf_gate = state["buffer_kv"], state["buffer_gate"]
        if buf_kv is not None and buf_kv.shape[1]:
            kv = mx.concatenate([buf_kv, kv], axis=1)
            gate = mx.concatenate([buf_gate, gate], axis=1)
        usable = (kv.shape[1] // ratio) * ratio
        state["buffer_kv"] = kv[:, usable:]
        state["buffer_gate"] = gate[:, usable:]
        pool_base = max(0, start_pos) - (buf_kv.shape[1] if buf_kv is not None else 0)
        return kv[:, :usable], gate[:, :usable], pool_base

    def update_pool(self, new_pooled, state_key):
        state = self._branch_state(state_key)
        pool = state["pooled"]
        if new_pooled.shape[1] > 0:
            pool = new_pooled if pool is None else mx.concatenate([pool, new_pooled], axis=1)
            state["pooled"] = pool
        if pool is None:
            pool = mx.zeros((new_pooled.shape[0], 0, new_pooled.shape[-1]), new_pooled.dtype)
        return pool


class Compressor(nn.Module):
    def __init__(self, config, compress_ratio, head_dim):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.overlap = compress_ratio == 4
        self.out_dim = head_dim * (2 if self.overlap else 1)
        self.wkv = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self.ape = mx.zeros((compress_ratio, self.out_dim), dtype=mx.float32)
        self.norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)

    def _overlap_transform(self, x, fill_value):
        B, W, R, _ = x.shape
        out = mx.full((B, W, 2 * R, self.head_dim), fill_value, dtype=x.dtype)
        out[:, :, R:] = x[:, :, :, self.head_dim:]
        out[:, 1:, :R] = x[:, :-1, :, :self.head_dim]
        return out

    def __call__(self, x, rope, cache, start_pos, state_key="compressor_state"):
        B, _, _ = x.shape
        kv = self.wkv(x)
        gate = self.wgate(x)
        if cache is None:
            usable = (kv.shape[1] // self.compress_ratio) * self.compress_ratio
            ready_kv, ready_gate = kv[:, :usable], gate[:, :usable]
            pool_base = start_pos
        else:
            ready_kv, ready_gate, pool_base = cache.accumulate_windows(
                kv, gate, state_key, self.compress_ratio, start_pos
            )
        if ready_kv.shape[1] == 0:
            new_pooled = mx.zeros((B, 0, self.head_dim), dtype=x.dtype)
        else:
            W = ready_kv.shape[1] // self.compress_ratio
            kv = ready_kv.reshape(B, W, self.compress_ratio, self.out_dim)
            gate = ready_gate.reshape(B, W, self.compress_ratio, self.out_dim) + self.ape.astype(ready_gate.dtype)
            if self.overlap:
                kv = self._overlap_transform(kv, 0.0)
                gate = self._overlap_transform(gate, -float("inf"))
            weights = mx.softmax(gate.astype(mx.float32), axis=2, precise=True).astype(kv.dtype)
            new_pooled = (kv * weights).sum(axis=2)
            new_pooled = self.norm(new_pooled.astype(x.dtype))
            positions = (
                mx.arange(new_pooled.shape[1], dtype=mx.float32) * self.compress_ratio
                + pool_base
            )
            new_pooled = _apply_partial_rope(new_pooled[:, None], rope, positions=positions).squeeze(1)
        if cache is not None:
            return cache.update_pool(new_pooled, state_key)
        return new_pooled


class Indexer(nn.Module):
    def __init__(self, config, compress_ratio):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.compressor = Compressor(config, compress_ratio, self.head_dim)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x, q_residual, rope, position_rope, cache, start_pos):
        B, L, _ = x.shape
        pooled = self.compressor(x, rope, cache, start_pos, state_key="indexer_state")
        if pooled.shape[1] == 0:
            return None
        offset = start_pos
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        q = _apply_partial_rope(q, position_rope, offset)
        scores = q.astype(mx.float32) @ pooled[:, None].swapaxes(-1, -2).astype(mx.float32)
        scores = mx.maximum(scores, 0) * self.scale
        weights = self.weights_proj(x).astype(mx.float32) * (self.n_heads ** -0.5)
        scores = (scores * weights.swapaxes(-1, -2)[..., None]).sum(axis=1)
        k = min(self.index_topk, pooled.shape[1])
        return mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]


def _mlx_apply_rotary_cis(x: mx.array, freqs_cis_real: mx.array) -> mx.array:
    """MLX port of DSV4's apply_rotary_emb.
    x: (..., rd) where rd is even.
    freqs_cis_real: (L, rd/2, 2) packed as [cos, sin] pairs.

    Returns rotated x with same shape. Handles any leading dims for x;
    freqs_cis broadcasts on the seq-len axis.

    CAUTION: unlike the torch reference this does NOT mutate x in place.
    """
    dtype = x.dtype
    shape = x.shape
    rd = shape[-1]
    x32 = x.astype(mx.float32).reshape(*shape[:-1], rd // 2, 2)
    xa = x32[..., 0]
    xb = x32[..., 1]
    # freqs_cis_real: (L, rd/2, 2) -> cos = [...,0], sin = [...,1]
    cos = freqs_cis_real[..., 0]
    sin = freqs_cis_real[..., 1]
    # Broadcast cos/sin over leading dims of xa/xb
    ya = xa * cos - xb * sin
    yb = xa * sin + xb * cos
    out = mx.stack([ya, yb], axis=-1)
    return mx.reshape(out, shape).astype(dtype)


def _precompute_freqs_cis_real(
    dim: int, seqlen: int, original_seq_len: int,
    base: float, factor: float, beta_fast: int, beta_slow: int,
) -> mx.array:
    """Precompute (seqlen, dim/2, 2) real-valued [cos, sin] pairs for YaRN RoPE.

    Matches PR #1192 DeepseekV4RoPE YaRN formula — notably `high` is clamped
    to `dim - 1` (not `dim // 2 - 1`). Previous `dim // 2 - 1` clamp gave a
    steeper smoothing ramp, producing rotated q/k that diverged from
    reference by ~12% RMS in attention output.

    Also mirrors reference's smoothing sign: `smooth = 1 - clip(ramp)`,
    freqs = (inv_freq / factor) * (1 - smooth) + inv_freq * smooth.
    """
    import math
    idx = mx.arange(0, dim, 2).astype(mx.float32)
    freqs = 1.0 / (base ** (idx / dim))
    if original_seq_len > 0 and factor > 1:
        def correction_dim(n):
            return dim * math.log(original_seq_len / (n * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = (mx.arange(dim // 2).astype(mx.float32) - low) / (high - low)
        smooth = 1 - mx.clip(ramp, 0, 1)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = mx.arange(seqlen).astype(mx.float32)
    theta = mx.outer(t, freqs)  # (seqlen, dim/2)
    cos = mx.cos(theta)
    sin = mx.sin(theta)
    return mx.stack([cos, sin], axis=-1)  # (seqlen, dim/2, 2)


class DeepseekV4Attention(nn.Module):
    """MLA with low-rank Q and grouped low-rank O.

    Per-layer RoPE: reference PR #1192 uses different rope configs based on
    `compress_ratio` for the layer. Layers with compress_ratio=0 (first + last)
    use base rope_theta=10000 with NO YaRN. Layers with compress_ratio>0
    (middle 41 layers) use compress_rope_theta=160000 WITH YaRN.

    We don't implement compressor/indexer yet, but we MUST still use the
    correct per-layer rope config or all middle layers drift catastrophically
    (std grows exponentially, hitting bf16 inf by layer 40).
    """
    def __init__(self, args: ModelArgs, layer_id: int = 0):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.hidden_size = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads  # typically 1 for DSV4
        self.head_dim = args.head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.nope_head_dim = args.head_dim - args.qk_rope_head_dim
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.o_groups = args.o_groups

        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        # Grouped low-rank O: wo_a(grouped input) → wo_b(concat to hidden)
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // self.o_groups,
            self.o_groups * self.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(self.o_groups * self.o_lora_rank, self.hidden_size, bias=False)
        self.attn_sink = mx.zeros((self.n_heads,))

        self.softmax_scale = self.head_dim ** -0.5

        # Resolve per-layer compress_ratio from args.compress_ratios (bundle
        # config.json carries this as an explicit list of 43+1=44 entries).
        compress_ratios = getattr(args, "compress_ratios", None)
        if compress_ratios and layer_id < len(compress_ratios):
            compress_ratio = compress_ratios[layer_id]
        else:
            n = args.num_hidden_layers
            if layer_id == 0 or layer_id == n - 1:
                compress_ratio = 0
            else:
                i = layer_id - 1
                compress_ratio = 4 if i % 2 else 128
        self.compress_ratio = compress_ratio

        # Per-layer RoPE: compress_ratio > 0 uses compress_rope_theta + YaRN.
        if compress_ratio:
            rope_theta = args.compress_rope_theta
            rope_scaling = args.rope_scaling
        else:
            rope_theta = args.rope_theta
            rope_scaling = None
        self.rope = DeepseekV4RoPE(
            args.qk_rope_head_dim, rope_theta, rope_scaling, args.max_position_embeddings,
        )
        self.compress_rope = self.rope

        # Instantiate Compressor + Indexer for layers with compress_ratio > 0.
        if compress_ratio:
            self.compressor = Compressor(args, compress_ratio, self.head_dim)
            if compress_ratio == 4:
                self.indexer = Indexer(args, compress_ratio)

    def __call__(self, x, mask=None, cache=None):
        # Match PR #1192 V4Attention forward. Handles compress_ratio>0 layers
        # via Compressor + Indexer, appending pooled global context to local KV.
        B, L, _ = x.shape
        local_cache = cache if isinstance(cache, DeepseekV4Cache) else cache
        offset = local_cache.offset if local_cache is not None else 0

        q_residual = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = q * mx.rsqrt(
            (q.astype(mx.float32) ** 2).mean(axis=-1, keepdims=True)
            + self.args.rms_norm_eps
        )
        q = q.astype(x.dtype)
        q = q.transpose(0, 2, 1, 3)

        kv = self.kv_norm(self.wkv(x)).reshape(B, L, 1, self.head_dim)
        kv = kv.transpose(0, 2, 1, 3)

        q = _apply_partial_rope(q, self.rope, offset)
        kv = _apply_partial_rope(kv, self.rope, offset)

        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, kv)
        full_kv = kv

        if self.compress_ratio:
            v4_cache = cache if isinstance(cache, DeepseekV4Cache) else None
            # FAST PATH: when NOT using DeepseekV4Cache (i.e., plain KVCache),
            # the compressor has no buffer state to accumulate. For L < compress_ratio
            # the pooled output is empty and gets no-op concat. Skip entirely to
            # save ~150 matmuls per token across 41 compress_ratio>0 layers.
            #
            # Only run Compressor/Indexer if:
            # - v4_cache is provided (state carries across calls), OR
            # - L >= compress_ratio (enough tokens to produce non-empty pool in one call)
            if v4_cache is not None or L >= self.compress_ratio:
                pooled = self.compressor(x, self.compress_rope, v4_cache, offset)
                if hasattr(self, "indexer") and pooled.shape[1] > 0:
                    topk = self.indexer(x, q_residual, self.compress_rope, self.rope, v4_cache, offset)
                    if topk is not None:
                        expanded = mx.broadcast_to(
                            pooled[:, None, None, :, :],
                            (B, 1, L, pooled.shape[1], self.head_dim),
                        )
                        idx = topk[:, None, :, :, None]
                        pooled = mx.take_along_axis(
                            expanded,
                            mx.broadcast_to(idx, idx.shape[:-1] + (self.head_dim,)),
                            axis=3,
                        ).reshape(B, 1, -1, self.head_dim)
                    else:
                        pooled = pooled[:, None]
                else:
                    pooled = pooled[:, None]
                if pooled.shape[2] > 0:
                    full_kv = mx.concatenate([full_kv, pooled], axis=2)

        if mask is not None and full_kv.shape[2] > mask.shape[-1]:
            pad = mx.ones(
                mask.shape[:-1] + (full_kv.shape[2] - mask.shape[-1],), dtype=mask.dtype
            )
            mask = mx.concatenate([mask, pad], axis=-1)

        out = scaled_dot_product_attention(
            q, full_kv, full_kv,
            cache=local_cache, scale=self.softmax_scale, mask=mask,
            sinks=self.attn_sink.astype(q.dtype),
        )
        out = _apply_partial_rope(out, self.rope, offset, inverse=True)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads * self.head_dim)
        out = self._grouped_output_projection(out)
        return self.wo_b(out)

    def _grouped_output_projection(self, out):
        """Match PR #1192 V4Attention._grouped_output_projection — handles
        both QuantizedLinear and plain paths for wo_a."""
        B, L = out.shape[:2]
        group_feat = (self.n_heads * self.head_dim) // self.o_groups
        out = out.reshape(B, L, self.o_groups, group_feat)

        if isinstance(self.wo_a, nn.QuantizedLinear):
            out = out.transpose(2, 0, 1, 3)
            weight = self.wo_a.weight.reshape(self.o_groups, self.o_lora_rank, -1)[:, None]
            scales = self.wo_a.scales.reshape(self.o_groups, self.o_lora_rank, -1)[:, None]
            biases = (
                None if self.wo_a.biases is None
                else self.wo_a.biases.reshape(self.o_groups, self.o_lora_rank, -1)[:, None]
            )
            out = mx.quantized_matmul(
                out, weight, scales=scales, biases=biases, transpose=True,
                group_size=self.wo_a.group_size, bits=self.wo_a.bits,
                mode=getattr(self.wo_a, "mode", "affine"),
            )
            out = out.transpose(1, 2, 0, 3).reshape(B, L, self.o_groups * self.o_lora_rank)
            if "bias" in self.wo_a:
                out = out + self.wo_a.bias
            return out

        weight = self.wo_a.weight.reshape(self.o_groups, self.o_lora_rank, group_feat)
        out = mx.einsum("bsgd,grd->bsgr", out, weight)
        out = out.reshape(B, L, self.o_groups * self.o_lora_rank)
        if "bias" in self.wo_a:
            out = out + self.wo_a.bias
        return out


# ---------- MoE ----------

@mx.compile
def sqrtsoftplus_select(
    gates: mx.array,
    bias: mx.array,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
):
    """DSV4 scoring: sqrt(softplus(gates)) + bias → top-k, then renorm.

    `gates` is expected to already be fp32 (caller must cast). Returns
    inds as int32 (required by mlx's gather_qmm path).
    """
    scores = mx.sqrt(mx.log1p(mx.exp(gates)))
    orig_scores = scores
    scores = scores + bias
    k = top_k
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k].astype(mx.int32)
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)
    scores = scores * routed_scaling_factor
    return inds, scores


class Gate(nn.Module):
    """DSV4 MoE gate. Supports both hash-routing (first N layers) and
    score-based (sqrtsoftplus + noaux_tc bias) modes."""
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.hash = layer_id < args.num_hash_layers
        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size))
        if self.hash:
            self.tid2eid = mx.zeros(
                (args.vocab_size, args.num_experts_per_tok), dtype=mx.int32,
            )
        else:
            self.bias = mx.zeros((args.n_routed_experts,))

    def __call__(self, x, input_ids=None):
        # Reference PR #1192: gate logits matmul in fp32 explicitly to avoid
        # bf16 accumulation error across 256 experts × hidden=4096.
        gates = x.astype(mx.float32) @ self.weight.T.astype(mx.float32)
        if self.hash:
            # Hash: deterministic per-token lookup (ignoring gates beyond
            # scoring for weights). Use original scores as weights.
            scores = mx.sqrt(mx.log1p(mx.exp(gates)))
            assert input_ids is not None, "hash-routed layer requires input_ids"
            inds = self.tid2eid[input_ids].astype(mx.int32)
            weights = mx.take_along_axis(scores, inds, axis=-1)
            if self.args.norm_topk_prob:
                weights = weights / mx.sum(weights, axis=-1, keepdims=True)
            weights = weights * self.args.routed_scaling_factor
            return inds, weights
        else:
            return sqrtsoftplus_select(
                gates, self.bias, self.args.num_experts_per_tok,
                self.args.routed_scaling_factor, self.args.norm_topk_prob,
            )


@mx.compile
def _dsv4_swiglu(gate, up, swiglu_limit: float):
    """DSV4 SwiGLU with gate/up clamping to ±swiglu_limit (gate is clamped
    to max only; up is clamped symmetrically). Without the clamp, deep MoE
    stacks diverge numerically.

    IMPORTANT: torch reference does `gate.float() * up.float()` — silu and
    multiply in fp32. We match that here to avoid per-layer precision drift.
    """
    out_dtype = gate.dtype
    gate = gate.astype(mx.float32)
    up = up.astype(mx.float32)
    if swiglu_limit > 0:
        up = mx.clip(up, a_min=-swiglu_limit, a_max=swiglu_limit)
        gate = mx.clip(gate, a_min=None, a_max=swiglu_limit)
    return (nn.silu(gate) * up).astype(out_dtype)


class _DSV4SwiGLU(nn.Module):
    def __init__(self, swiglu_limit: float):
        super().__init__()
        self.swiglu_limit = swiglu_limit

    def __call__(self, x_up, x_gate):
        return _dsv4_swiglu(x_gate, x_up, self.swiglu_limit)


class MLP(nn.Module):
    """SwiGLU expert / shared expert FFN. Uses mlx_lm naming convention."""
    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        d = args.hidden_size
        mi = intermediate_size if intermediate_size is not None else args.moe_intermediate_size
        self.swiglu_limit = getattr(args, "swiglu_limit", 10.0)
        self.gate_proj = nn.Linear(d, mi, bias=False)
        self.down_proj = nn.Linear(mi, d, bias=False)
        self.up_proj = nn.Linear(d, mi, bias=False)

    def __call__(self, x):
        # Match PR #1192 DeepseekV4MLP — no act_quant_sim wrapping.
        return self.down_proj(_dsv4_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.num_experts_per_tok = args.num_experts_per_tok
        self.gate = Gate(args, layer_id)
        swiglu_limit = getattr(args, "swiglu_limit", 10.0)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.n_routed_experts,
            activation=_DSV4SwiGLU(swiglu_limit),
        )
        self.shared_experts = MLP(args, intermediate_size=args.moe_intermediate_size)

    def __call__(self, x, input_ids=None):
        # Match PR #1192 DeepseekV4MoE forward exactly — no fp32 accumulation,
        # no act_quant_sim (that's for CUDA kernel fake-quant; MLX native
        # paths don't need it).
        inds, scores = self.gate(x, input_ids=input_ids)
        # Belt-and-suspenders int32 cast — mlx gather_qmm in QuantizedSwitchLinear
        # strictly requires int32; argpartition return dtype varies by mlx version.
        inds = inds.astype(mx.uint32)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype).reshape(x.shape)
        y = y + self.shared_experts(x)
        return y


# ---------- Block with mHC ----------

class DeepseekV4DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.self_attn = DeepseekV4Attention(args, layer_id=layer_id)
        self.mlp = MoE(args, layer_id)  # all DSV4 layers are MoE
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        mix_hc = (2 + args.hc_mult) * args.hc_mult
        hc_dim = args.hc_mult * args.hidden_size
        self.hc_attn_fn = mx.zeros((mix_hc, hc_dim))
        self.hc_ffn_fn = mx.zeros((mix_hc, hc_dim))
        self.hc_attn_base = mx.zeros((mix_hc,))
        self.hc_ffn_base = mx.zeros((mix_hc,))
        self.hc_attn_scale = mx.zeros((3,))
        self.hc_ffn_scale = mx.zeros((3,))

    def _hc_pre(self, x, fn, scale, base):
        # x: (B, L, hc_mult, D)
        shape = x.shape
        x_flat = mx.flatten(x, start_axis=2).astype(mx.float32)
        rsqrt = mx.rsqrt(mx.mean(x_flat.square(), axis=-1, keepdims=True) + self.args.rms_norm_eps)
        mixes = (x_flat @ fn.T) * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes, scale, base, self.args.hc_mult,
            self.args.hc_sinkhorn_iters, self.args.hc_eps,
        )
        y = mx.sum(pre[..., None] * mx.reshape(x_flat, shape), axis=2)
        return y.astype(x.dtype), post, comb

    def _hc_post(self, x, residual, post, comb):
        # x: (B, L, D); residual: (B, L, hc_mult, D); return (B, L, hc_mult, D)
        # Reference: y[b,s,i,d] = post[b,s,i] * x[b,s,d]
        #                       + sum_j comb[b,s,i,j] * residual[b,s,j,d]
        # Contracts comb's LAST axis with residual's hc axis → equivalent to
        # `comb @ residual`. mlx_lm PR #1192 latest (commit ef8c95d6, 2026-04-24)
        # uses `mx.matmul(comb, residual)` directly — mlx matmul is faster than
        # einsum for this batched contraction because einsum adds string-parsing
        # + intermediate graph overhead.
        y = post[..., None] * x[..., None, :].astype(mx.float32) + mx.matmul(
            comb.astype(mx.float32), residual.astype(mx.float32)
        )
        return y.astype(x.dtype)

    def __call__(self, x, mask=None, cache=None, input_ids=None):
        residual = x
        x, post, comb = self._hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask=mask, cache=cache)
        x = self._hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self._hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.post_attention_layernorm(x)
        x = self.mlp(x, input_ids=input_ids)
        x = self._hc_post(x, residual, post, comb)
        return x


# ---------- Top-level model ----------

class DeepseekV4Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DeepseekV4DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        mix_hc = (2 + args.hc_mult) * args.hc_mult
        self.hc_head_fn = mx.zeros((args.hc_mult, args.hc_mult * args.hidden_size))
        self.hc_head_base = mx.zeros((args.hc_mult,))
        self.hc_head_scale = mx.zeros((1,))

    def _hc_head_reduce(self, x):
        # x: (B, L, hc_mult, D) → (B, L, D) via sigmoid-weighted sum
        shape = x.shape
        x_flat = mx.flatten(x, start_axis=2).astype(mx.float32)
        rsqrt = mx.rsqrt(mx.mean(x_flat.square(), axis=-1, keepdims=True) + self.args.rms_norm_eps)
        mixes = (x_flat @ self.hc_head_fn.T) * rsqrt
        pre = mx.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.args.hc_eps
        y = mx.sum(pre[..., None] * mx.reshape(x_flat, shape), axis=2)
        return y.astype(x.dtype)

    def __call__(self, input_ids, cache=None, mask=None):
        h = self.embed(input_ids)
        # Expand to hc_mult copies for mHC. Must be materialized (not a broadcast
        # view) — matches torch reference `h.unsqueeze(2).repeat(1, 1, hc_mult, 1)`.
        # Subsequent `flatten(start_axis=2)` inside `_hc_pre` would see wrong
        # strided data from a broadcast view.
        h = mx.tile(h[..., None, :], (1, 1, self.args.hc_mult, 1))
        if cache is None:
            cache = [None] * len(self.layers)
        if mask is None:
            # Match PR #1192 reference: pass an explicit mask array (not
            # "causal" string), with sliding-window semantics. Native SDPA
            # needs an array mask for the `sinks` code path to work.
            first_cache = cache[0]
            mask = create_attention_mask(
                h[:, :, 0, :], first_cache,
                window_size=self.args.sliding_window,
                return_array=True,
            )
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask=mask, cache=c, input_ids=input_ids)
        h = self._hc_head_reduce(h)
        return self.norm(h)


class Model(nn.Module):
    """mlx_lm entry-point class — what load_jangtq_model / mlx-lm factory expects."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = DeepseekV4Model(args)
        # Tied weight option not confirmed for DSV4 — use separate lm_head
        # (config has tie_word_embeddings=false)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, input_ids, cache=None, mask=None):
        h = self.model(input_ids, cache=cache, mask=mask)
        # CRITICAL: reference does lm_head matmul in FP32
        # (inference/model.py ParallelHead.get_logits: `F.linear(x[:, -1].float(), self.weight)`
        # with self.weight stored as fp32). Accumulating 4096-dim contraction
        # in bf16 can add ~0.5 error per logit — comparable to the margin
        # between correct vs incorrect arithmetic answers.
        w = self.lm_head.weight
        if hasattr(self.lm_head, "scales"):
            # Quantized lm_head — dequantize then fp32 matmul
            w_f = mx.dequantize(
                self.lm_head.weight, self.lm_head.scales,
                getattr(self.lm_head, "biases", None),
                group_size=self.lm_head.group_size,
                bits=self.lm_head.bits,
                mode=getattr(self.lm_head, "mode", "affine"),
            ).astype(mx.float32)
        else:
            w_f = w.astype(mx.float32)
        h_f = h.astype(mx.float32)
        return h_f @ w_f.T

    def make_cache(self):
        """Build per-layer cache objects.
        SHORT-PROMPT-SAFE default: use plain KVCache for all layers. Compressor
        + Indexer fast-path is taken in DeepseekV4Attention (cache is None for
        v4-state, so pooled is empty and skipped). This makes prompts up to
        sliding_window=128 tokens behave identically to the pre-make_cache path.
        For >128 tokens, attention falls back to local-only sliding-window context
        (still coherent, but loses pooled-global benefit). To enable full
        long-context behavior with Compressor + Indexer, set the env var
        DSV4_LONG_CTX=1 — then compress_ratio>0 layers get DeepseekV4Cache.
        """
        from mlx_lm.models.cache import KVCache, RotatingKVCache
        import os
        long_ctx = os.environ.get("DSV4_LONG_CTX", "0") == "1"
        caches = []
        for layer in self.model.layers:
            if long_ctx and layer.self_attn.compress_ratio:
                caches.append(DeepseekV4Cache(self.args.sliding_window))
            else:
                caches.append(KVCache())
        return caches

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        """Map DSV4 source keys → mlx_lm conventions + stack experts.

        DSV4 ckpt conventions:
          embed.weight                               → model.embed.weight
          head.weight                                → lm_head.weight
          norm.weight                                → model.norm.weight
          layers.N.attn.{wq_a/wq_b/wkv/kv_norm/q_norm/wo_a/wo_b}.weight
                                                     → model.layers.N.self_attn.{...}.weight
          layers.N.attn_norm.weight                  → model.layers.N.input_layernorm.weight
          layers.N.ffn_norm.weight                   → model.layers.N.post_attention_layernorm.weight
          layers.N.ffn.gate.{weight|bias|tid2eid}    → model.layers.N.mlp.gate.{...}
          layers.N.ffn.shared_experts.{w1|w2|w3}.*   → model.layers.N.mlp.shared_experts.{gate/down/up}_proj.*
          layers.N.ffn.experts.E.{w1|w2|w3}.*        → STACK into model.layers.N.mlp.switch_mlp.{gate/down/up}_proj.*
          layers.N.attn.attn_sink                    → model.layers.N.self_attn.attn_sink
          layers.N.hc_{attn/ffn}_{fn/base/scale}     → model.layers.N.hc_{...}
          layers.N.attn.compressor.*                 → model.layers.N.self_attn.compressor.* (unused Phase 7.5B.2)
          layers.N.attn.indexer.*                    → model.layers.N.self_attn.indexer.*   (unused Phase 7.5B.2)
          mtp.0.*                                    → dropped (MTP not run at inference)
          hc_head_{fn/base/scale}                    → model.hc_head_{...}

        W1→gate_proj, W2→down_proj, W3→up_proj (per DSV convention).
        """
        import mlx.core as mx
        import re

        w1w2w3 = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}

        out = {}
        for k, v in weights.items():
            # Drop MTP at inference
            if k.startswith("mtp."):
                continue
            # Keep compressor/indexer weights — needed for DSV4-Flash layers
            # with compress_ratio > 0 (most layers) to produce correct attention
            # over compressed global context. Without them, residual stream
            # explodes over 43 layers.
            # Global
            if k == "embed.weight":
                out["model.embed.weight"] = v; continue
            if k == "head.weight" or k == "head.biases" or k == "head.scales":
                # Map quantized head's .weight/.scales/.biases
                out["lm_head." + k[len("head."):]] = v; continue
            if k == "norm.weight":
                out["model.norm.weight"] = v; continue
            if k in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                out["model." + k] = v; continue

            m = re.match(r"layers\.(\d+)\.(.+)", k)
            if not m:
                out["model." + k] = v  # pass-through
                continue
            L, rest = m.group(1), m.group(2)
            pfx = f"model.layers.{L}"

            # Norms
            if rest == "attn_norm.weight":
                out[f"{pfx}.input_layernorm.weight"] = v; continue
            if rest == "ffn_norm.weight":
                out[f"{pfx}.post_attention_layernorm.weight"] = v; continue

            # mHC
            if rest.startswith("hc_"):
                out[f"{pfx}.{rest}"] = v; continue

            # Attention (including compressor.* and indexer.* sub-modules)
            if rest.startswith("attn."):
                inner = rest[len("attn."):]
                out[f"{pfx}.self_attn.{inner}"] = v; continue

            # FFN
            if rest.startswith("ffn."):
                inner = rest[len("ffn."):]
                # Gate
                if inner.startswith("gate."):
                    out[f"{pfx}.mlp.gate.{inner[len('gate.'):]}"] = v; continue
                # Shared experts
                m2 = re.match(r"shared_experts\.(w[123])\.(weight|scales|biases)$", inner)
                if m2:
                    proj = w1w2w3[m2.group(1)]
                    out[f"{pfx}.mlp.shared_experts.{proj}.{m2.group(2)}"] = v; continue
                # Routed experts — collect for stacking
                m3 = re.match(r"experts\.(\d+)\.(w[123])\.(weight|scales|biases)$", inner)
                if m3:
                    # Temporary marker — will be stacked below
                    out[f"__TEMP__{pfx}.mlp.experts.{m3.group(1)}.{w1w2w3[m3.group(2)]}.{m3.group(3)}"] = v
                    continue
                # Fallback
                out[f"{pfx}.mlp.{inner}"] = v; continue

            out[f"{pfx}.{rest}"] = v

        # Stack routed experts across all layers
        n_experts = self.args.n_routed_experts
        for L in range(self.args.num_hidden_layers):
            pfx = f"model.layers.{L}.mlp"
            for proj in ("gate_proj", "down_proj", "up_proj"):
                for kind in ("weight", "scales", "biases"):
                    keys_e = [f"__TEMP__{pfx}.experts.{e}.{proj}.{kind}" for e in range(n_experts)]
                    if keys_e[0] in out:
                        stacked = mx.stack([out.pop(k) for k in keys_e])
                        out[f"{pfx}.switch_mlp.{proj}.{kind}"] = stacked

        # Final guard: no __TEMP__ keys should remain
        leftovers = [k for k in out if k.startswith("__TEMP__")]
        if leftovers:
            raise RuntimeError(f"sanitize left {len(leftovers)} unstacked TEMP keys, "
                               f"e.g. {leftovers[0]}")
        return out

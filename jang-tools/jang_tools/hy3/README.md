# Hy3 (Tencent Hy3-preview) — JANG runtime

`model_type=hy_v3` — Tencent Hy3-preview, 295B/21B-active text-only MoE.
Architecture: 80 dense GQA layers + 1 MTP draft layer, 192 routed
experts top-8 + 1 shared expert, sigmoid router with `expert_bias`
(DSV3-style aux-free balancing), 256K context, qk_norm RMSNorm pre-RoPE,
fp32 lm_head accumulation.

The architecture is identical to mlx_lm's `dots1` family with field-name
and tensor-key differences; this package wraps `dots1` with the
remapping logic.

## Convert

```
python -m jang_tools.convert_hy3_jangtq <bf16_src> <out> JANGTQ2
```

Bit policy (JANGTQ2):

- routed expert MLPs (192 experts × 79 sparse layers): MXTQ 2-bit
- attention q/k/v/o, shared expert, dense layer-0 MLP, embed/lm_head,
  MTP matmuls: affine 8-bit, group_size 64
- norms, router gate, expert_bias: fp16 passthrough

## Load (Python)

```python
from jang_tools.load_jangtq import load_jangtq_model

model, tokenizer = load_jangtq_model("/path/to/Hy3-preview-JANGTQ2")
```

`load_jangtq_model` auto-registers `model_type=hy_v3` before it builds the
MLX skeleton. Direct `register_mlx_lm_hy3()` calls are still fine in custom
scripts, but they are no longer required for normal JANGTQ loading.

For BF16 / affine-only Hy3 bundles use `jang_tools.hy3.load_hy3_model`
instead. JANGTQ bundles MUST go through `load_jangtq_model` so routed
experts hydrate through the TurboQuant kernel path.

## Capabilities

`build_capabilities` produces:

```
family=hy_v3
reasoning_parser=qwen3
tool_parser=hunyuan
think_in_template=True
supports_thinking=True
cache_type=kv
modality=text
```

## Architecture facts

```
hidden_size            4096
num_hidden_layers      80 (+1 MTP)
num_attention_heads    64
num_key_value_heads    8 (GQA, group=8)
head_dim               128
qk_norm                True (RMSNorm Q/K pre-RoPE)
rope_theta             11158840.0 (rope_type=default)
max_position_embeddings 262144 (256K)
num_experts            192
num_experts_per_tok    8
num_shared_experts     1
moe_intermediate_size  1536
intermediate_size      13312 (dense layer 0)
first_k_dense_replace  1
moe_router_use_sigmoid True
moe_router_enable_expert_bias True
route_norm             True
router_scaling_factor  2.826
num_nextn_predict_layers 1 (MTP)
enable_lm_head_fp32    True
vocab_size             120832
tie_word_embeddings    False
```

## Tensor namespace differences vs `dots1`

| Bundle (`hy_v3`) | `dots1` |
|---|---|
| `mlp.router.gate.weight` | `mlp.gate.weight` |
| `mlp.expert_bias` | `mlp.gate.e_score_correction_bias` |
| `mlp.shared_mlp.{gate,up,down}_proj.*` | `mlp.shared_experts.{gate,up,down}_proj.*` |
| `mlp.experts.{e}.{gate,up,down}_proj.*` (per-expert) | `mlp.experts.{gate,up,down}_proj.*` (stacked) |
| `model.layers.80.*` (MTP) | (preserved_disabled — dropped at sanitize) |
| `rope_parameters.rope_theta` | `rope_theta` (flattened) |
| `num_experts` / `num_shared_experts` / `route_norm` / `router_scaling_factor` | `n_routed_experts` / `n_shared_experts` / `norm_topk_prob` / `routed_scaling_factor` |

`Model.sanitize` performs all the renames + per-expert → SwitchGLU
stacking + drops the MTP layer.

## Critical runtime fixes

This package corrects two bugs that produce silent garbage output:

1. **fp32 lm_head**. `enable_lm_head_fp32=True` in the bundle config; the
   bf16 4096-dim contraction otherwise drifts logits enough to flip
   sensible top-k token picks toward high-baseline-energy junk tokens
   (doubled letters, `OO`/`HH`/`CCC`). `Model.__call__` dequantizes
   `lm_head.weight` and accumulates in fp32. Mirrors DSV4's pattern in
   `mlx_model.py`.

2. **qk_norm under JANGTQ P18 QKV fusion**. JANGTQ's QKV-fusion patch
   replaces `Dots1Attention.__call__` and applies `q_norm`/`k_norm` only
   when the attention class declares `use_qk_norm=True`. Without that
   flag, the norms are silently skipped, attention degenerates, and the
   model produces incoherent output. Even with the flag, `nn.RMSNorm`
   over flat `[B, L, n_heads * head_dim]` would normalize only the last
   `head_dim` of `n_heads * head_dim` elements.

   `Hy3Attention` declares `use_qk_norm=True` and uses `Hy3HeadRMSNorm`,
   which auto-reshapes flat input to `[..., n_heads, head_dim]`,
   normalizes per head, and reshapes back. Pre-reshaped input passes
   through directly.

## MTP status

Bundle ships MTP layer (`model.layers.80.*`) with
`mtp_mode=preserved_disabled`. The runtime drops the MTP layer at
sanitize and decodes one token per forward pass. Speculative decoding
requires a separate accept/reject loop that has not been implemented in
JANG / vmlx / vmlx-swift-lm yet.

## Validated

- 80 layers materialize, JANGTQ-stacked routed experts hydrate via
  TurboQuantLinear (79 instances).
- `decode("What is 2 + 2? Answer briefly.")` → `4<｜hy_eos｜>` at
  ~15 tok/s on M5 Max 128 GB with `reasoning_effort=no_think`.
- Sanity completions:
  - "The capital of France is" → top1 ` Paris` (logit 19.13)
  - "def fibonacci(n):" → top1 `\n`, top3 includes ` return`
  - "Once upon a time" → top1 `,`, top2 ` there`

Validation boundary: these are local smoke results for short prompts, not
benchmark claims. MTP speculative decode remains disabled.

## vmlx integration

`vmlx_engine.loaders.load_jangtq_hy3` should re-export
`load_hy3_model` from this module — mirrors the `load_jangtq_dsv4` /
`load_jangtq_kimi_vlm` pattern. The vmlx wrapper does not need its own
model code; this package is the source of truth.

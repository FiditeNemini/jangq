# ZAYA1-8B Attention Architecture Notes

These notes are for JANG/JANGTQ/MXFP4 conversion and runtime integration of
`Zyphra/ZAYA1-8B`. They are based on the local source snapshot at:

```text
/Users/eric/jang/models/Zyphra/ZAYA1-8B
```

The snapshot config reports `model_type=zaya` and
`architectures=["ZayaForCausalLM"]`. The local download was pinned during prep
to Hugging Face commit `2b008c91b7f0004636394dbd2d7b4ca2c2e820e7`.

## Layer Schedule

ZAYA is an alternating hybrid decoder:

```text
layers 0,2,4,...,78:  CCA attention layers
layers 1,3,5,...,79:  top-1 ZAYA MoE layers
```

There are 80 total decoder layers: 40 attention layers and 40 MoE layers.

Relevant config values:

```text
hidden_size = 2048
num_attention_heads = 16
num_query_groups = 2
num_key_value_heads = 2
cca = true
cca_num_q_heads = 8
kv_channels = 128
partial_rotary_factor = 0.5
rope_theta = 5000000
rope_scaling = false
sliding_window = null
max_position_embeddings = 131072
attention_bias = false
attention_dropout = 0.0
mamba_cache_dtype = float32
```

`head_dim = hidden_size / num_attention_heads = 128`.

## CCA Attention Weights

Layer 0 has the representative attention tensor layout:

```text
model.layers.0.self_attn.qkv.linear_q.weight       BF16 [1024, 2048]
model.layers.0.self_attn.qkv.linear_k.weight       BF16 [256, 2048]
model.layers.0.self_attn.qkv.val_proj1.weight      BF16 [128, 2048]
model.layers.0.self_attn.qkv.val_proj2.weight      BF16 [128, 2048]
model.layers.0.self_attn.qkv.conv_qk.0.weight      BF16 [1280, 1, 2]
model.layers.0.self_attn.qkv.conv_qk.0.bias        BF16 [1280]
model.layers.0.self_attn.qkv.conv_qk.1.weight      BF16 [1280, 128, 2]
model.layers.0.self_attn.qkv.conv_qk.1.bias        BF16 [1280]
model.layers.0.self_attn.qkv.temp                  BF16 [2]
model.layers.0.self_attn.o_proj.weight             BF16 [2048, 1024]
```

The CCA query width is `cca_num_q_heads * head_dim = 8 * 128 = 1024`.
The standard grouped-KV width is `num_query_groups * head_dim = 2 * 128 = 256`.
The CCA convolution channel count is:

```text
(cca_num_q_heads + num_query_groups) * head_dim = (8 + 2) * 128 = 1280
```

Runtime implementations should treat `conv_qk.*` and `temp` as part of the CCA
attention mechanism, not as normal dense projection weights.

## Cache Contract

Each attention layer has two independent state families:

```text
standard KV cache:   K/V [batch, 2, tokens, 128]
CCA convolution:     conv_state [batch, 1280, 2]
CCA previous hidden: prev_hs [batch, 2048]
```

The standard KV cache is pageable. The CCA state is not interchangeable with
standard KV pages; it must move with the sequence slot and must be restored with
the same prompt boundary.

## Batching, Prefix, And Paged KV

Continuous batching is compatible when every live sequence owns all of:

```text
KV pages
conv_state
prev_hs
position offset / RoPE offset
```

Paged KV can be used for the standard K/V tensors. Do not mark a paged-prefix
restore as complete unless the corresponding CCA `conv_state` and `prev_hs` are
also restored for the exact same prefix length.

Prefix caching should stay disabled for the first JANG/vMLX port. The official
vLLM launch path for ZAYA treats it as a hybrid model and disables prefix
caching. A correct prefix cache key for this architecture has to include the
usual token/template/model identity plus the CCA state payload, not just KV page
identity.

Chunked prefill has the same constraint: chunk boundaries must carry exact CCA
state. Do not split a prefill into chunks until an exact-match test proves that
single-shot prefill and chunked prefill produce the same logits.

## TurboQuant KV / Cache Encoding

TurboQuant cache encoding, if enabled, should only target the standard K/V
cache first:

```text
encode:    K/V pages for attention layers
do not encode: conv_state, prev_hs, RoPE offsets, router state
```

Keep CCA runtime state in float32 for the initial implementation. The source
config requests `mamba_cache_dtype=float32`; even though this is CCA rather than
a stock Mamba block, it is the strongest local signal for the hybrid state
precision floor.

## MoE Interaction

Odd layers are top-1 ZAYA MoE layers:

```text
num_experts = 16
moe_router_topk = 1
activation_func = swiglu
zaya_use_eda = true
zaya_use_mod = true
```

Each source expert stores:

```text
linear_fc1.weight BF16 [4096, 2048]
linear_fc2.weight BF16 [2048, 2048]
```

For JANGTQ and MXFP4 bundles, `linear_fc1` is split into SwiGLU halves:

```text
linear_fc1[:2048, :] -> gate_proj
linear_fc1[2048:, :] -> up_proj
linear_fc2           -> down_proj
```

The converted runtime-facing expert namespace is:

```text
model.layers.{L}.zaya_block.experts.switch_mlp.gate_proj
model.layers.{L}.zaya_block.experts.switch_mlp.up_proj
model.layers.{L}.zaya_block.experts.switch_mlp.down_proj
```

New JANGTQ bundles must emit these expert tensors pre-stacked across expert
axis 0, not as per-expert `.tq_*` keys.

## Quantization Policy

For `JANGTQ2` and `JANGTQ4`:

```text
routed experts: MXTQ 2-bit or 4-bit, pre-stacked switch_mlp
attention linears: 8-bit affine, group_size 32
embed/lm_head: 8-bit affine, group_size 32
router path: passthrough
conv_qk/temp: passthrough
norms/residual scaling/biases/balancing_biases: passthrough
```

Every JANGTQ bundle must include:

```text
config.json weight_format = "mxtq"
config.json mxtq_bits as a per-role dictionary
jang_config.json tq_in_features for exact TQ logical widths
jangtq_runtime.safetensors for Swift/runtime sidecar loading
```

For `MXFP4`:

```text
large 2-D linears: 4-bit affine, group_size 32
pre-stacked expert groups: 4-bit affine, group_size 32
embed/lm_head: 8-bit affine first pass
same passthrough floor as JANGTQ for router, CCA conv, temp, norms, and residual state
```

This keeps the first converted bundles conservative around the hybrid attention
state and router path while still making the expert bulk compact.

## Integration Checklist

Before a runtime claims ZAYA compatibility:

```text
1. Load real config metadata rather than matching only on model name.
2. Allocate one hybrid cache object per attention layer, not a KV-only cache.
3. Keep CCA conv_state and prev_hs attached to the sequence slot during batching.
4. Leave prefix caching disabled until KV+CCA restore is tested.
5. Page only standard K/V tensors unless the CCA state pager is implemented.
6. Treat JANGTQ expert tensors as pre-stacked switch_mlp groups.
7. Require mxtq_bits dictionary decoding and jangtq_runtime.safetensors.
8. Preserve tokenizer/chat template files from the source snapshot.
```

## vMLX Typed Restore Handoff

The initial vMLX Python runtime represents each CCA attention layer as:

```text
CacheList(KVCache(), ArraysCache(2))
ArraysCache[0] = convolution tail state, runtime layout [B, total_padding, packed_qk_dim]
ArraysCache[1] = previous hidden state, runtime layout [B, 1, hidden_size]
```

Odd MoE layers have no recurrent state and should use an explicit no-state
placeholder so cache lists stay aligned to the 80-layer model schedule. A
restore path that reconstructs only the 40 CCA attention layers is incomplete.

For prefix/paged/L2 enablement, do not rely on a generic `CacheList` or generic
hybrid-SSM serializer. The typed restore record should preserve:

```text
layer index
layer role: cca_attention | zaya_moe_no_state
standard KV state for CCA layers
CCA convolution tail state
CCA previous hidden state
prompt length / N-1 refeed convention
model + tokenizer/template + cache schema hash
```

Runtime TurboQuant KV, if later enabled for ZAYA, must be a typed partial codec:
only the standard KV sub-cache may be encoded. The CCA convolution state,
previous hidden state, router path, RoPE offsets, and no-state placeholders must
remain native until separate numeric parity tests prove otherwise.

Current vMLX Python handoff state:

```text
typed record: zaya_cca_v1
CCA record payload: standard KV pages + terminal conv_state + terminal prev_hs
MoE payload: explicit ZayaNoStateCache/no_state slot
generic TurboQuant KV: disabled for ZAYA CCA
native status fields: /health, /v1/cache/stats, /v1/models/{id}/capabilities
```

The source contract now includes prompt-boundary restore, block-disk
round-trip, and fresh L2 disk-hit continuation-logit checks on the small ZAYA
runtime. A direct full-bundle ZAYA MXFP4 probe also passed typed in-memory
prefix restore and fresh L2 disk restore with continuation-logit `maxdiff=0.0`.
Full production sign-off still requires server/API ZAYA JANGTQ4/MXFP4
multi-turn cache-hit, L2-restart, and tool-row tests; JANGTQ2 remains a
separate quality investigation.

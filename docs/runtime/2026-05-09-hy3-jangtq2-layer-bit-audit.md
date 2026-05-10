# Hy3 JANGTQ2 Layer And Bit Audit

Created 2026-05-09.

Target bundle: `Hy3-preview-JANGTQ2`.

## Profile

`JANGTQ2` is the first 128 GB release candidate.

| Tensor family | Expected full-source count | Storage policy | Reason |
|---|---:|---|---|
| Routed expert `model.layers.1..79.mlp.experts.*.{gate,up,down}_proj.weight` | 45504 | MXTQ 2-bit | Main memory reduction. Hy3 has 79 sparse MoE layers, 192 experts per layer, 3 projections per expert. |
| Shared expert `model.layers.1..79.mlp.shared_mlp.{gate,up,down}_proj.weight` | 237 | 8-bit affine, group 64 | Always active expert path; keep higher precision first. |
| Dense layer-0 MLP `model.layers.0.mlp.{gate,up,down}_proj.weight` | 3 | 8-bit affine, group 64 | First layer is dense by `first_k_dense_replace=1`; do not treat as routed MoE. |
| Attention `q/k/v/o_proj.weight` | 320 | 8-bit affine, group 64 | Dense GQA attention path; 64 Q heads, 8 KV heads. |
| Q/K norms `self_attn.{q_norm,k_norm}.weight` | 160 | passthrough fp16 | Precision-sensitive pre-attention normalization. |
| Layer norms `{input,post_attention}_layernorm.weight` | 160 | passthrough fp16 | Normalization tensors stay unquantized. |
| Router gate `mlp.router.gate.weight` | 79 | passthrough fp16 | Sigmoid top-k routing is precision-sensitive. |
| Expert correction bias `mlp.expert_bias` | 79 | passthrough fp16 | Top-k correction bias must match upstream routing. |
| `embed_tokens.weight`, `lm_head.weight` | 2 | 8-bit affine, group 64 | First pass keeps output/input projections higher than routed experts. |
| MTP tensors | unknown until final index | 8-bit affine for matmuls, passthrough for norms/router/bias | Bundle preserves MTP for future speculative decode, but runtime mode is `preserved_disabled`. |

## Runtime Rules

Hy3 runtime must implement:

- dense causal GQA KV cache, not MLA/SSM/CCA/VL
- Q/K RMSNorm before RoPE
- sigmoid router scoring
- expert-bias correction for expert choice
- selected sigmoid weights normalized by sum and multiplied by `router_scaling_factor=2.826`
- always-active shared expert output added to routed expert output
- dense layer 0 branch before sparse MoE layers
- MTP state separate from accepted base KV when MTP is later enabled

## Current Low-RAM Evidence

Partial source state during audit:

```text
dry-run profile: JANGTQ2
tensors seen so far: 32360
MXTQ: 31979
affine: 304
passthrough: 77
```

Role census from the same partial-download window:

```text
attention_affine_8: 216
embed_lm_head_affine_8: 1
expert_bias_passthrough_16: 13
norm_or_bias_passthrough_16: 52
routed_expert_mxtq_2: 33683
router_gate_passthrough_16: 12
shared_expert_affine_8: 87
default_affine_8_REVIEW: 0
```

The source download is still incomplete, so these are not final counts. The final gate must use `model.safetensors.index.json`.

Later full-source pre-flight observed:

```text
index tensors: 47138
source shards: 112
missing shards: 0
full policy aggregate: 45504 routed MXTQ, 1146 affine, 488 passthrough
```

Do not reuse partial conversion output from the interrupted 2026-05-09 run. The partial `/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2` directory was deleted after duplicate conversion coordination risk.

Current 128 GB estimate:

```text
estimated bundle: 88.5 GB
4K KV cache: 1.34 GB
estimated runtime total with 12 GB headroom: 101.8 GB
verdict: tight, but realistic for 128 GB
```

## Final Gates

Before conversion:

- `model.safetensors.index.json` exists.
- Every shard referenced by the index exists.
- Full tensor census confirms MTP namespace.
- No unexpected tensor role falls into the default 8-bit rule without review.

After conversion:

- `config.json` and `jang_config.json` stamp `profile=JANGTQ2`.
- `mxtq_bits.routed_expert=2`.
- `runtime.mtp_mode=preserved_disabled`.
- `jangtq_runtime.safetensors` exists.
- `verify_directory` passes.
- Output index has no missing shard references.
- README/model card states 128 GB status as measured, not assumed.

# Hy3 Runtime Handoff For vmlx Python And vmlx-swift-lm

Created 2026-05-09.

Scope: implementation notes and copyable examples for future agents working in:

- Python engine: `../vmlx`
- Swift engine: `../vmlx-swift-lm`
- JANG/JANGTQ bundle tooling: `/Users/eric/jang`

Do not edit `../vmlx` or `../vmlx-swift-lm` from this handoff. Copy the relevant patterns into those repos in a dedicated runtime pass.

## Current Conversion State

The active conversion is `Hy3-preview-JANGTQ2`, started after the earlier duplicate-writer cleanup. Current rule: only one writer may target `/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2`.

Expected first bundle policy:

| Tensor family | Policy |
|---|---|
| Routed experts | MXTQ 2-bit |
| Attention q/k/v/o | affine 8-bit |
| Shared expert | affine 8-bit |
| Dense layer-0 MLP | affine 8-bit |
| Embeddings/lm_head | affine 8-bit |
| MTP matmuls | affine 8-bit |
| Norms/router/expert_bias | fp16 passthrough |

## Family Comparison

| Family | Attention/cache | MoE/router | MTP/spec decode | Parser/tool notes | Runtime lesson for Hy3 |
|---|---|---|---|---|---|
| Hy3 | Dense causal GQA KV, 64 Q heads, 8 KV heads, q/k RMSNorm, default RoPE theta 11158840, 262K context | 192 experts, top-8, sigmoid router, expert bias, route norm, one shared expert, dense layer 0 | `num_nextn_predict_layers=1`; first bundle mode `preserved_disabled` | Reasoning effort `no_think|low|high`; `<think>` tags; Hunyuan/Tencent tool tags | Needs new `hy_v3` model path, not a Qwen/MiniMax alias. |
| MiniMax M2.7 | Dense causal GQA KV | MoE with sigmoid-style routing; JANGTQ_K precedent | Local JANGTQ configs show no active MTP | `minimax` tool parser, qwen3-style reasoning | Best existing MoE/JANGTQ runtime analog, but not an MTP test target locally. |
| DeepSeek-V4 Flash | MLA/hybrid cache, compressor/indexer-like spec path | DeepSeek-style MoE | MTP-adjacent compressor/indexer, not classical Hy3 MTP | DeepSeek reasoning/tool parsers | Do not copy its cache topology into Hy3; only reuse the explicit runtime-status discipline. |
| Qwen3.5/3.6 | Hybrid family in local notes; Qwen parsers | MoE variants exist | No MTP in inspected local configs | qwen3 reasoning, qwen tools | Useful parser/regression baseline, not Hy3 attention topology. |
| Gemma 4 | KV cache, Gemma parser surface | MoE variants | No MTP in inspected local config | Gemma parser/tool quirks | Non-MTP control for parser/cache regression. |
| ZAYA/ZAYA1-VL | CCA/hybrid; VL path has Qwen2.5-VL ViT | Top-1 MoE/MOD details | No MTP | Zaya XML tools, non-thinking gate | Do not reuse ZAYA cache or VL plumbing for Hy3; Hy3 is text-only KV. |

## Hy3 Attention Architecture

Hy3 uses standard dense causal grouped-query attention:

```text
hidden_size = 4096
num_attention_heads = 64
num_key_value_heads = 8
head_dim = 128
num_key_value_groups = 8
qk_norm = true
rope_parameters = { rope_type: default, rope_theta: 11158840.0 }
max_position_embeddings = 262144
```

Implementation requirements:

- project Q, K, V, O normally
- reshape Q to `[batch, tokens, 64, 128]`
- reshape K/V to `[batch, tokens, 8, 128]`
- apply RMSNorm to Q and K per head before RoPE
- apply default RoPE using `rope_theta=11158840`
- repeat/broadcast KV heads across 8 query groups
- use standard causal SDPA mask
- append accepted K/V to normal KV cache

Not present:

- no MLA absorb/compressed KV path
- no sliding-window cache
- no SSM/Mamba/ArraysCache
- no CCA cache
- no VLM/media salt
- no image tokens or processor

## Hy3 MoE Runtime

Sparse layers are layers `1..79`; layer `0` is dense FFN by `first_k_dense_replace=1`.

Sparse layer rule:

1. compute router logits
2. apply sigmoid
3. add expert correction bias for top-k choice
4. select top 8 experts
5. gather original sigmoid weights for selected experts
6. normalize selected weights by their sum when `route_norm=true`
7. multiply by `router_scaling_factor=2.826`
8. run selected expert MLPs
9. add always-active shared expert output

Precision requirements:

- router gate: fp16 passthrough in bundle, compute preferably fp32 or numerically stable fp16/fp32 boundary
- expert bias: fp16 passthrough
- q/k norms and RMSNorms: fp16 passthrough
- routed expert weights: JANGTQ2 2-bit MXTQ via TurboQuant kernels
- shared/dense/attention/MTP matmuls: affine 8-bit first bundle

## Reasoning And Tool Surface

Hy3 chat template facts:

- special role tokens: `<｜hy_User｜>`, `<｜hy_Assistant｜>`, `<｜hy_eos｜>`
- reasoning tags: `<think>...</think>`
- reasoning effort token: `<｜reasoning_mode｜>reasoning_effort:{no_think|low|high}`
- default if unspecified: `no_think`
- tool calls:

```text
<tool_calls>
<tool_call>{function-name}<tool_sep>
<arg_key>{key}</arg_key>
<arg_value>{value}</arg_value>
...
</tool_call>
</tool_calls>
```

Parser requirements:

- support `reasoning_effort` request metadata and template vars
- strip or route `<think>...</think>` into reasoning content
- parse Hunyuan/Tencent tool-call tags, not Hermes/Qwen JSON-only assumptions
- ensure `no_think` emits closed `<think></think>` prefill and does not leak reasoning into content
- test tool calls both with and without reasoning effort

## MTP Runtime Contract

Current bundle status:

```text
bundle_has_mtp = true
mtp_layers = 1
mtp_mode = preserved_disabled
```

`preserved_disabled` means the weights are shipped but runtime uses normal autoregressive decode. Do not call this MTP enabled.

Future enabled mode requires:

- separate draft state from accepted base KV
- accept/reject verification before committing a drafted token
- cancellation and rejection discard draft state
- batch slot-local draft state if used with BatchEngine
- cache key salt includes `mtp_mode`, model revision, quant profile, parser mode, and chat-template salt

## Python Engine Checklist

1. Add `hy_v3` model-family detection.
2. Reuse standard KV cache, not hybrid cache patches.
3. Add Hy3 config parser for `rope_parameters`, q/k norm, expert bias, route norm, shared expert, dense layer 0, MTP fields.
4. Add Hunyuan/Tencent tool parser.
5. Add reasoning effort plumbing into chat-template rendering.
6. Wire JANGTQ routed expert weights to existing TurboQuant load path.
7. Wire affine 8-bit non-routed weights to existing affine load path.
8. First runtime proof: normal decode with MTP disabled.
9. Second proof: two-turn cache continuation.
10. Later: MTP accept/reject reference path.

## Swift Engine Checklist

1. Add `hy_v3` dispatch in model factory.
2. Add `Hy3Configuration` decoding.
3. Add `Hy3Attention` with q/k RMSNorm and GQA KV dimensions.
4. Add `Hy3MoE` with sigmoid + expert-bias top-k + shared expert.
5. Add dense layer-0 branch.
6. Add JANGTQ2 routed expert decode using TurboQuant kernel path.
7. Add affine 8-bit non-routed linear path.
8. Add parser registration for Hunyuan/Tencent tool tags and qwen3-compatible reasoning.
9. Add `MTPDraftState` type but keep runtime mode disabled until tests exist.
10. Add tests: config decode, cache dimensions, parser smoke, fresh generation, continuation generation, no reasoning leak in `no_think`.

## Example Files

Use these as starting points:

- `jang-tools/examples/hy3/python_runtime/hy3_jangtq_runtime_skeleton.py`
- `jang-tools/examples/hy3/python_runtime/hy3_parser_contract.py`
- `jang-tools/examples/hy3/swift_runtime/Hy3JANGTQRuntimeSkeleton.swift`
- `jang-tools/examples/hy3/swift_runtime/Hy3ParserContract.swift`


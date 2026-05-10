# Multi-Token Prediction (MTP) layers in JANG bundles

Created 2026-05-09. Living doc — update with each MTP-bearing bundle.

## What MTP is

Multi-Token Prediction (MTP) is a technique where a model is trained with one or more **extra prediction heads** that forecast tokens at positions `t+2`, `t+3`, ... in addition to the usual next-token head at `t+1`. At inference time these heads can be used as **draft proposals** in speculative decoding: the main model verifies the draft tokens before they are committed to the accepted KV cache. Throughput gain depends on acceptance rate, runtime implementation, prompt shape, and batching. Do not claim a speedup until measured on the target runtime.

Key properties:

- The MTP layer(s) live **inside the same checkpoint** as the main transformer. They are extra layers / extra heads, not a separate distilled draft model.
- The MTP layer typically reuses the main model's `embed_tokens` and `lm_head`. It often has its own attention + MLP stack (smaller than the main layers).
- One MTP layer = predicts +1 token ahead per generation step. Multiple MTP layers compound.
- Acceptance rate is content-dependent — for code/math (deterministic continuations) it's higher than for free-form prose.

## Models in the JANG ecosystem with MTP

| Model | `model_type` | MTP layers | MTP namespace in source | JANG bundle policy |
|---|---|---|---|---|
| **Tencent Hy3-preview** | `hy_v3` | 1 (`num_nextn_predict_layers=1`) | TBD - discovered post-download (likely `model.layers.80.*` or `model.nextn_layers.0.*`) | Quantize MTP attention / MLP at 8-bit affine (same as dense FFN); MTP norms passthrough; weights ship in the bundle. First runtime status is `preserved_disabled` until accept/reject decoding is implemented. |
| **DeepSeek-V4-Flash** | `deepseek_v4` | 1 (built into the architecture) | `model.compressor.*`, `model.indexer.*` (MLA-style hybrid heads, not classical MTP but used the same way for spec decoding) | Compressor / Indexer kept passthrough or 8-bit affine per existing converter; `cache_subtype=deepseek_v4_hybrid`. |
| **Bailing v2.5 / Ling-2.6** | `bailing_moe_v2_5` | 1 | `model.mtp_layer.*` (varies by HF release) | Quantized at 8-bit affine in current converter; MTP runtime path pending. |
| **MiniMax M2.7 local JANGTQ bundles** | `minimax_m2` | 0 in inspected local configs (`use_mtp=false`, `num_mtp_modules=0`, `mtp_transformer_layers=0`) | n/a | MiniMax remains the closest MoE/router runtime analog for Hy3, but the current local M2.7 JANGTQ bundles are not an MTP validation target. Verify these three keys for every new MiniMax source before making a family-wide claim. |
| **ZAYA1-8B / ZAYA1-VL-8B** | `zaya`, `zaya1_vl` | 0 (no MTP) | n/a | n/a |
| **Kimi K2.6** | `kimi_k2` | 0 (no MTP) | n/a | n/a |

## How JANG converters preserve MTP

JANG converters should classify tensors by **tensor role**, not only by layer index. Any MTP-layer tensor that ships under one of the standard transformer prefixes (`self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`, `*_norm`, `embed_tokens`, `lm_head`) can use the same precision policy as the main transformer:

```
attention QKV/O          → 8-bit affine
dense FFN (gate/up/down) → 8-bit affine
norms (q_norm/k_norm/    → passthrough fp16
 input_layernorm/...)
embed_tokens / lm_head   → 8-bit affine (or fp32-sensitive flag honored)
```

No special handling is required only when the MTP namespace follows the standard transformer naming convention. If a future model uses non-standard names (for example `nextn_head.*` or `mtp_predictor.*`), add a passthrough rule for non-2D tensors and an affine rule for 2D matmuls before conversion.

The MTP layer's existence is recorded in `config.json` via `num_nextn_predict_layers` (Hy3) or model-specific keys (DSV4 has `compressor_dim`, `indexer_dim`). Bundles preserve those keys verbatim — no JANG-side rewriting.

## How to use MTP at inference time

### Today (no JANG MTP-aware runtime yet)

MTP-bearing JANG bundles should preserve the MTP layer weights, so any future runtime that grows MTP support can pick them up without re-converting. **Today's vmlx-swift-lm and jang-tools runtimes do NOT use Hy3 MTP for speculative decoding** - they decode one token per forward pass.

That means current decode speed on these bundles is "main model only." The MTP tensors are preserved for future compatibility, but they must be documented as disabled until an accept/reject speculative path is present.

### Reference recipes from upstream

For Hy3-preview, Tencent ships these recipes with MTP enabled:

```bash
# vLLM
vllm serve tencent/Hy3-preview \
  --tensor-parallel-size 8 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 1 \
  --tool-call-parser hy_v3 \
  --reasoning-parser hy_v3 \
  --enable-auto-tool-choice \
  --served-model-name hy3-preview

# SGLang
python3 -m sglang.launch_server \
  --model tencent/Hy3-preview \
  --tp 8 \
  --tool-call-parser hunyuan \
  --reasoning-parser hunyuan \
  --speculative-num-steps 1 \
  --speculative-algorithm EAGLE
```

For DSV4-Flash, the reference Python recipe (jang-tools `examples/dsv4_flash/`) similarly invokes the indexer/compressor; that's the closest existing JANG-side path that resembles MTP-style spec decoding.

### When JANG runtime grows MTP

Future runtime (Swift in `jang-runtime/Sources/JANG/<family>/` or Python in `jang-tools/jang_tools/<family>/runtime.py`) needs to:

1. Detect the MTP namespace at load time (read `num_nextn_predict_layers` from `config.json`; locate the layer's weights via the bundle's index).
2. At each decode step:
   - Run the main model forward to produce token `t+1` and its hidden state.
   - Run the MTP layer over the same hidden state to produce a draft for token `t+2`.
   - Accept the draft only if the main model's next-step forward (which is going to happen anyway) confirms it.
3. Cache the MTP-side intermediate state across steps (the MTP layer's KV cache is its own).

For Hy3 specifically, Tencent's transformers branch (5.6.0) implements this through their `HYV3ForCausalLM.generate` path with `speculative_config={method: 'mtp'}`. The reference can guide a Swift/MLX port.

## Per-bundle README integration

Every JANG bundle README (Osaurus + JANGQ-AI) for an MTP-bearing model should include this stanza in the runtime-status matrix:

```
| MTP spec decoding | not yet enabled in JANG runtime; weights preserved in bundle | upstream uses Tencent transformers / vLLM `--speculative-config.method mtp` |
```

The MTP-bearing bundles published to date:

- (pending) `OsaurusAI/Hy3-preview-JANGTQ2`
- `OsaurusAI/DSV4-Flash-JANGTQ` (compressor / indexer hybrid, MTP-adjacent)
- `OsaurusAI/Bailing-2.6-Ling-Flash-JANGTQ` (one MTP layer, runtime path pending)

## Open work

- Confirm Hy3 MTP tensor namespace (`model.layers.80.*` vs `model.nextn_layers.*`) once the source download finishes.
- Add a `bundle_has_mtp: true` flag to `jang_config.json` so runtimes can toggle MTP-aware decoding when supported.
- Build the first MTP-aware decode loop for Hy3 in a small Python reference path first, then port the proven accept/reject/cache rules to vmlx-swift-lm Swift.
- Cross-validate acceptance rate against Tencent's vLLM reference on a small prompt suite.

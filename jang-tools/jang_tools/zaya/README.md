# ZAYA (Zyphra) — JANG runtime

`model_type=zaya` — Zyphra ZAYA1-8B text-only. Alternates CCA attention
layers (even index) with top-1 MOD/MoE blocks (odd index). Bundles ship
pre-stacked routed experts under `zaya_block.experts.switch_mlp.*`.

Architecture summary:
- 80 layers (40 attention + 40 MoE), hidden 2048, 16 attn heads, 2 KV heads
- CCA attention: causal qk conv (ZayaQKV) + GQA + value-shifted projections
- Partial RoPE (`partial_rotary_factor=0.5`, `rope_theta=5e6`)
- Top-1 MoE with 16 experts + optional MOD skip route (`zaya_use_mod=True`)
- 131 K context, residual stream in fp32

For VL (`zaya1_vl`, 40 layers + Qwen2.5-VL ViT) the runtime is not yet
implemented in any engine.

## Convert

```
# JANGTQ2 / JANGTQ4 (recommended)
python -m jang_tools.convert_zaya_jangtq <bf16_src> <out> [JANGTQ2|JANGTQ4]

# MXFP4
python -m jang_tools.convert_zaya_mxfp4 <bf16_src> <out>
```

## Load (Python)

```python
from jang_tools.zaya import load_zaya_model

model, tokenizer = load_zaya_model("/path/to/ZAYA1-8B-JANGTQ2")
```

JANGTQ/MXTQ ZAYA bundles should go through
`jang_tools.load_jangtq.load_jangtq_model` instead — it replaces the
routed expert projections with TurboQuant kernels.

## Capabilities

`build_capabilities` produces:

```
family=zaya
reasoning_parser=qwen3
tool_parser=zaya_xml
think_in_template=False     # template emits closed </think>
supports_thinking=False     # qwen3 parser would route everything to reasoning_content
cache_type=hybrid
modality=text
```

The `think_in_template=False` rule is critical: ZAYA's chat template
defaults `enable_thinking=false` and renders a CLOSED `</think>` rather
than auto-opening one. With `think_in_template=True` the qwen3 reasoning
parser would assume the assistant already opened a think block and route
ALL output to `reasoning_content`, leaving `content` empty.

## Cache topology

Even (attention) layers use `CacheList(KVCache(), ArraysCache(2))` — the
`ArraysCache(2)` slots hold the conv state and the previous hidden state
for the value-shifted projection. Odd (MoE) layers use
`ZayaNoStateCache` (no recurrent state, but slice-safe under
`BatchGenerator.extract`).

## vmlx integration

`vmlx_engine.loaders.load_zaya.load_zaya_model` re-exports this module's
`load_zaya_model` so existing vmlx import paths keep working.

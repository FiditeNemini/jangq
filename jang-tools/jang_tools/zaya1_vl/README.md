# ZAYA1-VL (Zyphra) — JANG runtime

`model_type=zaya1_vl` — Zyphra ZAYA1-VL-8B vision-language model. Reuses
a Qwen2.5-VL vision tower for image inputs; the language trunk is the
ZAYA text-only architecture **truncated to 40 layers** (vs `zaya`'s 80).
Each text layer has CCA attention + top-1 MOD/MoE with optional
image-token LoRA paths on both sublayers.

Bundle layout:
- 40 layers (20 CCA attention + 20 MoE), hidden 2048, 16 attn / 2 KV heads
- Qwen2.5-VL vision tower (hidden 1280)
- Vision-LoRA: rank-8 on attention, rank-32 on MLP, gated to image-token
  positions only — text positions decode unmodified
- `image_token_id=262147`, `vision_start_token_id=255999`,
  `vision_end_token_id=256000`
- Routed experts pre-stacked under `zaya_block.experts.switch_mlp.*`

## Convert

```
# JANGTQ2 / JANGTQ4 (recommended)
python -m jang_tools.convert_zaya1_vl_jangtq <bf16_src> <out> [JANGTQ2|JANGTQ4]

# MXFP4
python -m jang_tools.convert_zaya1_vl_mxfp4 <bf16_src> <out>
```

## Load (Python)

```python
from jang_tools.zaya1_vl import load_zaya1_vl_model

# Requires mlx_vlm: pip install mlx_vlm
model, processor = load_zaya1_vl_model("/path/to/ZAYA1-VL-8B-JANGTQ2")
```

JANGTQ/MXTQ ZAYA1-VL bundles should normally go through
`jang_tools.load_jangtq.load_jangtq_model` instead — that function calls
`register_mlx_vlm_zaya1_vl()` then swaps routed expert projections with
TurboQuantLinear after load.

## Capabilities

`build_capabilities` produces:

```
family=zaya1_vl
reasoning_parser=qwen3
tool_parser=zaya_xml
think_in_template=False     # template default emits closed </think>
supports_thinking=True      # model reasons by default — measured live (2026-05-10)
cache_type=hybrid
modality=vision
```

`enable_thinking=True` in `apply_chat_template` opens the think block;
`enable_thinking=False` closes it but the model still produces
chain-of-thought in plain text afterward.

## Tensor namespace

Adapter `Model.sanitize` performs three transforms before mlx_vlm load:
- `model.*` → `language_model.model.*` (mlx_vlm's expected layout)
- `lm_head.*` → `language_model.lm_head.*`
- `layers.{n}.zaya_block.*` → `layers.{n}.mlp.zaya_block.*` (some
  upstream checkpoints ship this without the `mlp.` prefix)
- `patch_embed.proj.weight` axis transpose if shipped in (in_ch, out_ch,
  ...) form rather than mlx-vlm's expected layout
- `self_attn.qkv.conv_qk.weight` swapaxes (mirrors zaya text trunk fix)

## Runtime dependencies

- `mlx` ≥ 0.23
- `mlx_lm` (for shared switch_layers + base classes)
- `mlx_vlm` (for `qwen2_5_vl.VisionModel` + `InputEmbeddingsFeatures` +
  `LanguageModelOutput`)

`mlx_vlm` is an optional package extra for `jang`, but it is required before
importing this adapter because the model class subclasses `mlx_vlm` base
classes and reuses the Qwen2.5-VL vision tower.

## Source

- Ported from the vmlx ZAYA1-VL adapter used during local runtime bring-up.
- Sibling helpers reused from `jang_tools.zaya.model`:
  `ResidualScaleMerge`, `ZayaNoStateCache`, `ZayaPartialRoPE`,
  `ZayaRouter`. The text-trunk fix for `ZayaRouter.balancing_biases`
  (MOD slot init at -1.0) is now also in `jang_tools.zaya.model`.

## vmlx integration

`vmlx_engine.loaders.load_jangtq_vlm` should re-export
`load_zaya1_vl_model` from this module — mirrors the `load_jangtq_dsv4`
/ `load_jangtq_kimi_vlm` pattern. The vmlx wrapper does not need its
own model code; this package is the source of truth.

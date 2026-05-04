# Mistral-Medium-3.5 runtime examples

Working examples for the Mistral-Medium-3.5-128B (mistral3 + ministral3 + pixtral VL) JANGTQ bundle.

| # | File | What it shows |
|---|---|---|
| 00 | `00_verify.py` | Bundle metadata + smoke greedy decode |
| 01 | `01_chat.py` | Multi-prompt chat with the Mistral instruct chat template |

## Default bundle

```
~/.mlxstudio/models/_bundles/Mistral-Medium-3.5-128B-JANGTQ
```

## Architecture notes

`model_type=mistral3` is a wrapper around:
- `ministral3` text decoder (40 layers, FP8 e4m3 source quantized to MXTQ)
- `pixtral` vision tower (24 layers, kept fp16 passthrough)
- `multi_modal_projector` (kept fp16 passthrough)

JANGTQ converts the text decoder linears + embed_tokens to MXTQ; vision tower / projector / lm_head are explicitly listed in `modules_to_not_convert` and pass through as fp16.

## Vision input — current state

Vision input is wired through `jang_tools.mistral3.runtime.encode_with_image` + `jang_tools.vl.pixtral.encode_image_pixtral`. The runtime currently strips the vision tower at load time (text-only path); when we wire the vision fold-in step (place pixtral patch embeddings at placeholder positions) the same example shape will exercise multimodal end-to-end.

## Reasoning + tools

Mistral 3.5 does **not** ship with trained `<think>` reasoning mode. Prompts run normally; no reasoning split needed. Tool calls use the standard Mistral function-calling syntax (JSON in the assistant message); we don't have a parser baked in here yet.

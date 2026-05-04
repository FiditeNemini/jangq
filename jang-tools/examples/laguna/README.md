# Laguna runtime examples

Working examples for Laguna-XS.2 (33B/3B agentic-coding MoE) JANGTQ on Apple Silicon.

| # | File | What it shows |
|---|---|---|
| 00 | `00_verify.py` | Bundle metadata + smoke decode → verifies hydration |
| 01 | `01_code_completion.py` | Three programming prompts (FIM / typed / bug-fix) |

## Default bundle

```
~/.mlxstudio/models/_bundles/Laguna-XS.2-JANGTQ
```
Pass any other path as the first argument.

## Architecture notes

Laguna (`model_type=laguna`) is a hybrid SWA + full-attention MoE:

| field | value |
|---|---|
| layers | 40 |
| layer_types | mix of `full_attention` and `sliding_attention` |
| heads/layer | 48 on full / 64 on SWA |
| RoPE | YaRN on full, default on SWA |
| experts | 256 routed top-8 + 1 shared |
| MoE intermediate | 512 |

The runtime in `jang_tools.laguna.runtime` handles auto-format detection (bf16 / JANG affine / JANGTQ / MXFP4) and routes per-layer head count + RoPE flavor automatically.

## Reasoning + tools

Laguna **does** support `<think>` reasoning — based on `laguna_glm_thinking_v5/chat_template.jinja`. Pass `enable_thinking=True` to `apply_chat_template`:

```python
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Refactor this function..."}],
    tokenize=False, add_generation_prompt=True,
    enable_thinking=True,
)
```

The chat template prefills `<think>` (thinking on) or `</think>` (thinking off) at the assistant turn. The model emits its reasoning inside the `<think>...</think>` block, then the final answer.

Tool calling: GLM4-style `<tool_call>...</tool_call>` blocks. The Swift side dispatches via `ToolCallFormat.glm4` for `laguna` model_type (`vmlx-swift-lm/Libraries/MLXLMCommon/Tool/ToolCallFormat.swift:236`).

Reasoning split (Python): same partition pattern as DSV4 — `text.partition("</think>")` after generation.

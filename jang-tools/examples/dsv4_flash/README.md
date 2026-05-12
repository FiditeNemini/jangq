# DSV4-Flash runtime examples

End-to-end working examples for the DeepSeek-V4-Flash JANGTQ bundle on Apple Silicon.

| # | File | What it shows |
|---|---|---|
| 00 | `00_verify.py` | Bundle metadata + smoke greedy decode → verifies hydration |
| 01 | `01_text_only.py` | All 3 modes (chat / think / think_max) end-to-end |
| 02 | `02_thinking.py` | Reasoning vs content split + leak audit on a hard math question |
| 03 | `03_tool_calling.py` | DSML tool-call parsing (`｜DSML｜` markers) |
| 04 | `04_long_context.py` | Needle-in-haystack at long context — exercises HSA + CSA path |
| 05 | `05_streaming_generation.py` | Token-streaming reasoning/content splitter (server pattern) |

## Default bundle

All scripts default to:
```
~/.mlxstudio/models/_bundles/DeepSeek-V4-Flash-JANGTQ
```
Pass any other path as the first argument.

## Runtime knobs

| Env var | Default | Purpose |
|---|---|---|
| `DSV4_LONG_CTX` | `1` | Tri-mode HSA + CSA + SWA. Set `0` to fall back to legacy SWA-only (loses ≥7pp MMLU). |
| `DSV4_POOL_QUANT` | `0` | Correctness default. Pool quant is opt-in until compressor/indexer pool restore is proven for the exact runtime. |
| `JANG_MEMORY_LIMIT_GB` | `200` | mx.set_memory_limit. Bump if you have >200 GB unified. |

## Reasoning mode glossary

| Mode | `enable_thinking` | `reasoning_effort` | Default `max_tokens` |
|---|---|---|---|
| `chat` | False | None | 2048 |
| `think` | True | None | 8192 |
| `think_max` | True | `max` | 32768 |
| `fim` | (no template) | — | 1024 |

`think` and `think_max` need their token budgets — under-budgeting cuts off `</think>` and the runtime sets `truncated=True` on the result.

## Leak guarantees

`runtime.generate()` returns `GenerateResult` with separate `.reasoning_content` and `.content` fields. The `.content` is post-`</think>` only. Streaming pattern in 05 maintains the same invariant on a per-frame basis (handles split-across-token boundaries).

The DSML tool parser in `jang_tools.dsv4.test_chat.parse_dsml_tool_calls` reads `<｜DSML｜invoke …>` blocks. Note: those are **fullwidth** pipes (U+FF5C), not ASCII pipes — DeepSeek's choice, not ours.

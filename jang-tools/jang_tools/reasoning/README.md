# Reasoning parsers (mirrored from vmlx-engine)

Mirrors `vmlx_engine/reasoning/` — these strip thinking blocks
(`<think>…</think>`, `[THINK]…[/THINK]`, etc) before downstream extraction.
Used by `jang_tools.eval.mmlu` for the reasoning pass.

| parser | tag style | for |
|---|---|---|
| `MistralReasoningParser` | `[THINK]…[/THINK]` (special tokens 34/35) | mistral3, ministral3, mistral4 |
| `Qwen3ReasoningParser` | `<think>…</think>` | qwen3, qwen3_5_moe, laguna (compatible) |
| `DeepseekR1ReasoningParser` | `<think>…</think>` w/ explicit close | deepseek_v3, dsv4 |
| `Gemma4ReasoningParser` | gemma-specific markers | gemma4 |
| `GptOssReasoningParser` | gpt-oss control tokens | gpt-oss family |

All inherit `BaseThinkingReasoningParser` (`think_parser.py`) with
streaming-safe extraction. Original implementation: Jinho Jang
(eric@jangq.ai) for vMLX / mlxstudio.

Usage:
```python
from jang_tools.reasoning.mistral_parser import MistralReasoningParser
parser = MistralReasoningParser()
content = parser.extract_content(model_output)   # everything after [/THINK]
reasoning = parser.extract_reasoning(model_output)
```

`jang_tools.eval.mmlu._extract_answer` calls the right parser by
`config.text_config.model_type` automatically.

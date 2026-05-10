"""DeepSeek-V4 chat-template constants.

The authoritative Python runtime path is ``encoding/encoding_dsv4.py`` inside
each bundle. This Jinja template is the portable tokenizer_config fallback used
by Swift and generic HF-style tokenizer loaders.
"""

DSV4_CHAT_TEMPLATE_JINJA = """{%- set thinking_on = (enable_thinking is defined and enable_thinking) -%}
{%- set effort = (reasoning_effort if reasoning_effort is defined else None) -%}
{%- set ns = namespace(last_user_idx=-1) -%}
{%- for m in messages -%}
{%- if m.role == 'user' or m.role == 'developer' -%}
{%- set ns.last_user_idx = loop.index0 -%}
{%- endif -%}
{%- endfor -%}
<｜begin▁of▁sentence｜>
{%- if thinking_on and effort == 'max' -%}
Reasoning Effort: Absolute maximum with no shortcuts permitted.
You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.
Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.

{% endif -%}
{%- for m in messages -%}
{%- set idx = loop.index0 -%}
{%- if m.role == 'system' -%}
{{- m.content -}}
{%- elif m.role == 'user' or m.role == 'developer' -%}
<｜User｜>{{- m.content -}}
{%- elif m.role == 'assistant' -%}
<｜Assistant｜>
{%- set rc = m.reasoning_content if m.reasoning_content is defined else '' -%}
{%- if thinking_on and rc and idx > ns.last_user_idx -%}
<think>{{- rc -}}</think>
{%- endif -%}
{{- m.content -}}
<｜end▁of▁sentence｜>
{%- elif m.role == 'tool' -%}
<tool_result>{{- m.content -}}</tool_result>
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<｜Assistant｜>
{%- if thinking_on -%}<think>{%- else -%}</think>{%- endif -%}
{%- endif -%}
"""

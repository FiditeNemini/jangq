"""Regression gates for Inkling affine bundle metadata."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from jang_tools.capabilities import verify_directory
from jang_tools.convert_inkling_jang_affine import (
    INKLING_EOS_TOKEN,
    INKLING_EOS_TOKEN_ID,
    _normalize_inkling_chat_template,
    repair_inkling_bundle_metadata,
)


VENDOR_TEMPLATE = """\
{%- set effort_map = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "max": 0.99} -%}
{%- set role_token = {"user": "<|message_user|>", "assistant": "<|message_model|>", "system": "<|message_system|>", "tool": "<|message_tool|>"} -%}

{%- macro emit_thinking_effort() -%}
    {%- set eff = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 0.9 -%}
    {%- if eff is string -%}
        {%- set key = eff | trim -%}
        {%- if key not in effort_map -%}
            {{- raise_exception("Unknown reasoning_effort: " ~ eff) -}}
        {%- endif -%}
        {%- set num = effort_map[key] -%}
    {%- else -%}
        {%- set num = eff | float -%}
    {%- endif -%}
    {%- if num < 0.0 or num > 0.99 -%}
        {{- raise_exception("reasoning_effort must be in [0.0, 0.99]") -}}
    {%- endif -%}
    {{- "<|message_system|><|content_text|>Thinking effort level: " -}}
    {%- if num == 0.0 -%}0{%- else -%}{{ num }}{%- endif -%}
    {{- "<|end_message|>" -}}
{%- endmacro -%}

{%- if not state.effort_emitted -%}
    {{- emit_thinking_effort() -}}
{%- endif -%}
{%- if add_generation_prompt -%}
    {{- "<|message_model|>" -}}
{%- endif -%}
"""


def _render_effort(template: str, **kwargs: object) -> str:
    macro_only = template.split(
        "{%- if not state.effort_emitted -%}", 1
    )[0] + "{{- emit_thinking_effort() -}}"
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(message: str) -> None:
        raise ValueError(message)

    env.globals["raise_exception"] = raise_exception
    return env.from_string(macro_only).render(**kwargs)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "Inkling-Small-JANG"
    bundle.mkdir()
    _write_json(
        bundle / "config.json",
        {
            "model_type": "inkling_mm_model",
            "eos_token_id": INKLING_EOS_TOKEN_ID,
            "vision_config": {"vision_encoder_type": "hmlp"},
            "audio_config": {"audio_mode": "dmel"},
        },
    )
    _write_json(
        bundle / "jang_config.json",
        {
            "format": "jang",
            "format_version": 2,
            "weight_format": "affine",
            "source_model": "/fixture/Inkling-Small-BF16",
        },
    )
    _write_json(
        bundle / "tokenizer_config.json",
        {
            "added_tokens_decoder": {
                str(INKLING_EOS_TOKEN_ID): {
                    "content": INKLING_EOS_TOKEN,
                    "special": True,
                }
            }
        },
    )
    _write_json(
        bundle / "special_tokens_map.json",
        {"additional_special_tokens": [INKLING_EOS_TOKEN]},
    )
    _write_json(
        bundle / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 1},
            "weight_map": {
                "model.visual.encoder.weight": "model-00001-of-00001.safetensors",
                "model.audio.encoder.weight": "model-00001-of-00001.safetensors",
            },
        },
    )
    (bundle / "chat_template.jinja").write_text(
        VENDOR_TEMPLATE, encoding="utf-8"
    )
    return bundle


def test_chat_template_preserves_native_effort_scalar_and_default_on():
    normalized = _normalize_inkling_chat_template(VENDOR_TEMPLATE)

    assert normalized == VENDOR_TEMPLATE
    assert "enable_thinking" not in normalized
    assert _render_effort(normalized).endswith(
        "Thinking effort level: 0.9<|end_message|>"
    )
    # Inkling's source does not consume the generic boolean. An engine must
    # forward reasoning_effort rather than silently treating this as Qwen.
    assert _render_effort(normalized, enable_thinking=False).endswith(
        "Thinking effort level: 0.9<|end_message|>"
    )
    assert _render_effort(normalized, enable_thinking=True).endswith(
        "Thinking effort level: 0.9<|end_message|>"
    )
    assert _render_effort(
        normalized, reasoning_effort="low", enable_thinking=False
    ).endswith("Thinking effort level: 0.2<|end_message|>")
    assert _render_effort(normalized, reasoning_effort="max").endswith(
        "Thinking effort level: 0.99<|end_message|>"
    )
    with pytest.raises(ValueError, match="Unknown reasoning_effort"):
        _render_effort(normalized, reasoning_effort="xhigh")


def test_chat_template_normalizer_only_undoes_the_old_local_boolean_patch():
    old_local = VENDOR_TEMPLATE.replace(
        '"high": 0.9, "max": 0.99',
        '"high": 0.9, "xhigh": 0.99, "max": 0.99',
        1,
    ).replace(
        "    {%- set eff = reasoning_effort if reasoning_effort is defined "
        "and reasoning_effort is not none else 0.9 -%}",
        """\
    {%- if reasoning_effort is defined and reasoning_effort is not none -%}
        {%- set eff = reasoning_effort -%}
    {%- elif enable_thinking is defined -%}
        {%- set eff = 0.9 if enable_thinking else 0.0 -%}
    {%- else -%}
        {%- set eff = 0.9 -%}
    {%- endif -%}""",
        1,
    )

    assert _normalize_inkling_chat_template(old_local) == VENDOR_TEMPLATE


def test_metadata_repair_stamps_eos_parsers_capabilities_and_is_idempotent(
    tmp_path: Path,
):
    bundle = _fixture_bundle(tmp_path)
    result = repair_inkling_bundle_metadata(bundle)

    generation = json.loads((bundle / "generation_config.json").read_text())
    tokenizer = json.loads((bundle / "tokenizer_config.json").read_text())
    special = json.loads((bundle / "special_tokens_map.json").read_text())
    config = json.loads((bundle / "config.json").read_text())
    jang = json.loads((bundle / "jang_config.json").read_text())

    assert result["eos_token_id"] == INKLING_EOS_TOKEN_ID
    assert generation["eos_token_id"] == INKLING_EOS_TOKEN_ID
    assert generation["reasoning_parser"] == "inkling"
    assert generation["tool_call_parser"] == "inkling"
    assert generation["default_chat_template_kwargs"] == {
        "reasoning_effort": "high"
    }
    assert not {
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "max_new_tokens",
    } & generation.keys()

    assert tokenizer["eos_token"] == INKLING_EOS_TOKEN
    assert tokenizer["chat_template"] == (
        bundle / "chat_template.jinja"
    ).read_text()
    assert special["eos_token"] == INKLING_EOS_TOKEN

    assert config["capabilities"] == jang["capabilities"]
    assert config["default_chat_template_kwargs"] == {
        "reasoning_effort": "high"
    }
    assert jang["capabilities"]["family"] == "inkling"
    assert jang["capabilities"]["reasoning_parser"] == "inkling"
    assert jang["capabilities"]["tool_parser"] == "inkling"
    assert jang["capabilities"]["modality"] == "multimodal"
    assert jang["chat"]["reasoning"]["default_effort"] == 0.9
    assert jang["chat"]["reasoning"]["default_mode"] == "high"
    assert jang["chat"]["reasoning"]["control_argument"] == "reasoning_effort"
    assert jang["chat"]["reasoning"]["boolean_enable_thinking_supported"] is False
    assert jang["chat"]["reasoning"]["omitted_argument_effort"] == 0.9
    assert jang["chat"]["default_chat_template_kwargs"] == {
        "reasoning_effort": "high"
    }
    assert jang["chat"]["sampling_defaults"] == {}

    ok, message = verify_directory(bundle)
    assert ok, message

    controlled = (
        "config.json",
        "jang_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
    )
    before = {name: (bundle / name).read_bytes() for name in controlled}
    repair_inkling_bundle_metadata(bundle)
    after = {name: (bundle / name).read_bytes() for name in controlled}
    assert after == before


def test_every_inkling_converter_runs_the_shared_default_on_finalizer():
    package = Path(__file__).parents[1] / "jang_tools"
    for filename in (
        "convert_inkling_jang_affine.py",
        "convert_inkling_jangtq.py",
    ):
        source = (package / filename).read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "repair_inkling_bundle_metadata" in calls, filename
        assert '"generation_config.json"' in source, filename

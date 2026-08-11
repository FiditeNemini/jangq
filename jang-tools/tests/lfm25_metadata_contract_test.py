"""LFM2.5 converter policy + bundle metadata contract tests.

Created by Jinho Jang (eric@jangq.ai) — 2026-08-04.

Two halves:
  1. Policy unit tests — tier rules, AWQ scale math, template-evidence
     think_in_template lift, pack parity. Always run.
  2. Bundle contract tests — run against any built
     ~/.mlxstudio/models/JANGQ-AI/LFM2.5-2.6B-{MXFP8,JANG_6M} bundle found
     on disk (skipped when absent), asserting the full metadata contract:
     sampling two-file agreement, reasoning always-on, capabilities
     verify_directory pass, vendor-MLX-parity config keys.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jang_tools.capabilities import _template_preopens_think, verify_directory
from jang_tools.lfm25.convert import (
    CARD_DOCUMENTED_SAMPLING,
    PROFILES,
    _awq_scale,
    tier_bits,
)
from jang_tools.lfm25.qat_gptq import pack_self_test

BUNDLES = [
    Path("~/.mlxstudio/models/JANGQ-AI/LFM2.5-2.6B-MXFP8").expanduser(),
    Path("~/.mlxstudio/models/JANGQ-AI/LFM2.5-2.6B-JANG_6M").expanduser(),
]
_built = [b for b in BUNDLES if (b / "jang_config.json").is_file()]


# ── policy units ──────────────────────────────────────────────────────────


def test_profiles_are_dense_safe_tiers_only():
    assert set(PROFILES) == {"MXFP8", "JANG_6M"}
    assert PROFILES["MXFP8"] == {
        "mode": "mxfp8", "operator": 8, "ffn": 8, "embed": 8, "top": 8}
    assert PROFILES["JANG_6M"] == {
        "mode": "affine", "operator": 8, "ffn": 6, "embed": 6, "top": 8}


@pytest.mark.parametrize("profile_name", sorted(PROFILES))
def test_tier_rules(profile_name):
    p = PROFILES[profile_name]
    # token-mixing operators are CRITICAL 8-bit in every profile
    assert tier_bits("model.layers.2.self_attn.q_proj.weight", p) == 8
    assert tier_bits("model.layers.2.self_attn.out_proj.weight", p) == 8
    assert tier_bits("model.layers.0.conv.in_proj.weight", p) == 8
    assert tier_bits("model.layers.0.conv.out_proj.weight", p) == 8
    # FFN + tied embedding follow the profile
    assert tier_bits("model.layers.0.feed_forward.w1.weight", p) == p["ffn"]
    assert tier_bits("model.layers.0.feed_forward.w2.weight", p) == p["ffn"]
    assert tier_bits("model.layers.0.feed_forward.w3.weight", p) == p["ffn"]
    assert tier_bits("model.embed_tokens.weight", p) == p["embed"]
    # fp16 passthrough set
    for n in (
        "model.layers.0.conv.conv.weight",
        "model.layers.2.self_attn.q_layernorm.weight",
        "model.layers.2.self_attn.k_layernorm.weight",
        "model.layers.0.operator_norm.weight",
        "model.layers.0.ffn_norm.weight",
        "model.embedding_norm.weight",
    ):
        assert tier_bits(n, p) is None, n


def test_tier_rules_refuse_unknown_tensors():
    with pytest.raises(SystemExit):
        tier_bits("model.layers.0.mystery.weight", PROFILES["JANG_6M"])


def test_awq_scale_is_geomean_normalized_and_clipped():
    m = np.array([1e-4, 0.1, 1.0, 10.0, 1e4], dtype=np.float32)
    s = _awq_scale(m)
    assert s.shape == m.shape
    assert (s >= 0.5).all() and (s <= 2.0).all()
    # monotone in the activation magnitude
    assert (np.diff(s) >= 0).all()
    # uniform stats -> no-op fold
    assert np.allclose(_awq_scale(np.full(8, 3.0, dtype=np.float32)), 1.0)


def test_card_documented_sampling_matches_vendor_card():
    # LiquidAI card, LFM2.5-2.6B "Generation parameters"
    assert CARD_DOCUMENTED_SAMPLING == {
        "temperature": 0.1, "top_k": 50, "repetition_penalty": 1.1}


def test_pack_parity():
    pack_self_test()


def test_template_preopens_think_static_analysis(tmp_path):
    # LFM2.5-2.6B shape: unconditional literal ending in <think>
    (tmp_path / "chat_template.jinja").write_text(
        '{%- if add_generation_prompt -%}\n'
        '    {{- "<|im_start|>assistant\\n<think>" -}}\n'
        '{%- endif -%}\n')
    assert _template_preopens_think(tmp_path)
    # old-LFM2 shape: generation prompt ends at assistant\n
    (tmp_path / "chat_template.jinja").write_text(
        '{%- if add_generation_prompt -%}\n'
        '    {{- "<|im_start|>assistant\\n" -}}\n'
        '{%- endif -%}\n')
    assert not _template_preopens_think(tmp_path)
    # conditional thinking (bailing/qwen3 style) keeps the table value
    (tmp_path / "chat_template.jinja").write_text(
        '{%- if add_generation_prompt -%}\n'
        '  {{- "<|im_start|>assistant\\n" -}}\n'
        '  {%- if enable_thinking -%}{{- "<think>" -}}{%- endif -%}\n'
        '{%- endif -%}\n')
    assert not _template_preopens_think(tmp_path)
    # zaya-style closed prefill is NOT a pre-open
    (tmp_path / "chat_template.jinja").write_text(
        '{%- if add_generation_prompt -%}\n'
        '    {{- "<|im_start|>assistant\\n<think></think>" -}}\n'
        '{%- endif -%}\n')
    assert not _template_preopens_think(tmp_path)
    # no template file -> never guesses True
    (tmp_path / "chat_template.jinja").unlink()
    assert not _template_preopens_think(tmp_path)


# ── built-bundle contract ─────────────────────────────────────────────────


@pytest.mark.parametrize("bundle", _built or [None],
                         ids=[b.name for b in _built] or ["no-bundle"])
def test_bundle_contract(bundle):
    if bundle is None:
        pytest.skip("no built LFM2.5 bundle on disk")
    config = json.loads((bundle / "config.json").read_text())
    jang = json.loads((bundle / "jang_config.json").read_text())
    gen = json.loads((bundle / "generation_config.json").read_text())

    # vendor-MLX-parity keys mlx_lm ModelArgs needs
    assert config["block_ff_dim"] == config["intermediate_size"] == 10752
    assert config["rope_theta"] == config["rope_parameters"]["rope_theta"]
    assert isinstance(config["eos_token_id"], list)
    assert config["model_type"] == "lfm2"

    # quantization stanzas agree and carry a mode
    assert config["quantization"] == config["quantization_config"]
    assert config["quantization"]["mode"] in ("affine", "mxfp8")
    if jang["profile"] == "JANG_6M":
        assert config["quantization"]["bits"] == 8
        ov = config["quantization"]["model.layers.0.feed_forward.w1"]
        assert ov == {"group_size": 32, "bits": 6, "mode": "affine"}
        assert config["quantization"]["model.embed_tokens"]["bits"] == 6
    else:
        assert config["quantization"]["mode"] == "mxfp8"

    # sampling two-file agreement (the whole point of the gate)
    sd = jang["chat"]["sampling_defaults"]
    for k, v in sd.items():
        assert gen.get(k) == v, f"generation_config[{k}]={gen.get(k)} != {v}"
    for k, v in CARD_DOCUMENTED_SAMPLING.items():
        assert sd.get(k) == v

    # reasoning always-on, no off mode
    r = jang["chat"]["reasoning"]
    assert r["supported"] and r["default_enabled"] and r["always_on"]
    assert r["modes"] == ["think"]
    assert jang["chat"]["tool_calling"]["parser"] == "lfm2"

    # capabilities: text-only, think-in-template, verify gate passes
    caps = jang["capabilities"]
    assert caps == config["capabilities"]
    assert caps["think_in_template"] is True
    assert caps["family"] == "lfm2"
    assert caps["cache_type"] == "hybrid"
    assert caps["modality"] == "text"
    assert not caps["has_vision"] and not caps["has_audio"] and not caps["has_video"]
    ok, msg = verify_directory(bundle)
    assert ok, msg

    # template shipped verbatim + inlined into tokenizer_config
    tmpl = (bundle / "chat_template.jinja").read_text()
    assert tmpl.rstrip().endswith('{{- "<|im_start|>assistant\\n<think>" -}}\n{%- endif -%}')
    tok_cfg = json.loads((bundle / "tokenizer_config.json").read_text())
    assert tok_cfg.get("chat_template") == tmpl

    # jang_runtime hybrid cache contract
    rt = config["jang_runtime"]
    assert rt["full_attention_layers"] == [2, 5, 9, 13, 17, 21, 24, 27]
    assert rt["num_conv_layers"] == 22
    assert rt["conv_L_cache"] == 3
    assert rt["norm_convention"] == "llama_rmsnorm_no_plus_one"
    assert rt["tied_embeddings"] is True

    # AWQ + QAT provenance stamped
    q = jang["quantization"]
    assert q["awq"]["applied"] is True
    assert q["qat"]["applied"] is True
    assert q["qat"]["tensors"] == 90  # 30 layers x w1/w2/w3
    assert (bundle / "qat_report.json").is_file()

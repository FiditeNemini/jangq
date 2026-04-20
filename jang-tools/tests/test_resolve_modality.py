"""M127 (iter 50): `_resolve_modality` text_config-is-vision fallback bug.

Peer-helper audit of `_resolve_family_str` vs `_resolve_modality` in
capabilities.py surfaced that the modality fallback (`"text_config" in
config or "vision_config" in config`) misclassifies text-only MoE models
(Qwen3-MoE, Qwen3.5-MoE, GLM-MoE, Mistral 4) as vision whenever the
jang_config is missing a `has_vision` stamp — because those configs wrap
their text params under `text_config` WITHOUT a `vision_config`.

For our own converters this isn't an issue because we always stamp
`has_vision` in `jang.architecture.has_vision`. But legacy jang_configs
(pre-stamp), third-party JANG models, and user-edited jang_configs can
hit the fallback and get wrong routing at the vmlx CapabilityDetector
layer (loads via VLMModelFactory instead of LLMModelFactory → crash).

Iter 50 tightens the fallback to `vision_config in config` only.
"""
from jang_tools.capabilities import _resolve_modality


# ──────────────────────────── Baseline behaviors (preserved) ─────────────────


def test_modality_explicit_has_vision_true():
    """Top-level jang.has_vision wins over everything else."""
    assert _resolve_modality({"has_vision": True}, {}) == "vision"


def test_modality_explicit_has_vision_false():
    """Top-level jang.has_vision=false wins even if config looks vision-y."""
    assert _resolve_modality(
        {"has_vision": False}, {"vision_config": {}}
    ) == "text"


def test_modality_arch_has_vision_true():
    """architecture.has_vision (stamped by convert.py) wins over config probe."""
    assert _resolve_modality(
        {"architecture": {"has_vision": True}}, {}
    ) == "vision"


def test_modality_arch_has_vision_false_with_vision_config():
    """architecture.has_vision=false wins even if config has vision_config.
    Scenario: converter ran with VL weights stripped, stamped has_vision=false."""
    assert _resolve_modality(
        {"architecture": {"has_vision": False}},
        {"vision_config": {}},
    ) == "text"


def test_modality_fallback_vision_config_detected():
    """No has_vision stamp, but config has vision_config → vision."""
    assert _resolve_modality({}, {"vision_config": {"hidden_size": 1152}}) == "vision"


def test_modality_fallback_no_hints_defaults_to_text():
    """No has_vision, no vision_config, no text_config → text."""
    assert _resolve_modality({}, {"model_type": "llama"}) == "text"


# ───────── M127: the fix. `text_config` alone must NOT imply vision ──────────


def test_modality_text_config_without_vision_is_text_for_qwen3_moe():
    """Qwen3-MoE stores its text params under `text_config` but has NO
    vision component. A legacy jang_config without has_vision stamping must
    classify this as text, not vision.

    Pre-iter-50 behavior: misclassified as 'vision' → vmlx routes through
    VLMModelFactory → model class mismatch at runtime.
    """
    qwen3_moe_config = {
        "model_type": "qwen3_moe",
        "text_config": {
            "model_type": "qwen3_moe",
            "num_experts": 128,
            "hidden_size": 4096,
        },
    }
    assert _resolve_modality({}, qwen3_moe_config) == "text"


def test_modality_text_config_without_vision_is_text_for_qwen3_5_moe():
    """Same pathology for Qwen3.5-MoE (hybrid SSM)."""
    qwen35_moe_config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe",
            "num_experts": 256,
            "hidden_size": 3072,
        },
    }
    assert _resolve_modality({}, qwen35_moe_config) == "text"


def test_modality_text_config_without_vision_is_text_for_glm_moe():
    """GLM-5 MoE — same pathology."""
    glm_config = {
        "model_type": "glm_moe_dsa",
        "text_config": {"model_type": "glm_moe_dsa", "n_routed_experts": 256},
    }
    assert _resolve_modality({}, glm_config) == "text"


def test_modality_text_plus_vision_config_is_still_vision():
    """A real VL model has BOTH text_config AND vision_config. The fix must
    not regress the real-VL fallback path."""
    vl_config = {
        "model_type": "qwen3_vl",
        "text_config": {"model_type": "qwen3_5_moe"},
        "vision_config": {"hidden_size": 1152},
    }
    assert _resolve_modality({}, vl_config) == "vision"

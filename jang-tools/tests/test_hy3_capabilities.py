from __future__ import annotations

import importlib
import sys
from pathlib import Path

from jang_tools.capabilities import build_capabilities


def test_hy3_capabilities_match_request_dependent_thinking_contract():
    caps = build_capabilities(
        {"source_model": {"architecture": "hy_v3"}},
        {"model_type": "hy_v3"},
    )

    assert caps is not None
    assert caps["family"] == "hy_v3"
    assert caps["reasoning_parser"] == "qwen3"
    assert caps["tool_parser"] == "hunyuan"
    assert caps["supports_thinking"] is True
    assert caps["think_in_template"] is False
    assert caps["cache_type"] == "kv"
    assert caps["modality"] == "text"


def test_load_jangtq_auto_registers_hy3_before_skeleton_load():
    src = (Path(__file__).parents[1] / "jang_tools/load_jangtq.py").read_text()

    assert '_model_type == "hy_v3"' in src
    assert "import jang_tools.hy3" in src
    assert src.index('_model_type == "hy_v3"') < src.index("_load_skeleton(")


def test_importing_hy3_registers_mlx_lm_model_module():
    """load_jangtq's Hy3 branch imports jang_tools.hy3 before mlx_lm load.

    That import must be enough to make `model_type=hy_v3` resolvable by
    mlx_lm.utils.load_model; a source-string test can miss this runtime side
    effect.
    """
    sys.modules.pop("mlx_lm.models.hy_v3", None)

    import jang_tools.hy3

    importlib.reload(jang_tools.hy3)
    module = importlib.import_module("mlx_lm.models.hy_v3")

    assert module.Model.__name__ == "Model"


def test_hy3_converter_stamps_long_output_safe_chat_defaults():
    src = (Path(__file__).parents[1] / "jang_tools/convert_hy3_jangtq.py").read_text()

    assert '"sampling_defaults"' in src
    assert '"temperature": 0.0' in src
    assert '"top_p": 1.0' in src
    assert '"top_k": 0' in src
    assert '"max_new_tokens": 2048' in src


def test_hy3_converter_patches_generation_config_after_copy():
    src = (Path(__file__).parents[1] / "jang_tools/convert_hy3_jangtq.py").read_text()

    assert "HY3_CHAT_SAMPLING_DEFAULTS" in src
    assert "gen_cfg_path = OUT / \"generation_config.json\"" in src
    assert "gen_cfg.update(HY3_GENERATION_CONFIG_OVERRIDES)" in src

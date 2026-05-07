from jang_tools.capabilities import build_capabilities
from jang_tools.convert_zaya_common import CAPABILITIES
from pathlib import Path


def test_zaya_converter_stamps_tools_only_thinking_disabled():
    assert CAPABILITIES["family"] == "zaya"
    assert CAPABILITIES["tool_parser"] == "zaya_xml"
    assert CAPABILITIES["reasoning_parser"] is None
    assert CAPABILITIES["think_in_template"] is False
    assert CAPABILITIES["supports_thinking"] is False
    assert CAPABILITIES["cache_type"] == "hybrid"


def test_zaya_capability_builder_matches_converter_contract():
    caps = build_capabilities(
        {
            "source_model": {
                "architecture": "zaya",
            },
        },
        {"model_type": "zaya"},
    )

    assert caps is not None
    assert caps["family"] == "zaya"
    assert caps["tool_parser"] == "zaya_xml"
    assert caps["reasoning_parser"] is None
    assert caps["think_in_template"] is False
    assert caps["supports_thinking"] is False
    assert caps["cache_type"] == "hybrid"


def test_zaya_converter_console_scripts_are_registered():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text()

    assert 'jang-convert-zaya-jangtq = "jang_tools.convert_zaya_jangtq:main"' in text
    assert 'jang-convert-zaya-mxfp4 = "jang_tools.convert_zaya_mxfp4:main"' in text

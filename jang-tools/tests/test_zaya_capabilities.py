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


# Regression pins (2026-05-09): two unrelated bugs in capabilities.py.
#
# 1. FAMILY_MAP["deepseek_v4"] previously stamped tool_parser="deepseek",
#    but DSV4 emits its own DSML tool-call format. Plain
#    deepseek_tool_parser cannot parse DSML.
# 2. valid_tool did not include "dsml" or "zaya_xml" — verify_directory
#    rejected legitimate freshly-stamped DSV4 / Zaya bundles.


def test_deepseek_v4_family_map_uses_dsml_not_deepseek():
    """DSV4 must map to dsml tool parser, not deepseek."""
    from jang_tools.capabilities import FAMILY_MAP

    family, reasoning, tool, think_in_template, cache_type = FAMILY_MAP["deepseek_v4"]
    assert tool == "dsml", (
        f"FAMILY_MAP['deepseek_v4'] tool_parser is {tool!r}, expected 'dsml'. "
        "DSV4 emits DSML tool-call format; plain deepseek_tool_parser cannot "
        "parse it."
    )
    assert family == "deepseek_v4"
    assert reasoning == "deepseek_r1"
    assert think_in_template is True
    assert cache_type == "mla"


def test_dsv4_capability_builder_stamps_dsml_tool_parser():
    """End-to-end: build_capabilities for DSV4 produces tool_parser='dsml'."""
    caps = build_capabilities(
        {"source_model": {"architecture": "deepseek_v4"}},
        {"model_type": "deepseek_v4"},
    )
    assert caps is not None
    assert caps["family"] == "deepseek_v4"
    assert caps["tool_parser"] == "dsml"
    assert caps["reasoning_parser"] == "deepseek_r1"


def test_validate_capabilities_accepts_dsml_and_zaya_xml():
    """verify_directory's valid_tool must include both DSML and Zaya XML.

    Inspect the function source since valid_tool is a function-local set
    with no public accessor.
    """
    import inspect

    from jang_tools import capabilities as cap_module

    source = inspect.getsource(cap_module.verify_directory)
    assert '"dsml"' in source, (
        "verify_directory must accept dsml in valid_tool — DSV4 bundles "
        "stamp tool_parser='dsml' and the validator was rejecting them"
    )
    assert '"zaya_xml"' in source, (
        "verify_directory must accept zaya_xml in valid_tool — Zaya bundles "
        "stamp tool_parser='zaya_xml' and the validator was rejecting them"
    )

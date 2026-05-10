"""Hy3 JANGTQ1 profile contract checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_hy3_converter_declares_jangtq1_and_shape_guard():
    src = (ROOT / "jang_tools" / "convert_hy3_jangtq.py").read_text(
        encoding="utf-8"
    )

    assert '"JANGTQ1": 1' in src
    assert "_validate_tq_packing_shape" in src
    assert "moe_intermediate_size" in src
    assert "hidden_size" in src


def test_hy3_fit_estimator_knows_jangtq1():
    path = ROOT / "examples" / "mtp" / "estimate_jangtq_fit.py"
    spec = importlib.util.spec_from_file_location("estimate_jangtq_fit", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.profile_bits("JANGTQ1") == 1.0
    assert mod.profile_bits("jangtq1") == 1.0

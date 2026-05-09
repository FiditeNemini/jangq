from pathlib import Path


def test_jangtq_switchglu_fusion_failures_are_hard_errors():
    source = (Path(__file__).resolve().parents[1] / "jang_tools/load_jangtq.py").read_text()

    assert "JANGTQ SwitchGLU fusion failed" in source
    assert "DSV4 JANGTQ SwitchGLU fusion failed" in source
    assert "refusing to continue with " in source
    assert "stock SwitchGLU" in source
    assert "SwitchGLU fusion skipped" not in source

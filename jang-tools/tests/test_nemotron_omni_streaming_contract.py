"""Source contracts for real-time Nemotron Omni decode callbacks."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _function(source_path: Path, class_name: str, function_name: str) -> ast.FunctionDef:
    tree = ast.parse(source_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise AssertionError(f"missing {class_name}.{function_name} in {source_path}")


def _arg_names(function: ast.FunctionDef) -> set[str]:
    return {argument.arg for argument in function.args.args}


def test_stage1_omni_session_threads_live_decode_callback_and_usage():
    path = ROOT / "jang_tools" / "nemotron_omni_session.py"
    turn = _function(path, "OmniSession", "turn")
    decode = _function(path, "OmniSession", "_decode_turn")
    source = path.read_text()

    assert "token_callback" in _arg_names(turn)
    assert "token_callback" in _arg_names(decode)
    assert "token_callback=token_callback" in source
    assert "detokenizer.last_segment" in source
    assert "self._last_prompt_tokens" in source
    assert "self._last_completion_tokens" in source
    assert "self._last_finish_reason" in source


def test_stage2_omni_model_emits_from_native_decode_loop():
    path = ROOT / "jang_tools" / "nemotron_omni" / "model.py"
    turn = _function(path, "NemotronHOmni", "turn")
    source = path.read_text()

    assert "token_callback" in _arg_names(turn)
    assert "detokenizer.last_segment" in source
    assert "token_callback(token_id, segment)" in source
    assert "self._last_prompt_tokens" in source
    assert "self._last_completion_tokens" in source
    assert "self._last_finish_reason" in source

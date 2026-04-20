"""Unit tests for jang_tools.jangspec.manifest."""

from pathlib import Path

import pytest

from jang_tools.jangspec.manifest import Manifest, load_manifest, write_manifest


def test_manifest_roundtrip(tmp_path: Path):
    m = Manifest(
        bundle_version=1,
        source_jang="Gemma-4-26B-A4B-it-JANG_4M",
        source_jang_dir="/Users/eric/jang/models/Gemma-4-26B-A4B-it-JANG_4M",
        target_arch="gemma4_moe",
        n_layers=48,
        n_experts_per_layer=128,
        target_top_k=8,
        tokenizer_hash="sha256:abc123",
        hot_core_tensors=["model.embed_tokens.weight", "layers.0.self_attn.q_proj.weight"],
        expert_tensor_names=["layers.N.switch_mlp.gate_proj", "layers.N.switch_mlp.up_proj", "layers.N.switch_mlp.down_proj"],
        n_experts_total=48 * 128,
        hot_core_bytes=12_000_000_000,
        expert_bytes=40_000_000_000,
        has_draft=False,
        has_router_prior=False,
    )
    path = tmp_path / "jangspec.json"
    write_manifest(path, m)

    loaded = load_manifest(path)
    assert loaded == m


def test_manifest_rejects_wrong_version(tmp_path: Path):
    path = tmp_path / "jangspec.json"
    path.write_text('{"bundle_version": 99}')
    with pytest.raises(ValueError, match="bundle_version"):
        load_manifest(path)


# M148 (iter 70): load_manifest error-path hardening. Every failure mode
# must produce a ValueError with the bundle path in the message — matching
# iter-43 M120's pattern on inspect_source/recommend and iter-69 M147's
# pattern on AppSettings.load.


def test_manifest_rejects_malformed_json(tmp_path: Path):
    path = tmp_path / "jangspec.json"
    path.write_text("{ this is not valid json")
    with pytest.raises(ValueError) as excinfo:
        load_manifest(path)
    msg = str(excinfo.value)
    assert "not valid JSON" in msg
    assert str(path) in msg, f"error must include path, got: {msg}"


def test_manifest_rejects_non_dict_root(tmp_path: Path):
    path = tmp_path / "jangspec.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(ValueError) as excinfo:
        load_manifest(path)
    msg = str(excinfo.value)
    assert "expected a JSON object" in msg
    assert str(path) in msg


def test_manifest_rejects_missing_required_field(tmp_path: Path):
    # Schema migration simulation: missing required `source_jang` field.
    import json as _json
    path = tmp_path / "jangspec.json"
    path.write_text(_json.dumps({
        "bundle_version": 1,
        # no other fields — will trigger Manifest(**data) TypeError.
    }))
    with pytest.raises(ValueError) as excinfo:
        load_manifest(path)
    msg = str(excinfo.value)
    assert "schema validation" in msg
    assert "different jang-tools version" in msg
    assert str(path) in msg


def test_manifest_missing_file_raises_value_error(tmp_path: Path):
    # OSError should be converted to ValueError with path context.
    path = tmp_path / "nonexistent.json"
    with pytest.raises(ValueError) as excinfo:
        load_manifest(path)
    msg = str(excinfo.value)
    assert "could not read manifest" in msg
    assert str(path) in msg

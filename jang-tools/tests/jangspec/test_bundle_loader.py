"""Round-trip tests for jang_tools.jangspec.bundle_loader."""

from pathlib import Path

import numpy as np
import pytest

from jang_tools.jangspec.bundle_loader import load_weights_from_bundle


def test_bundle_loader_returns_hot_core_tensors(jangspec_fixture_model: Path, tmp_path: Path):
    # Use the conftest fixture (Gemma-4-26B-A4B-it-JANG_4M) to build a
    # bundle, then verify the loader returns at least one hot-core tensor.
    from jang_tools.jangspec.builder import JangSpecBuilder

    out = tmp_path / "fx.jangspec"
    JangSpecBuilder(source_dir=jangspec_fixture_model, out_dir=out, write_streaming=True).build()

    weights = load_weights_from_bundle(out)
    # Hot core has embeddings, attention, norms, lm_head — pick one we know
    # must be present in any Gemma-4 JANG bundle.
    assert any(k.endswith("embed_tokens.weight") for k in weights), (
        "expected embed_tokens.weight in bundle weights"
    )
    assert any(k.endswith("self_attn.q_proj.weight") for k in weights), (
        "expected self_attn.q_proj.weight in bundle weights"
    )


def test_bundle_loader_reconstructs_expert_3d_stacks(
    jangspec_fixture_model: Path, tmp_path: Path
):
    from jang_tools.jangspec.builder import JangSpecBuilder
    from jang_tools.jangspec.manifest import load_manifest
    from jang_tools.jangspec import format as fmt

    out = tmp_path / "fx.jangspec"
    JangSpecBuilder(source_dir=jangspec_fixture_model, out_dir=out, write_streaming=True).build()

    weights = load_weights_from_bundle(out)
    manifest = load_manifest(out / fmt.MANIFEST_FILENAME)

    # For every expert base name in the manifest, the loader must emit
    # a 3D stacked tensor with leading dim == n_experts_per_layer.
    for base in manifest.expert_tensor_names:
        weight_key = f"{base}.weight"
        scales_key = f"{base}.scales"
        biases_key = f"{base}.biases"

        assert weight_key in weights, f"missing {weight_key}"
        assert scales_key in weights, f"missing {scales_key}"
        assert biases_key in weights, f"missing {biases_key}"

        wt = weights[weight_key]
        assert wt.ndim == 3, f"{weight_key} should be 3D, got {wt.shape}"
        assert wt.shape[0] == manifest.n_experts_per_layer, (
            f"{weight_key} leading dim {wt.shape[0]} != "
            f"n_experts_per_layer {manifest.n_experts_per_layer}"
        )


def test_bundle_loader_byte_parity_against_source(
    jangspec_fixture_model: Path, tmp_path: Path
):
    """The reconstructed expert 3D stacks should be byte-identical to slicing
    the source safetensors directly."""
    import json
    from safetensors import safe_open
    from jang_tools.jangspec.builder import JangSpecBuilder

    out = tmp_path / "fx.jangspec"
    JangSpecBuilder(source_dir=jangspec_fixture_model, out_dir=out, write_streaming=True).build()
    weights = load_weights_from_bundle(out)

    # Pick layer 0 and compare the gate_proj 3D tensor end-to-end.
    st_index = json.loads(
        (jangspec_fixture_model / "model.safetensors.index.json").read_text()
    )["weight_map"]
    base = next(
        b for b in (
            "model.language_model.layers.0.switch_mlp.gate_proj",
            "model.layers.0.switch_mlp.gate_proj",
            "language_model.model.layers.0.switch_mlp.gate_proj",
        )
        if f"{b}.weight" in st_index
    )
    src_shard = jangspec_fixture_model / st_index[f"{base}.weight"]
    with safe_open(src_shard, framework="numpy", device="cpu") as f:
        src_qweight = f.get_tensor(f"{base}.weight")
        src_scales = f.get_tensor(f"{base}.scales")
        src_biases = f.get_tensor(f"{base}.biases")

    rec_qweight = weights[f"{base}.weight"]
    rec_scales = weights[f"{base}.scales"]
    rec_biases = weights[f"{base}.biases"]

    # The bundle loader returns mx.array for MLX consumption; convert for
    # numpy equality comparison.
    import mlx.core as mx
    np.testing.assert_array_equal(np.array(rec_qweight, copy=False), src_qweight)
    np.testing.assert_array_equal(np.array(rec_scales, copy=False), src_scales)
    np.testing.assert_array_equal(np.array(rec_biases, copy=False), src_biases)

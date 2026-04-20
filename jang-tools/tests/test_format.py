"""Tests for JANG format writer/reader — end-to-end roundtrip."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from jang_tools.quantize import quantize_tensor
from jang_tools.format.writer import write_jang_model
from jang_tools.format.reader import load_jang_model, is_jang_model


class TestFormatRoundtrip:
    """Write a JANG model, read it back, verify everything matches."""

    def test_write_read_roundtrip(self):
        """Full roundtrip: quantize → write → read → verify."""
        rng = np.random.default_rng(42)

        # Create fake weight tensors
        weights = {
            "layers.0.self_attn.q_proj": rng.standard_normal((256, 256)).astype(np.float32) * 0.02,
            "layers.0.self_attn.k_proj": rng.standard_normal((256, 256)).astype(np.float32) * 0.02,
            "layers.0.mlp.gate_proj": rng.standard_normal((512, 256)).astype(np.float32) * 0.02,
        }

        # Quantize each tensor — different bit widths per tensor (tier-based)
        quantized = {}
        tensor_bits = {
            "layers.0.self_attn.q_proj": 6,   # CRITICAL
            "layers.0.self_attn.k_proj": 6,   # CRITICAL
            "layers.0.mlp.gate_proj": 2,       # COMPRESS
        }
        for name, w in weights.items():
            n_blocks = (w.size) // 64
            bits = tensor_bits[name]
            bit_alloc = np.full(n_blocks, bits, dtype=np.uint8)

            qt = quantize_tensor(w, bit_alloc, block_size=64)
            quantized[name] = qt

        # Write to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "test-model-JANG-3bit"

            model_config = {"hidden_size": 256, "num_layers": 1}
            jang_config = {
                "quantization": {
                    "method": "jang-importance",
                    "target_bits": 3.0,
                    "actual_bits": 3.1,
                    "block_size": 64,
                    "scoring_method": "awq+hessian",
                    "bit_widths_used": [2, 4, 6],
                },
                "source_model": {
                    "name": "test-model",
                    "dtype": "float32",
                    "parameters": "1M",
                },
            }

            write_jang_model(
                output_dir=model_dir,
                quantized_tensors=quantized,
                model_config=model_config,
                jang_config=jang_config,
            )

            # Verify files exist
            assert is_jang_model(model_dir)
            assert (model_dir / "jang_config.json").exists()
            assert (model_dir / "config.json").exists()
            assert (model_dir / "model.jang.index.json").exists()

            # Read back
            model = load_jang_model(model_dir)
            assert model.target_bits == 3.0
            assert model.source_model == "test-model"

            # Verify weight names
            weight_names = model.weight_names
            assert len(weight_names) == 3
            assert "layers.0.self_attn.q_proj" in weight_names

            # Verify tensor data matches
            for name in weight_names:
                original = quantized[name]
                loaded = model.get_quantized_tensor(name)

                np.testing.assert_array_equal(loaded.qweight, original.qweight)
                np.testing.assert_array_equal(loaded.scales, original.scales)
                np.testing.assert_array_equal(loaded.biases, original.biases)
                assert loaded.bits == original.bits

            # Verify summary
            summary = model.summary()
            assert summary["total_weight_names"] == 3
            assert summary["total_blocks"] > 0
            assert "2-bit" in summary["histogram"]
            assert "6-bit" in summary["histogram"]
            assert "6-bit" in summary["histogram"]

    def test_not_mxq_model(self):
        """Non-JANG directory should be detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert not is_jang_model(tmpdir)

    def test_invalid_format_rejected(self):
        """Model with wrong format field should be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"format": "not-mxq", "format_version": "1.0"}
            (Path(tmpdir) / "jang_config.json").write_text(json.dumps(config))

            with pytest.raises(ValueError, match="Invalid format"):
                load_jang_model(tmpdir)


class TestFormatReaderErrorDiagnostics:
    """M149 (iter 71): every JSON read in load_jang_model must produce an
    actionable ValueError with the file path + purpose in the message.
    Same template as M120 (inspect_source), M147 (AppSettings.load),
    M148 (jangspec.manifest)."""

    def test_malformed_jang_config_raises_with_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "jang_config.json"
            cfg_path.write_text("{ this is not json")
            with pytest.raises(ValueError) as excinfo:
                load_jang_model(tmpdir)
            msg = str(excinfo.value)
            assert "not valid JSON" in msg
            assert str(cfg_path) in msg, f"error must include path, got: {msg}"
            assert "JANG config" in msg

    def test_non_dict_root_jang_config_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "jang_config.json"
            cfg_path.write_text("[1, 2, 3]")
            with pytest.raises(ValueError) as excinfo:
                load_jang_model(tmpdir)
            assert "expected a JSON object" in str(excinfo.value)

    def test_malformed_model_config_raises_with_purpose(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # jang_config OK but config.json broken.
            (Path(tmpdir) / "jang_config.json").write_text(
                json.dumps({"format": "jang", "format_version": "1.0"})
            )
            mc_path = Path(tmpdir) / "config.json"
            mc_path.write_text("not json at all")
            with pytest.raises(ValueError) as excinfo:
                load_jang_model(tmpdir)
            msg = str(excinfo.value)
            assert "model config" in msg, f"purpose must be in the error: {msg}"
            assert str(mc_path) in msg

    def test_malformed_shard_index_raises_with_purpose(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "jang_config.json").write_text(
                json.dumps({"format": "jang", "format_version": "1.0"})
            )
            idx_path = Path(tmpdir) / "model.jang.index.json"
            idx_path.write_text("{ broken")
            with pytest.raises(ValueError) as excinfo:
                load_jang_model(tmpdir)
            msg = str(excinfo.value)
            assert "shard index" in msg
            assert str(idx_path) in msg

    def test_shard_index_missing_weight_map_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "jang_config.json").write_text(
                json.dumps({"format": "jang", "format_version": "1.0"})
            )
            # Index is valid JSON but missing weight_map.
            (Path(tmpdir) / "model.jang.index.json").write_text(
                json.dumps({"metadata": {"total_size": 0}})
            )
            with pytest.raises(ValueError) as excinfo:
                load_jang_model(tmpdir)
            msg = str(excinfo.value)
            assert "weight_map" in msg
            assert "corrupted" in msg or "incompatible" in msg

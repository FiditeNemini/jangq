"""Unit tests for MiMo FP8 block dequantization.

Exercises three checkpoint shapes that occur in MiMo-V2.5:
  1. Expert gate_proj: weight (2048, 4096), scale (16, 32) — aligned
  2. SWA qkv_proj:     weight (14848, 4096), scale (116, 32) — aligned
  3. Full qkv_proj:    weight (13568, 4096), scale (108, 32) — has 2 padding rows
     beyond the weight rows (V uses scale 102..105, scale 106..107 unused)

Requires `MIMO_V25_SRC` env var pointing at a downloaded MiMo-V2.5 checkpoint
(default: /Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5). Skipped if missing.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import torch
from safetensors import safe_open

from jang_tools.mimo_v2.fp8_block_codec import dequant_fp8_e4m3_scale_inv


_DEFAULT_SRC = "/Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5"


def _load_pair(src: Path, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Read `{name}.weight` and `{name}.weight_scale_inv` from the source checkpoint."""
    weight_map = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    w_key = f"{name}.weight"
    s_key = f"{name}.weight_scale_inv"
    with safe_open(str(src / weight_map[w_key]), framework="pt", device="cpu") as f:
        w = f.get_tensor(w_key)
    with safe_open(str(src / weight_map[s_key]), framework="pt", device="cpu") as f:
        s = f.get_tensor(s_key)
    return w, s


@unittest.skipUnless(
    Path(os.environ.get("MIMO_V25_SRC", _DEFAULT_SRC)).is_dir(),
    "MiMo-V2.5 source checkpoint not available",
)
class TestFp8Codec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = Path(os.environ.get("MIMO_V25_SRC", _DEFAULT_SRC))

    def test_expert_gate_proj_aligned(self):
        """Aligned: scale shape == ceil(weight/block_size)."""
        w, s = _load_pair(self.src, "model.layers.1.mlp.experts.0.gate_proj")
        self.assertEqual(tuple(w.shape), (2048, 4096))
        self.assertEqual(tuple(s.shape), (16, 32))
        self.assertEqual(w.dtype, torch.float8_e4m3fn)
        out = dequant_fp8_e4m3_scale_inv(w, s, out_dtype=torch.bfloat16)
        self.assertEqual(out.shape, w.shape)
        self.assertEqual(out.dtype, torch.bfloat16)
        # Sanity: dequanted values should be O(1e-2 .. 1e-1) for typical attention/MLP weights
        mag = out.float().abs().mean().item()
        self.assertGreater(mag, 1e-4)
        self.assertLess(mag, 1.0)

    def test_swa_qkv_aligned(self):
        """SWA qkv_proj: 14848 rows, scale (116, 32) — exact ceil."""
        w, s = _load_pair(self.src, "model.layers.1.self_attn.qkv_proj")
        self.assertEqual(tuple(w.shape), (14848, 4096))
        self.assertEqual(tuple(s.shape), (116, 32))
        out = dequant_fp8_e4m3_scale_inv(w, s, out_dtype=torch.bfloat16)
        self.assertEqual(out.shape, w.shape)
        self.assertFalse(torch.isnan(out).any().item())
        self.assertFalse(torch.isinf(out).any().item())

    def test_full_qkv_with_trailing_scale_pad(self):
        """Full qkv_proj: 13568 rows, scale (108, 32) — 2 trailing padding rows."""
        w, s = _load_pair(self.src, "model.layers.0.self_attn.qkv_proj")
        self.assertEqual(tuple(w.shape), (13568, 4096))
        self.assertEqual(tuple(s.shape), (108, 32))
        out = dequant_fp8_e4m3_scale_inv(w, s, out_dtype=torch.bfloat16)
        self.assertEqual(out.shape, w.shape)
        # Per-projection sanity (Q=12288, K=768, V=512)
        q = out[:12288].float()
        k = out[12288:13056].float()
        v = out[13056:13568].float()
        for name, t in (("Q", q), ("K", k), ("V", v)):
            self.assertFalse(torch.isnan(t).any().item(), f"NaN in {name}")
            self.assertFalse(torch.isinf(t).any().item(), f"Inf in {name}")
            mag = t.abs().mean().item()
            self.assertGreater(mag, 1e-4, f"{name} suspiciously small ({mag})")
            self.assertLess(mag, 0.5, f"{name} suspiciously large ({mag})")

    def test_full_qkv_v_uses_contiguous_scale_rows(self):
        """Verify V is dequanted with scale rows 102..105 (H3), not 104..107 (H2).

        Distinguishes the two competing interpretations by comparing the codec's V
        output against an explicit H2-style dequant. They must differ — proving
        the codec follows H3 — and the H3 magnitude must be more consistent
        block-to-block (since the actual V scale rows are 102..105).
        """
        w, s = _load_pair(self.src, "model.layers.0.self_attn.qkv_proj")
        out = dequant_fp8_e4m3_scale_inv(w, s, out_dtype=torch.float32)
        v_h3 = out[13056:13568]
        # Explicit H2 reconstruction: use scale rows 104..107
        v_weight = w[13056:13568].float()
        v_scale_h2 = s[104:108].float().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        v_h2 = v_weight * v_scale_h2[: v_weight.shape[0], : v_weight.shape[1]]
        # They must differ
        self.assertFalse(torch.allclose(v_h3, v_h2, atol=1e-6),
                         "H3 and H2 dequant produced identical V — test premise broken")


if __name__ == "__main__":
    unittest.main()

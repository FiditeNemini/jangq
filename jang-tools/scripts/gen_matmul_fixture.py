#!/usr/bin/env python3
"""
Generate a deterministic 4-bit quantized matmul fixture for Plan 4's Swift test.

Produces a safetensors file containing:
    W.weight   uint32   (out, in/8)     - packed 4-bit quantized weights
    W.scales   float16  (out, n_groups) - per-group scale
    W.biases   float16  (out, n_groups) - per-group bias
    x          float16  (in,)           - input vector
    y_ref      float32  (out,)          - reference output W_dq @ x

Plus a side-car fixture_info.json with shapes and metadata.
"""

import json
from pathlib import Path

import numpy as np
import mlx.core as mx
from safetensors.numpy import save_file

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "jang-runtime" / "Tests" / "JANGCoreMetalTests" / "fixtures"
SEED = 0xBADC0FFEE
BITS = 4
GROUP_SIZE = 64
IN_FEATURES = 64
OUT_FEATURES = 128


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    W_full = rng.standard_normal((OUT_FEATURES, IN_FEATURES)).astype(np.float32)
    x_f16 = rng.standard_normal((IN_FEATURES,)).astype(np.float16)

    W_mx = mx.array(W_full)
    q, s, b = mx.quantize(W_mx, bits=BITS, group_size=GROUP_SIZE)

    q_np = np.array(q, dtype=np.uint32, copy=True)
    s_np = np.array(s, dtype=np.float16, copy=True)
    b_np = np.array(b, dtype=np.float16, copy=True)

    W_dq = mx.dequantize(q, s, b, group_size=GROUP_SIZE, bits=BITS)
    W_dq_np = np.array(W_dq, dtype=np.float32, copy=True)
    y_ref = (W_dq_np.astype(np.float32) @ x_f16.astype(np.float32)).astype(np.float32)

    assert q_np.shape == (OUT_FEATURES, IN_FEATURES * BITS // 32), q_np.shape
    assert s_np.shape == (OUT_FEATURES, IN_FEATURES // GROUP_SIZE), s_np.shape
    assert b_np.shape == (OUT_FEATURES, IN_FEATURES // GROUP_SIZE), b_np.shape

    save_file(
        {
            "W.weight": q_np,
            "W.scales": s_np,
            "W.biases": b_np,
            "x": x_f16,
            "y_ref": y_ref,
        },
        str(OUT_DIR / "matmul_4bit_64x128.safetensors"),
    )

    info = {
        "seed": SEED,
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "in_features": IN_FEATURES,
        "out_features": OUT_FEATURES,
        "qweight_shape": list(q_np.shape),
        "scales_shape": list(s_np.shape),
        "biases_shape": list(b_np.shape),
        "x_shape": list(x_f16.shape),
        "y_ref_shape": list(y_ref.shape),
        "mlx_pack_convention": "LSB-first 4-bit: position k in [0..7] at bit k*4 of each uint32",
        "dequant_formula": "val = q_int * scale + bias, per-group",
    }
    (OUT_DIR / "fixture_info.json").write_text(json.dumps(info, indent=2) + "\n")

    print(f"  wrote {OUT_DIR / 'matmul_4bit_64x128.safetensors'}")
    print(f"  qweight {q_np.shape} scales {s_np.shape} biases {b_np.shape}")
    print(f"  x {x_f16.shape}  y_ref {y_ref.shape}  y_ref[:4]={y_ref[:4].tolist()}")


if __name__ == "__main__":
    main()

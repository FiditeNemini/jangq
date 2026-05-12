#!/usr/bin/env python3
"""Probe the experimental TurboQuant MPP dense lane.

This script is intentionally a proof harness, not a production benchmark. It
compares the current fused TurboQuant kernel against the opt-in MPP dense lane
that materializes a dense half matrix and calls MPP tensor_ops matmul2d.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from jang_tools.turboquant.codebook import compute_codebook
from jang_tools.turboquant.mpp_dense_kernel import (
    mpp_tensorops_available,
    tq_matmul_mpp_dense,
)
from jang_tools.turboquant.mpp_nax_kernel import (
    mpp_nax_tensorops_available,
    tq_matmul_mpp_nax,
)
from jang_tools.turboquant.pipeline import pack_bits
from jang_tools.turboquant.rotation import generate_random_signs
from jang_tools.turboquant.tq_kernel import tq_matmul


def _quantize_rows(
    weight: np.ndarray, bits: int
) -> tuple[mx.array, mx.array, mx.array]:
    out_features, in_features = weight.shape
    codebook = mx.array(compute_codebook(in_features, bits), dtype=mx.float32)
    w = mx.array(weight.astype(np.float32))
    norms = mx.sqrt(mx.sum(w * w, axis=1, keepdims=True))
    w_normed = w / mx.maximum(norms, mx.array(1e-8))
    boundaries = (codebook[:-1] + codebook[1:]) / 2.0
    indices = mx.zeros(w_normed.shape, dtype=mx.uint8)
    for boundary in boundaries:
        indices = indices + (w_normed > boundary).astype(mx.uint8)
    vals_per_u32 = 32 // bits
    pad = (-in_features) % vals_per_u32
    if pad:
        indices = mx.pad(indices, [(0, 0), (0, pad)])
    packed = pack_bits(indices.reshape(-1), bits).reshape(out_features, -1)
    mx.eval(packed, norms)
    return packed, norms.squeeze(-1).astype(mx.float16), codebook


def _time_ms(fn, repeat: int) -> float:
    for _ in range(2):
        mx.eval(fn())
    start = time.perf_counter()
    for _ in range(repeat):
        mx.eval(fn())
    return (time.perf_counter() - start) * 1000.0 / repeat


def run_case(batch: int, in_features: int, out_features: int, bits: int, repeat: int):
    rng = np.random.default_rng(1000 + batch + in_features + out_features + bits)
    x = mx.array(rng.standard_normal((batch, in_features)).astype(np.float32))
    signs = mx.array(generate_random_signs(in_features, seed=123), dtype=mx.float32)
    w_rot = rng.standard_normal((out_features, in_features)).astype(np.float32)
    packed, norms, codebook = _quantize_rows(w_rot, bits)

    current = tq_matmul(x, packed, norms, codebook, signs, in_features, bits)
    mpp_dense = tq_matmul_mpp_dense(x, packed, norms, codebook, signs, in_features, bits)
    mpp_nax = tq_matmul_mpp_nax(x, packed, norms, codebook, signs, in_features, bits)
    mx.eval(current, mpp_dense, mpp_nax)
    dense_max_abs_err = float(
        mx.max(mx.abs(current.astype(mx.float32) - mpp_dense.astype(mx.float32))).item()
    )
    nax_max_abs_err = float(
        mx.max(mx.abs(current.astype(mx.float32) - mpp_nax.astype(mx.float32))).item()
    )

    current_ms = _time_ms(
        lambda: tq_matmul(x, packed, norms, codebook, signs, in_features, bits),
        repeat,
    )
    mpp_dense_ms = _time_ms(
        lambda: tq_matmul_mpp_dense(x, packed, norms, codebook, signs, in_features, bits),
        repeat,
    )
    mpp_nax_ms = _time_ms(
        lambda: tq_matmul_mpp_nax(x, packed, norms, codebook, signs, in_features, bits),
        repeat,
    )
    return {
        "batch": batch,
        "in_features": in_features,
        "out_features": out_features,
        "bits": bits,
        "mpp_dense_max_abs_err": dense_max_abs_err,
        "mpp_nax_max_abs_err": nax_max_abs_err,
        "current_ms": current_ms,
        "mpp_dense_ms": mpp_dense_ms,
        "mpp_dense_speedup": current_ms / mpp_dense_ms if mpp_dense_ms else None,
        "mpp_nax_ms": mpp_nax_ms,
        "mpp_nax_speedup": current_ms / mpp_nax_ms if mpp_nax_ms else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--case",
        action="append",
        metavar="BATCH,IN,OUT,BITS",
        help="Add a case, e.g. 1,1024,1024,4. Can be repeated.",
    )
    args = parser.parse_args()

    cases = args.case or [
        "1,1024,1024,4",
        "16,1024,1024,4",
        "64,1024,1024,4",
        "16,2048,2048,4",
    ]
    result = {
        "mpp_tensorops_available": mpp_tensorops_available(),
        "mpp_nax_tensorops_available": mpp_nax_tensorops_available(),
        "production_default_changed": False,
        "note": (
            "MPP dense materializes dense half weights and is only a proof lane. "
            "MPP NAX unpacks JANGTQ codebook values directly into cooperative "
            "TensorOps fragments and is the candidate production lane."
        ),
        "cases": [],
    }
    if result["mpp_tensorops_available"] and result["mpp_nax_tensorops_available"]:
        for raw in cases:
            batch, in_features, out_features, bits = [int(part) for part in raw.split(",")]
            result["cases"].append(
                run_case(batch, in_features, out_features, bits, args.repeat)
            )

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

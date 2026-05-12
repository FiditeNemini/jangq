#!/usr/bin/env python3
"""Probe the full JANGTQ MoE expert cluster with and without MPP/NAX.

This measures the synthetic prefill-shaped path that matters for whole-model
PP speed:

    _gather_sort -> fused gate/up/SwiGLU -> down gather -> _scatter_unsort

It intentionally uses the same public TurboQuant kernel functions that the
JANGTQ loader patches into SwitchGLU. It does not claim model quality; it is a
kernel-cluster timing and parity harness.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

from jang_tools.turboquant.codebook import compute_codebook
from jang_tools.turboquant.fused_gate_up_kernel import fused_gate_up_swiglu_matmul
from jang_tools.turboquant.gather_tq_kernel import gather_tq_matmul
from jang_tools.turboquant.hadamard_kernel import hadamard_rotate_metal
from jang_tools.turboquant.mpp_nax_kernel import mpp_nax_tensorops_available
from jang_tools.turboquant.mpp_nax_kernel import (
    build_sorted_group_tiles,
    fused_gate_up_swiglu_mpp_nax_grouped_from_rot_with_tiles,
    gather_tq_matmul_mpp_nax_grouped_from_rot_with_tiles,
)
from jang_tools.turboquant.pipeline import pack_bits
from jang_tools.turboquant.rotation import generate_random_signs


@contextmanager
def _env(name: str, value: str | None):
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _quantize_rows(weight: np.ndarray, bits: int) -> tuple[mx.array, mx.array, mx.array]:
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


def _expert_weights(
    rng: np.random.Generator,
    n_experts: int,
    out_features: int,
    in_features: int,
    bits: int,
) -> tuple[mx.array, mx.array, mx.array]:
    packed_rows = []
    norm_rows = []
    codebook = None
    for _ in range(n_experts):
        packed, norms, codebook = _quantize_rows(
            rng.standard_normal((out_features, in_features)).astype(np.float32),
            bits,
        )
        packed_rows.append(packed)
        norm_rows.append(norms)
    assert codebook is not None
    return mx.stack(packed_rows, axis=0), mx.stack(norm_rows, axis=0), codebook


def _time_ms(fn, repeat: int) -> float:
    for _ in range(2):
        mx.eval(fn())
    start = time.perf_counter()
    for _ in range(repeat):
        mx.eval(fn())
    return (time.perf_counter() - start) * 1000.0 / repeat


def run_case(
    tokens: int,
    top_k: int,
    hidden: int,
    intermediate: int,
    n_experts: int,
    bits: int,
    repeat: int,
    nax_mode: str,
    precompute_tiles: bool,
) -> dict:
    rng = np.random.default_rng(10_000 + tokens + top_k + hidden + intermediate)
    x = mx.array(rng.standard_normal((tokens, hidden)).astype(np.float32))
    indices = mx.array(
        rng.integers(0, n_experts, size=(tokens, top_k), dtype=np.uint32)
    )
    gate_packed, gate_norms, gate_codebook = _expert_weights(
        rng, n_experts, intermediate, hidden, bits
    )
    up_packed, up_norms, _ = _expert_weights(
        rng, n_experts, intermediate, hidden, bits
    )
    down_packed, down_norms, down_codebook = _expert_weights(
        rng, n_experts, hidden, intermediate, bits
    )
    gate_signs = mx.array(generate_random_signs(hidden, seed=17), dtype=mx.float32)
    down_signs = mx.array(
        generate_random_signs(intermediate, seed=19), dtype=mx.float32
    )

    def cluster() -> mx.array:
        x_exp = mx.expand_dims(x, (-2, -3))
        idx = indices
        inv_order = None
        do_sort = indices.size >= 64
        if do_sort:
            x_exp, idx, inv_order = _gather_sort(x_exp, indices)
        x_act = fused_gate_up_swiglu_matmul(
            x_exp,
            gate_packed,
            gate_norms,
            up_packed,
            up_norms,
            gate_codebook,
            gate_signs,
            idx,
            bits=bits,
        )
        x_out = gather_tq_matmul(
            x_act,
            down_packed,
            down_norms,
            down_codebook,
            down_signs,
            idx,
            bits=bits,
            sorted_indices=do_sort,
        )
        if do_sort:
            x_out = _scatter_unsort(x_out, inv_order, indices.shape)
        return x_out.squeeze(-2)

    precomputed = None
    if precompute_tiles:
        x_exp = mx.expand_dims(x, (-2, -3))
        x_sorted, idx_sorted, inv_order = _gather_sort(x_exp, indices)
        tiles = build_sorted_group_tiles(idx_sorted)
        precomputed = (x_sorted, idx_sorted, inv_order, tiles)

    def cluster_precomputed_tiles() -> mx.array:
        if precomputed is None:
            return cluster()
        x_sorted, _idx_sorted, inv_order, tiles = precomputed
        x_flat = x_sorted
        while x_flat.ndim > 2 and x_flat.shape[-2] == 1:
            x_flat = x_flat.squeeze(-2)
        x_flat = x_flat.reshape(-1, hidden)
        tile_starts, tile_counts, tile_experts = tiles
        gate_rot = hadamard_rotate_metal(x_flat.astype(mx.float32), gate_signs)
        x_act = fused_gate_up_swiglu_mpp_nax_grouped_from_rot_with_tiles(
            gate_rot,
            gate_packed,
            gate_norms,
            up_packed,
            up_norms,
            gate_codebook,
            tile_starts,
            tile_counts,
            tile_experts,
            hidden,
            intermediate,
            bits,
        )
        down_rot = hadamard_rotate_metal(x_act.astype(mx.float32), down_signs)
        x_out = gather_tq_matmul_mpp_nax_grouped_from_rot_with_tiles(
            down_rot,
            down_packed,
            down_norms,
            down_codebook,
            tile_starts,
            tile_counts,
            tile_experts,
            intermediate,
            hidden,
            bits,
        )
        x_out = x_out.reshape(-1, 1, hidden)
        x_out = _scatter_unsort(x_out, inv_order, indices.shape)
        return x_out.squeeze(-2)

    with _env("JANGTQ_MPP_NAX", None):
        baseline = cluster()
        mx.eval(baseline)
        baseline_ms = _time_ms(cluster, repeat)

    nax_fn = cluster_precomputed_tiles if precompute_tiles else cluster
    with _env("JANGTQ_MPP_NAX", nax_mode):
        nax = nax_fn()
        mx.eval(nax)
        nax_ms = _time_ms(nax_fn, repeat)

    diff = baseline.astype(mx.float32) - nax.astype(mx.float32)
    max_abs_err = float(mx.max(mx.abs(diff)).item())
    max_ref_abs = float(mx.max(mx.abs(baseline.astype(mx.float32))).item())
    rel_l2 = float(
        (
            mx.sqrt(mx.sum(diff * diff))
            / mx.maximum(
                mx.sqrt(mx.sum(baseline.astype(mx.float32) * baseline.astype(mx.float32))),
                mx.array(1e-8),
            )
        ).item()
    )
    return {
        "tokens": tokens,
        "top_k": top_k,
        "dispatches": tokens * top_k,
        "hidden": hidden,
        "intermediate": intermediate,
        "n_experts": n_experts,
        "bits": bits,
        "nax_mode": nax_mode,
        "precompute_tiles": precompute_tiles,
        "baseline_ms": baseline_ms,
        "nax_ms": nax_ms,
        "speedup": baseline_ms / nax_ms if nax_ms else None,
        "max_abs_err": max_abs_err,
        "max_ref_abs": max_ref_abs,
        "rel_l2": rel_l2,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--case",
        action="append",
        metavar="TOKENS,TOPK,HIDDEN,INTERMEDIATE,EXPERTS,BITS",
        help="Add a case. Example: 128,2,1024,1024,8,4",
    )
    parser.add_argument(
        "--nax-mode",
        default="1",
        choices=["1", "auto"],
        help="JANGTQ_MPP_NAX mode to time against baseline.",
    )
    parser.add_argument(
        "--precompute-tiles",
        action="store_true",
        help="Precompute sorted expert tile metadata outside the timed NAX path.",
    )
    args = parser.parse_args()

    cases = args.case or [
        "32,2,1024,1024,8,4",
        "128,2,1024,1024,8,4",
        "512,2,1024,1024,8,4",
        "128,2,2048,2048,8,4",
    ]
    result = {
        "mpp_nax_tensorops_available": mpp_nax_tensorops_available(),
        "note": (
            "Synthetic full expert-cluster probe: sort, gate/up/SwiGLU, down, "
            "scatter. Uses current opt-in NAX hooks, including CPU metadata in "
            "the grouped gather helper; production should remove that CPU sync."
        ),
        "cases": [],
    }
    if result["mpp_nax_tensorops_available"]:
        for raw in cases:
            values = [int(part) for part in raw.split(",")]
            result["cases"].append(
                run_case(
                    *values,
                    repeat=args.repeat,
                    nax_mode=args.nax_mode,
                    precompute_tiles=args.precompute_tiles,
                )
            )

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Probe JANGTQ sorted-prefill MoE MLP geometry on M5 TensorOps.

This is a proof harness for the proposed production path:

    sorted expert dispatch rows
    -> same-expert M=16 grouped TensorOps tiles
    -> gate/up SwiGLU
    -> grouped down projection

It compares the current JANGTQ sorted-prefill SwitchGLU path against the
candidate grouped MPP/NAX path using real model-like geometry. The script uses
synthetic packed weights because the performance question is kernel geometry,
not model quality.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from jang_tools.turboquant.codebook import compute_codebook
from jang_tools.turboquant.fused_gate_up_kernel import fused_gate_up_swiglu_matmul
from jang_tools.turboquant.gather_tq_kernel import gather_tq_matmul
from jang_tools.turboquant.hadamard_kernel import hadamard_rotate_metal
from jang_tools.turboquant.mpp_nax_kernel import (
    fused_gate_up_swiglu_mpp_nax_grouped_from_rot,
    gather_tq_matmul_mpp_nax_grouped_from_rot,
    mpp_nax_tensorops_available,
)
from jang_tools.turboquant.rotation import generate_random_signs


@dataclass(frozen=True)
class Case:
    name: str
    tokens: int
    top_k: int
    n_experts: int
    hidden: int
    intermediate: int
    bits: int

    @property
    def dispatches(self) -> int:
        return self.tokens * self.top_k


def _parse_case(raw: str) -> Case:
    parts = raw.split(",")
    if len(parts) != 7:
        raise argparse.ArgumentTypeError(
            "case must be name,tokens,top_k,n_experts,hidden,intermediate,bits"
        )
    name, tokens, top_k, n_experts, hidden, intermediate, bits = parts
    return Case(
        name=name,
        tokens=int(tokens),
        top_k=int(top_k),
        n_experts=int(n_experts),
        hidden=int(hidden),
        intermediate=int(intermediate),
        bits=int(bits),
    )


def _vals_per_u32(bits: int) -> int:
    return 32 // bits


def _random_packed(
    rng: np.random.Generator,
    n_experts: int,
    out_features: int,
    in_features: int,
    bits: int,
) -> mx.array:
    packed_cols = (in_features + _vals_per_u32(bits) - 1) // _vals_per_u32(bits)
    data = rng.integers(
        0,
        np.iinfo(np.uint32).max,
        size=(n_experts, out_features, packed_cols),
        dtype=np.uint32,
    )
    return mx.array(data, dtype=mx.uint32)


def _random_norms(
    rng: np.random.Generator, n_experts: int, out_features: int
) -> mx.array:
    data = rng.uniform(0.6, 1.4, size=(n_experts, out_features)).astype(np.float16)
    return mx.array(data, dtype=mx.float16)


def _balanced_sorted_indices(case: Case) -> mx.array:
    # Sorted rows model the post-mlx_lm _gather_sort() prefill path. Balanced
    # runs are the optimistic but realistic long-prompt case; if real router
    # histograms are skewed, grouping only improves.
    counts = np.full(case.n_experts, case.dispatches // case.n_experts, dtype=np.int64)
    counts[: case.dispatches % case.n_experts] += 1
    idx = np.repeat(np.arange(case.n_experts, dtype=np.uint32), counts)
    return mx.array(idx, dtype=mx.uint32)


def _time_ms(fn, repeat: int) -> float:
    mx.eval(fn())
    start = time.perf_counter()
    for _ in range(repeat):
        mx.eval(fn())
    return (time.perf_counter() - start) * 1000.0 / repeat


def _current_sorted_mlp(
    x_dispatch: mx.array,
    idx: mx.array,
    pg: mx.array,
    ng: mx.array,
    pu: mx.array,
    nu: mx.array,
    pd: mx.array,
    nd: mx.array,
    cb_in: mx.array,
    cb_down: mx.array,
    signs_in: mx.array,
    signs_down: mx.array,
    bits: int,
) -> mx.array:
    x_act = fused_gate_up_swiglu_matmul(
        x_dispatch,
        pg,
        ng,
        pu,
        nu,
        cb_in,
        signs_in,
        idx,
        bits,
    )
    return gather_tq_matmul(
        x_act,
        pd,
        nd,
        cb_down,
        signs_down,
        idx,
        bits,
        sorted_indices=True,
    )


def _candidate_grouped_mlp(
    x_dispatch: mx.array,
    idx: mx.array,
    pg: mx.array,
    ng: mx.array,
    pu: mx.array,
    nu: mx.array,
    pd: mx.array,
    nd: mx.array,
    cb_in: mx.array,
    cb_down: mx.array,
    signs_in: mx.array,
    signs_down: mx.array,
    bits: int,
) -> mx.array:
    hidden = int(x_dispatch.shape[-1])
    dispatches = int(idx.shape[0])
    inter = int(pg.shape[1])
    x_flat = x_dispatch.reshape(dispatches, hidden)
    x_rot = hadamard_rotate_metal(x_flat.astype(mx.float32), signs_in)
    x_act = fused_gate_up_swiglu_mpp_nax_grouped_from_rot(
        x_rot, pg, ng, pu, nu, cb_in, idx, hidden, inter, bits
    )
    x_act_rot = hadamard_rotate_metal(x_act.astype(mx.float32), signs_down)
    out = gather_tq_matmul_mpp_nax_grouped_from_rot(
        x_act_rot, pd, nd, cb_down, idx, inter, hidden, bits
    )
    return out.reshape(dispatches, 1, hidden)


def _component_timings_ms(
    x_dispatch: mx.array,
    idx: mx.array,
    pg: mx.array,
    ng: mx.array,
    pu: mx.array,
    nu: mx.array,
    pd: mx.array,
    nd: mx.array,
    cb_in: mx.array,
    cb_down: mx.array,
    signs_in: mx.array,
    signs_down: mx.array,
    bits: int,
    repeat: int,
) -> dict:
    hidden = int(x_dispatch.shape[-1])
    dispatches = int(idx.shape[0])
    inter = int(pg.shape[1])
    x_flat = x_dispatch.reshape(dispatches, hidden)
    x_rot = hadamard_rotate_metal(x_flat.astype(mx.float32), signs_in)
    gate = gather_tq_matmul_mpp_nax_grouped_from_rot(
        x_rot, pg, ng, cb_in, idx, hidden, inter, bits
    )
    up = gather_tq_matmul_mpp_nax_grouped_from_rot(
        x_rot, pu, nu, cb_in, idx, hidden, inter, bits
    )
    x_act = (gate / (1.0 + mx.exp(-gate))) * up
    mx.eval(x_rot, gate, up, x_act)

    current_gate_up_ms = _time_ms(
        lambda: fused_gate_up_swiglu_matmul(
            x_dispatch, pg, ng, pu, nu, cb_in, signs_in, idx, bits
        ),
        repeat,
    )
    current_act = fused_gate_up_swiglu_matmul(
        x_dispatch, pg, ng, pu, nu, cb_in, signs_in, idx, bits
    )
    mx.eval(current_act)
    current_down_ms = _time_ms(
        lambda: gather_tq_matmul(
            current_act, pd, nd, cb_down, signs_down, idx, bits, sorted_indices=True
        ),
        repeat,
    )
    candidate_x_rotate_ms = _time_ms(
        lambda: hadamard_rotate_metal(x_flat.astype(mx.float32), signs_in),
        repeat,
    )
    candidate_gate_ms = _time_ms(
        lambda: gather_tq_matmul_mpp_nax_grouped_from_rot(
            x_rot, pg, ng, cb_in, idx, hidden, inter, bits
        ),
        repeat,
    )
    candidate_up_ms = _time_ms(
        lambda: gather_tq_matmul_mpp_nax_grouped_from_rot(
            x_rot, pu, nu, cb_in, idx, hidden, inter, bits
        ),
        repeat,
    )
    candidate_activation_ms = _time_ms(
        lambda: (gate / (1.0 + mx.exp(-gate))) * up,
        repeat,
    )
    candidate_fused_gate_up_ms = _time_ms(
        lambda: fused_gate_up_swiglu_mpp_nax_grouped_from_rot(
            x_rot, pg, ng, pu, nu, cb_in, idx, hidden, inter, bits
        ),
        repeat,
    )
    x_act_fused = fused_gate_up_swiglu_mpp_nax_grouped_from_rot(
        x_rot, pg, ng, pu, nu, cb_in, idx, hidden, inter, bits
    )
    mx.eval(x_act_fused)
    candidate_act_rotate_ms = _time_ms(
        lambda: hadamard_rotate_metal(x_act_fused.astype(mx.float32), signs_down),
        repeat,
    )
    x_act_rot = hadamard_rotate_metal(x_act_fused.astype(mx.float32), signs_down)
    mx.eval(x_act_rot)
    candidate_down_ms = _time_ms(
        lambda: gather_tq_matmul_mpp_nax_grouped_from_rot(
            x_act_rot, pd, nd, cb_down, idx, inter, hidden, bits
        ),
        repeat,
    )
    measured_grouped_fused_mlp_ms = (
        candidate_x_rotate_ms
        + candidate_fused_gate_up_ms
        + candidate_act_rotate_ms
        + candidate_down_ms
    )
    return {
        "current_gate_up_ms": current_gate_up_ms,
        "current_down_ms": current_down_ms,
        "candidate_x_rotate_ms": candidate_x_rotate_ms,
        "candidate_grouped_gate_ms": candidate_gate_ms,
        "candidate_grouped_up_ms": candidate_up_ms,
        "candidate_activation_ms": candidate_activation_ms,
        "candidate_fused_grouped_gate_up_ms": candidate_fused_gate_up_ms,
        "candidate_act_rotate_ms": candidate_act_rotate_ms,
        "candidate_grouped_down_ms": candidate_down_ms,
        "measured_grouped_fused_mlp_ms": measured_grouped_fused_mlp_ms,
    }


def run_case(case: Case, repeat: int) -> dict:
    rng = np.random.default_rng(
        50_000
        + case.tokens
        + case.top_k * 13
        + case.n_experts * 17
        + case.hidden
        + case.intermediate
    )
    idx = _balanced_sorted_indices(case)
    x_dispatch = mx.array(
        rng.standard_normal((case.dispatches, 1, case.hidden)).astype(np.float32)
    )
    signs_in = mx.array(
        generate_random_signs(case.hidden, seed=case.hidden), dtype=mx.float32
    )
    signs_down = mx.array(
        generate_random_signs(case.intermediate, seed=case.intermediate),
        dtype=mx.float32,
    )
    cb_in = mx.array(compute_codebook(case.hidden, case.bits), dtype=mx.float32)
    cb_down = mx.array(
        compute_codebook(case.intermediate, case.bits), dtype=mx.float32
    )

    pg = _random_packed(
        rng, case.n_experts, case.intermediate, case.hidden, case.bits
    )
    pu = _random_packed(
        rng, case.n_experts, case.intermediate, case.hidden, case.bits
    )
    pd = _random_packed(
        rng, case.n_experts, case.hidden, case.intermediate, case.bits
    )
    ng = _random_norms(rng, case.n_experts, case.intermediate)
    nu = _random_norms(rng, case.n_experts, case.intermediate)
    nd = _random_norms(rng, case.n_experts, case.hidden)
    mx.eval(x_dispatch, idx, signs_in, signs_down, cb_in, cb_down, pg, pu, pd, ng, nu, nd)

    current = _current_sorted_mlp(
        x_dispatch, idx, pg, ng, pu, nu, pd, nd, cb_in, cb_down, signs_in, signs_down, case.bits
    )
    candidate = _candidate_grouped_mlp(
        x_dispatch, idx, pg, ng, pu, nu, pd, nd, cb_in, cb_down, signs_in, signs_down, case.bits
    )
    mx.eval(current, candidate)
    max_abs_err = float(
        mx.max(mx.abs(current.astype(mx.float32) - candidate.astype(mx.float32))).item()
    )

    current_ms = _time_ms(
        lambda: _current_sorted_mlp(
            x_dispatch, idx, pg, ng, pu, nu, pd, nd, cb_in, cb_down, signs_in, signs_down, case.bits
        ),
        repeat,
    )
    candidate_ms = _time_ms(
        lambda: _candidate_grouped_mlp(
            x_dispatch, idx, pg, ng, pu, nu, pd, nd, cb_in, cb_down, signs_in, signs_down, case.bits
        ),
        repeat,
    )
    components = _component_timings_ms(
        x_dispatch,
        idx,
        pg,
        ng,
        pu,
        nu,
        pd,
        nd,
        cb_in,
        cb_down,
        signs_in,
        signs_down,
        case.bits,
        repeat,
    )
    fused_ms = components["measured_grouped_fused_mlp_ms"]
    return {
        "name": case.name,
        "tokens": case.tokens,
        "top_k": case.top_k,
        "dispatches": case.dispatches,
        "n_experts": case.n_experts,
        "hidden": case.hidden,
        "intermediate": case.intermediate,
        "bits": case.bits,
        "current_sorted_mlp_ms": current_ms,
        "candidate_grouped_mpp_nax_mlp_ms": candidate_ms,
        "candidate_speedup": current_ms / candidate_ms if candidate_ms else None,
        "measured_grouped_fused_mpp_nax_mlp_ms": fused_ms,
        "measured_grouped_fused_speedup": (
            current_ms / fused_ms if fused_ms else None
        ),
        "component_ms": components,
        "max_abs_err": max_abs_err,
        "dispatches_per_expert_balanced": case.dispatches / case.n_experts,
        "note": (
            "Synthetic sorted-prefill MoE MLP only. It includes gate/up/down "
            "and Hadamard rotations, but not router logits, attention, cache, "
            "sampling, or server overhead."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        help=(
            "name,tokens,top_k,n_experts,hidden,intermediate,bits. "
            "Defaults cover ZAYA1 and Qwen3.6-35B-A3B JANGTQ4 geometry."
        ),
    )
    args = parser.parse_args()

    cases = args.case or [
        Case("zaya1_512tok", 512, 1, 16, 2048, 4096, 4),
        Case("qwen36_a3b_512tok", 512, 8, 256, 2048, 512, 4),
    ]
    result = {
        "mpp_nax_tensorops_available": mpp_nax_tensorops_available(),
        "production_default_changed": False,
        "cases": [],
    }
    if result["mpp_nax_tensorops_available"]:
        for case in cases:
            result["cases"].append(run_case(case, args.repeat))

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

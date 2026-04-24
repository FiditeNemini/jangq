"""
AWQ per-channel scale computation from captured activations.
Created by Jinho Jang (eric@jangq.ai)

Given:
  - awq_activations.safetensors — per-module per-input-channel max(|x|)
  - FP8 source model weights — per-tensor

Computes per-layer scale vectors s such that W' = W * s improves 2-bit
quantization MSE.  Grid-searches alpha in [0, 1]; selects the alpha that
minimizes Hadamard+MXTQ roundtrip MSE on the actual weight.

For MoE routed experts: one scale per (layer, projection), shared across
all 256 experts in that layer (they see same input distribution).

Output: awq_scales.safetensors — keys match awq_activations keys exactly.
"""
import sys, json, time
from pathlib import Path

import numpy as np
import mlx.core as mx
from safetensors import safe_open
from safetensors.numpy import save_file, load_file

from .fp8 import load_fp8_tensor
from .calibrate import _load_bf16_tensor
from .turboquant.rotation import (
    generate_random_signs,
    hadamard_rotate,
    hadamard_inverse,
)
from .turboquant.codebook import compute_codebook


ALPHA_GRID = np.arange(0.0, 1.01, 0.05)
SEED = 42


def _load_weight_matrix(model_path: Path, weight_map: dict,
                         tensor_name: str) -> np.ndarray:
    """Load a single FP8 weight by full tensor name."""
    shard = weight_map.get(tensor_name)
    if shard is None:
        return None
    sf_path = model_path / shard
    with safe_open(str(sf_path), framework="numpy") as f:
        if tensor_name not in f.keys():
            return None
        shape = list(f.get_slice(tensor_name).get_shape())
        sk = tensor_name + "_scale_inv"
        scale = None
        try:
            scale = f.get_tensor(sk)
        except Exception:
            scale = None
    try:
        return load_fp8_tensor(sf_path, tensor_name, shape, scale).astype(np.float32)
    except Exception:
        return _load_bf16_tensor(sf_path, tensor_name, shape).astype(np.float32)


def _mxtq_roundtrip(W_np: np.ndarray, bits: int, signs: mx.array,
                     codebook: np.ndarray) -> np.ndarray:
    """Full Hadamard + per-row-norm + codebook roundtrip (as in real converter)."""
    W_rot = np.array(hadamard_rotate(mx.array(W_np), signs))
    norms = np.linalg.norm(W_rot, axis=1, keepdims=True) + 1e-12
    W_n = W_rot / norms
    d = np.abs(W_n[..., None] - codebook[None, None, :])
    q_idx = np.argmin(d, axis=-1)
    W_q_rot = codebook[q_idx] * norms
    return np.array(hadamard_inverse(mx.array(W_q_rot), signs))


def _compute_scale_for_module(W: np.ndarray, x_mag: np.ndarray, bits: int,
                                signs_cache: dict, cb_cache: dict) -> tuple:
    """Grid-search alpha; return (best_alpha, best_scale, best_rel_err).

    Uses synthetic activations matched to x_mag for output-MSE evaluation.
    """
    out_f, in_f = W.shape
    W_max = np.abs(W).max(axis=0) + 1e-8  # per-input-channel weight max

    if in_f not in signs_cache:
        signs_cache[in_f] = mx.array(generate_random_signs(in_f, seed=SEED))
        cb_cache[in_f] = np.array(compute_codebook(in_f, bits), dtype=np.float32)
    signs = signs_cache[in_f]
    codebook = cb_cache[in_f]

    # Synthesize evaluation activations matching x_mag distribution
    rng = np.random.default_rng(42)
    N = 1024
    base = rng.standard_normal((N, in_f)).astype(np.float32)
    # Scale each column so column-max ≈ x_mag[j]
    base_max = np.abs(base).max(axis=0) + 1e-8
    X_eval = base * (x_mag / base_max)[None, :]
    Y_ref = X_eval @ W.T
    y_norm = np.linalg.norm(Y_ref) + 1e-8

    best_alpha = 0.0
    best_rel = float("inf")
    best_scale = np.ones(in_f, dtype=np.float32)

    for alpha in ALPHA_GRID:
        if alpha == 0.0:
            s = np.ones(in_f, dtype=np.float32)
        else:
            s = (x_mag ** alpha) / (W_max ** (1.0 - alpha))
            s = s / np.exp(np.mean(np.log(s + 1e-12)))  # geomean = 1
            s = np.clip(s, 1e-2, 1e2).astype(np.float32)

        W_scaled = W * s[None, :]
        W_quant = _mxtq_roundtrip(W_scaled, bits, signs, codebook)
        W_recovered = W_quant / s[None, :]

        Y = X_eval @ W_recovered.T
        rel = float(np.linalg.norm(Y_ref - Y) / y_norm)
        if rel < best_rel:
            best_rel = rel
            best_alpha = float(alpha)
            best_scale = s.copy()

    return best_alpha, best_scale, best_rel


def compute_scales(model_path: str, activations_path: str, output_path: str,
                    bits: int = 2, sample_tensor_per_layer: int = 1):
    """Compute AWQ scales for every MoE layer.

    Args:
        model_path: FP8 source root.
        activations_path: awq_activations.safetensors from awq_capture_fp8.
        output_path: where to write awq_scales.safetensors.
        bits: target bit width for routed experts (2 for JANGTQ_2L).
        sample_tensor_per_layer: how many expert weights per layer to use
            for grid search (1 = use just expert 0; higher = average).

    Output schema: one tensor per key from activations_path.
    For MoE layers: key is `model.layers.L.mlp.switch_mlp.{gate|up|down}_proj`,
    shape (in_features,) = scale vector.
    """
    model_path = Path(model_path)
    activations = load_file(activations_path)
    print(f"Loaded {len(activations)} activation keys", flush=True)

    # Build weight map.
    idx_path = model_path / "model.safetensors.index.json"
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]

    signs_cache, cb_cache = {}, {}
    scales = {}
    log_rows = []

    for key, x_mag in sorted(activations.items()):
        # Only process routed expert keys for now (JANGTQ_2L only needs those).
        # Attention + dense MLP stay FP16 passthrough.
        if "switch_mlp" not in key:
            continue

        # Map AWQ key -> FP8 source tensor for expert 0 as representative.
        # e.g. model.layers.10.mlp.switch_mlp.gate_proj
        #   -> model.layers.10.mlp.experts.0.gate_proj.weight
        parts = key.split(".")
        # switch_mlp position
        try:
            sm_idx = parts.index("switch_mlp")
        except ValueError:
            continue
        proj = parts[sm_idx + 1]  # gate_proj / up_proj / down_proj
        layer_prefix = ".".join(parts[:sm_idx])  # model.layers.L.mlp

        t_key = f"{layer_prefix}.experts.0.{proj}.weight"
        W = _load_weight_matrix(model_path, weight_map, t_key)
        if W is None:
            print(f"  WARN: no weight for {t_key}; skipping", flush=True)
            continue

        # Sanity: activation in_features should match weight's in_features
        if W.shape[1] != x_mag.shape[0]:
            print(f"  SHAPE MISMATCH {key}: W.in={W.shape[1]} "
                  f"x_mag={x_mag.shape[0]}; skipping", flush=True)
            continue

        t0 = time.time()
        alpha, scale, rel = _compute_scale_for_module(
            W, x_mag, bits, signs_cache, cb_cache
        )
        t_el = time.time() - t0
        log_rows.append((key, alpha, rel))
        scales[key] = scale.astype(np.float16)  # fp16 runtime storage
        print(f"  {key}: alpha={alpha:.2f} rel_err={rel:.4f} ({t_el:.1f}s)",
              flush=True)

    print(f"Computed {len(scales)} scales", flush=True)
    if scales:
        save_file({k: v.astype(np.float32) for k, v in scales.items()},
                  output_path)
        print(f"wrote {output_path}", flush=True)
    else:
        print("NO SCALES COMPUTED — check activation keys + weight names",
              flush=True)

    alphas = [r[1] for r in log_rows]
    rels = [r[2] for r in log_rows]
    if alphas:
        print(f"Summary: alpha mean={np.mean(alphas):.2f} "
              f"median={np.median(alphas):.2f}, "
              f"rel_err mean={np.mean(rels):.4f} median={np.median(rels):.4f}",
              flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: python3 -m jang_tools.awq_scales "
              "<fp8_src> <activations_safetensors> <output_path> [bits]",
              file=sys.stderr)
        sys.exit(1)
    src = sys.argv[1]
    acts = sys.argv[2]
    out = sys.argv[3]
    bits = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    compute_scales(src, acts, out, bits=bits)

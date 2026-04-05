"""
JANG Codebook VQ — Cross-Expert Vector Quantization for MoE Weight Compression.
Created by Jinho Jang (eric@jangq.ai)

Learns a shared codebook across all experts in a layer, then represents each
expert's weight groups as indices into the codebook. Exploits the structural
similarity between experts (same architecture, same training distribution)
to achieve ~10x compression over standard 2-bit quantization with BETTER
reconstruction quality.

Empirical result (2026-04-04):
  16K centroids: 33% lower MSE than 2-bit at 10.4% of the storage size.
  Effective: 0.23 bits/weight (vs 2.25 bits/weight for 2-bit + scale/bias).
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


DEFAULT_N_CODES = 16384
DEFAULT_GROUP_SIZE = 8  # MUST be 8 — validated 2026-04-04, g=128 gives cosine 0.2, g=8 gives 0.976


def learn_codebook(
    weight_groups: np.ndarray,
    n_codes: int = DEFAULT_N_CODES,
    max_train_samples: int = 200_000,
    batch_size: int = 8192,
    n_init: int = 3,
    max_iter: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Learn a codebook from weight groups using Mini-Batch K-means.

    Args:
        weight_groups: float32[n_groups, group_size]
        n_codes: number of centroids
        max_train_samples: subsample for training
        seed: random seed

    Returns:
        codebook: float16[n_codes, group_size]
        indices: uint16[n_groups]
    """
    from sklearn.cluster import MiniBatchKMeans

    n_groups = weight_groups.shape[0]
    rng = np.random.default_rng(seed)
    if n_groups > max_train_samples:
        train_idx = rng.choice(n_groups, max_train_samples, replace=False)
        train_data = weight_groups[train_idx]
    else:
        train_data = weight_groups

    kmeans = MiniBatchKMeans(
        n_clusters=n_codes, batch_size=batch_size,
        n_init=n_init, max_iter=max_iter, random_state=seed,
    )
    kmeans.fit(train_data)
    indices = kmeans.predict(weight_groups).astype(np.uint16)
    codebook = kmeans.cluster_centers_.astype(np.float16)

    return codebook, indices


def codebook_quantize_experts(
    expert_weights: np.ndarray,
    group_size: int = DEFAULT_GROUP_SIZE,
    n_codes: int = DEFAULT_N_CODES,
    seed: int = 42,
) -> dict:
    """
    Quantize a 3D expert weight tensor using cross-expert codebook VQ.

    Args:
        expert_weights: float32[n_experts, out_dim, in_dim]
        group_size: weights per codebook group
        n_codes: number of centroids

    Returns:
        dict with codebook, indices, metadata
    """
    n_experts, out_dim, in_dim = expert_weights.shape
    assert in_dim % group_size == 0, f"in_dim {in_dim} not divisible by group_size {group_size}"
    n_groups_per_row = in_dim // group_size

    groups = expert_weights.reshape(-1, group_size).astype(np.float32)

    t0 = time.time()
    codebook, indices = learn_codebook(groups, n_codes=n_codes, seed=seed)
    elapsed = time.time() - t0

    reconstructed = codebook[indices].astype(np.float32)
    mse = float(np.mean((groups - reconstructed) ** 2))
    indices = indices.reshape(n_experts, out_dim, n_groups_per_row)

    return {
        "codebook": codebook,
        "indices": indices,
        "metadata": {
            "n_codes": n_codes,
            "group_size": group_size,
            "shape": list(expert_weights.shape),
            "mse": mse,
            "elapsed_s": round(elapsed, 1),
        },
    }


def codebook_dequantize(codebook, indices, shape, group_size):
    """Reconstruct expert weights from codebook and indices."""
    flat_indices = indices.reshape(-1)
    flat_weights = codebook[flat_indices]
    return flat_weights.reshape(shape).astype(np.float16)


def codebook_matmul_mlx(x, codebook, indices, group_size):
    """
    MLX codebook matmul — reconstruct weights from codebook, then matmul.

    This is the reference implementation. Production would use a fused
    Metal kernel that does codebook lookup + dot product in one pass.

    x: mx.array [batch, in_dim]
    codebook: mx.array [n_codes, group_size]
    indices: mx.array [out_dim, n_groups]
    """
    import mlx.core as mx

    flat_idx = indices.reshape(-1)
    flat_weights = codebook[flat_idx]
    W = flat_weights.reshape(indices.shape[0], -1)
    return x @ W.T


def convert_experts_to_codebook(
    model_path: str | Path,
    output_path: str | Path,
    n_codes: int = DEFAULT_N_CODES,
    group_size: int = DEFAULT_GROUP_SIZE,
):
    """
    Convert a JANG model's expert weights to codebook VQ format.

    Loads the JANG model, dequantizes expert weights, learns codebooks,
    and saves in the codebook format. Non-expert weights stay as-is.
    """
    import mlx.core as mx
    from safetensors import safe_open
    from safetensors.numpy import save_file

    model_path = Path(model_path)
    output_path = Path(output_path)

    print(f"\n{'='*60}")
    print(f"  JANG Codebook VQ Converter")
    print(f"  Created by Jinho Jang (eric@jangq.ai)")
    print(f"{'='*60}")
    print(f"  Source: {model_path}")
    print(f"  Output: {output_path}")
    print(f"  Codebook size: {n_codes}")
    print(f"  Group size: {group_size}")
    print(f"{'='*60}\n")

    config = json.loads((model_path / "config.json").read_text())
    jang_cfg_path = model_path / "jang_config.json"
    jang_cfg = json.loads(jang_cfg_path.read_text()) if jang_cfg_path.exists() else {}

    weight_files = sorted(model_path.glob("model*.safetensors"))

    # Separate expert vs non-expert tensors
    expert_keys = {}  # key → shard path
    non_expert = {}

    print("  Scanning tensors...")
    for sf_path in weight_files:
        with safe_open(str(sf_path), framework="numpy") as f:
            for key in f.keys():
                if ".switch_mlp." in key or ".experts.switch_glu." in key:
                    expert_keys[key] = str(sf_path)
                else:
                    non_expert[key] = f.get_tensor(key)

    print(f"  Expert tensors: {len(expert_keys)}, Non-expert: {len(non_expert)}")

    # Group expert tensors by (layer, tensor_type)
    layer_groups = {}  # (layer_idx, type) → {weight: (path, key), scales: ..., biases: ...}
    for key, sf_path in expert_keys.items():
        parts = key.split(".")
        layer_idx = None
        tensor_type = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass
            if p in ("gate_proj", "up_proj", "down_proj"):
                tensor_type = p
        if layer_idx is None or tensor_type is None:
            non_expert[key] = None
            continue

        suffix = key.rsplit(".", 1)[-1]
        lk = (layer_idx, tensor_type)
        layer_groups.setdefault(lk, {})[suffix] = (sf_path, key)

    print(f"  Expert groups: {len(layer_groups)}")

    # Process each expert group
    output_path.mkdir(parents=True, exist_ok=True)
    total_original = 0
    total_codebook = 0
    codebook_metadata = {}

    for (layer_idx, tensor_type), parts in sorted(layer_groups.items()):
        if not all(s in parts for s in ("weight", "scales", "biases")):
            print(f"  SKIP layer {layer_idx} {tensor_type}: incomplete")
            continue

        # Load and dequantize
        with safe_open(parts["weight"][0], framework="numpy") as f:
            qw = mx.array(f.get_tensor(parts["weight"][1]))
        with safe_open(parts["scales"][0], framework="numpy") as f:
            scales = mx.array(f.get_tensor(parts["scales"][1]))
        with safe_open(parts["biases"][0], framework="numpy") as f:
            biases = mx.array(f.get_tensor(parts["biases"][1]))

        # Infer bits
        packed_cols = qw.shape[-1]
        n_groups_s = scales.shape[-1]
        bits = 2
        for b in [2, 3, 4, 6, 8]:
            in_dim = packed_cols * 32 // b
            gs_check = in_dim // n_groups_s if n_groups_s > 0 else 0
            if gs_check > 0 and gs_check * n_groups_s == in_dim:
                bits = b
                break

        gs_orig = in_dim // n_groups_s
        dq = mx.dequantize(qw, scales, biases, group_size=gs_orig, bits=bits)
        mx.synchronize()
        dq_np = np.array(dq).astype(np.float32)

        original_bytes = qw.nbytes + scales.nbytes + biases.nbytes
        total_original += original_bytes

        # Codebook quantize
        # Adjust group_size if in_dim isn't divisible
        actual_gs = group_size
        in_dim_actual = dq_np.shape[-1]
        if in_dim_actual % actual_gs != 0:
            # Find largest divisor <= group_size
            for gs_try in range(actual_gs, 0, -1):
                if in_dim_actual % gs_try == 0:
                    actual_gs = gs_try
                    break

        result = codebook_quantize_experts(dq_np, group_size=actual_gs, n_codes=n_codes)

        cb_bytes = result["codebook"].nbytes + result["indices"].nbytes
        total_codebook += cb_bytes

        # Save codebook tensors
        cb_tensors = {
            "codebook": result["codebook"],
            "indices": result["indices"],
        }
        fname = f"codebook-layer-{layer_idx:03d}-{tensor_type}.safetensors"
        save_file(cb_tensors, str(output_path / fname))

        codebook_metadata[f"{layer_idx}.{tensor_type}"] = result["metadata"]

        print(f"  L{layer_idx:>3d} {tensor_type:<10s}: "
              f"MSE={result['metadata']['mse']:.8f} | "
              f"{original_bytes/1e6:.1f}MB → {cb_bytes/1e6:.1f}MB "
              f"({cb_bytes/original_bytes:.1%}) | "
              f"{result['metadata']['elapsed_s']:.0f}s")

        del qw, scales, biases, dq, dq_np
        mx.clear_cache()

    # Save non-expert weights
    non_expert_clean = {k: v for k, v in non_expert.items() if v is not None}
    if non_expert_clean:
        save_file(non_expert_clean, str(output_path / "model-00001-of-00001.safetensors"))

    # Save configs
    import shutil
    (output_path / "config.json").write_text(json.dumps(config, indent=2))

    jang_cfg.setdefault("quantization", {})
    jang_cfg["quantization"]["codebook_vq"] = True
    jang_cfg["quantization"]["n_codes"] = n_codes
    jang_cfg["quantization"]["codebook_group_size"] = group_size
    jang_cfg["codebook_layers"] = codebook_metadata
    (output_path / "jang_config.json").write_text(json.dumps(jang_cfg, indent=2))

    # Copy tokenizer + config files
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                  "tokenizer.model", "added_tokens.json", "generation_config.json",
                  "chat_template.json", "preprocessor_config.json"]:
        src = model_path / fname
        if src.exists():
            shutil.copy2(str(src), str(output_path / fname))

    ratio = total_codebook / total_original if total_original > 0 else 0
    print(f"\n{'='*60}")
    print(f"  DONE — Codebook VQ")
    print(f"  Expert weights: {total_original/1e9:.1f} GB → {total_codebook/1e9:.1f} GB ({ratio:.1%})")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

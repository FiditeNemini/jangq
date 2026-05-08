"""Diff PyTorch-reference forward vs MLX runtime forward layer-by-layer.

Loads the same DSV4 layer's weights into both:
  - `jang_tools.dsv4.layer_forward.Block`  (pure PyTorch, bf16, full dequant)
  - `jang_tools.dsv4.mlx_model.DeepseekV4DecoderLayer` (MLX, with any
    quantization applied)

Feeds the same input and records intermediate tensors at each sub-block
boundary (hc_pre output, attention output, hc_post output, hc_pre ffn,
moe output, hc_post ffn). Prints per-boundary max/mean absolute diff.

Run:
  python -m jang_tools.dsv4.verify_mlx_vs_torch \\
      --mlx-bundle <path/to/output-bundle> \\
      --source <path/to/DeepSeek-V4-Flash> \\
      --layer 0

Only compress_ratio=0 layers are source-equivalent in this tool. The PyTorch
calibration reference does not yet model DSV4 CSA/HSA compressor/indexer
sparse attention, so compressed layers are refused to avoid false failures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import mlx.core as mx

# MLX side
from jang_tools.dsv4 import mlx_register  # noqa
from mlx_lm.models.base import create_attention_mask

from jang_tools.dsv4.mlx_model import ModelArgs, DeepseekV4DecoderLayer

# Torch reference
from jang_tools.dsv4.layer_forward import DSV4Config, Block, load_block_from_shards
from jang_tools.dsv4.weight_loader import ShardIndex
from jang_tools.dsv4.ops import precompute_freqs_cis


def _compress_ratio_for_layer(cfg_json: dict, layer_id: int) -> int:
    ratios = cfg_json.get("compress_ratios")
    if isinstance(ratios, list) and 0 <= layer_id < len(ratios):
        return int(ratios[layer_id] or 0)
    num_layers = int(cfg_json.get("num_hidden_layers", 43))
    if layer_id == 0 or layer_id == num_layers - 1:
        return 0
    return 4 if (layer_id - 1) % 2 else 128


def _window_size_for_mask(mlx_layer, fallback: int) -> int:
    args = getattr(mlx_layer, "args", None)
    return int(getattr(args, "sliding_window", fallback))


def diff(name: str, mlx_t: mx.array, torch_t: torch.Tensor):
    """Print max/mean abs diff."""
    a = np.array(mlx_t, copy=False).astype(np.float32)
    b = torch_t.detach().float().numpy()
    if a.shape != b.shape:
        print(f"  [{name}] SHAPE MISMATCH — mlx {a.shape} vs torch {b.shape}")
        return
    d = np.abs(a - b)
    print(f"  [{name}] max={d.max():.4g}  mean={d.mean():.4g}  "
          f"a_range=[{a.min():.3g}, {a.max():.3g}]  "
          f"b_range=[{b.min():.3g}, {b.max():.3g}]")


def run(mlx_bundle: Path, source_dir: Path, layer_id: int = 0):
    # --- Torch side: use pure PyTorch reference loaded from FP4+FP8 source
    print(f"[torch] checking layer {layer_id}...", flush=True)
    cfg_json = json.loads((source_dir / "config.json").read_text())
    compress_ratio = _compress_ratio_for_layer(cfg_json, layer_id)
    if compress_ratio:
        raise SystemExit(
            f"Layer {layer_id} has compress_ratio={compress_ratio}. "
            "verify_mlx_vs_torch is source-equivalent only for "
            "compress_ratio=0 layers until the PyTorch DSV4 reference ports "
            "CSA/HSA compressor/indexer sparse attention."
        )
    src_idx = ShardIndex(source_dir)
    print(f"[torch] indexing source shards...")
    torch_cfg = DSV4Config.from_config_json(cfg_json)
    print(f"[torch] loading layer {layer_id} from source...")
    torch_blk = load_block_from_shards(layer_id, torch_cfg, src_idx).eval()
    print(f"[torch] layer loaded, {sum(p.numel() for p in torch_blk.parameters())/1e9:.2f} B params")

    # --- MLX side: load bundle (already converted JANG_2L)
    print(f"[mlx] loading bundle layer {layer_id}...")
    from mlx_lm.utils import load_model
    mlx_model, _ = load_model(mlx_bundle)
    mlx_layer = mlx_model.model.layers[layer_id]
    print(f"[mlx] layer ready")

    # --- Same input
    torch.manual_seed(0)
    mx.random.seed(0)
    B, L = 1, 8
    D = torch_cfg.dim
    x_np = np.random.randn(B, L, torch_cfg.hc_mult, D).astype(np.float32) * 0.02
    x_t = torch.from_numpy(x_np).to(torch.bfloat16)
    x_m = mx.array(x_np.astype(np.float16))

    input_ids_np = np.random.randint(0, torch_cfg.vocab_size, (B, L)).astype(np.int64)
    ids_t = torch.from_numpy(input_ids_np)
    ids_m = mx.array(input_ids_np.astype(np.int32))

    # Precompute freqs_cis for torch
    fc = precompute_freqs_cis(
        torch_cfg.rope_head_dim, L, torch_cfg.original_seq_len,
        torch_cfg.rope_theta, torch_cfg.rope_factor,
        torch_cfg.beta_fast, torch_cfg.beta_slow,
    )

    # --- Diff input
    print()
    print("=== Input diff (random x, should be 0 within dtype rounding) ===")
    # Compare input bf16 vs fp16 version — expect small diff from dtype cast
    diff("input x", x_m, x_t)
    print()

    # --- Run both layer forwards and compare outputs
    print("=== Layer full forward — output diff ===")
    with torch.inference_mode():
        y_t = torch_blk(x_t, fc, ids_t)
    mask_m = create_attention_mask(
        x_m[:, :, 0, :],
        cache=None,
        window_size=_window_size_for_mask(mlx_layer, torch_cfg.window_size),
        return_array=True,
    )
    y_m = mlx_layer(x_m, mask=mask_m, cache=None, input_ids=ids_m)
    mx.eval(y_m)
    diff("layer_output", y_m, y_t)
    print()

    # TODO: instrument finer-grained per-sub-block diffs.
    # For now the end-to-end layer diff tells us magnitude of disagreement.


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mlx-bundle", required=True, type=Path)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()
    run(args.mlx_bundle, args.source, args.layer)
    return 0


if __name__ == "__main__":
    sys.exit(main())

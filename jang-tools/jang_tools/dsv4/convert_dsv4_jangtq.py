"""DeepSeek-V4-Flash FP4+FP8 source → JANGTQ bundle.

No REAP prune — converts all 256 experts as-is. Profile:
  JANGTQ2 (default): routed experts 2-bit MXTQ, attention + embed +
                     lm_head + shared experts 8-bit affine, norms +
                     router + mHC params fp16 passthrough
  JANGTQ4: routed experts 4-bit MXTQ (bigger bundle, better fidelity)

Usage:
  python -m jang_tools.dsv4.convert_dsv4_jangtq \\
      --src <path/to/DeepSeek-V4-Flash> \\
      --dst ~/.mlxstudio/models/JANGQ-AI/DSV4-Flash-JANGTQ \\
      --profile 2
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file as sf_save_np

from jang_tools.dsv4.weight_loader import ShardIndex


SEED = 42
FORMAT = "jangtq"  # set by main() from --format flag: "jang" or "jangtq"


def classify(name: str, profile_bits: int) -> tuple[int, str, int]:
    """Map tensor name → (bits, method, group_size).

    Uses uniform **group_size=32** across all quantized tensors — matches
    FP4's native block size, minimal extra scale overhead for FP8
    source tensors (~3% vs group=128). Keeps mlx_lm's single-config
    path simple (one group_size value).

    - FP4-origin routed experts:
      * profile 8 → 8-bit affine g=32 (max fidelity; ~0.5% RMS)
      * profile 4 → 4-bit affine g=32 (matches FP4 source; ~9% RMS)
      * profile 2 → 2-bit MXTQ codebook (aggressive; lossy)
    - FP8-origin attention/shared/embed/head: 8-bit affine g=32
    - Norms/small: fp16 passthrough
    """
    n = name

    # Norms + small tensors → fp16 passthrough
    if ("norm" in n or n.endswith(".bias") or "attn_sink" in n
            or ".ape" in n or "tid2eid" in n or n.startswith("hc_")
            or re.search(r"^layers\.\d+\.hc_", n)
            or re.search(r"^mtp\.\d+\.hc_", n)
            or re.search(r"^layers\.\d+\.ffn\.gate\.(weight|bias)$", n)
            ):
        return 16, "passthrough", 0

    # Router gate.weight → fp16 passthrough
    if n.endswith(".gate.weight") and "experts" not in n:
        return 16, "passthrough", 0

    # Routed experts
    if re.search(r"ffn\.experts\.\d+\.(w1|w2|w3)\.weight$", n):
        # JANG format: always use standard affine (all bit-widths).
        # JANGTQ format: MXTQ codebook for 2-bit, affine for 3+.
        if FORMAT == "jang":
            return profile_bits, "affine", 32
        if profile_bits in (3, 4, 5, 6, 8):
            return profile_bits, "affine", 32
        return profile_bits, "mxtq", 0

    # Shared experts + attention + embed + head: default 8-bit affine g=32.
    # Env DSV4_HIGH_PRECISION=1 keeps these at bf16 (no quant) to eliminate
    # compound quant error for arithmetic-sensitive reasoning. Trade-off:
    # bundle +5-8 GB, but removes all quant noise from non-routed path.
    import os as _os
    hp = _os.environ.get("DSV4_HIGH_PRECISION", "0") == "1"
    if "shared_experts" in n and n.endswith(".weight"):
        return (16, "passthrough", 0) if hp else (8, "affine", 32)
    if re.search(r"attn\.(wq_a|wq_b|wkv|wo_a|wo_b)\.weight$", n):
        return (16, "passthrough", 0) if hp else (8, "affine", 32)
    if n == "embed.weight" or n == "head.weight":
        return (16, "passthrough", 0) if hp else (8, "affine", 32)

    if n.endswith(".weight"):
        return 8, "affine", 32

    return 16, "passthrough", 0


def convert(src: Path, dst: Path, profile_bits: int) -> None:
    import mlx.core as mx
    from jang_tools.turboquant.linear import tq_quantize_weight

    dst.mkdir(parents=True, exist_ok=True)
    idx = ShardIndex(src)
    print(f"[convert] source: {src}")
    print(f"[convert] target: {dst}")
    profile_name = f"JANG_{profile_bits}L" if FORMAT == "jang" else f"JANGTQ{profile_bits}"
    print(f"[convert] profile: {profile_name} (format={FORMAT})")
    print(f"[convert] scanning for .weight keys (skip sibling .scale)...")
    weight_keys = [k for k in idx.keys if not k.endswith(".scale")]
    print(f"[convert] {len(weight_keys)} logical tensors to process")

    MAX_SHARD_BYTES = 1_000_000_000
    shard_idx = 1
    shard_bytes = 0
    shard_buf: dict[str, np.ndarray] = {}
    shard_map: dict[str, str] = {}

    totals = {"mxtq": 0, "affine": 0, "passthrough": 0}
    t_start = time.time()

    def flush_shard():
        nonlocal shard_idx, shard_bytes, shard_buf
        if not shard_buf:
            return
        shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        sf_save_np(shard_buf, str(dst / shard_name))
        for k in shard_buf:
            shard_map[k] = shard_name
        print(f"    shard {shard_idx}: {len(shard_buf)} tensors, "
              f"{shard_bytes / 1e9:.2f} GB  "
              f"(elapsed {time.time() - t_start:.0f}s)", flush=True)
        shard_buf = {}
        shard_bytes = 0
        shard_idx += 1

    def add_tensor(name: str, arr: np.ndarray):
        nonlocal shard_bytes
        shard_buf[name] = arr
        shard_bytes += arr.nbytes
        if shard_bytes >= MAX_SHARD_BYTES:
            flush_shard()

    for i, name in enumerate(weight_keys):
        bits, method, gsz = classify(name, profile_bits)

        if method == "passthrough":
            t = idx.read_tensor(name, out_dtype=torch.float16)
            arr = t.numpy() if t.dtype != torch.bfloat16 else t.float().numpy().astype(np.float16)
            add_tensor(name, arr)
            totals["passthrough"] += 1

        elif method == "affine":
            t = idx.read_tensor(name, out_dtype=torch.float32)
            w = mx.array(t.numpy())
            # FP4-origin routed experts at 4-bit: use MXFP4 mode which
            # exactly replicates source FP4 16-level log-spaced codebook.
            # All other tensors (FP8-origin attention, shared, embed, head):
            # use standard affine — they're FP8 source so linear 8-bit
            # affine represents them losslessly within 0.5% RMS.
            is_routed_expert = re.search(r"ffn\.experts\.\d+\.(w1|w2|w3)\.weight$", name) is not None
            # Direct-copy only applies if source is still in FP4 format (int8
            # packed + float8_e8m0fnu scale). BF16-dequant sources have no
            # .scale sibling — fall through to mx.quantize on the bf16 tensor.
            raw_w = idx.read_raw(name) if is_routed_expert and bits == 4 else None
            src_is_fp4 = raw_w is not None and raw_w.dtype == torch.int8
            if is_routed_expert and bits == 4 and src_is_fp4:
                # BIT-EXACT preservation: source is already MXFP4 format
                # (int8 packed FP4 + UE8M0 scale). MLX's mxfp4 uint32 layout
                # matches source int8 byte-for-byte (little-endian packing of
                # 4 source bytes per uint32, nibbles LSB→MSB). So we can
                # DIRECT-COPY without going through bf16 intermediate.
                sk = name[:-len(".weight")] + ".scale" if name.endswith(".weight") else name + ".scale"
                raw_s = idx.read_raw(sk)    # float8_e8m0fnu torch tensor
                # int8 (out, in/2) → reinterpret as uint8 bytes → pack into uint32
                w_bytes = raw_w.numpy().view(np.uint8)  # (out, in/2)
                out_dim, packed_in = w_bytes.shape
                in_dim = packed_in * 2
                assert in_dim % 8 == 0, f"in_dim {in_dim} not multiple of 8"
                # View as (out, in/8, 4 bytes) then as little-endian uint32
                w_u32 = w_bytes.reshape(out_dim, in_dim // 8, 4).copy().view(np.uint32).reshape(out_dim, in_dim // 8)
                # float8_e8m0fnu doesn't support .numpy() directly; reinterpret as uint8
                s_bytes = raw_s.view(torch.uint8).numpy()   # (out, in/32)
                assert s_bytes.shape == (out_dim, in_dim // 32), f"scale shape {s_bytes.shape}"
                base = name[:-len(".weight")] if name.endswith(".weight") else name
                add_tensor(f"{base}.weight", np.ascontiguousarray(w_u32))
                add_tensor(f"{base}.scales", np.ascontiguousarray(s_bytes))
                totals["affine"] += 1
            else:
                qw, qs, qb = mx.quantize(w, group_size=gsz or 64, bits=bits)
                base = name[:-len(".weight")] if name.endswith(".weight") else name
                add_tensor(f"{base}.weight", np.array(qw))
                add_tensor(f"{base}.scales", np.array(qs).astype(np.float16))
                add_tensor(f"{base}.biases", np.array(qb).astype(np.float16))
                totals["affine"] += 1

        elif method == "mxtq":
            t = idx.read_tensor(name, out_dtype=torch.float32)
            arr = t.numpy()
            result = tq_quantize_weight(arr, bits=bits, seed=SEED)
            base = name[:-len(".weight")] if name.endswith(".weight") else name
            add_tensor(f"{base}.tq_packed", np.asarray(result["packed"]))
            add_tensor(f"{base}.tq_norms", np.asarray(result["norms"]))
            add_tensor(f"{base}.tq_bits", np.array([bits], dtype=np.int32))
            totals["mxtq"] += 1
        else:
            raise ValueError(f"unknown method {method} for {name}")

        if (i + 1) % 500 == 0:
            print(f"    processed {i + 1}/{len(weight_keys)}  "
                  f"mxtq={totals['mxtq']} affine={totals['affine']} "
                  f"passthrough={totals['passthrough']}  "
                  f"({time.time() - t_start:.0f}s)", flush=True)

    flush_shard()

    # Rename shards to final count
    for k in range(1, shard_idx):
        old = dst / f"model-{k:05d}-of-XXXXX.safetensors"
        new = dst / f"model-{k:05d}-of-{shard_idx - 1:05d}.safetensors"
        if old.exists():
            old.rename(new)
    final_map = {k: v.replace("XXXXX", f"{shard_idx - 1:05d}") for k, v in shard_map.items()}
    total_bytes = sum((dst / fn).stat().st_size for fn in set(final_map.values()))
    (dst / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_bytes},
        "weight_map": final_map,
    }, indent=2))

    # config.json + jang_config.json
    src_cfg = json.loads((src / "config.json").read_text())
    src_cfg.pop("quantization_config", None)
    # Default global quantization: 8-bit affine group=32 for attention+shared+head+embed.
    # Per-module overrides for mxfp4 routed experts written below.
    quant_cfg = {"group_size": 32, "bits": 8}
    # Routed expert overrides to mxfp4 (only when profile=4 JANGTQ).
    if FORMAT == "jangtq" and profile_bits == 4:
        n_layers = src_cfg.get("num_hidden_layers", 43)
        for L in range(n_layers):
            for proj in ("gate_proj", "down_proj", "up_proj"):
                path = f"model.layers.{L}.mlp.switch_mlp.{proj}"
                quant_cfg[path] = {
                    "group_size": 32, "bits": 4, "mode": "mxfp4",
                }
    elif FORMAT == "jangtq" and profile_bits == 2:
        # 2-bit MXTQ is our own codebook — not mxfp4
        quant_cfg["bits"] = profile_bits
    else:
        # JANG_2L etc — uniform affine at profile_bits
        quant_cfg["bits"] = profile_bits
    src_cfg["quantization"] = quant_cfg
    src_cfg["_name_or_path"] = f"DSV4-Flash-{profile_name}"
    (dst / "config.json").write_text(json.dumps(src_cfg, indent=2))

    (dst / "jang_config.json").write_text(json.dumps({
        "weight_format": "mxfp4_mixed" if FORMAT == "jangtq" and profile_bits == 4 else (
            "affine" if FORMAT == "jang" else "mxtq"
        ),
        "profile": profile_name,
        "mxtq_seed": SEED,
        "source_model": str(src),
        "source_config": {
            "n_routed_experts": src_cfg.get("n_routed_experts"),
            "num_hidden_layers": src_cfg.get("num_hidden_layers"),
            "n_hash_layers": src_cfg.get("num_hash_layers"),
        },
        "mxtq_bits": {
            "routed_expert": profile_bits,
            "attention": 8,
            "shared_expert": 8,
            "embed_tokens": 8,
            "lm_head": 8,
            "norms_router_hc": 16,
        },
        # DSV4-Flash chat + reasoning + tool-parser metadata for runtime wiring.
        # Python loader can use this to auto-wire chat encoding; Swift port
        # reads this to know which encoder to use.
        "model_family": "deepseek_v4",
        "chat": {
            "encoder": "encoding_dsv4",  # Python module in ./encoding/
            "encoder_fn": "encode_messages",
            "chat_template_source": "builtin_encoding_module",
            "has_tokenizer_chat_template": False,  # tokenizer_config.json has no chat_template
            "bos_token": "<｜begin▁of▁sentence｜>",
            "eos_token": "<｜end▁of▁sentence｜>",
            "bos_token_id": 0,
            "eos_token_id": 1,
            "role_tokens": {
                "user": "<｜User｜>",
                "assistant": "<｜Assistant｜>",
                "latest_reminder": "<｜latest_reminder｜>",
            },
            "reasoning": {
                "supported": True,
                "modes": ["chat", "thinking"],
                "default_mode": "chat",
                "thinking_start": "<think>",
                "thinking_end": "</think>",
                # "chat" mode: prompt ends with <Assistant></think> (empty reasoning closed)
                # "thinking" mode: prompt ends with <Assistant><think> (open, model fills)
                "reasoning_effort_levels": ["max", "high", None],
                "drop_earlier_reasoning": True,  # drop_thinking in encode_messages
            },
            "tool_calling": {
                "supported": True,
                "parser": "dsml",  # DeepSeek Markup Language (｜DSML｜...)
                "dsml_token": "｜DSML｜",
                "tool_calls_block": "tool_calls",
                "invoke_block": "invoke",
                "parameter_block": "parameter",
                "tool_output_tag": "tool_result",
            },
            "sampling_defaults": {
                "temperature": 0.6,   # from inference/generate.py default
                "top_p": 0.95,
                "max_new_tokens": 300,
            },
        },
    }, indent=2))

    # Copy aux files (tokenizer, chat template, modeling files if any)
    copied = 0
    for p in src.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") \
                and p.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(p, dst / p.name)
            copied += 1
    # Also copy encoding/ directory (DSV4's Python chat-template impl)
    enc = src / "encoding"
    if enc.is_dir():
        shutil.copytree(enc, dst / "encoding", dirs_exist_ok=True)
        copied += 1
    print(f"[convert] copied {copied} aux files/dirs")

    elapsed = time.time() - t_start
    print(f"\nDONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  mxtq={totals['mxtq']}  affine={totals['affine']}  passthrough={totals['passthrough']}")
    print(f"  output size: {total_bytes / 1e9:.1f} GB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--profile", type=int, default=2, choices=(2, 3, 4, 5, 6, 8),
                    help="Routed-expert bit count.")
    ap.add_argument("--format", default="jangtq", choices=("jang", "jangtq"),
                    help="jang=standard affine everywhere; jangtq=MXTQ for 2-bit routed.")
    args = ap.parse_args()
    global FORMAT
    FORMAT = args.format
    convert(args.src, args.dst, args.profile)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
GLM-5.1-JANGTQ_2L Conversion
Created by Jinho Jang (eric@jangq.ai)

JANGTQ_2L = JANGTQ_1L + two quality upgrades:

  1. FP16 attention / shared_experts / embed / lm_head (vs 8-bit affine in 1L)
     Fixes the MLA bf16 SDPA drift — attention stays at full precision.

  2. AWQ per-channel weight scaling on routed experts (2-bit mxtq, as 1L).
     Scale is stored per-layer (shared across 256 experts per layer) and
     applied at runtime by dividing x by the scale before matmul.

Bundle size: ~215 GB (vs JANGTQ_1L's 191 GB, +23 GB for FP16 non-routed).

Usage:
    python3 -m jang_tools.convert_glm51_jangtq_2l
"""
import sys, json, gc, time, shutil, re
from pathlib import Path

import numpy as np
import mlx.core as mx
from tqdm import tqdm
from safetensors import safe_open
from safetensors.numpy import save_file, load_file


# ── Module-level: only pure helpers, no side effects ────────────────
SEED = 42

# AWQ key mapping: per-expert weight -> per-layer runtime module key.
_EXPERT_TO_SWITCH = re.compile(r"\.experts\.\d+\.")


def awq_key_for(tensor_name: str) -> str:
    """Map an FP8 per-expert weight name to the runtime module name the
    AWQ scale is keyed under."""
    base = tensor_name.replace(".weight", "")
    return _EXPERT_TO_SWITCH.sub(".switch_mlp.", base)


def get_bits_and_method(tensor_name: str) -> tuple:
    """Returns (bits, method) where method is one of 'mxtq',
    'mxtq_awq', 'affine', or 'passthrough' for JANGTQ_2L."""
    name = tensor_name.lower()
    if "norm" in name or tensor_name.endswith(".bias"):
        return (16, "passthrough")
    # Router / gate (not attention's gate_proj — MoE gate only)
    if ".gate." in name and "gate_proj" not in name:
        return (16, "passthrough")
    # Embeddings / lm_head -> FP16
    if "embed_tokens" in name or "lm_head" in name:
        return (16, "passthrough")
    # Attention (MLA) -> FP16
    if "self_attn" in name:
        return (16, "passthrough")
    # Shared experts -> FP16 (per FP8 source: `shared_experts` plural)
    if "shared_expert" in name:  # matches both singular + plural
        return (16, "passthrough")
    # Routed experts -> 2-bit mxtq with AWQ scaling
    if "experts" in name or "switch_mlp" in name:
        return (2, "mxtq_awq")
    return (16, "passthrough")


def _load_awq_scales(path):
    if path is None:
        print("  no AWQ scales file provided; falling back to RTN", flush=True)
        return {}
    if not path.exists():
        print(f"  WARNING: no AWQ scales at {path}; falling back to RTN",
              flush=True)
        return {}
    scales = load_file(str(path))
    print(f"  loaded AWQ scales for {len(scales)} modules", flush=True)
    return {k: v.astype(np.float32) for k, v in scales.items()}


def _setup_capabilities(jang_config, config, out_path):
    """Fill jang.capabilities for vmlx compatibility. Module-scoped
    import to avoid heavy imports when only using helpers."""
    from jang_tools.capabilities import build_capabilities, verify_directory
    caps = build_capabilities(jang_config, config, out_path)
    if caps is not None:
        jang_config["capabilities"] = caps
        print(f"  capabilities: family={caps['family']} "
              f"reasoning={caps['reasoning_parser']} "
              f"tool={caps['tool_parser']} cache={caps['cache_type']} "
              f"modality={caps['modality']}",
              flush=True)
    return caps


# ── Main conversion (only runs when called explicitly) ──────────────

def main(src=None, out=None, awq_scales_file=None):
    # Lazy imports of heavy deps (only when actually running conversion).
    sys.path.insert(0, "/opt/homebrew/lib/python3.14/site-packages")
    from jang_tools.fp8 import load_fp8_tensor
    from jang_tools.calibrate import _load_bf16_tensor
    from jang_tools.turboquant.linear import tq_quantize_weight

    if not src or not out:
        raise SystemExit(
            "convert_glm51_jangtq_2l.main(): src and out paths are required.\n"
            "  src  = GLM-5.1 FP8 source model directory\n"
            "  out  = output directory for the JANGTQ_2L bundle\n"
            "  awq_scales_file = optional pre-computed AWQ scales .safetensors"
        )
    SRC = Path(src)
    OUT = Path(out)
    AWQ_FILE = Path(awq_scales_file) if awq_scales_file else None
    OUT.mkdir(parents=True, exist_ok=True)

    with open(SRC / "config.json") as _f:
        config = json.load(_f)
    tc = config.get("text_config", config)
    n_layers = tc["num_hidden_layers"]
    first_dense = tc.get("first_k_dense_replace", 0)
    n_experts = tc.get("n_routed_experts", 256)

    awq_scales = _load_awq_scales(AWQ_FILE)

    print("=" * 60)
    print("  GLM-5.1-JANGTQ_2L Conversion")
    print("  Created by Jinho Jang (eric@jangq.ai)")
    print("=" * 60)
    print(f"  Source: {SRC}")
    print(f"  Output: {OUT}")
    print(f"  Layers: {n_layers} ({first_dense} dense, {n_layers - first_dense} MoE)")
    print(f"  Experts: {n_experts}")
    print(f"  Profile: JANGTQ_2L")
    print(f"    attn=fp16, shared_experts=fp16, embed/lm_head=fp16")
    print(f"    routed=mxtq-2 + AWQ per-channel scale")
    print(f"  AWQ scales: {len(awq_scales)} module keys", flush=True)

    # Scan source tensors
    print("\n  Scanning source...", flush=True)
    all_tensors = []
    for sf in sorted(SRC.glob("model-*.safetensors")):
        with safe_open(str(sf), framework="numpy") as f:
            for k in f.keys():
                if k.endswith("_scale_inv"):
                    continue
                shape = list(f.get_slice(k).get_shape())
                all_tensors.append((k, shape, sf))
    print(f"  Found {len(all_tensors)} tensors", flush=True)

    # Conversion loop
    shard_idx = 0
    shard_tensors = {}
    shard_bytes = 0
    MAX_SHARD = 1_000_000_000
    totals = {"mxtq": 0, "mxtq_awq": 0, "affine": 0, "passthrough": 0, "awq_applied": 0}
    shard_map = {}

    def flush_shard():
        nonlocal shard_idx, shard_tensors, shard_bytes
        if not shard_tensors:
            return
        shard_idx += 1
        fname = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        save_file(shard_tensors, str(OUT / fname))
        for k in shard_tensors:
            shard_map[k] = fname
        print(f"    Shard {shard_idx}: {len(shard_tensors)} tensors, "
              f"{shard_bytes/1e9:.1f} GB", flush=True)
        shard_tensors = {}
        shard_bytes = 0

    def add_tensor(name, arr):
        nonlocal shard_tensors, shard_bytes
        shard_tensors[name] = arr
        shard_bytes += arr.nbytes
        if shard_bytes >= MAX_SHARD:
            flush_shard()

    print("\n  Converting...", flush=True)
    awq_scales_written = set()
    for tensor_name, shape, sf_path in tqdm(all_tensors, desc="  Processing"):
        bits, method = get_bits_and_method(tensor_name)

        with safe_open(str(sf_path), framework="numpy") as f:
            scale_key = tensor_name + "_scale_inv"
            try:
                fp8_scale = f.get_tensor(scale_key)
            except Exception:
                fp8_scale = None
            try:
                tensor = load_fp8_tensor(sf_path, tensor_name, shape, fp8_scale)
            except Exception:
                try:
                    tensor = f.get_tensor(tensor_name)
                    if not isinstance(tensor, np.ndarray):
                        tensor = np.array(tensor)
                except (TypeError, Exception):
                    tensor = _load_bf16_tensor(sf_path, tensor_name, shape)

        tensor = tensor.astype(np.float32) if tensor.dtype != np.float32 else tensor

        if method == "passthrough":
            t16 = tensor.astype(np.float16) if tensor.dtype != np.float16 else tensor
            add_tensor(tensor_name, t16)
            totals["passthrough"] += 1

        elif method == "affine":
            w = mx.array(tensor.astype(np.float16))
            qw, qs, qb = mx.quantize(w, group_size=64, bits=bits)
            base = tensor_name.replace(".weight", "") if tensor_name.endswith(".weight") \
                else tensor_name
            add_tensor(f"{base}.weight", np.array(qw))
            add_tensor(f"{base}.scales", np.array(qs).astype(np.float16))
            add_tensor(f"{base}.biases", np.array(qb).astype(np.float16))
            totals["affine"] += 1

        elif method in ("mxtq", "mxtq_awq"):
            scale = None
            if method == "mxtq_awq":
                key = awq_key_for(tensor_name)
                scale = awq_scales.get(key)
            if scale is not None:
                if scale.shape[0] != tensor.shape[1]:
                    raise RuntimeError(
                        f"AWQ scale shape mismatch for {tensor_name}: "
                        f"tensor in={tensor.shape[1]}, scale={scale.shape[0]}"
                    )
                tensor_scaled = tensor * scale[None, :]
                totals["awq_applied"] += 1
            else:
                tensor_scaled = tensor

            result = tq_quantize_weight(tensor_scaled, bits=bits, seed=SEED)
            base = tensor_name.replace(".weight", "") if tensor_name.endswith(".weight") \
                else tensor_name
            add_tensor(f"{base}.tq_packed", result["packed"])
            add_tensor(f"{base}.tq_norms", result["norms"])
            add_tensor(f"{base}.tq_bits", np.array([bits], dtype=np.uint8))
            # Write AWQ scale once per per-layer module (not per-expert) to save space.
            if scale is not None:
                awq_key = awq_key_for(tensor_name)
                if awq_key not in awq_scales_written:
                    add_tensor(f"{awq_key}.awq_scale",
                               scale.astype(np.float16))
                    awq_scales_written.add(awq_key)

            if method == "mxtq_awq":
                totals["mxtq_awq"] += 1
            else:
                totals["mxtq"] += 1

        del tensor
        if (totals["mxtq"] + totals["mxtq_awq"]) % 100 == 0 and \
           (totals["mxtq"] + totals["mxtq_awq"]) > 0:
            gc.collect()

    flush_shard()

    # Index: write placeholder, then rename shards, then final.
    print(f"\n  Writing index ({shard_idx} shards)...", flush=True)
    index = {"metadata": {"total_size": 0}, "weight_map": shard_map}
    with open(OUT / "model.safetensors.index.json", "w") as _f:
        json.dump(index, _f, indent=2)
    for i in range(1, shard_idx + 1):
        old = OUT / f"model-{i:05d}-of-XXXXX.safetensors"
        new = OUT / f"model-{i:05d}-of-{shard_idx:05d}.safetensors"
        if old.exists():
            old.rename(new)
    shard_map_final = {k: v.replace("XXXXX", f"{shard_idx:05d}")
                        for k, v in shard_map.items()}
    index["weight_map"] = shard_map_final
    with open(OUT / "model.safetensors.index.json", "w") as _f:
        json.dump(index, _f, indent=2)

    # Config / jang_config
    config.pop("quantization_config", None)
    config["quantization"] = {"group_size": 64, "bits": 2}
    jang_config = {
        "weight_format": "mxtq",
        "profile": "JANGTQ_2L",
        "mxtq_seed": SEED,
        "mxtq_bits": {
            "attention": 16,
            "shared_expert": 16,
            "routed_expert": 2,
            "embed_tokens": 16,
            "lm_head": 16,
        },
        "method": "passthrough+mxtq_awq",
        "awq_enabled": bool(awq_scales),
        "awq_alpha_searched": True,
    }
    _setup_capabilities(jang_config, config, OUT)
    config["jang"] = jang_config
    with open(OUT / "config.json", "w") as _f:
        json.dump(config, _f, indent=2)
    with open(OUT / "jang_config.json", "w") as _f:
        json.dump(jang_config, _f, indent=2)

    # Tokenizer / chat template / modeling py copies
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "generation_config.json", "chat_template.jinja",
              "chat_template.json", "merges.txt", "vocab.json",
              "preprocessor_config.json", "video_preprocessor_config.json",
              "configuration.json"]:
        src_f = SRC / f
        if src_f.exists():
            shutil.copy2(str(src_f), str(OUT / f))
    for f in SRC.glob("*.py"):
        shutil.copy2(str(f), str(OUT / f.name))

    # Osaurus compat
    _tok_cfg = OUT / "tokenizer_config.json"
    if _tok_cfg.exists():
        try:
            with open(_tok_cfg) as _f:
                _tc = json.load(_f)
            if _tc.get("tokenizer_class") == "TokenizersBackend":
                _tc["tokenizer_class"] = "ChatGLMTokenizer"
                with open(_tok_cfg, "w") as _f:
                    json.dump(_tc, _f, indent=2)
                print("  [osaurus-fix] tokenizer_class -> ChatGLMTokenizer",
                      flush=True)
        except Exception as _e:
            print(f"  [osaurus-fix] skipped: {_e}", flush=True)

    # Sidecar
    print(f"\n  Building jangtq_runtime.safetensors sidecar...", flush=True)
    try:
        from jang_tools.build_jangtq_sidecar import main as _build_sidecar
        _saved_argv = sys.argv
        sys.argv = ["build_jangtq_sidecar", str(OUT)]
        try:
            _build_sidecar()
        finally:
            sys.argv = _saved_argv
    except (Exception, SystemExit) as _e:
        print(f"  [sidecar] FAILED: {_e}", flush=True)

    print(f"\n{'='*60}")
    print(f"  DONE — GLM-5.1-JANGTQ_2L")
    print(f"  Output: {OUT}")
    for k, v in totals.items():
        print(f"    {k}: {v}")
    du = sum(f.stat().st_size for f in OUT.glob("*") if f.is_file())
    print(f"  Total size: {du/1e9:.1f} GB")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()

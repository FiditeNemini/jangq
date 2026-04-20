"""
MiniMax M2.7 → JANGTQ Conversion
Created by Jinho Jang (eric@jangq.ai)

Mixed-precision TurboQuant for MiniMax: MXTQ 2-bit experts, affine 8-bit attention.
Output is loadable via load_jangtq.py with TurboQuantLinear Metal kernel.
"""
import sys, json, gc, shutil
import argparse
import numpy as np
import mlx.core as mx
from pathlib import Path
from tqdm import tqdm
from safetensors import safe_open
from safetensors.numpy import save_file

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--progress", choices=["json", "off"], default="off")
_ap.add_argument("--quiet-text", action="store_true")
_args, _rest = _ap.parse_known_args()
sys.argv = [sys.argv[0]] + _rest

from jang_tools.progress import ProgressEmitter
progress = ProgressEmitter(
    json_to_stderr=(_args.progress == "json"),
    quiet_text=_args.quiet_text,
)

from jang_tools.fp8 import load_fp8_tensor
from jang_tools.calibrate import _load_bf16_tensor
from jang_tools.turboquant.linear import tq_quantize_weight

# === Config ===
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Volumes/EricsLLMDrive/MiniMax-M2.7-FP8")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/Volumes/EricsLLMDrive/MiniMax-M2.7-JANGTQ2")
PROFILE = sys.argv[3] if len(sys.argv) > 3 else "JANGTQ2"
SEED = 42

try:
    # Canonical profile names: JANGTQ{N}. Legacy JANGTQ_2L / _4M etc accepted.
    _PROFILE_BITS = {
        "JANGTQ2": 2, "JANGTQ_2L": 2, "JANGTQ_2S": 2,
        "JANGTQ3": 3, "JANGTQ_3L": 3, "JANGTQ_3S": 3,
        "JANGTQ4": 4, "JANGTQ_4M": 4, "JANGTQ_4K": 4,
    }
    _PROFILE_NORM = PROFILE.upper()
    if _PROFILE_NORM not in _PROFILE_BITS:
        raise ValueError(f"unknown profile {PROFILE!r}; expected one of {sorted(_PROFILE_BITS)}")
    EXPERT_BITS = _PROFILE_BITS[_PROFILE_NORM]
    PROFILE = f"JANGTQ{EXPERT_BITS}"

    OUT.mkdir(parents=True, exist_ok=True)

    with open(SRC / "config.json") as f:
        config = json.load(f)
    n_layers = config.get("num_hidden_layers", 62)
    n_experts = config.get("num_local_experts", 256)


    # === Bit assignment ===
    # attn=8 affine, embed/lm_head=8 affine, routed expert MLP=EXPERT_BITS mxtq
    # (2/3/4 by profile), gate/router/norms=fp16.
    def get_bits_and_method(tensor_name):
        name = tensor_name.lower()

        # Norms (1D), biases — passthrough
        if "norm" in name or tensor_name.endswith(".bias") or "e_score_correction_bias" in name:
            return (16, "passthrough")

        # MoE router/gate (small but precision-critical) — passthrough float16
        # Match .gate.weight but NOT gate_proj
        if (".gate.weight" in name or name.endswith(".gate.weight")) and "gate_proj" not in name:
            return (16, "passthrough")

        # Embeddings, lm_head — affine 8-bit
        if "embed_tokens" in name or "lm_head" in name:
            return (8, "affine")

        # Attention Q/K/V/O — affine 8-bit
        if "self_attn" in name and (".weight" in name or "_proj" in name):
            return (8, "affine")

        # Shared expert (none on MiniMax, but handle it)
        if "shared_expert" in name:
            return (8, "affine")

        # Routed expert MLP (w1/w2/w3 = gate/down/up) — MXTQ EXPERT_BITS
        if "experts" in name and (".w1" in name or ".w2" in name or ".w3" in name):
            return (EXPERT_BITS, "mxtq")

        # Default — affine 8-bit (catch unmapped)
        return (8, "affine")


    print("=" * 60)
    print(f"  MiniMax M2.7 → {PROFILE} JANGTQ Conversion")
    print(f"  Created by Jinho Jang (eric@jangq.ai)")
    print("=" * 60)
    print(f"  Source: {SRC}")
    print(f"  Output: {OUT}")
    print(f"  Layers: {n_layers}, Experts: {n_experts}")
    print(f"  Profile: attn=affine-8, expert=mxtq-{EXPERT_BITS}, gate=fp16")
    print(flush=True)

    # Scan tensors
    progress.phase(1, 3, "scan")
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

    # === Process ===
    shard_idx = 0
    shard_tensors = {}
    shard_bytes = 0
    MAX_SHARD = 1_000_000_000  # 1 GB
    total_mxtq = 0
    total_affine = 0
    total_passthrough = 0
    shard_map = {}


    def flush_shard():
        global shard_idx, shard_tensors, shard_bytes
        if not shard_tensors:
            return
        shard_idx += 1
        fname = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        save_file(shard_tensors, str(OUT / fname))
        for k in shard_tensors:
            shard_map[k] = fname
        print(f"    Shard {shard_idx}: {len(shard_tensors)} tensors, {shard_bytes/1e9:.1f} GB", flush=True)
        shard_tensors = {}
        shard_bytes = 0


    def add_tensor(name, arr):
        global shard_bytes
        shard_tensors[name] = arr
        shard_bytes += arr.nbytes
        if shard_bytes >= MAX_SHARD:
            flush_shard()


    # === Resume: scan existing shards ===
    done_keys = set()
    existing_shards = sorted(OUT.glob("model-*-of-XXXXX.safetensors"))
    if existing_shards:
        print(f"\n  Resume: found {len(existing_shards)} existing shards", flush=True)
        import struct as _struct
        for sf in existing_shards:
            # Read safetensors header to get tensor names
            with open(sf, "rb") as f:
                hsize = _struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hsize))
            fname = sf.name
            for k in hdr:
                if k == "__metadata__":
                    continue
                done_keys.add(k)
                shard_map[k] = fname
            # Track shard_idx
            idx_str = sf.name.split("-")[1]
            shard_idx = max(shard_idx, int(idx_str))
        print(f"  Resume: {len(done_keys)} keys already written, continuing from shard {shard_idx + 1}", flush=True)


    def is_already_done(source_name, method):
        """Check if all output keys for this source tensor exist."""
        if method == "passthrough":
            return source_name in done_keys
        elif method == "affine":
            base = source_name.replace(".weight", "") if source_name.endswith(".weight") else source_name
            return (f"{base}.weight" in done_keys and
                    f"{base}.scales" in done_keys and
                    f"{base}.biases" in done_keys)
        elif method == "mxtq":
            base = source_name.replace(".weight", "") if source_name.endswith(".weight") else source_name
            return (f"{base}.tq_packed" in done_keys and
                    f"{base}.tq_norms" in done_keys and
                    f"{base}.tq_bits" in done_keys)
        return False


    progress.phase(2, 3, "convert")
    print("\n  Converting...", flush=True)
    skipped_resume = 0
    for tensor_name, shape, sf_path in tqdm(all_tensors, desc="  Processing"):
        bits, method = get_bits_and_method(tensor_name)

        # Skip if already done from previous run
        if done_keys and is_already_done(tensor_name, method):
            skipped_resume += 1
            if method == "mxtq":
                total_mxtq += 1
            elif method == "affine":
                total_affine += 1
            else:
                total_passthrough += 1
            continue

        # Load tensor
        with safe_open(str(sf_path), framework="numpy") as f:
            scale_key = tensor_name + "_scale_inv"
            try:
                scale = f.get_tensor(scale_key)
            except Exception:
                scale = None
            try:
                tensor = load_fp8_tensor(sf_path, tensor_name, shape, scale)
            except Exception:
                try:
                    tensor = f.get_tensor(tensor_name)
                    if not isinstance(tensor, np.ndarray):
                        tensor = np.array(tensor)
                except Exception:
                    tensor = _load_bf16_tensor(sf_path, tensor_name, shape)

        if tensor.dtype != np.float32:
            tensor = tensor.astype(np.float32)

        if method == "passthrough":
            t16 = tensor.astype(np.float16)
            add_tensor(tensor_name, t16)
            total_passthrough += 1

        elif method == "affine":
            w = mx.array(tensor.astype(np.float16))
            qw, qs, qb = mx.quantize(w, group_size=64, bits=bits)
            base = tensor_name.replace(".weight", "") if tensor_name.endswith(".weight") else tensor_name
            add_tensor(f"{base}.weight", np.array(qw))
            add_tensor(f"{base}.scales", np.array(qs).astype(np.float16))
            add_tensor(f"{base}.biases", np.array(qb).astype(np.float16))
            total_affine += 1
            del w, qw, qs, qb

        elif method == "mxtq":
            # MiniMax expert tensors are per-expert 2D (experts.N.w1.weight)
            result = tq_quantize_weight(tensor, bits=bits, seed=SEED)
            base = tensor_name.replace(".weight", "") if tensor_name.endswith(".weight") else tensor_name
            add_tensor(f"{base}.tq_packed", result["packed"])
            add_tensor(f"{base}.tq_norms", result["norms"])
            add_tensor(f"{base}.tq_bits", np.array([bits], dtype=np.uint8))
            total_mxtq += 1

        del tensor
        if (total_mxtq + total_affine) % 200 == 0:
            gc.collect()

    flush_shard()

    if skipped_resume > 0:
        print(f"\n  Resume: skipped {skipped_resume} tensors from previous run", flush=True)

    # Rename shards
    progress.phase(3, 3, "write")
    print(f"\n  Renaming {shard_idx} shards...", flush=True)
    for i in range(1, shard_idx + 1):
        old = OUT / f"model-{i:05d}-of-XXXXX.safetensors"
        new = OUT / f"model-{i:05d}-of-{shard_idx:05d}.safetensors"
        if old.exists():
            old.rename(new)
    shard_map = {k: v.replace("XXXXX", f"{shard_idx:05d}") for k, v in shard_map.items()}

    # Write index
    index = {"metadata": {"format": "jangtq", "total_size": sum(v.nbytes for v in shard_tensors.values())},
             "weight_map": shard_map}
    # M125 (iter 48): use a context manager so the write is flushed + closed
    # deterministically. Prior `json.dump(..., open(p, "w"))` relied on CPython
    # refcount-GC to close the file promptly — if the converter crashed or
    # was backgrounded between the dump and GC, a partial JSON write could
    # land on disk and brick the model.
    with open(OUT / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Write config
    config.pop("quantization_config", None)
    config["quantization"] = {"group_size": 64, "bits": EXPERT_BITS}
    config["weight_format"] = "mxtq"
    config["mxtq_bits"] = EXPERT_BITS
    with open(OUT / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Write jang_config
    jang_config = {
        "version": 2,
        "weight_format": "mxtq",
        "profile": PROFILE,
        "source_model": {
            "name": "MiniMax-M2.7",
            "org": "MiniMaxAI",
            "architecture": "minimax_m2",
        },
        "mxtq_seed": SEED,
        "mxtq_bits": {
            "attention": 8,
            "shared_expert": 8,
            "routed_expert": EXPERT_BITS,
            "embed_tokens": 8,
            "lm_head": 8,
        },
        "quantization": {
            "method": "affine+mxtq",
            "group_size": 64,
            "bits_default": EXPERT_BITS,
        },
    }
    # Stamp Tier-1 capabilities for vmlx CapabilityDetector. Idempotent.
    from jang_tools.capabilities import build_capabilities
    caps = build_capabilities(jang_config, config, OUT)
    if caps is not None:
        jang_config["capabilities"] = caps
        print(f"  capabilities: family={caps['family']} reasoning={caps['reasoning_parser']} "
              f"tool={caps['tool_parser']} cache={caps['cache_type']} modality={caps['modality']}",
              flush=True)
    else:
        print("  WARNING: could not resolve capabilities — vmlx will fall back to "
              "silver/bronze. Add the family to jang_tools/capabilities.py::FAMILY_MAP.",
              flush=True)

    with open(OUT / "jang_config.json", "w") as f:
        json.dump(jang_config, f, indent=2)

    # Validate the final jang_config — catch schema drift / typos / missing keys.
    from jang_tools.capabilities import verify_directory
    _ok, _msg = verify_directory(OUT)
    print(f"  verify: {_msg}")
    if not _ok:
        raise SystemExit(f"capabilities verify failed: {_msg}")

    # Copy tokenizer / chat-template / VL preprocessor / custom .py files.
    # Downstream consumers (mlx-vlm, transformers) need the preprocessor configs
    # to do image/video inference — if they're missing they have to fish them
    # out of the source HF repo by hand.
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "generation_config.json", "chat_template.jinja",
              "chat_template.json", "merges.txt", "vocab.json",
              "preprocessor_config.json", "video_preprocessor_config.json",
              "configuration.json",
              "modeling_minimax_m2.py", "configuration_minimax_m2.py"]:
        src_f = SRC / f
        if src_f.exists():
            shutil.copy2(str(src_f), str(OUT / f))

    # --- Osaurus / swift-transformers compatibility fix ---------------------------
    # Remap tokenizer_class: "TokenizersBackend" → a concrete class swift-transformers
    # can parse. For MiniMax (Mixtral-family vocab) use GPT2Tokenizer as the stable
    # fallback; swift-transformers routes it through its generic BPE path. Change
    # this if an explicit MiniMaxTokenizer class is added upstream.
    _tok_cfg = OUT / "tokenizer_config.json"
    if _tok_cfg.exists():
        try:
            with open(_tok_cfg) as f:
                _tc = json.load(f)
            if _tc.get("tokenizer_class") == "TokenizersBackend":
                _tc["tokenizer_class"] = "GPT2Tokenizer"
                with open(_tok_cfg, "w") as f:
                    json.dump(_tc, f, indent=2)
                print("  [osaurus-fix] tokenizer_class: TokenizersBackend → GPT2Tokenizer", flush=True)
        except Exception as _e:
            print(f"  [osaurus-fix] skipped: {_e}", flush=True)

    # Build Swift runtime sidecar so Swift runtimes (vmlx-swift-lm, vmlxctl,
    # Osaurus) can load without fatalError. Python loader doesn't need it, but
    # uploading without it is a footgun — see research/JANGTQ-REFERENCE.md.
    print(f"\n  Building jangtq_runtime.safetensors sidecar...")
    try:
        from jang_tools.build_jangtq_sidecar import main as _build_sidecar
        _saved_argv = sys.argv
        sys.argv = ["build_jangtq_sidecar", str(OUT)]
        try:
            _build_sidecar()
        finally:
            sys.argv = _saved_argv
    except (Exception, SystemExit) as _e:
        print(f"  [sidecar] FAILED: {_e} — run `python3 -m jang_tools.build_jangtq_sidecar {OUT}` manually before upload", flush=True)

    print(f"\n  Done!")
    print(f"  MXTQ tensors: {total_mxtq}")
    print(f"  Affine tensors: {total_affine}")
    print(f"  Passthrough: {total_passthrough}")
    print(f"  Output: {OUT}")
    progress.done(ok=True, output=str(OUT))

except Exception as _exc:
    progress.done(ok=False, error=f"{type(_exc).__name__}: {_exc}")
    raise

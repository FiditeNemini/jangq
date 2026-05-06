"""
Ling-2.6-flash (Bailing-V2.5 hybrid) → JANGTQ Conversion
Created by Jinho Jang (eric@jangq.ai)

Mixed-precision TurboQuant for Bailing-V2.5 hybrid (MLA + Lightning Linear
Attention + MoE + MTP):
  routed-expert MLPs   → MXTQ bits per profile (2 / 3 / 4)
  attention (MLA + LA) → affine 8-bit
  shared-expert MLP    → affine 8-bit
  dense MLP (layer 0)  → affine 8-bit
  MTP eh_proj          → affine 8-bit
  embed / lm_head      → affine 8-bit
  norms / router gate / expert_bias / slope → fp16 / fp32 passthrough

Output is loadable via load_jangtq.py + the bailing_hybrid model class.
"""
import sys
import json
import gc
import shutil
import argparse
import struct as _struct
import numpy as np
import mlx.core as mx
from pathlib import Path
from tqdm import tqdm
from safetensors import safe_open
from safetensors.numpy import save_file


_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--progress", choices=["json", "off"], default="off")
_ap.add_argument("--quiet-text", action="store_true")
# Codex 2026-05-05 #2: post-process per-expert output through rebundle()
# so the SHIPPED bundle is JANGTQ-PRESTACK-SPEC compliant. Default ON.
_ap.add_argument("--no-prestack", action="store_true",
                 help="skip rebundle post-process (debug only; produces non-spec bundle)")
_args, _rest = _ap.parse_known_args()
sys.argv = [sys.argv[0]] + _rest

from jang_tools.progress import ProgressEmitter

progress = ProgressEmitter(
    json_to_stderr=(_args.progress == "json"),
    quiet_text=_args.quiet_text,
)

from jang_tools.calibrate import _load_bf16_tensor
from jang_tools.turboquant.linear import tq_quantize_weight


# === CLI ===
if len(sys.argv) < 3:
    print(
        "usage: python -m jang_tools.convert_ling_jangtq <src_bf16_dir> <out_dir> [profile]\n"
        "  <src_bf16_dir>  path to a Ling-2.6-flash (bailing_hybrid) bf16 source\n"
        "  <out_dir>       output directory for the JANGTQ bundle\n"
        "  [profile]       JANGTQ2 (default), JANGTQ3, or JANGTQ4",
        file=sys.stderr,
    )
    sys.exit(2)

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
PROFILE = sys.argv[3] if len(sys.argv) > 3 else "JANGTQ2"
SEED = 42

try:
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
    n_layers = config.get("num_hidden_layers", 32)
    n_experts = config.get("num_experts", 256)
    n_mtp = config.get("num_nextn_predict_layers", 0)
    hidden_size = config.get("hidden_size", 4096)
    moe_intermediate = config.get("moe_intermediate_size", 1024)
    intermediate = config.get("intermediate_size", 9216)
    layer_group_size = config.get("layer_group_size", 8)
    first_k_dense_replace = config.get("first_k_dense_replace", 1)

    print("=" * 60)
    print(f"  Ling-2.6-flash (bailing_hybrid) → {PROFILE} JANGTQ Conversion")
    print(f"  Created by Jinho Jang (eric@jangq.ai)")
    print("=" * 60)
    print(f"  Source: {SRC}")
    print(f"  Output: {OUT}")
    print(
        f"  Layers: {n_layers} (+{n_mtp} MTP)  Experts: {n_experts}  "
        f"layer_group_size={layer_group_size}  first_dense={first_k_dense_replace}"
    )
    print(
        f"  Profile: routed_expert=mxtq-{EXPERT_BITS}  attn/shared/dense_mlp/embed/lm_head=affine-8  "
        f"norms/gate/biases=fp16/fp32"
    )
    print(flush=True)

    # === Bit assignment ===
    def get_bits_and_method(name: str):
        n = name

        # --- pure-passthrough (norms / scalar buffers / 1-D weights / biases) ---
        # Catch every *.norm, *.layernorm, *_layernorm, *.g_norm, model.norm, MTP
        # final/e/h_norm, and every q/k/kv layernorm. Also bias / expert_bias.
        if (
            "norm" in n
            or n.endswith(".bias")
            or "expert_bias" in n
            or "e_score_correction_bias" in n
        ):
            return (16, "passthrough")

        # --- MoE router gate.weight — passthrough fp16 (sigmoid is sensitive) ---
        # Match `....mlp.gate.weight` precisely; do NOT match `gate_proj.weight`.
        if n.endswith(".mlp.gate.weight"):
            return (16, "passthrough")

        # --- routed expert MLPs: MXTQ at EXPERT_BITS ---
        # `model.layers.{l}.mlp.experts.{e}.{gate_proj|up_proj|down_proj}.weight`
        if ".mlp.experts." in n and (
            ".gate_proj." in n or ".up_proj." in n or ".down_proj." in n
        ):
            return (EXPERT_BITS, "mxtq")

        # --- shared expert: affine 8-bit ---
        if ".mlp.shared_experts." in n:
            return (8, "affine")

        # --- dense MLP (layer 0): affine 8-bit ---
        # Matches `model.layers.0.mlp.{gate_proj|up_proj|down_proj}.weight`
        # — no `.experts.` segment.
        if ".mlp." in n and ".experts." not in n and (
            ".gate_proj." in n or ".up_proj." in n or ".down_proj." in n
        ):
            return (8, "affine")

        # --- attention (MLA + Linear) projections: affine 8-bit ---
        # MLA: q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, dense
        # Linear: query_key_value, dense, g_proj
        if ".attention." in n and n.endswith(".weight"):
            return (8, "affine")

        # --- MTP eh_proj: affine 8-bit ---
        if ".eh_proj.weight" in n:
            return (8, "affine")

        # --- embed / lm_head: affine 8-bit ---
        if "word_embeddings" in n or n == "lm_head.weight":
            return (8, "affine")

        # --- model.norm.weight already caught by `norm in n` rule above ---

        # --- catchall: affine 8-bit (so we don't silently passthrough huge mats) ---
        return (8, "affine")

    # === Scan source ===
    progress.phase(1, 3, "scan")
    print("\n  Scanning source...", flush=True)
    all_tensors = []
    for sf in sorted(SRC.glob("model-*.safetensors")):
        with safe_open(str(sf), framework="numpy") as f:
            for k in f.keys():
                shape = list(f.get_slice(k).get_shape())
                all_tensors.append((k, shape, sf))
    print(f"  Found {len(all_tensors)} tensors", flush=True)

    # === Stream-quantize, sharded write ===
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
        print(
            f"    Shard {shard_idx}: {len(shard_tensors)} tensors, {shard_bytes/1e9:.1f} GB",
            flush=True,
        )
        shard_tensors = {}
        shard_bytes = 0

    def add_tensor(name, arr):
        global shard_bytes
        shard_tensors[name] = arr
        shard_bytes += arr.nbytes
        if shard_bytes >= MAX_SHARD:
            flush_shard()

    # === Resume support ===
    done_keys = set()
    existing_shards = sorted(OUT.glob("model-*-of-XXXXX.safetensors"))
    if existing_shards:
        print(f"\n  Resume: found {len(existing_shards)} existing shards", flush=True)
        for sf in existing_shards:
            with open(sf, "rb") as f:
                hsize = _struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hsize))
            fname = sf.name
            for k in hdr:
                if k == "__metadata__":
                    continue
                done_keys.add(k)
                shard_map[k] = fname
            idx_str = sf.name.split("-")[1]
            shard_idx = max(shard_idx, int(idx_str))
        print(
            f"  Resume: {len(done_keys)} keys already written, "
            f"continuing from shard {shard_idx + 1}",
            flush=True,
        )

    def is_already_done(source_name, method):
        if method == "passthrough":
            return source_name in done_keys
        base = source_name.replace(".weight", "") if source_name.endswith(".weight") else source_name
        if method == "affine":
            return (
                f"{base}.weight" in done_keys
                and f"{base}.scales" in done_keys
                and f"{base}.biases" in done_keys
            )
        if method == "mxtq":
            return (
                f"{base}.tq_packed" in done_keys
                and f"{base}.tq_norms" in done_keys
                and f"{base}.tq_bits" in done_keys
            )
        return False

    # === Convert ===
    progress.phase(2, 3, "convert")
    print("\n  Converting...", flush=True)
    skipped_resume = 0
    for tensor_name, shape, sf_path in tqdm(all_tensors, desc="  Processing"):
        bits, method = get_bits_and_method(tensor_name)

        if done_keys and is_already_done(tensor_name, method):
            skipped_resume += 1
            if method == "mxtq":
                total_mxtq += 1
            elif method == "affine":
                total_affine += 1
            else:
                total_passthrough += 1
            continue

        # Bf16 source → load via the project's existing bf16 loader.
        try:
            with safe_open(str(sf_path), framework="numpy") as f:
                tensor = f.get_tensor(tensor_name)
            if not isinstance(tensor, np.ndarray):
                tensor = np.array(tensor)
        except (TypeError, ValueError):
            # numpy can't decode bf16 → fall back to manual reader.
            tensor = _load_bf16_tensor(sf_path, tensor_name, shape)

        if tensor.dtype != np.float32:
            tensor = tensor.astype(np.float32)

        if method == "passthrough":
            # Keep expert_bias at fp32 (source dtype). Everything else fp16.
            if "expert_bias" in tensor_name:
                add_tensor(tensor_name, tensor.astype(np.float32))
            else:
                add_tensor(tensor_name, tensor.astype(np.float16))
            total_passthrough += 1

        elif method == "affine":
            w = mx.array(tensor.astype(np.float16))
            qw, qs, qb = mx.quantize(w, group_size=64, bits=bits)
            base = (
                tensor_name.replace(".weight", "")
                if tensor_name.endswith(".weight")
                else tensor_name
            )
            add_tensor(f"{base}.weight", np.array(qw))
            add_tensor(f"{base}.scales", np.array(qs).astype(np.float16))
            add_tensor(f"{base}.biases", np.array(qb).astype(np.float16))
            total_affine += 1
            del w, qw, qs, qb

        elif method == "mxtq":
            result = tq_quantize_weight(tensor, bits=bits, seed=SEED)
            base = (
                tensor_name.replace(".weight", "")
                if tensor_name.endswith(".weight")
                else tensor_name
            )
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

    # === Rename to final shard count ===
    progress.phase(3, 3, "write")
    print(f"\n  Renaming {shard_idx} shards...", flush=True)
    for i in range(1, shard_idx + 1):
        old = OUT / f"model-{i:05d}-of-XXXXX.safetensors"
        new = OUT / f"model-{i:05d}-of-{shard_idx:05d}.safetensors"
        if old.exists():
            old.rename(new)
    shard_map = {k: v.replace("XXXXX", f"{shard_idx:05d}") for k, v in shard_map.items()}

    # === Index ===
    # Compute total_size from on-disk shard files (shard_tensors is empty after
    # the final flush_shard()).
    total_size = 0
    for fname in set(shard_map.values()):
        p = OUT / fname
        if p.exists():
            total_size += p.stat().st_size
    index = {
        "metadata": {
            "format": "jangtq",
            "total_size": total_size,
        },
        "weight_map": shard_map,
    }
    with open(OUT / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # === config.json — write FULL JANGTQ metadata block (avoid the silent
    # bits=2 vs affine-8 fallback bug; mirror DSV4-Flash convention).
    config.pop("quantization_config", None)
    mxtq_bits_map = {
        "routed_expert": EXPERT_BITS,
        "attention": 8,
        "shared_expert": 8,
        "dense_mlp": 8,
        "embed_tokens": 8,
        "lm_head": 8,
        "mtp_eh_proj": 8,
        "norms_router_biases": 16,
    }
    config["quantization"] = {
        "bits": 8,                    # default for affine paths (CRITICAL: not 2)
        "mode": "affine",
        "group_size": 64,
        "routed_expert_bits": EXPERT_BITS,
        "mxtq_bits": mxtq_bits_map,
    }
    config["mxtq_bits"] = mxtq_bits_map  # top-level for §418 backwards compat
    config["weight_format"] = "mxtq"

    with open(OUT / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # === jang_config.json — Tier-1 capabilities + JANGTQ profile metadata ===
    jang_config = {
        "version": 2,
        "weight_format": "mxtq",
        "profile": PROFILE,
        "source_model": {
            "name": "Ling-2.6-flash",
            "org": "inclusionAI",
            "architecture": "bailing_hybrid",
        },
        "mxtq_seed": SEED,
        "mxtq_bits": mxtq_bits_map,
        "quantization": {
            "method": "affine+mxtq",
            "group_size": 64,
            "bits_default": EXPERT_BITS,
        },
    }

    try:
        from jang_tools.capabilities import build_capabilities
        caps = build_capabilities(jang_config, config, OUT)
        if caps is not None:
            jang_config["capabilities"] = caps
            print(
                f"  capabilities: family={caps['family']} reasoning={caps['reasoning_parser']} "
                f"tool={caps['tool_parser']} cache={caps['cache_type']} modality={caps['modality']}",
                flush=True,
            )
        else:
            # Bailing-hybrid is new; capabilities map may need an entry.
            print(
                "  WARNING: could not resolve capabilities — vmlx will fall back to "
                "silver/bronze. Add 'bailing_hybrid' to jang_tools/capabilities.py::FAMILY_MAP.",
                flush=True,
            )
    except Exception as _e:
        print(f"  WARNING: build_capabilities failed: {_e}", flush=True)

    with open(OUT / "jang_config.json", "w") as f:
        json.dump(jang_config, f, indent=2)

    try:
        from jang_tools.capabilities import verify_directory
        _ok, _msg = verify_directory(OUT)
        print(f"  verify: {_msg}")
        if not _ok:
            print(
                f"  WARNING: capabilities verify failed: {_msg} "
                "— fix jang_tools/capabilities.py before HF upload.",
                flush=True,
            )
    except Exception:
        pass

    # === Copy tokenizer / chat-template / custom modeling files ===
    for f in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
        "configuration_bailing_moe_v2_5.py",
        "modeling_bailing_moe_v2_5.py",
        "configuration.json",
    ]:
        src_f = SRC / f
        if src_f.exists():
            shutil.copy2(str(src_f), str(OUT / f))

    # === Build Swift runtime sidecar (codebooks + signs) ===
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
        print(
            f"  [sidecar] FAILED: {_e} — run "
            f"`python3 -m jang_tools.build_jangtq_sidecar {OUT}` manually before upload",
            flush=True,
        )

    # Codex 2026-05-05 #2: prestack post-process for spec compliance.
    if not _args.no_prestack:
        print(f"\n  Prestacking (JANGTQ-PRESTACK-SPEC compliance)...")
        try:
            from jang_tools.rebundle_jangtq_stacked import rebundle
            import shutil as _shutil
            _tmp = OUT.parent / (OUT.name + ".prestack_tmp")
            if _tmp.exists():
                _shutil.rmtree(_tmp)
            rebundle(OUT, _tmp)
            _backup = OUT.parent / (OUT.name + ".per_expert_backup")
            if _backup.exists():
                _shutil.rmtree(_backup)
            OUT.rename(_backup)
            _tmp.rename(OUT)
            _shutil.rmtree(_backup)
            print(f"  Prestack complete.")
        except Exception as _re:
            print(f"  [prestack] FAILED: {_re} — bundle remains per-expert layout. "
                  f"Run `python -m jang_tools.rebundle_jangtq_stacked {OUT} {OUT}_prestacked` manually.",
                  flush=True)

    print(f"\n  Done!")
    print(f"  MXTQ tensors:        {total_mxtq}")
    print(f"  Affine tensors:      {total_affine}")
    print(f"  Passthrough tensors: {total_passthrough}")
    print(f"  Output:              {OUT}")
    progress.done(ok=True, output=str(OUT))

except Exception as _exc:
    progress.done(ok=False, error=f"{type(_exc).__name__}: {_exc}")
    raise

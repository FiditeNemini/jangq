"""
Ling-2.6-flash (Bailing-V2.5 hybrid) → MXFP4 Conversion
Created by Jinho Jang (eric@jangq.ai)

MXFP4 path: stock MLX 4-bit affine grouped quantization (group_size=32, bits=4),
no TurboQuant codec. Loads via `mlx_lm.load()` once `mlx_lm/models/bailing_hybrid.py`
is present (required for MLA + Linear-Attention + MTP dispatch).

Design:
  - Pre-stack routed experts at convert time so stock mlx_lm.load() works:
      input  : model.layers.N.mlp.experts.E.{gate_proj,up_proj,down_proj}.weight
      output : model.layers.N.mlp.switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}
  - Stacked along axis 0 → mx.quantize once per (layer, projection).
  - All non-routed weights also affine-4 quantized except norms / router gate /
    biases / slopes which stay fp16 / fp32 passthrough.

What's quantized 4-bit affine g=32:
  routed experts (stacked), shared experts, attention (MLA q_a/q_b/kv_a/kv_b/dense
  + Linear query_key_value/dense/g_proj), MTP eh_proj, dense MLP layer-0,
  embed, lm_head.

What's passthrough fp16/fp32:
  all *.norm / *.layernorm / model.norm, expert_bias (fp32),
  e_score_correction_bias, router gate.weight, MTP enorm/hnorm/final_layernorm.
"""
import sys
import json
import gc
import shutil
import argparse
import struct as _struct
import re
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

from jang_tools.calibrate import _load_bf16_tensor


if len(sys.argv) < 3:
    print(
        "usage: python -m jang_tools.convert_ling_mxfp4 <src_bf16_dir> <out_dir> [bits] [group_size]\n"
        "  bits        default 4\n"
        "  group_size  default 32",
        file=sys.stderr,
    )
    sys.exit(2)

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
BITS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
GROUP = int(sys.argv[4]) if len(sys.argv) > 4 else 32

# Routed-expert key pattern. group(1)=layer_idx, group(2)=expert_idx, group(3)=proj
EXPERT_KEY_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


try:
    OUT.mkdir(parents=True, exist_ok=True)

    with open(SRC / "config.json") as f:
        config = json.load(f)
    n_layers = config.get("num_hidden_layers", 32)
    n_experts = config.get("num_experts", 256)
    n_mtp = config.get("num_nextn_predict_layers", 0)
    hidden_size = config.get("hidden_size", 4096)
    moe_intermediate = config.get("moe_intermediate_size", 1024)
    first_k_dense_replace = config.get("first_k_dense_replace", 1)

    print("=" * 60)
    print(f"  Ling-2.6-flash (bailing_hybrid) → MXFP4 ({BITS}-bit affine g={GROUP})")
    print(f"  Created by Jinho Jang (eric@jangq.ai)")
    print("=" * 60)
    print(f"  Source: {SRC}")
    print(f"  Output: {OUT}")
    print(
        f"  Layers: {n_layers} (+{n_mtp} MTP)  Experts: {n_experts}  "
        f"first_dense={first_k_dense_replace}"
    )
    print(flush=True)

    # === Bit assignment ===
    def is_passthrough(name: str) -> bool:
        n = name
        if "norm" in n:
            return True
        if n.endswith(".bias"):
            return True
        if "expert_bias" in n or "e_score_correction_bias" in n:
            return True
        if n.endswith(".mlp.gate.weight"):
            return True
        return False

    # === Scan source — separate routed-expert keys from regular keys ===
    progress.phase(1, 3, "scan")
    print("\n  Scanning source...", flush=True)
    regular_keys = []         # (key, shape, src_path)
    expert_groups: dict[tuple[int, str], list[tuple[int, Path]]] = {}
    # expert_groups[(layer_idx, proj_name)] = [(expert_idx, src_path), ...]
    for sf in sorted(SRC.glob("model-*.safetensors")):
        with safe_open(str(sf), framework="numpy") as f:
            for k in f.keys():
                m = EXPERT_KEY_RE.match(k)
                if m:
                    layer_idx = int(m.group(1))
                    expert_idx = int(m.group(2))
                    proj = m.group(3)
                    expert_groups.setdefault((layer_idx, proj), []).append(
                        (expert_idx, sf)
                    )
                else:
                    shape = list(f.get_slice(k).get_shape())
                    regular_keys.append((k, shape, sf))

    print(
        f"  Found {len(regular_keys)} regular tensors + "
        f"{len(expert_groups)} expert groups",
        flush=True,
    )

    # === Sharded write ===
    shard_idx = 0
    shard_tensors = {}
    shard_bytes = 0
    MAX_SHARD = 1_000_000_000
    total_quantized = 0
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

    def affine_quantize_and_emit(base: str, tensor_np: np.ndarray):
        """tensor_np: float32, any rank. Quantize and emit base.{weight,scales,biases}."""
        # Skip if all output keys exist already (resume path).
        if (
            f"{base}.weight" in done_keys
            and f"{base}.scales" in done_keys
            and f"{base}.biases" in done_keys
        ):
            return False
        w = mx.array(tensor_np.astype(np.float16))
        qw, qs, qb = mx.quantize(w, group_size=GROUP, bits=BITS)
        add_tensor(f"{base}.weight", np.array(qw))
        add_tensor(f"{base}.scales", np.array(qs).astype(np.float16))
        add_tensor(f"{base}.biases", np.array(qb).astype(np.float16))
        return True

    def load_bf16(sf_path, key, shape):
        """Load a bf16 tensor as float32 numpy array."""
        try:
            with safe_open(str(sf_path), framework="numpy") as f:
                arr = f.get_tensor(key)
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr)
        except (TypeError, ValueError):
            arr = _load_bf16_tensor(sf_path, key, shape)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return arr

    # === Process regular keys ===
    progress.phase(2, 3, "convert")
    print("\n  Converting regular tensors...", flush=True)
    for tensor_name, shape, sf_path in tqdm(
        regular_keys, desc="  Regular  "
    ):
        if is_passthrough(tensor_name):
            if tensor_name in done_keys:
                total_passthrough += 1
                continue
            arr = load_bf16(sf_path, tensor_name, shape)
            if "expert_bias" in tensor_name:
                add_tensor(tensor_name, arr.astype(np.float32))
            else:
                add_tensor(tensor_name, arr.astype(np.float16))
            total_passthrough += 1
            del arr
            continue

        # Quantize
        base = (
            tensor_name.replace(".weight", "")
            if tensor_name.endswith(".weight")
            else tensor_name
        )
        # Special-case: rename `mlp.gate.weight` to `mlp.gate.gate_proj.weight`
        # at convert time so the bailing_hybrid sanitize doesn't have to.
        # (router gate.weight is passthrough, so this branch is for non-router gates only.)
        # The router gate.weight already passed through the is_passthrough check.
        arr = load_bf16(sf_path, tensor_name, shape)
        affine_quantize_and_emit(base, arr)
        total_quantized += 1
        del arr

    # === Process expert groups (pre-stack then quantize) ===
    print("\n  Stacking + quantizing routed experts...", flush=True)
    for (layer_idx, proj), members in tqdm(
        sorted(expert_groups.items()), desc="  Experts  "
    ):
        if len(members) != n_experts:
            print(
                f"  WARNING: layer {layer_idx} {proj} has {len(members)} experts "
                f"(expected {n_experts})"
            )

        out_base = f"model.layers.{layer_idx}.mlp.switch_mlp.{proj}"
        if (
            f"{out_base}.weight" in done_keys
            and f"{out_base}.scales" in done_keys
            and f"{out_base}.biases" in done_keys
        ):
            total_quantized += 1
            continue

        # Sort by expert_idx so stacking aligns
        members.sort(key=lambda t: t[0])
        # Load + stack
        stacked = None
        for expert_idx, sf_path in members:
            key = (
                f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}.weight"
            )
            arr = load_bf16(sf_path, key, None)
            if stacked is None:
                stacked = np.empty((n_experts,) + arr.shape, dtype=np.float32)
            stacked[expert_idx] = arr
            del arr
        # Quantize once on the [n_experts, out, in] stack
        affine_quantize_and_emit(out_base, stacked)
        total_quantized += 1
        del stacked
        gc.collect()

    # === Rename router gate.weight → gate.gate_proj.weight (sanitize-equivalent) ===
    # The bailing_hybrid model class expects `mlp.gate.gate_proj.weight`. The source
    # ships it as `mlp.gate.weight`. We've already passthrough-written it under the
    # source name; rename in shard_tensors / on-disk now to skip the runtime sanitize.
    # But because shards are already written, we instead emit a key alias: write
    # both names to the index. The runtime sanitize will pop the source name. To
    # avoid duplicating bytes we just rewrite the target name in the index.
    # (Simpler approach: rely on the model's sanitize. mlx_lm.load runs it.)

    flush_shard()

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
    total_size = 0
    for fname in set(shard_map.values()):
        p = OUT / fname
        if p.exists():
            total_size += p.stat().st_size
    index = {
        "metadata": {
            "format": "mxfp4",
            "total_size": total_size,
        },
        "weight_map": shard_map,
    }
    with open(OUT / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # === config.json — write stock-quantized metadata ===
    config.pop("quantization_config", None)
    config["quantization"] = {
        "bits": BITS,
        "group_size": GROUP,
        "mode": "affine",
    }
    config["weight_format"] = "mxfp4"
    with open(OUT / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # === jang_config.json — even though MXFP4 is stock, emit jang_config so the
    # capabilities pipeline + Tier-1 metadata still apply.
    jang_config = {
        "version": 2,
        "weight_format": "mxfp4",
        "profile": f"MXFP{BITS}",
        "source_model": {
            "name": "Ling-2.6-flash",
            "org": "inclusionAI",
            "architecture": "bailing_hybrid",
        },
        "quantization": {
            "method": "affine",
            "group_size": GROUP,
            "bits": BITS,
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
    except Exception as _e:
        print(f"  WARNING: build_capabilities failed: {_e}", flush=True)

    with open(OUT / "jang_config.json", "w") as f:
        json.dump(jang_config, f, indent=2)

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

    print(f"\n  Done!")
    print(f"  Quantized tensors:   {total_quantized}")
    print(f"  Passthrough tensors: {total_passthrough}")
    print(f"  Output:              {OUT}")
    progress.done(ok=True, output=str(OUT))

except Exception as _exc:
    progress.done(ok=False, error=f"{type(_exc).__name__}: {_exc}")
    raise

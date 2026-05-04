"""
Build the Nemotron-3-Nano-Omni multimodal addon bundle.

Extracts vision_model + sound_encoder + mlp1 + sound_projection weights from the
BF16 source as fp16, copies the source modeling/processing .py files, and writes
a small jang_omni_addon.json sidecar so the runtime can pair this addon with any
of our LLM bundles (MXFP4 / JANGTQ4 / JANGTQ2).

Output bundle is ~3 GB (vs 66 GB full BF16 source). User downloads ONE LLM bundle
+ ONE addon → multimodal works.

Usage:
  python3 -m jang_tools.build_nemotron_omni_addon <src_bf16_dir> <out_dir>
"""
import sys, json, gc, shutil
import numpy as np
from pathlib import Path
from safetensors import safe_open
from safetensors.numpy import save_file
from tqdm import tqdm


if len(sys.argv) < 3:
    print(
        "usage: python -m jang_tools.build_nemotron_omni_addon <src_bf16_dir> <out_dir>",
        file=sys.stderr,
    )
    sys.exit(2)
SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])

OUT.mkdir(parents=True, exist_ok=True)

print(f"  Source: {SRC}")
print(f"  Output: {OUT}")

# Multimodal-only key prefixes
MM_PREFIXES = (
    "vision_model.",
    "sound_encoder.",
    "mlp1.",
    "sound_projection.",
)

with open(SRC / "config.json") as f:
    full_config = json.load(f)

# Scan source
print("\n  Scanning source...")
mm_tensors = []  # (key, shape, sf)
for sf in sorted(SRC.glob("model-*.safetensors")):
    with safe_open(str(sf), framework="numpy") as f:
        for k in f.keys():
            if any(k.startswith(p) for p in MM_PREFIXES):
                shape = list(f.get_slice(k).get_shape())
                mm_tensors.append((k, shape, sf))
print(f"  Found {len(mm_tensors)} multimodal tensors")

# Bf16-as-fp16 (lossless cast) → keep multimodal weights at full precision
shard_idx = 0
shard_tensors: dict[str, np.ndarray] = {}
shard_bytes = 0
MAX_SHARD = 1_000_000_000
shard_map: dict[str, str] = {}


def flush_shard():
    global shard_idx, shard_tensors, shard_bytes
    if not shard_tensors:
        return
    shard_idx += 1
    fname = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
    save_file(shard_tensors, str(OUT / fname))
    for k in shard_tensors:
        shard_map[k] = fname
    print(f"    Shard {shard_idx}: {len(shard_tensors)} tensors, {shard_bytes/1e9:.2f} GB")
    shard_tensors = {}
    shard_bytes = 0


def add_tensor(name: str, arr: np.ndarray):
    global shard_bytes
    shard_tensors[name] = arr
    shard_bytes += arr.nbytes
    if shard_bytes >= MAX_SHARD:
        flush_shard()


print("\n  Copying multimodal tensors at fp16...")
for src_name, shape, sf_path in tqdm(mm_tensors):
    with safe_open(str(sf_path), framework="numpy") as f:
        try:
            t = f.get_tensor(src_name)
            if not isinstance(t, np.ndarray):
                t = np.array(t)
        except Exception:
            from jang_tools.calibrate import _load_bf16_tensor
            t = _load_bf16_tensor(sf_path, src_name, shape)
    if t.dtype not in (np.float16, np.float32, np.uint8):
        t = t.astype(np.float32)
    if t.dtype == np.float32:
        t = t.astype(np.float16)
    add_tensor(src_name, t)
flush_shard()

print(f"\n  Renaming {shard_idx} shards...")
for i in range(1, shard_idx + 1):
    old = OUT / f"model-{i:05d}-of-XXXXX.safetensors"
    new = OUT / f"model-{i:05d}-of-{shard_idx:05d}.safetensors"
    if old.exists():
        old.rename(new)
shard_map = {k: v.replace("XXXXX", f"{shard_idx:05d}") for k, v in shard_map.items()}

total_bytes = sum(sf.stat().st_size for sf in OUT.glob("model-*-of-*.safetensors"))
index = {"metadata": {"format": "fp16", "total_size": total_bytes},
         "weight_map": shard_map}
with open(OUT / "model.safetensors.index.json", "w") as f:
    json.dump(index, f, indent=2)

# Copy source preprocessor + modeling .py — needed for the runtime to invoke
# image / video / audio processing and call the wrapper if user wants pure-PyTorch.
print("\n  Copying source .py and tokenizer files...")
for f in [
    "configuration.py", "configuration_radio.py", "configuration_nemotron_h.py",
    "modeling.py", "modeling_nemotron_h.py", "audio_model.py",
    "image_processing.py", "video_processing.py", "video_io.py",
    "processing.py", "processing_utils.py", "evs.py",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "generation_config.json", "chat_template.jinja", "chat_template.json",
    "preprocessor_config.json", "video_preprocessor_config.json",
    "config.json",  # the OMNI wrapper config
]:
    src_f = SRC / f
    if src_f.exists():
        shutil.copy2(str(src_f), str(OUT / f))

# Sidecar: identify this as a multimodal addon
with open(OUT / "jang_omni_addon.json", "w") as f:
    json.dump({
        "version": 1,
        "addon_type": "nemotron_omni_multimodal",
        "source_arch": "NemotronH_Nano_Omni_Reasoning_V3",
        "components": ["vision_model", "sound_encoder", "mlp1", "sound_projection"],
        "compatible_llm_bundles": [
            "Nemotron-3-Nano-Omni-30B-A3B-MXFP4",
            "Nemotron-3-Nano-Omni-30B-A3B-JANGTQ4",
            "Nemotron-3-Nano-Omni-30B-A3B-JANGTQ2",
        ],
        "weight_dtype": "float16",
        "vision_hidden": full_config.get("vit_hidden_size", 1280),
        "projector_hidden": full_config.get("projector_hidden_size", 20480),
        "llm_hidden": full_config.get("llm_config", {}).get("hidden_size", 2688),
        "downsample_ratio": full_config.get("downsample_ratio", 0.5),
        "force_image_size": full_config.get("force_image_size", 512),
        "patch_size": full_config.get("patch_size", 16),
        "img_context_token_id": full_config.get("img_context_token_id"),
        "video_context_token_id": full_config.get("video_context_token_id"),
        "sound_context_token_id": full_config.get("sound_context_token_id"),
        "sound_config": full_config.get("sound_config"),
        "ps_version": full_config.get("ps_version", "v2"),
        "video_pruning_rate": full_config.get("video_pruning_rate", 0.7),
    }, f, indent=2)

print(f"\n  ✅ Done — {shard_idx} shards, {total_bytes/1e9:.2f} GB total")
print(f"  Output: {OUT}")

"""
Nemotron-3-Nano-Omni-30B-A3B (nemotron_h) → JANGTQ Conversion
Created by Jinho Jang (eric@jangq.ai)

Adapted from convert_minimax_jangtq.py for the nemotron_h hybrid Mamba+Attention+MoE
architecture. Source is BF16 (not FP8). Bundle ships TEXT-ONLY LLM weights —
vision_model (RADIO ViT) and sound_encoder (parakeet) are dropped because mlx_lm
has no nemotron_h_omni multimodal model. Future work can keep those tensors in a
separate sidecar bundle for an omni runtime.

Output bundle uses tensor names compatible with mlx_lm.models.nemotron_h:
  language_model.backbone.* → backbone.*
  language_model.lm_head    → lm_head

Per-expert tensors are kept unstacked; mlx_lm.nemotron_h.sanitize() stacks them
into switch_mlp.{fc1,fc2}.weight at load time:
  experts.E.up_proj.weight   → switch_mlp.fc1.weight  (stacked over E)
  experts.E.down_proj.weight → switch_mlp.fc2.weight  (stacked over E)

Layer types per hybrid_override_pattern (52 layers, MEMEM*EMEMEM*...):
  M = Mamba2 (mixer.{conv1d, in_proj, out_proj, A_log, D, dt_bias, norm})
  E = MoE   (mixer.{gate, experts.E.{up_proj,down_proj}, shared_experts.{up_proj,down_proj}})
  * = Attention (mixer.{q_proj, k_proj, v_proj, o_proj})

Quantization plan:
  routed experts (.experts.E.{up,down}_proj.weight)  → MXTQ EXPERT_BITS (2/3/4)
  attention q/k/v/o, shared_experts, embed, lm_head  → affine 8-bit
  Mamba in_proj, out_proj                            → affine 8-bit
  Mamba A_log, D, dt_bias, conv1d.{weight,bias}      → fp16 passthrough
  router gate.weight, e_score_correction_bias        → fp16 passthrough
  norms, all 1D                                       → fp16 passthrough
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

from jang_tools.calibrate import _load_bf16_tensor
from jang_tools.turboquant.linear import tq_quantize_weight

# === CLI ===
if len(sys.argv) < 3:
    print(
        "usage: python -m jang_tools.convert_nemotron_jangtq <src_bf16_dir> <out_dir> [profile]\n"
        "  <src_bf16_dir>  path to Nemotron-3-Nano-Omni source (BF16)\n"
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
        full_config = json.load(f)

    # Lift llm_config to top level — mlx_lm reads model_type from top level.
    llm_config = full_config.get("llm_config", {})
    if not llm_config:
        raise SystemExit(f"{SRC}/config.json has no llm_config — wrong source?")

    n_layers = llm_config.get("num_hidden_layers", 52)
    n_experts = llm_config.get("n_routed_experts", 128)
    pattern = llm_config.get("hybrid_override_pattern", "")
    print(f"  hybrid pattern ({len(pattern)} layers): {pattern}")
    print(f"  M={pattern.count('M')} E={pattern.count('E')} *={pattern.count('*')}")

    # === Bit-assignment policy =================================================
    # Names use SOURCE keys (with `language_model.` prefix). Destination renaming
    # is applied later in `dst_name()`.
    #
    # Method codes:
    #   "drop"        — skip entirely (multimodal towers, MTP)
    #   "passthrough" — store as fp16 (1-D, A_log, D, dt_bias, norm, gate, conv1d)
    #   "affine"      — mx.quantize bits=B group_size=64 (attn, shared, embed, head, mamba in/out_proj)
    #   "mxtq"        — TurboQuant codec at EXPERT_BITS (routed experts only)
    def get_method(tensor_name: str) -> tuple[int, str]:
        n = tensor_name

        # Drop multimodal towers + projections.
        if (n.startswith("vision_model.") or n.startswith("sound_encoder.")
                or n.startswith("mlp1.") or n.startswith("sound_projection.")):
            return (0, "drop")

        # Drop MTP (none on this model but keep guard for future).
        if n.startswith("language_model.mtp.") or n.startswith("mtp."):
            return (0, "drop")

        # Norms (RMSNorm.weight — 1-D, ~hidden_size).
        if n.endswith(".norm.weight") or "norm_f.weight" in n or n.endswith(".norm.bias"):
            return (16, "passthrough")
        if "mixer.norm.weight" in n:
            return (16, "passthrough")

        # Mamba 1-D parameters.
        if n.endswith(".A_log") or n.endswith(".D") or n.endswith(".dt_bias"):
            return (16, "passthrough")

        # Conv1d (small Conv1d weight + optional bias). Conv1d isn't a stock
        # nn.Linear so MLX won't quantize it; passthrough fp16.
        if "conv1d.weight" in n or "conv1d.bias" in n:
            return (16, "passthrough")

        # MoE router gate (n_routed_experts × hidden_size; precision-critical
        # routing math). Plus its e_score_correction_bias.
        if n.endswith(".mixer.gate.weight") or n.endswith(".mixer.gate.e_score_correction_bias"):
            return (16, "passthrough")

        # All remaining 1-D / bias terms.
        if n.endswith(".bias"):
            return (16, "passthrough")

        # Routed experts (per-expert MLP weights) — MXTQ at EXPERT_BITS.
        if ".mixer.experts." in n and (".up_proj.weight" in n or ".down_proj.weight" in n):
            return (EXPERT_BITS, "mxtq")

        # Shared expert MLP — affine 8-bit.
        if ".mixer.shared_experts." in n and (".up_proj.weight" in n or ".down_proj.weight" in n):
            return (8, "affine")

        # Mamba in_proj / out_proj — affine 8-bit.
        if (".mixer.in_proj.weight" in n) or (".mixer.out_proj.weight" in n):
            return (8, "affine")

        # Attention Q/K/V/O — affine 8-bit.
        if ".mixer." in n and any(p in n for p in (".q_proj.weight", ".k_proj.weight",
                                                    ".v_proj.weight", ".o_proj.weight")):
            return (8, "affine")

        # Embeddings + lm_head — affine 8-bit.
        if "embeddings.weight" in n or n.endswith("lm_head.weight"):
            return (8, "affine")

        # Default: 16-bit passthrough so we never silently miss a tensor.
        return (16, "passthrough")

    def dst_name(src: str) -> str:
        """Strip language_model. prefix; mlx_lm.nemotron_h expects backbone.* / lm_head."""
        if src.startswith("language_model."):
            return src[len("language_model."):]
        return src

    print("=" * 70)
    print(f"  Nemotron-3-Nano-Omni-30B-A3B → {PROFILE} JANGTQ Conversion")
    print(f"  Created by Jinho Jang (eric@jangq.ai)")
    print("=" * 70)
    print(f"  Source:  {SRC}")
    print(f"  Output:  {OUT}")
    print(f"  Layers:  {n_layers}  Experts: {n_experts}")
    print(f"  Profile: routed_expert=mxtq-{EXPERT_BITS}, attn/shared/embed/head=affine-8")
    print(f"  Mode:    text-only LLM (vision_model + sound_encoder dropped)")
    print(flush=True)

    # === Scan source tensors ==================================================
    progress.phase(1, 3, "scan")
    print("\n  Scanning source...", flush=True)
    all_tensors = []
    for sf in sorted(SRC.glob("model-*.safetensors")):
        with safe_open(str(sf), framework="numpy") as f:
            for k in f.keys():
                if k.endswith("_scale_inv"):
                    continue  # FP8 scales (none on this BF16 source, but defensive)
                shape = list(f.get_slice(k).get_shape())
                all_tensors.append((k, shape, sf))
    print(f"  Found {len(all_tensors)} source tensors", flush=True)

    # === Sharding output ======================================================
    shard_idx = 0
    shard_tensors: dict[str, np.ndarray] = {}
    shard_bytes = 0
    MAX_SHARD = 1_000_000_000  # 1 GB
    total_mxtq = 0
    total_affine = 0
    total_passthrough = 0
    total_dropped = 0
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
        print(f"    Shard {shard_idx}: {len(shard_tensors)} tensors, {shard_bytes/1e9:.2f} GB",
              flush=True)
        shard_tensors = {}
        shard_bytes = 0

    def add_tensor(name: str, arr: np.ndarray):
        global shard_bytes
        shard_tensors[name] = arr
        shard_bytes += arr.nbytes
        if shard_bytes >= MAX_SHARD:
            flush_shard()

    # === Resume support =======================================================
    done_keys: set[str] = set()
    existing_shards = sorted(OUT.glob("model-*-of-XXXXX.safetensors"))
    if existing_shards:
        print(f"\n  Resume: found {len(existing_shards)} existing shards", flush=True)
        import struct as _struct
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
        print(f"  Resume: {len(done_keys)} keys already written, continuing from shard {shard_idx + 1}",
              flush=True)

    def is_already_done(dst_n: str, method: str) -> bool:
        if method == "passthrough":
            return dst_n in done_keys
        elif method == "affine":
            base = dst_n[:-len(".weight")] if dst_n.endswith(".weight") else dst_n
            return (f"{base}.weight" in done_keys
                    and f"{base}.scales" in done_keys
                    and f"{base}.biases" in done_keys)
        elif method == "mxtq":
            base = dst_n[:-len(".weight")] if dst_n.endswith(".weight") else dst_n
            return (f"{base}.tq_packed" in done_keys
                    and f"{base}.tq_norms" in done_keys
                    and f"{base}.tq_bits" in done_keys)
        return False

    # === Convert ==============================================================
    progress.phase(2, 3, "convert")
    print("\n  Converting...", flush=True)
    skipped_resume = 0
    for src_name, shape, sf_path in tqdm(all_tensors, desc="  Processing"):
        bits, method = get_method(src_name)

        if method == "drop":
            total_dropped += 1
            continue

        dst_n = dst_name(src_name)

        if done_keys and is_already_done(dst_n, method):
            skipped_resume += 1
            if method == "mxtq":      total_mxtq += 1
            elif method == "affine":  total_affine += 1
            else:                     total_passthrough += 1
            continue

        # Load tensor (BF16 source).
        with safe_open(str(sf_path), framework="numpy") as f:
            try:
                tensor = f.get_tensor(src_name)
                if not isinstance(tensor, np.ndarray):
                    tensor = np.array(tensor)
            except Exception:
                tensor = _load_bf16_tensor(sf_path, src_name, shape)

        if tensor.dtype != np.float32:
            tensor = tensor.astype(np.float32)

        if method == "passthrough":
            add_tensor(dst_n, tensor.astype(np.float16))
            total_passthrough += 1

        elif method == "affine":
            # 1-D guard: never feed mx.quantize a 1-D tensor.
            if tensor.ndim < 2:
                add_tensor(dst_n, tensor.astype(np.float16))
                total_passthrough += 1
            else:
                w = mx.array(tensor.astype(np.float16))
                qw, qs, qb = mx.quantize(w, group_size=64, bits=bits)
                base = dst_n[:-len(".weight")] if dst_n.endswith(".weight") else dst_n
                add_tensor(f"{base}.weight", np.array(qw))
                add_tensor(f"{base}.scales", np.array(qs).astype(np.float16))
                add_tensor(f"{base}.biases", np.array(qb).astype(np.float16))
                total_affine += 1
                del w, qw, qs, qb

        elif method == "mxtq":
            # Routed expert weight: 2-D (intermediate × hidden) for up_proj
            # or (hidden × intermediate) for down_proj. tq_quantize_weight
            # handles both via per-row codebook + Hadamard rotation.
            result = tq_quantize_weight(tensor, bits=bits, seed=SEED)
            base = dst_n[:-len(".weight")] if dst_n.endswith(".weight") else dst_n
            add_tensor(f"{base}.tq_packed", result["packed"])
            add_tensor(f"{base}.tq_norms",  result["norms"])
            add_tensor(f"{base}.tq_bits",   np.array([bits], dtype=np.uint8))
            total_mxtq += 1

        del tensor
        if (total_mxtq + total_affine) % 200 == 0:
            gc.collect()

    flush_shard()

    if skipped_resume > 0:
        print(f"\n  Resume: skipped {skipped_resume} tensors from previous run", flush=True)

    # === Rename shards to final NNNNN-of-NNNNN form ===========================
    progress.phase(3, 3, "write")
    print(f"\n  Renaming {shard_idx} shards...", flush=True)
    for i in range(1, shard_idx + 1):
        old = OUT / f"model-{i:05d}-of-XXXXX.safetensors"
        new = OUT / f"model-{i:05d}-of-{shard_idx:05d}.safetensors"
        if old.exists():
            old.rename(new)
    shard_map = {k: v.replace("XXXXX", f"{shard_idx:05d}") for k, v in shard_map.items()}

    # Total size for index metadata
    total_bytes = 0
    for sf in sorted(OUT.glob("model-*-of-*.safetensors")):
        total_bytes += sf.stat().st_size
    index = {"metadata": {"format": "jangtq", "total_size": total_bytes},
             "weight_map": shard_map}
    with open(OUT / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # === Build flattened nemotron_h config.json ==============================
    # mlx_lm reads top-level model_type / hidden_size / etc. We start from
    # llm_config (nemotron_h sub-config) and add quant + capability metadata.
    out_config = dict(llm_config)
    out_config["model_type"] = "nemotron_h"
    # Pre-stacking is what mlx_lm.nemotron_h.sanitize() does at load — keep
    # the per-expert layout on disk so a future omni runtime can re-read.
    out_config["weight_format"] = "mxtq"
    out_config["mxtq_bits"] = EXPERT_BITS
    out_config["quantization"] = {"group_size": 64, "bits": EXPERT_BITS}
    # Stamp source provenance
    out_config["_jang_source"] = full_config.get("model_type", "NemotronH_Nano_Omni_Reasoning_V3")
    out_config["_jang_modality"] = "text"  # vision/audio dropped in this bundle

    with open(OUT / "config.json", "w") as f:
        json.dump(out_config, f, indent=2)

    # === jang_config.json =====================================================
    jang_config = {
        "version": 2,
        "weight_format": "mxtq",
        "profile": PROFILE,
        "source_model": {
            "name": "Nemotron-3-Nano-Omni-30B-A3B-Reasoning",
            "org": "nvidia",
            "architecture": "nemotron_h",
            "wrapper_arch": "NemotronH_Nano_Omni_Reasoning_V3",
            "modality": "text",  # text-only; vision/sound dropped
        },
        "mxtq_seed": SEED,
        "mxtq_bits": {
            "attention": 8,
            "shared_expert": 8,
            "mamba_proj": 8,
            "routed_expert": EXPERT_BITS,
            "embed_tokens": 8,
            "lm_head": 8,
        },
        "quantization": {
            "method": "affine+mxtq",
            "group_size": 64,
            "bits_default": EXPERT_BITS,
        },
        "hybrid_pattern": pattern,
    }
    # Stamp Tier-1 capabilities for vmlx CapabilityDetector. Idempotent.
    from jang_tools.capabilities import build_capabilities
    caps = build_capabilities(jang_config, out_config, OUT)
    if caps is not None:
        jang_config["capabilities"] = caps
        print(f"  capabilities: family={caps['family']} reasoning={caps['reasoning_parser']} "
              f"tool={caps['tool_parser']} cache={caps['cache_type']} modality={caps['modality']}",
              flush=True)
    else:
        print("  WARNING: could not resolve capabilities — vmlx will fall back to "
              "silver/bronze. Add nemotron_h to jang_tools/capabilities.py::FAMILY_MAP.",
              flush=True)

    with open(OUT / "jang_config.json", "w") as f:
        json.dump(jang_config, f, indent=2)

    # Validate the final jang_config — catch schema drift / typos / missing keys.
    from jang_tools.capabilities import verify_directory
    _ok, _msg = verify_directory(OUT)
    print(f"  verify: {_msg}")
    if not _ok:
        raise SystemExit(f"capabilities verify failed: {_msg}")

    # === Copy tokenizer + chat template + custom .py files ====================
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "generation_config.json", "chat_template.jinja", "chat_template.json",
              "merges.txt", "vocab.json",
              "configuration_nemotron_h.py", "modeling_nemotron_h.py"]:
        src_f = SRC / f
        if src_f.exists():
            shutil.copy2(str(src_f), str(OUT / f))

    # Osaurus / swift-transformers compatibility — Nemotron uses a tiktoken-style
    # tokenizer; if it ships as TokenizersBackend, swift-transformers can't parse.
    _tok_cfg = OUT / "tokenizer_config.json"
    if _tok_cfg.exists():
        try:
            with open(_tok_cfg) as f:
                _tc = json.load(f)
            if _tc.get("tokenizer_class") == "TokenizersBackend":
                _tc["tokenizer_class"] = "GPT2Tokenizer"
                with open(_tok_cfg, "w") as f:
                    json.dump(_tc, f, indent=2)
                print("  [osaurus-fix] tokenizer_class: TokenizersBackend → GPT2Tokenizer",
                      flush=True)
        except Exception as _e:
            print(f"  [osaurus-fix] skipped: {_e}", flush=True)

    # === Build Swift runtime sidecar ==========================================
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
        print(f"  [sidecar] FAILED: {_e} — run "
              f"`python3 -m jang_tools.build_jangtq_sidecar {OUT}` manually before upload",
              flush=True)

    print(f"\n  Done!")
    print(f"  MXTQ tensors:        {total_mxtq}")
    print(f"  Affine tensors:      {total_affine}")
    print(f"  Passthrough tensors: {total_passthrough}")
    print(f"  Dropped (vision/audio): {total_dropped}")
    print(f"  Output: {OUT}")
    progress.done(ok=True, output=str(OUT))

except Exception as _exc:
    progress.done(ok=False, error=f"{type(_exc).__name__}: {_exc}")
    raise

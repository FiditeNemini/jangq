"""
JANG Convert — End-to-end model quantization pipeline.
Created by Jinho Jang (eric@jangq.ai)

Takes a HuggingFace model directory and produces a JANG v2 model.
v2 outputs MLX-native safetensors — loads instantly via mx.load() mmap.

Pipeline: detect arch → calibrate → allocate bits → quantize → write
"""

import json
import re
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm
from safetensors import safe_open

from .format.spec import DEFAULT_BLOCK_SIZE
from .format.writer import write_jang_v2_model
from .architectures import detect_architecture, get_layer_config, get_skip_tensors, summarize_architecture
from .calibrate import calibrate_from_weights, _load_bf16_tensor
from .allocate import allocate_bits_greedy, allocate_bits_profile, summarize_allocation, JANG_PROFILES


# Pattern for per-expert 2D tensors (MiniMax/Mixtral style)
# e.g. model.layers.0.block_sparse_moe.experts.5.w1.weight
_PER_EXPERT_PATTERN = re.compile(
    r"(.+)\.experts\.(\d+)\.(w[123]|gate_proj|up_proj|down_proj)\.weight$"
)

# MiniMax/Mixtral name mapping: w1→gate_proj, w2→down_proj, w3→up_proj
_EXPERT_NAME_MAP = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def _load_bf16_from_header(sf_path: str, key: str):
    """Load a bfloat16 tensor from safetensors by reading raw bytes.

    numpy can't handle bfloat16 — this reads the raw bytes and converts
    to float32 (bfloat16 is the upper 16 bits of float32).
    Returns None if key not found. Works for scalars and any shape.
    """
    import struct as _struct
    with open(sf_path, "rb") as fh:
        header_size = _struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_size))
        if key not in header:
            return None
        info = header[key]
        data_start = 8 + header_size
        fh.seek(data_start + info["data_offsets"][0])
        raw = fh.read(info["data_offsets"][1] - info["data_offsets"][0])
    shape = info.get("shape", [])
    n_elements = max(1, len(raw) // 2)  # 2 bytes per bfloat16
    # Convert bfloat16 → float32: each bf16 is upper 16 bits of f32
    bf16_arr = np.frombuffer(raw, dtype=np.uint16)
    f32_bits = bf16_arr.astype(np.uint32) << 16
    result = f32_bits.view(np.float32)
    if shape:
        result = result.reshape(shape)
    return result


def _get_tensor_group_size(tensor_name: str, default_gs: int, num_experts: int = 0) -> int:
    """
    Get the optimal group_size for a specific tensor.

    Rules (from CRACK abliteration research on MiniMax, Mar 2026):
    - MoE router/gate: ALWAYS gs=64 (precision-critical, tiny tensor)
    - Expert MLP with 150+ experts at 2-4 bit: gs=128 (gather_qmm cache pressure)
    - Everything else: gs=64 (standard, best precision)

    The default_gs is the model's global group_size (set by auto-detection).
    This function overrides it for specific tensors that need different values.
    """
    name_lower = tensor_name.lower()

    # MoE router/gate — ALWAYS gs=64 for precision (tiny tensor, speed irrelevant)
    # Matches: mlp.gate.weight, block_sparse_moe.gate.weight, shared_expert_gate.weight
    if ".gate." in name_lower or name_lower.endswith(".gate"):
        return 64
    if "shared_expert_gate" in name_lower:
        return 64

    # Everything else uses the model's global group_size
    # (which is 128 for 150+ expert models, 64 otherwise)
    return default_gs


def convert_model(
    model_path: str | Path,
    output_path: str | Path,
    target_bits: float = 2.5,
    block_size: int = DEFAULT_BLOCK_SIZE,
    calibration_method: str = "weights",
    quantization_method: str = "mse",
    imatrix_path: Optional[str | Path] = None,
    use_awq: bool = False,
    awq_alpha: float = 0.25,
    profile: Optional[str] = None,
    hadamard: bool = False,
    gptq_hessian_dir: Optional[str | Path] = None,
) -> dict:
    """
    Convert a HuggingFace model to JANG v2 format.

    v2 stores weights as MLX-native uint32 packed + float16 scales/biases
    in standard safetensors. Loading is instant via mx.load() mmap — no
    repacking step at load time.

    Args:
        model_path: path to source model directory
        output_path: path for output JANG model directory
        target_bits: target average bits per weight
        block_size: weights per quantization block
        calibration_method: "weights" (fast) or "activations" (better quality)
        quantization_method: "rtn" (fast) or "mse" (better quality)
        imatrix_path: pre-computed importance matrix (skip calibration if provided)

    Returns:
        dict with conversion results and quality metrics
    """
    model_path = Path(model_path)
    output_path = Path(output_path)

    print(f"\n{'='*60}")
    print(f"  JANG Convert v2")
    print(f"  Created by Jinho Jang (eric@jangq.ai)")
    print(f"{'='*60}")
    print(f"  Source: {model_path}")
    print(f"  Output: {output_path}")
    print(f"  Target: {target_bits} bits/weight")
    print(f"  Block size: {block_size}")
    print(f"  Calibration: {calibration_method}")
    print(f"  Quantization: {quantization_method}")
    print(f"  Format: v2 (MLX-native, instant load)")
    print(f"  AWQ scaling: {'yes (alpha=' + str(awq_alpha) + ')' if use_awq else 'no'}")
    print(f"  Hadamard rotation: {'yes' if hadamard else 'no'}")
    print(f"{'='*60}\n")

    # Check source model exists
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(
            f"No config.json in {model_path} — is this a HuggingFace model directory?"
        )

    # Check tie_word_embeddings early
    _raw_config = json.loads((model_path / "config.json").read_text())
    _tie_embeddings = _raw_config.get("tie_word_embeddings", False) or _raw_config.get("text_config", {}).get("tie_word_embeddings", False)

    # Step 1: Detect architecture
    print("  [1/5] Detecting architecture...")
    arch_config = detect_architecture(model_path)
    print(f"  {summarize_architecture(arch_config)}\n")

    # Auto-detect optimal group_size for MoE models with many experts.
    # Models with 150+ experts suffer 15-25% speed regression at group_size=64
    # due to gather_qmm kernel cache pressure. group_size=128 eliminates this.
    # Discovered via CRACK abliteration research (Mar 5 2026).
    num_experts = getattr(arch_config, 'num_experts', 0)
    has_shared_mlp = getattr(arch_config, 'has_shared_mlp', False)
    if has_shared_mlp:
        print(f"  Shared MLP: parallel dense MLP alongside MoE → classified as CRITICAL")
    if block_size == DEFAULT_BLOCK_SIZE and arch_config.has_moe_layers:
        if num_experts >= 150:
            # Check MLA dim compatibility: sanitize reshapes kv_b_proj to (..., qk_nope_head_dim).
            # If qk_nope_head_dim isn't divisible by 128, must use 64 or loading crashes.
            _qk_nope = _raw_config.get("qk_nope_head_dim", _raw_config.get("text_config", {}).get("qk_nope_head_dim", 0))
            if _qk_nope > 0 and _qk_nope % 128 != 0:
                block_size = 64
                print(f"  Auto group_size: {num_experts} experts + qk_nope_head_dim={_qk_nope} (not ÷128) → group_size=64")
            else:
                block_size = 128
                print(f"  Auto group_size: {num_experts} experts detected → group_size=128 (speed fix)")
        elif num_experts >= 64:
            # 64-149 experts: warn but keep 64 (quality vs speed tradeoff)
            print(f"  Note: {num_experts} experts. Consider -b 128 if speed is priority.")

    # MLP asymmetry fix for 512+ expert models.
    # gate_proj→IMPORTANT (4-bit), down_proj→3-bit floor, up_proj→COMPRESS (2-bit OK).
    # Prevents SiLU amplification → float16 overflow → NaN (proven on 397B + Nemotron).
    # See research/397B-MLP-ASYMMETRY.md for full analysis.
    if num_experts >= 512:
        print(f"  MLP asymmetry: 512+ experts → gate_proj=IMPORTANT(4-bit), down_proj=3-bit floor")

    # Step 2: Calibrate (compute importance scores)
    # Skip calibration for profile and K-quant allocation — they use tier
    # classification only, never importance scores. Saves 10-30 seconds.
    from .allocate import is_k_quant
    needs_calibration = not (profile and (profile.upper() in JANG_PROFILES or is_k_quant(profile.upper())))
    # Also skip for integer target bits (routed to K-quant in step 3)
    if not needs_calibration and target_bits in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        needs_calibration = False

    if needs_calibration:
        print("  [2/5] Calibrating...")
        if imatrix_path:
            from safetensors.numpy import load_file
            importance_data = load_file(str(imatrix_path))
            print(f"  Loaded pre-computed imatrix: {imatrix_path}")
        elif calibration_method == "weights":
            # Save imatrix to output dir (not source dir) to avoid polluting source
            if output_path:
                output_path.mkdir(parents=True, exist_ok=True)
            imatrix_out = output_path / "jang_imatrix.safetensors" if output_path else None
            importance_data = calibrate_from_weights(model_path, block_size, output_path=imatrix_out)
        else:
            from .calibrate import calibrate_with_activations
            imatrix_out = output_path / "jang_imatrix.safetensors" if output_path else None
            importance_data = calibrate_with_activations(model_path, block_size=block_size, output_path=imatrix_out)
    else:
        print("  [2/5] Skipping calibration (not needed for profile/K-quant allocation)")
        importance_data = {}

    # Step 2b: AWQ activation norms (optional)
    awq_norms = {}
    if use_awq:
        print("\n  [2b] Collecting AWQ activation norms...")
        try:
            from .awq import collect_activation_norms_mlx
            awq_norms = collect_activation_norms_mlx(str(model_path))
        except ImportError:
            print("  WARNING: MLX not available, skipping AWQ scaling")
            use_awq = False

    # Step 3: Load weights and allocate bits
    print("\n  [3/5] Allocating bits...")
    skip_patterns = get_skip_tensors(arch_config)
    weight_files = sorted(model_path.glob("*.safetensors"))

    # Collect all weight tensor info
    all_tensors_info = []  # (name, shape, n_blocks, sf_path)

    # Compact allocation: per-tensor classification saves 100+ GB on large models.
    # Profile/budget allocation assigns bits per-tensor, not per-block.
    # GLM-5 (256 experts × 78 layers) has 5.89B blocks → 100 GB in per-block arrays.
    use_compact = (
        (profile and (profile.upper() in JANG_PROFILES or is_k_quant(profile.upper())))
        or target_bits in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
    )
    if not use_compact:
        all_importance = []
        all_tensor_names_for_alloc = []

    for sf_path in weight_files:
        with safe_open(str(sf_path), framework="numpy") as f:
            for tensor_name in f.keys():
                if any(skip in tensor_name for skip in skip_patterns):
                    continue
                if tensor_name.endswith(".bias"):
                    continue
                if "layernorm" in tensor_name.lower() or "rmsnorm" in tensor_name.lower():
                    continue
                if "weight_scale_inv" in tensor_name or "_scale_inv" in tensor_name:
                    continue
                if "activation_scale" in tensor_name:
                    continue
                # Gemma 4: tiny scalar/norm tensors that must stay full precision
                if tensor_name.endswith(".layer_scalar"):
                    continue
                if "q_norm" in tensor_name or "k_norm" in tensor_name:
                    continue
                if "lm_head" in tensor_name and _tie_embeddings:
                    continue
                if ".visual." in tensor_name or "vision_tower" in tensor_name or "vision_model" in tensor_name or "vision_encoder" in tensor_name or "embed_vision" in tensor_name:
                    continue

                shape = f.get_slice(tensor_name).get_shape()
                if len(shape) < 2:
                    continue

                n_weights = 1
                for d in shape:
                    n_weights *= d
                if len(shape) > 2 and n_weights < 100_000:
                    continue

                n_blocks = (n_weights + block_size - 1) // block_size

                all_tensors_info.append((tensor_name, shape, n_blocks, sf_path))
                if not use_compact:
                    base_name = tensor_name
                    if base_name.endswith(".weight"):
                        base_name = base_name[:-7]
                    imp_key = f"{base_name}.importance"
                    if imp_key in importance_data:
                        imp = importance_data[imp_key]
                    else:
                        imp = np.ones(n_blocks, dtype=np.float32) * 0.5
                    all_importance.append(imp)
                    all_tensor_names_for_alloc.extend([tensor_name] * n_blocks)

    # Run bit allocation → produces _tensor_bits dict (tensor_name → bits)
    if use_compact:
        # Compact path: classify each tensor once, no per-block arrays.
        # Reduces allocation memory from ~100 GB to <100 MB on 744B+ models.
        from .allocate import (
            allocate_bits_profile_compact, allocate_bits_budget_compact,
            summarize_allocation_compact, k_quant_target,
        )
        tensor_info = [(name, n_blocks) for name, _, n_blocks, _ in all_tensors_info]

        if profile and is_k_quant(profile.upper()):
            k_target = k_quant_target(profile.upper())
            print(f"  Using K-quant: {profile} (target: {k_target} avg bits)")
            _tensor_bits = allocate_bits_budget_compact(
                tensor_info, target_bits=k_target,
                num_experts=num_experts, has_shared_mlp=has_shared_mlp,
            )
        elif profile:
            print(f"  Using profile: {profile}")
            _tensor_bits = allocate_bits_profile_compact(
                tensor_info, profile,
                num_experts=num_experts, has_shared_mlp=has_shared_mlp,
            )
        else:
            print(f"  Using K-quant allocation (target: {target_bits} avg bits)")
            _tensor_bits = allocate_bits_budget_compact(
                tensor_info, target_bits=target_bits,
                num_experts=num_experts, has_shared_mlp=has_shared_mlp,
            )

        alloc_summary = summarize_allocation_compact(_tensor_bits, tensor_info, num_experts)
        actual_bits = alloc_summary["average_bits"]
        print(f"  Compact allocation: {len(all_tensors_info)} tensors classified directly")

    else:
        # Original per-block path (greedy/DP allocation)
        global_importance = np.concatenate(all_importance)

        layer_indices = set()
        for name in all_tensor_names_for_alloc:
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_indices.add(int(parts[i + 1]))
                    except ValueError:
                        pass
        n_layers = max(layer_indices) + 1 if layer_indices else 1

        from .allocate import k_quant_target, allocate_bits_budget
        if profile and is_k_quant(profile):
            k_target = k_quant_target(profile)
            print(f"  Using K-quant: {profile} (target: {k_target} avg bits)")
            global_bit_alloc = allocate_bits_budget(
                all_tensor_names_for_alloc, target_bits=k_target,
                num_experts=num_experts, has_shared_mlp=has_shared_mlp,
            )
        elif profile:
            print(f"  Using profile: {profile}")
            global_bit_alloc = allocate_bits_profile(
                all_tensor_names_for_alloc, profile,
                num_experts=num_experts, has_shared_mlp=has_shared_mlp,
            )
        elif target_bits in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
            print(f"  Using K-quant allocation (target: {target_bits} avg bits)")
            global_bit_alloc = allocate_bits_budget(
                all_tensor_names_for_alloc, target_bits=target_bits,
                num_experts=num_experts, has_shared_mlp=has_shared_mlp,
            )
        else:
            global_bit_alloc = allocate_bits_greedy(
                global_importance, target_bits, all_tensor_names_for_alloc,
                n_layers=n_layers, block_size=block_size,
            )

        alloc_summary = summarize_allocation(global_bit_alloc, all_tensor_names_for_alloc, num_experts)
        actual_bits = alloc_summary["average_bits"]

        # Build _tensor_bits from per-block allocation
        _tensor_bits = {}
        _offset = 0
        for tensor_name, shape, n_blocks, sf_path in all_tensors_info:
            _tensor_bits[tensor_name] = int(global_bit_alloc[_offset])
            _offset += n_blocks

        # Free the giant arrays — recovers ~100 GB on 744B models
        del all_tensor_names_for_alloc, global_bit_alloc, all_importance, global_importance
        import gc; gc.collect()
        print(f"  Freed allocation arrays ({len(_tensor_bits)} tensor bit assignments retained)")

    # Print summary (common for both paths)
    print(f"  Target bits: {target_bits}")
    print(f"  Actual bits: {actual_bits:.2f}")
    print(f"  Total blocks: {alloc_summary['total_blocks']:,}")
    for bw, info in alloc_summary["histogram"].items():
        print(f"    {bw}: {info['count']:,} blocks ({info['percent']}%)")

    # ── Safety check: precision floor warnings ──────────────────────
    from .allocate import classify_tensor, Tier
    _cfg = json.loads((model_path / "config.json").read_text())
    hidden_size = _cfg.get("hidden_size",
                    _cfg.get("text_config", {}).get("hidden_size", 0))
    danger_tensors = {}
    for tname, bits in _tensor_bits.items():
        tier = classify_tensor(tname, num_experts)
        if "shared_expert" in tname and "gate" not in tname and bits < 4 and hidden_size >= 4096:
            danger_tensors.setdefault(tname, bits)
        if tier == Tier.CRITICAL and bits < 4:
            danger_tensors.setdefault(tname, bits)
        if num_experts >= 512 and bits < 4 and "shared_expert" not in tname:
            tname_lower = tname.lower()
            if "gate_proj" in tname_lower or "gate_up_proj" in tname_lower:
                danger_tensors.setdefault(tname, bits)
        if num_experts >= 512 and bits < 3 and "shared_expert" not in tname:
            tname_lower = tname.lower()
            if "down_proj" in tname_lower:
                danger_tensors.setdefault(tname, bits)

    if danger_tensors:
        print(f"\n  ⚠ PRECISION WARNING: {len(set(danger_tensors.values()))} tensor types below safe floor")
        shown = set()
        for tname, bits in danger_tensors.items():
            base = tname.split(".")[-1] if "." in tname else tname
            if base not in shown:
                print(f"    {tname}: {bits}-bit (min recommended: 4-bit)")
                shown.add(base)
                if len(shown) >= 5:
                    break
        if num_experts >= 512:
            print(f"    512+ experts with hidden_size={hidden_size}: MLP asymmetry risk.")
            print(f"    gate_proj needs 4-bit min (SiLU amplifier), down_proj needs 3-bit min.")
        print(f"    This may cause float16 overflow (NaN) on large models.")
        print(f"    Consider using a higher-bit profile or reclassifying these tensors.\n")

    # Step 4: Quantize and build MLX-native tensors
    print(f"\n  [4/5] Quantizing to MLX-native format ({quantization_method})...")

    # v2 output: flat dict of tensor_name → numpy array (MLX format)
    v2_tensors = {}
    # Buffer for per-expert 2D tensors that need stacking
    expert_buffer = {}  # (layer_prefix, wtype) → {expert_id: {weight, scales, biases}}
    passthrough = {}

    for tensor_name, shape, n_blocks, sf_path in tqdm(all_tensors_info, desc="  Quantizing"):
        # Skip vision conv weights — Conv3d/Conv2d needs float, not uint32.
        # These are passthrough tensors that get saved as float16.
        name_lower = tensor_name.lower()
        if ("patch_embed" in name_lower or "temporal_embed" in name_lower or "patch_conv" in name_lower) and ".weight" in name_lower:
            with safe_open(str(sf_path), framework="numpy") as f:
                try:
                    w = f.get_tensor(tensor_name)
                except TypeError:
                    w = _load_bf16_tensor(sf_path, tensor_name, shape)
                w_out = w.astype(np.float16) if w.dtype != np.float16 else w
                # Transpose 4D conv weights: PyTorch OIHW → MLX OHWI
                if len(w_out.shape) == 4:
                    w_out = np.transpose(w_out, (0, 2, 3, 1))
                passthrough[tensor_name] = w_out
            continue

        with safe_open(str(sf_path), framework="numpy") as f:
            try:
                weights = f.get_tensor(tensor_name).astype(np.float32)
            except (TypeError, AttributeError):
                from .fp8 import load_fp8_tensor
                from .calibrate import _load_bf16_tensor
                scale_key = f"{tensor_name}_scale_inv"
                scale_inv = None
                try:
                    scale_inv = f.get_tensor(scale_key)
                except Exception:
                    # bfloat16 scale_inv can't be loaded by numpy —
                    # read raw bytes from safetensors and convert manually
                    scale_inv = _load_bf16_from_header(str(sf_path), scale_key)
                try:
                    weights = load_fp8_tensor(sf_path, tensor_name, shape, scale_inv)
                except Exception:
                    weights = _load_bf16_tensor(sf_path, tensor_name, shape)

        # AWQ scaling
        awq_scales = None
        if use_awq:
            parts = tensor_name.split(".")
            layer_key = None
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        layer_key = f"layers.{layer_idx}.attn_input"
                    except ValueError:
                        pass
                    break
            if layer_key and layer_key in awq_norms:
                from .awq import compute_awq_scales, apply_awq_scaling
                in_features = shape[1] if len(shape) == 2 else shape[-1]
                norms = awq_norms[layer_key]
                if len(norms) == in_features:
                    awq_scales = compute_awq_scales(norms, alpha=awq_alpha)
                    weights = apply_awq_scaling(weights, awq_scales)

        # MoE router/gate: store as float16 passthrough (no quantization).
        # Gate routing is extremely precision-sensitive — even 8-bit quantization
        # with different intermediate dtypes (f32 vs bf16) changes expert selection.
        # Gate is tiny (128 × 4096 = 0.5 MB per layer) so size impact is negligible.
        # This ensures maximum routing precision and avoids bf16/f16 dtype issues.
        _tn_lower = tensor_name.lower()
        _is_gate = (".gate.weight" in _tn_lower or _tn_lower.endswith(".gate.weight"))
        _is_gate = _is_gate and "gate_proj" not in _tn_lower and "gate_up" not in _tn_lower
        if _is_gate and num_experts > 0:
            passthrough[tensor_name] = weights.astype(np.float16)
            # offset already incremented at line 412 — do NOT increment again
            print(f"    Gate passthrough (f16): {tensor_name} → {weights.shape}")
            continue

        bits = _tensor_bits[tensor_name]
        w_shape = weights.shape
        is_3d = len(w_shape) >= 3

        # --- Check for GPTQ Hessian for this tensor ---
        _gptq_hessian = None
        _gptq_hinv = None
        if gptq_hessian_dir is not None:
            import re
            _layer_match = re.search(r'layers\.(\d+)', tensor_name)
            _is_expert = "experts" in tensor_name or "switch_mlp" in tensor_name
            if _layer_match and bits <= 4 and (_is_expert or is_3d):
                _layer_idx = int(_layer_match.group(1))
                _h_path = Path(gptq_hessian_dir) / f"H_FP8_L{_layer_idx}.npy"
                if _h_path.exists():
                    _gptq_hessian = np.load(str(_h_path)).astype(np.float64)
                    # Cache H_inv per layer (shared across all 256 experts)
                    if not hasattr(convert_model, '_gptq_hinv_cache'):
                        convert_model._gptq_hinv_cache = {}
                    if _layer_idx not in convert_model._gptq_hinv_cache:
                        damp = 0.01 * np.mean(np.diag(_gptq_hessian))
                        _H = _gptq_hessian.copy()
                        _H[np.diag_indices_from(_H)] += damp
                        try:
                            _L = np.linalg.cholesky(_H)
                            _hinv = np.linalg.solve(_L.T, np.linalg.solve(_L, np.eye(_H.shape[0], dtype=np.float64)))
                        except np.linalg.LinAlgError:
                            _H[np.diag_indices_from(_H)] += damp * 100
                            _hinv = np.linalg.inv(_H)
                        convert_model._gptq_hinv_cache[_layer_idx] = _hinv
                    _gptq_hinv = convert_model._gptq_hinv_cache[_layer_idx]

        # --- Quantize via mx.quantize() or GPTQ → MLX-native uint32 output ---
        try:
            import mlx.core as mx

            # Per-tensor group_size: router/gate gets gs=64, experts get model default
            tensor_gs = _get_tensor_group_size(tensor_name, block_size, num_experts)

            if is_3d:
                weights = weights.reshape(-1, weights.shape[-1])

            # Hadamard rotation: rotate weight rows before quantization.
            # Spreads outliers across dimensions → more uniform distribution →
            # less quantization error at the same bit width.
            # At inference, input is rotated with the same transform: signs cancel.
            # Math: y = hadamard_rotate(x, signs) @ W_rot^T = x @ W^T (exact)
            # Skip: MoE gates (float16 passthrough), norms, embeddings, lm_head
            if hadamard and bits >= 3:
                from .turboquant.rotation import generate_random_signs, hadamard_rotate
                in_dim = weights.shape[-1]
                _tn_lower_h = tensor_name.lower()
                _skip_rotation = (
                    "embed" in _tn_lower_h
                    or "lm_head" in _tn_lower_h
                    or "norm" in _tn_lower_h
                    or ".gate." in _tn_lower_h
                )
                if not _skip_rotation:
                    signs = generate_random_signs(in_dim, seed=42)
                    w_mx_pre = mx.array(weights.astype(np.float32))
                    w_rotated = hadamard_rotate(w_mx_pre, signs)
                    weights = np.array(w_rotated, dtype=np.float32)
                    del w_mx_pre, w_rotated
                    mx.clear_cache()

            # GPTQ path: use error feedback with Hessian for expert tensors
            if _gptq_hessian is not None and bits <= 4:
                if not hasattr(convert_model, '_gptq_logged'):
                    print(f"    GPTQ activated: {tensor_name} (bits={bits})")
                    convert_model._gptq_logged = True
                from .gptq_mlx import gptq_quantize_fast_with_hinv
                _gptq_w = weights.astype(np.float32)
                # For down_proj, Hessian is for hidden_size input but weight has
                # intermediate_size input — use identity-scaled H_inv
                if _gptq_hinv is not None and _gptq_hinv.shape[0] == _gptq_w.shape[1]:
                    _hinv_use = _gptq_hinv
                else:
                    # Fallback: identity H_inv (no error feedback, just MSE-optimal clipping)
                    _hinv_use = np.eye(_gptq_w.shape[1], dtype=np.float64) / max(np.mean(np.diag(_gptq_hessian)), 1e-10)
                mlx_weight, mlx_scales, mlx_biases = gptq_quantize_fast_with_hinv(
                    _gptq_w, _hinv_use, bits=bits, group_size=tensor_gs,
                )
                del _gptq_w
            else:
                # Standard RTN path via mx.quantize
                n_elements = weights.shape[0] * weights.shape[1]
                if n_elements > 100_000_000:
                    chunk_rows = max(1, 100_000_000 // weights.shape[1])
                    all_qw, all_s, all_b = [], [], []
                    for start in range(0, weights.shape[0], chunk_rows):
                        end = min(start + chunk_rows, weights.shape[0])
                        chunk = mx.array(weights[start:end].astype(np.float16))
                        cqw, cs, cb = mx.quantize(chunk, group_size=tensor_gs, bits=bits)
                        all_qw.append(np.array(cqw))
                        all_s.append(np.array(cs))
                        all_b.append(np.array(cb))
                        mx.synchronize()
                    mlx_weight = np.concatenate(all_qw, axis=0)
                    mlx_scales = np.concatenate(all_s, axis=0).astype(np.float16)
                    mlx_biases = np.concatenate(all_b, axis=0).astype(np.float16)
                else:
                    w_mx = mx.array(weights.astype(np.float16))
                    qw, scales, biases = mx.quantize(w_mx, group_size=tensor_gs, bits=bits)
                    mlx_weight = np.array(qw)
                    mlx_scales = np.array(scales).astype(np.float16)
                    mlx_biases = np.array(biases).astype(np.float16)

            # Reshape back to 3D for expert weights
            if is_3d:
                n_experts = w_shape[0]
                expert_out = w_shape[1]
                mlx_weight = mlx_weight.reshape(n_experts, expert_out, -1)
                mlx_scales = mlx_scales.reshape(n_experts, expert_out, -1)
                mlx_biases = mlx_biases.reshape(n_experts, expert_out, -1)

        except (ImportError, ValueError) as _exc:
            # Fallback: RTN quantization → convert to uint32 shaped
            if isinstance(_exc, ImportError):
                print(f"  WARNING: mlx not available for {tensor_name}, using RTN fallback (lower quality)")
                print(f"           Install mlx for best results: pip install 'jang[mlx]'")
            from .quantize import quantize_tensor
            _fallback_alloc = np.full(n_blocks, bits, dtype=np.uint8)
            qt = quantize_tensor(weights.reshape(-1, weights.shape[-1]) if is_3d else weights,
                                 _fallback_alloc, tensor_gs, method="rtn")
            out_dim = weights.reshape(-1, weights.shape[-1]).shape[0] if is_3d else weights.shape[0]
            in_dim = weights.shape[-1]
            packed_per_row = (in_dim * bits + 31) // 32
            n_groups = (in_dim + tensor_gs - 1) // tensor_gs

            packed_bytes = qt.qweight.tobytes()
            pad_needed = (4 - len(packed_bytes) % 4) % 4
            if pad_needed:
                packed_bytes += b'\x00' * pad_needed
            mlx_weight = np.frombuffer(packed_bytes, dtype=np.uint32)[:out_dim * packed_per_row].copy()
            mlx_weight = mlx_weight.reshape(out_dim, packed_per_row)
            mlx_scales = qt.scales.reshape(out_dim, n_groups)
            mlx_biases = qt.biases.reshape(out_dim, n_groups)

            if is_3d:
                n_experts = w_shape[0]
                expert_out = w_shape[1]
                mlx_weight = mlx_weight.reshape(n_experts, expert_out, -1)
                mlx_scales = mlx_scales.reshape(n_experts, expert_out, -1)
                mlx_biases = mlx_biases.reshape(n_experts, expert_out, -1)

        # Clear Metal cache between tensors
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass

        # --- Store with MLX-ready names and shapes ---
        base_name = tensor_name
        if base_name.endswith(".weight"):
            base_name = base_name[:-7]

        if awq_scales is not None:
            passthrough[f"{base_name}.awq_scales"] = awq_scales.astype(np.float16)

        # Handle gate_up_proj splitting (Qwen3.5 MoE fused projections)
        if "gate_up_proj" in base_name:
            if is_3d:
                # 3D: (n_experts, 2*inter, packed) → split into gate + up
                mid = mlx_weight.shape[1] // 2
                gate_base = base_name.replace("experts.gate_up_proj", "switch_mlp.gate_proj")
                up_base = base_name.replace("experts.gate_up_proj", "switch_mlp.up_proj")
                v2_tensors[f"{gate_base}.weight"] = mlx_weight[:, :mid, :]
                v2_tensors[f"{gate_base}.scales"] = mlx_scales[:, :mid, :]
                v2_tensors[f"{gate_base}.biases"] = mlx_biases[:, :mid, :]
                v2_tensors[f"{up_base}.weight"] = mlx_weight[:, mid:, :]
                v2_tensors[f"{up_base}.scales"] = mlx_scales[:, mid:, :]
                v2_tensors[f"{up_base}.biases"] = mlx_biases[:, mid:, :]
            else:
                # 2D: (2*inter, packed) → split into gate + up
                mid = mlx_weight.shape[0] // 2
                gate_base = base_name.replace("gate_up_proj", "gate_proj")
                up_base = base_name.replace("gate_up_proj", "up_proj")
                v2_tensors[f"{gate_base}.weight"] = mlx_weight[:mid, :]
                v2_tensors[f"{gate_base}.scales"] = mlx_scales[:mid, :]
                v2_tensors[f"{gate_base}.biases"] = mlx_biases[:mid, :]
                v2_tensors[f"{up_base}.weight"] = mlx_weight[mid:, :]
                v2_tensors[f"{up_base}.scales"] = mlx_scales[mid:, :]
                v2_tensors[f"{up_base}.biases"] = mlx_biases[mid:, :]

        # Handle 3D expert down_proj renaming
        elif is_3d and "experts" in base_name and "down_proj" in base_name:
            sw_base = base_name.replace("experts.down_proj", "switch_mlp.down_proj")
            v2_tensors[f"{sw_base}.weight"] = mlx_weight
            v2_tensors[f"{sw_base}.scales"] = mlx_scales
            v2_tensors[f"{sw_base}.biases"] = mlx_biases

        # Handle per-expert 2D tensors (MiniMax/Mixtral/GLM-5: experts.N.w1)
        # Stack into 3D as soon as all experts for a group are collected,
        # then move to v2_tensors for flushing. Prevents OOM from accumulating
        # 228 groups × 256 experts (~200 GB) in expert_buffer.
        elif _PER_EXPERT_PATTERN.match(tensor_name):
            m = _PER_EXPERT_PATTERN.match(tensor_name)
            prefix = m.group(1)
            expert_id = int(m.group(2))
            wtype = m.group(3)
            group_key = (prefix, wtype)
            if group_key not in expert_buffer:
                expert_buffer[group_key] = {}
            expert_buffer[group_key][expert_id] = {
                "weight": mlx_weight,
                "scales": mlx_scales,
                "biases": mlx_biases,
            }
            # Stack and flush as soon as all experts for this group are collected
            if not hasattr(convert_model, '_n_experts_expected'):
                convert_model._n_experts_expected = num_experts
            if len(expert_buffer[group_key]) >= convert_model._n_experts_expected:
                new_name = _EXPERT_NAME_MAP.get(wtype, wtype)
                sw_key = f"{prefix}.switch_mlp.{new_name}"
                n_exp = max(expert_buffer[group_key].keys()) + 1
                v2_tensors[f"{sw_key}.weight"] = np.stack(
                    [expert_buffer[group_key][e]["weight"] for e in range(n_exp)])
                v2_tensors[f"{sw_key}.scales"] = np.stack(
                    [expert_buffer[group_key][e]["scales"] for e in range(n_exp)])
                v2_tensors[f"{sw_key}.biases"] = np.stack(
                    [expert_buffer[group_key][e]["biases"] for e in range(n_exp)])
                del expert_buffer[group_key]

        # Standard tensor — store directly
        else:
            v2_tensors[f"{base_name}.weight"] = mlx_weight
            v2_tensors[f"{base_name}.scales"] = mlx_scales
            v2_tensors[f"{base_name}.biases"] = mlx_biases

        del weights

        # Incremental shard flush: write to disk when buffer exceeds 4 GB.
        # Prevents OOM on 744B+ models where v2_tensors would exceed RAM.
        _buf_bytes = sum(arr.nbytes for arr in v2_tensors.values())
        if _buf_bytes > 500 * 1024 ** 2:  # 500 MB — aggressive flush for 744B+ models
            if not hasattr(convert_model, '_shard_idx'):
                convert_model._shard_idx = 0
                output_path.mkdir(parents=True, exist_ok=True)
            convert_model._shard_idx += 1
            _shard_name = f"model-{convert_model._shard_idx:05d}-of-NNNNN.safetensors"
            _shard_path = output_path / _shard_name
            from safetensors.numpy import save_file as _save_shard
            _save_shard(v2_tensors, str(_shard_path), metadata={"format": "mlx"})
            if not hasattr(convert_model, '_shard_map'):
                convert_model._shard_map = {}
            for _k in v2_tensors:
                convert_model._shard_map[_k] = _shard_name
            print(f"    Flushed shard {convert_model._shard_idx}: {len(v2_tensors)} tensors, {_buf_bytes / 1e9:.1f} GB")
            v2_tensors.clear()
            import gc; gc.collect()

    # --- Stack per-expert 2D weights into 3D QuantizedSwitchLinear format ---
    if expert_buffer:
        print(f"  Stacking {len(expert_buffer)} expert groups into 3D...")
        for (prefix, wtype), experts in expert_buffer.items():
            new_name = _EXPERT_NAME_MAP.get(wtype, wtype)
            sw_key = f"{prefix}.switch_mlp.{new_name}"
            n_experts = max(experts.keys()) + 1

            v2_tensors[f"{sw_key}.weight"] = np.stack(
                [experts[e]["weight"] for e in range(n_experts)])
            v2_tensors[f"{sw_key}.scales"] = np.stack(
                [experts[e]["scales"] for e in range(n_experts)])
            v2_tensors[f"{sw_key}.biases"] = np.stack(
                [experts[e]["biases"] for e in range(n_experts)])
        expert_buffer.clear()

    # Step 4b: Collect non-quantized tensors (norms, biases, vision)
    print("  Collecting non-quantized tensors...")
    quantized_bases = set()
    # Include current v2_tensors AND pre-flushed shard tensors
    _all_quant_keys = list(v2_tensors.keys())
    _all_quant_keys.extend(getattr(convert_model, '_shard_map', {}).keys())
    for key in _all_quant_keys:
        if key.endswith(".weight"):
            quantized_bases.add(key[:-7])
        elif key.endswith(".scales") or key.endswith(".biases"):
            quantized_bases.add(key[:-7])

    for sf_path in weight_files:
        # Skip imatrix files (calibration-only, not model weights)
        if sf_path.name == "jang_imatrix.safetensors":
            continue
        with safe_open(str(sf_path), framework="numpy") as f:
            for tensor_name in f.keys():
                # Skip importance tensors (calibration-only)
                if tensor_name.endswith(".importance"):
                    continue
                base = tensor_name.replace(".weight", "")
                if base in quantized_bases:
                    continue

                is_norm = ("norm" in tensor_name.lower())
                is_bias = tensor_name.endswith(".bias")
                is_vision = (".visual." in tensor_name or "vision_tower" in tensor_name
                             or "vision_model" in tensor_name or "vision_encoder" in tensor_name
                             or "embed_vision" in tensor_name)
                shape = f.get_slice(tensor_name).get_shape()
                is_small = len(shape) == 1
                n_el = 1
                for d in shape:
                    n_el *= d
                is_tiny_nd = len(shape) > 2 and n_el < 100_000

                if is_norm or is_bias or is_small or is_tiny_nd or is_vision:
                    try:
                        tensor = f.get_tensor(tensor_name)
                    except TypeError:
                        tensor = _load_bf16_tensor(sf_path, tensor_name, shape)

                    if tensor.dtype == np.float32:
                        tensor = tensor.astype(np.float16)
                    elif tensor.dtype != np.float16:
                        tensor = tensor.astype(np.float16)

                    # Transpose 4D conv weights: PyTorch OIHW → MLX OHWI
                    if len(tensor.shape) == 4:
                        tensor = np.transpose(tensor, (0, 2, 3, 1))

                    passthrough[tensor_name] = tensor

    print(f"  Found {len(passthrough)} non-quantized tensors (norms, biases)")

    # Merge passthrough into v2_tensors
    v2_tensors.update(passthrough)

    # Step 5: Write JANG v2 model
    print(f"\n  [5/5] Writing JANG v2 model (MLX-native)...")

    # Account for pre-flushed shards (incremental shard flush for large models)
    _preflushed_map = getattr(convert_model, '_shard_map', {})
    _preflushed_idx = getattr(convert_model, '_shard_idx', 0)
    if _preflushed_map:
        print(f"  {_preflushed_idx} shards pre-flushed ({len(_preflushed_map)} tensors on disk)")

    model_config = json.loads((model_path / "config.json").read_text())
    # Strip source quantization_config (FP8 metadata) — leaving it causes mlx_lm
    # to misinterpret the model format. JANG uses "quantization" key instead.
    model_config.pop("quantization_config", None)
    bit_widths_used = sorted(set(_tensor_bits.values()))
    total_weight_bytes = sum(
        arr.nbytes for name, arr in v2_tensors.items()
        if name.endswith(".weight") and arr.dtype == np.uint32
    )

    jang_config = {
        "quantization": {
            "method": "jang-importance",
            "profile": profile if profile else None,
            "target_bits": target_bits,
            "actual_bits": round(actual_bits, 2),
            "block_size": block_size,
            "calibration_method": calibration_method,
            "quantization_method": quantization_method,
            "scoring_method": "weight-magnitude" if calibration_method == "weights" else "awq+hessian",
            "bit_widths_used": bit_widths_used,
            "quantization_scheme": "asymmetric",
            "quantization_backend": "mx.quantize",
            "hadamard_rotation": hadamard,
        },
        "source_model": {
            "name": model_path.name,
            "dtype": "bfloat16",
            "parameters": _count_params_str(model_config),
        },
        "architecture": {
            "type": arch_config.arch_type.value,
            "attention": arch_config.attention_type.value,
            # has_vision is true only if vision weights are preserved as float
            # (patch_embed skip in converter). If vision was quantized, VLM won't work.
            "has_vision": arch_config.has_vision_encoder and any(
                k for k in v2_tensors if "patch_embed" in k and v2_tensors[k].dtype != np.uint32
            ) if arch_config.has_vision_encoder else False,
            "has_ssm": arch_config.has_ssm_layers,
            "has_moe": arch_config.has_moe_layers,
        },
        "runtime": {
            "total_weight_bytes": total_weight_bytes,
            "total_weight_gb": round(total_weight_bytes / (1024 ** 3), 2),
        },
    }

    # Copy tokenizer files
    tokenizer_files = {}
    for tok_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "tokenizer.model", "merges.txt", "vocab.json", "added_tokens.json"]:
        tok_path = model_path / tok_file
        if tok_path.exists():
            if tok_file.endswith(".json"):
                tokenizer_files[tok_file] = json.loads(tok_path.read_text())
            else:
                tokenizer_files[tok_file] = tok_path.read_text()

    # ── eos_token_id auto-fix ──────────────────────────────────
    # Qwen3.5 source ships with eos_token_id=248044 (<|endoftext|>) which is WRONG.
    # Correct: 248046 (<|im_end|>). Wrong eos causes infinite thinking loops.
    # Fix in BOTH config.json and tokenizer_config.json.
    _eos_fixes = {
        "qwen3_5_moe": {248044: 248046},  # <|endoftext|> → <|im_end|>
        "qwen3_5": {248044: 248046},
    }
    model_type = model_config.get("model_type", "")
    text_cfg = model_config.get("text_config", {})
    if not model_type:
        model_type = text_cfg.get("model_type", "")
    eos_fix_map = _eos_fixes.get(model_type, {})
    if eos_fix_map:
        # Fix in text_config
        old_eos = text_cfg.get("eos_token_id")
        if old_eos in eos_fix_map:
            new_eos = eos_fix_map[old_eos]
            text_cfg["eos_token_id"] = new_eos
            model_config["text_config"] = text_cfg
            print(f"  eos_token_id fix: {old_eos} → {new_eos} (text_config)")
        # Fix in top-level config
        old_eos_top = model_config.get("eos_token_id")
        if old_eos_top in eos_fix_map:
            model_config["eos_token_id"] = eos_fix_map[old_eos_top]
            print(f"  eos_token_id fix: {old_eos_top} → {eos_fix_map[old_eos_top]} (top-level)")
        # Fix in tokenizer_config.json
        if "tokenizer_config.json" in tokenizer_files:
            tc = tokenizer_files["tokenizer_config.json"]
            if tc.get("eos_token_id") in eos_fix_map:
                tc["eos_token_id"] = eos_fix_map[tc["eos_token_id"]]

    # Copy VL processor, chat template, and extra config files.
    # Chat templates are CRITICAL — missing or wrong template causes:
    #   - Qwen3.5: infinite thinking loops if eos_token_id wrong
    #   - MiniMax: loops if enable_thinking toggle missing from template
    output_path.mkdir(parents=True, exist_ok=True)
    extra_configs = ["preprocessor_config.json", "video_preprocessor_config.json",
                     "chat_template.json", "chat_template.jinja",
                     "generation_config.json"]
    for extra_file in extra_configs:
        extra_path = model_path / extra_file
        if extra_path.exists():
            shutil.copy2(str(extra_path), str(output_path / extra_file))

    # Copy ALL custom .py files (trust_remote_code models: Nemotron, MiniMax, etc.)
    # Auto-detect instead of hardcoding specific filenames.
    py_files_copied = []
    for f in model_path.iterdir():
        if f.suffix == ".py" and f.is_file():
            shutil.copy2(str(f), str(output_path / f.name))
            py_files_copied.append(f.name)
    if py_files_copied:
        print(f"  Custom .py files: {', '.join(py_files_copied)}")

    # Verify chat template exists in tokenizer_config or as .jinja file
    has_inline_template = False
    if "tokenizer_config.json" in tokenizer_files:
        tc = tokenizer_files["tokenizer_config.json"]
        has_inline_template = bool(tc.get("chat_template"))
    has_jinja = (output_path / "chat_template.jinja").exists()
    if not has_inline_template and not has_jinja:
        print("  ⚠ WARNING: No chat template found (inline or .jinja). Model may loop during inference.")

    write_jang_v2_model(
        output_dir=output_path,
        tensors=v2_tensors,
        model_config=model_config,
        jang_config=jang_config,
        tokenizer_files=tokenizer_files,
        importance_data=importance_data,
        preflushed_map=_preflushed_map,
    )

    # Clean up incremental flush state
    if hasattr(convert_model, '_shard_idx'):
        del convert_model._shard_idx
    if hasattr(convert_model, '_shard_map'):
        del convert_model._shard_map

    results = {
        "source_model": str(model_path),
        "output_path": str(output_path),
        "target_bits": target_bits,
        "actual_bits": round(actual_bits, 2),
        "total_blocks": alloc_summary["total_blocks"],
        "total_weight_gb": round(total_weight_bytes / (1024 ** 3), 2),
        "histogram": alloc_summary["histogram"],
        "bit_widths_used": bit_widths_used,
    }

    print(f"\n{'='*60}")
    print(f"  DONE — JANG v2 (MLX-native)")
    print(f"  Output: {output_path}")
    print(f"  Size: {results['total_weight_gb']} GB")
    print(f"  Avg bits: {results['actual_bits']}")
    print(f"  Load time: instant (mx.load mmap)")
    print(f"{'='*60}\n")

    return results


def _count_params_str(config: dict) -> str:
    """Estimate parameter count from model config (best-effort, cosmetic only)."""
    # Check both top-level and text_config for VLM models
    tc = config.get("text_config", {})
    def _get(key, default=0):
        return config.get(key, tc.get(key, default))

    hidden = _get("hidden_size")
    n_layers = _get("num_hidden_layers")
    intermediate = _get("intermediate_size")
    vocab = _get("vocab_size")
    num_experts = _get("num_local_experts", _get("num_experts")) or 0

    attn_params = 4 * hidden * hidden
    mlp_params = 3 * hidden * intermediate
    if num_experts > 1:
        mlp_params *= num_experts  # MoE: multiply by expert count
    layer_params = attn_params + mlp_params
    total = vocab * hidden + n_layers * layer_params

    if total > 1e9:
        return f"{total / 1e9:.1f}B"
    elif total > 1e6:
        return f"{total / 1e6:.0f}M"
    else:
        return f"{total:.0f}"

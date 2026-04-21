"""
JANGTQ loader — drop-in MLX inference path for `weight_format: mxtq` models.
Created by Jinho Jang (eric@jangq.ai)

JANGTQ ("Turbo Quant") is the most-compressed, highest-quality JANG quant
format. Weights stay in their codebook+Hadamard form at runtime — no
dequant to affine. The matmul path uses custom Metal kernels (P2-P18)
that read packed uint32 weights, look up centroids in a 4-entry codebook,
and accumulate dot products against a Hadamard-rotated input.

This module is fully self-contained: the only externals it needs are
`mlx`, `mlx_lm`, and `jang_tools.{loader, turboquant}`. Drop into any
inference engine by calling `load_jangtq_model(path)` and using the
returned `(model, tokenizer)` pair with `mlx_lm.generate` (or any equivalent
generate loop).

What gets applied on load (all class-level monkeypatches that survive
the function call so subsequent decodes are fast):

  * P3  — single-dispatch multi-block Hadamard (`jang_tools.turboquant.hadamard_kernel`)
  * P15 — `mx.compile`'d router math + compile-friendly `_mlp` fast path
  * P17 — thread-tiling sweet spot OPT=10 (fused_gate_up_swiglu) and
          OPT=20 (gather_tq_matmul) — re-swept per Apple GPU generation
  * P18 — QKV fusion in attention (3 quantized matmuls → 1 + slice views)

Architectures supported:
  * MiniMax M2.7 (`minimax_m2`) — standard Q/K/V attention, sigmoid+bias router,
    no shared expert; all patches apply. Measured 44.3 tok/s on M3 Ultra.
  * Qwen3.5 / Qwen3.6 / Qwen3-Next (`qwen3_5_moe`, `qwen3_next`) — hybrid
    linear_attn + full_attn, softmax+topk router, shared expert with sigmoid
    gate, pre-stacked experts with combined gate_up_proj (split at load).
    P3/P15 apply to all MoE layers; P18 QKV fusion applies only to full_attn
    layers (linear_attn layers skip silently — they have a different layout).
  * GLM-5.1 (`glm_moe_dsa`) — MLA attention; P18 QKV fusion will silently
    skip because it has q_a_proj/kv_a_proj_with_mqa instead of q/k/v_proj.
    The MoE-side P3/P15 still apply. Shared expert (if present) handled.

See `research/JANGTQ-REFERENCE.md` for full architecture, math, kernel
inventory, and known traps.
"""
import json, gc, re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from jang_tools.turboquant.tq_kernel import TurboQuantLinear, TurboQuantSwitchLinear


def _apply_wired_limit_safe_default():
    """Set mx.wired_limit to ~70% of total RAM if the caller hasn't already.

    Why: for per-expert JANGTQ bundles (e.g. GLM-5.1-JANGTQ_1L, ~190 GB),
    the loader's `mx.stack(packed_list)` at inference-time creates fresh
    non-mmap-backed stacked expert tensors. If MLX's wired_limit is too
    high (e.g. 240 GB on a 256 GB machine), macOS has no pagecache
    headroom to spill the materialization spike and SIGKILLs the process
    silently (no OOM traceback — just vanishes).

    Ralph iter-14 measurement: setting wired_limit=180 GB on 256 GB
    Mac Studio let the forward complete in 80 s (cold I/O) instead of
    SIGKILL. 70% of total RAM is a safe general-case default; callers
    who have already set their own limit (higher or lower) are respected.

    Enabled only on Apple Silicon macOS where psutil is available.
    No-op elsewhere.
    """
    try:
        import psutil, sys, mlx.core as _mx
        if sys.platform != "darwin":
            return
        total_gb = psutil.virtual_memory().total / 1e9
        target_gb = int(total_gb * 0.70)
        # Clamp to reasonable range: at least 32 GB, at most 220 GB
        # (leaving enough headroom even on very-large-RAM machines).
        target_gb = max(32, min(target_gb, 220))
        target_bytes = target_gb * 1000 * 1000 * 1000
        _mx.set_wired_limit(target_bytes)
        print(f"  [wired_limit] auto-set to {target_gb} GB "
              f"(~70% of {total_gb:.0f} GB total RAM; ralph iter-14 tuning)",
              flush=True)
    except Exception as _e:
        # Non-fatal: older MLX, no psutil, non-Apple OS, etc.
        pass


def load_jangtq_model(model_path, skip_params_eval=False):
    """Load JANGTQ model with TurboQuantLinear (Metal kernel, no dequant).

    Automatically caps MLX's wired_limit at ~70% of total RAM to avoid the
    forward-pass OOM seen on large per-expert bundles (GLM-5.1-JANGTQ_1L)
    where non-mmap stacked expert tensors need pagecache headroom to
    materialize on first forward. See `_apply_wired_limit_safe_default`
    for the Ralph iter-14 investigation.
    """
    _apply_wired_limit_safe_default()

    model_path = Path(model_path)
    # M125 (iter 48): context-manage reads so fds close deterministically.
    with open(model_path / "config.json") as f:
        config = json.load(f)

    jang_cfg_path = model_path / "jang_config.json"
    if jang_cfg_path.exists():
        with open(jang_cfg_path) as f:
            jang_cfg = json.load(f)
    else:
        jang_cfg = {}
    mxtq_seed = jang_cfg.get("mxtq_seed", 42)
    mxtq_bits_map = jang_cfg.get("mxtq_bits", {})

    print(f"Loading JANGTQ: {model_path.name}", flush=True)
    print(f"  seed={mxtq_seed}, bits_map={mxtq_bits_map}", flush=True)

    from mlx_lm.utils import load_config, load_model as _load_skeleton, load_tokenizer
    model_config = load_config(model_path)
    if "quantization" not in model_config:
        model_config["quantization"] = {"group_size": 64, "bits": 2}

    model, model_config = _load_skeleton(
        model_path, lazy=True, strict=False, model_config=model_config
    )

    _hydrate_jangtq_model(
        model=model,
        model_path=model_path,
        mxtq_seed=mxtq_seed,
        mxtq_bits_map=mxtq_bits_map,
        model_config=model_config,
        skip_params_eval=skip_params_eval,
    )

    eos_ids = config.get("eos_token_id")
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    tokenizer = load_tokenizer(model_path, eos_token_ids=eos_ids)
    print(f"  Done (eos_token_ids={eos_ids})", flush=True)
    return model, tokenizer


def _vlm_minimal_sanitize(model, weights):
    """VLM weight sanitize that skips the (already-applied) expert split.

    Mirrors the LLM-path mlx_lm.qwen3_5.sanitize behavior — same as what
    `load_jangtq_model` triggers via `model.sanitize(regular)` — but
    avoids mlx_vlm's qwen3_5_moe.sanitize which would also try to
    re-split routed experts (we already split them via TQ above) and
    unconditionally shift norm weights (mlx_lm only shifts conditionally
    on `has_unsanitized_conv1d`, and our artifact's conv1d shape `(out,
    1, kernel)` always trips that flag, so the shift IS needed).

    Steps applied:
      * drop mtp.* (speculative decode extras — unused at inference time)
      * drop lm_head.weight when tie_word_embeddings
      * rename `model.language_model.*` → `language_model.model.*`
      * rename `model.visual.*` → `vision_tower.*` (alternate HF layout)
      * rename `lm_head.*` → `language_model.lm_head.*`
      * conv1d.weight: moveaxis(2, 1) so shape (out, 1, k) → (out, k, 1)
      * RMSNorm.weight: `+= 1.0` (same shift mlx_lm applies for our
        artifact's pre-shift convention)
    """
    text_tied = False
    tc = getattr(model.config, "text_config", None)
    if tc is not None:
        text_tied = bool(getattr(tc, "tie_word_embeddings", False))

    norm_suffixes = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    )

    out = {}
    for key, value in weights.items():
        if "mtp." in key:
            continue
        if text_tied and key == "lm_head.weight":
            continue
        if key.startswith("model.language_model"):
            key = key.replace("model.language_model", "language_model.model")
        elif key.startswith("model.visual"):
            key = key.replace("model.visual", "vision_tower")
        elif key.startswith("lm_head"):
            key = key.replace("lm_head", "language_model.lm_head")

        # GatedDeltaNet conv1d weights are stored on disk as
        # (out, 1, kernel) but mlx's `nn.Conv1d` expects (out, kernel, 1).
        # Same fix mlx_vlm + mlx_lm do in their sanitize passes.
        if "conv1d.weight" in key and value.ndim == 3 and value.shape[-1] != 1:
            value = value.moveaxis(2, 1)

        # RMSNorm `+= 1.0`: matches mlx_lm.qwen3_5.sanitize behavior when
        # `has_unsanitized_conv1d` (always true for our artifact). The
        # text-path verifies this is correct: norms saved at ~0.03 mean,
        # decoded after shift to ~1.03, produces coherent text at 96 tok/s.
        if value.ndim == 1 and any(key.endswith(sfx) for sfx in norm_suffixes):
            value = value + 1.0

        out[key] = value
    return out


def _hydrate_jangtq_model(model, model_path, mxtq_seed, mxtq_bits_map,
                          model_config, skip_params_eval=False):
    """Apply JANGTQ TQ replacement, weight load, and runtime patches in-place.

    Shared by `load_jangtq_model` (LLM-only) and `load_jangtq_vlm.load_jangtq_vlm_model`
    (VLM with vision_tower). Caller is responsible for building `model` and (for
    LLM path) the tokenizer/processor.
    """
    model_path = Path(model_path)
    weight_files = sorted(model_path.glob("model-*.safetensors"))
    print(f"  {len(weight_files)} shards", flush=True)

    # Collect TQ groups + regular weights
    tq_groups = {}  # base_path -> {packed, norms, bits}
    regular = {}
    for shard_i, sf_path in enumerate(weight_files):
        weights = mx.load(str(sf_path))
        for k, v in weights.items():
            if k.endswith(".tq_packed"):
                tq_groups.setdefault(k[:-10], {})["packed"] = v
            elif k.endswith(".tq_norms"):
                tq_groups.setdefault(k[:-9], {})["norms"] = v
            elif k.endswith(".tq_bits"):
                tq_groups.setdefault(k[:-8], {})["bits"] = int(v[0].item())
            else:
                regular[k] = v
        if (shard_i + 1) % 40 == 0:
            print(f"  shard {shard_i + 1}/{len(weight_files)}", flush=True)

    print(f"  TQ groups: {len(tq_groups)}, regular: {len(regular)}", flush=True)

    # Stack per-expert TQ tensors into 3D switch_mlp tensors.
    # Three naming conventions handled:
    #   GLM-5.1:       model.layers.L.mlp.experts.E.{gate_proj,up_proj,down_proj}
    #   MiniMax M2.7:  model.layers.L.block_sparse_moe.experts.E.{w1,w2,w3}
    #                  (w1=gate, w2=down, w3=up — Mixtral convention)
    #   Qwen3.5/3.6:   [model.language_model.]layers.L.mlp.experts.{gate_up_proj,down_proj}
    #                  ALREADY 3D-stacked ([n_experts, ...]); gate_up_proj is combined (split at mid).
    # All converge to: prefix + switch_mlp.{gate_proj,up_proj,down_proj}
    # Three prefix flavors appear across arches and sanitize passes:
    #   model.                         — standard MiniMax / GLM / Qwen3
    #   model.language_model.          — raw HF Qwen3.5/3.6 VLM
    #   language_model.model.          — post-sanitize Qwen3.5/3.6 VLM
    _VLM_PREFIX = r"(?:model\.language_model\.|language_model\.model\.|model\.)"
    glm_pat = re.compile(rf"^({_VLM_PREFIX}layers\.\d+\.mlp\.)experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")
    mm_pat  = re.compile(rf"^({_VLM_PREFIX}layers\.\d+\.block_sparse_moe\.)experts\.(\d+)\.(w[123])$")
    qw_pat  = re.compile(rf"^({_VLM_PREFIX}layers\.\d+\.mlp\.)experts\.(gate_up_proj|down_proj|gate_proj|up_proj)$")
    mm_map  = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
    grouped_experts = {}
    # Pre-stacked tensors go straight into tq_groups with switch_mlp naming (no stacking).
    prestacked = {}  # new_base -> dict(packed/norms/bits)

    for base in list(tq_groups.keys()):
        m = glm_pat.match(base)
        if m:
            layer_prefix = m.group(1)
            expert_id = int(m.group(2))
            proj_name = m.group(3)
            grouped_experts.setdefault((layer_prefix, proj_name), {})[expert_id] = tq_groups.pop(base)
            continue
        m = mm_pat.match(base)
        if m:
            layer_prefix = m.group(1)
            expert_id = int(m.group(2))
            proj_name = mm_map[m.group(3)]
            grouped_experts.setdefault((layer_prefix, proj_name), {})[expert_id] = tq_groups.pop(base)
            continue
        m = qw_pat.match(base)
        if m:
            layer_prefix = m.group(1)
            proj_name = m.group(2)
            parts = tq_groups.pop(base)
            # Expect packed.ndim == 3 (pre-stacked [n_experts, out, packed_in]).
            # If converter already split gate_up → gate_proj + up_proj, proj_name
            # is gate_proj/up_proj — handle like any other pre-stacked projection.
            if proj_name == "gate_up_proj":
                # Split combined gate+up along the output-row axis.
                # packed: (n_exp, 2*inter_packed, in_packed_cols)
                # norms:  (n_exp, 2*inter_packed)
                packed = parts["packed"]
                norms = parts["norms"]
                bits = parts["bits"]
                # In MLX's qwen3_5_moe.sanitize: mid = gate_up.shape[-2] // 2
                mid = packed.shape[-2] // 2
                for half, name in ((slice(None, mid), "gate_proj"),
                                   (slice(mid, None), "up_proj")):
                    new_base = f"{layer_prefix}switch_mlp.{name}"
                    prestacked[new_base] = {
                        "packed": packed[..., half, :],
                        "norms": norms[..., half],
                        "bits": bits,
                    }
            else:
                new_base = f"{layer_prefix}switch_mlp.{proj_name}"
                prestacked[new_base] = parts
            continue

    print(f"  Expert groups to stack: {len(grouped_experts)}, pre-stacked: {len(prestacked)}", flush=True)

    for (layer_prefix, proj_name), experts in grouped_experts.items():
        n_exp = max(experts.keys()) + 1
        packed_list = [experts[e]["packed"] for e in range(n_exp)]
        norms_list = [experts[e]["norms"] for e in range(n_exp)]
        bits = experts[0]["bits"]
        stacked_packed = mx.stack(packed_list)
        stacked_norms = mx.stack(norms_list)
        new_base = f"{layer_prefix}switch_mlp.{proj_name}"
        tq_groups[new_base] = {
            "packed": stacked_packed,
            "norms": stacked_norms,
            "bits": bits,
        }
    del grouped_experts
    # Merge pre-stacked (Qwen3.5/3.6) groups straight in.
    for new_base, parts in prestacked.items():
        tq_groups[new_base] = parts
    del prestacked
    gc.collect()
    print(f"  After stacking: {len(tq_groups)} TQ groups", flush=True)

    # Replace modules with TurboQuant variants
    print("  Replacing modules with TurboQuantLinear...", flush=True)
    n_replaced = 0

    def get_module(root, dotted):
        cur = root
        for p in dotted.split("."):
            if p.isdigit():
                cur = cur[int(p)]
            else:
                cur = getattr(cur, p)
        return cur

    def set_module(root, dotted, new_mod):
        parts = dotted.split(".")
        cur = root
        for p in parts[:-1]:
            if p.isdigit():
                cur = cur[int(p)]
            else:
                cur = getattr(cur, p)
        last = parts[-1]
        if last.isdigit():
            cur[int(last)] = new_mod
        else:
            setattr(cur, last, new_mod)

    for base, parts in list(tq_groups.items()):
        packed = parts["packed"]
        norms = parts["norms"]
        bits = parts["bits"]
        vals_per_u32 = 32 // bits

        try:
            existing = get_module(model, base)
        except (AttributeError, IndexError, KeyError):
            print(f"    Skip (not in model): {base}", flush=True)
            continue

        if packed.ndim == 3:
            n_exp, out_feat, packed_cols = packed.shape
            in_features = packed_cols * vals_per_u32
            new_module = TurboQuantSwitchLinear(
                in_features=in_features, out_features=out_feat,
                num_experts=n_exp, bits=bits, bias=False, seed=mxtq_seed,
            )
        else:
            out_feat, packed_cols = packed.shape
            in_features = packed_cols * vals_per_u32
            new_module = TurboQuantLinear(
                in_features=in_features, out_features=out_feat,
                bits=bits, bias=False, seed=mxtq_seed,
            )
        new_module.packed = packed
        new_module.norms = norms

        set_module(model, base, new_module)
        n_replaced += 1

    print(f"  Replaced {n_replaced} modules", flush=True)
    del tq_groups
    gc.collect()

    # Load regular weights
    if hasattr(model, "sanitize"):
        # mlx_vlm's qwen3_5_moe.sanitize would re-split experts (already split
        # via TQ above) and unconditionally shift norm weights by +1.0 (already
        # baked into our converted artifact). Detect VLM models by the presence
        # of vision_tower and use a minimal sanitize that only handles key
        # renames + lm_head tie + mtp drops.
        is_vlm_model = hasattr(model, "vision_tower") and hasattr(model, "language_model")
        if is_vlm_model:
            regular = _vlm_minimal_sanitize(model, regular)
        else:
            regular = model.sanitize(regular)
    model.load_weights(list(regular.items()), strict=False)
    del regular
    gc.collect()

    # P7 + P15: replace SwitchGLU.__call__ with a two-path version:
    #
    #   Fast path (decode, batch=1, K<64, not training):
    #     Uses mx.compile'd closure that bakes meta/grid/threadgroup as Python
    #     constants and runs: rotate → fused_gate_up_swiglu → rotate → gather_dn.
    #     Separate `gp.codebook` (keyed on in_features=hidden) and `dp.codebook`
    #     (keyed on in_features=inter) MUST be passed — codebooks differ by
    #     exactly sqrt(inter/hidden) so mixing them silently scales output.
    #
    #   Slow path (prefill, sorted routing, training):
    #     Calls the dynamic-shape-detection helpers
    #     `fused_gate_up_swiglu_matmul` + `gather_tq_matmul` directly.
    #
    # Class-level monkey-patch because Python looks up `__call__` on the type.
    try:
        from mlx_lm.models.switch_layers import SwitchGLU, _gather_sort, _scatter_unsort
        from jang_tools.turboquant.fused_gate_up_kernel import (
            fused_gate_up_matmul, fused_gate_up_swiglu_matmul,
            make_fused_gate_up_swiglu_decode,
        )
        from jang_tools.turboquant.gather_tq_kernel import (
            gather_tq_matmul, make_gather_tq_decode_per_row,
            make_fused_rot_gather_decode,
        )
        from jang_tools.turboquant.hadamard_kernel import hadamard_rotate_metal

        # P15: compiled decode helpers, cached per (in_f, out_f, bits, K).
        _DECODE_COMPILED = {}

        def _get_compiled_decode(in_f, out_f, bits, K):
            key = (in_f, out_f, bits, K)
            if key in _DECODE_COMPILED:
                return _DECODE_COMPILED[key]
            fused_gu = make_fused_gate_up_swiglu_decode(in_f, out_f, bits, K)
            # P19 (fused rot+gather) was tested but regressed 44.6 → 34.2 tok/s
            # in real decode despite +3 μs microbench win. Likely shmem/occupancy
            # interaction with other kernels. Reverted to split path.
            gather_dn = make_gather_tq_decode_per_row(out_f, in_f, bits, K)

            def _mlp(x_flat, pg, ng, pu, nu, pd, nd, cb_gate, cb_down, signs_in, signs_dn, idx):
                x_rot = hadamard_rotate_metal(x_flat, signs_in)
                x_act = fused_gu(x_rot, pg, ng, pu, nu, cb_gate, idx)  # (K, out_f)
                x_act_rot = hadamard_rotate_metal(x_act, signs_dn)
                y = gather_dn(x_act_rot, pd, nd, cb_down, idx)  # (K, in_f)
                return y

            _DECODE_COMPILED[key] = mx.compile(_mlp)
            return _DECODE_COMPILED[key]

        def _fused_switchglu_call(self, x, indices):
            # Fallback for non-TQ switch layers
            gp = self.gate_proj
            up = self.up_proj
            dp = self.down_proj
            if not isinstance(gp, TurboQuantSwitchLinear) or not isinstance(up, TurboQuantSwitchLinear):
                return _ORIG_SWITCHGLU_CALL(self, x, indices)

            # Decode fast path: batch=1, K=topk, broadcast mode.
            # Detect by: x has flattenable-to-(1, in_f) layout AND indices is 1D-equivalent.
            x_sq = x
            while x_sq.ndim > 2 and x_sq.shape[-2] == 1:
                x_sq = x_sq.squeeze(-2)
            x_flat = x_sq.reshape(-1, gp.in_features)
            batch = x_flat.shape[0]
            K = indices.shape[-1] if indices.ndim > 0 else 1
            do_sort_ok = indices.ndim >= 1 and indices.size < 64
            can_fast = (batch == 1 and K > 0 and do_sort_ok and not getattr(self, "training", False))

            if can_fast:
                idx_flat = indices.reshape(-1).astype(mx.uint32)
                compiled_mlp = _get_compiled_decode(gp.in_features, gp.out_features, gp.bits, K)
                y = compiled_mlp(
                    x_flat.astype(mx.float32),
                    gp.packed, gp.norms, up.packed, up.norms,
                    dp.packed, dp.norms,
                    gp.codebook, dp.codebook,
                    gp.signs, dp.signs, idx_flat,
                )  # (K, in_f)
                out = y.reshape(*indices.shape[:-1], K, 1, gp.in_features)
                if out.dtype != x.dtype:
                    out = out.astype(x.dtype)
                return out.squeeze(-2)

            # Slow path: original fused call with dynamic shape detection.
            x_exp = mx.expand_dims(x, (-2, -3))
            do_sort = indices.size >= 64
            idx = indices
            inv_order = None
            if do_sort:
                x_exp, idx, inv_order = _gather_sort(x_exp, indices)
            if getattr(self, "training", False):
                idx = mx.stop_gradient(idx)

            x_act = fused_gate_up_swiglu_matmul(
                x_exp,
                gp.packed, gp.norms,
                up.packed, up.norms,
                gp.codebook, gp.signs,
                idx,
                bits=gp.bits,
            )
            x_out = self.down_proj(x_act, idx, sorted_indices=do_sort)
            if do_sort:
                x_out = _scatter_unsort(x_out, inv_order, indices.shape)
            return x_out.squeeze(-2)

        _ORIG_SWITCHGLU_CALL = SwitchGLU.__call__
        SwitchGLU.__call__ = _fused_switchglu_call

        # Count how many instances will benefit
        patched = sum(
            1 for name, m in model.named_modules()
            if isinstance(m, SwitchGLU)
            and isinstance(getattr(m, "gate_proj", None), TurboQuantSwitchLinear)
            and isinstance(getattr(m, "up_proj", None), TurboQuantSwitchLinear)
        )
        print(f"  Patched SwitchGLU class for fused gate+up ({patched} TQ instances)", flush=True)
    except Exception as _e:
        print(f"  SwitchGLU fusion skipped: {_e}", flush=True)

    # P15: mx.compile ONLY pure router math (mlx_lm pattern — free function,
    # no closures over self, shapeless so prefill/decode share one graph).
    # Safer than patching __call__ which thrashes the compile cache.
    try:
        # Cache compiled variants keyed on (k, bits).
        _ROUTER_CACHE = {}
        _MLP_CACHE = {}

        # Variant A: sigmoid + e_score_correction_bias topk (MiniMax, GLM, DeepSeek V3).
        def _get_compiled_router_sigmoid_bias(k):
            key = ("sigbias", k)
            if key in _ROUTER_CACHE:
                return _ROUTER_CACHE[key]
            def _router(gates_f32, e_bias):
                scores = mx.sigmoid(gates_f32)
                orig = scores
                scores = scores + e_bias
                inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
                sel = mx.take_along_axis(orig, inds, axis=-1)
                sel = sel / (mx.sum(sel, axis=-1, keepdims=True) + 1e-20)
                return inds, sel
            _ROUTER_CACHE[key] = mx.compile(_router)
            return _ROUTER_CACHE[key]

        # Variant B: softmax → topk → optional renormalize (Qwen3-Next, Qwen3.5/3.6).
        # Matches mlx_lm.models.qwen3_next.Qwen3NextSparseMoeBlock.__call__ semantics.
        def _get_compiled_router_softmax(k, renorm):
            key = ("softmax", k, bool(renorm))
            if key in _ROUTER_CACHE:
                return _ROUTER_CACHE[key]
            if renorm:
                def _router(gates_f32):
                    scores = mx.softmax(gates_f32, axis=-1, precise=True)
                    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
                    sel = mx.take_along_axis(scores, inds, axis=-1)
                    sel = sel / (mx.sum(sel, axis=-1, keepdims=True) + 1e-20)
                    return inds, sel
            else:
                def _router(gates_f32):
                    scores = mx.softmax(gates_f32, axis=-1, precise=True)
                    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
                    sel = mx.take_along_axis(scores, inds, axis=-1)
                    return inds, sel
            _ROUTER_CACHE[key] = mx.compile(_router)
            return _ROUTER_CACHE[key]

        # Back-compat name (kept so any external references don't break).
        _get_compiled_router = _get_compiled_router_sigmoid_bias

        # Compiling the MLP path regressed 35.5 → 30 tok/s AND changed output
        # length (likely recompile thrash — metal_kernel internal mx.array alloc
        # for `meta` invalidates the trace every call). Router-only is safe.
        #
        # Unified patch handles three arch families via hasattr probes:
        #   - `e_score_correction_bias` present → sigmoid-bias router (MiniMax/GLM/DSV3)
        #   - `e_score_correction_bias` absent  → softmax router (Qwen3-Next/3.5/3.6)
        #   - `shared_expert` + `shared_expert_gate` present → add gated shared output
        _n_patched = 0
        _patched_classes = set()
        for _, _mod in model.named_modules():
            _cls = _mod.__class__
            if _cls in _patched_classes or "SparseMoeBlock" not in _cls.__name__:
                continue

            def _patched_moe(self, x):
                gates = self.gate(x.astype(mx.float32))
                # MiniMax stores K as `num_experts_per_tok`, Qwen3-Next as `top_k`,
                # some GLM variants as `num_routed_experts`. Probe in order.
                k = getattr(self, "num_experts_per_tok",
                            getattr(self, "top_k",
                                    getattr(self, "num_routed_experts", None)))
                if k is None:
                    raise AttributeError(
                        f"{type(self).__name__} missing one of "
                        "num_experts_per_tok / top_k / num_routed_experts"
                    )
                if hasattr(self, "e_score_correction_bias"):
                    inds, scores = _get_compiled_router_sigmoid_bias(k)(
                        gates, self.e_score_correction_bias
                    )
                else:
                    renorm = getattr(self, "norm_topk_prob", True)
                    inds, scores = _get_compiled_router_softmax(k, renorm)(gates)
                scores = scores.astype(x.dtype)
                y = self.switch_mlp(x, inds)
                y = (y * scores[..., None]).sum(axis=-2)
                # Shared expert (always active, gated): Qwen3-Next/3.5/3.6; some GLM configs.
                shared = getattr(self, "shared_expert", None)
                if shared is not None:
                    sh_out = shared(x)
                    gate_lin = getattr(self, "shared_expert_gate", None)
                    if gate_lin is not None:
                        sh_out = sh_out * mx.sigmoid(gate_lin(x))
                    y = y + sh_out
                return y

            _cls.__call__ = _patched_moe
            _patched_classes.add(_cls)
            _n_patched += 1
        print(f"  P15 mx.compile(router-only) applied to {_n_patched} MoE class(es)", flush=True)
    except Exception as _e:
        print(f"  P15 skipped: {_e}", flush=True)

    # Fix mixed bit widths in remaining QuantizedLinear/QuantizedEmbedding
    # (embed_tokens, lm_head, attention may be at different bits than config default)
    # Self-contained — uses jang_tools.loader's bit-width fixer. No vMLX dependency.
    from jang_tools.loader import _fix_quantized_bits
    _fix_quantized_bits(model, {})

    # MLA models (GLM-5.1 glm_moe_dsa, Mistral-4 MLA, etc.) have an extra
    # `QuantizedMultiLinear` class (mlx_lm.models.mla) for the embed_q/unembed_out
    # absorbed projections. These are NOT handled by _fix_quantized_bits (which
    # only knows QuantizedLinear/Embedding/SwitchLinear), so their `.bits` stays
    # at the skeleton default while the actual packed weights are 8-bit → shape
    # mismatch on first forward. Fix by walking the model and inferring bits
    # from weight/scales shapes, same as _fix_quantized_bits does for 2D tensors.
    try:
        from mlx_lm.models.mla import QuantizedMultiLinear
        _mla_fixed = 0
        for _name, _mod in model.named_modules():
            if not isinstance(_mod, QuantizedMultiLinear):
                continue
            if not (hasattr(_mod, "weight") and hasattr(_mod, "scales")):
                continue
            # weight: (num_heads, out, in_packed), scales: (num_heads, out, in/gs)
            w_cols = _mod.weight.shape[-1]
            s_cols = _mod.scales.shape[-1]
            for try_gs in (_mod.group_size, 64, 128):
                in_dim = s_cols * try_gs
                if in_dim <= 0:
                    continue
                if (w_cols * 32) % in_dim != 0:
                    continue
                try_bits = (w_cols * 32) // in_dim
                if try_bits in (2, 3, 4, 5, 6, 8):
                    if try_bits != _mod.bits:
                        _mod.bits = try_bits
                    if try_gs != _mod.group_size:
                        _mod.group_size = try_gs
                    _mla_fixed += 1
                    break
        if _mla_fixed:
            print(f"  MLA QuantizedMultiLinear bits fixed: {_mla_fixed} modules", flush=True)
    except ImportError:
        pass  # mlx_lm version without mla.py — MLA models unsupported anyway

    # P18: QKV fusion — must run AFTER _fix_quantized_bits so the attention
    # QuantizedLinear modules report the correct bits (8 instead of default 2).
    # Concat q/k/v weights into one matmul: 3 dispatches → 1 per layer.
    try:
        import mlx.nn as _nn
        _qkv_fused = {}  # id(attn_instance) -> (fused_linear, Hq, Hk, Hv)
        _patched_attn_classes = set()
        for _, _mod in model.named_modules():
            if not (hasattr(_mod, "q_proj") and hasattr(_mod, "k_proj") and hasattr(_mod, "v_proj")):
                continue
            q, k, v = _mod.q_proj, _mod.k_proj, _mod.v_proj
            if not all(isinstance(p, _nn.QuantizedLinear) for p in (q, k, v)):
                continue
            if not (q.bits == k.bits == v.bits and q.group_size == k.group_size == v.group_size):
                continue
            # Skip P18 fusion when q_proj is doubled for the
            # attn_output_gate split (Qwen 3.5/3.6). The original
            # __call__ knows to split q into (queries | gate); the
            # patched fused path here doesn't, and would feed an
            # over-wide query into SDPA. Detection: q.weight rows are
            # 2x the standard Hq when num_heads * head_dim * 2 matches.
            num_heads = getattr(_mod, "num_attention_heads", None) \
                or getattr(_mod, "n_heads", None) or 0
            head_dim = getattr(_mod, "head_dim", 0)
            if num_heads and head_dim and q.weight.shape[0] == num_heads * head_dim * 2:
                continue
            Hq, Hk, Hv = q.weight.shape[0], k.weight.shape[0], v.weight.shape[0]
            in_f = q.weight.shape[1] * (32 // q.bits)
            fused = _nn.QuantizedLinear(
                input_dims=in_f, output_dims=Hq + Hk + Hv,
                bias=False, group_size=q.group_size, bits=q.bits,
            )
            fused.weight = mx.concatenate([q.weight, k.weight, v.weight], axis=0)
            fused.scales = mx.concatenate([q.scales, k.scales, v.scales], axis=0)
            fused.biases = mx.concatenate([q.biases, k.biases, v.biases], axis=0)
            _qkv_fused[id(_mod)] = (fused, Hq, Hk, Hv)
            _patched_attn_classes.add(_mod.__class__)

        for _attn_cls in _patched_attn_classes:
            _orig_attn_call = _attn_cls.__call__
            def _make_patched(orig_call):
                def _patched(self, x, mask=None, cache=None):
                    info = _qkv_fused.get(id(self))
                    if info is None:
                        return orig_call(self, x, mask=mask, cache=cache)
                    fused, Hq, Hk, Hv = info
                    B, L, _ = x.shape
                    qkv = fused(x)
                    queries = qkv[..., :Hq]
                    keys = qkv[..., Hq:Hq + Hk]
                    values = qkv[..., Hq + Hk:]
                    if getattr(self, "use_qk_norm", False):
                        queries = self.q_norm(queries)
                        keys = self.k_norm(keys)
                    queries = queries.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
                    keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
                    values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
                    if cache is not None:
                        queries = self.rope(queries, offset=cache.offset)
                        keys = self.rope(keys, offset=cache.offset)
                        keys, values = cache.update_and_fetch(keys, values)
                    else:
                        queries = self.rope(queries); keys = self.rope(keys)
                    from mlx_lm.models.base import scaled_dot_product_attention
                    out = scaled_dot_product_attention(
                        queries, keys, values, cache=cache, scale=self.scale, mask=mask,
                    )
                    out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
                    return self.o_proj(out)
                return _patched
            _attn_cls.__call__ = _make_patched(_orig_attn_call)
        print(f"  P18 QKV fusion: {len(_patched_attn_classes)} class(es), {len(_qkv_fused)} instances", flush=True)
    except Exception as _e:
        print(f"  P18 QKV fusion skipped: {_e}", flush=True)

    # bfloat16 for MLA models. model_config can be a dict (mlx_lm path) or
    # a dataclass (mlx_vlm path); normalize via getattr/getitem fallback.
    def _cfg_get(cfg, key, default=None):
        if hasattr(cfg, key):
            return getattr(cfg, key)
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return default
    tc = _cfg_get(model_config, "text_config", model_config)
    if _cfg_get(tc, "kv_lora_rank", 0) > 0 or _cfg_get(tc, "model_type", "") == "glm_moe_dsa":
        model.set_dtype(mx.bfloat16)
        print("  bfloat16 enabled", flush=True)

    if not skip_params_eval:
        mx.synchronize()
    print("  Hydration complete", flush=True)

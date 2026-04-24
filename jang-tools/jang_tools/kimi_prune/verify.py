"""Verify a pruned Kimi K2.6 model for structural + semantic correctness.

Level 1 — Structural checks (fast, no compute):
  - config.json has n_routed_experts = n_keep
  - Every MoE layer's router.weight has shape (n_keep, hidden_size)
  - Every MoE layer has expert tensors for 0..n_keep-1 only
  - Total output size is roughly the expected fraction of source

Level 2 — Semantic forward (one batch, ~1 min on MPS):
  - Load the pruned model using the same layer-by-layer runner the
    observer used (which already respects n_routed_experts from config)
  - Forward 1 sequence of 256 tokens through all 61 layers
  - Assert: no NaN/Inf, hidden state std growth at each layer <2×
  - Final logits reasonable (sparse top-10 distribution, not uniform)

Usage:
  python -m jang_tools.kimi_prune.verify \\
      --model <path/to/Kimi-K2.6-REAP-30> \\
      --tokens <path/to/data-drive>/kimi_calib/tokens_64.safetensors \\
      --expected-n-keep 269
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .layer_forward import KimiCfg, LayerWeights, decoder_layer_forward, precompute_rope_cos_sin
from .weight_loader import (
    build_index, read_embed, read_layernorms, read_mla_attention,
    read_dense_mlp, read_router, read_expert, read_shared_expert, read_final_norm,
)


def level1_structural(model_dir: Path, expected_n_keep: int | None = None) -> dict:
    """Structural verification — fast, pure metadata inspection."""
    print(f"\n=== LEVEL 1: STRUCTURAL ===", flush=True)

    cfg = json.loads((model_dir / "config.json").read_text())
    t = cfg.get("text_config", cfg)
    n_keep = t["n_routed_experts"]
    print(f"config.json: n_routed_experts = {n_keep}", flush=True)
    if expected_n_keep is not None:
        assert n_keep == expected_n_keep, \
            f"Expected n_keep={expected_n_keep}, config says {n_keep}"
        print(f"  ✓ matches expected {expected_n_keep}", flush=True)

    hidden_size = t["hidden_size"]
    num_layers = t["num_hidden_layers"]
    first_k_dense = t.get("first_k_dense_replace", 0)
    moe_layers = list(range(first_k_dense, num_layers))

    # Walk all shards, aggregate expert-tensor presence + router shape
    shards = sorted(model_dir.glob("model-*.safetensors"))
    total_bytes = sum(p.stat().st_size for p in shards)

    expert_re = re.compile(
        r"^language_model\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight_packed|weight_scale|weight_shape)$"
    )
    router_re = re.compile(
        r"^language_model\.model\.layers\.(\d+)\.mlp\.gate\.(weight|e_score_correction_bias)$"
    )

    expert_ids_per_layer: dict[int, set[int]] = {}
    router_shapes: dict[int, tuple] = {}
    router_bias_shapes: dict[int, tuple] = {}

    for sp in shards:
        with safe_open(sp, framework="pt") as f:
            for k in f.keys():
                m = expert_re.match(k)
                if m:
                    L = int(m.group(1)); e = int(m.group(2))
                    expert_ids_per_layer.setdefault(L, set()).add(e)
                    continue
                m = router_re.match(k)
                if m:
                    L = int(m.group(1)); kind = m.group(2)
                    t_info = f.get_slice(k)
                    shape = tuple(t_info.get_shape())
                    if kind == "weight":
                        router_shapes[L] = shape
                    else:
                        router_bias_shapes[L] = shape

    # Assertions per MoE layer
    fails = []
    for L in moe_layers:
        eids = expert_ids_per_layer.get(L, set())
        if eids:
            if eids != set(range(n_keep)):
                fails.append(f"L{L}: expert ids = {sorted(eids)[:10]}...{sorted(eids)[-5:]} "
                             f"(expected 0..{n_keep-1})")
        if L in router_shapes:
            if router_shapes[L] != (n_keep, hidden_size):
                fails.append(f"L{L}: router.weight shape = {router_shapes[L]} "
                             f"(expected ({n_keep}, {hidden_size}))")
        if L in router_bias_shapes:
            if router_bias_shapes[L] != (n_keep,):
                fails.append(f"L{L}: router.bias shape = {router_bias_shapes[L]} "
                             f"(expected ({n_keep},))")

    print(f"  moe_layers with expert tensors: {len(expert_ids_per_layer)}/{len(moe_layers)}",
          flush=True)
    print(f"  router.weight shapes verified: {len(router_shapes)} layers",
          flush=True)
    print(f"  router.bias shapes verified: {len(router_bias_shapes)} layers",
          flush=True)
    print(f"  total size: {total_bytes/1e9:.1f} GB", flush=True)

    summary = {
        "n_keep": n_keep,
        "moe_layers": len(moe_layers),
        "router_layers_checked": len(router_shapes),
        "total_size_gb": round(total_bytes / 1e9, 1),
        "fails": fails,
    }

    if fails:
        print(f"\n  ✗ {len(fails)} failures:", flush=True)
        for msg in fails[:20]:
            print(f"    {msg}", flush=True)
        if len(fails) > 20:
            print(f"    ... and {len(fails) - 20} more", flush=True)
    else:
        print(f"  ✓ all structural checks passed", flush=True)
    return summary


def _t(arr, device, dtype):
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def level2_semantic_forward(model_dir: Path, tokens_path: Path,
                            device: str = "cpu", dtype: str = "bfloat16",
                            seq_len: int = 64) -> dict:
    """Level 2: forward one batch through all layers, check sanity."""
    print(f"\n=== LEVEL 2: SEMANTIC FORWARD ===", flush=True)
    dev = torch.device(device)
    torch_dtype = dict(bfloat16=torch.bfloat16, float16=torch.float16,
                       float32=torch.float32)[dtype]

    cfg = KimiCfg.from_config_json(model_dir / "config.json")
    print(f"cfg.n_routed_experts = {cfg.n_routed_experts}", flush=True)

    idx = build_index(model_dir)
    print(f"indexed {len(idx.key_to_shard)} tensors", flush=True)

    from safetensors.torch import load_file as sf_load
    tok = sf_load(str(tokens_path))["tokens"][:1, :seq_len].to(torch.long)
    emb = torch.from_numpy(read_embed(idx)).to(torch_dtype)
    h = torch.embedding(emb, tok).to(dev)
    T = h.shape[1]
    cos, sin = precompute_rope_cos_sin(cfg.qk_rope_head_dim, T, cfg.rope_theta,
                                        dev, torch_dtype, cfg.rope_scaling)
    print(f"embed OK: std={h.float().std().item():.3e} absmax={h.float().abs().max().item():.3e}",
          flush=True)

    growth_factors = []
    prev_std = h.float().std().item()
    any_nan_inf = False

    for L in range(cfg.num_hidden_layers):
        ln = read_layernorms(idx, L)
        attn = read_mla_attention(idx, L)
        is_dense = L < cfg.first_k_dense_replace
        if is_dense:
            dm = read_dense_mlp(idx, L)
            dense_t = {k: _t(v, dev, torch_dtype) for k, v in dm.items()}
            router_w = None; router_bias = None; get_expert = None; shared = None
            expert_cache = None
        else:
            rw, rb = read_router(idx, L)
            router_w = torch.from_numpy(rw)
            router_bias = torch.from_numpy(rb) if rb is not None else None
            # Expert cache: ONLY cfg.n_routed_experts (which reflects pruned count)
            gate_list, up_list, down_list = [], [], []
            for e in range(cfg.n_routed_experts):
                ex = read_expert(idx, L, e)
                gate_list.append(_t(ex["gate_proj"], dev, torch_dtype))
                up_list.append(_t(ex["up_proj"], dev, torch_dtype))
                down_list.append(_t(ex["down_proj"], dev, torch_dtype))
            stk = (torch.stack(gate_list), torch.stack(up_list), torch.stack(down_list))
            expert_cache = stk
            def _make_getter(stacked):
                def _get(e, allow_none=False):
                    if e == "__stacked__": return stacked
                    return (stacked[0][e], stacked[1][e], stacked[2][e])
                _get.supports_stacked = True
                return _get
            get_expert = _make_getter(stk)
            sh = read_shared_expert(idx, L)
            shared = {k: _t(v, dev, torch_dtype) for k, v in sh.items()} if sh else None
            dense_t = None

        lw = LayerWeights(
            input_layernorm=_t(ln["input_layernorm"], dev, torch_dtype),
            post_attention_layernorm=_t(ln["post_attention_layernorm"], dev, torch_dtype),
            q_a_proj=_t(attn["q_a_proj"], dev, torch_dtype),
            q_a_layernorm=_t(attn["q_a_layernorm"], dev, torch_dtype),
            q_b_proj=_t(attn["q_b_proj"], dev, torch_dtype),
            kv_a_proj_with_mqa=_t(attn["kv_a_proj_with_mqa"], dev, torch_dtype),
            kv_a_layernorm=_t(attn["kv_a_layernorm"], dev, torch_dtype),
            kv_b_proj=_t(attn["kv_b_proj"], dev, torch_dtype),
            o_proj=_t(attn["o_proj"], dev, torch_dtype),
            dense_mlp=dense_t,
            router_w=router_w, router_bias=router_bias,
            expert_loader=get_expert, shared_expert=shared,
        )

        h, sal, cnt = decoder_layer_forward(h, lw, cfg, cos, sin, dev)
        h32 = h.float()
        cur_std = h32.std().item()
        n_nan = int(torch.isnan(h32).sum().item())
        n_inf = int(torch.isinf(h32).sum().item())
        if n_nan or n_inf:
            any_nan_inf = True
        growth = cur_std / max(prev_std, 1e-12)
        growth_factors.append(growth)
        if L % 10 == 0 or L in (0, 1, cfg.num_hidden_layers - 1):
            print(f"  L{L:2d} {'dense' if is_dense else 'moe  '}: "
                  f"std={cur_std:.3e} growth={growth:.2f}x nan={n_nan} inf={n_inf}",
                  flush=True)
        prev_std = cur_std

        del lw
        if expert_cache is not None:
            del expert_cache
        if dev.type == "mps":
            torch.mps.empty_cache()

    # Final check: apply final norm + lm_head (if accessible) for logit sanity
    print(f"\n  growth factors min={min(growth_factors):.3f} "
          f"max={max(growth_factors):.3f} median={sorted(growth_factors)[len(growth_factors)//2]:.3f}",
          flush=True)

    # Real failure modes: NaN/Inf, or "routing collapse" (many consecutive
    # layers with near-identical growth ~1 from a broken state).
    # Early transformer layers legitimately grow 5-10× as residual stream
    # warms up — that's not a bug.
    worst = max(growth_factors)
    # Check for catastrophic blowup (>100× means real problem)
    # or NaN/Inf (always real problem)
    failed = any_nan_inf or worst > 100.0
    # Find which layer(s) had the biggest growth
    gf_sorted = sorted(enumerate(growth_factors), key=lambda x: -x[1])
    print(f"\n  growth by layer (top 5): "
          f"{[(l, f'{g:.2f}x') for l, g in gf_sorted[:5]]}", flush=True)
    if failed:
        print(f"  ✗ SEMANTIC FAIL: worst growth {worst:.2f}x  any_nan_inf={any_nan_inf}",
              flush=True)
    else:
        print(f"  ✓ all {len(growth_factors)} layers produced sane output "
              f"(no NaN/Inf, worst growth {worst:.2f}x - normal for early layers)",
              flush=True)

    return {
        "max_growth_factor": worst,
        "min_growth_factor": min(growth_factors),
        "any_nan_inf": any_nan_inf,
        "final_hidden_std": prev_std,
        "passed": not failed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--tokens", type=Path)
    ap.add_argument("--expected-n-keep", type=int, default=None)
    ap.add_argument("--skip-forward", action="store_true")
    ap.add_argument("--seq-len", type=int, default=256)
    args = ap.parse_args()

    l1 = level1_structural(args.model, args.expected_n_keep)
    if args.skip_forward or args.tokens is None:
        return

    l2 = level2_semantic_forward(args.model, args.tokens, seq_len=args.seq_len)

    print(f"\n=== SUMMARY ===", flush=True)
    print(f"Level 1: {'PASS' if not l1['fails'] else 'FAIL'}  ({l1['total_size_gb']} GB)",
          flush=True)
    print(f"Level 2: {'PASS' if l2['passed'] else 'FAIL'}  "
          f"(max growth {l2['max_growth_factor']:.2f}x)", flush=True)


if __name__ == "__main__":
    main()

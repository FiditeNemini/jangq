"""JANGReap: Kimi K2.6 expert pruning via REAP saliency.

Layer-by-layer observer that streams Kimi K2.6 weights off disk, runs
one DSV3 decoder layer at a time on MPS (Apple Silicon GPU), and
accumulates the REAP saliency score per (layer, expert):

    S[L, e] = mean over x ∈ X_e of ( g_e(x) · ||f_e(x)||_2 )

where X_e = {tokens selecting expert e in top-k}, g_e is the
post-top-k-renorm router gate, ||f_e||_2 is the L2 norm of the raw
expert output.

Outputs:
  saliency_sum.safetensors   (L, E) f32  — Σ g·||f||
  counts.safetensors         (L, E) i64  — |X_e|
  prune_plan_30.json, prune_plan_50.json   (keep/drop per layer)

Peak memory: ~20-30 GB (one layer's weights + current activation cache).

Usage:
  python -m jang_tools.kimi_prune.jangreap \\
      --model <path/to/sources>/Kimi_K2_6_FP8 \\
      --tokens <path/to/data-drive>/kimi_calib/tokens.safetensors \\
      --out-dir <path/to/data-drive>/kimi_calib/jangreap_run/ \\
      --ratios 0.30 0.50 \\
      --device mps
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as sf_load, save_file as sf_save

from .layer_forward import (
    KimiCfg, LayerWeights, decoder_layer_forward,
    precompute_rope_cos_sin, rms_norm, swiglu_mlp,
)
from .weight_loader import (
    ShardIndex, build_index, read_embed, read_final_norm,
    read_layernorms, read_mla_attention, read_dense_mlp,
    read_router, read_expert, read_shared_expert,
)


def _to_t(arr: np.ndarray, device, dtype=torch.bfloat16) -> torch.Tensor:
    """numpy float32 -> torch tensor on device with cast."""
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def embed_tokens(tokens: torch.Tensor, embed_w: torch.Tensor) -> torch.Tensor:
    """tokens: (B, T) int32, embed_w: (V, H) bf16. Returns (B, T, H) bf16."""
    return torch.embedding(embed_w, tokens.long())


def _make_expert_loader(idx: ShardIndex, layer: int, device: torch.device,
                        dtype=torch.bfloat16, n_experts: int = 384):
    """Return a stacked-tensor expert loader.

    Loads all `n_experts` experts for layer `layer` into three stacked
    torch tensors on `device`:
      gate_stk: (E, Im, H)
      up_stk:   (E, Im, H)
      down_stk: (E, H, Im)

    The returned callable has `.supports_stacked = True` and responds to
    `loader("__stacked__", allow_none=True)` with the tuple. Also supports
    `loader(e)` for per-expert access (for compat).
    """
    from .weight_loader import read_expert
    gate_list: list[torch.Tensor] = []
    up_list: list[torch.Tensor] = []
    down_list: list[torch.Tensor] = []
    t0 = time.time()
    for e in range(n_experts):
        exp = read_expert(idx, layer, e)
        gate_list.append(_to_t(exp["gate_proj"], device, dtype))
        up_list.append(_to_t(exp["up_proj"], device, dtype))
        down_list.append(_to_t(exp["down_proj"], device, dtype))
        if (e + 1) % 64 == 0:
            print(f"    loaded {e+1}/{n_experts} experts  ({time.time()-t0:.1f}s)",
                  flush=True)
    gate_stk = torch.stack(gate_list, dim=0)
    up_stk = torch.stack(up_list, dim=0)
    down_stk = torch.stack(down_list, dim=0)
    # Free python list refs so only the stacked tensors hold the memory
    del gate_list, up_list, down_list

    cache = (gate_stk, up_stk, down_stk)

    def _get(e, allow_none: bool = False):
        if e == "__stacked__":
            return cache
        return (cache[0][e], cache[1][e], cache[2][e])

    _get.supports_stacked = True  # type: ignore[attr-defined]
    return _get, cache


def run_observer(
    model_dir: Path,
    tokens_path: Path,
    out_dir: Path,
    device: str = "mps",
    dtype: str = "bfloat16",
    batch_size: int = 1,
):
    """Layer-by-layer forward that accumulates REAP saliency on MoE layers."""
    device = torch.device(device)
    torch_dtype = dict(bfloat16=torch.bfloat16, float16=torch.float16,
                       float32=torch.float32)[dtype]

    cfg = KimiCfg.from_config_json(model_dir / "config.json")
    print(f"[jangreap] cfg: {cfg.num_hidden_layers} layers, {cfg.n_routed_experts} "
          f"experts/layer, topk={cfg.num_experts_per_tok}, "
          f"H={cfg.hidden_size}, heads={cfg.num_attention_heads}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[jangreap] indexing shards in {model_dir}", flush=True)
    t0 = time.time()
    idx = build_index(model_dir)
    print(f"  {len(idx.key_to_shard):,} tensors in {time.time()-t0:.1f}s", flush=True)

    print(f"[jangreap] loading tokens from {tokens_path}", flush=True)
    td = sf_load(str(tokens_path))
    tokens = td["tokens"]  # (N, T) int32
    N, T = tokens.shape
    print(f"  N={N} seqs × T={T} = {N*T:,} tokens", flush=True)

    # Precompute RoPE cos/sin (shared across all layers — same base + T).
    cos, sin = precompute_rope_cos_sin(
        cfg.qk_rope_head_dim, T, cfg.rope_theta,
        device=device, dtype=torch_dtype, scaling=cfg.rope_scaling,
    )
    print(f"  RoPE cos/sin: {cos.shape}  dtype={cos.dtype}", flush=True)

    # Load embed table → initial hidden states (stored on CPU, streamed per batch)
    print("[jangreap] loading embed_tokens...", flush=True)
    t0 = time.time()
    embed_w_np = read_embed(idx)
    print(f"  embed: {embed_w_np.shape}  {time.time()-t0:.1f}s", flush=True)
    embed_w = torch.from_numpy(embed_w_np).to(dtype=torch_dtype)

    # Hidden-state cache on disk: emitted one batch at a time.
    hs_dir = out_dir / "hs_cache"
    hs_dir.mkdir(exist_ok=True)

    def _emit_initial_hs():
        """Embed lookup for each batch; write (N, T, H) to disk in chunks."""
        t0 = time.time()
        for i in range(0, N, batch_size):
            b = tokens[i:i+batch_size].to(torch.long)
            hs = torch.embedding(embed_w, b).to(torch_dtype)
            sf_save({"h": hs}, str(hs_dir / f"L0_in_{i:04d}.safetensors"))
        print(f"  embed → disk in {time.time()-t0:.1f}s  "
              f"({N} batches × {batch_size} seqs)", flush=True)

    _emit_initial_hs()

    # Accumulators
    saliency_sum = torch.zeros(cfg.num_hidden_layers, cfg.n_routed_experts,
                               dtype=torch.float32, device=device)
    counts = torch.zeros(cfg.num_hidden_layers, cfg.n_routed_experts,
                         dtype=torch.int64, device=device)

    # Layer-by-layer forward
    for L in range(cfg.num_hidden_layers):
        t_layer = time.time()
        print(f"[jangreap] === layer {L} ===", flush=True)

        # Load layer L weights
        t0 = time.time()
        ln = read_layernorms(idx, L)
        attn = read_mla_attention(idx, L)
        lw_attn = {
            "input_layernorm": _to_t(ln["input_layernorm"], device, torch_dtype),
            "post_attention_layernorm": _to_t(ln["post_attention_layernorm"], device, torch_dtype),
            "q_a_proj": _to_t(attn["q_a_proj"], device, torch_dtype),
            "q_a_layernorm": _to_t(attn["q_a_layernorm"], device, torch_dtype),
            "q_b_proj": _to_t(attn["q_b_proj"], device, torch_dtype),
            "kv_a_proj_with_mqa": _to_t(attn["kv_a_proj_with_mqa"], device, torch_dtype),
            "kv_a_layernorm": _to_t(attn["kv_a_layernorm"], device, torch_dtype),
            "kv_b_proj": _to_t(attn["kv_b_proj"], device, torch_dtype),
            "o_proj": _to_t(attn["o_proj"], device, torch_dtype),
        }
        print(f"  attention weights loaded in {time.time()-t0:.1f}s", flush=True)

        is_dense = L < cfg.first_k_dense_replace
        dense_mlp_t = None
        router_w = None; router_bias = None
        shared_mlp_t = None
        get_expert = None
        expert_cache = None
        if is_dense:
            dm = read_dense_mlp(idx, L)
            dense_mlp_t = {k: _to_t(v, device, torch_dtype) for k, v in dm.items()}
            print(f"  dense MLP loaded", flush=True)
        else:
            rw, rb = read_router(idx, L)
            router_w = torch.from_numpy(rw)  # (E, H) f32
            router_bias = torch.from_numpy(rb) if rb is not None else None
            shared = read_shared_expert(idx, L)
            if shared is not None:
                shared_mlp_t = {k: _to_t(v, device, torch_dtype) for k, v in shared.items()}
            # Experts loaded eagerly + stacked for fast bmm-friendly indexing.
            get_expert, expert_cache = _make_expert_loader(
                idx, L, device, torch_dtype, n_experts=cfg.n_routed_experts,
            )
            print(f"  MoE router + {cfg.n_routed_experts} experts stacked in "
                  f"{time.time()-t0:.1f}s", flush=True)

        lw = LayerWeights(
            input_layernorm=lw_attn["input_layernorm"],
            post_attention_layernorm=lw_attn["post_attention_layernorm"],
            q_a_proj=lw_attn["q_a_proj"],
            q_a_layernorm=lw_attn["q_a_layernorm"],
            q_b_proj=lw_attn["q_b_proj"],
            kv_a_proj_with_mqa=lw_attn["kv_a_proj_with_mqa"],
            kv_a_layernorm=lw_attn["kv_a_layernorm"],
            kv_b_proj=lw_attn["kv_b_proj"],
            o_proj=lw_attn["o_proj"],
            dense_mlp=dense_mlp_t,
            router_w=router_w,
            router_bias=router_bias,
            expert_loader=get_expert,
            shared_expert=shared_mlp_t,
        )

        # Forward each cached HS batch
        t_fwd = time.time()
        for i in range(0, N, batch_size):
            hs_path = hs_dir / f"L{L}_in_{i:04d}.safetensors"
            hs_next_path = hs_dir / f"L{L+1}_in_{i:04d}.safetensors"
            x = sf_load(str(hs_path))["h"].to(device=device, dtype=torch_dtype)
            out, sal, cnt = decoder_layer_forward(x, lw, cfg, cos, sin, device)
            if sal is not None:
                saliency_sum[L] += sal
                counts[L] += cnt
            sf_save({"h": out.cpu()}, str(hs_next_path))
            # Free per-batch
            del x, out
        print(f"  forward {N} batches in {time.time()-t_fwd:.1f}s", flush=True)

        # Clean up L-1's cache files (not needed anymore).
        if L > 0:
            for i in range(0, N, batch_size):
                p = hs_dir / f"L{L}_in_{i:04d}.safetensors"
                if p.exists():
                    p.unlink()

        # Free layer weights
        del lw, lw_attn, dense_mlp_t, shared_mlp_t, expert_cache, get_expert
        if device.type == "mps":
            torch.mps.empty_cache()
        print(f"  layer {L} total: {time.time()-t_layer:.1f}s", flush=True)

        # Snapshot saliency after each layer
        sf_save({
            "saliency_sum": saliency_sum.cpu(),
            "counts": counts.cpu(),
        }, str(out_dir / "saliency_running.safetensors"))

    # Final
    sf_save({
        "saliency_sum": saliency_sum.cpu(),
        "counts": counts.cpu(),
    }, str(out_dir / "saliency.safetensors"))
    print(f"[jangreap] observer DONE — saved saliency to {out_dir}", flush=True)


def score_and_plan(out_dir: Path, cfg_path: Path, ratios: list[float]):
    """Compute REAP scores and write per-ratio prune plans."""
    cfg = KimiCfg.from_config_json(cfg_path)
    d = sf_load(str(out_dir / "saliency.safetensors"))
    sal = d["saliency_sum"].to(torch.float64)  # (L, E)
    cnt = d["counts"].to(torch.float64)
    S = sal / (cnt + 1e-12)  # (L, E)

    L, E = S.shape
    print(f"[score] saliency shape {S.shape}", flush=True)

    for ratio in ratios:
        n_drop = int(round(E * ratio))
        n_keep = E - n_drop
        plan = {
            "base_ratio": ratio,
            "n_layers": int(L),
            "n_experts_per_layer": int(E),
            "n_keep_per_layer": int(n_keep),
            "n_drop_per_layer": int(n_drop),
            "global_kept_fraction": 1.0 - ratio,
            "per_layer": [],
        }
        for Li in range(L):
            # Skip dense layers (saliency is zero there because no MoE forward happened).
            if Li < cfg.first_k_dense_replace:
                continue
            s = S[Li]
            # Handle experts that were never selected (count=0): treat as 0 saliency.
            order = torch.argsort(s, descending=True)
            keep = sorted(order[:n_keep].tolist())
            drop = sorted(order[n_keep:].tolist())
            plan["per_layer"].append({
                "layer": int(Li),
                "n_keep": int(n_keep),
                "n_drop": int(n_drop),
                "keep": keep,
                "drop": drop,
                "score_kept_mean": float(s[keep].mean()),
                "score_dropped_mean": float(s[drop].mean()),
            })
        out_path = out_dir / f"prune_plan_{int(ratio*100)}.json"
        with out_path.open("w") as f:
            json.dump(plan, f, indent=2)
        print(f"  wrote {out_path}  ({len(plan['per_layer'])} MoE layers)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--tokens", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--device", default="mps",
                    choices=("mps", "cpu", "cuda"))
    ap.add_argument("--dtype", default="bfloat16",
                    choices=("bfloat16", "float16", "float32"))
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--ratios", type=float, nargs="*", default=[0.30, 0.50])
    ap.add_argument("--skip-observer", action="store_true",
                    help="Reuse existing saliency.safetensors from a prior run.")
    args = ap.parse_args()

    if not args.skip_observer:
        run_observer(
            args.model, args.tokens, args.out_dir,
            device=args.device, dtype=args.dtype, batch_size=args.batch_size,
        )
    score_and_plan(args.out_dir, args.model / "config.json", args.ratios)


if __name__ == "__main__":
    main()

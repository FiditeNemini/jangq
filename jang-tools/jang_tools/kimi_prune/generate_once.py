"""Single-shot prefill + top-k logit inspector for a pruned Kimi K2.6 model.

Uses the layer-by-layer forward from jangreap.py to push a prompt through
all 61 layers, applies final norm + lm_head, and prints the top-10 predicted
next tokens with probabilities.

Coherence test: if the pruned model returns sensible continuations for
factual prompts, pruning preserved semantic behavior.

Usage:
  python -m jang_tools.kimi_prune.generate_once \\
      --model <path/to/Kimi-K2.6-REAP-30> \\
      --tokenizer <path/to/sources>/Kimi_K2_6_FP8 \\
      --prompt "The capital of France is" \\
      --device cpu --top-k 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .layer_forward import (
    KimiCfg, LayerWeights, decoder_layer_forward, precompute_rope_cos_sin,
    rms_norm,
)
from .weight_loader import (
    build_index, read_embed, read_layernorms, read_mla_attention,
    read_dense_mlp, read_router, read_expert, read_shared_expert,
    read_final_norm,
)


def _t(arr, device, dtype):
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def prefill_once(model_dir: Path, tokenizer_path: Path, prompt: str,
                 device: str = "cpu", dtype: str = "bfloat16",
                 top_k: int = 10, max_tokens: int = 256):
    dev = torch.device(device)
    torch_dtype = dict(bfloat16=torch.bfloat16, float16=torch.float16,
                       float32=torch.float32)[dtype]

    print(f"[generate] model={model_dir}", flush=True)
    cfg = KimiCfg.from_config_json(model_dir / "config.json")
    print(f"  n_routed_experts = {cfg.n_routed_experts}", flush=True)

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    ids = tok.encode(prompt, add_special_tokens=True)
    ids = ids[:max_tokens]
    print(f"  prompt={prompt!r}  -> {len(ids)} tokens", flush=True)
    print(f"  tokens: {tok.convert_ids_to_tokens(ids)[:20]}", flush=True)

    idx = build_index(model_dir)
    print(f"  indexed {len(idx.key_to_shard)} tensors", flush=True)

    # Embed
    t0 = time.time()
    emb = torch.from_numpy(read_embed(idx)).to(torch_dtype)
    x = torch.tensor([ids], dtype=torch.long)
    h = torch.embedding(emb, x).to(dev)
    T = h.shape[1]
    print(f"  embed: {h.shape}  ({time.time()-t0:.1f}s)", flush=True)

    cos, sin = precompute_rope_cos_sin(
        cfg.qk_rope_head_dim, T, cfg.rope_theta,
        device=dev, dtype=torch_dtype, scaling=cfg.rope_scaling,
    )

    # Forward through all layers
    t_fwd = time.time()
    for L in range(cfg.num_hidden_layers):
        t0 = time.time()
        ln = read_layernorms(idx, L)
        attn = read_mla_attention(idx, L)
        is_dense = L < cfg.first_k_dense_replace
        dense_t = None; router_w = None; router_bias = None
        get_expert = None; shared = None
        if is_dense:
            dm = read_dense_mlp(idx, L)
            dense_t = {k: _t(v, dev, torch_dtype) for k, v in dm.items()}
        else:
            rw, rb = read_router(idx, L)
            router_w = torch.from_numpy(rw)
            router_bias = torch.from_numpy(rb) if rb is not None else None
            gate_list, up_list, down_list = [], [], []
            for e in range(cfg.n_routed_experts):
                ex = read_expert(idx, L, e)
                gate_list.append(_t(ex["gate_proj"], dev, torch_dtype))
                up_list.append(_t(ex["up_proj"], dev, torch_dtype))
                down_list.append(_t(ex["down_proj"], dev, torch_dtype))
            stk = (torch.stack(gate_list), torch.stack(up_list), torch.stack(down_list))
            def _make_get(stacked):
                def _get(e, allow_none=False):
                    if e == "__stacked__": return stacked
                    return (stacked[0][e], stacked[1][e], stacked[2][e])
                _get.supports_stacked = True
                return _get
            get_expert = _make_get(stk)
            sh = read_shared_expert(idx, L)
            shared = {k: _t(v, dev, torch_dtype) for k, v in sh.items()} if sh else None

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
            dense_mlp=dense_t, router_w=router_w, router_bias=router_bias,
            expert_loader=get_expert, shared_expert=shared,
        )
        h, _, _ = decoder_layer_forward(h, lw, cfg, cos, sin, dev)
        print(f"  L{L:2d}: h std={h.float().std().item():.3e}  "
              f"{time.time()-t0:.0f}s", flush=True)
        del lw
        if not is_dense:
            del stk
        if dev.type == "mps":
            torch.mps.empty_cache()

    # Final RMS norm + lm_head
    print(f"\n  total forward: {time.time()-t_fwd:.0f}s", flush=True)
    final_norm_w = _t(read_final_norm(idx), dev, torch_dtype)
    h = rms_norm(h, final_norm_w, cfg.rms_norm_eps)

    # lm_head: load last logit projection. For Kimi K2.6 check if tied embeddings
    lm_head_key = "language_model.lm_head"
    if lm_head_key + ".weight_packed" in idx.key_to_shard:
        from .weight_loader import read_int4_weight
        lm_head_w = _t(read_int4_weight(idx, lm_head_key), dev, torch_dtype)
    elif lm_head_key + ".weight" in idx.key_to_shard:
        lm_head_w = _t(read_final_norm.__globals__["read_tensor"](idx, lm_head_key + ".weight"),
                       dev, torch_dtype)
    else:
        # Tied — use embed
        lm_head_w = emb.to(dev, torch_dtype)
    print(f"  lm_head: {lm_head_w.shape}", flush=True)

    # Logits at LAST position
    last = h[0, -1]  # (H,)
    logits = last @ lm_head_w.T.to(last.dtype)  # (V,)
    logits_f = logits.float()
    probs = F.softmax(logits_f, dim=-1)
    top_p, top_i = probs.topk(top_k)

    print(f"\n=== TOP-{top_k} PREDICTED NEXT TOKENS for {prompt!r} ===", flush=True)
    for rank, (p, i) in enumerate(zip(top_p.tolist(), top_i.tolist())):
        tok_str = tok.decode([i])
        print(f"  {rank+1:>2}. {i:>7}  {p:.4f}  {tok_str!r}", flush=True)

    # Decode first argmax for next-token
    pred = tok.decode([int(top_i[0].item())])
    print(f"\n>> next predicted continuation: {pred!r}", flush=True)
    print(f">> argmax-decoded 1 token: {prompt!r} + {pred!r}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--tokenizer", required=True, type=Path,
                    help="Tokenizer path (can be unpruned model — tokenizer is the same)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    prefill_once(args.model, args.tokenizer, args.prompt,
                 device=args.device, dtype=args.dtype,
                 top_k=args.top_k, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()

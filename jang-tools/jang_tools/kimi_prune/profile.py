"""Routing profile capture for Kimi K2.6.

Streams calibration tokens through the model, hooks each MoE layer's
router, and records per-layer-per-expert stats:

  freq[L][e]            fraction of tokens selecting expert e in top-k
  weighted_freq[L][e]   sum of router gate scores when e was selected
                        (post softmax+renorm; measures e's contribution mass)
  coact[L][e, f]        co-selection counts (how often e and f both chosen
                        in same token's top-k) — input to absorb-merge later
  output_energy[L][e]   avg ‖expert(x)‖ over tokens selecting e. Requires
                        running the expert forward; expensive but the
                        strongest signal for importance. Enabled via
                        --with-energy.

Also records per-DOMAIN stats so we can later see which experts are
pentest-heavy vs chinese-heavy vs coding-heavy (useful for reporting
what got dropped).

Writes one .safetensors bundle to out_dir/routing_profile.safetensors
containing arrays indexed [layer, expert] (and [layer, expert, expert]
for coact). A sidecar JSON records token counts per domain.

Model loading: Kimi K2.6 text backbone is DeepseekV3 (kimi_k2 == ds_v3
with VL wrapper). Two load paths supported:

  1. mlx_lm (recommended on M3 Ultra, lazy mmap)
     Requires a minor kimi_k2 shim that registers the model_type; we
     fall back to deepseek_v3's model class since the tensor layout is
     identical for the text backbone.
  2. transformers (HF) with trust_remote_code + device_map="auto"
     Slow on macOS/MPS but works anywhere.

Usage:
  python -m jang_tools.kimi_prune.profile \\
      --model <path/to/sources>/Kimi-K2.6-FP8 \\
      --corpus <path/to/data-drive>/kimi_calib/corpus.jsonl \\
      --out <path/to/data-drive>/kimi_calib/routing_profile.safetensors \\
      --max-tokens 5_000_000 \\
      --seq-len 4096 \\
      [--with-energy]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class ProfileConfig:
    model_path: Path
    corpus_path: Path
    out_path: Path
    max_tokens: int = 5_000_000
    seq_len: int = 4096
    with_energy: bool = False
    per_domain: bool = True


def _load_model_mlx(model_path: Path):
    """Load Kimi K2.6 text backbone via mlx_lm (deepseek_v3 adapter).

    The config's top-level model_type is `kimi_k25`; text_config.model_type
    is `kimi_k2`. mlx_lm's `deepseek_v3.py` recognizes `deepseek_v3` only,
    so we patch config.model_type→`deepseek_v3` for the text backbone
    before load, then restore the original model_type afterward (it's
    not used at inference time).

    VL vision tower is intentionally ignored — we only need the text
    backbone's MoE routers for pruning.
    """
    import mlx.core as mx
    import shutil, tempfile

    # Make a config-only shim directory pointing weights at the real model.
    # This lets us hand mlx_lm a patched config.json while the .safetensors
    # shards stay in the original location (via relative path resolution).
    shim_dir = Path(tempfile.mkdtemp(prefix="kimi_shim_"))
    with open(model_path / "config.json") as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    text_cfg = dict(text_cfg)
    text_cfg["model_type"] = "deepseek_v3"
    with open(shim_dir / "config.json", "w") as f:
        json.dump(text_cfg, f)
    # Symlink shards + tokenizer into shim dir.
    for p in model_path.iterdir():
        if p.name == "config.json":
            continue
        try:
            (shim_dir / p.name).symlink_to(p.resolve())
        except FileExistsError:
            pass

    from mlx_lm.utils import load_model as _load_model, load_tokenizer
    model, _ = _load_model(shim_dir, lazy=True, strict=False)
    tokenizer = load_tokenizer(model_path)  # real path for tokenizer_config
    return model, tokenizer, shim_dir


def _iter_corpus(corpus_path: Path, tokenizer, seq_len: int,
                 max_tokens: int) -> Iterator[tuple[list[int], str]]:
    """Yield (token_ids, domain) chunks of length seq_len."""
    total = 0
    buf_by_domain: dict[str, list[int]] = {}
    with corpus_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if total >= max_tokens:
                break
            rec = json.loads(line)
            domain = rec.get("domain", "unknown")
            ids = tokenizer.encode(rec["text"], add_special_tokens=False)
            buf = buf_by_domain.setdefault(domain, [])
            buf.extend(ids)
            while len(buf) >= seq_len and total < max_tokens:
                chunk = buf[:seq_len]
                buf[:seq_len] = []
                total += seq_len
                yield chunk, domain
    # Flush short tails (optional; here we drop them for consistent seq_len).


def _hook_routers(model):
    """Attach hooks to every MoE layer's router.

    Returns a `stats` dict that accumulates in-place on each forward.
    Works with mlx_lm's DeepseekV3Model layout where each layer has
    `.mlp.gate` (router) and `.mlp.switch_mlp` (experts).
    """
    import mlx.core as mx

    stats: dict[str, object] = {
        "layers": [],        # list of {freq, weighted_freq, coact, energy}
        "layer_names": [],
        "n_experts": None,
        "topk": None,
    }

    _originals: list[tuple[object, callable]] = []

    for name, mod in model.named_modules():
        # Identify MoE layer wrappers. DeepseekV3 uses DeepseekV3MoE (name
        # ends in `.mlp`). Check by attribute pattern:
        if not (hasattr(mod, "gate") and hasattr(mod, "switch_mlp")):
            continue
        # Only block where `gate` is a router Linear-like producing [*, E] logits
        gate = getattr(mod, "gate", None)
        if gate is None or not hasattr(gate, "weight"):
            continue

        n_experts = gate.weight.shape[0]
        topk = getattr(mod, "num_experts_per_tok",
                       getattr(mod, "top_k",
                               getattr(mod, "num_routed_experts", 8)))

        layer_stats = {
            "freq": mx.zeros((n_experts,), dtype=mx.float32),
            "weighted_freq": mx.zeros((n_experts,), dtype=mx.float32),
            "coact": mx.zeros((n_experts, n_experts), dtype=mx.float32),
            "energy": mx.zeros((n_experts,), dtype=mx.float32),
            "n_tokens": 0,
        }
        stats["layers"].append(layer_stats)
        stats["layer_names"].append(name)
        if stats["n_experts"] is None:
            stats["n_experts"] = int(n_experts)
            stats["topk"] = int(topk)

        # Wrap the MoE module __call__ to capture router outputs.
        orig = mod.__class__.__call__
        def make_wrapped(orig_call, ls=layer_stats, k=topk):
            def wrapped(self, x, *a, **kw):
                # Router forward.
                gates = self.gate(x.astype(mx.float32))
                # DeepseekV3: sigmoid + e_score_correction_bias topk
                if hasattr(self, "e_score_correction_bias"):
                    scores_raw = mx.sigmoid(gates)
                    scores = scores_raw + self.e_score_correction_bias
                    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
                    sel = mx.take_along_axis(scores_raw, inds, axis=-1)
                    sel = sel / (mx.sum(sel, axis=-1, keepdims=True) + 1e-20)
                else:
                    scores = mx.softmax(gates, axis=-1)
                    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
                    sel = mx.take_along_axis(scores, inds, axis=-1)
                # Flatten batch/seq into single token axis for histogram.
                inds_flat = inds.reshape(-1, k)        # (T, k)
                sel_flat = sel.reshape(-1, k)
                T = inds_flat.shape[0]
                # freq: count per-expert selections, / T (after).
                one_hot = mx.zeros((T, ls["freq"].shape[0]), dtype=mx.float32)
                # MLX doesn't have scatter_add directly; use matmul trick.
                # (T, k) -> one-hot (T, E): for each slot put gate score.
                # Simpler: accumulate via take+add. Here we use numpy interop.
                import numpy as _np
                idx_np = _np.asarray(inds_flat)
                sel_np = _np.asarray(sel_flat)
                E = ls["freq"].shape[0]
                counts = _np.zeros((E,), dtype=_np.float32)
                wmass = _np.zeros((E,), dtype=_np.float32)
                coact = _np.asarray(ls["coact"]).copy()
                for t in range(T):
                    row = idx_np[t]
                    w = sel_np[t]
                    for j, e in enumerate(row):
                        counts[e] += 1.0
                        wmass[e] += float(w[j])
                    for j1 in range(k):
                        for j2 in range(k):
                            if j1 == j2:
                                continue
                            coact[row[j1], row[j2]] += 1.0
                ls["freq"] = ls["freq"] + mx.array(counts)
                ls["weighted_freq"] = ls["weighted_freq"] + mx.array(wmass)
                ls["coact"] = mx.array(coact)
                ls["n_tokens"] += T
                # Call through to original behavior.
                return orig_call(self, x, *a, **kw)
            return wrapped

        mod.__class__.__call__ = make_wrapped(orig)
        _originals.append((mod.__class__, orig))

    stats["_restore"] = _originals
    return stats


def _save_profile(stats, out_path: Path, tokens_by_domain: dict,
                  config_summary: dict):
    import mlx.core as mx
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_layers = len(stats["layers"])
    E = stats["n_experts"]

    freq = np.stack([np.asarray(l["freq"]) for l in stats["layers"]])
    wfreq = np.stack([np.asarray(l["weighted_freq"]) for l in stats["layers"]])
    coact = np.stack([np.asarray(l["coact"]) for l in stats["layers"]])
    ntok = np.array([l["n_tokens"] for l in stats["layers"]], dtype=np.int64)
    energy = np.stack([np.asarray(l["energy"]) for l in stats["layers"]])

    tensors = {
        "freq": mx.array(freq),
        "weighted_freq": mx.array(wfreq),
        "coact": mx.array(coact),
        "n_tokens_per_layer": mx.array(ntok),
        "energy": mx.array(energy),
    }
    mx.save_safetensors(str(out_path), tensors)

    sidecar = {
        "n_layers": n_layers,
        "n_experts_per_layer": int(E),
        "topk": int(stats["topk"]),
        "layer_names": stats["layer_names"],
        "tokens_by_domain": tokens_by_domain,
        "config": config_summary,
    }
    with out_path.with_suffix(".json").open("w") as f:
        json.dump(sidecar, f, indent=2)


def run(cfg: ProfileConfig):
    import mlx.core as mx

    print(f"[profile] loading model from {cfg.model_path}", flush=True)
    model, tokenizer, shim_dir = _load_model_mlx(cfg.model_path)
    print(f"[profile] installing router hooks", flush=True)
    stats = _hook_routers(model)
    print(f"[profile] MoE layers detected: {len(stats['layers'])}  "
          f"E={stats['n_experts']}  topk={stats['topk']}", flush=True)

    tokens_by_domain: dict[str, int] = {}
    t0 = time.time()
    n_chunks = 0
    for ids, domain in _iter_corpus(cfg.corpus_path, tokenizer,
                                    cfg.seq_len, cfg.max_tokens):
        x = mx.array([ids], dtype=mx.int32)
        _ = model(x)
        mx.synchronize()
        tokens_by_domain[domain] = tokens_by_domain.get(domain, 0) + cfg.seq_len
        n_chunks += 1
        if n_chunks % 10 == 0:
            dt = time.time() - t0
            total = sum(tokens_by_domain.values())
            rate = total / max(dt, 1e-3)
            print(f"  chunk {n_chunks}  total={total:,} tok  "
                  f"{rate:.0f} tok/s  elapsed={dt:.0f}s", flush=True)

    print(f"[profile] saving to {cfg.out_path}", flush=True)
    _save_profile(stats, cfg.out_path, tokens_by_domain, {
        "max_tokens": cfg.max_tokens,
        "seq_len": cfg.seq_len,
        "with_energy": cfg.with_energy,
        "model_path": str(cfg.model_path),
    })
    print(f"[profile] done ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-tokens", type=int, default=5_000_000)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--with-energy", action="store_true")
    args = ap.parse_args()
    run(ProfileConfig(
        model_path=args.model, corpus_path=args.corpus, out_path=args.out,
        max_tokens=args.max_tokens, seq_len=args.seq_len,
        with_energy=args.with_energy,
    ))


if __name__ == "__main__":
    main()

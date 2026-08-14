"""Turn the Qwen3.6-27B calibration capture into a measured bit allocation.

Score per quantizable module:

    s = tr(H) * ||W||_F^2

`tr(H)` comes from the capture (`sum_c E[x_c^2]`, the Hessian diagonal);
`||W||_F^2` is read from the source weights. The product approximates the output
error a unit relative perturbation of that module would cause, so ranking by it
tells us where bits actually buy coherence — rather than inferring it from the
tensor's *name*, which is what `TIER_RULES` does and what has broken on every
new architecture we've touched.

Emits a per-module bit map under a target size budget, plus the evidence
(scores, traces, norms) so the choice is auditable.

    python -m jang_tools.qwen36_allocate <model_dir> <calib.json> <out.json> \
        [--target-gib 10.0] [--base-bits 2] [--group-size 128]
"""
from __future__ import annotations

import glob
import json
import struct
import sys
from pathlib import Path

import numpy as np

# Modules that must never be driven by the score alone.
FORCED = {
    # vision tower: proven to collapse under aggressive quant / AWQ on this
    # family (QWEN36-A3B-JANGTQ4-COHERENCE-BUG). Floor it.
    "vision_min_bits": 4,
    # full attention is the coherence anchor and is only ~6% of params.
    "attn_min_bits": 8,
    # embeddings / untied head
    "embed_min_bits": 4,
}
# in_features not divisible by any MLX group size (32/64/128) -> cannot quantize
UNQUANTIZABLE_IN_FEATURES = {4304}


def source_key(path: str) -> str:
    p = path.replace("language_model.model.", "model.language_model.")
    p = p.replace("language_model.lm_head", "lm_head")
    p = p.replace("vision_tower.", "model.visual.")
    return p + ".weight"


def read_norms(src: Path, keys: set[str]) -> dict[str, tuple[float, int]]:
    """||W||_F^2 and numel per tensor, streaming shard by shard."""
    import mlx.core as mx
    out = {}
    for shard in sorted(glob.glob(str(src / "model-*.safetensors"))):
        with open(shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        want = [k for k in hdr if k in keys]
        if not want:
            continue
        arrs = mx.load(shard)
        for k in want:
            w = arrs[k].astype(mx.float32)
            fro = float((w * w).sum().item())
            out[k] = (fro, int(w.size))
        del arrs
    return out


def classify(path: str) -> str:
    if path.startswith("vision_tower"):
        return "vision"
    if "self_attn" in path:
        return "attn"
    if "linear_attn" in path:
        return "gdn"
    if ".mlp." in path:
        return "mlp"
    if "lm_head" in path or "embed" in path:
        return "embed"
    return "other"


def main(argv) -> int:
    if len(argv) < 4:
        print(__doc__)
        return 1
    src, calib_json, out_p = Path(argv[1]), Path(argv[2]), Path(argv[3])
    target_gib, base_bits, gs = 10.0, 2, 128
    for i, a in enumerate(argv):
        if a == "--target-gib":
            target_gib = float(argv[i + 1])
        if a == "--base-bits":
            base_bits = int(argv[i + 1])
        if a == "--group-size":
            gs = int(argv[i + 1])

    stats = json.loads(calib_json.read_text())["stats"]
    keys = {source_key(p) for p in stats}
    print(f"  reading ||W||_F^2 for {len(keys)} tensors ...", flush=True)
    norms = read_norms(src, keys)
    print(f"  read {len(norms)}", flush=True)

    mods = []
    for path, st in stats.items():
        k = source_key(path)
        if k not in norms:
            continue
        fro, numel = norms[k]
        tr = st["trace"]
        mods.append({
            "path": path, "group": classify(path),
            "trace": tr, "fro2": fro, "numel": numel,
            "in_features": st["in_features"],
            "score": tr * fro,
            "quantizable": st["in_features"] not in UNQUANTIZABLE_IN_FEATURES
                           and st["in_features"] % gs == 0,
        })

    # normalise score to a 0..1 rank
    qs = [m for m in mods if m["quantizable"]]
    order = sorted(qs, key=lambda m: -m["score"])
    for i, m in enumerate(order):
        m["rank"] = i
        m["pct"] = i / max(len(order) - 1, 1)

    # ── allocate ──────────────────────────────────────────────────────────
    # Floors first (safety), then spend the remaining budget top-down by score.
    for m in mods:
        g = m["group"]
        if not m["quantizable"]:
            m["bits"] = 16          # fp16 passthrough (e.g. in_features 4304)
            m["reason"] = "in_features not divisible by an MLX group size"
        elif g == "attn":
            m["bits"] = FORCED["attn_min_bits"]; m["reason"] = "coherence anchor (forced)"
        elif g == "vision":
            m["bits"] = FORCED["vision_min_bits"]; m["reason"] = "vision floor (collapse precedent)"
        elif g == "embed":
            m["bits"] = FORCED["embed_min_bits"]; m["reason"] = "embed/head floor"
        else:
            m["bits"] = base_bits; m["reason"] = "base"

    def total_gib():
        b = 0
        for m in mods:
            if m["bits"] >= 16:
                b += m["numel"] * 2
            else:
                b += m["numel"] * (m["bits"] + 32.0 / gs) / 8
        return b / 2**30

    base_size = total_gib()
    # promote highest-score MLP/GDN modules while budget allows
    promoted = 0
    for m in order:
        if m["group"] not in ("mlp", "gdn") or m["bits"] != base_bits:
            continue
        cost = m["numel"] * 1.0 / 8 / 2**30      # +1 bit
        if total_gib() + cost > target_gib:
            continue
        m["bits"] = base_bits + 1
        m["reason"] = f"promoted +1 (rank {m['rank']}/{len(order)})"
        promoted += 1
        # a second promotion for the very top of the ranking
        if m["pct"] < 0.10 and total_gib() + cost <= target_gib:
            m["bits"] = base_bits + 2
            m["reason"] = f"promoted +2 (top decile, rank {m['rank']})"

    final = total_gib()
    import collections
    dist = collections.Counter(m["bits"] for m in mods)
    print(f"\n  base @ {base_bits}-bit gs{gs}: {base_size:.2f} GiB")
    print(f"  after promotion       : {final:.2f} GiB  (target {target_gib})")
    print(f"  promoted modules      : {promoted}")
    print(f"  bit distribution      : {dict(sorted(dist.items()))}")
    print()
    bygrp = collections.defaultdict(collections.Counter)
    for m in mods:
        bygrp[m["group"]][m["bits"]] += 1
    for g, c in sorted(bygrp.items()):
        print(f"    {g:8s} {dict(sorted(c.items()))}")

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({
        "source": str(src), "target_gib": target_gib, "base_bits": base_bits,
        "group_size": gs, "projected_gib": final,
        "method": "hessian-trace x frobenius (tr(H)*||W||_F^2)",
        "modules": sorted(mods, key=lambda m: -m["score"]),
    }, indent=1))
    print(f"\n  bit map -> {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

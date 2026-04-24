"""Execute a prune plan on Kimi K2.6: drop experts, rewrite shards.

Pure prune (no absorb-merge — REAP paper shows merging hurts generative
tasks via functional subspace collapse).

Per source shard, walk tensors:
  - For routed expert tensors (.mlp.experts.E.{gate,up,down}_proj.{weight_packed,
    weight_scale,weight_shape}):
    * drop if E ∈ plan.drop[L]
    * renumber to 0..n_keep-1 if E ∈ plan.keep[L]
  - For router (.mlp.gate.weight): keep only rows for kept experts in kept order
  - For router bias (.mlp.gate.e_score_correction_bias): keep only kept entries
  - Everything else (layernorms, attention, shared expert, embed, lm_head,
    dense layer 0): passthrough unchanged

Config.json: update text_config.n_routed_experts = n_keep.
Shards: same count, same structure, fewer expert tensor keys.

Usage:
  python -m jang_tools.kimi_prune.prune \\
      --src <path/to/sources>/Kimi_K2_6_FP8 \\
      --dst <path/to/Kimi-K2.6-REAP-30> \\
      --plan <path/to/data-drive>/kimi_calib/jangreap_run/prune_plan_30.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


# Kimi K2.6 tensor key patterns (prefix = language_model.model.layers.L)
EXPERT_RE = re.compile(
    r"^(?P<prefix>language_model\.model\.layers\.)"
    r"(?P<layer>\d+)"
    r"\.mlp\.experts\."
    r"(?P<expert>\d+)"
    r"\.(?P<proj>gate_proj|up_proj|down_proj)"
    r"\.(?P<kind>weight_packed|weight_scale|weight_shape)$"
)
ROUTER_WEIGHT_RE = re.compile(
    r"^(?P<prefix>language_model\.model\.layers\.)"
    r"(?P<layer>\d+)"
    r"\.mlp\.gate\.weight$"
)
ROUTER_BIAS_RE = re.compile(
    r"^(?P<prefix>language_model\.model\.layers\.)"
    r"(?P<layer>\d+)"
    r"\.mlp\.gate\.e_score_correction_bias$"
)


@dataclass
class PruneStats:
    passthrough: int = 0
    dropped_expert_keys: int = 0
    renumbered_expert_keys: int = 0
    router_rewrites: int = 0


def build_renumber(plan: dict) -> dict[int, dict[int, int]]:
    """renumber[L][old_expert_idx] = new_slot, or -1 if dropped.

    Layers not in plan (e.g. dense layer 0) get identity map.
    """
    rn: dict[int, dict[int, int]] = {}
    for row in plan["per_layer"]:
        L = row["layer"]
        keep_sorted = sorted(row["keep"])
        drop_set = set(row["drop"])
        m: dict[int, int] = {}
        for new_slot, old in enumerate(keep_sorted):
            m[old] = new_slot
        for old in drop_set:
            m[old] = -1
        rn[L] = m
    return rn


def execute(src: Path, dst: Path, plan_path: Path, compress_level: int = 0):
    plan = json.loads(plan_path.read_text())
    n_keep = plan["n_keep_per_layer"]
    n_drop = plan["n_drop_per_layer"]
    n_exp = plan["n_experts_per_layer"]
    renumber = build_renumber(plan)

    dst.mkdir(parents=True, exist_ok=True)
    shards = sorted(src.glob("model-*.safetensors"))
    print(f"[prune] src={src}  dst={dst}  plan={plan_path}", flush=True)
    print(f"[prune] {len(shards)} shards, {n_keep}/{n_exp} experts kept/layer "
          f"({(1-n_keep/n_exp)*100:.0f}% pruned)", flush=True)

    stats = PruneStats()
    t0 = time.time()

    for shard_i, sp in enumerate(shards):
        out: dict[str, torch.Tensor] = {}
        with safe_open(sp, framework="pt") as f:
            for k in f.keys():
                m = EXPERT_RE.match(k)
                if m:
                    L = int(m.group("layer"))
                    e = int(m.group("expert"))
                    proj = m.group("proj")
                    kind = m.group("kind")
                    rn_L = renumber.get(L)
                    if rn_L is None:
                        # Layer not in plan — keep all (shouldn't happen for MoE layers)
                        out[k] = f.get_tensor(k)
                        stats.passthrough += 1
                        continue
                    new_slot = rn_L.get(e, e)
                    if new_slot == -1:
                        stats.dropped_expert_keys += 1
                        continue
                    if new_slot == e:
                        out[k] = f.get_tensor(k)
                    else:
                        new_key = (f"{m.group('prefix')}{L}.mlp.experts."
                                   f"{new_slot}.{proj}.{kind}")
                        out[new_key] = f.get_tensor(k)
                    stats.renumbered_expert_keys += 1
                    continue

                m = ROUTER_WEIGHT_RE.match(k)
                if m:
                    L = int(m.group("layer"))
                    rn_L = renumber.get(L)
                    if rn_L is None:
                        out[k] = f.get_tensor(k)
                        stats.passthrough += 1
                        continue
                    keep_sorted = sorted([e for e, s in rn_L.items() if s != -1])
                    w = f.get_tensor(k)  # (E, H) bf16
                    idx_t = torch.tensor(keep_sorted, dtype=torch.long)
                    out[k] = w.index_select(0, idx_t).contiguous()
                    stats.router_rewrites += 1
                    continue

                m = ROUTER_BIAS_RE.match(k)
                if m:
                    L = int(m.group("layer"))
                    rn_L = renumber.get(L)
                    if rn_L is None:
                        out[k] = f.get_tensor(k)
                        stats.passthrough += 1
                        continue
                    keep_sorted = sorted([e for e, s in rn_L.items() if s != -1])
                    b = f.get_tensor(k)  # (E,)
                    idx_t = torch.tensor(keep_sorted, dtype=torch.long)
                    out[k] = b.index_select(0, idx_t).contiguous()
                    stats.router_rewrites += 1
                    continue

                # Passthrough: layernorms, attention, shared_experts, dense
                # layer 0, embed, lm_head, etc.
                out[k] = f.get_tensor(k)
                stats.passthrough += 1

        dst_shard = dst / sp.name
        save_file(out, str(dst_shard))
        print(f"  shard {shard_i+1}/{len(shards)}  {sp.name}  "
              f"({len(out)} tensors  {time.time()-t0:.0f}s total)", flush=True)

    # Update config.json
    src_cfg = json.loads((src / "config.json").read_text())
    txt = src_cfg.get("text_config", src_cfg)
    txt["n_routed_experts"] = n_keep
    if "text_config" in src_cfg:
        src_cfg["text_config"] = txt
    else:
        src_cfg = txt
    with (dst / "config.json").open("w") as f:
        json.dump(src_cfg, f, indent=2)
    print(f"[prune] config.json updated: n_routed_experts = {n_keep}", flush=True)

    # Copy non-weight files: tokenizer, processor, README, chat template, etc.
    copied = 0
    for p in src.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") \
                and p.name != "config.json":
            shutil.copy2(p, dst / p.name)
            copied += 1
    print(f"[prune] copied {copied} aux files", flush=True)

    # Record plan used
    (dst / "jangreap_plan.json").write_text(plan_path.read_text())

    # Write a prune_info.json summary
    info = {
        "source_model": str(src),
        "source_plan": str(plan_path),
        "base_ratio": plan["base_ratio"],
        "n_keep_per_layer": n_keep,
        "n_drop_per_layer": n_drop,
        "n_experts_per_layer_source": n_exp,
        "stats": {
            "passthrough_tensors": stats.passthrough,
            "dropped_expert_keys": stats.dropped_expert_keys,
            "renumbered_expert_keys": stats.renumbered_expert_keys,
            "router_rewrites": stats.router_rewrites,
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with (dst / "prune_info.json").open("w") as f:
        json.dump(info, f, indent=2)

    total_size = sum(p.stat().st_size for p in dst.glob("model-*.safetensors"))
    print(f"[prune] DONE in {time.time()-t0:.0f}s", flush=True)
    print(f"  passthrough={stats.passthrough} dropped={stats.dropped_expert_keys} "
          f"renumbered={stats.renumbered_expert_keys} router_rewrites={stats.router_rewrites}",
          flush=True)
    print(f"  total output shard size: {total_size/1e9:.1f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--plan", required=True, type=Path)
    args = ap.parse_args()
    execute(args.src, args.dst, args.plan)


if __name__ == "__main__":
    main()

"""
Rebundle a JANGTQ bundle from per-expert TQ keys to pre-stacked switch_mlp.
Created by Jinho Jang (eric@jangq.ai)

Per JANGTQ-PRESTACK-SPEC.md, every JANGTQ bundle should ship routed-expert
tensors pre-stacked along axis 0 in the main shards (no `jangtq_stacked.*`
sidecar, no per-expert keys). This utility takes an existing per-expert
JANGTQ bundle and produces a pre-stacked bundle WITHOUT re-quantizing —
the bytes per (layer, expert, projection, value) are identical, only the
file layout changes.

Architecture map (auto-detected from existing key shapes):

  DSV4         model.layers.{L}.ffn.experts.{E}.{w1|w2|w3}.tq_*
            →  model.layers.{L}.mlp.switch_mlp.{gate_proj|down_proj|up_proj}.tq_*

  Bailing       model.layers.{L}.mlp.experts.{E}.{gate_proj|up_proj|down_proj}.tq_*
            →  model.layers.{L}.mlp.switch_mlp.{...}.tq_*

  MiniMax       model.layers.{L}.block_sparse_moe.experts.{E}.{w1|w2|w3}.tq_*
            →  model.layers.{L}.block_sparse_moe.switch_mlp.{...}.tq_*

  GLM/Qwen3.6   model.layers.{L}.mlp.experts.{E}.{gate_proj|up_proj|down_proj}.tq_*
            →  model.layers.{L}.mlp.switch_mlp.{...}.tq_*

  Nemotron-H    backbone.layers.{L}.mixer.experts.{E}.{up_proj|down_proj}.tq_*
            →  backbone.layers.{L}.mixer.switch_mlp.{fc1|fc2}.tq_*

Output bundle:
- New main shards with pre-stacked switch_mlp.* keys + all non-routed tensors passed through unchanged.
- Updated model.safetensors.index.json
- jangtq_stacked.{safetensors,json} deleted (no longer needed — bundle IS pre-stacked).
- Other metadata copied (config.json, jang_config.json, generation_config, tokenizer, chat_template, modeling .py).
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import struct as _struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
from tqdm import tqdm


# Per-architecture regex + projection rename map.
# Each rule: (input_regex, prefix_replace, proj_rename_dict, label).
#   input_regex matches the per-expert KEY BASE (without .tq_packed/_norms/_bits suffix).
#   capture group 1 = layer prefix (e.g. "model.layers.7.ffn.")
#   capture group 2 = expert index
#   capture group 3 = projection name on disk (w1/w2/w3 or gate_proj/...)
#   prefix_replace: tuple (old, new) applied to capture 1 to remap (e.g. ".ffn." → ".mlp.")
#                   or None to keep prefix as-is
#   proj_rename_dict: maps disk projection name → output projection name
_PREFIX = r"(?:model\.|model\.language_model\.|language_model\.model\.|backbone\.|)"
# Match either `model.layers.N....` (HF convention) or bare `layers.N....`
# (DSV4 / Bailing post-sanitize / some converters). The leading optional
# group handles VLM and bare-layers variants.
RULES: list[tuple[re.Pattern, tuple | None, dict[str, str], str]] = [
    # DSV4 — DeepSeek V4 family. Disk uses `[model.]layers.N.ffn.experts.E.w[123]`;
    # the in-memory module path is `.mlp.switch_mlp.X` after the loader's
    # sanitize/streaming hydrate (jang_tools.dsv4.mlx_model). Pre-stacked output
    # uses the post-sanitize prefix directly so no further translation is needed
    # at load time. Capture group 1 includes `ffn.` so the prefix_replace can
    # swap it for `mlp.`.
    (
        re.compile(rf"^({_PREFIX}layers\.\d+\.ffn\.)experts\.(\d+)\.(w[123])$"),
        (".ffn.", ".mlp."),
        {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"},
        "dsv4",
    ),
    # MiniMax M2 — `[model.]layers.N.block_sparse_moe.experts.E.w[123]` stays
    # under the same block prefix (no rename). Capture includes `block_sparse_moe.`.
    (
        re.compile(rf"^({_PREFIX}layers\.\d+\.block_sparse_moe\.)experts\.(\d+)\.(w[123])$"),
        None,
        {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"},
        "minimax_m2",
    ),
    # Bailing / GLM / Qwen / Holo — `[model.]layers.N.mlp.experts.E.{gate_proj|up_proj|down_proj}`.
    # Capture includes `mlp.` so output stays under `mlp.switch_mlp.X`.
    (
        re.compile(rf"^({_PREFIX}layers\.\d+\.mlp\.)experts\.(\d+)\.(gate_proj|up_proj|down_proj)$"),
        None,
        {"gate_proj": "gate_proj", "up_proj": "up_proj", "down_proj": "down_proj"},
        "bailing/glm/qwen",
    ),
    # Nemotron-H — `backbone.layers.N.mixer.experts.E.{up_proj|down_proj}` →
    # `backbone.layers.N.mixer.switch_mlp.{fc1|fc2}`. Nemotron is GLU (not SwiGLU)
    # so gate_proj never appears.
    (
        re.compile(r"^(backbone\.layers\.\d+\.mixer\.)experts\.(\d+)\.(up_proj|down_proj)$"),
        None,
        {"up_proj": "fc1", "down_proj": "fc2"},
        "nemotron_h",
    ),
]


def parse_tq_key(key: str) -> tuple[str, str] | None:
    """Strip the .tq_packed / .tq_norms / .tq_bits suffix; return (base, part) or None."""
    for suffix, part in [(".tq_packed", "packed"), (".tq_norms", "norms"), (".tq_bits", "bits")]:
        if key.endswith(suffix):
            return key[: -len(suffix)], part
    return None


def match_per_expert_rule(base: str) -> tuple[str, int, str, str] | None:
    """
    Try every architecture rule. On hit, return:
      (out_base, expert_id, label, original_proj)
    where out_base = "<remapped_layer_prefix>switch_mlp.<output_proj>".
    """
    for pat, prefix_replace, proj_map, label in RULES:
        m = pat.match(base)
        if not m:
            continue
        layer_prefix = m.group(1)
        expert_id = int(m.group(2))
        disk_proj = m.group(3)
        if disk_proj not in proj_map:
            return None  # unknown projection name in this arch
        out_proj = proj_map[disk_proj]
        if prefix_replace is not None:
            old, new = prefix_replace
            layer_prefix = layer_prefix.replace(old, new)
        return f"{layer_prefix}switch_mlp.{out_proj}", expert_id, label, disk_proj
    return None


def rebundle(src, dst, *, shard_bytes: int = 1_000_000_000, dry_run: bool = False):
    """Programmatic entry point for converters to call after their per-expert
    quantization pass completes. Equivalent to running the CLI on (src, dst).

    Codex 2026-05-05 #2: every JANGTQ converter that emits per-expert layout
    should call this as a final step so the SHIPPED bundle is prestack-spec
    compliant. Old per-expert bundles need the CLI form to migrate.
    """
    import argparse as _ap
    args = _ap.Namespace(
        src=Path(src),
        dst=Path(dst),
        shard_bytes=shard_bytes,
        dry_run=dry_run,
    )
    return _rebundle_impl(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("src", type=Path, help="source bundle (per-expert JANGTQ)")
    ap.add_argument("dst", type=Path, help="destination bundle (will be created)")
    ap.add_argument("--shard-bytes", type=int, default=1_000_000_000,
                    help="max bytes per output shard (default 1 GB)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report planned mapping without writing")
    args = ap.parse_args()
    return _rebundle_impl(args)


def _rebundle_impl(args):
    SRC: Path = args.src.expanduser().resolve()
    DST: Path = args.dst.expanduser().resolve()
    if not SRC.is_dir():
        sys.exit(f"src bundle not found: {SRC}")
    if SRC == DST:
        sys.exit("dst must differ from src (we don't overwrite in place)")
    DST.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  JANGTQ rebundle (per-expert → pre-stacked switch_mlp)")
    print(f"  src: {SRC}")
    print(f"  dst: {DST}")
    print("=" * 70, flush=True)

    # Phase 1: scan source — collect per-expert TQ groups + non-routed tensors.
    print("\n[1/3] scanning source shards...", flush=True)
    src_shards = sorted(SRC.glob("model-*.safetensors"))
    if not src_shards:
        sys.exit(f"no model-*.safetensors in {SRC}")

    tq_per_expert: dict[tuple[str, int], dict[str, tuple[Path, str]]] = defaultdict(dict)
    # tq_per_expert[(out_base, expert_id)][part] = (shard_path, original_disk_key)
    passthrough_keys: list[tuple[str, Path]] = []
    arch_label_counts: dict[str, int] = defaultdict(int)
    n_tq_per_expert = 0

    for sf_path in tqdm(src_shards, desc="scan"):
        with safe_open(str(sf_path), framework="numpy") as f:
            for key in f.keys():
                tq = parse_tq_key(key)
                if tq is None:
                    passthrough_keys.append((key, sf_path))
                    continue
                base, part = tq
                hit = match_per_expert_rule(base)
                if hit is None:
                    # Could be an already-pre-stacked key (e.g. switch_mlp.* on
                    # a partially-rebundled bundle), or a non-routed TQ key
                    # (attention/embed). Pass through.
                    passthrough_keys.append((key, sf_path))
                    continue
                out_base, expert_id, label, _orig = hit
                tq_per_expert[(out_base, expert_id)][part] = (sf_path, key)
                arch_label_counts[label] += 1
                if part == "packed":
                    n_tq_per_expert += 1

    if not tq_per_expert:
        sys.exit("found 0 per-expert TQ groups — nothing to rebundle")

    # Group expert IDs per (out_base) so we know each group's expert count.
    per_out_base: dict[str, dict[int, dict[str, tuple[Path, str]]]] = defaultdict(dict)
    for (out_base, eid), parts in tq_per_expert.items():
        per_out_base[out_base][eid] = parts

    # Sanity: every group should have the same expert count.
    expert_counts = {ob: len(d) for ob, d in per_out_base.items()}
    n_experts_unique = sorted(set(expert_counts.values()))
    print(f"  per-expert TQ groups: {n_tq_per_expert}", flush=True)
    print(f"  output (layer × proj) groups: {len(per_out_base)}", flush=True)
    print(f"  expert counts seen across groups: {n_experts_unique}", flush=True)
    print(f"  passthrough non-TQ-per-expert keys: {len(passthrough_keys)}", flush=True)
    if arch_label_counts:
        for label, count in sorted(arch_label_counts.items()):
            print(f"  arch matched: {label} ({count // 3} groups, {count} tensors)", flush=True)

    if args.dry_run:
        print("\n[dry-run] sample 5 mappings:")
        for (out_base, eid), parts in list(tq_per_expert.items())[:5]:
            sample = next(iter(parts.values()))
            print(f"  {sample[1]}  →  {out_base}.tq_{next(iter(parts))}  (expert {eid})")
        print("\n[dry-run] sample 5 passthrough keys:")
        for k, _ in passthrough_keys[:5]:
            print(f"  {k}")
        print("\n[dry-run] no files written.")
        return

    # Phase 2: stack + emit.
    print("\n[2/3] stacking + writing...", flush=True)
    shard_idx = 0
    shard_buf: dict[str, np.ndarray] = {}
    shard_bytes = 0
    weight_map: dict[str, str] = {}

    def flush_shard():
        nonlocal shard_idx, shard_buf, shard_bytes
        if not shard_buf:
            return
        shard_idx += 1
        fname = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        save_file(shard_buf, str(DST / fname))
        for k in shard_buf:
            weight_map[k] = fname
        print(f"    shard {shard_idx}: {len(shard_buf)} tensors, {shard_bytes/1e9:.2f} GB", flush=True)
        shard_buf = {}
        shard_bytes = 0

    def emit(key: str, arr: np.ndarray):
        nonlocal shard_bytes
        shard_buf[key] = arr
        shard_bytes += arr.nbytes
        if shard_bytes >= args.shard_bytes:
            flush_shard()

    # Group A: per-(out_base) stack the per-expert tensors.
    sorted_bases = sorted(per_out_base.keys())
    for out_base in tqdm(sorted_bases, desc="stack"):
        experts = per_out_base[out_base]
        n_exp = max(experts.keys()) + 1
        if len(experts) != n_exp or not all(i in experts for i in range(n_exp)):
            missing = [i for i in range(n_exp) if i not in experts]
            sys.exit(f"non-contiguous experts for {out_base}: missing {missing[:5]}…")

        # Materialize per-expert tensors.
        packed_list, norms_list, bits = [], [], None
        for eid in range(n_exp):
            parts = experts[eid]
            for needed in ("packed", "norms", "bits"):
                if needed not in parts:
                    sys.exit(f"missing tq_{needed} for {out_base} expert {eid}")
            sf_p, k_p = parts["packed"]
            sf_n, k_n = parts["norms"]
            sf_b, k_b = parts["bits"]
            with safe_open(str(sf_p), framework="numpy") as f:
                packed_list.append(f.get_tensor(k_p))
            with safe_open(str(sf_n), framework="numpy") as f:
                norms_list.append(f.get_tensor(k_n))
            if bits is None:
                with safe_open(str(sf_b), framework="numpy") as f:
                    bits_arr = f.get_tensor(k_b)
                    bits = int(bits_arr[0])

        stacked_packed = np.stack(packed_list, axis=0)   # [n_exp, out, packed_in]
        stacked_norms = np.stack(norms_list, axis=0)     # [n_exp, out]
        bits_tensor = np.array([bits], dtype=np.uint8)

        emit(f"{out_base}.tq_packed", stacked_packed)
        emit(f"{out_base}.tq_norms", stacked_norms)
        emit(f"{out_base}.tq_bits", bits_tensor)

        del packed_list, norms_list, stacked_packed, stacked_norms
        gc.collect()

    # Group B: passthrough keys — load + write through unchanged.
    # Skip files inside .cache/ subdirs and the redundant jangtq_stacked sidecar.
    print(f"\n  passthrough phase: {len(passthrough_keys)} keys", flush=True)
    seen_passthrough_keys: set[str] = set()
    for key, sf_path in tqdm(passthrough_keys, desc="passthrough"):
        if key in seen_passthrough_keys:
            continue
        seen_passthrough_keys.add(key)
        try:
            with safe_open(str(sf_path), framework="numpy") as f:
                arr = f.get_tensor(key)
        except (TypeError, ValueError):
            # bf16 — load as bytes via mlx fallback (rare for JANGTQ).
            from jang_tools.calibrate import _load_bf16_tensor
            shape = list(safe_open(str(sf_path), framework="numpy").get_slice(key).get_shape())
            arr = _load_bf16_tensor(sf_path, key, shape)
        emit(key, arr)
        del arr

    flush_shard()

    # Phase 3: rename shards to final XX-of-YY pattern + write index.
    print("\n[3/3] finalizing index + cleanup...", flush=True)
    total_shards = shard_idx
    for i in range(1, total_shards + 1):
        old = DST / f"model-{i:05d}-of-XXXXX.safetensors"
        new = DST / f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        if old.exists():
            old.rename(new)
    weight_map = {k: v.replace("XXXXX", f"{total_shards:05d}") for k, v in weight_map.items()}

    total_size = sum((DST / fname).stat().st_size for fname in set(weight_map.values()))
    index = {
        "metadata": {
            "format": "jangtq",
            "total_size": total_size,
            "rebundled_from": str(SRC),
            "rebundled_layout": "prestacked-switch_mlp",
        },
        "weight_map": weight_map,
    }
    with open(DST / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Copy non-shard files (config.json, jang_config.json, generation_config,
    # tokenizer, chat template, modeling .py, capabilities, sidecar, logos).
    # Skip the redundant `jangtq_stacked.*` files — pre-stacked bundle does not
    # need them.
    print(f"\n  copying metadata files...", flush=True)
    skip_names = {"jangtq_stacked.safetensors", "jangtq_stacked.json"}
    skip_dirs = {".cache", ".git"}
    for f in SRC.iterdir():
        if f.name in skip_names:
            continue
        if f.is_dir() and f.name in skip_dirs:
            continue
        if f.is_file() and f.suffix == ".safetensors" and f.name.startswith("model-"):
            continue  # source shards — replaced by new ones
        if f.name == "model.safetensors.index.json":
            continue  # we write our own
        dst_path = DST / f.name
        if f.is_file():
            shutil.copy2(str(f), str(dst_path))
        elif f.is_dir():
            shutil.copytree(str(f), str(dst_path), dirs_exist_ok=True)

    # Mark in jang_config that this is pre-stacked so loaders can opt for the
    # fast path without inspecting tensor shapes.
    jc_path = DST / "jang_config.json"
    if jc_path.exists():
        try:
            jc = json.load(open(jc_path))
            jc["routed_expert_layout"] = "prestacked"
            json.dump(jc, open(jc_path, "w"), indent=2)
            print(f"  jang_config.json: routed_expert_layout=prestacked")
        except Exception as e:
            print(f"  warn: jang_config.json patch failed: {e}")

    src_size = sum(p.stat().st_size for p in SRC.glob("**/*") if p.is_file())
    dst_size = sum(p.stat().st_size for p in DST.glob("**/*") if p.is_file())
    print(f"\n  Source total:    {src_size/1e9:6.2f} GB")
    print(f"  Output total:    {dst_size/1e9:6.2f} GB")
    print(f"  Output shards:   {total_shards}")
    print(f"  Output:          {DST}")
    print(f"  Done.")


if __name__ == "__main__":
    main()

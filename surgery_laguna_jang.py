#!/usr/bin/env python3
"""
Qwen3.6-35B-A3B JANGTQ CRACK surgery (dual-pathway binary patch).

Forked from surgery_jangtq_xshard.py. Differences from MiniMax surgery:
  * Target tensor names use Qwen 3.6 VLM layout:
      model.layers.{L}.self_attn.o_proj.{weight,scales,biases}   (FA)
      model.layers.{L}.linear_attn.out_proj.{weight,scales,biases}  (SSM)
  * Dual-pathway: applies the same shared refusal vector to FA o_proj AND
    SSM out_proj at adjacent layer pairs (per 397B JANG_1L finding).
  * Single shared vector (SVSA) -- default reads layer_39 from the JANGTQ4
    probe because that layer has the highest projected_norm (7.60) and
    cleanest cos(r,p)=0.988. The vector is quant-invariant so it transfers
    to JANGTQ2 without regeneration.
  * No M2.7 eos/chat_template post-fix. Qwen 3.6 config is already correct.

All target tensors are 8-bit affine, so standard dequant -> CRACK ablate
W' = W - s * v_hat @ (v_hat^T @ W) -> requant works in place. 2-bit/4-bit
routed experts are NOT touched (refusal lives in residual-stream space and
never routes through a quantized codebook).

Output: binary-identical shard copy with only the targeted tensor bytes
replaced; untouched shards are symlinked for speed.
"""

import argparse, gc, json, os, struct, shutil, sys, time
from pathlib import Path

import mlx.core as mx
import numpy as np

_materialize = mx.eval  # MLX lazy materialization, not Python eval().


def parse_safetensors_header(filepath):
    with open(filepath, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    data_start = 8 + header_len
    tensors = {}
    for key, info in header.items():
        if key == "__metadata__":
            continue
        tensors[key] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "start": data_start + info["data_offsets"][0],
            "end": data_start + info["data_offsets"][1],
            "size": info["data_offsets"][1] - info["data_offsets"][0],
        }
    return tensors, data_start


def detect_tensor_bits(weight_shape, scales_shape):
    packed_cols = weight_shape[-1]
    scales_cols = scales_shape[-1]
    candidates = []
    for bits in [8, 6, 5, 4, 3, 2]:
        # MLX packs real_cols*bits total bits into uint32 columns. This differs
        # from packed_cols*(32//bits) for bit-widths that don't divide 32 (3/5/6).
        real_cols = packed_cols * 32 // bits
        if (real_cols * bits + 31) // 32 != packed_cols:  # not an exact packing
            continue
        if scales_cols > 0 and real_cols % scales_cols == 0:
            gs = real_cols // scales_cols
            if gs in [32, 64, 128, 256, 512]:
                candidates.append((bits, gs))
    if not candidates:
        return 8, 64
    # Prefer HIGHER BITS (more precision = the correct weight layout in JANG mixed-precision;
    # lower-bit candidates also parse valid but correspond to a different notional intermediate
    # size). Verified: switch_mlp shape (256,3072,96)+scales (256,3072,16) gives valid (6,32)
    # AND (3,64) — the (3,64) reading yields 0% ablation because it dequantizes to the wrong
    # intermediate; (6,32) is correct per jang_config bits_by_role.routed_expert=6.
    candidates.sort(key=lambda x: (-x[0], -x[1]))
    return candidates[0]


def ablate_weight(w_float, v, strength, mode="standard"):
    """Remove the v_hat direction from W.

    Standard: W' = W - s * v_hat @ (v_hat^T @ W). Row norms can drop when v is
    aligned with W's rows. Compounds across layers -> coherence damage at
    high strength / wide coverage.

    MPOA (Magnitude-Preserving Orthogonal Ablation): decompose W into row
    magnitudes + unit-direction rows, ablate v from the directions only,
    restore original row norms. Guarantees ||W_new[i,:]|| == ||W[i,:]||.
    Safer at higher strengths; used by 397B JANG_1L (96.2% HB at s=3/2).
    """
    w = w_float.astype(mx.float32)
    v = v.astype(mx.float32)
    v_hat = v / (mx.sqrt(mx.sum(v * v)) + 1e-10)

    if mode == "standard":
        proj = v_hat @ w
        w = w - strength * mx.outer(v_hat, proj)
        _materialize(w)
        return w

    if mode == "mpoa":
        # Match reference impl in crack/engine/surgeon.py._abliterate_weight_mpoa
        # for v_on_axis0 (our case: v is output-space = axis 0 of W [out, in]).
        row_norms = mx.linalg.norm(w, axis=1, keepdims=True)       # [out, 1]
        safe_norms = mx.maximum(row_norms, 1e-8)
        w_dir = w / safe_norms                                     # unit rows

        proj_coeffs = v_hat @ w_dir                                # [in]
        w_dir_ablated = w_dir - strength * mx.outer(v_hat, proj_coeffs)

        new_norms = mx.linalg.norm(w_dir_ablated, axis=1, keepdims=True)
        safe_new_norms = mx.maximum(new_norms, 1e-8)
        w_dir_new = w_dir_ablated / safe_new_norms                 # renormalize
        w_new = row_norms * w_dir_new                              # restore magnitudes
        _materialize(w_new)
        return w_new

    raise ValueError(f"unknown ablation mode: {mode!r}")


def ablate_weight_3d(w_float_3d, v, strength, mode="mpoa"):
    """Batched ablation for switch_mlp.down_proj [E=experts, out, in].
    Same v (out-space, dim=out) applies to every expert independently.
    Loops per-expert (safer for large tensors — fp16 sum-of-squares overflows
    on 400M-element batches; keeps each expert's math in fp32 within safe range)."""
    v = v.astype(mx.float32)
    v_hat = v / (mx.sqrt(mx.sum(v * v)) + 1e-10)
    E = w_float_3d.shape[0]
    out_list = []
    for e in range(E):
        w_e = w_float_3d[e].astype(mx.float32)   # [out, in] fp32 per expert
        assert v.shape[0] == w_e.shape[0], f"v {v.shape[0]} != out {w_e.shape[0]}"
        if mode == "standard":
            proj = v_hat @ w_e
            w_new_e = w_e - strength * mx.outer(v_hat, proj)
        elif mode == "mpoa":
            row_norms = mx.linalg.norm(w_e, axis=1, keepdims=True)
            safe = mx.maximum(row_norms, 1e-8)
            w_dir = w_e / safe
            proj = v_hat @ w_dir
            w_dir_ab = w_dir - strength * mx.outer(v_hat, proj)
            new_norms = mx.linalg.norm(w_dir_ab, axis=1, keepdims=True)
            safe_new = mx.maximum(new_norms, 1e-8)
            w_new_e = (row_norms * (w_dir_ab / safe_new))
        else:
            raise ValueError(f"unknown mode: {mode!r}")
        _materialize(w_new_e)
        out_list.append(w_new_e[None, ...])
    stacked = mx.concatenate(out_list, axis=0)
    _materialize(stacked)
    return stacked


FA_DEFAULT = [19, 23, 27, 31, 35, 39]
SSM_DEFAULT = [18, 22, 26, 30, 34, 38]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Source JANGTQ model dir")
    p.add_argument("--vectors", required=True, help="Probe vectors safetensors")
    p.add_argument("--vector-key", default="layer_39",
                   help="Which layer's vector to use (SVSA mode, shared across all targets)")
    p.add_argument("--per-layer", action="store_true",
                   help="Use each target's own layer vector instead of shared SVSA. "
                        "Required when per-layer cos(v_L, v_39) diverges significantly "
                        "(rotating refusal direction through residual stream).")
    p.add_argument("--output", "-o", required=True, help="Output model dir")
    p.add_argument("--fa-layers", default=",".join(str(x) for x in FA_DEFAULT),
                   help="Comma-separated FA layer indices for o_proj surgery")
    p.add_argument("--ssm-layers", default=",".join(str(x) for x in SSM_DEFAULT),
                   help="Comma-separated SSM layer indices for out_proj surgery")
    p.add_argument("--fa-strength", type=float, default=5.0,
                   help="Ablation strength for FA o_proj (default 5.0)")
    p.add_argument("--ssm-strength", type=float, default=3.0,
                   help="Ablation strength for SSM out_proj (default 3.0)")
    p.add_argument("--shared-layers", default="",
                   help="Comma-separated layer indices for mlp.shared_expert.down_proj surgery "
                        "(shared expert runs on every forward pass, writes to residual stream).")
    p.add_argument("--shared-strength", type=float, default=5.0,
                   help="Ablation strength for shared_expert.down_proj (default 5.0)")
    p.add_argument("--switch-layers", default="",
                   help="Comma-separated layer indices for mlp.switch_mlp.down_proj surgery "
                        "(batched 3D routed experts). Same v applies to all experts.")
    p.add_argument("--switch-strength", type=float, default=2.0,
                   help="Ablation strength for switch_mlp.down_proj routed experts (default 2.0)")
    p.add_argument("--strength", "-s", type=float, default=None,
                   help="Uniform strength; overrides --fa-strength/--ssm-strength")
    p.add_argument("--mode", choices=["standard", "mpoa"], default="standard",
                   help="Ablation mode. mpoa preserves per-row magnitudes; "
                        "safer at high strength / wide layer coverage.")
    args = p.parse_args()

    source = Path(args.model)
    output = Path(args.output)
    if not source.exists():
        sys.exit(f"ERROR: source not found: {source}")

    fa_layers = [int(x) for x in args.fa_layers.split(",") if x.strip()]
    ssm_layers = [int(x) for x in args.ssm_layers.split(",") if x.strip()]
    shared_layers = [int(x) for x in args.shared_layers.split(",") if x.strip()]
    switch_layers = [int(x) for x in args.switch_layers.split(",") if x.strip()]

    vdata = mx.load(args.vectors)
    shared_v = None
    if not args.per_layer:
        if args.vector_key not in vdata:
            sys.exit(f"ERROR: vector key {args.vector_key!r} not in {args.vectors}; "
                     f"available: {sorted(vdata.keys())[:5]}...")
        shared_v = vdata[args.vector_key].astype(mx.float32)
        if bool(mx.any(mx.isnan(shared_v))):
            sys.exit(f"ERROR: NaN in {args.vector_key}")

    def get_vec_for_layer(li):
        if not args.per_layer:
            return shared_v
        key = f"layer_{li}"
        if key not in vdata:
            sys.exit(f"ERROR: {key} missing from {args.vectors} (per-layer mode)")
        v = vdata[key].astype(mx.float32)
        if bool(mx.any(mx.isnan(v))):
            sys.exit(f"ERROR: NaN in {key}")
        return v

    fa_s = args.strength if args.strength is not None else args.fa_strength
    ssm_s = args.strength if args.strength is not None else args.ssm_strength
    shared_s = args.strength if args.strength is not None else args.shared_strength
    switch_s = args.strength if args.strength is not None else args.switch_strength

    print("=" * 60)
    print("  Qwen3.6-35B JANGTQ CRACK Surgery (dual-pathway)")
    print("=" * 60)
    print(f"  Source:     {source.name}")
    if args.per_layer:
        print(f"  Vectors:    PER-LAYER from {Path(args.vectors).name}")
    else:
        print(f"  Vector:     {args.vector_key} from {Path(args.vectors).name}")
    print(f"  Mode:       {args.mode}")
    print(f"  FA layers:  {fa_layers}  s={fa_s}")
    print(f"  SSM layers: {ssm_layers}  s={ssm_s}")
    if shared_layers:
        print(f"  Shared layers: {shared_layers}  s={shared_s}")
    if switch_layers:
        print(f"  Switch (routed) layers: {switch_layers}  s={switch_s}  [256 experts batched per layer]")

    idx_path = source / "model.safetensors.index.json"
    if not idx_path.exists():
        sys.exit(f"ERROR: missing {idx_path}")
    weight_map = json.loads(idx_path.read_text())["weight_map"]

    plan = []
    for li in fa_layers:
        plan.append(("FA", li, f"model.layers.{li}.self_attn.o_proj", fa_s))
    for li in ssm_layers:
        plan.append(("SSM", li, f"model.layers.{li}.linear_attn.out_proj", ssm_s))
    for li in shared_layers:
        plan.append(("SHR", li, f"model.layers.{li}.mlp.shared_expert.down_proj", shared_s))
    for li in switch_layers:
        plan.append(("SW", li, f"model.layers.{li}.mlp.switch_mlp.down_proj", switch_s))

    resolved = []
    shards_touched = set()
    for kind, li, base, strength in plan:
        wkey, skey, bkey = base + ".weight", base + ".scales", base + ".biases"
        for key in (wkey, skey, bkey):
            if key not in weight_map:
                sys.exit(f"ERROR: missing {key} in weight map")
        entry = {
            "kind": kind, "li": li, "strength": strength,
            "wkey": wkey, "skey": skey, "bkey": bkey,
            "wshard": weight_map[wkey],
            "sshard": weight_map[skey],
            "bshard": weight_map[bkey],
        }
        resolved.append(entry)
        shards_touched.update([entry["wshard"], entry["sshard"], entry["bshard"]])

    print(f"  Targets:    {len(resolved)} tensors across {len(shards_touched)} shards")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    for f in source.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, output / f.name)

    shard_files = sorted(source.glob("model-*.safetensors"))
    for sf in shard_files:
        dest = output / sf.name
        if sf.name in shards_touched:
            shutil.copy2(sf, dest)
        else:
            dest.symlink_to(sf)

    # Read jang_config.json for authoritative per-role bits (detect_tensor_bits alone
    # is ambiguous — e.g. 2L switch_mlp shape parses valid as both (6,32) AND (3,64);
    # config says routed_expert.down_proj=3 gs=64. Wrong bits → 100% RT diff → garbage.
    jang_cfg_path = source / "jang_config.json"
    role_bits = {}
    cfg_gs = None
    if jang_cfg_path.exists():
        jcfg = json.loads(jang_cfg_path.read_text())
        qcfg = jcfg.get("quantization", {})
        cfg_gs = qcfg.get("group_size")
        br = qcfg.get("bits_by_role", {})
        # routed_expert may be int or dict {gate_proj, up_proj, down_proj}
        r = br.get("routed_expert")
        role_bits["SW"] = (r["down_proj"] if isinstance(r, dict) else r, cfg_gs) if r is not None else None
        role_bits["SHR"] = (br.get("shared_expert"), cfg_gs) if br.get("shared_expert") else None
        role_bits["FA"] = (br.get("attention"), cfg_gs) if br.get("attention") else None
        role_bits["SSM"] = role_bits["FA"]

    headers = {}
    def get_header(shard_name):
        if shard_name not in headers:
            headers[shard_name], _ = parse_safetensors_header(output / shard_name)
        return headers[shard_name]

    total_patched = 0
    t_start = time.perf_counter()
    for entry in resolved:
        kind, li, strength = entry["kind"], entry["li"], entry["strength"]
        wkey, skey, bkey = entry["wkey"], entry["skey"], entry["bkey"]
        wshard, sshard, bshard = entry["wshard"], entry["sshard"], entry["bshard"]

        w_data = mx.load(str(source / wshard))
        s_data = mx.load(str(source / sshard)) if sshard != wshard else w_data
        b_data = mx.load(str(source / bshard)) if bshard not in (wshard, sshard) else (
            s_data if bshard == sshard else w_data
        )
        w_q = w_data[wkey]
        scales = s_data[skey]
        biases = b_data[bkey]

        # Prefer authoritative jang_config bits_by_role; fall back to shape detection.
        rb = role_bits.get(kind)
        if rb is not None and rb[0] is not None:
            bits, gs = rb
        else:
            bits, gs = detect_tensor_bits(w_q.shape, scales.shape)
        w_float = mx.dequantize(w_q, scales, biases, gs, bits)
        _materialize(w_float)

        v_for_layer = get_vec_for_layer(li)
        if w_float.ndim == 3:                   # routed switch_mlp: [E, out, in]
            w_new = ablate_weight_3d(w_float, v_for_layer, strength, mode=args.mode)
            # per-expert avg change% (avoid fp16 overflow on 400M-element sum)
            E = w_float.shape[0]
            deltas = []; norms = []
            for e in [0, E // 2, E - 1]:  # sample 3 experts for a stable %
                w_e_f32 = w_float[e].astype(mx.float32)
                d = float(mx.sqrt(mx.sum((w_new[e] - w_e_f32) ** 2)))
                n = float(mx.sqrt(mx.sum(w_e_f32 ** 2)))
                deltas.append(d); norms.append(n)
            change_pct = 100 * sum(deltas) / sum(norms) if sum(norms) > 0 else 0.0
        else:
            w_new = ablate_weight(w_float, v_for_layer, strength, mode=args.mode)
            pre_norm = float(mx.sqrt(mx.sum(w_float * w_float)))
            delta = float(mx.sqrt(mx.sum((w_new - w_float) * (w_new - w_float))))
            change_pct = (delta / pre_norm) * 100 if pre_norm > 0 else 0.0

        w_new_q, new_scales, new_biases = mx.quantize(w_new, group_size=gs, bits=bits)
        new_scales = new_scales.astype(mx.float16)
        new_biases = new_biases.astype(mx.float16)
        _materialize(w_new_q, new_scales, new_biases)

        for key, value, shard_name in [
            (wkey, w_new_q, wshard),
            (skey, new_scales, sshard),
            (bkey, new_biases, bshard),
        ]:
            tensors_in_shard = get_header(shard_name)
            info = tensors_in_shard.get(key)
            if info is None:
                sys.exit(f"ERROR: {key} not found in shard {shard_name} header")
            raw = np.array(value).tobytes()
            assert len(raw) == info["size"], (
                f"size mismatch for {key}: got {len(raw)} expected {info['size']}"
            )
            with open(output / shard_name, "r+b") as f:
                f.seek(info["start"])
                f.write(raw)

        total_patched += 1
        print(f"    {kind:>3} L{li:>2}  {bits}-bit s={strength:.2f}  change={change_pct:.2f}%")

        del w_data, s_data, b_data, w_float, w_new, w_new_q, new_scales, new_biases
        gc.collect()
        mx.clear_cache()

    elapsed = time.perf_counter() - t_start

    print("\n  Verifying shard sizes...")
    ok = True
    for sf in shard_files:
        dest = output / sf.name
        if dest.is_symlink():
            continue
        if sf.stat().st_size != dest.stat().st_size:
            print(f"    MISMATCH: {sf.name}")
            ok = False
    print("    All patched shards BYTE-IDENTICAL in size" if ok else "    MISMATCH DETECTED")

    symlinked = sum(1 for sf in shard_files if (output / sf.name).is_symlink())
    print("\n" + "=" * 60)
    print(f"DONE: {total_patched} tensors patched, {symlinked} shards symlinked, {int(elapsed)}s")
    print(f"Output: {output}")
    print("=" * 60)


if __name__ == "__main__":
    main()

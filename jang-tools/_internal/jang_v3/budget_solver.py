"""JANG v3 Phase 3: Bit-budget solver.

Operates on **logical groups** (one entry per fused module — e.g. all 256
routed experts of layer L's `switch_mlp.gate_proj` are one group), as
produced by `jang_v3.consolidate`.

Inputs:
  - logical groups JSON: {name: {shape_per_unit, n_units, disk_names}}
  - importance JSON:     {name: float}  (Phase 2 output, mlx-lm namespace)
  - budget bytes
  - optional fixed_bits override

Output: {logical_name: bits} — one bit choice per group; the encoder fans
this out to every disk tensor in `disk_names`.

Algorithm:
  1. Start every group at the LOWEST-considered tier.
  2. Consider every legal upgrade (group, current_bits → next_bits).
  3. priority = importance_per_unit × bytes_saved_avoided_per_dollar
              = importance × (MSE@cur - MSE@next) / (bytes@next - bytes@cur)
     (importance is per-unit; we apply it to *each* unit in the group via
      n_units multiplier so that big groups still need real upside to win)
  4. Greedily pop best, bump tier, repeat until budget exhausted or every
     group at max tier.

The greedy is provably within (1+ε) of LP optimum for diminishing-returns
MSE-vs-bits curves.

Usage:
    python -m jang_tools.jang_v3.budget_solver \
        --groups /tmp/dsv4_logical_groups.json \
        --importance /tmp/importance_v3.json \
        --budget-gb 90 \
        --output /tmp/bit_plan_v3.json
"""
import argparse
import json
from pathlib import Path

# Relative MSE per bit (uniform-quant baseline; outliers handled by AWQ).
# Each extra bit roughly halves quant variance.
_REL_MSE = {2: 1.000, 3: 0.450, 4: 0.200, 5: 0.090, 6: 0.040, 8: 0.008, 16: 0.0}
# Tier ladder (low → high). 16 reserved for direct passthrough on tiny
# tensors that aren't worth quantizing (e.g. routing gate, hc_*).
_TIERS = [2, 3, 4, 6, 8]

# Group-name patterns that get DEFAULT importance when calib didn't see
# them. We treat them roughly per their structural role.
_DEFAULT_IMP = {
    "embed_tokens":      150.0,   # vocabulary lookup — keep accurate
    "lm_head":           150.0,   # final logits
    "self_attn.compressor":  20.0,   # long-ctx pool projection
    "self_attn.indexer":     30.0,   # long-ctx top-k selection
    "self_attn.wo_a":        50.0,   # attn output low-rank
    "self_attn.wq_a":        25.0,   # attn query low-rank
    "self_attn.wq_b":        40.0,
    "self_attn._wq_a_kv_fused": 35.0,
    "self_attn.wkv":         60.0,   # KV joint projection
    "self_attn.wo_b":        80.0,   # attn output
    "mtp.":              50.0,   # MTP path (used for spec dec)
    "mlp.gate":              60.0,   # router decision
    "shared_experts":        50.0,
    "switch_mlp":            40.0,
    "e_proj":                40.0,
    "h_proj":                40.0,
}


def _default_imp(name: str) -> float:
    for pat, v in _DEFAULT_IMP.items():
        if pat in name:
            return v
    return 1.0


# Patterns that are rarely on the critical path — fix them at the floor.
# Indexer/Compressor only fire under VMLX_DSV4_LONG_CTX, and MTP is the
# speculative-decode head (currently dead weight). Don't waste upgrade
# budget on them.
_FIX_AT_FLOOR_PATTERNS = (
    "self_attn.indexer",
    "self_attn.compressor",
    "mtp.",
)


def _is_pinned_floor(name: str) -> bool:
    return any(p in name for p in _FIX_AT_FLOOR_PATTERNS)


# Activation-frequency multiplier — token-fraction times this tensor is
# touched on the critical decode path. Routed experts only fire for top-K
# of N selected, so per-tensor expected hit rate is K/N. Indexer/MTP are
# 0 above so don't matter (already pinned).
_ACTIVATION_FREQ = {
    # routed: top-8 of 256 experts on Flash → 8/256 = 0.031 hit per tensor
    "switch_mlp": 0.031,
    # everything else: always-on (attn, shared, gate, embed/head)
}


def _activation_freq(name: str) -> float:
    for pat, f in _ACTIVATION_FREQ.items():
        if pat in name:
            return f
    return 1.0


def _group_size_for(name: str, routed_gs: int, default_gs: int) -> int:
    """Routed experts get the fatter group size to amortize overhead;
    attention/embed/etc stay at default for accuracy."""
    if "switch_mlp" in name:
        return routed_gs
    return default_gs


def _bytes_per_group(shape: list[int], n_units: int, bits: int,
                     group_size: int = 32) -> int:
    """Approximate on-disk bytes of an affine-quantized weight group.

    bits-per-element + 16-bit scale + 16-bit bias per group_size cluster.
    For passthrough (16-bit), no scales/biases.
    """
    if not shape:
        return 0
    n_per_unit = 1
    for d in shape:
        n_per_unit *= d
    if bits == 16:
        return int(n_per_unit * n_units * 2)
    in_dim = shape[-1]
    n_groups_per_row = max(1, in_dim // group_size)
    other = n_per_unit // in_dim
    weight_b = n_per_unit * n_units * bits / 8
    overhead_b = other * n_groups_per_row * n_units * 4  # scale + bias = 4 bytes
    return int(weight_b + overhead_b)


def solve(groups: dict[str, dict],
          importance: dict[str, float],
          budget_bytes: int,
          start_bits: int = 2,
          tiers: list[int] | None = None,
          fixed_bits: dict[str, int] | None = None,
          floor_bits: dict[str, int] | None = None,
          routed_gs: int = 64,
          default_gs: int = 32) -> dict[str, int]:
    """floor_bits: pattern → min bits. e.g. {"embed_tokens": 4, "lm_head": 4,
    "mlp.gate": 4} sets baseline above start_bits for those tensors before
    upgrade selection runs."""
    fixed_bits = fixed_bits or {}
    floor_bits = floor_bits or {}
    tiers = tiers or _TIERS

    def _floor_for(name: str) -> int:
        # Use endswith() for precision: pattern must END the logical name.
        # Avoids "mlp.gate" matching "mlp.shared_experts.gate_proj".
        best = start_bits
        for pat, b in floor_bits.items():
            if name.endswith(pat) or name == pat:
                if b > best:
                    best = b
        return best

    plan = {name: _floor_for(name) for name in groups}
    plan.update(fixed_bits)

    def gbytes(name: str, bits: int) -> int:
        g = groups[name]
        gs = _group_size_for(name, routed_gs, default_gs)
        return _bytes_per_group(g["shape_per_unit"], g["n_units"], bits, gs)

    total = sum(gbytes(n, plan[n]) for n in groups)
    base_gb = total / 1e9
    print(f"  baseline (start={start_bits}-bit + floors): {base_gb:.2f} GB")
    print(f"  budget:                          {budget_bytes/1e9:.2f} GB")
    print(f"  headroom:                        {(budget_bytes-total)/1e9:.2f} GB")

    if total >= budget_bytes:
        print(f"  ⚠ baseline exceeds budget; returning all-{start_bits}")
        return plan

    # Routed experts can only use (2, 4)-bit MXTQ in the current production
    # DSV4 JANGTQ path. 8-bit affine routed layers would require affine
    # per-expert prestack + post-sanitize QuantizedSwitchLinear hydration and
    # live mixed-codec DSV4 gates, none of which is proven yet.
    routed_tiers = [b for b in tiers if b in (2, 4)]

    upgrades = 0
    while True:
        best = None
        for name in groups:
            if name in fixed_bits:
                continue
            if _is_pinned_floor(name):
                continue
            cur = plan[name]
            is_routed = ".mlp.switch_mlp." in name or name.endswith(".SWITCH_MLP_LAYER")
            t_list = routed_tiers if is_routed else tiers
            if cur not in t_list:
                continue
            ti = t_list.index(cur)
            if ti >= len(t_list) - 1:
                continue
            nxt = t_list[ti + 1]
            mse_red = _REL_MSE[cur] - _REL_MSE[nxt]
            cost = gbytes(name, nxt) - gbytes(name, cur)
            if cost <= 0 or mse_red <= 0:
                continue
            imp = importance.get(name, _default_imp(name))
            # Multiply by n_units × activation_freq so each tensor's
            # priority reflects its expected per-token contribution.
            n_units = groups[name]["n_units"]
            freq = _activation_freq(name)
            priority = imp * n_units * freq * mse_red / cost
            if best is None or priority > best[0]:
                best = (priority, name, nxt, cost)
        if best is None:
            break
        prio, name, nxt, cost = best
        if total + cost > budget_bytes:
            break
        plan[name] = nxt
        total += cost
        upgrades += 1

    # Post-process: enforce same-bit across (layer.X.switch_mlp.{gate,down,up}_proj).
    # The fused TQ decode kernel + mlx-lm gather_qmm cannot mix formats within
    # one layer's MoE expert tuple — they all need the same codec/bits.
    # If any of the 3 was upgraded, lift the other two to match (max wins).
    import re as _re
    layer_coupled = 0
    layer_buckets: dict[str, list[str]] = {}
    for name in plan:
        m = _re.match(r"^(.+\.mlp\.switch_mlp)\.(gate|down|up)_proj$", name)
        if m:
            layer_buckets.setdefault(m.group(1), []).append(name)
    for prefix, members in layer_buckets.items():
        if len(members) <= 1:
            continue
        max_bits = max(plan[n] for n in members)
        for n in members:
            if plan[n] != max_bits:
                plan[n] = max_bits
                layer_coupled += 1
    if layer_coupled:
        # Recompute total size after coupling
        total = sum(gbytes(n, plan[n]) for n in plan if n in groups)
        print(f"  layer-coupling lifted {layer_coupled} routed projs to match siblings")

    print(f"  upgrades:                        {upgrades}")
    print(f"  final size:                      {total/1e9:.2f} GB")
    dist = {}
    for b in plan.values():
        dist[b] = dist.get(b, 0) + 1
    print(f"  bit distribution:")
    for b in sorted(dist):
        print(f"    {b}-bit: {dist[b]:>5d} groups")
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    ap.add_argument("--importance", required=True)
    ap.add_argument("--budget-gb", type=float, required=True)
    ap.add_argument("--start-bits", type=int, default=2)
    ap.add_argument("--routed-gs", type=int, default=64,
                    help="group_size for routed_experts switch_mlp tensors")
    ap.add_argument("--default-gs", type=int, default=32,
                    help="group_size for everything else")
    ap.add_argument("--floor", action="append", default=[],
                    help="pattern=bits floor, e.g. --floor embed_tokens=4 (repeatable)")
    ap.add_argument("--hash-bits", type=int, default=4,
                    help="Min bits for first num-hash-layers routed experts (DSV4 has 3 hash-routed layers)")
    ap.add_argument("--num-hash-layers", type=int, default=3)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    floor_bits = {}
    for spec in args.floor:
        pat, b = spec.split("=", 1)
        floor_bits[pat] = int(b)

    groups = json.loads(Path(args.groups).read_text())
    importance = json.loads(Path(args.importance).read_text())
    budget = int(args.budget_gb * 1e9)

    print(f"=== JANG v3 budget solver ===")
    print(f"  groups:        {len(groups)}")
    print(f"  importance:    {len(importance)}")
    print(f"  budget:        {args.budget_gb} GB")

    # Pre-coalesce routed projs: the runtime needs all 3 projs of one layer
    # at the same bit width (fused TQ kernel + gather_qmm dispatch can't
    # mix MXTQ codec across gate/up/down). Treat (layer.X.switch_mlp.gate_proj,
    # .down_proj, .up_proj) as a single virtual "switch_mlp.layer_X" group
    # whose shape sums all 3 projs and importance is max of the 3 (so the
    # most-sensitive proj drives the upgrade decision).
    import re as _re_pre
    routed_pat = _re_pre.compile(r"^(.+\.mlp\.switch_mlp)\.(gate|down|up)_proj$")
    routed_layers: dict[str, list[str]] = {}
    for n in list(groups.keys()):
        m = routed_pat.match(n)
        if m:
            routed_layers.setdefault(m.group(1), []).append(n)
    coalesced = 0
    for prefix, members in routed_layers.items():
        if len(members) <= 1:
            continue
        # Sum total elements across the 3 projs (per unit). For DSV4:
        #   gate_proj: shape [intermediate=2048, hidden=4096] = 8.4M elem
        #   up_proj:   shape [intermediate=2048, hidden=4096] = 8.4M elem
        #   down_proj: shape [hidden=4096, intermediate=2048] = 8.4M elem
        #   Total per expert: 3 × 8.4M = 25.2M
        # Encode as a synthetic shape that gives the right element count and
        # the right "in_dim" for group_size overhead computation. Use the
        # most common in_dim across the 3 projs (gate/up=hidden, down=interm)
        # weighted by element count — they're typically equal so just use the
        # gate_proj's in_dim. Approximate.
        first = groups[members[0]]
        n_units = first["n_units"]
        total_elem_per_unit = 0
        for m in members:
            sh = groups[m]["shape_per_unit"]
            n = 1
            for d in sh:
                n *= d
            total_elem_per_unit += n
        in0 = first["shape_per_unit"][-1]  # gate_proj in_dim (= hidden)
        out_combined = total_elem_per_unit // in0  # synthetic out dim
        combined_name = f"{prefix}.SWITCH_MLP_LAYER"
        combined_disk = []
        for m in members:
            combined_disk.extend(groups[m]["disk_names"])
        groups[combined_name] = {
            "shape_per_unit": [out_combined, in0],
            "n_units": n_units,
            "disk_names": combined_disk,
            "_routed_layer_members": members,
        }
        # Aggregate importance across the 3 projs (max)
        max_imp = max(importance.get(m, 0) for m in members)
        importance[combined_name] = max_imp
        # Remove individual members
        for m in members:
            del groups[m]
            if m in importance:
                del importance[m]
        coalesced += 1
    if coalesced:
        print(f"  routed-layer coalesce: {coalesced} layers (3 projs → 1 unit)")

    # MTP exclusion: the DSV4 converter skips all `mtp.*` source tensors
    # (`weight_keys = [k for k in idx.keys if not k.startswith("mtp.")]`),
    # so any plan entries for MTP groups are no-ops at build time. Drop
    # them from the input groups dict so the solver doesn't waste budget
    # picking upgrades that won't get applied.
    mtp_groups = [n for n in groups if n.startswith("mtp.")]
    for n in mtp_groups:
        del groups[n]
    if mtp_groups:
        print(f"  mtp dropped: {len(mtp_groups)} groups (converter excludes mtp.* tensors)")

    # Hash-routed layers (first num_hash_layers) need higher bits — DSV4's
    # deterministic hash routing means quant noise compounds harder there.
    # After routed-layer coalescing, hash floor uses the SWITCH_MLP_LAYER key.
    hash_floor: dict[str, int] = {}
    if args.hash_bits and args.num_hash_layers:
        for L in range(args.num_hash_layers):
            hash_floor[f"model.layers.{L}.mlp.switch_mlp.SWITCH_MLP_LAYER"] = args.hash_bits
    fixed_bits_combined = dict(hash_floor)
    if hash_floor:
        print(f"  hash-routed floor: {len(hash_floor)} layer-groups at {args.hash_bits}-bit (first {args.num_hash_layers} layers)")

    plan = solve(groups, importance, budget, start_bits=args.start_bits,
                 routed_gs=args.routed_gs, default_gs=args.default_gs,
                 floor_bits=floor_bits,
                 fixed_bits=fixed_bits_combined)
    out = {
        "plan": plan,
        "config": {
            "budget_gb": args.budget_gb,
            "start_bits": args.start_bits,
            "routed_gs": args.routed_gs,
            "default_gs": args.default_gs,
            "floor_bits": floor_bits,
        },
    }
    Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nWrote bit plan to {args.output}")

    # Sample by tier
    by_tier = {}
    for n, b in plan.items():
        by_tier.setdefault(b, []).append(n)
    print(f"\nSample plans per tier:")
    for b in sorted(by_tier):
        names = by_tier[b]
        print(f"  {b}-bit ({len(names)} groups):")
        for n in sorted(names)[:3]:
            print(f"    {n}")
        if len(names) > 3:
            print(f"    ... and {len(names)-3} more")


if __name__ == "__main__":
    main()

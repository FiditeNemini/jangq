"""JANG v3 Phase 4: Encoder integration.

Reads a bit plan JSON (`{plan: {logical_name: bits}, config: {...}}`) and
exposes a `lookup(disk_tensor_name)` that maps each on-disk source tensor
to its target (bits, method, group_size). The DSV4 converter calls this
when env `DSV4_V3_PLAN_PATH` is set.

This module also has a thin CLI wrapper that:
  - validates the plan against the source DSV4 model
  - sets DSV4_V3_PLAN_PATH + FORMAT=jangtq
  - invokes convert_dsv4_jangtq with the right output dir suffix

Usage (via converter, env-driven):
    DSV4_V3_PLAN_PATH=/tmp/bit_plan_v3_95gb.json FORMAT=jangtq \
        python3 -m jang_tools.dsv4.convert_dsv4_jangtq \
        --src /Users/eric/sources/DeepSeek-V4-Flash \
        --dst /tmp/DSV4-Flash-JANG_V3_95gb \
        --profile 2 --variant V3

Usage (direct CLI wrapper):
    python3 -m jang_tools.jang_v3.encode \
        --plan /tmp/bit_plan_v3_95gb.json \
        --src /Users/eric/sources/DeepSeek-V4-Flash \
        --dst /tmp/DSV4-Flash-JANG_V3_95gb
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


_PLAN_CACHE: dict | None = None
_CONFIG_CACHE: dict | None = None


def _load_plan() -> tuple[dict, dict] | tuple[None, None]:
    global _PLAN_CACHE, _CONFIG_CACHE
    if _PLAN_CACHE is not None:
        return _PLAN_CACHE, _CONFIG_CACHE
    path = os.environ.get("DSV4_V3_PLAN_PATH", "").strip()
    if not path:
        return None, None
    data = json.loads(Path(path).read_text())
    if "plan" in data and isinstance(data["plan"], dict):
        _PLAN_CACHE = data["plan"]
        _CONFIG_CACHE = data.get("config", {})
    else:
        # backwards-compat: bare {name: bits}
        _PLAN_CACHE = data
        _CONFIG_CACHE = {}
    return _PLAN_CACHE, _CONFIG_CACHE


def _logical_from_source(n: str) -> Optional[str]:
    """Mirror of jang_v3.consolidate._logical_name — kept inline so the
    converter doesn't need to import consolidate (avoid heavy imports
    inside the per-tensor loop)."""
    if n.endswith((".scales", ".biases", ".scale")):
        return None
    if "_norm." in n or n.endswith("norm.weight"):
        return None
    if n.endswith((".attn_sink", ".bias", ".ape", ".tid2eid")):
        return None
    if n.endswith((".weight",)) is False:
        return None
    if "hc_" in n and n.endswith(("_base", "_fn", "_scale")):
        return None

    m = re.match(r"^mtp\.(\d+)\.(.+)$", n)
    if m:
        midx, sub = m.groups()
        prefix = f"mtp.{midx}"
        rest = sub
    else:
        m = re.match(r"^layers\.(\d+)\.(.+)$", n)
        if m:
            lidx, rest = m.groups()
            prefix = f"model.layers.{lidx}"
        else:
            if n == "embed.weight":
                return "model.embed_tokens"
            if n == "head.weight":
                return "lm_head"
            return None

    m = re.match(r"^ffn\.experts\.\d+\.(w[123])\.weight$", rest)
    if m:
        proj = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}[m.group(1)]
        return f"{prefix}.mlp.switch_mlp.{proj}"
    m = re.match(r"^ffn\.shared_experts\.(w[123])\.weight$", rest)
    if m:
        proj = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}[m.group(1)]
        return f"{prefix}.mlp.shared_experts.{proj}"
    if rest == "ffn.gate.weight":
        return f"{prefix}.mlp.gate"
    m = re.match(r"^attn\.(wq_a|wq_b|wkv|wo_a|wo_b)\.weight$", rest)
    if m:
        sub = m.group(1)
        if sub in ("wq_a", "wkv"):
            return f"{prefix}.self_attn._wq_a_kv_fused"
        return f"{prefix}.self_attn.{sub}"
    m = re.match(r"^attn\.indexer\.(wq_b|weights_proj)\.weight$", rest)
    if m:
        return f"{prefix}.self_attn.indexer.{m.group(1)}"
    m = re.match(r"^attn\.indexer\.compressor\.(wkv|wgate)\.weight$", rest)
    if m:
        return f"{prefix}.self_attn.indexer.compressor.{m.group(1)}"
    m = re.match(r"^attn\.compressor\.(wkv|wgate)\.weight$", rest)
    if m:
        return f"{prefix}.self_attn.compressor.{m.group(1)}"
    if rest in ("e_proj.weight", "h_proj.weight"):
        return f"{prefix}.{rest.replace('.weight','')}"
    return None


def lookup(n: str) -> Optional[tuple[int, str, int]]:
    """Return (bits, method, group_size) for source tensor name `n`, or
    None if the v3 plan doesn't apply (caller should fall through to the
    default classify()).

    Defers to default classify() for tensors that the runtime expects at
    fp16 regardless of plan: router gate (mlp.gate.weight without
    "experts" — runtime uses raw `.T` matmul, can't dispatch quantized).
    """
    plan, cfg = _load_plan()
    if plan is None:
        return None
    logical = _logical_from_source(n)
    if logical is None:
        return None
    # Coalesced layer-routed key: solver may emit one entry per layer (3 projs
    # share bits). Plan key is `<prefix>.mlp.switch_mlp.SWITCH_MLP_LAYER`;
    # convert per-proj logical name to the coalesced key for lookup.
    if ".mlp.switch_mlp." in logical and (logical.endswith(".gate_proj")
                                          or logical.endswith(".down_proj")
                                          or logical.endswith(".up_proj")):
        coalesced_key = logical.rsplit(".", 1)[0] + ".SWITCH_MLP_LAYER"
        if coalesced_key in plan:
            logical = coalesced_key
    # Router gate.weight: runtime path in dsv4/mlx_model.py does
    # `x @ self.weight.T` directly — not through QuantizedLinear. Forcing
    # it to fp16 passthrough is correct for any v3 plan; quantizing the
    # router would crash with a shape mismatch on first forward.
    # Match the converter's existing rule: ".gate.weight" without "experts".
    if n.endswith(".gate.weight") and "experts" not in n:
        return 16, "passthrough", 0
    bits = plan.get(logical)
    if bits is None:
        return None
    cfg = cfg or {}
    routed_gs = int(cfg.get("routed_gs", 64))
    default_gs = int(cfg.get("default_gs", 32))
    if bits == 16:
        return 16, "passthrough", 0
    gs = routed_gs if "switch_mlp" in logical else default_gs
    # Routed experts: only (2, 4)-bit MXTQ is production-proven in the current
    # DSV4 JANGTQ prestack/runtime path. 8-bit affine routed layers need a
    # separate affine-routed prestack + hydration proof before this helper can
    # advertise them.
    if "switch_mlp" in logical:
        if bits in (2, 4):
            return bits, "mxtq", 0
        raise ValueError(
            f"DSV4 V3 affine routed plan is not currently supported for "
            f"{logical}: requested {bits}-bit. Use routed 2/4-bit MXTQ "
            "or implement/prove affine-routed prestack/runtime support."
        )
    return bits, "affine", gs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="Bit plan JSON from budget_solver")
    ap.add_argument("--src", required=True, help="Source DSV4 model dir")
    ap.add_argument("--dst", required=True, help="Output bundle dir")
    ap.add_argument("--profile", type=int, default=2,
                    help="Base routed-expert bit count passed to convert_dsv4_jangtq")
    ap.add_argument("--awq-norms", default=None,
                    help="AWQ activation norms (optional)")
    ap.add_argument("--awq-alpha", type=float, default=0.25)
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate plan + show stats only")
    args = ap.parse_args()

    plan_data = json.loads(Path(args.plan).read_text())
    if "plan" in plan_data:
        plan = plan_data["plan"]
        cfg = plan_data.get("config", {})
    else:
        plan = plan_data
        cfg = {}

    print(f"=== JANG v3 encode ===")
    print(f"  plan:    {args.plan}  ({len(plan)} groups)")
    print(f"  src:     {args.src}")
    print(f"  dst:     {args.dst}")
    print(f"  config:  {cfg}")
    dist = {}
    for b in plan.values():
        dist[b] = dist.get(b, 0) + 1
    print(f"  bit dist:")
    for b in sorted(dist):
        print(f"    {b}-bit: {dist[b]} groups")

    if args.dry_run:
        return

    env = os.environ.copy()
    env["DSV4_V3_PLAN_PATH"] = str(Path(args.plan).resolve())
    env["FORMAT"] = "jangtq"
    cmd = [
        sys.executable, "-m", "jang_tools.dsv4.convert_dsv4_jangtq",
        "--src", str(args.src),
        "--dst", str(args.dst),
        "--profile", str(args.profile),
        "--variant", "V3",
    ]
    if args.awq_norms:
        cmd += ["--awq-norms", args.awq_norms,
                    "--awq-alpha", str(args.awq_alpha)]
    print(f"\n  invoking: {' '.join(cmd)}")
    print(f"  with env DSV4_V3_PLAN_PATH={env['DSV4_V3_PLAN_PATH']}")
    res = subprocess.run(cmd, env=env)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()

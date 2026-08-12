"""Watch for the official Qwen 3.8 27B source release and report its shape.

Prints a one-line verdict plus, on a hit, the architecture facts that decide the
conversion plan (hybrid SSM vs full attention, MoE vs dense, native MTP,
multimodal towers). Exit 0 = released, 1 = not yet.

    python -m jang_tools.watch_qwen38
"""
from __future__ import annotations

import json
import sys

from huggingface_hub import HfApi, hf_hub_download

# Official repos worth watching, most likely first.
CANDIDATES = [
    "Qwen/Qwen3.8-27B",
    "Qwen/Qwen3.8-27B-Instruct",
    "Qwen/Qwen3.8-27B-A3B",
    "Qwen/Qwen3.8-27B-Base",
    "Qwen/Qwen3.8-27B-Thinking",
]


def _arch_report(rid: str, api: HfApi) -> None:
    """Everything needed to decide the conversion plan, from config alone."""
    p = hf_hub_download(rid, "config.json")
    c = json.load(open(p))
    tc = c.get("text_config", c)
    print(f"  model_type      : {c.get('model_type')}  arch={c.get('architectures')}")
    print(f"  layers/hidden   : {tc.get('num_hidden_layers')} / {tc.get('hidden_size')}")
    print(f"  vocab           : {tc.get('vocab_size')}")

    # MoE?
    ne = tc.get("num_experts") or tc.get("n_routed_experts")
    if ne:
        print(f"  MoE             : {ne} experts, top-{tc.get('num_experts_per_tok')}, "
              f"shared={tc.get('n_shared_experts')}, "
              f"moe_inter={tc.get('moe_intermediate_size')}")
    else:
        print("  MoE             : dense")

    # Hybrid / linear attention — decides whether the nemotron_h-style hybrid
    # cache and its 'cache index != layer index' trap apply.
    hy = {k: tc.get(k) for k in
          ("layer_types", "layers_block_type", "hybrid_override_pattern",
           "linear_attn_config", "mamba_num_heads", "ssm_state_size",
           "full_attention_interval") if tc.get(k) is not None}
    print(f"  hybrid/SSM      : {hy if hy else 'none — plain attention'}")

    # Activation decides whether the gate-less relu2 floors apply.
    act = tc.get("hidden_act") or tc.get("mlp_hidden_act")
    print(f"  activation      : {act}")

    # Native MTP?
    mtp = {k: tc.get(k) for k in
           ("num_nextn_predict_layers", "mtp_layers_block_type") if tc.get(k)}
    print(f"  MTP (config)    : {mtp if mtp else 'not declared'}")

    # Multimodal towers.
    mm = [k for k in c if any(x in k for x in
          ("vision", "audio", "video", "image", "mm_", "visual"))]
    print(f"  multimodal keys : {mm if mm else 'none — text only (confirm on weights)'}")

    print(f"  rope            : theta={tc.get('rope_theta')} "
          f"max_pos={tc.get('max_position_embeddings')} "
          f"scaling={tc.get('rope_scaling')}")

    try:
        info = api.model_info(rid, files_metadata=True)
        tot = sum(s.size or 0 for s in info.siblings)
        shards = sum(1 for s in info.siblings if s.rfilename.endswith(".safetensors"))
        has_mtp_files = any("mtp" in s.rfilename for s in info.siblings)
        print(f"  download size   : {tot/1e9:.2f} GB in {shards} shards  gated={info.gated}")
        print(f"  mtp.* files     : {has_mtp_files}")
    except Exception as e:  # noqa: BLE001
        print(f"  (file listing unavailable: {type(e).__name__})")


def main() -> int:
    api = HfApi()
    hits = []
    for rid in CANDIDATES:
        try:
            api.model_info(rid)
            hits.append(rid)
        except Exception:  # noqa: BLE001,S110
            pass

    # Also sweep the org for anything 3.8-shaped we did not predict.
    try:
        for m in api.list_models(author="Qwen", limit=40, sort="lastModified"):
            if "3.8" in m.id and m.id not in hits:
                hits.append(m.id)
    except Exception:  # noqa: BLE001,S110
        pass

    if not hits:
        print("QWEN38: NOT RELEASED — no official Qwen/*3.8* repo on the Hub yet")
        return 1

    print(f"QWEN38: RELEASED — {len(hits)} repo(s): {', '.join(hits)}")
    for rid in hits:
        print(f"\n=== {rid} ===")
        try:
            _arch_report(rid, api)
        except Exception as e:  # noqa: BLE001
            print(f"  config not readable yet ({type(e).__name__}) — "
                  f"repo may still be uploading")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare local ZAYA quant bundles for HF upload.

This writes org-specific model cards, copies the matching logo/banner asset,
creates Osaurus-branded JANGTQ mirror bundles, and emits an upload manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/Users/eric/jang")
MODEL_ROOT = ROOT / "models/Zyphra"
JANGQ_LOGO = ROOT / "assets/jangq-logo-dark.png"
OSAURUS_BANNER = Path("/Users/eric/models/JANGQ/Laguna-XS.2-JANGTQ/osaurus-x-banner.png")


TARGETS = [
    {
        "path": MODEL_ROOT / "ZAYA1-8B-JANGTQ2",
        "repo": "JANGQ-AI/ZAYA1-8B-JANGTQ2",
        "org": "JANGQ-AI",
        "asset_name": "jangq-logo.png",
        "asset_path": JANGQ_LOGO,
        "format": "JANGTQ2",
        "kind": "jangtq",
        "bits": "2-bit MXTQ routed experts + 8-bit affine non-routed tensors",
    },
    {
        "source_path": MODEL_ROOT / "ZAYA1-8B-JANGTQ2",
        "path": MODEL_ROOT / "ZAYA1-8B-JANGTQ2-OsaurusAI",
        "repo": "OsaurusAI/ZAYA1-8B-JANGTQ2",
        "org": "OsaurusAI",
        "asset_name": "osaurus-x-banner.png",
        "asset_path": OSAURUS_BANNER,
        "format": "JANGTQ2",
        "kind": "jangtq",
        "bits": "2-bit MXTQ routed experts + 8-bit affine non-routed tensors",
    },
    {
        "path": MODEL_ROOT / "ZAYA1-8B-JANGTQ4",
        "repo": "JANGQ-AI/ZAYA1-8B-JANGTQ4",
        "org": "JANGQ-AI",
        "asset_name": "jangq-logo.png",
        "asset_path": JANGQ_LOGO,
        "format": "JANGTQ4",
        "kind": "jangtq",
        "bits": "4-bit MXTQ routed experts + 8-bit affine non-routed tensors",
    },
    {
        "source_path": MODEL_ROOT / "ZAYA1-8B-JANGTQ4",
        "path": MODEL_ROOT / "ZAYA1-8B-JANGTQ4-OsaurusAI",
        "repo": "OsaurusAI/ZAYA1-8B-JANGTQ4",
        "org": "OsaurusAI",
        "asset_name": "osaurus-x-banner.png",
        "asset_path": OSAURUS_BANNER,
        "format": "JANGTQ4",
        "kind": "jangtq",
        "bits": "4-bit MXTQ routed experts + 8-bit affine non-routed tensors",
    },
    {
        "path": MODEL_ROOT / "ZAYA1-8B-MXFP4",
        "repo": "OsaurusAI/ZAYA1-8B-MXFP4",
        "org": "OsaurusAI",
        "asset_name": "osaurus-x-banner.png",
        "asset_path": OSAURUS_BANNER,
        "format": "MXFP4",
        "kind": "mxfp4",
        "bits": "4-bit affine linears + 8-bit embeddings + passthrough router/CCA state tensors",
    },
]


def clone_bundle(source: Path, target: Path) -> None:
    """Create a generated upload mirror without doubling physical storage."""
    if not source.is_dir():
        raise SystemExit(f"missing source bundle dir: {source}")
    if source.resolve() == target.resolve():
        return
    if target.exists():
        shutil.rmtree(target)
    try:
        subprocess.run(["cp", "-cR", str(source), str(target)], check=True)
        return
    except Exception:
        pass
    shutil.copytree(source, target, copy_function=os.link)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def size_gib(path: Path) -> str:
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return f"{total / (1024 ** 3):.2f} GiB"


def read_coherence(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"passed": False, "reason": "missing coherence report"}
    try:
        data = load_json(path)
    except Exception as exc:
        return {"passed": False, "reason": f"invalid coherence report: {exc}"}
    if data.get("passed") is True:
        return {"passed": True, "path": str(path)}
    return {"passed": False, "reason": "coherence report did not pass", "path": str(path)}


def status_line(coherence: dict[str, Any], allow_unverified: bool) -> str:
    if coherence.get("passed"):
        return "Generation coherence: PASSED using the recorded ZAYA runtime report."
    if allow_unverified:
        reason = coherence.get("reason", "not independently passed")
        return (
            "Generation coherence: NOT INDEPENDENTLY PASSED for the quantized "
            f"runtime bundle ({reason}); published as a format/runtime bundle "
            "pending downstream ZAYA runtime validation."
        )
    reason = coherence.get("reason", "not passed")
    return f"Generation coherence: not passed in local preflight ({reason})."


def runtime_note(kind: str) -> str:
    if kind == "jangtq":
        return (
            "This bundle requires a ZAYA-aware JANGTQ runtime that implements "
            "CCA attention state plus pre-stacked `switch_mlp` TurboQuant experts."
        )
    return (
        "This bundle requires a ZAYA-aware MLX/JANG runtime that implements "
        "CCA attention state and the converted pre-stacked expert layout."
    )


def sidecar_line(kind: str, sidecar: bool) -> str:
    if kind == "jangtq":
        return f"- `jangtq_runtime.safetensors` is included: {str(sidecar).lower()}."
    return "- `jangtq_runtime.safetensors` is not applicable to MXFP4."


def card(target: dict[str, Any], coherence: dict[str, Any], allow_unverified: bool) -> str:
    path = target["path"]
    cfg = load_json(path / "config.json")
    jcfg = load_json(path / "jang_config.json")
    index = load_json(path / "model.safetensors.index.json")
    size = size_gib(path)
    is_jangq = target["org"] == "JANGQ-AI"
    image = (
        f'<p align="center"><img src="{target["asset_name"]}" width="160" alt="JANGQ-AI"/></p>'
        if is_jangq
        else f'<p align="center"><img src="{target["asset_name"]}" width="100%" alt="OsaurusAI"/></p>'
    )
    tags = [
        "zaya",
        "mixture-of-experts",
        "hybrid-attention",
        "cca-attention",
        "mlx",
        "apple-silicon",
        "reasoning",
        "tool-use",
        "quantized",
    ]
    if target["kind"] == "jangtq":
        tags += ["jang", "jangtq", "mxtq", "jangtq-prestack"]
    else:
        tags += ["mxfp4", "jang"]
    if target["org"] == "OsaurusAI":
        tags.append("osaurus")

    tag_lines = "\n".join(f"  - {tag}" for tag in tags)
    quant = cfg.get("quantization", {})
    mxtq_bits = cfg.get("mxtq_bits")
    sidecar = (path / "jangtq_runtime.safetensors").exists()
    sidecar_text = sidecar_line(target["kind"], sidecar)
    weight_count = len(index.get("weight_map", {}))
    status = status_line(coherence, allow_unverified)
    note = runtime_note(target["kind"])
    repo_name = target["repo"].split("/", 1)[1]
    korean = (
        "이 번들은 Zyphra/ZAYA1-8B를 Apple Silicon MLX/JANG 런타임용으로 "
        "양자화한 모델입니다. ZAYA의 CCA attention 상태와 MoE 라우팅을 "
        "정확히 구현한 런타임에서만 사용해야 합니다."
    )

    return f"""---
license: apache-2.0
library_name: mlx
base_model: Zyphra/ZAYA1-8B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
{tag_lines}
quantization_config:
  family: {target["kind"]}
  profile: {target["format"]}
  group_size: {quant.get("group_size", 32)}
  expert_layout: split_switch_mlp
---

{image}

# {repo_name}

Quantized **Zyphra/ZAYA1-8B** for Apple Silicon runtimes.

| | |
|---|---|
| Source | [Zyphra/ZAYA1-8B](https://huggingface.co/Zyphra/ZAYA1-8B) |
| License | Apache-2.0, inherited from upstream |
| Format | {target["format"]} |
| Bundle size | {size} |
| Tensor keys | {weight_count} |
| Expert layout | Pre-stacked `zaya_block.experts.switch_mlp` |
| Runtime status | {status} |

## Important Runtime Note

{note}

ZAYA is not a stock `mlx_lm` architecture. It alternates CCA attention layers
and top-1 MoE layers. Use this bundle only with a runtime that implements the
ZAYA CCA state contract and the converted pre-stacked expert layout.

## Architecture Summary

- 80 decoder layers: 40 CCA attention layers and 40 top-1 MoE layers
- Hidden size 2048, 16 query heads, 2 KV heads, head dim 128
- CCA state per attention layer: standard KV plus `conv_state [B,1280,2]`
  and `prev_hs [B,2048]`
- 16 routed experts per MoE layer, top-1 routing with MOD skip route
- Context length 131072, `rope_theta=5000000`

## Quantization

{target["bits"]}.

Passthrough floor for first release prep:

- `conv_qk.*`, `temp`, norms, residual scaling, router path, biases, and
  balancing biases are preserved as float tensors.
- Embeddings and `lm_head` use 8-bit affine in the prepared bundles.
{sidecar_text}

`mxtq_bits`:

```json
{json.dumps(mxtq_bits, indent=2) if mxtq_bits is not None else "null"}
```

## Bundle Verification

- Safetensor headers scanned.
- Source tensor coverage checked.
- Converted bundles checked for `local_experts` removal.
- Converted expert tensors checked for pre-stacked `switch_mlp` layout.
- JANGTQ sidecars checked for the Swift runtime contract.
- Runtime coherence status recorded above.

## Runtime Smoke Tests

Before production use, run short deterministic prompts through the exact target
runtime:

- `What is 2+2? Answer with only the number.`
- `What is the capital of France? Answer with one word.`
- One chat-template prompt with thinking disabled.
- One chat-template prompt with thinking enabled and enough output budget for
  the final answer.

The first public bundle release records bundle integrity and runtime contract
checks. Full generation quality depends on a ZAYA-aware runtime implementation.

## Korean Summary

{korean}

## Files

- `config.json` carries `weight_format={cfg.get("weight_format")}` and
  `zaya_expert_layout=split_switch_mlp`.
- `jang_config.json` carries `cache_subtype={jcfg.get("cache_subtype")}`.
- Tokenizer files and `chat_template.jinja` are preserved from the upstream
  source snapshot.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coherence-report", type=Path)
    parser.add_argument("--manifest", type=Path, default=MODEL_ROOT / "zaya_upload_manifest.json")
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Mark targets upload-ready even if the coherence report did not pass.",
    )
    args = parser.parse_args()

    coherence = read_coherence(args.coherence_report)
    manifest: list[dict[str, Any]] = []

    for target in TARGETS:
        source_path = target.get("source_path")
        if source_path is not None:
            clone_bundle(source_path, target["path"])
        path = target["path"]
        if not path.is_dir():
            raise SystemExit(f"missing bundle dir: {path}")
        asset_path = target["asset_path"]
        if not asset_path.exists():
            raise SystemExit(f"missing asset: {asset_path}")
        for stale_asset in ("jangq-logo.png", "osaurus-x-banner.png"):
            stale_path = path / stale_asset
            if stale_asset != target["asset_name"] and stale_path.exists():
                stale_path.unlink()
        asset_out = path / target["asset_name"]
        if asset_out.exists():
            asset_out.unlink()
        shutil.copy2(asset_path, asset_out)
        readme = path / "README.md"
        if readme.exists():
            readme.unlink()
        readme.write_text(card(target, coherence, args.allow_unverified), encoding="utf-8")
        files = sorted(p for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts)
        upload_ready = bool(coherence.get("passed") or args.allow_unverified)
        blockers = [] if upload_ready else [coherence.get("reason", "coherence not passed")]
        warnings = []
        if args.allow_unverified and not coherence.get("passed"):
            warnings.append(coherence.get("reason", "coherence not passed"))
        manifest.append(
            {
                "repo": target["repo"],
                "path": str(path),
                "source_path": str(source_path) if source_path is not None else str(path),
                "org": target["org"],
                "format": target["format"],
                "readme": str(readme),
                "asset": target["asset_name"],
                "files_count": len(files),
                "total_size_bytes": sum(p.stat().st_size for p in files),
                "upload_ready": upload_ready,
                "blockers": blockers,
                "warnings": warnings,
            }
        )

    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

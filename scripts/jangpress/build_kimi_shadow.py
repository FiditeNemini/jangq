#!/usr/bin/env python3
"""Build a shadow directory of a Kimi-K2.6 JANGTQ bundle for vmlxctl JANGPress.

Two issues with Kimi-K2.6-* bundles + the current vmlxctl release:
  1. vMLX ModelDetector routes any bundle with a `text_config` wrapper
     to VLMModelFactory, but VLMModelFactory doesn't dispatch `kimi_k25`
     yet. (Fix: patch jang_config.has_vision = false → Tier-1
     authoritative LLM route.)
  2. After (1) routes to LLM, vMLX's `makeDeepseekV3OrJANGTQ` doesn't
     unwrap `text_config` → DeepseekV3JANGTQConfiguration's Codable
     fails because dims live inside text_config (not top-level), AND
     `model_type` reads as `kimi_k2` (the inner LLM type, not in the
     LLM factory's dispatch map). vmlx-swift-lm has the unwrap fix; vmlx
     does not. (Proper fix: port the unwrap; see
     `kimi_jangpress_agent_fix.patch`.)

This script's workaround for #2 — promote text_config keys to top-level
so the existing decoders see a flat config. Symlinks every other file
(safetensors shards, sidecar, tokenizer, chat_template) so the shadow
dir adds < 100 KB to disk.

Usage:
    python3 build_kimi_shadow.py <bundle_dir> [shadow_root]

Output:
    <shadow_root>/<bundle_name>/  (shadow_root defaults to /tmp/kimi-shadow)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def write_text_if_changed(path: Path, text: str) -> None:
    """Write text only when content changed, preserving mtime otherwise."""
    if path.exists() and path.read_text() == text:
        return
    path.write_text(text)


def symlink_if_changed(target: Path, source: Path) -> None:
    """Create target -> source without refreshing symlink mtime unnecessarily."""
    source = source.resolve()
    if target.is_symlink():
        try:
            if target.resolve() == source:
                return
        except FileNotFoundError:
            pass
        target.unlink()
    elif target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    os.symlink(source, target)


def patched_tokenizer_config(src: Path) -> dict | None:
    """Return a tokenizer_config.json payload with Kimi template fixes.

    Kimi-K2.6 bundles can ship the chat template as a sidecar
    chat_template.jinja. vMLX's Swift tokenizer loader expects the template
    inline in tokenizer_config.json. Also, the upstream Kimi template uses a
    `thinking` variable while vMLX's common request field is
    `enable_thinking`; add a tiny compatibility alias so either spelling
    renders the same chat/thinking mode.
    """
    tok_path = src / "tokenizer_config.json"
    if not tok_path.exists():
        return None

    tok = json.loads(tok_path.read_text())
    sidecar = src / "chat_template.jinja"
    template = tok.get("chat_template")
    if (not template or "include" in template.lower()) and sidecar.exists():
        template = sidecar.read_text()

    if isinstance(template, str) and "thinking is defined" in template:
        alias = (
            "{%- if thinking is not defined and enable_thinking is defined -%}"
            "{%- set thinking = enable_thinking -%}"
            "{%- endif -%}\n"
        )
        if alias not in template:
            marker = "{%- set preserve_thinking = preserve_thinking | default(false) -%}"
            if marker in template:
                template = template.replace(marker, alias + marker, 1)
            else:
                template = alias + template
        tok["chat_template"] = template

    return tok


def find_tokenizer_json(src: Path) -> Path | None:
    """Resolve tokenizer.json for Kimi bundles that omitted it.

    All current Kimi-K2.6 JANGTQ variants share the same tokenizer.json
    (verified by sha256 between Med and the Small prestack cache). Prefer
    an explicit env override, then the source bundle, then sibling/cached
    Kimi locations on this host.
    """
    candidates: list[Path] = []
    if env := os.environ.get("KIMI_TOKENIZER_JSON"):
        candidates.append(Path(env).expanduser())
    candidates.append(src / "tokenizer.json")

    parent = src.parent
    for name in [
        "Kimi-K2.6-Med-JANGTQ",
        "Kimi-K2.6-Small-JANGTQ",
        "Kimi-K2.6-Large-JANGTQ",
    ]:
        candidates.append(parent / name / "tokenizer.json")

    cache_roots = [
        Path.home() / "Library/Caches/vmlx-swift-lm/jangpress-prestack",
        Path.home() / "Library/Caches/vmlx/jangpress-prestack",
    ]
    for root in cache_roots:
        if root.exists():
            candidates.extend(root.glob("Kimi-K2.6-*-JANGTQ-*/tokenizer.json"))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def build_shadow(src: Path, dst: Path) -> Path:
    if not src.is_dir():
        sys.exit(f"FATAL: {src} is not a directory")
    cfg_path = src / "config.json"
    if not cfg_path.exists():
        sys.exit(f"FATAL: {cfg_path} missing")

    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(cfg_path.read_text())

    text_config = cfg.get("text_config")
    if text_config is None:
        sys.exit(
            f"FATAL: {src} has no `text_config` block — "
            "this script is for Kimi-K2.6 bundles only. Use the bundle "
            "directly with vmlxctl serve."
        )

    flat: dict = dict(text_config)
    for k in [
        "weight_format",
        "mxtq_bits",
        "mxtq_seed",
        "quantization",
        "routed_expert_bits",
        "group_size",
    ]:
        if k not in flat and k in cfg:
            flat[k] = cfg[k]
    # weight_format = "mxtq" is canonically in jang_config.json on
    # Kimi bundles (the converter writes it there, not config.json).
    # vMLX's FormatSniff reads from config.json — copy it across.
    jang_src = src / "jang_config.json"
    if jang_src.exists():
        jcfg_outer = json.loads(jang_src.read_text())
        if flat.get("weight_format") is None and jcfg_outer.get("weight_format"):
            flat["weight_format"] = jcfg_outer["weight_format"]
    flat["model_type"] = "kimi_k25"
    flat["architectures"] = ["KimiK25ForCausalLM"]
    flat.pop("auto_map", None)

    write_text_if_changed(dst / "config.json", json.dumps(flat, indent=2))
    print(f"[shadow] wrote flat config.json to {dst}")
    print(f"  model_type:    {flat.get('model_type')}")
    print(f"  architectures: {flat.get('architectures')}")
    print(f"  hidden_size:   {flat.get('hidden_size')}")
    print(f"  num_hidden_layers: {flat.get('num_hidden_layers')}")
    print(f"  weight_format: {flat.get('weight_format')}")
    print(f"  routed_expert_bits: {flat.get('routed_expert_bits')}")

    jang_path = dst / "jang_config.json"
    if (src / "jang_config.json").exists():
        jcfg = json.loads((src / "jang_config.json").read_text())
        jcfg["has_vision"] = False
        write_text_if_changed(jang_path, json.dumps(jcfg, indent=2))
        print(f"[shadow] wrote jang_config.json with has_vision=false")

    tok_cfg = patched_tokenizer_config(src)
    skip = {"config.json", "jang_config.json"}
    if tok_cfg is not None:
        write_text_if_changed(dst / "tokenizer_config.json", json.dumps(tok_cfg, indent=2))
        skip.add("tokenizer_config.json")
        tmpl_len = len(tok_cfg.get("chat_template") or "")
        print(f"[shadow] wrote tokenizer_config.json (chat_template_len={tmpl_len})")

    tokenizer_json = find_tokenizer_json(src)
    if tokenizer_json is None:
        sys.exit(
            "FATAL: tokenizer.json missing and no compatible Kimi tokenizer "
            "was found. Set KIMI_TOKENIZER_JSON=/path/to/tokenizer.json."
        )
    if tokenizer_json.parent != src:
        target = dst / "tokenizer.json"
        symlink_if_changed(target, tokenizer_json)
        skip.add("tokenizer.json")
        print(f"[shadow] linked tokenizer.json from {tokenizer_json}")

    linked = 0
    for entry in src.iterdir():
        if entry.name in skip:
            continue
        target = dst / entry.name
        symlink_if_changed(target, entry)
        linked += 1
    print(f"[shadow] symlinked {linked} files/dirs from source bundle")

    return dst


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = Path(sys.argv[1]).expanduser().resolve()
    shadow_root = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/kimi-shadow")
    dst = shadow_root / src.name
    print(f"[shadow] source: {src}")
    print(f"[shadow] target: {dst}")
    build_shadow(src, dst)
    print(f"\nShadow dir ready: {dst}")
    print("Use this path as --model in vmlxctl serve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

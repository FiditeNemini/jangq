"""Canonical family → parser/cache capability table for JANG / JANGTQ models.

This module owns the mapping from a model's family identifier (HF model_type
or architecture string) to the runtime hints vmlx's `CapabilityDetector`
reads at Tier-1: reasoning parser, tool parser, think-in-template flag,
cache type, modality, supports_tools/supports_thinking.

Cross-checked against:
  - vmlx `Sources/vMLXEngine/ModelCapabilities.swift::ModelTypeTable` (silver)
  - vmlx `Sources/vMLXEngine/Parsers/ParserRegistry.swift` (registered names)
  - vLLM upstream recipes (Qwen 3.5/3.6 → qwen3 + qwen3_coder)

Schemas accepted:
  jang_config["source_model"]["architecture"]      ← string  (JANGTQ converters)
  jang_config["architecture"]["type"]              ← string inside dict (convert.py)

Both shapes resolve through `build_capabilities`. Safe to re-run on already-
stamped artifacts (idempotent).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# (family, reasoning_parser, tool_parser, think_in_template, cache_type)
FAMILY_MAP: dict[str, tuple[str, str, str, bool, str]] = {
    # Qwen 3.5 / 3.6 family (hybrid SSM + attention)
    "qwen3_5":          ("qwen3_5",     "qwen3",       "qwen",     True,  "hybrid"),
    "qwen3_5_text":     ("qwen3_5",     "qwen3",       "qwen",     True,  "hybrid"),
    "qwen3_5_moe":      ("qwen3_5_moe", "qwen3",       "qwen",     True,  "hybrid"),
    "qwen3_5_moe_text": ("qwen3_5_moe", "qwen3",       "qwen",     True,  "hybrid"),
    "qwen3_next":       ("qwen3_next",  "qwen3",       "qwen",     True,  "hybrid"),
    "qwen3":            ("qwen3",       "qwen3",       "qwen",     True,  "kv"),
    # MiniMax M2.x
    "minimax_m2":       ("minimax_m2",  "qwen3",       "minimax",  True,  "kv"),
    "minimax_m2_5":     ("minimax_m2",  "qwen3",       "minimax",  True,  "kv"),
    "minimax":          ("minimax_m2",  "qwen3",       "minimax",  True,  "kv"),
    # GLM 5.x (MLA + DSA)
    "glm_moe_dsa":      ("glm5",        "deepseek_r1", "deepseek", True,  "mla"),
    "glm5":             ("glm5",        "deepseek_r1", "deepseek", True,  "mla"),
    # GLM 4 (dense + MoE, no MLA)
    "glm4":             ("glm4",        "deepseek_r1", "glm47",    False, "kv"),
    "glm4_moe":         ("glm4_moe",    "deepseek_r1", "glm47",    False, "kv"),
    # Nemotron (hybrid SSM)
    "nemotron_h":       ("nemotron_h",  "deepseek_r1", "nemotron", True,  "hybrid"),
    "nemotron_h_v2":    ("nemotron_h",  "deepseek_r1", "nemotron", True,  "hybrid"),
    # Mistral 4 (MLA)
    "mistral3":         ("mistral4",    "mistral",     "mistral",  False, "mla"),
    "mistral4":         ("mistral4",    "mistral",     "mistral",  False, "mla"),
    # Gemma 4 / 3
    "gemma4":           ("gemma4",      "gemma4",      "gemma4",   False, "kv"),
    "gemma4_text":      ("gemma4",      "gemma4",      "gemma4",   False, "kv"),
    "gemma3":           ("gemma4",      "deepseek_r1", "gemma4",   False, "kv"),
    "gemma3_text":      ("gemma4",      "deepseek_r1", "gemma4",   False, "kv"),
    "gemma3n":          ("gemma4",      "gemma4",      "gemma4",   False, "hybrid"),
    # Llama 3.x (dense) — base + instruct
    "llama":            ("llama",       None,          "llama",    False, "kv"),
    "llama3":           ("llama",       None,          "llama",    False, "kv"),
    # idefics3 (SmolVLM) — llama text decoder + SigLIP vision encoder
    "idefics3":         ("idefics3",    None,          "llama",    False, "kv"),
}


def _resolve_family_str(jang: dict, config: dict) -> tuple[str | None, list[str]]:
    """Find the source-arch string from any known location.

    Priority:
      1. `jang_config["source_model"]["architecture"]`  (string form — JANGTQ converters)
      2. `jang_config["architecture"]["type"]`           (string-in-dict — convert.py)
      3. `config["text_config"]["model_type"]`           (HF VLM wrapper)
      4. `config["model_type"]`                          (HF top-level)
    """
    candidates: list[str] = []

    src_dict = jang.get("source_model") or {}
    if isinstance(src_dict.get("architecture"), str):
        candidates.append(src_dict["architecture"])

    arch_dict = jang.get("architecture")
    if isinstance(arch_dict, dict) and isinstance(arch_dict.get("type"), str):
        candidates.append(arch_dict["type"])

    text_cfg = config.get("text_config") or {}
    if isinstance(text_cfg.get("model_type"), str):
        candidates.append(text_cfg["model_type"])

    if isinstance(config.get("model_type"), str):
        candidates.append(config["model_type"])

    # Filter empty strings, dedupe preserving order.
    seen = set()
    unique = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            unique.append(c)

    for c in unique:
        if c.lower() in FAMILY_MAP:
            return c.lower(), unique

    # Substring fallback — match the longest known key contained in any candidate.
    joined = " ".join(c.lower() for c in unique)
    for key in sorted(FAMILY_MAP.keys(), key=len, reverse=True):
        if key in joined:
            return key, unique

    return None, unique


def _resolve_modality(jang: dict, config: dict, model_path: Path | None = None) -> str:
    """text | vision. jang.has_vision is authoritative when present.

    M127 (iter 50): the fallback used to return "vision" if EITHER
    ``text_config`` OR ``vision_config`` appeared in the HF config. But many
    text-only MoE families (qwen3_moe, qwen3_5_moe, glm_moe_dsa, mistral4)
    wrap their text params under ``text_config`` with NO ``vision_config``,
    so any jang_config missing a ``has_vision`` stamp (legacy v1 files,
    third-party JANG models, manually-edited configs) got misclassified as
    vision. vmlx's CapabilityDetector would then route through
    VLMModelFactory and fail to load. Tightened to require ``vision_config``
    specifically — text_config alone is NOT a vision signal.
    """
    if "has_vision" in jang:
        return "vision" if jang["has_vision"] else "text"
    arch_dict = jang.get("architecture")
    if isinstance(arch_dict, dict) and "has_vision" in arch_dict:
        return "vision" if arch_dict["has_vision"] else "text"
    if "vision_config" in config:
        return "vision"
    return "text"


def build_capabilities(
    jang: dict,
    config: dict | None = None,
    model_path: Path | None = None,
) -> dict | None:
    """Return the canonical `capabilities` block for this model, or None.

    None means the family couldn't be resolved — the caller should warn and
    skip stamping. The block is safe to assign as `jang["capabilities"] = ...`
    and is idempotent (re-running over an already-stamped jang produces the
    same dict).

    Does NOT mutate any inputs.
    """
    if config is None:
        config = {}
    matched, candidates = _resolve_family_str(jang, config)
    if matched is None:
        return None
    family, reasoning, tool, think_in_template, cache_type = FAMILY_MAP[matched]
    modality = _resolve_modality(jang, config, model_path)
    return {
        "reasoning_parser": reasoning,
        "tool_parser": tool,
        "think_in_template": think_in_template,
        "supports_tools": True,
        "supports_thinking": reasoning is not None,
        "family": family,
        "modality": modality,
        "cache_type": cache_type,
    }


def verify_directory(model_dir: Path) -> tuple[bool, str]:
    """Re-read jang_config.json after a converter wrote it and confirm:

      1. capabilities block is present
      2. all required keys are populated
      3. parser names are in the registered set
      4. family/cache/modality round-trip via build_capabilities

    Returns (ok, message). Use at the end of every converter:

        from jang_tools.capabilities import verify_directory
        ok, msg = verify_directory(OUT)
        if not ok:
            sys.exit(f'capabilities verify failed: {msg}')
    """
    jang_path = model_dir / "jang_config.json"
    cfg_path = model_dir / "config.json"

    # `convert_mxtq.py` (legacy) inlines jang under config["jang"] — handle both.
    # M125 (iter 48): wrap open() in `with` so fds close deterministically.
    if not jang_path.exists():
        if cfg_path.exists():
            with open(cfg_path) as fh:
                cfg = json.load(fh)
            inline = cfg.get("jang")
            if isinstance(inline, dict):
                jang = inline
                config = cfg
            else:
                return False, f"no jang_config.json and no config.json::jang inline at {model_dir}"
        else:
            return False, f"no jang_config.json at {model_dir}"
    else:
        with open(jang_path) as fh:
            jang = json.load(fh)
        if cfg_path.exists():
            with open(cfg_path) as fh:
                config = json.load(fh)
        else:
            config = {}

    caps = jang.get("capabilities")
    if caps is None:
        return False, "missing `capabilities` block (converter forgot to stamp)"

    required = {"reasoning_parser", "tool_parser", "think_in_template",
                "supports_tools", "supports_thinking", "family", "modality",
                "cache_type"}
    missing = required - set(caps.keys())
    if missing:
        return False, f"capabilities missing keys: {sorted(missing)}"

    valid_reasoning = {"qwen3", "deepseek_r1", "mistral", "gemma4",
                       "openai_gptoss", None}
    valid_tool = {"qwen", "qwen3", "hermes", "llama", "mistral", "deepseek",
                  "kimi", "granite", "nemotron", "step3p5", "xlam",
                  "functionary", "glm47", "minimax", "gemma4", "native"}
    valid_cache = {"kv", "hybrid", "mla", "mamba"}
    valid_modality = {"text", "vision", "embedding", "rerank", "image"}

    if caps["reasoning_parser"] not in valid_reasoning:
        return False, f"reasoning_parser={caps['reasoning_parser']!r} not in {sorted(valid_reasoning - {None})}"
    if caps["tool_parser"] not in valid_tool:
        return False, f"tool_parser={caps['tool_parser']!r} not in {sorted(valid_tool)}"
    if caps["cache_type"] not in valid_cache:
        return False, f"cache_type={caps['cache_type']!r} not in {sorted(valid_cache)}"
    if caps["modality"] not in valid_modality:
        return False, f"modality={caps['modality']!r} not in {sorted(valid_modality)}"

    expected = build_capabilities(jang, config, model_dir)
    if expected is not None and expected != caps:
        # Stamp drift: file says one family, build_capabilities computes another.
        # Most often happens when converter stamps a stale value before later
        # mutating jang_config (e.g. flipping has_vision after capabilities).
        return False, (
            f"capabilities mismatch — stamped block {caps} but recomputing from "
            f"the same jang_config + config yields {expected}. Re-stamp at the "
            "very end of the converter, after all jang_config mutations."
        )

    return True, f"capabilities OK (family={caps['family']})"


def stamp_directory(model_dir: Path, write: bool = False, verbose: bool = True) -> bool:
    """Convenience: read/build/(write-back) jang_config.json for a directory.

    Returns True if a write would happen (or did). Safe to call after a
    converter finishes — reads jang_config.json and config.json from the
    output dir, builds the capabilities block, stamps it back.
    """
    jang_path = model_dir / "jang_config.json"
    cfg_path = model_dir / "config.json"
    if not jang_path.exists():
        if verbose:
            print(f"  [capabilities] SKIP {model_dir.name} — no jang_config.json")
        return False
    with open(jang_path) as fh:
        jang = json.load(fh)
    if cfg_path.exists():
        with open(cfg_path) as fh:
            config = json.load(fh)
    else:
        config = {}
    caps = build_capabilities(jang, config, model_dir)
    if caps is None:
        if verbose:
            _, cands = _resolve_family_str(jang, config)
            print(f"  [capabilities] WARN {model_dir.name} — no family match (candidates={cands})")
        return False
    existing = jang.get("capabilities")
    if existing == caps:
        if verbose:
            print(f"  [capabilities] OK   {model_dir.name} (family={caps['family']})")
        return False
    jang["capabilities"] = caps
    if verbose:
        tag = "UPD" if existing else "NEW"
        print(
            f"  [capabilities] {tag}  {model_dir.name} → "
            f"family={caps['family']} reasoning={caps['reasoning_parser']} "
            f"tool={caps['tool_parser']} cache={caps['cache_type']}"
        )
    if write:
        with open(jang_path, "w") as f:
            json.dump(jang, f, indent=2)
            f.write("\n")
    return True

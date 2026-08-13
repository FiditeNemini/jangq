"""Stamp the LFM2.5-VL-3B bundle contract.

Vendor-documented sampling (README + generation_config.json agree):
    temperature 0.2, top_k 50, repetition_penalty 1.0.
`top_p` is specified NOWHERE, so it is stamped 1.0 (disabled) rather than
invented — same discipline as top_k=0 on Nemotron 3.5 Lightning.

REASONING: LFM2.5-VL-3B is **not** a reasoning model. The card states it
"answers directly instead of reasoning", the chat template emits a bare
`<|im_start|>assistant\\n` generation prompt with no `<think>` prefill, and
there is no `enable_thinking` kwarg. `build_capabilities` stamps
`supports_thinking: true` off the family row (lfm2 -> qwen3 reasoning parser),
which is wrong here — corrected to false. The template only *tolerates*
`<think>` in prior assistant turns (`preserve_thinking`, default false), which
is history handling, not a reasoning rail.

TOOLS: same mlx_lm detection bug as the rest of the LFM2.5 line — the template
never contains the literal `<|tool_list_start|>`, so mlx_lm infers no parser.
Applies the same fix as `fix_lfm_tool_parser`.

VISION: weight-gated true (437 `model.vision_tower.*` tensors, kept at source
precision). Also mirrors `image_token_id` -> `image_token_index` and stamps
`text_config.block_ff_dim`, both of which mlx_vlm/mlx_lm require, and adds
de-prefixed aliases for per-module quant overrides so mlx_vlm's module-path
lookup finds them.

    python -m jang_tools.stamp_lfm25_vl3b <bundle_dir> [more...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TEMPERATURE = 0.2
TOP_K = 50
TOP_P = 1.0          # unspecified upstream -> disabled, not invented
REPETITION_PENALTY = 1.0
EOS_IDS = [124900]
BOS_ID = 124894
PAD_ID = 124893
IMAGE_TOKEN_ID = 124907

HINT = "{#- <|tool_list_start|> detection hint for mlx_lm -#}"
MARKER = "<|tool_list_start|>"
TOOL_PARSER = "pythonic"
SOURCE_TAG = "vendor_card+generation_config_2026-08-12"


def stamp(b: Path) -> dict:
    cfg_p, jang_p, gen_p = (b / "config.json", b / "jang_config.json",
                            b / "generation_config.json")
    cfg = json.loads(cfg_p.read_text())
    jang = json.loads(jang_p.read_text()) if jang_p.exists() else {}
    gen = json.loads(gen_p.read_text()) if gen_p.exists() else {}

    # ── config keys the MLX runtimes require ──────────────────────────────
    tc = cfg.setdefault("text_config", {})
    if "block_ff_dim" not in tc and "intermediate_size" in tc:
        tc["block_ff_dim"] = tc["intermediate_size"]
    if "image_token_id" in cfg:
        cfg["image_token_index"] = cfg["image_token_id"]

    # mlx_vlm looks up per-module quant overrides by post-sanitize module path
    # (no `model.` prefix). Keep both spellings.
    q = cfg.get("quantization", {})
    for k in [k for k in list(q) if isinstance(q[k], dict) and k.startswith("model.")]:
        q.setdefault(k[len("model."):], q[k])

    # ── sampling: both files must agree ───────────────────────────────────
    gen.update({
        "do_sample": True, "temperature": TEMPERATURE, "top_k": TOP_K,
        "top_p": TOP_P, "repetition_penalty": REPETITION_PENALTY,
        "eos_token_id": EOS_IDS, "bos_token_id": BOS_ID, "pad_token_id": PAD_ID,
    })
    chat = jang.get("chat") or {}
    chat["sampling_defaults"] = {
        "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY, "source": SOURCE_TAG,
    }
    chat["stop_token_ids"] = EOS_IDS
    jang["chat"] = chat

    # ── reasoning: this model does NOT reason ─────────────────────────────
    jang["reasoning"] = {
        "supported": False,
        "default": "off",
        "think_in_template": False,
        "tiers": None,
        "note": ("LFM2.5-VL-3B answers directly; the card states it does not "
                 "reason. The template has no <think> prefill and no "
                 "enable_thinking kwarg. `preserve_thinking` (default false) "
                 "only controls whether <think> in PRIOR assistant turns is "
                 "kept — it is history handling, not a reasoning rail."),
    }

    jang["tools"] = {
        "supported": True, "parser": "lfm2", "mlx_lm_parser": TOOL_PARSER,
        "dialect": "pythonic",
        "format": "<|tool_call_start|>[fn(arg='v')]<|tool_call_end|>",
        "tools_in_system_prompt": True,
    }
    jang["vision"] = {
        "supported": True,
        "tower": "siglip2_vision_model (so400m-patch16-naflex)",
        "tower_precision": "source (not quantized)",
        "image_token_id": IMAGE_TOKEN_ID,
        "max_image_tokens": cfg.get("max_image_tokens"),
        "max_tiles": cfg.get("max_tiles"),
        "processor": "Lfm2VlProcessor (processor_config.json)",
    }

    caps = jang.get("capabilities") or cfg.get("capabilities") or {}
    caps.update({
        "has_vision": True, "has_audio": False, "has_video": False,
        "modality": "vision",
        "modalities": {"text": True, "vision": True, "audio": False, "video": False},
        "supports_tools": True,
        "supports_thinking": False,      # corrected: not a reasoning model
        "default_reasoning": "off",
        "think_in_template": False,
        "tool_parser": "lfm2",
        "family": "lfm2",
        "cache_type": "hybrid",
    })
    jang["capabilities"] = caps
    cfg["capabilities"] = caps

    # ── mlx_lm tool-parser detection ──────────────────────────────────────
    tcp = b / "tokenizer_config.json"
    tk = json.loads(tcp.read_text())
    tk["tool_parser_type"] = TOOL_PARSER
    if isinstance(tk.get("chat_template"), str) and MARKER not in tk["chat_template"]:
        tk["chat_template"] += "\n" + HINT
    tcp.write_text(json.dumps(tk, indent=2, ensure_ascii=False) + "\n")

    jp = b / "chat_template.jinja"
    if jp.exists():
        t = jp.read_text()
        if MARKER not in t:
            jp.write_text(t + "\n" + HINT)

    gen_p.write_text(json.dumps(gen, indent=2) + "\n")
    jang_p.write_text(json.dumps(jang, indent=2) + "\n")
    cfg_p.write_text(json.dumps(cfg, indent=2) + "\n")
    return {"bundle": b.name, "vision": True, "reasoning": "off (not a reasoning model)"}


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    for d in argv[1:]:
        r = stamp(Path(d).expanduser())
        print(f"  stamped {r['bundle']:28s} T={TEMPERATURE} top_p={TOP_P} "
              f"top_k={TOP_K} eos={EOS_IDS} vision=True reasoning={r['reasoning']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

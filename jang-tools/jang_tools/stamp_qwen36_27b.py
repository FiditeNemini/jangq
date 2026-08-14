"""Stamp the Qwen3.6-27B bundle contract (vision + video + MTP + reasoning).

Everything here is taken from the model card and `generation_config.json` of
`Qwen/Qwen3.6-27B`, verified against the downloaded source on 2026-08-12.

SAMPLING — the card documents **three** modes, not one. We stamp all three and
make the thinking/general one the default, because that is what upstream's
`generation_config.json` carries:

    thinking / general   temp 1.0  top_p 0.95  top_k 20  min_p 0  presence 0.0
    thinking / coding    temp 0.6  top_p 0.95  top_k 20  min_p 0  presence 0.0
    instruct (no-think)  temp 0.7  top_p 0.80  top_k 20  min_p 0  presence 1.5

repetition_penalty is 1.0 in all three. Shipping only one number would lose the
coding preset, which is the one that matters most for agentic use — the same
class of loss as the Laguna XS `top_k=20` incident
(`feedback_sampling_defaults_two_file_contract`).

REASONING — thinking is **ON by default**. Verified empirically: the generation
prompt with no kwarg is byte-identical to `enable_thinking=True` and ends
`<|im_start|>assistant\\n<think>\\n`; `enable_thinking=False` instead prefills a
*closed* block `<think>\\n\\n</think>\\n\\n`. So reasoning-off is a prefill, not
an omission. Reasoning parser upstream is `qwen3`.

TOOLS — `qwen3_coder`. mlx_lm already infers this correctly from the template
(it contains the `<tool_call>\\n<function=` literal), so unlike the LFM2.5 line
no detection hint is needed. We still stamp it explicitly.

MODALITY — vision **and video**: the source ships both
`preprocessor_config.json` and `video_preprocessor_config.json`, and carries 333
`model.visual.*` tensors. No audio.

MTP — 15 real `mtp.*` tensors (`mtp.fc` + one transformer block), shared
embeddings (`mtp_use_dedicated_embeddings: False`). Upstream drives it as
`qwen3_next_mtp` with 2 speculative tokens. `runtime_available` stays False
until an MLX runtime actually decodes with it.

    python -m jang_tools.stamp_qwen36_27b <bundle_dir> [more...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SAMPLING_MODES = {
    "thinking_general": {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                         "min_p": 0.0, "presence_penalty": 0.0,
                         "repetition_penalty": 1.0},
    "thinking_coding": {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                        "min_p": 0.0, "presence_penalty": 0.0,
                        "repetition_penalty": 1.0},
    "instruct_nothinking": {"temperature": 0.7, "top_p": 0.80, "top_k": 20,
                            "min_p": 0.0, "presence_penalty": 1.5,
                            "repetition_penalty": 1.0},
}
DEFAULT_MODE = "thinking_general"

EOS_IDS = [248046, 248044]
BOS_ID = 248044
PAD_ID = 248044
SOURCE_TAG = "vendor_card+generation_config_2026-08-12"


def stamp(b: Path) -> dict:
    cfg_p, jang_p, gen_p = (b / "config.json", b / "jang_config.json",
                            b / "generation_config.json")
    cfg = json.loads(cfg_p.read_text())
    jang = json.loads(jang_p.read_text()) if jang_p.exists() else {}
    gen = json.loads(gen_p.read_text()) if gen_p.exists() else {}

    d = SAMPLING_MODES[DEFAULT_MODE]

    # ── two-file sampling contract ────────────────────────────────────────
    gen.update({
        "do_sample": True,
        "temperature": d["temperature"], "top_p": d["top_p"], "top_k": d["top_k"],
        "repetition_penalty": d["repetition_penalty"],
        "eos_token_id": EOS_IDS, "bos_token_id": BOS_ID, "pad_token_id": PAD_ID,
    })
    chat = jang.get("chat") or {}
    chat["sampling_defaults"] = dict(SAMPLING_MODES[DEFAULT_MODE],
                                     source=SOURCE_TAG, mode=DEFAULT_MODE)
    chat["sampling_modes"] = SAMPLING_MODES
    chat["stop_token_ids"] = EOS_IDS
    jang["chat"] = chat

    # ── reasoning ─────────────────────────────────────────────────────────
    jang["reasoning"] = {
        "supported": True,
        "parser": "qwen3",
        "default": "on",
        "think_in_template": True,
        "enable_kwarg": "enable_thinking",
        "off_is_prefilled_closed_block": True,
        "off_prefill": "<think>\n\n</think>\n\n",
        "tiers": None,
        "note": ("Thinking is ON by default — the no-kwarg generation prompt is "
                 "byte-identical to enable_thinking=True. Disabling it prefills "
                 "a CLOSED think block; it does not omit the block."),
    }

    jang["tools"] = {"supported": True, "parser": "qwen3_coder",
                     "dialect": "xml_function",
                     "mlx_lm_autodetected": True}

    # ── modality: weight-gated ────────────────────────────────────────────
    jang["vision"] = {
        "supported": True,
        "video_supported": (b / "video_preprocessor_config.json").exists(),
        "tower": "qwen3_5 ViT (27 layers, hidden 1152)",
        "processor": "preprocessor_config.json",
        "video_processor": "video_preprocessor_config.json",
    }

    idx = b / "model.safetensors.index.json"
    mtp_n = 0
    if idx.exists():
        wm = json.loads(idx.read_text()).get("weight_map", {})
        mtp_n = sum(1 for k in wm if k.startswith("mtp."))
    jang["mtp"] = {
        "artifact_available": mtp_n > 0,
        "tensor_count": mtp_n,
        "runtime_available": False,
        "num_hidden_layers": 1,
        "dedicated_embeddings": False,
        "upstream_method": "qwen3_next_mtp",
        # Upstream vLLM setting — 2 drafts/step. That is D3 in our notation and
        # is measured HARMFUL on Apple silicon (Nemotron: D3 = 0.48x, accept
        # 54%). Do not copy it. The recommendation below is ours.
        "upstream_num_speculative_tokens": 2,
        "recommended_depth": "D2",
        "recommended_num_drafts": 1,
        "depth_note": ("D2 (1 draft) max on Macs. The head is depth-1-trained; "
                       "recursion (D3+) is out-of-distribution and measured a "
                       "net loss. Greedy D2 must be token-identical to D1."),
    }

    caps = jang.get("capabilities") or cfg.get("capabilities") or {}
    caps.update({
        "has_vision": True,
        "has_video": bool(jang["vision"]["video_supported"]),
        "has_audio": False,
        "modality": "video" if jang["vision"]["video_supported"] else "vision",
        "modalities": {"text": True, "vision": True,
                       "video": bool(jang["vision"]["video_supported"]),
                       "audio": False},
        "supports_tools": True, "tool_parser": "qwen3_coder",
        "supports_thinking": True, "default_reasoning": "on",
        "think_in_template": True, "reasoning_parser": "qwen3",
        "family": "qwen3_5", "cache_type": "hybrid",
    })
    jang["capabilities"] = caps
    cfg["capabilities"] = caps

    gen_p.write_text(json.dumps(gen, indent=2) + "\n")
    jang_p.write_text(json.dumps(jang, indent=2) + "\n")
    cfg_p.write_text(json.dumps(cfg, indent=2) + "\n")
    return {"bundle": b.name, "mtp": mtp_n,
            "video": jang["vision"]["video_supported"]}


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    d = SAMPLING_MODES[DEFAULT_MODE]
    for x in argv[1:]:
        r = stamp(Path(x).expanduser())
        print(f"  stamped {r['bundle']:34s} T={d['temperature']} top_p={d['top_p']} "
              f"top_k={d['top_k']} eos={EOS_IDS} reasoning=on vision=True "
              f"video={r['video']} mtp={r['mtp']} (+3 sampling modes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

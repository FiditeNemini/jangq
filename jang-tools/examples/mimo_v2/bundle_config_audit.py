"""MiMo-V2.5 JANG bundle CONFIG-SURFACE audit.

Checks every config/runtime surface a quantized MiMo bundle must carry, beyond
tensor shapes (see bundle_audit.py for those). Reports PASS / WARN / FAIL per
surface so testing actually enforces: model config, quantization metadata,
generation config (sampling + eos traps), tokenizer/special tokens, chat
template (thinking + tool calling), VL/video preprocessor + stamps, and the
MTP/audio preservation contract.

Exit code 0 if no FAIL (WARNs allowed), 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Canonical MiMo-V2.5 special-token ids (source tokenizer_config).
TOKENS = {
    "im_end": 151645,
    "endoftext": 151643,
    "mimo_audio_eod": 151672,
    "vision_start": 151652,
    "vision_end": 151653,
    "image_pad": 151655,
    "video_pad": 151656,
    "tool_call": 151657,
    "tool_call_end": 151658,
    "tool_response": 151665,
    "think": 151667,
    "think_end": 151668,
}
EXPECTED_EOS_LIST = [151643, 151645, 151672]


class Audit:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.failed = False

    def add(self, level: str, surface: str, msg: str) -> None:
        self.rows.append((level, surface, msg))
        if level == "FAIL":
            self.failed = True

    def ok(self, s, m):
        self.add("PASS", s, m)

    def warn(self, s, m):
        self.add("WARN", s, m)

    def fail(self, s, m):
        self.add("FAIL", s, m)

    def report(self) -> None:
        order = {"FAIL": 0, "WARN": 1, "PASS": 2}
        for level, surface, msg in sorted(self.rows, key=lambda r: (order[r[0]], r[1])):
            print(f"  [{level}] {surface}: {msg}")
        n_fail = sum(1 for r in self.rows if r[0] == "FAIL")
        n_warn = sum(1 for r in self.rows if r[0] == "WARN")
        print(f"\n  {len(self.rows)} checks: {n_fail} FAIL, {n_warn} WARN")


def _load(bundle: Path, name: str):
    p = bundle / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--expect-vl", action="store_true",
                        help="Require has_vision/has_video stamps + VL preprocessor.")
    parser.add_argument("--expect-mtp", action="store_true",
                        help="Require num_nextn_predict_layers preserved.")
    args = parser.parse_args()
    a = Audit()
    b = args.bundle

    # --- model config -----------------------------------------------------
    cfg = _load(b, "config.json")
    if cfg is None:
        a.fail("config.json", "missing")
        print(f"bundle={b}")
        a.report()
        return 1
    if cfg.get("model_type") == "mimo_v2":
        a.ok("config.model_type", "mimo_v2")
    else:
        a.fail("config.model_type", f"got {cfg.get('model_type')}")
    for field in ("sliding_window", "hybrid_layer_pattern", "n_routed_experts",
                  "num_experts_per_tok", "attention_value_scale"):
        if field in cfg:
            a.ok(f"config.{field}", str(cfg[field])[:40])
        else:
            a.fail(f"config.{field}", "absent")
    # rope: transformers >=4.50 needs rope_parameters with rope_theta inside
    rp = cfg.get("rope_parameters")
    if isinstance(rp, dict) and "rope_theta" in rp:
        a.ok("config.rope_parameters", f"theta={rp['rope_theta']} partial={rp.get('partial_rotary_factor')}")
    else:
        a.fail("config.rope_parameters", "missing rope_parameters{rope_theta} (tokenizer will drop chat_template)")

    # --- quantization metadata -------------------------------------------
    q = cfg.get("quantization")
    if isinstance(q, dict) and "bits" in q and "group_size" in q:
        overrides = sum(1 for k, v in q.items() if isinstance(v, dict))
        a.ok("quantization", f"top bits={q['bits']} gs={q['group_size']} mode={q.get('mode')} + {overrides} overrides")
        if overrides == 0:
            a.warn("quantization.overrides", "no per-module overrides (uniform bits?)")
    else:
        a.fail("quantization", "missing top-level {bits,group_size}")

    # --- generation config (sampling + eos traps) ------------------------
    gc = _load(b, "generation_config.json")
    if gc is None:
        a.fail("generation_config.json", "missing")
    else:
        if gc.get("do_sample") is False:
            a.warn("generation.do_sample",
                   "do_sample=false (GREEDY default) — greedy+thinking collapses; callers MUST override to T=0.6")
        eos = gc.get("eos_token_id")
        if isinstance(eos, list) and set(eos) >= set(EXPECTED_EOS_LIST):
            a.ok("generation.eos_token_id", f"{eos}")
        else:
            a.warn("generation.eos_token_id", f"expected superset of {EXPECTED_EOS_LIST}, got {eos}")
        cfg_eos = cfg.get("eos_token_id")
        if isinstance(cfg_eos, int) and isinstance(eos, list) and cfg_eos in eos and len(eos) > 1:
            a.warn("config.eos_token_id",
                   f"config.json has single {cfg_eos} but generation_config lists {eos}; stopping must read the list")

    # --- tokenizer + special tokens --------------------------------------
    tc = _load(b, "tokenizer_config.json")
    has_jinja = (b / "chat_template.jinja").exists()
    ct = ""
    if has_jinja:
        ct = (b / "chat_template.jinja").read_text(encoding="utf-8")
    elif tc and isinstance(tc.get("chat_template"), str):
        ct = tc["chat_template"]
    if not ct:
        a.fail("chat_template", "no chat_template.jinja and none in tokenizer_config")
    else:
        a.ok("chat_template", f"present ({len(ct)} chars)")
        # thinking
        if "<think>" in ct and "enable_thinking" in ct:
            a.ok("chat_template.thinking", "<think> + enable_thinking flag present")
        else:
            a.warn("chat_template.thinking", "missing <think> or enable_thinking")
        # tool calling
        if "<tool_call>" in ct and "tool_response" in ct:
            a.ok("chat_template.tools", "<tool_call> + tool_response present")
        else:
            a.warn("chat_template.tools", "tool-calling tokens missing from template")

    if tc:
        at = tc.get("added_tokens_decoder", {})
        present = {int(k) for k in at}
        for name, tid in TOKENS.items():
            if tid not in present:
                a.warn("special_tokens", f"{name} ({tid}) not in added_tokens_decoder")
        if all(t in present for t in (TOKENS["think"], TOKENS["tool_call"], TOKENS["image_pad"], TOKENS["video_pad"])):
            a.ok("special_tokens", "think/tool/image/video pad tokens all present")

    # --- VL / video surface ----------------------------------------------
    if "vision_config" in cfg:
        a.ok("vision_config", "present")
    else:
        (a.fail if args.expect_vl else a.warn)("vision_config", "absent")
    pp = _load(b, "preprocessor_config.json")
    if pp:
        a.ok("preprocessor_config.json", pp.get("image_processor_type", "present"))
    else:
        (a.fail if args.expect_vl else a.warn)("preprocessor_config.json", "absent")
    for stamp in ("has_vision", "has_video"):
        if cfg.get(stamp) is True:
            a.ok(f"config.{stamp}", "true")
        else:
            (a.fail if args.expect_vl else a.warn)(
                f"config.{stamp}", "not stamped (capabilities will keep MiMo text-only)")

    # --- MTP / audio preservation ----------------------------------------
    if "num_nextn_predict_layers" in cfg:
        a.ok("config.num_nextn_predict_layers", str(cfg["num_nextn_predict_layers"]))
    else:
        (a.fail if args.expect_mtp else a.warn)(
            "config.num_nextn_predict_layers", "absent (MTP dropped)")

    print(f"bundle={b}")
    print(f"surfaces audited (PART A/B of the complete record):\n")
    a.report()
    return 1 if a.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Activation-aware source-side QDQ probe for MiMo-V2.5.

This is a diagnostic gate before a full rebuild. It keeps the current MLX
affine min/max codec, but tests whether a runtime-feasible AWQ-style per-layer
MoE input scale reduces drift compared with plain min/max affine.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from jang_tools.mimo_v2.convert_jang import ExpertKeepMap, load_expert_keep_map
from jang_tools.mimo_v2.awq_qdq import (
    awq_channel_scale,
    quant_dequant_awq_weight,
    quant_dequant_minmax_affine,
)

from layer_diff_probe import SourceRunner, rmsnorm
from source_profile_probe import ProbeProfile, final_logits, top_tokens, torch_rel_stats


CALIB_PROMPTS = [
    "What is 2 + 2? Answer in one short sentence.",
    "Name the capital city of France.",
    "Write exactly three comma-separated colors.",
    "A user says: remember Paris. What country is that city in?",
    "def add(a, b):\n    return a + b\n\nWhat does add(2, 5) return?",
    "The quick brown fox jumps over the lazy dog.",
]


class CalibrationRunner(SourceRunner):
    def __init__(self, src: Path):
        super().__init__(src)
        self.layer_input_max: dict[int, torch.Tensor] = {}
        self.used_experts: dict[int, set[int]] = defaultdict(set)

    def _merge_max(self, layer_idx: int, x: torch.Tensor) -> None:
        mag = x.reshape(-1, x.shape[-1]).abs().amax(dim=0).float().cpu()
        prev = self.layer_input_max.get(layer_idx)
        self.layer_input_max[layer_idx] = mag if prev is None else torch.maximum(prev, mag)

    def moe(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        self._merge_max(layer_idx, x)
        bsz, seq_len, hidden = x.shape
        xf = x.reshape(-1, hidden)
        gate_w = self.idx.read_passthrough(f"model.layers.{layer_idx}.mlp.gate.weight", out_dtype=torch.float32)
        bias = self.idx.read_passthrough(
            f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias",
            out_dtype=torch.float32,
        )
        scores = torch.sigmoid(F.linear(xf.float(), gate_w.float()))
        _, topk_idx = torch.topk(scores + bias.view(1, -1), k=self.top_k, dim=-1, sorted=False)
        self.used_experts[layer_idx].update(int(x) for x in torch.unique(topk_idx).tolist())
        return super().moe(layer_idx, x)


def run_calibration(src: Path, tokenizer, *, max_layer: int, thinking: str) -> CalibrationRunner:
    runner = CalibrationRunner(src)
    for text in CALIB_PROMPTS:
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if thinking == "on":
            template_kwargs["enable_thinking"] = True
        elif thinking == "off":
            template_kwargs["enable_thinking"] = False
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}], **template_kwargs)
        ids = tokenizer.encode(prompt)
        h = runner.embed(ids)
        for layer_idx in range(min(runner.cfg["num_hidden_layers"], max_layer + 1)):
            h = runner.layer(layer_idx, h)
    return runner


def layer_awq_scale(
    runner: CalibrationRunner,
    layer_idx: int,
    *,
    alpha: float,
    group_size: int,
) -> torch.Tensor:
    act_max = runner.layer_input_max[layer_idx]
    experts = sorted(runner.used_experts[layer_idx])
    if not experts:
        raise ValueError(f"layer {layer_idx}: calibration observed no routed experts")
    weight_col_max = torch.zeros_like(act_max)
    for expert_idx in experts:
        for proj in ("gate_proj", "up_proj"):
            w = runner.cached_tensor(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}.weight")
            weight_col_max = torch.maximum(weight_col_max, w.float().abs().amax(dim=0).cpu())
    # Use a synthetic row so the common helper's activation/weight balancing
    # applies to the observed layer-wide expert bank.
    scale = awq_channel_scale(
        act_max,
        weight_col_max.reshape(1, -1),
        alpha=alpha,
        floor=1.0,
    )
    if scale.numel() % group_size != 0:
        raise ValueError(f"layer {layer_idx}: scale dim {scale.numel()} not divisible by group_size={group_size}")
    return scale


class AwqQuantizedRunner(SourceRunner):
    def __init__(self, src: Path, profile: ProbeProfile, input_scales: dict[int, torch.Tensor]):
        super().__init__(src)
        self.profile = profile
        self.input_scales = input_scales
        self._q_cache: dict[tuple[str, int, int, str], torch.Tensor] = {}

    @lru_cache(maxsize=128)
    def tensor(self, name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        w = super().tensor(name, dtype=torch.float32)
        expert_bits = self.profile.expert_bits_for(name)
        if expert_bits is not None:
            return quant_dequant_minmax_affine(
                w,
                bits=expert_bits,
                group_size=self.profile.expert_group_for(name),
            )
        if name.endswith(".weight"):
            return quant_dequant_minmax_affine(
                w,
                bits=self.profile.bits_for_weight(name),
                group_size=self.profile.bookend_group_size,
            )
        return w

    @lru_cache(maxsize=128)
    def cached_tensor(self, name: str) -> torch.Tensor:
        return self.tensor(name)

    def _expert_awq_weight(self, name: str, bits: int, group_size: int, input_scale: torch.Tensor) -> torch.Tensor:
        key = (name, bits, group_size, "moe-input-awq")
        cached = self._q_cache.get(key)
        if cached is not None:
            return cached
        w = super().tensor(name, dtype=torch.float32)
        q, _ = quant_dequant_awq_weight(w, input_scale=input_scale, bits=bits, group_size=group_size)
        self._q_cache[key] = q
        return q

    def moe(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        xf = x.reshape(-1, hidden)
        gate_w = self.idx.read_passthrough(f"model.layers.{layer_idx}.mlp.gate.weight", out_dtype=torch.float32)
        bias = self.idx.read_passthrough(
            f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias",
            out_dtype=torch.float32,
        )
        scores = torch.sigmoid(F.linear(xf.float(), gate_w.float()))
        _, topk_idx = torch.topk(scores + bias.view(1, -1), k=self.top_k, dim=-1, sorted=False)
        topk_w = scores.gather(1, topk_idx)
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
        input_scale = self.input_scales[layer_idx].to(device=xf.device, dtype=torch.float32)
        scaled_xf = xf.float() / input_scale
        out = torch.zeros_like(xf)
        for expert_idx in torch.unique(topk_idx).tolist():
            slots = topk_idx == int(expert_idx)
            token_idx, slot_idx = torch.where(slots)
            if token_idx.numel() == 0:
                continue
            bits = self.profile.expert_layer_bits.get(layer_idx, self.profile.default_expert_bits)
            gate_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"
            up_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"
            gate = self._expert_awq_weight(
                gate_name,
                bits["gate_proj"],
                self.profile.expert_group_for(gate_name),
                input_scale,
            )
            up = self._expert_awq_weight(
                up_name,
                bits["up_proj"],
                self.profile.expert_group_for(up_name),
                input_scale,
            )
            down = self.cached_tensor(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight")
            expert_x = scaled_xf[token_idx]
            expert_y = F.linear(F.silu(F.linear(expert_x, gate)) * F.linear(expert_x, up), down)
            out.index_add_(0, token_idx, expert_y * topk_w[token_idx, slot_idx].unsqueeze(-1))
        self._q_cache.clear()
        return out.view(bsz, seq_len, hidden)




class PrunedAwqRunner(AwqQuantizedRunner):
    """AWQ-quantized source path restricted to an expert keep-map (prune+AWQ)."""

    def __init__(self, src: Path, profile: ProbeProfile, input_scales: dict[int, torch.Tensor], expert_keep_map: ExpertKeepMap):
        super().__init__(src, profile, input_scales)
        self.expert_keep_map = expert_keep_map

    def moe(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        keep = self.expert_keep_map.indices_for_layer(layer_idx)
        keep_t = torch.tensor(keep, dtype=torch.long)
        bsz, seq_len, hidden = x.shape
        xf = x.reshape(-1, hidden)
        gate_w = self.idx.read_passthrough(
            f"model.layers.{layer_idx}.mlp.gate.weight", out_dtype=torch.float32
        ).index_select(0, keep_t)
        bias = self.idx.read_passthrough(
            f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", out_dtype=torch.float32
        ).index_select(0, keep_t)
        scores = torch.sigmoid(F.linear(xf.float(), gate_w.float()))
        _, topk_local = torch.topk(scores + bias.view(1, -1), k=self.top_k, dim=-1, sorted=False)
        topk_w = scores.gather(1, topk_local)
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
        input_scale = self.input_scales[layer_idx].to(device=xf.device, dtype=torch.float32)
        scaled_xf = xf.float() / input_scale
        out = torch.zeros_like(xf)
        bits = self.profile.expert_layer_bits.get(layer_idx, self.profile.default_expert_bits)
        for local_expert_idx in torch.unique(topk_local).tolist():
            slots = topk_local == int(local_expert_idx)
            token_idx, slot_idx = torch.where(slots)
            if token_idx.numel() == 0:
                continue
            expert_idx = keep[int(local_expert_idx)]
            gate_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"
            up_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"
            gate = self._expert_awq_weight(gate_name, bits["gate_proj"], self.profile.expert_group_for(gate_name), input_scale)
            up = self._expert_awq_weight(up_name, bits["up_proj"], self.profile.expert_group_for(up_name), input_scale)
            down = self.cached_tensor(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight")
            expert_x = scaled_xf[token_idx]
            expert_y = F.linear(F.silu(F.linear(expert_x, gate)) * F.linear(expert_x, up), down)
            out.index_add_(0, token_idx, expert_y * topk_w[token_idx, slot_idx].unsqueeze(-1))
        self._q_cache.clear()
        return out.view(bsz, seq_len, hidden)


def prompt_ids(tokenizer, text: str, thinking: str, messages_json: str | None = None) -> list[int]:
    import json as _json

    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if thinking == "on":
        template_kwargs["enable_thinking"] = True
    elif thinking == "off":
        template_kwargs["enable_thinking"] = False
    messages = _json.loads(messages_json) if messages_json else [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    return tokenizer.encode(prompt)


def evaluate_profile(src: Path, tokenizer, profile: ProbeProfile, input_scales: dict[int, torch.Tensor], args) -> None:
    source = SourceRunner(src)
    if getattr(args, "expert_keep_map", None) and getattr(args, "keep_experts", 0):
        keep_map = load_expert_keep_map(args.expert_keep_map, keep_experts=args.keep_experts)
        quant = PrunedAwqRunner(src, profile, input_scales, keep_map)
        print(f"pruned_awq keep_experts={args.keep_experts}", flush=True)
    else:
        quant = AwqQuantizedRunner(src, profile, input_scales)
    ids = prompt_ids(tokenizer, args.prompt, args.thinking, getattr(args, 'messages_json', None))
    h_src = source.embed(ids)
    h_q = quant.embed(ids)
    print(f"profile={profile.name} alpha={args.alpha} tokens={len(ids)}")
    rel, last_rel, maxerr = torch_rel_stats(h_src, h_q)
    print(f"embed rel={rel:.6f} last_rel={last_rel:.6f} max={maxerr:.6f}")
    report_layers = {int(x) for x in args.report_layers.split(",") if x.strip()}
    for layer_idx in range(min(source.cfg["num_hidden_layers"], args.max_layer + 1)):
        h_src = source.layer(layer_idx, h_src)
        h_q = quant.layer(layer_idx, h_q)
        if layer_idx in report_layers:
            rel, last_rel, maxerr = torch_rel_stats(h_src, h_q)
            print(f"layer {layer_idx:02d} rel={rel:.6f} last_rel={last_rel:.6f} max={maxerr:.6f}")
    if args.max_layer >= source.cfg["num_hidden_layers"] - 1:
        logits_src = final_logits(source, h_src)
        logits_q = final_logits(quant, h_q)
        rel, last_rel, maxerr = torch_rel_stats(logits_src, logits_q)
        print(f"final_logits rel={rel:.6f} last_rel={last_rel:.6f} max={maxerr:.6f}")
        print("source_top:")
        for token_id, text, value in top_tokens(tokenizer, logits_src):
            print(f"  {token_id:6d} {text!r} {value:.6f}")
        print("quant_top:")
        for token_id, text, value in top_tokens(tokenizer, logits_q):
            print(f"  {token_id:6d} {text!r} {value:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--prompt", default="Name the capital city of France.")
    parser.add_argument("--messages-json", default=None,
                        help="JSON list of chat messages; overrides --prompt for multi-turn probes.")
    parser.add_argument("--thinking", choices=("default", "on", "off"), default="off")
    parser.add_argument("--max-layer", type=int, default=8)
    parser.add_argument("--report-layers", default="0,1,2,3,4,5,6,7,8")
    parser.add_argument("--save-scales", type=Path, default=None,
                        help="Write calibrated per-layer input scales to this JSON for convert_jang --awq-scales.")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--load-scales", type=Path, default=None,
                        help="Reuse previously saved scales JSON instead of recalibrating.")
    parser.add_argument("--expert-keep-map", type=Path, default=None)
    parser.add_argument("--keep-experts", type=int, default=0)
    args = parser.parse_args()

    profile = ProbeProfile.parse(args.profile)
    tokenizer = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)
    if args.load_scales:
        import json as _json

        payload = _json.loads(args.load_scales.read_text(encoding="utf-8"))
        input_scales = {int(k): torch.tensor(v, dtype=torch.float32) for k, v in payload["layers"].items()}
        print(f"LOADED_SCALES {args.load_scales} ({len(input_scales)} layers, alpha={payload.get('alpha')})", flush=True)
        if args.save_scales:
            args.save_scales.write_text(_json.dumps(payload), encoding="utf-8")
        if not args.skip_eval:
            evaluate_profile(args.src, tokenizer, profile, input_scales, args)
        return 0
    calib = run_calibration(args.src, tokenizer, max_layer=args.max_layer, thinking=args.thinking)
    input_scales = {}
    for layer_idx in sorted(calib.layer_input_max):
        input_scales[layer_idx] = layer_awq_scale(
            calib,
            layer_idx,
            alpha=args.alpha,
            group_size=profile.expert_group_size,
        )
        experts = len(calib.used_experts[layer_idx])
        s = input_scales[layer_idx]
        print(
            f"calib layer {layer_idx:02d}: experts={experts} "
            f"scale_min={float(s.min()):.4f} scale_max={float(s.max()):.4f}",
            flush=True,
        )
    if args.save_scales:
        import json

        payload = {
            "alpha": args.alpha,
            "group_size": profile.expert_group_size,
            "calib_prompts": CALIB_PROMPTS,
            "layers": {str(k): [float(x) for x in v.tolist()] for k, v in input_scales.items()},
        }
        args.save_scales.write_text(json.dumps(payload), encoding="utf-8")
        print(f"WROTE_SCALES {args.save_scales} ({len(input_scales)} layers)", flush=True)
    if not args.skip_eval:
        evaluate_profile(args.src, tokenizer, profile, input_scales, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

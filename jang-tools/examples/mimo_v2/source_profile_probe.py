"""Source-side MiMo profile probe.

This simulates a JANG affine profile directly from the FP8/BF16 source
checkpoint without building a safetensors bundle. It is intentionally slow and
diagnostic: the goal is to decide which profile is worth a full conversion.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from jang_tools.mimo_v2.convert_jang import ExpertKeepMap, load_expert_keep_map

from layer_diff_probe import SourceRunner, rmsnorm


_EXPERT_RE = re.compile(
    r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\.\d+\.(?P<proj>gate_proj|up_proj|down_proj)\.weight"
)


@dataclass(frozen=True)
class ProbeProfile:
    name: str
    bookend_bits: int
    default_expert_bits: dict[str, int]
    expert_layer_bits: dict[int, dict[str, int]]
    expert_group_size: int = 128
    expert_group_overrides: dict[str, int] = field(default_factory=dict)
    bookend_group_size: int = 64
    qkv_bits: int | None = None
    layer0_dense_bits: int | None = None
    o_proj_bits: int | None = None
    lm_head_bits: int | None = 0
    token_io_bf16: bool = False
    non_expert_text_bf16: bool = False

    @classmethod
    def parse(cls, raw: str) -> "ProbeProfile":
        key = raw.lower().replace("_", "").replace("-", "")
        base = {"gate_proj": 2, "up_proj": 2, "down_proj": 2}
        critical8 = {"gate_proj": 8, "up_proj": 8, "down_proj": 8}

        if key in {"all2", "2all", "2fit"}:
            return cls("ALL2", 8, base, {})

        if key in {"c4", "2lc4"}:
            return cls("C4", 8, base, {layer: critical8 for layer in range(1, 5)})

        m = re.fullmatch(r"c4l([1-4])(?:b([68]))?(?:h8)?", key)
        if m:
            late_count = int(m.group(1))
            bookend_bits = int(m.group(2) or 8)
            late4 = {"gate_proj": 4, "up_proj": 4, "down_proj": 4}
            layers = {layer: critical8 for layer in range(1, 5)}
            layers.update({layer: late4 for layer in range(48 - late_count, 48)})
            lm_head_bits = 8 if key.endswith("h8") else 0
            return cls(
                f"C4L{late_count}B{bookend_bits}" + ("H8" if lm_head_bits else ""),
                bookend_bits,
                base,
                layers,
                lm_head_bits=lm_head_bits,
            )

        m = re.fullmatch(r"c4l([1-4])x3(?:b([68]))?(?:h8)?", key)
        if m:
            late_count = int(m.group(1))
            bookend_bits = int(m.group(2) or 8)
            late3 = {"gate_proj": 3, "up_proj": 3, "down_proj": 3}
            layers = {layer: critical8 for layer in range(1, 5)}
            layers.update({layer: late3 for layer in range(48 - late_count, 48)})
            lm_head_bits = 8 if key.endswith("h8") else 0
            return cls(
                f"C4L{late_count}x3B{bookend_bits}" + ("H8" if lm_head_bits else ""),
                bookend_bits,
                base,
                layers,
                lm_head_bits=lm_head_bits,
            )

        late8 = re.fullmatch(r"([2348][2348][2348])g(32|64|128)l([1-9]|1[0-6])x8(t16|n16)?", key)
        if late8:
            digits = late8.group(1)
            group_size = int(late8.group(2))
            late_count = int(late8.group(3))
            bf16_suffix = late8.group(4)
            token_io_bf16 = bf16_suffix == "t16"
            non_expert_text_bf16 = bf16_suffix == "n16"
            bits = {
                "gate_proj": int(digits[0]),
                "up_proj": int(digits[1]),
                "down_proj": int(digits[2]),
            }
            late = {"gate_proj": 8, "up_proj": 8, "down_proj": 8}
            layers = {layer: late for layer in range(48 - late_count, 48)}
            return cls(
                f"{digits.upper()}G{group_size}L{late_count}X8" + ("T16" if token_io_bf16 else ""),
                8,
                bits,
                layers,
                expert_group_size=group_size,
                token_io_bf16=token_io_bf16,
                non_expert_text_bf16=non_expert_text_bf16,
            )

        early8 = re.fullmatch(r"([2348][2348][2348])g(32|64|128)e([1-9]|1[0-6])x8(t16|n16)?", key)
        if early8:
            digits = early8.group(1)
            group_size = int(early8.group(2))
            early_count = int(early8.group(3))
            bf16_suffix = early8.group(4)
            token_io_bf16 = bf16_suffix == "t16"
            non_expert_text_bf16 = bf16_suffix == "n16"
            bits = {
                "gate_proj": int(digits[0]),
                "up_proj": int(digits[1]),
                "down_proj": int(digits[2]),
            }
            early = {"gate_proj": 8, "up_proj": 8, "down_proj": 8}
            layers = {layer: early for layer in range(1, early_count + 1)}
            return cls(
                f"{digits.upper()}G{group_size}E{early_count}X8" + ("T16" if token_io_bf16 else ""),
                8,
                bits,
                layers,
                expert_group_size=group_size,
                token_io_bf16=token_io_bf16,
                non_expert_text_bf16=non_expert_text_bf16,
            )

        if key == "223mix":
            # GGUF-informed mixed-group sub-110 no-prune: gate/up 2-bit g64
            # (probed best margin), down 3-bit g128 for size.
            bits = {"gate_proj": 2, "up_proj": 2, "down_proj": 3}
            return cls(
                "223MIX",
                8,
                bits,
                {},
                expert_group_size=64,
                expert_group_overrides={"down_proj": 128},
                qkv_bits=5,
                o_proj_bits=6,
            )

        if key == "223g128u16":
            # unsloth UD-Q2_K_XL mirror boosts: layer 16 gate/up->3 down->4, layer 47 down->4.
            bits = {"gate_proj": 2, "up_proj": 2, "down_proj": 3}
            return cls(
                "223G128U16",
                8,
                bits,
                {
                    16: {"gate_proj": 3, "up_proj": 3, "down_proj": 4},
                    47: {"gate_proj": 2, "up_proj": 2, "down_proj": 4},
                },
                expert_group_size=128,
            )

        grouped = re.fullmatch(r"([2348][2348][2348])g(32|64|128)(t16|n16)?", key)
        if grouped:
            digits = grouped.group(1)
            group_size = int(grouped.group(2))
            bf16_suffix = grouped.group(3)
            token_io_bf16 = bf16_suffix == "t16"
            non_expert_text_bf16 = bf16_suffix == "n16"
            bits = {
                "gate_proj": int(digits[0]),
                "up_proj": int(digits[1]),
                "down_proj": int(digits[2]),
            }
            return cls(
                f"{digits.upper()}G{group_size}" + ("T16" if token_io_bf16 else ""),
                8,
                bits,
                {},
                expert_group_size=group_size,
                token_io_bf16=token_io_bf16,
                non_expert_text_bf16=non_expert_text_bf16,
            )

        d3e = re.fullmatch(r"322d(?:own)?3e(?:arly)?(\d+)(?:b([568]))?(?:q([4568]))?", key)
        if d3e:
            end_layer = int(d3e.group(1))
            bookend_bits = int(d3e.group(2) or 8)
            qkv_bits = int(d3e.group(3) or 6)
            if end_layer < 1 or end_layer > 47:
                raise ValueError(f"invalid early down3 end {end_layer}; expected 1..47")
            base = {"gate_proj": 3, "up_proj": 2, "down_proj": 2}
            early = {"gate_proj": 3, "up_proj": 2, "down_proj": 3}
            layers = {layer: early for layer in range(1, end_layer + 1)}
            return cls(
                f"322D3E{end_layer}" + (f"B{bookend_bits}" if bookend_bits != 8 else ""),
                bookend_bits,
                base,
                layers,
                qkv_bits=qkv_bits,
                layer0_dense_bits=6,
                o_proj_bits=4,
                lm_head_bits=bookend_bits,
            )

        e333 = re.fullmatch(r"(?:slim)?333e(?:arly)?(\d+)(?:b([568]))?(?:q([4568]))?", key)
        if e333:
            end_layer = int(e333.group(1))
            bookend_bits = int(e333.group(2) or 4)
            qkv_bits = int(e333.group(3) or 6)
            if end_layer < 1 or end_layer > 47:
                raise ValueError(f"invalid early 333 end {end_layer}; expected 1..47")
            base = {"gate_proj": 3, "up_proj": 2, "down_proj": 2}
            early = {"gate_proj": 3, "up_proj": 3, "down_proj": 3}
            layers = {layer: early for layer in range(1, end_layer + 1)}
            return cls(
                f"333E{end_layer}" + (f"B{bookend_bits}" if bookend_bits != 4 else ""),
                bookend_bits,
                base,
                layers,
                qkv_bits=qkv_bits,
                layer0_dense_bits=6,
                o_proj_bits=4,
                lm_head_bits=bookend_bits,
            )

        three_digit = re.fullmatch(r"([2348][2348][2348])(t16)?", key)
        if key == "2l" or three_digit:
            if key == "2l":
                bits = {"gate_proj": 4, "up_proj": 2, "down_proj": 3}
                token_io_bf16 = False
            else:
                digits = three_digit.group(1)
                token_io_bf16 = bool(three_digit.group(2))
                bits = {"gate_proj": int(digits[0]), "up_proj": int(digits[1]), "down_proj": int(digits[2])}
            return cls(key.upper(), 8, bits, {}, token_io_bf16=token_io_bf16)

        raise ValueError(
            f"unknown profile {raw!r}; use c4, c4l1..c4l4, c4l1x3..c4l4x3, "
            "optional b6/b8 and h8 suffixes, 2l, 322/323/333/448, 322g64, or 322d3eN"
        )

    def expert_bits_for(self, name: str) -> int | None:
        m = _EXPERT_RE.match(name)
        if not m:
            return None
        layer = int(m.group("layer"))
        proj = m.group("proj")
        return self.expert_layer_bits.get(layer, self.default_expert_bits)[proj]


    def expert_group_for(self, name: str) -> int:
        m = _EXPERT_RE.match(name)
        if m is not None:
            override = self.expert_group_overrides.get(m.group("proj"))
            if override is not None:
                return override
        return self.expert_group_size

    def bits_for_weight(self, name: str) -> int:
        if self.non_expert_text_bf16 and name.endswith(".weight") and self.expert_bits_for(name) is None:
            return 0
        if self.token_io_bf16 and name in {"model.embed_tokens.weight", "lm_head.weight"}:
            return 0
        if name == "lm_head.weight" and self.lm_head_bits is not None:
            return self.lm_head_bits
        if self.qkv_bits is not None and name.endswith(".self_attn.qkv_proj.weight"):
            return self.qkv_bits
        if (
            self.layer0_dense_bits is not None
            and name.startswith("model.layers.0.mlp.")
            and name.endswith("_proj.weight")
        ):
            return self.layer0_dense_bits
        if self.o_proj_bits is not None and name.endswith(".self_attn.o_proj.weight"):
            return self.o_proj_bits
        return self.bookend_bits


def quant_dequant_affine(weight: torch.Tensor, *, bits: int, group_size: int) -> torch.Tensor:
    if bits == 0:
        return weight.float()
    if bits not in {2, 3, 4, 5, 6, 8}:
        raise ValueError(f"unsupported affine bits={bits}")
    x = weight.float()
    if x.ndim != 2:
        raise ValueError(f"expected 2D weight, got shape={tuple(x.shape)}")
    rows, cols = x.shape
    if cols % group_size != 0:
        raise ValueError(f"cols={cols} not divisible by group_size={group_size}")
    groups = cols // group_size
    xr = x.reshape(rows, groups, group_size)
    minv = xr.amin(dim=2, keepdim=True)
    maxv = xr.amax(dim=2, keepdim=True)
    levels = (1 << bits) - 1
    scale = ((maxv - minv) / float(levels)).clamp_min(1e-7)
    q = torch.round((xr - minv) / scale).clamp_(0, (1 << bits) - 1)
    # Converter writes MLX affine sidecars as bf16 negative-scale/max-bias.
    mlx_scale = (-scale).to(torch.bfloat16).to(torch.float32)
    mlx_bias = maxv.to(torch.bfloat16).to(torch.float32)
    return ((levels - q) * mlx_scale + mlx_bias).reshape_as(x)


class QuantizedSourceRunner(SourceRunner):
    def __init__(self, src: Path, profile: ProbeProfile, expert_keep_map: ExpertKeepMap | None = None):
        super().__init__(src)
        self.profile = profile
        self.expert_keep_map = expert_keep_map

    @lru_cache(maxsize=128)
    def tensor(self, name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        w = super().tensor(name, dtype=torch.float32)
        expert_bits = self.profile.expert_bits_for(name)
        if expert_bits is not None:
            return quant_dequant_affine(
                w,
                bits=expert_bits,
                group_size=self.profile.expert_group_for(name),
            )
        if name.endswith(".weight"):
            return quant_dequant_affine(
                w,
                bits=self.profile.bits_for_weight(name),
                group_size=self.profile.bookend_group_size,
            )
        return w

    @lru_cache(maxsize=128)
    def cached_tensor(self, name: str) -> torch.Tensor:
        return self.tensor(name)

    def moe(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        if self.expert_keep_map is None:
            return super().moe(layer_idx, x)

        keep = self.expert_keep_map.indices_for_layer(layer_idx)
        keep_t = torch.tensor(keep, dtype=torch.long)
        bsz, seq_len, hidden = x.shape
        xf = x.reshape(-1, hidden)
        gate_w = self.idx.read_passthrough(
            f"model.layers.{layer_idx}.mlp.gate.weight",
            out_dtype=torch.float32,
        ).index_select(0, keep_t)
        bias = self.idx.read_passthrough(
            f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias",
            out_dtype=torch.float32,
        ).index_select(0, keep_t)
        scores = torch.sigmoid(F.linear(xf.float(), gate_w.float()))
        _, topk_local = torch.topk(scores + bias.view(1, -1), k=self.top_k, dim=-1, sorted=False)
        topk_w = scores.gather(1, topk_local)
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
        out = torch.zeros_like(xf)
        for local_expert_idx in torch.unique(topk_local).tolist():
            slots = topk_local == int(local_expert_idx)
            token_idx, slot_idx = torch.where(slots)
            if token_idx.numel() == 0:
                continue
            expert_idx = keep[int(local_expert_idx)]
            expert_x = xf[token_idx]
            gate = self.cached_tensor(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight")
            up = self.cached_tensor(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight")
            down = self.cached_tensor(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight")
            expert_y = F.linear(F.silu(F.linear(expert_x, gate)) * F.linear(expert_x, up), down)
            out.index_add_(0, token_idx, expert_y * topk_w[token_idx, slot_idx].unsqueeze(-1))
        return out.view(bsz, seq_len, hidden)


def final_logits(runner: SourceRunner, h: torch.Tensor, profile: ProbeProfile | None = None) -> torch.Tensor:
    norm_w = runner.idx.read_passthrough("model.norm.weight", out_dtype=torch.float32)
    h = rmsnorm(h, norm_w, runner.eps)
    lm_head = runner.tensor("lm_head.weight", dtype=torch.float32)
    return F.linear(h, lm_head)


def top_tokens(tokenizer, logits: torch.Tensor, k: int = 8) -> list[tuple[int, str, float]]:
    vals, idx = torch.topk(logits[0, -1].float(), k=k)
    return [(int(i), tokenizer.decode([int(i)]), float(v)) for v, i in zip(vals, idx)]


def torch_rel_stats(src: torch.Tensor, actual: torch.Tensor) -> tuple[float, float, float]:
    s = src.float()
    a = actual.float()
    d = s - a
    rmse = torch.sqrt(torch.mean(d * d))
    rms = torch.sqrt(torch.mean(s * s)) + 1e-12
    last_d = s[:, -1, :] - a[:, -1, :]
    last_rmse = torch.sqrt(torch.mean(last_d * last_d))
    last_rms = torch.sqrt(torch.mean(s[:, -1, :] * s[:, -1, :])) + 1e-12
    return float(rmse / rms), float(last_rmse / last_rms), float(d.abs().max())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--prompt", default="Name the capital city of France.")
    parser.add_argument("--thinking", choices=("default", "on", "off"), default="off")
    parser.add_argument("--report-layers", default="0,1,2,3,4,31,43,44,45,46,47")
    parser.add_argument("--max-layer", type=int, default=47,
                        help="Last decoder layer to run, inclusive. Use <47 for faster screening.")
    parser.add_argument("--expert-keep-map", type=Path,
                        help="Router trace or keep-map JSON used to simulate pruned routed experts.")
    parser.add_argument("--keep-experts", type=int,
                        help="Number of experts per MoE layer to keep when --expert-keep-map is set.")
    args = parser.parse_args()

    profile = ProbeProfile.parse(args.profile)
    expert_keep_map = None
    if args.expert_keep_map is not None:
        if args.keep_experts is None:
            parser.error("--keep-experts is required with --expert-keep-map")
        expert_keep_map = load_expert_keep_map(
            args.expert_keep_map.expanduser(),
            keep_experts=args.keep_experts,
        )
    report_layers = {int(x) for x in args.report_layers.split(",") if x.strip()}

    tokenizer = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if args.thinking == "on":
        template_kwargs["enable_thinking"] = True
    elif args.thinking == "off":
        template_kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], **template_kwargs)
    ids = tokenizer.encode(prompt)

    src = SourceRunner(args.src)
    qsrc = QuantizedSourceRunner(args.src, profile, expert_keep_map=expert_keep_map)
    h_src = src.embed(ids)
    h_q = qsrc.embed(ids)
    rel, last_rel, maxerr = torch_rel_stats(h_src, h_q)
    keep_suffix = f" keep_experts={expert_keep_map.keep_experts}" if expert_keep_map is not None else ""
    print(f"profile={profile.name}{keep_suffix} tokens={len(ids)}")
    print(f"embed rel={rel:.6f} last_rel={last_rel:.6f} max={maxerr:.6f}")

    for layer_idx in range(min(src.cfg["num_hidden_layers"], args.max_layer + 1)):
        h_src = src.layer(layer_idx, h_src)
        h_q = qsrc.layer(layer_idx, h_q)
        if layer_idx in report_layers:
            rel, last_rel, maxerr = torch_rel_stats(h_src, h_q)
            print(f"layer {layer_idx:02d} rel={rel:.6f} last_rel={last_rel:.6f} max={maxerr:.6f}")

    if args.max_layer >= src.cfg["num_hidden_layers"] - 1:
        logits_src = final_logits(src, h_src)
        logits_q = final_logits(qsrc, h_q, profile)
        rel, last_rel, maxerr = torch_rel_stats(logits_src, logits_q)
        print(f"final_logits rel={rel:.6f} last_rel={last_rel:.6f} max={maxerr:.6f}")
        print("source_top:")
        for token_id, text, value in top_tokens(tokenizer, logits_src):
            print(f"  {token_id:6d} {text!r} {value:.6f}")
        print("quant_top:")
        for token_id, text, value in top_tokens(tokenizer, logits_q):
            print(f"  {token_id:6d} {text!r} {value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

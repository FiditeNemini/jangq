"""Compare loaded MLX attention output against artifact-exact source QDQ."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load

from jang_tools.mimo_v2 import mlx_register  # noqa: F401
from jang_tools.mimo_v2.convert_jang import QuantProfile, classify
from layer_diff_probe import SourceRunner
from source_profile_probe import quant_dequant_affine


class ArtifactExactSourceRunner(SourceRunner):
    def __init__(self, src: Path, profile: QuantProfile):
        super().__init__(src)
        self.profile = profile

    def tensor(self, name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        bits, method, group_size = classify(name, self.profile)
        if method == "affine":
            source = self.idx.read_tensor(name, out_dtype=torch.float32)
            return quant_dequant_affine(source, bits=bits, group_size=group_size).to(dtype)
        if method in {"passthrough_bf16", "passthrough_fp32"}:
            return self.idx.read_tensor(name, out_dtype=dtype)
        return self.idx.read_tensor(name, out_dtype=dtype)


def torch_from_mx(x: mx.array) -> torch.Tensor:
    return torch.from_numpy(np.array(x.astype(mx.float32)))


def rel_stats(ref: torch.Tensor, actual: torch.Tensor) -> tuple[float, float, float, float]:
    ref = ref.float()
    actual = actual.float()
    diff = ref - actual
    rmse = torch.sqrt(torch.mean(diff * diff))
    rms = torch.sqrt(torch.mean(ref * ref)) + 1e-12
    last = diff[:, -1, :]
    last_ref = ref[:, -1, :]
    last_rel = torch.sqrt(torch.mean(last * last)) / (torch.sqrt(torch.mean(last_ref * last_ref)) + 1e-12)
    return float(rmse / rms), float(last_rel), float(diff.abs().max()), float(diff.abs().mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--profile", default="444g64")
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--prompt", default="What is 2 + 2? Answer with only the number.")
    parser.add_argument("--thinking", choices=("default", "on", "off"), default="off")
    args = parser.parse_args()

    model, tokenizer = load(str(args.bundle), lazy=True, tokenizer_config={"trust_remote_code": True})
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if args.thinking == "on":
        template_kwargs["enable_thinking"] = True
    elif args.thinking == "off":
        template_kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], **template_kwargs)
    ids = tokenizer.encode(prompt)

    h = model.model.embed_tokens(mx.array([ids], dtype=mx.int32))
    for idx in range(args.layer):
        layer = model.model.layers[idx]
        mask = create_attention_mask(h, None, window_size=layer.self_attn.sliding_window)
        h = layer(h, mask=mask, cache=None)
        mx.eval(h)

    layer = model.model.layers[args.layer]
    mask = create_attention_mask(h, None, window_size=layer.self_attn.sliding_window)
    attn_in = layer.input_layernorm(h)
    mlx_attn = layer.self_attn(attn_in, mask=mask, cache=None)
    mx.eval(attn_in, mlx_attn)

    src = ArtifactExactSourceRunner(args.src, QuantProfile.parse(args.profile))
    torch_attn = src.attention(args.layer, torch_from_mx(attn_in))
    rel, last_rel, maxerr, mae = rel_stats(torch_attn, torch_from_mx(mlx_attn))

    print(f"bundle={args.bundle}")
    print(f"profile={args.profile} layer={args.layer} prompt_tokens={len(ids)}")
    print(f"attn_in_shape={tuple(torch_from_mx(attn_in).shape)}")
    print(f"attention_output rel={rel:.8f} last_rel={last_rel:.8f} maxerr={maxerr:.8f} mae={mae:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

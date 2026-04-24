"""DSV4-Flash weight loader with FP4/FP8/BF16 auto-dispatch."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from jang_tools.dsv4.fp4_codec import dequant_fp4_blockwise
from jang_tools.dsv4.fp8_ue8m0_codec import dequant_fp8_ue8m0_blockwise


# safetensors returns dtype strings from get_slice().get_dtype();
# map to torch.dtype for cheap index-time inspection.
_DTYPE_MAP = {
    "I8": torch.int8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
    "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
}


class ShardIndex:
    """Map tensor_name → (shard_path, dtype) built by scanning all shards."""

    def __init__(self, src_dir: str | Path):
        self.src_dir = Path(src_dir)
        self.shards = sorted(self.src_dir.glob("model-*.safetensors"))
        if not self.shards:
            raise FileNotFoundError(
                f"no model-*.safetensors shards found under {self.src_dir}"
            )
        self._index: dict[str, tuple[Path, torch.dtype]] = {}
        for sp in self.shards:
            with safe_open(str(sp), framework="pt") as f:
                for k in f.keys():
                    slc = f.get_slice(k)
                    dt_str = slc.get_dtype()
                    dt = _DTYPE_MAP.get(dt_str)
                    if dt is None:
                        raise ValueError(f"unknown safetensors dtype {dt_str!r} for {k}")
                    self._index[k] = (sp, dt)
        print(f"[ShardIndex] {len(self.shards)} shards, {len(self._index)} tensors")

    @property
    def keys(self) -> list[str]:
        return list(self._index.keys())

    def dtype_of(self, name: str) -> torch.dtype:
        return self._index[name][1]

    def has(self, name: str) -> bool:
        return name in self._index

    def read_raw(self, name: str) -> torch.Tensor:
        sp, _ = self._index[name]
        with safe_open(str(sp), framework="pt") as f:
            return f.get_tensor(name)

    def read_tensor(
        self, name: str, *, out_dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        """Return the DEQUANTIZED tensor in out_dtype.

        Dispatches on source dtype:
          int8              → FP4 dequant (requires sibling .scale)
          float8_e4m3fn     → FP8 dequant (requires sibling .scale)
          bfloat16/fp32/f16 → passthrough (cast to out_dtype)
          int*              → passthrough as-is (e.g. tid2eid hash table)
        """
        sp, dtype = self._index[name]
        w = self.read_raw(name)
        if dtype == torch.int8:
            scale = self._read_scale_for(name)
            return dequant_fp4_blockwise(w, scale, out_dtype=out_dtype)
        if dtype == torch.float8_e4m3fn:
            scale = self._read_scale_for(name)
            return dequant_fp8_ue8m0_blockwise(w, scale, out_dtype=out_dtype)
        if dtype in (torch.bfloat16, torch.float32, torch.float16):
            return w.to(out_dtype) if dtype != out_dtype else w
        # int64 / int32 / int16 (e.g. hash-routing tables) — leave as-is
        return w

    def _read_scale_for(self, weight_name: str) -> torch.Tensor:
        sk = _scale_key(weight_name)
        if not self.has(sk):
            raise KeyError(f"no sibling scale for {weight_name!r} at {sk!r}")
        return self.read_raw(sk)


def _scale_key(weight_name: str) -> str:
    """DSV4 convention: `foo.weight` → `foo.scale` (not `.weight_scale_inv`)."""
    if weight_name.endswith(".weight"):
        return weight_name[: -len(".weight")] + ".scale"
    return weight_name + ".scale"

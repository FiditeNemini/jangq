"""
Tier classification: decide which source-JANG tensors go into the hot core
(pinned in RAM) versus which get sliced per-expert and streamed from SSD.

Rules (v1, MoE-only):
  - Anything matching `*.switch_mlp.{gate,up,down}_proj.*` is an expert tensor,
    serialized per-expert into ExpertBlobs.
  - Everything else goes into the hot core.

This keeps attention, router (mlp.gate), shared experts, norms, embeddings,
and lm_head resident at all times.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List

EXPERT_RE = re.compile(r"\.switch_mlp\.(gate_proj|up_proj|down_proj)\b")

# Strip trailing safetensors suffixes to get the base weight name.
_SUFFIXES = (".weight", ".scales", ".biases", ".qweight", ".bits", ".bit_map", ".block_offsets", ".zeros", ".shape")


def _base_name(tensor_name: str) -> str:
    for suf in _SUFFIXES:
        if tensor_name.endswith(suf):
            return tensor_name[: -len(suf)]
    return tensor_name


@dataclass
class TierSplit:
    hot_core: List[str] = field(default_factory=list)       # full tensor names (with suffix)
    expert_base_names: List[str] = field(default_factory=list)  # base names, sorted, unique


def classify_tensors(tensor_names: Iterable[str]) -> TierSplit:
    hot: List[str] = []
    expert_bases: set[str] = set()
    for name in tensor_names:
        if EXPERT_RE.search(name):
            expert_bases.add(_base_name(name))
        else:
            hot.append(name)
    return TierSplit(
        hot_core=sorted(hot),
        expert_base_names=sorted(expert_bases),
    )


def is_dense_model(tensor_names: Iterable[str]) -> bool:
    """Return True if no tensor matches the switch_mlp expert pattern."""
    for name in tensor_names:
        if EXPERT_RE.search(name):
            return False
    return True

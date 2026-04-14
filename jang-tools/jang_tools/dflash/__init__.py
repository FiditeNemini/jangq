"""JANG-DFlash: block-diffusion speculative-decoding drafter for JANG MoE targets.

Public API:
    JangDFlashConfig   - dataclass mirroring the Swift JangDFlashConfig
    JangDFlashDrafter  - PyTorch drafter module used by training
    dflash_loss        - weighted masked CE (DFlash Eq. 4)

Training / data modules are accessed via ``python -m`` — importing
them at package load would pull in torch/safetensors/mlx which are
not universally available on every host (distill runs on the M3
Ultra, training runs on the 5090, inference runs on the target Mac).
"""
from .config import JangDFlashConfig
from .drafter import JangDFlashDrafter, dflash_loss

__all__ = ["JangDFlashConfig", "JangDFlashDrafter", "dflash_loss"]

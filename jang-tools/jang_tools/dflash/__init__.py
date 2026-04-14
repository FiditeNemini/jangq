"""JANG-DFlash: block-diffusion speculative-decoding drafter for JANG MoE targets."""
from .config import JangDFlashConfig
from .drafter import JangDFlashDrafter, dflash_loss

__all__ = ["JangDFlashConfig", "JangDFlashDrafter", "dflash_loss"]

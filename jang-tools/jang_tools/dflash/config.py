"""Configuration for the JANG-DFlash block-diffusion drafter."""
from dataclasses import dataclass


@dataclass
class JangDFlashConfig:
    vocab_size: int = 200064
    hidden_dim: int = 1536
    num_layers: int = 5
    num_heads: int = 12
    num_kv_heads: int = 4
    ffn_dim: int = 4096
    block_size: int = 16
    mask_token_id: int = 200064
    tap_dim: int = 15360          # 5 tap layers * 3072 target hidden dim
    head_dim: int = 128
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    loss_gamma: float = 7.0

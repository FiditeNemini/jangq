"""Unit tests for the JANG-DFlash drafter (PyTorch model)."""
import pytest
import torch

from jang_tools.dflash import JangDFlashConfig, JangDFlashDrafter, dflash_loss


def _small_cfg(**overrides) -> JangDFlashConfig:
    cfg = JangDFlashConfig(
        vocab_size=256,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ffn_dim=128,
        block_size=8,
        mask_token_id=256,
        tap_dim=80,      # 5 * 16 for test purposes
        head_dim=16,
        loss_gamma=4.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_drafter_forward_shape_from_h_taps():
    cfg = _small_cfg()
    drafter = JangDFlashDrafter(cfg)
    B, L = 2, cfg.block_size
    block = torch.full((B, L), cfg.mask_token_id, dtype=torch.long)
    block[:, 0] = 42
    h_taps = torch.randn(B, L, cfg.tap_dim)
    logits = drafter(block, h_taps=h_taps)
    assert logits.shape == (B, L, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_drafter_forward_shape_from_h_ctx_kv():
    cfg = _small_cfg()
    drafter = JangDFlashDrafter(cfg)
    B, L = 2, cfg.block_size
    block = torch.full((B, L), cfg.mask_token_id, dtype=torch.long)
    block[:, 0] = 7
    h_ctx_kv = torch.randn(B, L, cfg.hidden_dim)
    logits = drafter(block, h_ctx_kv=h_ctx_kv)
    assert logits.shape == (B, L, cfg.vocab_size)


def test_drafter_rejects_both_or_neither_inputs():
    cfg = _small_cfg()
    drafter = JangDFlashDrafter(cfg)
    block = torch.zeros(1, cfg.block_size, dtype=torch.long)
    with pytest.raises(ValueError):
        drafter(block)
    with pytest.raises(ValueError):
        drafter(block, h_taps=torch.zeros(1, cfg.block_size, cfg.tap_dim),
                h_ctx_kv=torch.zeros(1, cfg.block_size, cfg.hidden_dim))


def test_loss_is_finite_and_weighted():
    cfg = _small_cfg()
    torch.manual_seed(0)
    logits = torch.randn(2, cfg.block_size, cfg.vocab_size)
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    loss = dflash_loss(logits, targets, cfg)
    assert torch.isfinite(loss)
    assert loss.shape == ()


def test_loss_zero_when_perfect():
    cfg = _small_cfg()
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits = torch.full((2, cfg.block_size, cfg.vocab_size), -20.0)
    logits.scatter_(-1, targets.unsqueeze(-1), 20.0)
    loss = dflash_loss(logits, targets, cfg)
    assert loss.item() < 1e-3


def test_loss_rejects_wrong_block_size():
    cfg = _small_cfg()
    logits = torch.randn(1, cfg.block_size + 1, cfg.vocab_size)
    targets = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 1))
    with pytest.raises(ValueError):
        dflash_loss(logits, targets, cfg)


def test_gradient_flow_through_fusion_mlp():
    cfg = _small_cfg()
    drafter = JangDFlashDrafter(cfg)
    block = torch.zeros(1, cfg.block_size, dtype=torch.long)
    h_taps = torch.randn(1, cfg.block_size, cfg.tap_dim, requires_grad=True)
    targets = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    logits = drafter(block, h_taps=h_taps)
    loss = dflash_loss(logits, targets, cfg)
    loss.backward()
    assert h_taps.grad is not None
    assert (h_taps.grad.abs() > 0).any()

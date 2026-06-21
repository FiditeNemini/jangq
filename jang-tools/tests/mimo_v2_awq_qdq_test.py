import torch

from jang_tools.mimo_v2.awq_qdq import awq_channel_scale, quant_dequant_awq_weight


def test_awq_channel_scale_balances_activation_and_weight_ranges():
    act_max = torch.tensor([1.0, 16.0], dtype=torch.float32)
    weight = torch.tensor([[8.0, 1.0], [-4.0, 2.0]], dtype=torch.float32)

    scale = awq_channel_scale(act_max, weight, alpha=0.5)

    assert scale.shape == (2,)
    assert scale[0] < scale[1]
    assert torch.isfinite(scale).all()


def test_quant_dequant_awq_weight_preserves_linear_shape_and_applies_inverse_input_scale():
    weight = torch.arange(64, dtype=torch.float32).reshape(2, 32)
    x = torch.arange(32, dtype=torch.float32).reshape(1, 32)
    scale = torch.linspace(1.0, 2.0, 32, dtype=torch.float32)

    q_weight, q_input = quant_dequant_awq_weight(weight, input_scale=scale, bits=4, group_size=32)

    assert q_weight.shape == weight.shape
    assert q_input(x).shape == x.shape
    assert torch.allclose(q_input(x), x / scale)

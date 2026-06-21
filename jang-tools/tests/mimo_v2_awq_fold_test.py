"""Contract tests for the converter-side AWQ fold (convert_jang --awq-scales)."""

import json

import torch
import torch.nn.functional as F

from jang_tools.mimo_v2.convert_jang import apply_awq_fold, load_awq_scales


def _rmsnorm(h: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    var = h.pow(2).mean(-1, keepdim=True)
    return h * torch.rsqrt(var + eps) * w


def test_awq_fold_preserves_full_precision_moe_function():
    torch.manual_seed(7)
    hidden, inter, n_exp, toks = 64, 32, 16, 8
    h = torch.randn(toks, hidden)
    norm_w = torch.rand(hidden) + 0.5
    router_w = torch.randn(n_exp, hidden)
    gate_w = torch.randn(inter, hidden)
    up_w = torch.randn(inter, hidden)
    down_w = torch.randn(hidden, inter)
    s = torch.rand(hidden) * 4 + 0.25
    scales = {3: s}

    def forward(nw, rw, gw, uw):
        xn = _rmsnorm(h, nw)
        scores = torch.sigmoid(F.linear(xn, rw))
        y = F.linear(F.silu(F.linear(xn, gw)) * F.linear(xn, uw), down_w)
        return scores, y

    base_scores, base_y = forward(norm_w, router_w, gate_w, up_w)

    folded_norm = apply_awq_fold("model.layers.3.post_attention_layernorm.weight", norm_w, scales)
    folded_router = apply_awq_fold("model.layers.3.mlp.gate.weight", router_w, scales)
    folded_gate = apply_awq_fold("model.layers.3.mlp.experts.5.gate_proj.weight", gate_w, scales)
    folded_up = apply_awq_fold("model.layers.3.mlp.experts.5.up_proj.weight", up_w, scales)
    fold_scores, fold_y = forward(folded_norm, folded_router, folded_gate, folded_up)

    assert torch.allclose(base_scores, fold_scores, atol=1e-5)
    assert torch.allclose(base_y, fold_y, atol=1e-4)


def test_awq_fold_only_touches_target_tensors_and_layers():
    s = torch.full((16,), 2.0)
    scales = {1: s}
    w = torch.randn(4, 16)
    # down_proj, other layers, attention, and non-mapped layers stay untouched
    for name in (
        "model.layers.1.mlp.experts.0.down_proj.weight",
        "model.layers.2.mlp.experts.0.gate_proj.weight",
        "model.layers.1.self_attn.qkv_proj.weight",
        "model.layers.1.input_layernorm.weight",
        "lm_head.weight",
    ):
        out = apply_awq_fold(name, w, scales)
        assert out is w, name
    folded = apply_awq_fold("model.layers.1.mlp.experts.0.gate_proj.weight", w, scales)
    assert torch.allclose(folded, w * 2.0)


def test_awq_fold_matches_probe_qdq_semantics():
    from jang_tools.mimo_v2.affine_codec import quantize_minmax_affine
    from jang_tools.mimo_v2.awq_qdq import quant_dequant_awq_weight

    torch.manual_seed(11)
    w = torch.randn(8, 128) * 0.05
    s = torch.rand(128) * 3 + 0.5
    probe_q, transform = quant_dequant_awq_weight(w, input_scale=s, bits=2, group_size=64)

    folded = apply_awq_fold("model.layers.4.mlp.experts.0.gate_proj.weight", w, {4: s})
    x = folded.float().reshape(8, 2, 64)
    minv, maxv = x.amin(2, keepdim=True), x.amax(2, keepdim=True)
    levels = 3
    pos = ((maxv - minv) / levels).clamp_min(1e-7)
    codes = levels - torch.round((x - minv) / pos).clamp_(0, levels)
    recon = (codes * (-pos).to(torch.bfloat16).float() + maxv.to(torch.bfloat16).float()).reshape(8, 128)
    assert torch.allclose(probe_q, recon, atol=0)
    # converter packer accepts the folded tensor unchanged
    qw, qs, qb = quantize_minmax_affine(folded, bits=2, group_size=64)
    assert qw.shape[0] == 8


def test_load_awq_scales_roundtrip(tmp_path):
    s = [1.0, 2.0, 0.5, 4.0]
    path = tmp_path / "scales.json"
    path.write_text(json.dumps({"alpha": 0.5, "layers": {"1": s, "47": s}}))
    scales = load_awq_scales(path)
    assert set(scales) == {1, 47}
    assert torch.allclose(scales[47], torch.tensor(s))

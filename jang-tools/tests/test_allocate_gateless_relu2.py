"""Pin the gate-less relu2 expert bit-floor behaviour (Nemotron-H family).

Nemotron 3.5 Lightning 30B-A3B (and the Nemotron-H family generally) has NO
gate_proj: the routed expert MLP is ``down_proj(relu2(up_proj(x)))``. The
SwiGLU-derived MLP_ASYMMETRY_FLOORS therefore encodes the wrong assumption
("up_proj: no floor, 2-bit OK when gate is protected") — there is no gate, and
up_proj feeds a SQUARING nonlinearity, making it the amplifier. Compounding it,
those floors only fire at >=256 experts and this model has 128, so neither the
gate nor the down_proj floor applied at all.

Measured before the fix, over the real 6513 tensor names at JANG_2L:
    routed up_proj   -> 2 bits   (squaring activation!)
    routed down_proj -> 2 bits
    backbone.embeddings -> 2 bits
    conv1d           -> 2 bits

These tests pin both the fix and — critically — its narrow blast radius:
``gateless_relu2`` defaults to False, so no existing model's allocation moves.
"""
import numpy as np

from jang_tools.allocate import (
    RELU2_ASYMMETRY_FLOORS,
    Tier,
    _apply_gateless_relu2_conv_floor,
    _apply_relu2_asymmetry_floor,
    allocate_bits_profile,
    allocate_bits_profile_compact,
    classify_tensor,
)

UP = "backbone.layers.1.mixer.experts.5.up_proj.weight"
DOWN = "backbone.layers.1.mixer.experts.5.down_proj.weight"
SHARED = "backbone.layers.1.mixer.shared_experts.up_proj.weight"
CONV = "backbone.layers.0.mixer.conv1d.weight"
EMB = "backbone.embeddings.weight"


def test_floors_are_intentional_values():
    """up_proj inherits gate_proj's 4-bit floor because relu2 squares it."""
    assert RELU2_ASYMMETRY_FLOORS["up_proj"] == 4
    assert RELU2_ASYMMETRY_FLOORS["down_proj"] == 3


def test_disabled_by_default_is_a_noop():
    """The whole point of the flag: default False changes nothing."""
    for name in (UP, DOWN, SHARED, CONV):
        assert _apply_relu2_asymmetry_floor(name, 2) == 2
        assert _apply_gateless_relu2_conv_floor(name, 2, 8) == 2


def test_up_and_down_floors_apply_regardless_of_expert_count():
    """128 experts is below _MLP_ASYMMETRY_MIN_EXPERTS=256, so the SwiGLU
    floors never fire here. These must not be expert-count gated."""
    assert _apply_relu2_asymmetry_floor(UP, 2, gateless_relu2=True) == 4
    assert _apply_relu2_asymmetry_floor(DOWN, 2, gateless_relu2=True) == 3


def test_shared_expert_exempt():
    """shared_expert is already CRITICAL — same carve-out as the SwiGLU path."""
    assert _apply_relu2_asymmetry_floor(SHARED, 8, gateless_relu2=True) == 8


def test_non_expert_tensors_pass_through():
    """Only routed expert MLPs are affected; a dense up_proj is untouched."""
    dense = "model.layers.0.mlp.up_proj.weight"
    assert _apply_relu2_asymmetry_floor(dense, 2, gateless_relu2=True) == 2


def test_floor_never_lowers_bits():
    assert _apply_relu2_asymmetry_floor(UP, 8, gateless_relu2=True) == 8
    assert _apply_relu2_asymmetry_floor(DOWN, 6, gateless_relu2=True) == 6


def test_conv1d_promoted_to_critical_only_when_enabled():
    assert _apply_gateless_relu2_conv_floor(CONV, 2, 8, gateless_relu2=True) == 8
    assert _apply_gateless_relu2_conv_floor(CONV, 2, 8, gateless_relu2=False) == 2


def test_backbone_embeddings_is_important_not_compress():
    """`backbone.embeddings` matched none of embed_tokens/wte/word_embeddings
    and fell through to the COMPRESS default — 2-bit on a 0.35B table."""
    assert classify_tensor(EMB, 128) == Tier.IMPORTANT


def test_embeddings_rule_does_not_capture_vision_or_audio_towers():
    """Anchored on `backbone.` deliberately: a bare "embeddings" pattern would
    re-tier every shipped VL/audio bundle from COMPRESS to IMPORTANT."""
    assert classify_tensor(
        "vision_model.embeddings.position_embedding.weight", 128
    ) == Tier.COMPRESS
    assert classify_tensor(
        "audio_tower.embeddings.conv1.weight", 128
    ) == Tier.COMPRESS


def test_end_to_end_jang_2l_protects_the_squaring_amplifier():
    info = [(UP, 1), (DOWN, 1), (CONV, 1), (EMB, 1)]
    off = allocate_bits_profile_compact(info, profile="JANG_2L", num_experts=128)
    on = allocate_bits_profile_compact(
        info, profile="JANG_2L", num_experts=128, gateless_relu2=True
    )
    assert off[UP] == 2 and on[UP] == 4
    assert off[DOWN] == 2 and on[DOWN] == 3
    assert off[CONV] == 2 and on[CONV] == 8
    # embeddings is fixed by the tier rule, so it is already correct with the
    # flag off — pin that so the two mechanisms don't get conflated.
    assert off[EMB] == 6 and on[EMB] == 6


def test_array_and_compact_allocators_agree():
    names = [UP, DOWN, CONV, EMB]
    arr = allocate_bits_profile(
        names, profile="JANG_2L", num_experts=128, gateless_relu2=True
    )
    cmp_ = allocate_bits_profile_compact(
        [(n, 1) for n in names], profile="JANG_2L", num_experts=128,
        gateless_relu2=True,
    )
    assert list(arr) == [cmp_[n] for n in names]
    assert arr.dtype == np.uint8

"""Pin the MLP asymmetry bit-floor behaviour for 256+ expert MoE models.

Memory ref: `project_mlp_asymmetry.md` (11 days old at iter 21). Threshold was
lowered from 512 to 256 on 2026-04-08 after GLM-5.1 (256 experts) degenerated
into repetition loops when all three expert MLP tensor classes were at pure
2-bit. The code had drifted back to a 512 check — iter 21's memory-cross-ref
caught it. These tests pin the invariant so future drift fails loudly.
"""
from jang_tools.allocate import (
    _MLP_ASYMMETRY_MIN_EXPERTS,
    _apply_mlp_asymmetry_floor,
    MLP_ASYMMETRY_FLOORS,
)


def test_threshold_matches_memory():
    """project_mlp_asymmetry.md says 256 is the threshold. Pin it so code
    and memory can't drift apart again."""
    assert _MLP_ASYMMETRY_MIN_EXPERTS == 256


def test_floors_are_intentional_values():
    """gate_proj=4, down_proj=3, up_proj=no-floor per the memory's root-cause
    analysis. Pin these so a future 'let's just lower everything' PR breaks."""
    assert MLP_ASYMMETRY_FLOORS["gate_proj"] == 4
    assert MLP_ASYMMETRY_FLOORS["down_proj"] == 3
    assert MLP_ASYMMETRY_FLOORS["w1"] == 4       # Mixtral alias for gate_proj
    assert MLP_ASYMMETRY_FLOORS["w2"] == 3       # Mixtral alias for down_proj
    assert "up_proj" not in MLP_ASYMMETRY_FLOORS, "up_proj must have NO floor"


def test_below_threshold_no_change():
    # 8 experts (e.g., Mixtral 8x7B) — no asymmetry floor applied.
    for bits in [2, 3, 4, 6, 8]:
        assert _apply_mlp_asymmetry_floor("layers.0.mlp.gate_proj.weight", bits, 8) == bits
        assert _apply_mlp_asymmetry_floor("layers.0.mlp.down_proj.weight", bits, 8) == bits
        assert _apply_mlp_asymmetry_floor("layers.0.mlp.up_proj.weight", bits, 8) == bits


def test_at_256_threshold_floor_applies():
    """GLM-5.1 has exactly 256 experts — must get the floor, else it enters
    repetition loops per the memory's root-cause analysis."""
    # 2-bit input for expert gate_proj → floor raises to 4-bit
    assert _apply_mlp_asymmetry_floor(
        "model.layers.0.mlp.experts.0.gate_proj.weight", 2, 256) == 4
    # 2-bit input for expert down_proj → floor raises to 3-bit
    assert _apply_mlp_asymmetry_floor(
        "model.layers.0.mlp.experts.0.down_proj.weight", 2, 256) == 3
    # up_proj has NO floor → stays at 2-bit
    assert _apply_mlp_asymmetry_floor(
        "model.layers.0.mlp.experts.0.up_proj.weight", 2, 256) == 2


def test_above_256_floor_still_applies():
    """397B / Nemotron (512+ experts) also need the floor — they were the
    original motivating case before the 2026-04-08 lowering."""
    assert _apply_mlp_asymmetry_floor(
        "layers.0.mlp.gate_proj.weight", 2, 512) == 4
    assert _apply_mlp_asymmetry_floor(
        "layers.0.mlp.down_proj.weight", 2, 512) == 3


def test_shared_expert_exempt():
    """shared_expert is already CRITICAL-tier in the profile — don't touch it.
    Applying the floor here would waste bits on a tensor that's already
    well-protected."""
    assert _apply_mlp_asymmetry_floor(
        "model.layers.0.mlp.shared_expert.gate_proj.weight", 2, 256) == 2
    assert _apply_mlp_asymmetry_floor(
        "model.layers.0.mlp.shared_expert.down_proj.weight", 2, 256) == 2


def test_non_expert_mlp_names_pass_through():
    """Non-MLP tensor names (e.g., attention proj) should never match a
    floor — they don't contain gate_proj / down_proj / w1 / w2 substrings."""
    assert _apply_mlp_asymmetry_floor(
        "model.layers.0.self_attn.q_proj.weight", 2, 256) == 2
    assert _apply_mlp_asymmetry_floor(
        "model.embed_tokens.weight", 2, 256) == 2


def test_floor_never_lowers_bits():
    """Floor is a MINIMUM — if the tensor is already at 6-bit, the floor
    doesn't DOWNGRADE it to 4. Sanity check for `max(bits, floor)`."""
    assert _apply_mlp_asymmetry_floor(
        "model.layers.0.mlp.experts.0.gate_proj.weight", 6, 256) == 6
    assert _apply_mlp_asymmetry_floor(
        "model.layers.0.mlp.experts.0.down_proj.weight", 8, 256) == 8


def test_mixtral_naming_also_covered():
    """Mixtral uses w1/w2 instead of gate_proj/down_proj. Both naming
    conventions must activate the floor."""
    # Mixtral 8x22B has 8 experts — below threshold. But w1/w2 naming on a
    # hypothetical 256-expert Mixtral variant must be covered.
    assert _apply_mlp_asymmetry_floor(
        "layers.0.block_sparse_moe.experts.0.w1.weight", 2, 256) == 4
    assert _apply_mlp_asymmetry_floor(
        "layers.0.block_sparse_moe.experts.0.w2.weight", 2, 256) == 3

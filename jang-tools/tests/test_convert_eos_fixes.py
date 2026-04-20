"""Pin the coverage of convert.py's EOS_FIXES map.

Memory ref: `feedback_chat_template_rules.md` (31 days old at iter 26).
Qwen3.5 source ships with eos_token_id=248044 (<|endoftext|>) which must
be fixed to 248046 (<|im_end|>) during conversion or the model loops
infinitely on inference.

Iter-26 Cat D cross-ref extended coverage to the VL variants of the Qwen3
family (`qwen3_vl`, `qwen3_moe_vl`, `qwen3_5_vl`) which share the Qwen2
tokenizer token IDs. The fix is idempotent on correct sources (only
applies when the source actually has 248044) so broadening coverage
carries zero regression risk.
"""
from jang_tools.convert import EOS_FIXES


def test_eos_fixes_dict_shape():
    """Every value must map 248044 (wrong) → 248046 (correct). Pin this so
    a future edit that accidentally reverses the mapping breaks the test
    instead of silently miscorrecting every Qwen3 conversion."""
    for model_type, mapping in EOS_FIXES.items():
        assert mapping == {248044: 248046}, \
            f"{model_type} mapping must be 248044 → 248046, got {mapping}"


def test_covers_qwen3_5_and_moe():
    """Original memory-documented coverage — must never be removed."""
    assert "qwen3_5" in EOS_FIXES
    assert "qwen3_5_moe" in EOS_FIXES


def test_covers_qwen3_vl_variants():
    """Iter-26 extension. The Qwen3.5/3.6 family includes image + video VL
    variants which ship with the same Qwen2Tokenizer token IDs, so the
    248044 wrong-eos bug can slip through on VL sources without these
    entries. Regression pin."""
    assert "qwen3_vl" in EOS_FIXES, "qwen3_vl (image VL) missing from EOS_FIXES"
    assert "qwen3_moe_vl" in EOS_FIXES, "qwen3_moe_vl (MoE VL) missing from EOS_FIXES"
    assert "qwen3_5_vl" in EOS_FIXES, "qwen3_5_vl (3.5 VL) missing from EOS_FIXES"


def test_no_coverage_for_unrelated_families():
    """Regression pin the NEGATIVE case — adding llama or mistral here would
    mis-correct tokens on those architectures, which don't use 248046 as
    <|im_end|>. They simply have correct eos in the source.
    """
    for mt in ["llama", "mistral", "gemma", "phi", "qwen2", "minimax_m2",
               "deepseek_v2", "deepseek_v3", "nemotron_h"]:
        assert mt not in EOS_FIXES, \
            f"{mt} must NOT be in EOS_FIXES — its eos is already correct in source"


def test_idempotent_on_correct_source():
    """Document the invariant: the fix only triggers when source.eos ∈
    EOS_FIXES[model_type].keys. A model that already has the correct
    248046 in its config passes through unchanged.

    This is enforced at the use site in convert_model:
      `if tc.get("eos_token_id") in eos_fix_map: tc["eos_token_id"] = ...`
    """
    mapping = EOS_FIXES["qwen3_5_moe"]
    # Simulate the check
    correct_source_eos = 248046
    assert correct_source_eos not in mapping, \
        "correct eos 248046 must NOT be a key in the fix map — the check `in eos_fix_map` guards against double-fix"
    # Only the wrong value is keyed
    assert 248044 in mapping


def test_fix_map_keys_are_wrong_eos_values():
    """Double-check: every KEY in every mapping is a 'wrong' eos that needs
    correction, and every VALUE is the 'right' eos. If somebody ever flips
    these, every conversion would set eos to 248044 (wrong) instead of
    leaving the 248046 source untouched."""
    for mt, mapping in EOS_FIXES.items():
        for wrong, right in mapping.items():
            assert wrong != right, f"{mt}: wrong → right must differ, got {wrong} = {right}"
            # 248044 = <|endoftext|>; 248046 = <|im_end|>. Numeric pin.
            assert wrong == 248044, f"{mt}: wrong token id must be 248044, got {wrong}"
            assert right == 248046, f"{mt}: right token id must be 248046, got {right}"

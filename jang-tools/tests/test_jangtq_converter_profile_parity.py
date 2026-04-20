"""M132 (iter 54): JANGTQ converter profile-validation parity.

Iter 52's meta-pattern applied to the two JANGTQ converters:
  - jang_tools/convert_minimax_jangtq.py
  - jang_tools/convert_qwen35_jangtq.py

Both take a ``PROFILE`` string and map it to an ``EXPERT_BITS`` int via a
nearly-identical ``_PROFILE_BITS`` / ``_EXPERT_BITS_BY_PROFILE`` dict.

**Asymmetry:** MiniMax raised ValueError on unknown profile (correct).
Qwen35 silently fell back to 2 (bug — "JANGTQ44" typo would produce a
2-bit conversion labeled JANGTQ4 in jang_config, no warning).

Iter 54 aligns the Qwen35 converter to MiniMax's behavior. This test
pins both converters together so any future divergence fails loudly.

Because the converters do heavy MLX work at module level we can't import
them directly in-process. These tests read the source files and inspect
the profile-validation block with a regex — we're verifying the
**code shape** rather than runtime behavior. The corresponding runtime
behavior is exercised by the existing end-to-end audit jobs.
"""
from __future__ import annotations

import re
from pathlib import Path

CONVERTERS = {
    "minimax": Path(__file__).resolve().parents[1] / "jang_tools" / "convert_minimax_jangtq.py",
    "qwen35": Path(__file__).resolve().parents[1] / "jang_tools" / "convert_qwen35_jangtq.py",
}


def _extract_profile_block(src: str) -> str:
    """Return the text surrounding the _PROFILE_BITS / _EXPERT_BITS_BY_PROFILE
    dict + the immediate next 30 lines (lookup + raise guard)."""
    for key in ("_PROFILE_BITS", "_EXPERT_BITS_BY_PROFILE"):
        idx = src.find(f"{key} = {{")
        if idx != -1:
            # Grab the dict + next ~30 lines to include the lookup + raise.
            tail = src[idx:]
            # Stop at the first "===" comment or 40 lines, whichever first.
            lines = tail.splitlines()[:40]
            return "\n".join(lines)
    return ""


def test_both_converters_raise_on_unknown_profile():
    """Both converters must raise ValueError for an unrecognized profile.
    Pre-iter-54, qwen35 silently fell back to 2-bit via dict.get(…, 2)."""
    for name, path in CONVERTERS.items():
        src = path.read_text(encoding="utf-8")
        block = _extract_profile_block(src)
        assert block, f"{name}: could not locate profile-bits dict"
        assert "raise ValueError" in block, (
            f"{name} converter is missing a ValueError guard for unknown profiles. "
            f"Silent fallback would produce a mis-labeled output (wrong bit-width "
            f"landing in jang_config while the name says something else). "
            f"Block:\n{block}"
        )


def test_no_silent_dict_get_fallback_on_profile_bits():
    """Regression: neither converter should use `dict.get(_PROFILE_NORM, <int>)`
    with a numeric default. The `, 2` silent fallback WAS the bug."""
    for name, path in CONVERTERS.items():
        src = path.read_text(encoding="utf-8")
        # Reject patterns like `_EXPERT_BITS_BY_PROFILE.get(_PROFILE_NORM, 2)`.
        bad = re.search(
            r"_(?:PROFILE_BITS|EXPERT_BITS_BY_PROFILE)\.get\(\s*_PROFILE_NORM\s*,\s*\d+\s*\)",
            src,
        )
        assert bad is None, (
            f"{name} uses a silent-fallback dict.get on the profile-bits "
            f"table: {bad.group(0)!r}. Use explicit membership + raise "
            f"ValueError instead (M132)."
        )


def test_both_converters_share_same_profile_bits_keys():
    """Keeps the two dicts in sync. If one converter adds a new legacy name
    the other won't silently diverge."""
    keys_by_converter = {}
    for name, path in CONVERTERS.items():
        src = path.read_text(encoding="utf-8")
        block = _extract_profile_block(src)
        # Match "JANGTQ*": N entries.
        keys = set(re.findall(r'"(JANGTQ[\w_]*)"\s*:', block))
        keys_by_converter[name] = keys
    minimax_keys = keys_by_converter["minimax"]
    qwen_keys = keys_by_converter["qwen35"]
    assert minimax_keys == qwen_keys, (
        f"Profile-bits dicts diverged between converters:\n"
        f"  minimax only: {minimax_keys - qwen_keys}\n"
        f"  qwen35 only:  {qwen_keys - minimax_keys}\n"
        f"Keep the canonical JANGTQ{{N}} + legacy aliases identical across "
        f"both converters so users can't convert a Qwen with a label the "
        f"MiniMax converter would reject (or vice-versa)."
    )

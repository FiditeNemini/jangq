"""M212 (iter 143): generation_config.json must be included in the
Qwen3.5-family eos_token_id auto-fix.

Pre-M212 the convert eos-fix in `jang-tools/jang_tools/convert.py`
applied `EOS_FIXES[model_type]` to:
  - model_config (top-level eos_token_id)
  - model_config.text_config (nested eos_token_id)
  - tokenizer_config.json (eos_token_id field)

It did NOT patch `generation_config.json`. HF's `GenerationMixin`
reads `generation_config.eos_token_id` at `.generate()` time with
PRIORITY over `config.eos_token_id`, so a Qwen3.5-family source
shipping `generation_config.json` with the stale scalar `248044`
would produce a bundle where:
  - config.json eos_token_id = 248046 (fixed)
  - generation_config.json eos_token_id = 248044 (unfixed)

At inference HF picks 248044 → infinite thinking loop → exactly the
bug A04's fix was supposed to eliminate, silently undone by the
un-patched generation_config.

M212 extends the eos-fix loop to load + patch + rewrite
generation_config.json when it exists in source AND contains a
matching eos ID (scalar OR list). The extra_configs byte-copy loop
skips generation_config.json when M212 already wrote the patched
version (so we don't overwrite the fix with stale source bytes).

This test pins the plumbing via source inspection — complements
the live-render check (iter-143 captured output against the real
Qwen3.6-35B fixture showed `[248046, 248044]` list-form which
survives correctly either way; scalar-form would fail without M212).
"""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
CONVERT_PY = REPO_ROOT / "jang-tools" / "jang_tools" / "convert.py"


def _live_code(path: Path) -> str:
    """Strip `#` single-line comments (Python) + blank-line normalize."""
    lines = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        # Keep inline comments' code portion; strip whole-line `#` lines
        # to avoid false-positives on rationale prose.
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _eos_fix_block() -> str:
    """Return the source slice that covers the eos-fix region —
    from the `# ── eos_token_id auto-fix ──` comment to just past
    the `extra_configs` byte-copy loop. Scoped so unrelated mentions
    of `generation_config.json` don't register."""
    content = CONVERT_PY.read_text(encoding="utf-8")
    start = content.find("── eos_token_id auto-fix ─")
    assert start != -1, "eos-fix region not found in convert.py"
    end = content.find("for extra_file in extra_configs:", start)
    assert end != -1
    # Extend `end` past the loop body — approximate.
    return content[start : end + 800]


def test_eos_fix_also_patches_generation_config():
    """Pin: the eos-fix block references generation_config.json and
    uses EOS_FIXES / eos_fix_map to patch it. Without this, the fix
    is incomplete and stale generation_config.eos_token_id silently
    undoes the config.json fix at inference."""
    block = _eos_fix_block()
    assert "generation_config.json" in block, (
        "M212 regression: eos-fix region no longer mentions "
        "generation_config.json. Pre-M212 this file was copied "
        "byte-for-byte without patching — any Qwen3.5-family source "
        "with stale scalar eos_token_id in generation_config would "
        "silently break at inference. The fix must load + patch + "
        "rewrite the file alongside the config.json + tokenizer_config "
        "patches."
    )
    # Pin the actual patch logic: load + assignment to eos_token_id.
    assert "eos_fix_map" in block, (
        "M212 regression: eos-fix block doesn't reference eos_fix_map. "
        "Expected to see the SAME map applied to generation_config "
        "as to config.json."
    )


def test_eos_fix_handles_both_scalar_and_list_forms():
    """HF generation_config.json's eos_token_id is EITHER a scalar
    (legacy Qwen3.5) or a list (modern Qwen3/3.5-VL). Fix must handle
    both — a scalar-only branch would miss list-form (where multiple
    IDs need independent mapping), and a list-only branch would miss
    scalar-form (which is the bug we're fixing)."""
    block = _eos_fix_block()
    assert "isinstance(old_eos, int)" in block, (
        "M212 regression: eos-fix for generation_config lacks a scalar "
        "(int) handler. Pre-M212 the bug was exactly this case — "
        "source had a SCALAR stale eos that the old fix missed."
    )
    assert "isinstance(old_eos, list)" in block, (
        "M212 regression: eos-fix for generation_config lacks a list "
        "handler. Modern Qwen3/Qwen3.5-VL uses list-form multi-EOS; "
        "each ID must be mapped through eos_fix_map independently."
    )


def test_extra_configs_copy_skips_patched_generation_config():
    """Pin: the extra_configs byte-copy loop must skip
    generation_config.json when M212 already wrote the patched
    version. Without this guard, the byte-copy would overwrite the
    fix with stale source bytes — double-bug. The flag
    `_eos_fixed_gen_cfg` signals the skip."""
    content = CONVERT_PY.read_text(encoding="utf-8")
    # Find the byte-copy loop.
    loop_idx = content.find("for extra_file in extra_configs:")
    assert loop_idx != -1
    loop_body = content[loop_idx : loop_idx + 800]
    assert "_eos_fixed_gen_cfg" in loop_body, (
        "M212 regression: extra_configs copy loop doesn't check "
        "_eos_fixed_gen_cfg. Without the skip guard, a successful "
        "eos-fix rewrite would be overwritten by the plain byte-copy "
        "a few lines later — silent double-bug."
    )
    assert 'extra_file == "generation_config.json"' in loop_body, (
        "M212 regression: copy-loop skip guard doesn't match on "
        "generation_config.json. Must be gated by the exact filename "
        "so other extra_configs (preprocessor_config.json etc.) aren't "
        "accidentally skipped."
    )


def test_eos_fix_map_covers_qwen3_5_family():
    """Regression pin on the EOS_FIXES table — Qwen3.5 family
    members (qwen3_5, qwen3_5_moe, qwen3_vl, qwen3_moe_vl, qwen3_5_vl)
    all map 248044 → 248046."""
    content = CONVERT_PY.read_text(encoding="utf-8")
    # Find EOS_FIXES dict.
    start = content.find("EOS_FIXES = {")
    # M177 (iter 111) promoted EOS_FIXES to module scope; the exact
    # assignment line name might vary slightly across versions.
    if start == -1:
        # Try alternate capitalization / prefix.
        m = re.search(r"EOS_FIXES\s*[:=]", content)
        assert m is not None, "EOS_FIXES declaration missing"
        start = m.start()
    block = content[start : start + 800]
    for model_type in ("qwen3_5", "qwen3_5_moe", "qwen3_vl", "qwen3_5_vl"):
        assert model_type in block, (
            f"M212 regression: EOS_FIXES no longer covers "
            f"{model_type!r}. Every Qwen3.5-family model_type must map "
            f"248044 → 248046 or the fix is incomplete."
        )
    # Values check — both the wrong ID and the right ID must appear.
    assert "248044" in block and "248046" in block, (
        "M212 regression: EOS_FIXES values don't include 248044 → 248046. "
        "These exact IDs are the Qwen3.5 <|endoftext|> and <|im_end|> "
        "tokens the fix was designed for."
    )

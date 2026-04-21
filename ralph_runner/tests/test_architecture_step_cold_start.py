"""M218 (iter 146): ArchitectureStep cold-start guidance invariant.

Pre-M218 a first-time JANGStudio user arriving at Step 2 saw:
  - "Model type: qwen3_5_moe" — jargon
  - "Layout: MoE · 256 experts" — more jargon
  - "Source dtype: BF16" — technical abbreviation
  - "Vision/Language: No" — OK
  - Large-expert info line (conditional)
  - DisclosureGroup("Advanced overrides") — unexplained
  - "Looks right → Profile" button — vague confirmation

No explanation of what the step is FOR, what MoE means, what dtype
shows up as, or what "just click Continue" would do.

M218 adds always-visible plain-English guidance:
  1. Section-top caption explaining what this step is for + the
     typical stranger action ("usually no change needed").
  2. Plain-English MoE/Dense caption under the Layout value.
  3. dtype caption with a one-sentence hint per dtype class.

Extends iter-140 M206's pattern (always-visible, not hover-only)
to the next wizard step.
"""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
ARCH_STEP = REPO_ROOT / "JANGStudio" / "JANGStudio" / "Wizard" / "Steps" / "ArchitectureStep.swift"


def _live_code(path: Path) -> str:
    lines = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("///") or stripped.startswith("//"):
            continue
        lines.append(line)
    return "\n".join(lines)


def test_section_top_caption_explains_purpose():
    """Pin: the Section must open with a plain-English caption
    explaining what the step is for. A stranger arriving at Step 2
    deserves to know before scanning technical fields."""
    live = _live_code(ARCH_STEP)
    assert "JANG Studio inspected your model" in live, (
        "M218 regression: ArchitectureStep missing the section-top "
        "caption. Cold-start users need a one-liner explaining what "
        "this step's purpose is. Pre-M218 users saw only technical "
        "fields with no framing."
    )
    assert "usually no change" in live or "no change is needed" in live, (
        "M218 regression: caption doesn't tell the stranger the "
        "typical action. Without that hint, users stare at Advanced "
        "overrides unsure whether they need to touch anything."
    )


def test_moe_has_plain_english_explanation():
    """When the detected layout is MoE, a plain-English caption must
    appear beneath the value explaining what MoE means — not just
    the number of experts."""
    live = _live_code(ARCH_STEP)
    assert "Mixture-of-Experts" in live, (
        "M218 regression: ArchitectureStep no longer spells out "
        "'Mixture-of-Experts' inline. `MoE · N experts` is jargon for "
        "beginners — the caption defuses it."
    )


def test_dense_has_plain_english_explanation():
    """Symmetric: the non-MoE case must also carry a plain-English
    caption. `Dense` is similarly jargon-y (a stranger thinks 'thick',
    not 'non-sparse')."""
    live = _live_code(ARCH_STEP)
    assert "all weights active" in live or "Standard transformer" in live, (
        "M218 regression: ArchitectureStep's Dense branch is missing "
        "the plain-English caption. `Dense` means something specific "
        "in ML but is confusing to a first-timer."
    )


def test_dtype_hint_helper_exists_and_covers_all_cases():
    """Pin the dtypeHint helper and verify it has a case for every
    SourceDtype enum value (exhaustive switch). If SourceDtype gains
    a new case (e.g. int8), the helper's switch must grow with it —
    otherwise the new dtype renders an empty hint."""
    content = ARCH_STEP.read_text(encoding="utf-8")
    assert re.search(
        r"func\s+dtypeHint\(_\s+\w+\s*:\s*SourceDtype\s*\)\s*->\s*String",
        content,
    ), (
        "M218 regression: dtypeHint helper missing or signature changed. "
        "Expected: `private func dtypeHint(_ d: SourceDtype) -> String`."
    )
    # Each currently-known SourceDtype must have a case in the switch.
    live = _live_code(ARCH_STEP)
    for dtype_case in ("bf16", "fp16", "fp8", "jangV2", "unknown"):
        assert f"case .{dtype_case}" in live or f"case .{dtype_case}:" in live, (
            f"M218 regression: dtypeHint's switch missing `case .{dtype_case}`. "
            f"A source detected as {dtype_case} would show NO plain-English "
            f"hint — defeating the cold-start guidance purpose."
        )


def test_dtype_caption_renders_below_labeled_content():
    """Pin that the dtype LabeledContent is wrapped in a VStack with
    the hint caption directly beneath. A plain label + a hint on the
    NEXT section row wouldn't visually associate them."""
    content = ARCH_STEP.read_text(encoding="utf-8")
    # Look for the pattern: VStack containing both `LabeledContent("Source dtype", ...)` and `dtypeHint(...)`.
    # Substring proxy — stricter AST parse would be overkill.
    assert "LabeledContent(\"Source dtype\"" in content
    # Find the VStack-wrapped dtype region.
    idx = content.find("LabeledContent(\"Source dtype\"")
    assert idx != -1
    # Look backward for "VStack" and forward for "dtypeHint" within a small window.
    backward = content[max(0, idx - 120) : idx]
    forward = content[idx : idx + 350]
    assert "VStack" in backward, (
        "M218 regression: Source dtype LabeledContent not wrapped in a "
        "VStack. Without a VStack the hint caption would float on its "
        "own row, disconnected from the label it explains."
    )
    assert "dtypeHint" in forward, (
        "M218 regression: dtypeHint call not adjacent to the Source "
        "dtype LabeledContent. The hint must render directly beneath "
        "the value to visually associate."
    )

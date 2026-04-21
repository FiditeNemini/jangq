"""M223 (iter 149): TestInferenceSheet cold-start guidance invariant.

§8 stage #6 (test-inference) audit. Pre-M223 a stranger arriving at
an empty Test Inference sheet saw only:
  - "No messages yet"
  - "Type a prompt to see how your converted model responds"
  - A blank prompt field
  - For VL: "Drop image here" (no format/size hint)

No example prompt → user types "Hello" → generic response → learns
nothing. Worse for reasoning models (Qwen3.6 / GLM-5.1 / MiniMax M2.7):
the default chat template wraps prompts in `<think>...</think>` blocks
which eat 100+ tokens before answering, so a 150-token smoke test
returns no visible answer. The "Skip thinking" Settings toggle fixes
this but is buried in the gear-icon popover.

M223 adds:
  1. Three suggested-prompt buttons (factual recall, reasoning,
     creativity) that prefill the prompt field on tap.
  2. A reasoning-model warning banner that appears for known
     reasoning model types (qwen3_5*, qwen3_6*, glm*, minimax*) AND
     when skipThinking is OFF — pointing the user at the Settings
     toggle that fixes the documented failure mode.

Extends the M206 / M218 cold-start-caption pattern to a sheet.
"""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
SHEET_PATH = REPO_ROOT / "JANGStudio" / "JANGStudio" / "Wizard" / "TestInferenceSheet.swift"


def _live_code(path: Path) -> str:
    lines = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("///") or stripped.startswith("//"):
            continue
        lines.append(line)
    return "\n".join(lines)


def test_empty_state_offers_sample_prompts():
    """Pin: TestInferenceSheet's empty state must offer at least
    one tappable sample prompt button. Without it, strangers type
    'Hello' as their first prompt and learn nothing about model
    capabilities."""
    live = _live_code(SHEET_PATH)
    assert "samplePromptButton" in live, (
        "M223 regression: TestInferenceSheet's empty state no longer "
        "offers samplePromptButton calls. Cold-start strangers need "
        "concrete prompt examples to demonstrate model capabilities."
    )
    # At least 3 distinct sample-prompt buttons (factual / reasoning / creative).
    button_call_count = live.count("samplePromptButton(")
    assert button_call_count >= 3, (
        f"M223 regression: only {button_call_count} samplePromptButton "
        f"calls in TestInferenceSheet. Pre-M223 there were 3 covering "
        f"factual recall + reasoning + creativity. Restore the variety "
        f"so strangers see different prompt styles."
    )


def test_sample_prompt_button_helper_exists():
    """Pin the helper function — keeps the call sites DRY and
    makes future reuse explicit."""
    content = SHEET_PATH.read_text(encoding="utf-8")
    assert re.search(
        r"func\s+samplePromptButton\(_\s+\w+\s*:\s*String\s*\)",
        content,
    ), (
        "M223 regression: samplePromptButton helper missing or "
        "signature changed. Expected: `private func samplePromptButton(_ prompt: String) -> some View`."
    )


def test_reasoning_model_hint_present():
    """Pin: a hint about the Skip-thinking Settings toggle must
    appear when the loaded model is a known reasoning type AND the
    user hasn't already enabled the toggle. Documented failure mode
    in feedback memory: 150-token smoke tests on reasoning models
    return empty answers because <think>...</think> eats the budget."""
    live = _live_code(SHEET_PATH)
    assert "Reasoning model detected" in live, (
        "M223 regression: TestInferenceSheet no longer surfaces the "
        "reasoning-model hint. Strangers running short smoke tests on "
        "Qwen3.6 / GLM-5.1 / MiniMax M2.7 will see empty answers and "
        "assume the model is broken."
    )
    assert "Skip thinking" in live, (
        "M223 regression: hint must reference the EXACT Settings toggle "
        "name 'Skip thinking' so users can find it. Renaming the toggle "
        "without updating the hint orphans the guidance."
    )
    # The hint should be gated on `!vm.skipThinking` so it disappears
    # once the user has enabled the fix.
    assert "!vm.skipThinking" in live, (
        "M223 regression: hint not gated on `!vm.skipThinking`. Without "
        "the gate, the hint stays visible after the user has fixed the "
        "issue — false-positive nag."
    )


def test_reasoning_model_classifier_covers_known_families():
    """Pin: isReasoningModelType helper must match all known
    reasoning families. If a future reasoning model family is added
    (e.g. deepseek_v32_thinking), the classifier must grow with it
    or its users won't see the hint."""
    live = _live_code(SHEET_PATH)
    assert re.search(
        r"func\s+isReasoningModelType\(_\s+\w+\s*:\s*String\s*\)\s*->\s*Bool",
        live,
    ), "M223 regression: isReasoningModelType helper missing or signature changed."
    # Each currently-known reasoning family must be in the classifier.
    for family in ("qwen3_5", "qwen3_6", "glm", "minimax"):
        assert f'"{family}"' in live or f"'{family}'" in live, (
            f"M223 regression: isReasoningModelType doesn't recognize "
            f"{family!r}. Reasoning models in this family won't show "
            f"the cold-start hint and users will hit the empty-answer "
            f"failure mode unaware."
        )


def test_skip_thinking_toggle_exists_in_settings_popover():
    """Pin the destination of the M223 hint. The hint tells the
    user to open Settings and enable Skip thinking — that toggle
    must actually exist in the settings popover, or the hint sends
    the user on a wild goose chase."""
    content = SHEET_PATH.read_text(encoding="utf-8")
    assert "$vm.skipThinking" in content, (
        "M223 regression: 'Skip thinking' toggle binding missing from "
        "TestInferenceSheet. The M223 hint references this toggle by "
        "name and a missing toggle would break the user's escape route."
    )

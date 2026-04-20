"""Tests for jang_tools.modelcard."""
import json
import subprocess
import sys
from pathlib import Path
import pytest

from jang_tools.modelcard import generate_card


@pytest.fixture
def dense_model_dir(tmp_path):
    d = tmp_path / "dense"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "num_hidden_layers": 28,
        "_name_or_path": "Qwen/Qwen3-0.6B-Base",
        "license": "apache-2.0",
    }))
    (d / "jang_config.json").write_text(json.dumps({
        "format": "jang", "family": "jang", "profile": "JANG_4K",
        "quantization": {"actual_bits_per_weight": 4.23, "block_size": 64, "bit_widths_used": [3, 4, 6, 8]},
    }))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": "{% for m in messages %}{{m.content}}{% endfor %}",
    }))
    # Create a fake shard so size_gb is non-zero
    (d / "model-00001-of-00001.safetensors").write_bytes(b"x" * 1_000_000)
    return d


def test_card_has_frontmatter(dense_model_dir):
    # M202 (iter 138): generate_card now returns (card, license_unknown).
    card, license_unknown = generate_card(dense_model_dir)
    assert card.startswith("---")
    assert "license:" in card
    assert "base_model:" in card
    assert "quantization_config:" in card
    assert "family: jang" in card
    assert "profile: JANG_4K" in card
    # dense_model_dir fixture explicitly sets license=apache-2.0 in
    # config.json, so license_unknown must be False.
    assert license_unknown is False


def test_card_has_usage_section(dense_model_dir):
    card, _ = generate_card(dense_model_dir)
    assert "Quick start" in card
    assert "```python" in card
    # M45 (iter 20): symbol is `load_jang_model`, not `load_model`. The
    # `load_model` substring appears INSIDE `load_jang_model` so the old
    # assertion was vacuously true — a pure import-name regression would
    # never have been caught here. Assert the full correct symbol AND
    # assert the bare `load_model(` call is NOT present.
    assert "load_jang_model" in card
    assert "load_model(" not in card, \
        "card contains bare `load_model(` — would ImportError for adopters"


def test_cli_json_output(dense_model_dir):
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "modelcard",
         "--model", str(dense_model_dir), "--json"],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    assert data["license"] == "apache-2.0"
    assert data["quantization_config"]["family"] == "jang"
    assert data["quantization_config"]["profile"] == "JANG_4K"
    assert "card_markdown" in data
    assert data["card_markdown"].startswith("---")


def test_cli_writes_file(dense_model_dir, tmp_path):
    out = tmp_path / "README.md"
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "modelcard",
         "--model", str(dense_model_dir), "--output", str(out)],
        capture_output=True, text=True, check=True,
    )
    assert out.exists()
    assert out.read_text().startswith("---")


# ────────────────────────────────────────────────────────────────────
# Iter 28: M91 — CLI emits skeleton-warning note on stderr
# ────────────────────────────────────────────────────────────────────

def test_cli_emits_skeleton_warning_to_stderr(dense_model_dir):
    """M91: the auto-generated card is a skeleton; publishing without MMLU
    scores, JANG-vs-MLX comparison, and Korean section violates
    feedback_readme_standards.md. CLI emits a stderr note so humans (and
    Ralph tail-reads) know before they publish.

    The note must land on STDERR, not stdout — stdout carries the card
    markdown / JSON payload and must stay machine-parseable.
    """
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "modelcard",
         "--model", str(dense_model_dir), "--json"],
        capture_output=True, text=True, check=True,
    )
    # stdout is clean JSON (would fail parse otherwise)
    json.loads(r.stdout)
    # stderr contains the skeleton note. Use a distinctive phrase from the
    # warning message (not just "skeleton" which can appear in tmp paths
    # named after the test case — pytest tmp_path dirs include the test name).
    assert "generated card is a skeleton" in r.stderr.lower(), \
        f"stderr missing skeleton warning: {r.stderr!r}"
    # Note points at the memory rule so future readers know the WHY
    assert "feedback_readme_standards" in r.stderr or "MMLU" in r.stderr
    # But stdout stays MACHINE-PARSEABLE — the distinctive warning phrase
    # must NOT leak onto stdout.
    assert "generated card is a skeleton" not in r.stdout.lower(), \
        "skeleton note must NOT appear on stdout (breaks --json consumers)"


# ────────────────────────────────────────────────────────────────────
# Iter 138: M202 — license must be explicit or marked as unknown
#
# Pre-M202 the template defaulted missing licenses to "apache-2.0" via
# Jinja's `| default("apache-2.0")` filter. HF convention puts the
# license in README YAML, not config.json — so most upstream configs
# (Qwen, Llama, etc.) have no license key. The silent default fabricated
# an apache-2.0 tag for every one. Legally concerning for derivatives
# of non-apache-2.0 models.
#
# Post-M202: modelcard.py coerces None → "other" (HF standard for
# custom/unknown) AND returns license_unknown=True; CLI emits a
# specific stderr warning; template renders a visible banner at the
# top of the card body.
# ────────────────────────────────────────────────────────────────────

@pytest.fixture
def no_license_model_dir(tmp_path):
    """Fixture mimicking upstream HF model configs that omit `license`
    (Qwen, Llama, many others)."""
    d = tmp_path / "no_license"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "num_hidden_layers": 28,
        "_name_or_path": "Qwen/Qwen3-0.6B-Base",
        # NO "license" key — mirrors real Qwen3 config.json on HF Hub.
    }))
    (d / "jang_config.json").write_text(json.dumps({
        "format": "jang", "family": "jang", "profile": "JANG_4K",
        "quantization": {"actual_bits_per_weight": 4.23, "block_size": 64, "bit_widths_used": [3, 4, 6, 8]},
    }))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": "{% for m in messages %}{{m.content}}{% endfor %}",
    }))
    (d / "model-00001-of-00001.safetensors").write_bytes(b"x" * 1_000_000)
    return d


def test_license_present_in_config_round_trips_to_card(dense_model_dir):
    """M202: when config.json DOES have `license`, the card must use
    that EXACT value — no placeholder coercion."""
    card, license_unknown = generate_card(dense_model_dir)
    assert license_unknown is False
    # YAML frontmatter must contain the source's license verbatim.
    assert "license: apache-2.0" in card, (
        f"M202 regression: card must preserve the explicit license "
        f"from config.json (apache-2.0) — got: {card[:200]!r}"
    )
    # Must NOT show the license-unknown warning banner.
    assert "License not detected" not in card, (
        "M202 regression: license was present in config but warning "
        "banner still rendered."
    )


def test_license_absent_in_config_renders_other_and_warning(no_license_model_dir):
    """M202: when config.json OMITS `license` (typical HF upstream),
    the card MUST use `license: other` (not silently default to
    apache-2.0 — that's a fabrication) AND include a visible warning
    banner in the body."""
    card, license_unknown = generate_card(no_license_model_dir)
    assert license_unknown is True
    # YAML frontmatter uses HF-standard "other" marker.
    assert "license: other" in card, (
        f"M202 regression: missing license must render `license: other`, "
        f"not silently default to apache-2.0 — got: {card[:200]!r}"
    )
    assert "license: apache-2.0" not in card, (
        "M202 regression: license-fabrication bug returned — missing "
        "license is now rendering as apache-2.0 again."
    )
    # Body must carry a user-visible warning banner pointing at the fix.
    assert "License not detected" in card, (
        "M202 regression: no warning banner in card body. User must be "
        "told to set the license before HF publishing."
    )


def test_cli_stderr_warns_when_license_absent(no_license_model_dir):
    """M202: CLI must emit a distinctive stderr note when license is
    unknown, separate from the M91 skeleton warning."""
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "modelcard",
         "--model", str(no_license_model_dir), "--json"],
        capture_output=True, text=True, check=True,
    )
    assert "no `license` key in config.json" in r.stderr, (
        f"M202 regression: stderr must warn about missing license — "
        f"got: {r.stderr!r}"
    )
    # Distinctive phrase must NOT leak onto stdout (breaks --json consumers).
    assert "no `license` key" not in r.stdout, (
        "M202 regression: license warning leaked onto stdout"
    )


def test_cli_stderr_silent_when_license_present(dense_model_dir):
    """M202: no false-positive license warning when config.json DOES
    have a license. Only emit the warning when it's actually missing."""
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "modelcard",
         "--model", str(dense_model_dir), "--json"],
        capture_output=True, text=True, check=True,
    )
    assert "no `license` key in config.json" not in r.stderr, (
        f"M202 regression: license warning fired when config HAS a "
        f"license — false positive. stderr: {r.stderr!r}"
    )


def test_cli_json_output_license_unknown_signal(no_license_model_dir):
    """M202: the --json payload must carry `license_unknown: true` so
    Swift PublishToHF / other consumers can prompt the user to set
    it before uploading."""
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "modelcard",
         "--model", str(no_license_model_dir), "--json"],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    assert data["license"] == "other", (
        "M202 regression: --json must emit `license: other` when "
        "source is missing, not silently fabricate apache-2.0."
    )
    assert data.get("license_unknown") is True, (
        "M202 regression: --json must expose `license_unknown: true` "
        "so programmatic consumers can gate HF upload on license set."
    )

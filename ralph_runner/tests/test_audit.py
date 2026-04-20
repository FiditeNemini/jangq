"""Unit tests for ralph_runner.audit — shape + registry.

MLX-dependent rows (a1-a5, a15) run only when mlx_lm is importable AND
a real model is available. For hermetic unit tests we stub and only test
dispatch + registry behavior.
"""
import json
import subprocess
import sys
from pathlib import Path
import pytest

from ralph_runner.audit import AUDIT_REGISTRY, audit_a7_size_estimate


@pytest.fixture
def mock_model_dir(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text('{"model_type":"qwen3"}')
    (d / "jang_config.json").write_text('{"format":"jang","format_version":"2.0","capabilities":{"arch":"qwen3"},"quantization":{"actual_bits_per_weight":4.23,"block_size":64,"bit_widths_used":[4]}}')
    (d / "tokenizer.json").write_text('{"model":{"type":"BPE"}}')
    (d / "tokenizer_config.json").write_text('{"tokenizer_class":"Qwen2Tokenizer"}')
    (d / "special_tokens_map.json").write_text('{}')
    # 1 MB fake shard
    (d / "model-00001-of-00001.safetensors").write_bytes(b"x" * 1_000_000)
    return d


def test_registry_has_expected_rows():
    for row in ["a1", "a2", "a3", "a4", "a5", "a7", "a15"]:
        assert row in AUDIT_REGISTRY
    # a1 and a15 are required
    assert AUDIT_REGISTRY["a1"][2] is True
    assert AUDIT_REGISTRY["a15"][2] is True


def test_a7_passes_when_no_prediction(mock_model_dir):
    r = audit_a7_size_estimate(mock_model_dir, predicted_bytes=None)
    assert r["status"] == "pass"
    assert r["actual_gb"] == 0.001   # 1 MB


def test_a7_warns_on_large_drift(mock_model_dir):
    r = audit_a7_size_estimate(mock_model_dir, predicted_bytes=10_000_000)  # predicted 10 MB, actual 1 MB
    assert r["status"] == "warn"
    assert r["ratio"] == 0.1


def test_a7_passes_near_estimate(mock_model_dir):
    r = audit_a7_size_estimate(mock_model_dir, predicted_bytes=1_050_000)  # 5% over
    assert r["status"] == "pass"


def test_cli_rejects_missing_model(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "ralph_runner.audit", "--model", str(tmp_path / "nope")],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 2
    data = json.loads(r.stdout)
    assert "error" in data


def test_cli_unknown_row_is_na(mock_model_dir):
    r = subprocess.run(
        [sys.executable, "-m", "ralph_runner.audit",
         "--model", str(mock_model_dir),
         "--rows", "a7,a999_unknown",
         "--json"],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["rows"]["a999_unknown"]["status"] == "n/a"
    assert data["rows"]["a7"]["status"] in ("pass", "warn")


def test_registry_includes_new_rows():
    for row in ["a8", "a9", "a16", "a17", "a18"]:
        assert row in AUDIT_REGISTRY, f"{row} missing from registry"
    # a9, a17, a18 are required
    assert AUDIT_REGISTRY["a9"][2] is True
    assert AUDIT_REGISTRY["a17"][2] is True
    assert AUDIT_REGISTRY["a18"][2] is True
    # a8, a16 are warn-only
    assert AUDIT_REGISTRY["a8"][2] is False
    assert AUDIT_REGISTRY["a16"][2] is False


def test_a8_na_when_no_source():
    from ralph_runner.audit import audit_a8_parser_preservation
    r = audit_a8_parser_preservation(Path("/tmp/anywhere"), source_dir=None)
    assert r["status"] == "n/a"


def test_a9_ok_on_nonempty_output_without_source(mock_model_dir):
    # Write a minimal special_tokens_map to the fixture
    (mock_model_dir / "special_tokens_map.json").write_text('{"bos_token":"<s>","eos_token":"</s>"}')
    from ralph_runner.audit import audit_a9_special_tokens
    r = audit_a9_special_tokens(mock_model_dir, source_dir=None)
    assert r["status"] == "pass"
    assert "bos_token" in r["output_keys"]


def test_a9_na_when_source_has_no_special_tokens_map(tmp_path, mock_model_dir):
    # Source dir with no special_tokens_map.json — A9 should be n/a (nothing to preserve)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    # No special_tokens_map.json in source
    from ralph_runner.audit import audit_a9_special_tokens
    r = audit_a9_special_tokens(mock_model_dir, source_dir=source_dir)
    assert r["status"] == "n/a"


def test_a17_fails_without_jang_config(tmp_path):
    # Empty dir — jang-tools modelcard will fail
    from ralph_runner.audit import audit_a17_modelcard_generatable
    r = audit_a17_modelcard_generatable(tmp_path)
    assert r["status"] == "fail"


def test_a11_na_when_not_vl(mock_model_dir):
    from ralph_runner.audit import audit_a11_vl_preprocessor_functional
    r = audit_a11_vl_preprocessor_functional(mock_model_dir)
    assert r["status"] == "n/a"


def test_a12_na_when_not_video(mock_model_dir):
    from ralph_runner.audit import audit_a12_video_preprocessor_functional
    r = audit_a12_video_preprocessor_functional(mock_model_dir)
    assert r["status"] == "n/a"


def test_registry_has_a11_a12():
    from ralph_runner.audit import AUDIT_REGISTRY
    assert "a11" in AUDIT_REGISTRY
    assert "a12" in AUDIT_REGISTRY
    # Both are warn-only
    assert AUDIT_REGISTRY["a11"][2] is False
    assert AUDIT_REGISTRY["a12"][2] is False


# ────────────────────────────────────────────────────────────────────
# Iter 15: M72 — a6 registered (was defined + dispatch-handled but unreachable)
# ────────────────────────────────────────────────────────────────────

def test_a6_wall_time_is_registered():
    """M72: a6 was defined + had a run_audits special-case dispatch but
    wasn't in AUDIT_REGISTRY, so `row not in AUDIT_REGISTRY: continue`
    skipped it — `--rows a6` returned `status=n/a, hint=unknown row a6`
    on a row that actually works. Regression test to keep it registered.
    """
    from ralph_runner.audit import AUDIT_REGISTRY
    assert "a6" in AUDIT_REGISTRY
    title, fn, required = AUDIT_REGISTRY["a6"]
    assert title == "Wall time vs baseline"
    # a6 is warn-only (a bad wall-time shouldn't fail the whole audit).
    assert required is False
    # Function reference must be the real audit_a6_wall_time.
    from ralph_runner.audit import audit_a6_wall_time
    assert fn is audit_a6_wall_time


def test_a6_dispatches_through_run_audits():
    """End-to-end: passing `--rows a6` to run_audits with a convert_wall_s
    must return a status (pass/warn/fail) — NOT `n/a, unknown row a6`."""
    from ralph_runner.audit import run_audits
    # Fresh tmp dir — a6 doesn't need a real model (it only checks wall times).
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        results = run_audits(
            Path(d), ["a6"], convert_wall_s=100.0, baseline_wall_s=80.0,
        )
        a6 = results["rows"]["a6"]
        assert a6["status"] != "n/a", f"a6 should dispatch, got {a6}"
        assert "unknown row" not in a6.get("hint", "")


def test_a6_baseline_none_returns_ok():
    """First-run behaviour: no baseline established yet — a6 should
    return ok with the wall time recorded, not fail."""
    from ralph_runner.audit import audit_a6_wall_time
    r = audit_a6_wall_time(convert_wall_s=42.0, baseline_s=None)
    assert r["status"] == "pass"
    assert r["wall_s"] == 42.0


def test_a6_over_150pct_baseline_fails():
    from ralph_runner.audit import audit_a6_wall_time
    r = audit_a6_wall_time(convert_wall_s=200.0, baseline_s=100.0)
    assert r["status"] == "fail"
    assert r["wall_s"] == 200.0


def test_a6_within_baseline_passes():
    from ralph_runner.audit import audit_a6_wall_time
    r = audit_a6_wall_time(convert_wall_s=110.0, baseline_s=100.0)
    assert r["status"] == "pass"
    # ratio is recorded
    assert r["ratio"] == 1.1


# ────────────────────────────────────────────────────────────────────
# Iter 15: M77 — a2 accepts chat_template.json as a third template form
# ────────────────────────────────────────────────────────────────────

class _NoChatTemplateTokenizer:
    """Mock tokenizer with no chat_template attribute — triggers the
    first-clause code path in audit_a2_chat_template that decides between
    the three file-based template forms."""
    chat_template = None


def test_a2_accepts_chat_template_json_file(tmp_path, monkeypatch):
    """M77: a model shipping only chat_template.json (newer HF convention,
    e.g. Qwen3-VL) was mis-graded as `n/a`. Now a2 treats it as the third
    valid chat-template form alongside inline + .jinja.

    We stub load_tokenizer so the test doesn't depend on having real HF
    tokenizer bytes — the regression we pin is the file-discovery logic
    inside audit_a2_chat_template.
    """
    import ralph_runner.audit as audit_mod
    monkeypatch.setattr(audit_mod, "load_tokenizer",
                        lambda _d: _NoChatTemplateTokenizer())

    d = tmp_path / "model"
    d.mkdir()
    (d / "chat_template.json").write_text(
        '{"chat_template":"{% for m in messages %}{{m.content}}{% endfor %}"}')

    r = audit_mod.audit_a2_chat_template(d)
    # With no inline template AND chat_template.json present, the old code
    # returned `n/a, "no chat template present in source"` and bailed.
    # The fix: a2 now progresses past the guard. The downstream
    # apply_chat_template call will still fail on our mock tokenizer (it's
    # a stub with no apply_chat_template method) → status=fail from the
    # downstream exception handler. That's the expected new behaviour:
    # we surface a real problem instead of silently skipping.
    assert r["status"] != "n/a" or "no chat template present" not in r.get("hint", ""), \
        f"chat_template.json must not be treated as 'no template'. got {r}"


def test_a2_still_na_when_no_template_anywhere(tmp_path, monkeypatch):
    """Negative: when none of the three forms exist, a2 still returns n/a.
    Defends against accidentally making a2 'always-passes' via the iter-15 fix.
    """
    import ralph_runner.audit as audit_mod
    monkeypatch.setattr(audit_mod, "load_tokenizer",
                        lambda _d: _NoChatTemplateTokenizer())

    d = tmp_path / "model"
    d.mkdir()
    # No chat_template.jinja, no chat_template.json, no inline
    r = audit_mod.audit_a2_chat_template(d)
    assert r["status"] == "n/a", f"got {r}"
    assert "no chat template present" in r.get("hint", "")


# ────────────────────────────────────────────────────────────────────
# Iter 16: M78 — a9 accepts structured ↔ string special_tokens equivalence
# ────────────────────────────────────────────────────────────────────

def test_a9_value_normalizer_plain_string():
    from ralph_runner.audit import _normalize_special_token_value
    assert _normalize_special_token_value("<s>") == "<s>"
    assert _normalize_special_token_value("") == ""


def test_a9_value_normalizer_structured_dict():
    from ralph_runner.audit import _normalize_special_token_value
    v = {"content": "<s>", "lstrip": False, "normalized": False,
         "rstrip": False, "single_word": False}
    assert _normalize_special_token_value(v) == "<s>"


def test_a9_value_normalizer_unrecognized_shape():
    """Shapes HF doesn't emit must return None so callers fall back to
    strict equality — we must not silently pass on a corrupted token file."""
    from ralph_runner.audit import _normalize_special_token_value
    assert _normalize_special_token_value(None) is None
    assert _normalize_special_token_value(42) is None
    assert _normalize_special_token_value([1, 2]) is None
    assert _normalize_special_token_value({"no_content_key": "x"}) is None
    assert _normalize_special_token_value({"content": 42}) is None  # non-string content


def _make_dir_with_tokens(root, name, content):
    d = root / name
    d.mkdir()
    (d / "special_tokens_map.json").write_text(json.dumps(content))
    return d


def test_a9_passes_when_structured_source_saved_as_string(tmp_path):
    """M78 regression: source has structured {"content": "<s>", ...}, output
    saved plain "<s>" — these are semantically equivalent but the old `!=`
    comparison false-failed. Required=True on this row means the old bug
    would have marked every such convert as FAILED in Ralph's matrix."""
    from ralph_runner.audit import audit_a9_special_tokens
    src = _make_dir_with_tokens(tmp_path, "src", {
        "bos_token": {"content": "<s>", "lstrip": False, "normalized": False,
                      "rstrip": False, "single_word": False},
        "eos_token": {"content": "</s>", "lstrip": False, "normalized": False,
                      "rstrip": False, "single_word": False},
    })
    out = _make_dir_with_tokens(tmp_path, "out", {
        "bos_token": "<s>",
        "eos_token": "</s>",
    })
    r = audit_a9_special_tokens(out, source_dir=src)
    assert r["status"] == "pass", f"structured→string must match, got {r}"
    assert sorted(r["preserved_keys"]) == ["bos_token", "eos_token"]


def test_a9_passes_when_string_source_saved_as_structured(tmp_path):
    """Reverse direction: source string, output structured."""
    from ralph_runner.audit import audit_a9_special_tokens
    src = _make_dir_with_tokens(tmp_path, "src", {"bos_token": "<s>"})
    out = _make_dir_with_tokens(tmp_path, "out", {
        "bos_token": {"content": "<s>", "lstrip": False, "normalized": False,
                      "rstrip": False, "single_word": False},
    })
    r = audit_a9_special_tokens(out, source_dir=src)
    assert r["status"] == "pass", f"string→structured must match, got {r}"


def test_a9_still_fails_on_genuine_content_mismatch(tmp_path):
    """Negative test: when the content strings actually differ, a9 must
    still fail. Defends against accidentally making a9 always-pass via the
    iter-16 normalization."""
    from ralph_runner.audit import audit_a9_special_tokens
    src = _make_dir_with_tokens(tmp_path, "src", {
        "bos_token": {"content": "<s>"},
    })
    out = _make_dir_with_tokens(tmp_path, "out", {
        "bos_token": {"content": "<DIFFERENT>"},
    })
    r = audit_a9_special_tokens(out, source_dir=src)
    assert r["status"] == "fail"
    assert any(m["source"] == "<s>" and m["output"] == "<DIFFERENT>"
               for m in r["mismatched"])


def test_a9_fails_on_missing_key_unchanged(tmp_path):
    """Other failure mode — missing key — must still be detected unchanged."""
    from ralph_runner.audit import audit_a9_special_tokens
    src = _make_dir_with_tokens(tmp_path, "src", {
        "bos_token": "<s>",
        "eos_token": "</s>",
    })
    out = _make_dir_with_tokens(tmp_path, "out", {"bos_token": "<s>"})
    r = audit_a9_special_tokens(out, source_dir=src)
    assert r["status"] == "fail"
    assert "eos_token" in r["missing"]


def test_a9_unnormalizable_shape_preserved_as_strict_comparison(tmp_path):
    """If HF ever ships a NEW shape (e.g. content as bytes, extra wrapper
    layer), we must fall back to strict equality rather than silently pass.
    This defends against future schema drift letting a real mismatch through."""
    from ralph_runner.audit import audit_a9_special_tokens
    # Two values that both fail the normaliser (non-dict-with-content) and
    # are also unequal under strict ==
    src = _make_dir_with_tokens(tmp_path, "src", {"weird": {"wrapper": "a"}})
    out = _make_dir_with_tokens(tmp_path, "out", {"weird": {"wrapper": "b"}})
    r = audit_a9_special_tokens(out, source_dir=src)
    assert r["status"] == "fail"
    assert "unnormalizable" in r


def test_a2_accepts_chat_template_jinja_file(tmp_path, monkeypatch):
    """Regression test for the pre-existing .jinja path — iter 15's fix
    must not have broken the middle of three valid forms."""
    import ralph_runner.audit as audit_mod
    monkeypatch.setattr(audit_mod, "load_tokenizer",
                        lambda _d: _NoChatTemplateTokenizer())
    d = tmp_path / "model"
    d.mkdir()
    (d / "chat_template.jinja").write_text("{% for m in messages %}{{m.content}}{% endfor %}")
    r = audit_mod.audit_a2_chat_template(d)
    # Same reasoning as the chat_template.json test: must not be the "no template" n/a.
    assert r["status"] != "n/a" or "no chat template present" not in r.get("hint", "")

"""Verify inspect-source returns valid JSON with expected keys."""
import json
import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_qwen"


def test_inspect_source_prints_valid_json():
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inspect-source", "--json", str(FIXTURE)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    assert data["model_type"] == "qwen3_5_moe"
    assert data["is_moe"] is True
    assert data["num_experts"] == 8
    assert data["dtype"] in ("bfloat16", "float16", "float8_e4m3fn", "unknown")
    assert "jangtq_compatible" in data
    assert data["jangtq_compatible"] is True   # qwen3_5_moe is in the v1 whitelist
    assert "is_video_vl" in data
    assert data["is_video_vl"] is False   # tiny_qwen fixture has no video_preprocessor_config.json
    assert "has_generation_config" in data


def test_inspect_source_video_vl_false_for_non_video_fixture():
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inspect-source", "--json", str(FIXTURE)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    assert data["is_video_vl"] is False
    assert "num_hidden_layers" in data
    assert data["num_hidden_layers"] == 2   # tiny_qwen fixture value


def test_inspect_source_missing_config_errors(tmp_path):
    (tmp_path / "README").write_text("nope")
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inspect-source", "--json", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode != 0
    assert "config.json" in r.stderr.lower()


def _assert_clean_error(r: subprocess.CompletedProcess, *, expect_phrase: str) -> None:
    """Shared invariants for inspect-source failure surfaces (M120).

    A well-formed error must:
      1. exit non-zero
      2. NOT print a Python traceback (no `Traceback (most recent call last)`)
      3. include the phrase describing WHAT went wrong so the wizard can
         relay it to the user in plain English
    """
    assert r.returncode != 0
    assert "Traceback" not in r.stderr, (
        "inspect-source leaked a Python traceback — "
        "user would see a cryptic multi-line stacktrace:\n" + r.stderr
    )
    assert expect_phrase in r.stderr.lower(), (
        f"stderr missing phrase {expect_phrase!r}, got:\n{r.stderr}"
    )


def test_inspect_source_malformed_json_errors_cleanly(tmp_path):
    """M120: a corrupt config.json must not crash with a bare JSONDecodeError."""
    (tmp_path / "config.json").write_text("{ this is not json")
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inspect-source", "--json", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    _assert_clean_error(r, expect_phrase="config.json")


def test_inspect_source_empty_config_errors_cleanly(tmp_path):
    """M120: empty config.json (zero-byte or whitespace only) is a common disk-
    failure mode — don't surface `Expecting value: line 1 column 1 (char 0)`."""
    (tmp_path / "config.json").write_text("")
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inspect-source", "--json", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    _assert_clean_error(r, expect_phrase="config.json")


def test_inspect_source_non_dict_config_errors_cleanly(tmp_path):
    """M120: valid JSON but not an object (e.g. someone dumps a list) used to
    AttributeError on `cfg.get(...)` — must fail with a clean diagnostic."""
    (tmp_path / "config.json").write_text("[1, 2, 3]")
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inspect-source", "--json", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    _assert_clean_error(r, expect_phrase="config.json")

"""M130 (iter 52): peer-helper parity between load_jang_model and load_jang_vlm_model.

Iter 51's meta-checklist applied to the two public loader entrypoints:
  - load_jang_model        → text path, routes to _load_jang_v1 or _load_jang_v2
  - load_jang_vlm_model    → VL path,   routes to _load_jang_v1_vlm or _load_jang_v2_vlm

Both validate the jang_config's format tag, but only the text path also
validates format_version. A future JANG v3 would load through v2_vlm with
confused internal errors instead of a clean "unsupported version" message.

Error-path parity: both entrypoints should raise ValueError with an
informative message for malformed or unsupported configs.
"""
import json
import sys
import subprocess
from pathlib import Path

import pytest


def _write_cfg(tmp_path: Path, cfg: dict) -> Path:
    """Create a minimal model dir with just a jang_config.json."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "jang_config.json").write_text(json.dumps(cfg))
    # Minimal config.json so _is_v2_model's format_version check fires.
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    return d


def _call_loader(model_dir: Path, entrypoint: str) -> subprocess.CompletedProcess:
    """Invoke the loader via a subprocess so MLX-availability errors are
    caught at the loader's own import check, not at test-collection time."""
    code = f"""
import sys
from pathlib import Path
from jang_tools.loader import {entrypoint}
try:
    {entrypoint}(Path({str(model_dir)!r}))
    print("LOADED_OK", file=sys.stderr)
    sys.exit(0)
except ValueError as e:
    print(f"VALUE_ERROR: {{e}}", file=sys.stderr)
    sys.exit(2)
except ImportError as e:
    # MLX not available on the test runner — the loader short-circuits
    # before the format checks. That's not a format-parity test signal
    # so we skip it upstream rather than treat it as a failure.
    print(f"IMPORT_ERROR: {{e}}", file=sys.stderr)
    sys.exit(3)
except FileNotFoundError as e:
    print(f"NOT_FOUND: {{e}}", file=sys.stderr)
    sys.exit(4)
except Exception as e:
    print(f"OTHER: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    sys.exit(5)
"""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=False,
    )


def _both_entrypoints_run_or_skip():
    """Ensure MLX is present by doing a quick import probe; otherwise skip
    the parity tests because they need the loader to reach its format gates."""
    r = subprocess.run(
        [sys.executable, "-c", "from jang_tools.loader import load_jang_model, load_jang_vlm_model"],
        capture_output=True, text=True, check=False,
    )
    if r.returncode != 0:
        pytest.skip(f"loader symbols not importable: {r.stderr.strip()}")


def test_text_path_rejects_unsupported_format_version(tmp_path):
    """Baseline: load_jang_model rejects format_version > 2 with a clean ValueError."""
    _both_entrypoints_run_or_skip()
    d = _write_cfg(tmp_path, {
        "format": "jang",
        "format_version": "3.0",  # hypothetical future
        "quantization": {"block_size": 64, "bit_widths_used": [4]},
    })
    r = _call_loader(d, "load_jang_model")
    if r.returncode == 3:  # MLX not available
        pytest.skip("MLX not available in test env")
    assert r.returncode == 2, f"expected VALUE_ERROR, got rc={r.returncode} stderr={r.stderr}"
    assert "unsupported" in r.stderr.lower()
    assert "3.0" in r.stderr


def test_vlm_path_also_rejects_unsupported_format_version(tmp_path):
    """M130: load_jang_vlm_model must match load_jang_model's version guard.
    Pre-iter-52 the VLM path skipped this check — a v3 artifact would try to
    load through _load_jang_v2_vlm and fail with an obscure inner error."""
    _both_entrypoints_run_or_skip()
    d = _write_cfg(tmp_path, {
        "format": "jang",
        "format_version": "3.0",
        "quantization": {"block_size": 64, "bit_widths_used": [4]},
    })
    r = _call_loader(d, "load_jang_vlm_model")
    if r.returncode == 3:
        pytest.skip("MLX not available in test env")
    assert r.returncode == 2, f"expected VALUE_ERROR, got rc={r.returncode} stderr={r.stderr}"
    assert "unsupported" in r.stderr.lower()
    assert "3.0" in r.stderr


def test_text_path_rejects_missing_format(tmp_path):
    """Baseline: load_jang_model rejects jang_config with no 'format' field."""
    _both_entrypoints_run_or_skip()
    d = _write_cfg(tmp_path, {
        "format_version": "2.0",
        # no "format" key
    })
    r = _call_loader(d, "load_jang_model")
    if r.returncode == 3:
        pytest.skip("MLX not available in test env")
    assert r.returncode == 2, f"expected VALUE_ERROR, got rc={r.returncode} stderr={r.stderr}"
    assert "format" in r.stderr.lower()


def test_vlm_path_also_rejects_missing_format(tmp_path):
    """M130 mirror: vlm path rejects missing 'format' too — already present
    but pinned here so any future refactor keeps both sides aligned."""
    _both_entrypoints_run_or_skip()
    d = _write_cfg(tmp_path, {"format_version": "2.0"})
    r = _call_loader(d, "load_jang_vlm_model")
    if r.returncode == 3:
        pytest.skip("MLX not available in test env")
    assert r.returncode == 2, f"expected VALUE_ERROR, got rc={r.returncode} stderr={r.stderr}"
    assert "format" in r.stderr.lower()


def test_both_paths_reject_non_jang_format(tmp_path):
    """Both paths: format=gguf (or any non-JANG string) must raise ValueError
    with a message that includes the bad value."""
    _both_entrypoints_run_or_skip()
    d = _write_cfg(tmp_path, {
        "format": "gguf",
        "format_version": "1.0",
        "quantization": {"block_size": 64, "bit_widths_used": [4]},
    })
    for entry in ("load_jang_model", "load_jang_vlm_model"):
        r = _call_loader(d, entry)
        if r.returncode == 3:
            pytest.skip("MLX not available in test env")
        assert r.returncode == 2, f"{entry}: expected VALUE_ERROR, got rc={r.returncode}"
        assert "gguf" in r.stderr.lower() or "not a jang" in r.stderr.lower()

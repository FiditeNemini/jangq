"""M151 (iter 74): loader.py entry-point and detection helpers must
produce clean diagnostics on corrupt JANG config files.

Pre-iter-74 all 4 entry surfaces (`_is_v2_model`, `_is_vlm_config`,
`load_jang_model`, `load_jang_vlm_model`) did bare `json.loads` and
raised cryptic JSONDecodeError tracebacks when pointed at a model dir
with a malformed config.

Fix:
  - Detection probes (`_is_v2_model`, `_is_vlm_config`) tolerate corrupt
    configs and return False via `_read_config_or_none`.
  - Entry-point loaders (`load_jang_model`, `load_jang_vlm_model`) raise
    ValueError via `_read_config_or_raise` with path + purpose context.

Tests cover both contracts.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest


# ──────────── Detection helpers tolerate corrupt JSON ────────────

def test_is_v2_model_tolerates_corrupt_jang_config(tmp_path):
    """_is_v2_model must return False (not raise) on corrupt jang_config
    so upstream detection can proceed via the fallback checks (standard
    safetensors presence, .jang.safetensors presence)."""
    from jang_tools.loader import _is_v2_model
    d = tmp_path / "model"
    d.mkdir()
    (d / "jang_config.json").write_text("{ this is not json")
    # No safetensors files either → all three detection paths fail → False.
    assert _is_v2_model(d) is False


def test_is_vlm_config_tolerates_corrupt_config_json(tmp_path):
    """_is_vlm_config must return False (not raise) on corrupt config.json.
    Detection probes cannot crash the caller."""
    from jang_tools.loader import _is_vlm_config
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text("not-json")
    assert _is_vlm_config(d) is False


def test_is_vlm_config_tolerates_non_dict_root(tmp_path):
    """`[1,2,3]` as config.json root — detection returns False."""
    from jang_tools.loader import _is_vlm_config
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text("[1,2,3]")
    assert _is_vlm_config(d) is False


# ──────────── Entry-point loaders raise ValueError with path ────────────

def _invoke_loader_subprocess(tmp_path: Path, entrypoint: str, code_template: str) -> subprocess.CompletedProcess:
    """Run the loader in a subprocess so MLX-import failures don't block
    our error-path test. We only care about the pre-MLX config-parse
    error surface."""
    code = f"""
from jang_tools.loader import {entrypoint}
{code_template.format(path=tmp_path)}
"""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=False,
    )


def test_load_jang_model_names_corrupt_jang_config(tmp_path):
    """load_jang_model's ValueError must name the corrupt file + decode
    location. Pre-M151 users got a raw JSONDecodeError traceback with
    no file-path context."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "jang_config.json").write_text("{ broken json")
    r = _invoke_loader_subprocess(
        d, "load_jang_model",
        "try:\n"
        "    load_jang_model(r'{path}')\n"
        "except ValueError as e:\n"
        "    import sys; print('VE:', e, file=sys.stderr); sys.exit(2)\n"
        "except ImportError as e:\n"
        "    import sys; print('IE:', e, file=sys.stderr); sys.exit(3)\n"
    )
    if r.returncode == 3:
        pytest.skip(f"MLX not available: {r.stderr.strip()}")
    assert r.returncode == 2, f"expected ValueError, got rc={r.returncode}\n{r.stderr}"
    assert "JANG config" in r.stderr, f"purpose missing in error: {r.stderr}"
    assert "jang_config.json" in r.stderr
    assert "not valid JSON" in r.stderr


def test_load_jang_vlm_model_names_corrupt_jang_config(tmp_path):
    """Symmetric for load_jang_vlm_model — M130 iter-52 made the VLM
    entry match the text entry's format-guard; iter-74 extends to the
    config-parse error surface too."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "jang_config.json").write_text("{ broken json")
    r = _invoke_loader_subprocess(
        d, "load_jang_vlm_model",
        "try:\n"
        "    load_jang_vlm_model(r'{path}')\n"
        "except ValueError as e:\n"
        "    import sys; print('VE:', e, file=sys.stderr); sys.exit(2)\n"
        "except ImportError as e:\n"
        "    import sys; print('IE:', e, file=sys.stderr); sys.exit(3)\n"
    )
    if r.returncode == 3:
        pytest.skip(f"MLX not available: {r.stderr.strip()}")
    assert r.returncode == 2, f"expected ValueError, got rc={r.returncode}\n{r.stderr}"
    assert "JANG config" in r.stderr
    assert "jang_config.json" in r.stderr
    assert "not valid JSON" in r.stderr

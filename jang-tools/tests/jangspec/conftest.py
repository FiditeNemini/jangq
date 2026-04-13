"""Shared fixtures for jangspec tests."""

import os
from pathlib import Path

import pytest

DEFAULT_FIXTURE = "/Users/eric/jang/models/Gemma-4-26B-A4B-it-JANG_4M"


@pytest.fixture(scope="session")
def jangspec_fixture_model() -> Path:
    """Return the path to a small MoE JANG model for integration tests.

    Set JANGSPEC_TEST_MODEL to override. Skips the test if the path is missing.
    """
    raw = os.environ.get("JANGSPEC_TEST_MODEL", DEFAULT_FIXTURE)
    path = Path(raw)
    if not (path / "config.json").exists():
        pytest.skip(f"fixture MoE model not found at {path}; set JANGSPEC_TEST_MODEL to override")
    return path

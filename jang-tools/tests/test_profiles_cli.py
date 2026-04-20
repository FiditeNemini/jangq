"""Tests for jang_tools.profiles_cli."""
import json
import subprocess
import sys

from jang_tools.profiles_cli import list_profiles


def test_list_profiles_shape():
    data = list_profiles()
    assert "jang" in data
    assert "jangtq" in data
    assert "default_profile" in data
    assert data["default_profile"] == "JANG_4K"
    # JANG must have at least 15 profiles
    assert len(data["jang"]) >= 15
    # JANGTQ must have exactly 3 profiles (JANGTQ2/3/4)
    assert len(data["jangtq"]) == 3


def test_default_profile_marked():
    data = list_profiles()
    defaults = [p for p in data["jang"] if p["is_default"]]
    assert len(defaults) == 1
    assert defaults[0]["name"] == "JANG_4K"


def test_kquant_profiles_marked():
    data = list_profiles()
    kquant = [p["name"] for p in data["jang"] if p["is_kquant"]]
    assert "JANG_4K" in kquant
    assert "JANG_3K" in kquant
    # JANG_2S is NOT K-quant
    jang_2s = next(p for p in data["jang"] if p["name"] == "JANG_2S")
    assert jang_2s["is_kquant"] is False


def test_cli_json_output():
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "profiles", "--json"],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    assert data["default_profile"] == "JANG_4K"
    assert len(data["jangtq"]) == 3


def test_cli_human_output_has_default_tag():
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "profiles"],
        capture_output=True, text=True, check=True,
    )
    assert "JANG_4K" in r.stdout
    assert "[DEFAULT]" in r.stdout

"""M178 (iter 113): webhook_url SSRF validator tests.

Pre-M178 `_fire_webhook` blindly POSTed JSON to any user-provided URL.
Attacker could target loopback / private LAN / instance-metadata
endpoints on the server's network.

Validator rules:
  - Empty URL → valid (no webhook requested).
  - Scheme must be http or https.
  - Hostname must resolve to a public IP (not private/loopback/
    link-local/multicast/reserved/unspecified).

We import the validator directly from server.py without spinning up
the FastAPI app — pure function testing.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

# Load server.py as a module without triggering FastAPI app start.
_spec = importlib.util.spec_from_file_location(
    "jang_server_module",
    Path(__file__).parent.parent / "server.py",
)
# FastAPI app construction happens at import time. That's fine for a
# validator unit test — we just want the function reference.
_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_server)

_validate = _server._validate_webhook_url


def test_validator_accepts_empty_url():
    assert _validate("") is None


def test_validator_accepts_public_https_url():
    # Well-known public target — example.com resolves to a public IP.
    assert _validate("https://example.com/hook") is None


def test_validator_rejects_non_http_scheme():
    assert _validate("file:///etc/passwd") is not None
    err = _validate("file:///etc/passwd")
    assert "http" in err.lower()


def test_validator_rejects_gopher_scheme():
    err = _validate("gopher://attacker.example.com/")
    assert err is not None
    assert "http" in err.lower()


def test_validator_rejects_loopback_ipv4():
    err = _validate("http://127.0.0.1:8080/admin")
    assert err is not None
    assert "loopback" in err.lower() or "private" in err.lower() or "SSRF" in err


def test_validator_rejects_private_10_network():
    err = _validate("http://10.0.0.5/api")
    assert err is not None
    assert "private" in err.lower() or "SSRF" in err


def test_validator_rejects_private_192_168():
    err = _validate("http://192.168.1.1/router/admin")
    assert err is not None
    assert "private" in err.lower() or "SSRF" in err


def test_validator_rejects_aws_metadata_endpoint():
    # The classic cloud-SSRF target. 169.254.169.254 is IANA-assigned
    # link-local and used by AWS/GCP/Azure for instance metadata.
    err = _validate("http://169.254.169.254/latest/meta-data/")
    assert err is not None
    assert "link-local" in err.lower() or "SSRF" in err


def test_validator_rejects_ipv6_loopback():
    err = _validate("http://[::1]/local")
    assert err is not None


def test_validator_rejects_localhost_hostname():
    # `localhost` resolves to 127.0.0.1 in most configs — must also
    # be caught, not just the literal IP.
    err = _validate("http://localhost:8080/admin")
    assert err is not None


def test_validator_rejects_nonexistent_hostname():
    # DNS failure is a validation error — we refuse to accept URLs
    # that can't resolve at submission time.
    err = _validate("http://this-host-definitely-does-not-exist-m178.invalid/hook")
    assert err is not None
    assert "resolve" in err.lower() or "did not" in err.lower()


def test_validator_rejects_missing_hostname():
    err = _validate("http:///no-host")
    assert err is not None

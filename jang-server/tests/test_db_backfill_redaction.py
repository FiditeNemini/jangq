"""M195 (iter 132): data-at-rest backfill for pre-M193/M194 persisted secrets.

M193 (iter 130) and M194 (iter 131) added redact_for_log at WRITE time.
But job rows persisted BEFORE those iters already have unredacted
tracebacks + phase_detail strings sitting on disk in WORK_DIR/jobs.db.
An operator who rotated their HF_UPLOAD_TOKEN after M181's leak still
has the old token embedded in persisted error tracebacks, readable by
anyone with disk access.

M195 adds `_backfill_redact_persisted_logs()` called at startup before
`_load_jobs_from_db`. Idempotent (redact_for_log sentinel doesn't match
any pattern), so re-applying on every restart is safe.

These tests exercise the real function against a temporary DB with
known-dirty rows, and pin the wiring via source inspection.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import tempfile
from pathlib import Path


SERVER_PY = Path(__file__).parent.parent / "server.py"


def _load_server_with_tmp_workdir(tmp_dir: Path):
    """Load server.py as a fresh module with JANG_WORK_DIR pointing at
    tmp_dir. Each test gets a clean DB. Uses a unique module name per
    call so the module cache doesn't reuse a stale DB_PATH."""
    import os
    os.environ["JANG_WORK_DIR"] = str(tmp_dir)
    mod_name = f"jang_server_backfill_{tmp_dir.name}"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, SERVER_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _seed_dirty_row(db_path: Path, row_id: str, phase_detail: str, error: str):
    """Insert a job row with the given dirty fields. Mimics a pre-M193
    row written before redaction existed."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    """)
    data = {
        "job_id": row_id,
        "model_id": "org/model",
        "profile": "JANG_4K",
        "user": "test",
        "priority": 0,
        "phase": "failed",
        "phase_detail": phase_detail,
        "error": error,
    }
    conn.execute(
        "INSERT INTO jobs (id, data, created_at, updated_at) VALUES (?, ?, 0, 0)",
        (row_id, json.dumps(data)),
    )
    conn.commit()
    conn.close()


def test_backfill_redacts_hf_token_in_error():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        db_path = tmp_dir / "jobs.db"
        _seed_dirty_row(
            db_path,
            row_id="rowA",
            phase_detail="Failed during upload",
            error="POST https://hf.co/api failed: token hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 rejected",
        )
        server = _load_server_with_tmp_workdir(tmp_dir)
        server._backfill_redact_persisted_logs()
        # Verify the row was redacted.
        conn = sqlite3.connect(str(db_path))
        (data_str,) = conn.execute(
            "SELECT data FROM jobs WHERE id = ?", ("rowA",)
        ).fetchone()
        conn.close()
        data = json.loads(data_str)
        assert "hf_ABCD" not in data["error"]
        assert "***REDACTED***" in data["error"]


def test_backfill_redacts_webhook_secret_in_phase_detail():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        db_path = tmp_dir / "jobs.db"
        _seed_dirty_row(
            db_path,
            row_id="rowB",
            phase_detail="Webhook delivered to https://hooks.slack.com/services/T0/B0/XXXXXXXXXXXXXXXX",
            error="",
        )
        server = _load_server_with_tmp_workdir(tmp_dir)
        server._backfill_redact_persisted_logs()
        conn = sqlite3.connect(str(db_path))
        (data_str,) = conn.execute(
            "SELECT data FROM jobs WHERE id = ?", ("rowB",)
        ).fetchone()
        conn.close()
        data = json.loads(data_str)
        assert "XXXXXXXXXXXXXXXX" not in data["phase_detail"]
        assert "hooks.slack.com" in data["phase_detail"]  # host kept for diagnostics


def test_backfill_is_idempotent():
    """Running the backfill twice must produce the same result as once.
    Protects against page-cache churn + ensures restart-safe."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        db_path = tmp_dir / "jobs.db"
        _seed_dirty_row(
            db_path,
            row_id="rowC",
            phase_detail="",
            error="Bearer abc.def.ghi1234567890jkl oops",
        )
        server = _load_server_with_tmp_workdir(tmp_dir)
        server._backfill_redact_persisted_logs()
        conn = sqlite3.connect(str(db_path))
        (after_once,) = conn.execute(
            "SELECT data FROM jobs WHERE id = ?", ("rowC",)
        ).fetchone()
        conn.close()
        # Second run should leave the row unchanged.
        server._backfill_redact_persisted_logs()
        conn = sqlite3.connect(str(db_path))
        (after_twice,) = conn.execute(
            "SELECT data FROM jobs WHERE id = ?", ("rowC",)
        ).fetchone()
        conn.close()
        assert after_once == after_twice


def test_backfill_leaves_clean_rows_untouched():
    """Rows that were written AFTER M193/M194 are already clean; the
    backfill should not rewrite them (avoids unnecessary page churn
    on a large DB)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        db_path = tmp_dir / "jobs.db"
        _seed_dirty_row(
            db_path,
            row_id="rowD",
            phase_detail="Completed in 120s",
            error="",  # no exception
        )
        server = _load_server_with_tmp_workdir(tmp_dir)
        server._backfill_redact_persisted_logs()
        conn = sqlite3.connect(str(db_path))
        (data_str,) = conn.execute(
            "SELECT data FROM jobs WHERE id = ?", ("rowD",)
        ).fetchone()
        conn.close()
        data = json.loads(data_str)
        assert data["phase_detail"] == "Completed in 120s"
        assert data["error"] == ""


def test_backfill_handles_corrupt_rows_gracefully():
    """A corrupt row (invalid JSON) must not kill the whole backfill.
    Other rows in the DB should still be processed."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        db_path = tmp_dir / "jobs.db"
        # Seed a good dirty row, then insert a bad one with malformed JSON.
        _seed_dirty_row(
            db_path,
            row_id="rowE_good",
            phase_detail="",
            error="token hf_BADGOODABCDEFGHIJKLMNOPQRSTUVWXYZ0123",
        )
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO jobs (id, data, created_at, updated_at) VALUES (?, ?, 0, 0)",
            ("rowE_bad", "not valid json{"),
        )
        conn.commit()
        conn.close()
        server = _load_server_with_tmp_workdir(tmp_dir)
        # Should not raise.
        server._backfill_redact_persisted_logs()
        # Good row should still be redacted.
        conn = sqlite3.connect(str(db_path))
        (data_str,) = conn.execute(
            "SELECT data FROM jobs WHERE id = ?", ("rowE_good",)
        ).fetchone()
        conn.close()
        data = json.loads(data_str)
        assert "hf_BADGOOD" not in data["error"]


def test_backfill_noop_when_db_does_not_exist():
    """DB hasn't been created yet (first run) — backfill should no-op."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        server = _load_server_with_tmp_workdir(tmp_dir)
        # Should not raise even though jobs.db doesn't exist.
        server._backfill_redact_persisted_logs()


def test_backfill_wired_into_startup():
    """Pin: the startup hook calls _backfill_redact_persisted_logs
    BEFORE _load_jobs_from_db. If a future edit swaps the order or
    drops the call, a restart would load unredacted rows into memory
    and expose them via GET /jobs/{id}."""
    content = SERVER_PY.read_text(encoding="utf-8")
    # Find the startup function body.
    idx = content.find("def startup()")
    assert idx != -1, "M195 regression: startup() function removed"
    body_end = content.find("\n\n", idx)
    body = content[idx:body_end]
    backfill_idx = body.find("_backfill_redact_persisted_logs()")
    load_idx = body.find("_load_jobs_from_db()")
    assert backfill_idx != -1, (
        "M195 regression: startup() does NOT call "
        "_backfill_redact_persisted_logs(). Pre-M193/M194 persisted "
        "secrets will remain on disk + be loaded into memory."
    )
    assert load_idx != -1, "M195 regression: _load_jobs_from_db() call removed"
    assert backfill_idx < load_idx, (
        "M195 regression: _backfill_redact_persisted_logs() must be called "
        "BEFORE _load_jobs_from_db() — otherwise the in-memory _jobs dict "
        "carries unredacted data until the next save."
    )

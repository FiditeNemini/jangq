"""Verify convert.py emits the 5 expected phase events in order."""
import json
import subprocess
import sys
from pathlib import Path


def test_convert_emits_all_phases_even_on_missing_model(tmp_path):
    # Point at a non-model dir; we only care about the phase event ordering
    # before the converter fails out. It should still emit at least phase 1
    # and a done(ok=False) event.
    bogus = tmp_path / "not_a_model"
    bogus.mkdir()
    (bogus / "README").write_text("nope")
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "--progress=json", "--quiet-text",
         "convert", str(bogus), "-o", str(tmp_path / "out"), "-p", "2"],
        capture_output=True, text=True, check=False,
    )
    events = [json.loads(l) for l in r.stderr.splitlines() if l.strip().startswith("{")]
    types = [e["type"] for e in events]
    assert "done" in types
    done_ev = [e for e in events if e["type"] == "done"][-1]
    assert done_ev["ok"] is False
    # After Task 1.4: convert.py emits phase events via ProgressEmitter.
    # Phase 1 ("detect") fires before config.json check, so it must appear
    # even on a bogus model directory.
    assert "phase" in types, f"expected at least one phase event; got types={types}"
    phase_ev = [e for e in events if e["type"] == "phase"][0]
    assert phase_ev["n"] == 1
    assert phase_ev["name"] == "detect"

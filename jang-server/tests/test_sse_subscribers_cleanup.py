"""M180 (iter 115): _sse_subscribers slow-leak regression test.

Pre-M180, when an SSE client disconnected, the event-generator's
`finally` block removed the client's queue from
`_sse_subscribers[job_id]` but left the dict ENTRY (with empty list
value) behind. Over thousands of job submissions + subscriber
disconnects, the dict accumulated ghost keys.

Iter-115 fix: when last subscriber leaves, drop the dict key.
Defense-in-depth: /admin/purge also clears subscribers for purged jobs.

This test pins both behaviors via source inspection.
"""
from __future__ import annotations

from pathlib import Path


SERVER_PY = Path(__file__).parent.parent / "server.py"


def test_sse_event_generator_finally_drops_empty_dict_entry():
    """The event_generator's finally block must delete the dict KEY when
    the subscriber list becomes empty, not just remove the queue from it."""
    src = SERVER_PY.read_text(encoding="utf-8")
    # Look for the M180 cleanup pattern: `if not subs and ... del _sse_subscribers[`
    assert "del _sse_subscribers[job_id]" in src, (
        "M180 regression: _sse_subscribers cleanup missing — last-subscriber "
        "departure must delete the dict key, not just remove the queue from "
        "the list. Without this, ghost {job_id: []} entries accumulate."
    )
    assert "if not subs" in src, (
        "M180 regression: cleanup guard `if not subs` missing — should only "
        "delete when the subscriber list is empty (not on every disconnect)."
    )


def test_admin_purge_also_clears_sse_subscribers():
    """/admin/purge cleans _jobs but pre-M180 left _sse_subscribers entries
    pointing at purged jobs. Defense-in-depth fix: also clear them."""
    src = SERVER_PY.read_text(encoding="utf-8")
    # Find the purge function's body — assert it includes _sse_subscribers cleanup.
    purge_start = src.find("def purge_old_jobs")
    assert purge_start != -1, "purge_old_jobs not found"
    # Look at the next ~50 lines for the M180 belt-and-suspenders cleanup.
    purge_body = src[purge_start:purge_start + 2000]
    assert "_sse_subscribers.pop" in purge_body, (
        "M180 defense-in-depth: /admin/purge must also clear "
        "_sse_subscribers entries for purged job IDs (use .pop(jid, None) "
        "inside _sse_lock). A subscriber that hasn't disconnected by purge "
        "time would otherwise leave a stale entry pointing at a purged job."
    )

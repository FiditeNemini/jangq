"""00: Full verification across all 3 quant levels.

Runs the canonical 3-turn multi-turn test on each bundle (MXFP4, JANGTQ4,
JANGTQ2). Used as a smoke test to confirm the runtime stack works end to end.

Run: python3 00_verify_all.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path.home() / ".mlxstudio/models/JANGQ-AI"
BUNDLES = [
    "Nemotron-3-Nano-Omni-30B-A3B-MXFP4",
    "Nemotron-3-Nano-Omni-30B-A3B-JANGTQ4",
    "Nemotron-3-Nano-Omni-30B-A3B-JANGTQ2",
]

EXPECT = {
    "T1": "paris",        # capital of France
    "T2": "berlin",       # of Germany
    "T3_must_have": ["france", "germany"],  # both countries recalled
}


def run_one(bundle_path: Path) -> dict:
    """Run the 3-turn multi-turn test, return pass/fail + outputs."""
    from jang_tools.nemotron_omni_session import OmniSession

    sess = OmniSession(str(bundle_path))
    t0 = time.time()
    r1 = sess.turn("What is the capital of France? Just the city name.",
                   max_tokens=60, temperature=0.0)
    r2 = sess.turn("And of Germany?", max_tokens=60, temperature=0.0)
    r3 = sess.turn("What were the two countries I just asked about?",
                   max_tokens=80, temperature=0.0)
    dt = time.time() - t0

    result = {
        "bundle": bundle_path.name,
        "wallclock_s": round(dt, 1),
        "T1": r1,
        "T2": r2,
        "T3": r3,
        "T1_pass": EXPECT["T1"] in r1.lower(),
        "T2_pass": EXPECT["T2"] in r2.lower(),
        "T3_pass": all(c in r3.lower() for c in EXPECT["T3_must_have"]),
    }
    result["all_pass"] = (
        result["T1_pass"] and result["T2_pass"] and result["T3_pass"]
    )
    return result


def main():
    failures = []
    for name in BUNDLES:
        bundle = ROOT / name
        if not bundle.exists():
            print(f"SKIP {name}: not on disk at {bundle}")
            continue
        print(f"\n{'='*70}\n  Verifying {name}\n{'='*70}", flush=True)
        try:
            r = run_one(bundle)
        except Exception as e:
            print(f"  ❌ EXCEPTION: {type(e).__name__}: {e}")
            failures.append((name, str(e)))
            continue
        print(f"  T1 ({'✅' if r['T1_pass'] else '❌'}): {r['T1']!r}")
        print(f"  T2 ({'✅' if r['T2_pass'] else '❌'}): {r['T2']!r}")
        print(f"  T3 ({'✅' if r['T3_pass'] else '❌'}): {r['T3']!r}")
        print(f"  wallclock: {r['wallclock_s']}s")
        if not r["all_pass"]:
            failures.append((name, "assertion"))

    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    if failures:
        print(f"  ❌ {len(failures)} failure(s):")
        for name, err in failures:
            print(f"     {name}: {err}")
        sys.exit(1)
    print("  ✅ All bundles passed")


if __name__ == "__main__":
    main()

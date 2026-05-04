"""03: Multi-turn multimodal — persistent KV+Mamba cache across turns.

Demonstrates `OmniSession` that holds the full hybrid cache across turns,
so follow-up questions can reference earlier text + images + audio without
replaying them.

Run: python3 03_multimodal_multiturn.py [bundle_path]
"""
from __future__ import annotations

import sys
from pathlib import Path

from jang_tools.nemotron_omni_session import OmniSession


def main():
    bundle = sys.argv[1] if len(sys.argv) > 1 else str(
        Path.home() / ".mlxstudio/models/JANGQ-AI/Nemotron-3-Nano-Omni-30B-A3B-MXFP4"
    )
    sess = OmniSession(bundle)

    # ── Turn 1: factual ──────────────────────────────────────────────────
    print("\n[T1] you> What is the capital of France? Just the city name.")
    r = sess.turn("What is the capital of France? Just the city name.",
                  max_tokens=50, temperature=0.0)
    print(f"[T1] asst> {r}")

    # ── Turn 2: follow-up that requires prior context ────────────────────
    print("\n[T2] you> And of Germany?")
    r = sess.turn("And of Germany?", max_tokens=50, temperature=0.0)
    print(f"[T2] asst> {r}")

    # ── Turn 3: meta-question — cache MUST hold to answer ────────────────
    print("\n[T3] you> What were the two countries I just asked about?")
    r = sess.turn("What were the two countries I just asked about?",
                  max_tokens=80, temperature=0.0)
    print(f"[T3] asst> {r}")

    # ── Multimodal multi-turn (uncomment if you have media files) ────────
    # MEDIA = Path.home() / "sources/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/media"
    # img = MEDIA / "example1a.jpeg"
    #
    # print("\n[T4] you> [image] Describe what you see briefly.")
    # r = sess.turn("Describe what you see briefly.", images=[str(img)],
    #               max_tokens=120)
    # print(f"[T4] asst> {r}")
    #
    # print("\n[T5] you> What was in the image I showed you, in 5 words?")
    # r = sess.turn("What was in the image I showed you, in 5 words?",
    #               max_tokens=40)
    # print(f"[T5] asst> {r}")

    # Reset between unrelated conversations
    # sess.reset()

    print("\n=== Session demo done ===")


if __name__ == "__main__":
    main()

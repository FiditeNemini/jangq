"""04: Interactive multi-turn REPL with media commands.

Type prompts to chat. Special commands:
  /image PATH    queue an image for the next turn
  /audio PATH    queue an audio file for the next turn
  /video PATH    queue a video file for the next turn
  /reset         wipe cache + history (start fresh)
  /quit          exit

Run: python3 04_interactive_repl.py [bundle_path]

This is the same as `python3 -m jang_tools.nemotron_omni_session <bundle>`
but kept here as an example for embedding in other apps.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from jang_tools.nemotron_omni_session import OmniSession


def main():
    bundle = sys.argv[1] if len(sys.argv) > 1 else str(
        Path.home() / ".mlxstudio/models/JANGQ-AI/Nemotron-3-Nano-Omni-30B-A3B-MXFP4"
    )
    sess = OmniSession(bundle)

    pending_images: List[str] = []
    pending_audio: Optional[str] = None
    pending_video: Optional[str] = None

    print("\nReady. Type messages or / commands.\n")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/reset":
            sess.reset()
            pending_images.clear()
            pending_audio = pending_video = None
            print("[reset]")
            continue
        if line.startswith("/image "):
            pending_images.append(line[len("/image "):].strip())
            print(f"[queued image: {pending_images[-1]}]")
            continue
        if line.startswith("/audio "):
            pending_audio = line[len("/audio "):].strip()
            print(f"[queued audio: {pending_audio}]")
            continue
        if line.startswith("/video "):
            pending_video = line[len("/video "):].strip()
            print(f"[queued video: {pending_video}]")
            continue

        reply = sess.turn(
            line,
            images=pending_images or None,
            audio=pending_audio,
            video=pending_video,
            max_tokens=300,
        )
        print(f"asst> {reply}\n")
        pending_images.clear()
        pending_audio = pending_video = None


if __name__ == "__main__":
    main()

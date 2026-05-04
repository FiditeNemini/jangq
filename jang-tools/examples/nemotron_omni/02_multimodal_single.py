"""02: Single-turn multimodal — image / audio / video / mixed.

Loads `OmniChat` once and runs separate calls per modality. Each call is
isolated (no cache carry-over). For multi-turn with cache use script 03.

Run:
  python3 02_multimodal_single.py [bundle_path] \\
      [--image PATH] [--audio PATH] [--video PATH] [--prompt TEXT]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jang_tools.nemotron_omni_chat import OmniChat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "bundle",
        nargs="?",
        default=str(Path.home() / ".mlxstudio/models/JANGQ-AI/Nemotron-3-Nano-Omni-30B-A3B-MXFP4"),
    )
    ap.add_argument("--image", action="append", default=[],
                    help="Path to an image (repeatable)")
    ap.add_argument("--audio", default=None, help="Path to an audio file")
    ap.add_argument("--video", default=None, help="Path to a video file")
    ap.add_argument("--prompt", default="Briefly describe what you see/hear.",
                    help="User prompt")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    print(f"\n=== Loading {Path(args.bundle).name} ===", flush=True)
    chat = OmniChat(bundle_path=args.bundle)

    print(f"\n=== Prompt ===\n{args.prompt}\n", flush=True)
    if args.image:
        print(f"  + {len(args.image)} image(s): {args.image}")
    if args.audio:
        print(f"  + audio: {args.audio}")
    if args.video:
        print(f"  + video: {args.video}")

    out = chat.chat(
        args.prompt,
        images=args.image or None,
        audio=args.audio,
        video=args.video,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(f"\n=== Response ===\n{out}\n", flush=True)


if __name__ == "__main__":
    main()

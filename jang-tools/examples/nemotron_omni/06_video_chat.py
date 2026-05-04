"""06: Native MLX video chat — text + video clip.

Uses NemotronHOmni's native MLX video path: imageio[ffmpeg] frame extraction,
bicubic resize, CLIP normalize, T=2 temporal stacking, RADIO video_embedder
forward, optional EVS pruning, mlp1 projection, LLM chat.

Run: python3 06_video_chat.py [bundle_path] [video_path] [prompt]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx

from jang_tools.nemotron_omni.model import NemotronHOmni


def main():
    bundle = sys.argv[1] if len(sys.argv) > 1 else str(
        Path.home() / ".mlxstudio/models/JANGQ-AI/Nemotron-3-Nano-Omni-30B-A3B-MXFP4"
    )
    video = sys.argv[2] if len(sys.argv) > 2 else str(
        Path.home() / "sources/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/media/demo.mp4"
    )
    prompt = sys.argv[3] if len(sys.argv) > 3 else \
        "Briefly describe what happens in this video."

    print(f"\n=== Loading {Path(bundle).name} ===")
    t0 = time.time()
    chat = NemotronHOmni(bundle, dtype=mx.float32)
    print(f"  loaded in {time.time()-t0:.1f}s")

    print(f"\n=== Native video chat ===")
    print(f"  prompt: {prompt}")
    print(f"  video:  {video}")
    t0 = time.time()
    reply = chat.turn(
        prompt, video=video,
        video_target_frames=8,         # 8 frames → 4 temporal groups
        video_apply_evs=True,          # drop 70% of redundant tokens
        max_tokens=300, temperature=0.0,
    )
    print(f"  ({time.time()-t0:.1f}s) reply:\n{reply}\n")


if __name__ == "__main__":
    main()

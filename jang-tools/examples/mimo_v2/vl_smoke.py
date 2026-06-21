"""End-to-end image smoke for MiMo-V2.5 JANG bundles (self-contained loader)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from jang_tools.mimo_v2.vlm.load import generate_vl, load_vlm


def make_test_image(path: Path) -> None:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 200, 200], fill="red")
    d.ellipse([240, 240, 410, 410], fill="blue")
    img.save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--prompt", default="Describe the shapes and their colors in this image briefly.")
    parser.add_argument("--max-tokens", type=int, default=80)
    args = parser.parse_args()

    from PIL import Image

    image_path = args.image
    if image_path is None:
        image_path = Path("/tmp/mimo_vl_smoke.png")
        make_test_image(image_path)

    t0 = time.monotonic()
    model, processor = load_vlm(args.bundle)
    print(f"loaded in {time.monotonic() - t0:.0f}s", flush=True)

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": args.prompt},
        ]}
    ]
    t1 = time.monotonic()
    text = generate_vl(model, processor, messages, images=[Image.open(image_path)], max_tokens=args.max_tokens)
    print(f"gen in {time.monotonic() - t1:.0f}s")
    print("OUTPUT:", repr(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

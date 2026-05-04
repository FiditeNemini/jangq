"""Native (numpy/PIL) image preprocessing for Nemotron-3-Nano-Omni.

Re-implements the source `image_processing.py` in pure numpy + PIL — no
PyTorch / torchvision dependency. Output is a `pixel_values` numpy array
of shape (num_tiles, 3, H, W) that the RADIO ViT consumes.

Reference: NVLM 1-D dynamic-tile preprocessing
  - SigLIP normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    Wait — actually the source uses the OpenAI CLIP norm:
    mean=[0.48145466, 0.4578275, 0.40821073]
    std=[0.26862954, 0.26130258, 0.27577711]
  - Force image size: 512
  - Patch size: 16 → 32×32 = 1024 patches per tile
  - Tile picker: `dynamic_preprocess` chooses N tiles based on aspect ratio
    so each tile is square (or near square) at 512×512
  - Optional thumbnail at start (use_thumbnail=True)
  - 1-D tagging: tiles are concatenated linearly (left-to-right or
    top-to-bottom) so the LLM sees them in order
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import Image

# OpenAI CLIP normalization (matches the source `norm_mean` / `norm_std`)
NORM_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
NORM_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    """Pick the (cols, rows) tile grid whose aspect best matches the input."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # Tie-break: prefer the ratio that gives more total tiles up to area
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    *,
    image_size: int = 512,
    min_num: int = 1,
    max_num: int = 12,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    """NVLM-style dynamic tiling. Returns a list of PIL images (the tiles)."""
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h

    # Build candidate (cols, rows) ratios where cols * rows in [min_num, max_num]
    target_ratios = sorted(
        {(c, r) for n in range(min_num, max_num + 1)
         for c in range(1, n + 1) for r in range(1, n + 1)
         if min_num <= c * r <= max_num},
        key=lambda x: x[0] * x[1],
    )
    cols, rows = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_w, orig_h, image_size,
    )

    target_w = image_size * cols
    target_h = image_size * rows
    blocks = cols * rows
    resized = image.resize((target_w, target_h), Image.BICUBIC)

    tiles: List[Image.Image] = []
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        tiles.append(resized.crop(box))

    if use_thumbnail and blocks != 1:
        thumb = image.resize((image_size, image_size), Image.BICUBIC)
        tiles.append(thumb)

    return tiles


def preprocess_images(
    images: List[Image.Image],
    *,
    image_size: int = 512,
    min_num: int = 1,
    max_num: int = 12,
    use_thumbnail: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """Process one or more PIL images into model-ready tile pixel values.

    Returns:
        pixel_values: float32 numpy array of shape (total_tiles, 3, H, W),
            mean+std normalized.
        tile_counts: list of int, number of tiles per input image.
    """
    all_tiles: List[Image.Image] = []
    tile_counts: List[int] = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        tiles = dynamic_preprocess(
            img, image_size=image_size, min_num=min_num, max_num=max_num,
            use_thumbnail=use_thumbnail,
        )
        all_tiles.extend(tiles)
        tile_counts.append(len(tiles))

    # Stack to (N, 3, H, W) and normalize
    arr = np.stack([np.asarray(t, dtype=np.float32) / 255.0 for t in all_tiles], axis=0)
    arr = arr.transpose(0, 3, 1, 2)  # (N, 3, H, W)
    # Normalize: (x - mean) / std, broadcast over channels
    mean = NORM_MEAN.reshape(1, 3, 1, 1)
    std = NORM_STD.reshape(1, 3, 1, 1)
    arr = (arr - mean) / std
    return arr.astype(np.float32), tile_counts

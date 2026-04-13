"""
jang-spec — SSD-streamed MoE speculative-decoding bundle format.
Created by Jinho Jang (eric@jangq.ai).

See docs/superpowers/specs/2026-04-13-jang-spec-design.md.
"""

from .format import (
    BUNDLE_VERSION,
    MANIFEST_FILENAME,
    INDEX_FILENAME,
    HOT_CORE_FILENAME,
    EXPERT_FILE_PATTERN,
    EXPERT_FILE_BYTES_MAX,
    BLOB_ALIGNMENT,
    BLOB_MAGIC,
    INDEX_MAGIC,
)

__all__ = [
    "BUNDLE_VERSION",
    "MANIFEST_FILENAME",
    "INDEX_FILENAME",
    "HOT_CORE_FILENAME",
    "EXPERT_FILE_PATTERN",
    "EXPERT_FILE_BYTES_MAX",
    "BLOB_ALIGNMENT",
    "BLOB_MAGIC",
    "INDEX_MAGIC",
]

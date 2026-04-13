//
// JANGCore — v2 JANG bundle loader in pure Swift.
// Created by Eric Jang (eric@jangq.ai).
//
// This module is format-focused: it parses a .jangspec bundle's
// manifest, expert index, and individual expert blobs. It does NOT
// dequantize, allocate Metal buffers, or run any inference. Those
// responsibilities belong to later plans.
//
// See docs/superpowers/specs/2026-04-13-jang-spec-design.md §5, §8.1.
//

import Foundation

public enum JANGCore {
    public static let version = "0.1.0"
}

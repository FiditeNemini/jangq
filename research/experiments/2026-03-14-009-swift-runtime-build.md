# Experiment 009: Swift Runtime First Build

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: PASS — builds clean, CLI reads JANG model

## Setup

- **Swift**: 6.0 (Xcode on macOS Tahoe)
- **Target**: macOS 15+
- **Dependencies**: swift-argument-parser 1.3.0

## Results

| Component | Status |
|-----------|--------|
| Package.swift | Valid, resolves dependencies |
| JANGMetal module (error types, Metal device) | Compiles |
| JANG module (config, safetensors reader, loader) | Compiles |
| JANGCLI module (argument parser CLI) | Compiles |
| `swift build` | Build complete (2.51s) |
| `jang info model/` | Correctly reads JANG model config |

### CLI Output (first successful run)

```
JANG Model Info
──────────────────────────────────
Source: Qwen2.5-0.5B
Format: JANG v1.0
Bits: 2.76 avg (2.5 target)
Block size: 64
Architecture: qwen2
Layers: 24
Hidden: 896
Vocab: 151936
Heads: 14 Q, 2 KV
Head dim: 64
Weights: 170 MB
```

## Architecture

```
JANGRuntime (Swift Package)
├── JANGMetal (Metal device, error types)
│   ├── JANGMetalDevice.swift — GPU device, pipeline management, metallib loading
│   └── JANGError.swift — error types
├── JANG (core library)
│   ├── JANGConfig.swift — model + quantization config parsing
│   ├── SafetensorsReader.swift — mmap-based safetensors file reader
│   ├── JANGLoader.swift — model loading into Metal buffers
│   └── JANG.swift — re-exports JANGMetal
└── JANGCLI (executable)
    └── main.swift — CLI with info subcommand
```

## Issues Fixed During Build

1. **Swift 6 Sendable**: MTLBuffer doesn't conform to Sendable — used `@unchecked Sendable`
2. **Bundle.module**: Not available without SPM resources — rewrote metallib loading to search paths
3. **Circular dependency**: JANGError needed by both modules — placed in JANGMetal (base module)
4. **Metal Toolchain**: Had to download 704MB Metal Toolchain for shader compilation

## Not Yet Implemented

- Actual model weight loading (SafetensorsReader → Metal buffers)
- Forward pass execution (dispatching Metal kernels)
- Tokenizer
- KV cache
- Sampling
- `jang run` command (inference)

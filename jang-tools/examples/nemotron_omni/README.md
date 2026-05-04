# Nemotron-3-Nano-Omni-30B-A3B — Examples & Reference

Concrete runnable scripts for Python (Apple MLX) and Swift (vMLX) agents to
use Nemotron-3-Nano-Omni-30B-A3B-Reasoning across all three quant levels
(MXFP4 / JANGTQ4 / JANGTQ2) with text, image, audio, video, and multi-turn
chat with persistent cache.

## What's verified working today (2026-04-28)

| Capability | MXFP4 | JANGTQ4 | JANGTQ2 |
|---|---|---|---|
| Text-only single-turn | ✅ | ✅ | ✅ |
| Text-only multi-turn (3-turn cache) | ✅ | ✅ | ✅ |
| Image input | ✅ | ✅ | ✅ |
| Audio input | ✅ | ✅ | ✅ |
| Reasoning ON (default `<think>`) | ✅ | ✅ | ✅ |
| Reasoning OFF (`enable_thinking=False`) | ✅ | ✅ | ✅ |
| Multi-turn cache reference back | ✅ | ✅ | ✅ |

Sample multi-turn run (verified with all 3 quant levels):

```
T1 you> What is the capital of France? Just the city name.
T1 asst> Paris

T2 you> And of Germany?
T2 asst> Berlin

T3 you> What were the two countries I just asked about?
T3 asst> France and Germany
```

## Bundles (HuggingFace, public)

```
OsaurusAI/Nemotron-3-Nano-Omni-30B-A3B-MXFP4     22.6 GB  ~113 tok/s  mlx_lm.load
OsaurusAI/Nemotron-3-Nano-Omni-30B-A3B-JANGTQ4   19.9 GB   ~82 tok/s  jang_tools.load_jangtq
OsaurusAI/Nemotron-3-Nano-Omni-30B-A3B-JANGTQ2   12.6 GB   ~85 tok/s  jang_tools.load_jangtq
```

Each bundle is **self-contained**: LLM (quantized) + vision tower (RADIO ViT,
fp16) + sound encoder (parakeet, fp16) + projectors (fp16) + all source `.py`
files. No separate addon download.

## Python — install

```bash
pip install jang_tools transformers torch torchaudio soundfile pillow \
            timm open_clip_torch librosa
```

## Python — text-only fast path (mlx_lm or load_jangtq)

See `01_text_only.py`. Demonstrates:
- Loading via `mlx_lm.load` for MXFP4
- Loading via `jang_tools.load_jangtq.load_jangtq_model` for JANGTQ4 / JANGTQ2
- Both reasoning on / off
- Greedy + temperature sampling

## Python — multimodal single-turn

See `02_multimodal_single.py`. Demonstrates:
- `OmniChat` for one-shot image / audio / video / mixed input
- Custom temperature + top_p
- Drops PyTorch encoder cache after each call

## Python — multimodal multi-turn (recommended)

See `03_multimodal_multiturn.py`. Demonstrates:
- `OmniSession` — KV + Mamba cache persists across turns
- Mix images with later text follow-ups
- `/reset` between unrelated conversations

## Python — interactive REPL

See `04_interactive_repl.py`. Drop-in chat loop with `/image`, `/audio`,
`/video`, `/reset`, `/quit` commands.

## Python — reasoning toggle benchmark

See `05_reasoning_compare.py`. Same prompt with reasoning ON vs OFF,
shows tok/s + answer quality difference.

## Swift — vMLX integration (stage-3, queued)

See `swift/` subdir for the architecture spec + stub files. Today the
Python path is the recommended runtime; Swift native multimodal is queued
work. The text-only Swift path **already works** via existing
`vmlx-swift-lm/Libraries/MLXLLM/Models/NemotronH.swift` for the LLM portion.

## Bundled assets

The Nemotron source `media/` folder ships with these test files:
- `example1a.jpeg`, `example1b.jpeg` — sample images (Chinese gate scene, etc.)
- `table.png`, `tech.png` — sample screenshots
- `2414-165385-0000.wav` — LibriVox speech sample (~5s, 16 kHz mono)
- `demo.mp4` — short video clip

For your own tests, point any image/audio/video script at your own files.

## Architecture summary (one diagram)

```
USER INPUT
   text + image + audio + video
            │
            ▼
   ┌─────────────────────────────────────────┐
   │ STAGE 1 (today, hybrid)                 │
   │                                          │
   │  vision_model (RADIO ViT)  ─PyTorch CPU─┐│
   │     ↓ bf16                              ││
   │   mlp1                                  ││
   │     ↓ image embeds                      ││
   │  sound_encoder (parakeet) ─PyTorch CPU──┤│  → numpy  →  MLX
   │     ↓ bf16                              ││
   │   sound_projection                      ││
   │     ↓ audio embeds                      ││
   │                                          │
   │  ┌────────────────────────────────────┐│
   │  │ MLX (Metal):                       ││
   │  │  • text token IDs → embed lookup   ││
   │  │  • inject image/audio embeds at    ││
   │  │    <image>/<so_embedding> tokens   ││
   │  │  • LLM 52-layer M+E+* hybrid       ││
   │  │  • prefill via inputs_embeds       ││
   │  │  • decode token-by-token           ││
   │  │  • CACHE: persistent across turns  ││
   │  │    - Mamba ArraysCache (M layers)  ││
   │  │    - KVCache (* layers)            ││
   │  └────────────────────────────────────┘│
   └─────────────────────────────────────────┘
                 │
                 ▼
              REPLY
```

Reasoning, FIM, special tokens, and cache mechanics are documented in the
Nemotron-Omni runtime guide alongside this example.

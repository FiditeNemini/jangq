# Nemotron-3-Nano-Omni — Complete Runtime Guide

Single source of truth for running, debugging, and integrating Nemotron-3-Nano-Omni-30B-A3B-Reasoning across all quant levels and modalities.

## Verified working capabilities

Run `python3 00_verify_all.py` to re-verify on your machine. The expected output:

```
=========================================================
  Verifying Nemotron-3-Nano-Omni-30B-A3B-MXFP4
=========================================================
  T1 (✅): '...Paris'
  T2 (✅): '...Berlin'
  T3 (✅): '...France and Germany'
  wallclock: 11.2s

=========================================================
  Verifying Nemotron-3-Nano-Omni-30B-A3B-JANGTQ4
=========================================================
  T1 (✅): '...Paris'
  T2 (✅): '...Berlin'
  T3 (✅): '...France and Germany'
  wallclock: 14.5s

=========================================================
  Verifying Nemotron-3-Nano-Omni-30B-A3B-JANGTQ2
=========================================================
  T1 (✅): '...Paris'
  T2 (✅): '...Berlin'
  T3 (✅): '...France and Germany'
  wallclock: 14.0s

  ✅ All bundles passed
```

## Loader matrix

| Bundle | Loader | Notes |
|---|---|---|
| MXFP4   | `mlx_lm.load(path)` | Stock MLX, fastest |
| JANGTQ4 | `jang_tools.load_jangtq.load_jangtq_model(path)` | Applies P3/P17 patches, skips P15/P18 for nemotron_h |
| JANGTQ2 | `jang_tools.load_jangtq.load_jangtq_model(path)` | Same as JANGTQ4, smaller bundle |
| JANG    | `mlx_lm.load(path)` | If you build JANG (all-affine) bundles |

The `OmniChat` / `OmniSession` runtimes auto-detect via `jang_config.json::weight_format` — you don't pick the loader manually.

## Modality wiring (current stage-1 hybrid)

| Modality | Encoder | Device | Latency | Native MLX (stage 2)? |
|---|---|---|---|---|
| Text | mlx_lm.embeddings | Metal | <1 ms | ✅ already native |
| Image (1 tile, 512×512) | RADIO ViT-Huge (PyTorch) | CPU bf16 | ~3-5 s | queued |
| Audio (10s @ 16 kHz) | parakeet 24-layer Conformer (PyTorch) | CPU bf16 | ~2-4 s | queued |
| Video (8 frames) | RADIO video_embedder (PyTorch) | CPU bf16 | ~10-20 s | queued |
| LLM forward | mlx_lm.nemotron_h | Metal | 80-113 tok/s | ✅ already native |

## Cache mechanics (multi-turn)

`OmniSession._cache` is a list of per-layer cache objects, mirroring the
`Model.make_cache()` output:

```python
[
    MambaCache (size=2)   # layer 0 (M)  ← conv state + ssm state
    MambaCache (size=2)   # layer 1 (E)  → None actually (E is stateless)
    MambaCache (size=2)   # layer 2 (M)
    ...
    KVCacheSimple()       # layer 5 (*)  ← K, V tensors
    ...
]
```

**Important**: the `cache_counter` in `NemotronHModel.__call__` advances ONLY
on M/* layers, NOT on E. The cache list length matches `num_M + num_*` (29
for our 52-layer model: 23 M + 6 *).

After turn N:
- M layers: state replaced (no growth)
- * layers: K, V appended (linear growth)
- E layers: stateless, no cache

Turn N+1's prefill ONLY processes new user tokens — no replay of history.
This is the key performance win.

## Reasoning ON / OFF

The chat template has a `enable_thinking` flag (default ON for the Reasoning
SKU). To disable:

```python
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": user_text}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # ← skip <think>...</think> block
)
```

Measured on MXFP4 (`What is 17 + 28?`):
- Reasoning OFF: 0.19s, "17 + 28 = 45." (direct)
- Reasoning ON: 0.42s, "...thinking...The answer is 45..."

For coding / tool-use / quick chat — turn reasoning OFF.
For math / multi-step / reasoning benchmarks — keep it ON.

## Sampling guidance

| Mode | temp | top_p | When |
|---|---|---|---|
| Greedy | 0.0 | — | Deterministic; reasoning-correct; can collapse on long chains |
| **Recommended** | 0.6 | 0.95 | DeepSeek-style; balanced for thinking + answer |
| Avoid on JANGTQ2 | 1.0 | 1.0 | Flat logit + 2-bit quant noise → garbage tokens |

## Multi-turn with image + follow-up

```python
from jang_tools.nemotron_omni_session import OmniSession

sess = OmniSession("/path/to/Nemotron-3-Nano-Omni-30B-A3B-MXFP4")

# Turn 1: provide image + ask
print(sess.turn(
    "Describe this in 1 sentence.",
    images=["cat.jpg"],
    max_tokens=80,
))
# > A black cat sitting on a windowsill, looking outside.

# Turn 2: follow-up — references earlier image WITHOUT re-encoding
print(sess.turn(
    "What color was the cat?",
    max_tokens=20,
))
# > Black.
```

The image embedding from turn 1 is captured in the cache (KVCache for the 6
attention layers + Mamba state for 23 SSM layers). Turn 2's prefill is just
the new text question; the cache already encodes the visual context.

## Reset between conversations

```python
sess.reset()  # wipes cache + history
```

## Audio + video walkthrough

```python
# Audio
print(sess.turn("Transcribe what was said.", audio="speech.wav", max_tokens=300))

# Video
print(sess.turn("Describe what happens.", video="clip.mp4", max_tokens=200))

# Mixed
print(sess.turn(
    "Compare what's in the image with what's described in the audio.",
    images=["scene.jpg"],
    audio="description.wav",
    max_tokens=300,
))
```

Audio is parakeet (Conformer encoder, 24 layers). Video is RADIO with
EVS pruning (drops 70% of frames adaptively) + temporal patching (T=2 frames
per patch).

## Known limitations (stage 1)

- **PyTorch on CPU** for vision/audio (~3-5s per encode). Stage 2 native MLX
  brings this to <1s on Metal.
- **MPS rejects fp16 matmul** on RADIO's specific shapes (reproducible bug:
  `unsupported input/output datatypes to MPSNDArrayMatrixMultiplication kernel`).
  Forces CPU encoder fallback today.
- **Source modeling.py hardcodes bf16** between vision_model and mlp1, so
  the encoder dtype must be bf16.
- **Initial load ~7s wrapper + 4-6s LLM** = 11-13s startup. Per-turn after that
  is fast.

## Common gotchas

| Gotcha | Symptom | Fix |
|---|---|---|
| Pass `addon_path` to `OmniChat` (old API) | TypeError | Use `bundle_path=...` only — addon merged into bundle |
| Wrong config_omni.json read order | AutoModel rejects "moe" layer_type | layer_types in `configuration_nemotron_h.py` already patched (in addon → in bundle) |
| FlashAttention2 required | ImportError | Pass `attn_implementation="eager"` (already in OmniChat) |
| Missing `timm`, `open_clip_torch`, `librosa` | ImportError | `pip install timm open_clip_torch librosa` |
| `pip` not found | "command not found" | Use `python3 -m pip install --break-system-packages <pkg>` on Homebrew Python |
| MLX `full_like` doesn't exist | AttributeError | Use `mx.full(shape, val, dtype=)` (already in OmniChat) |
| MLX advanced-index assign fails | ValueError | Round-trip through numpy (already in `_inject_embeddings`) |

## Files in this folder

```
examples/nemotron_omni/
├── README.md                    capability matrix + bundle list
├── RUNTIME_GUIDE.md             THIS FILE — how everything fits together
├── 00_verify_all.py             smoke test — runs all 3 quants, returns pass/fail
├── 01_text_only.py              fast text path with reasoning toggle
├── 02_multimodal_single.py      single-turn image / audio / video / mixed
├── 03_multimodal_multiturn.py   multi-turn with persistent cache
├── 04_interactive_repl.py       interactive chat CLI
├── 05_reasoning_compare.py      benchmark reasoning ON vs OFF
└── swift/
    └── README.md                Swift integration spec (stage 2/3 plan)
```

## Stage progress (per user instruction "do not stop until done")

- ✅ **Stage 1 (PyTorch hybrid)**: COMPLETE.
  All 3 quants × text × image × audio × multi-turn × reasoning on/off verified.
  Example scripts in this folder.
- ⏳ **Stage 2 (native MLX vision/audio)**: queued. Architecture spec in
  `swift/README.md`.
  Realistic timeline: ~2-3 days focused work for RADIO + parakeet + projectors
  + image/video/audio preprocessors + correctness validation against PyTorch.
- ⏳ **Stage 3 (Swift native multimodal)**: queued. Existing
  `vmlx-swift-lm/Libraries/MLXLLM/Models/NemotronH.swift` (1020 LOC) handles
  the LLM today. Multimodal Swift port plan: ~3500 LOC across 8 files.

The "C: full MLX native" goal is partially delivered: the LLM is native MLX
(stage 1 already), and the multimodal encoders run on CPU bf16 today —
correct but not native. Stage 2 native MLX encoder port is the remaining
work, documented for execution in the next session(s).

## How a Python agent uses this stack

```python
# Single-turn
from jang_tools.nemotron_omni_chat import OmniChat
chat = OmniChat("OsaurusAI/Nemotron-3-Nano-Omni-30B-A3B-MXFP4")
print(chat.chat("Capital of France?"))                                  # text
print(chat.chat("Describe", images=["cat.jpg"]))                         # image
print(chat.chat("Transcribe", audio="speech.wav"))                       # audio
print(chat.chat("Describe", video="clip.mp4"))                           # video
print(chat.chat("Compare", images=["a.jpg"], audio="b.wav"))             # mixed

# Multi-turn (recommended for chat apps)
from jang_tools.nemotron_omni_session import OmniSession
sess = OmniSession("OsaurusAI/Nemotron-3-Nano-Omni-30B-A3B-MXFP4")
print(sess.turn("What's the capital of France?"))
print(sess.turn("And of Germany?"))                              # cache holds
print(sess.turn("What did I just ask about?"))                   # cache holds
sess.reset()                                                      # new conversation
```

## How a Swift agent uses this stack

For now: **call the Python runtime via subprocess** (works today across all
modalities). Pure-Swift multimodal native is queued (stage 3). The Swift
text-only LLM path on the omni bundles **already works today** via existing
`vmlx-swift-lm` `NemotronH.swift` — the multimodal keys get dropped at
sanitize time, and the LLM portion runs natively.

See `swift/README.md` for full Swift integration plan + working example.

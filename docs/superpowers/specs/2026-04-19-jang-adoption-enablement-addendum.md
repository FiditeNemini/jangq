# JANG Adoption Enablement — Addendum to Production-Readiness Design

**Date:** 2026-04-19
**Author:** Jinho Jang
**Status:** Draft — awaiting user review
**Parent spec:** `docs/superpowers/specs/2026-04-19-ralph-loop-production-readiness-design.md`

**Why this addendum:** the parent spec covers how to *create* JANG models reliably. This one covers how to make JANG models *usable by anyone other than Eric* — which is the actual bar for "production ready." If a user finishes the wizard and gets a `.safetensors` directory they don't know how to use, adoption is zero regardless of how clean the conversion is.

---

## Part 8 — Adoption enablement (the bar the app must meet)

For every converted output, the app must answer seven questions without the user leaving JANG Studio:

| Question | UI action |
|---|---|
| Does this actually work? | **Test Inference** chat pane |
| How do I load it in my code? | **View Usage Examples** (Python/Swift/Server/HF tabs) |
| How do I serve it to agents? | **Serve Locally** (Osaurus HTTP server) |
| How do I publish it? | **Publish to HuggingFace** dialog |
| How do I share it to iOS/Swift apps? | Swift snippets + embedded JANGCore framework |
| Does my model support tool-calls / reasoning / VL? | Capability-aware snippets that show the right API |
| What's the performance? | Live tok/s + peak RAM in Test Inference |

---

## Part 9 — Runtime files physically bundled in `.app`

Current bundle (post-Phase 6) only contains the Python runtime. Updated layout:

```
JANGStudio.app/
  Contents/
    MacOS/
      JANGStudio                    # main SwiftUI app (existing)
      jang-cli                      # NEW: Swift JANGCLI release binary, for power users
    Resources/
      python/                       # existing ~305 MB Python + mlx + jang_tools
      usage-templates/              # NEW
        model-card.md.jinja
        python-snippet.py.jinja
        swift-snippet.swift.jinja
        server-snippet.sh.jinja
        hf-card.md.jinja
      example-assets/               # NEW — tiny test assets for the inference pane
        test-image.jpg              # 224x224 JPEG for image-VL
        test-video-frame.npy        # 16-frame numpy array for video-VL
        test-audio.wav              # for future audio models
    Frameworks/                     # NEW
      JANGCore.framework/           # Swift-callable loader + kernels
      libjang_metal.dylib           # Metal kernel dylib (loads JANGTQMatmul.metal etc.)
```

### Build script changes (`JANGStudio/Scripts/build-python-bundle.sh` + new siblings)

Three new scripts, chained after `build-python-bundle.sh` in CI:

```
Scripts/
  build-python-bundle.sh          # existing
  build-swift-runtime.sh          # NEW — compiles JANGCLI + JANGCore.framework
  build-metal-kernels.sh          # NEW — xcrun metal → .metallib
  build-usage-templates.sh        # NEW — lints Jinja templates + packs into Resources
```

`build-swift-runtime.sh`:
```bash
cd $JANG_ROOT/jang-runtime
swift build -c release --product jang --product jang-core
# Copy JANGCLI binary
cp .build/release/jang "$APP/Contents/MacOS/jang-cli"
# Copy JANGCore as a framework (needs lipo-style packaging)
...
```

`build-metal-kernels.sh`:
```bash
xcrun -sdk macosx metal -c JangV2QuantMatmul.metal -o JangV2QuantMatmul.air
xcrun -sdk macosx metal -c JANGTQMatmul.metal -o JANGTQMatmul.air
xcrun -sdk macosx metal -c JANGTQAffine8Matmul.metal -o JANGTQAffine8Matmul.air
xcrun -sdk macosx metal -c JANGTQDecodeOps.metal -o JANGTQDecodeOps.air
xcrun -sdk macosx metallib *.air -o libjang_metal.metallib
cp libjang_metal.metallib "$APP/Contents/Frameworks/"
```

CI workflow adds these as additional build steps before codesign.

---

## Part 10 — New `jang-tools` CLI commands (Python side)

Three new subcommands, all JSON-output-capable so Swift can consume them cleanly:

### `jang-tools inference`

```
python -m jang_tools inference --model <dir> --prompt "Hello" [--max-tokens N] [--temperature T] [--image PATH] [--video PATH] [--json]
```

- Loads via `jang_tools.loader.load_model()` (dispatches JANG vs JANGTQ automatically).
- Generates up to `--max-tokens` tokens.
- `--json` output: `{"output": "...", "tokens": 20, "tokens_per_sec": 42.3, "peak_rss_mb": 3412, "finish_reason": "stop"}`.
- For VL models: `--image` required; automatically uses `mlx_vlm.utils.generate` path.
- For video models: `--video` triggers `video_processor` fallback (already handled in `load_jangtq_vlm.py`).
- Clean error message if the model's architecture isn't supported by any loader.

### `jang-tools examples`

```
python -m jang_tools examples --model <dir> --lang python|swift|server|hf [--json]
```

Returns a language-specific code snippet, pre-substituted with the model's actual path + capabilities. Uses the `usage-templates/*.jinja` files bundled in `.app`.

- `--lang python`: `from jang_tools.loader import load_model; model, tokenizer = load_model("<path>"); ...`
- `--lang swift`: `import JANGCore; let model = try JANGModel(path: URL(...)); ...`
- `--lang server`: `python -m osaurus serve --model <path> --port 8080` + OpenAI-compatible curl example
- `--lang hf`: HuggingFace usage card with one-liner + full script

Each snippet is capability-aware:
- If model has a chat template → snippet shows `tokenizer.apply_chat_template(messages, ...)`.
- If model has `tool_call_parser` in config → snippet shows a tool-use example.
- If model has `reasoning_parser` or `enable_thinking` → snippet shows a `<think>…</think>` example.
- If model is VL → snippet shows `processor(images=img, text=...)`.

### `jang-tools publish`

```
python -m jang_tools publish --model <dir> --repo org/name [--private|--public] [--json]
```

- Wraps `huggingface_hub.create_repo` + `upload_folder`.
- Auto-generates a `README.md` (HF model card) using the `hf-card.md.jinja` template if one doesn't exist.
- Metadata: quantization family (jang/jangtq), profile, actual bits, source model, license pass-through, tags: `jang`, `jangtq`, `mlx`, `apple-silicon`.
- Respects HF_HUB_TOKEN env var; otherwise errors with "set HF_HUB_TOKEN".

### `jang-tools modelcard`

```
python -m jang_tools modelcard --model <dir> [--output README.md] [--json]
```

Standalone card generator (same output as `publish --dry-run`). Lets the UI's **Generate Model Card** button work without actually pushing to HF.

---

## Part 11 — Wizard UI: Step 5 adoption surface

Replace the current Step 5 "Finish" buttons with a two-row action bar:

**Row 1 — Post-convert actions (always visible after GREEN):**

| Button | Target | Notes |
|---|---|---|
| Test Inference | opens chat pane sheet | See Part 12 |
| View Usage Examples | opens tabbed sheet | Python / Swift / Server / HF |
| Generate Model Card | writes `README.md` to output dir | Uses `jang-tools modelcard` |
| Serve Locally | starts Osaurus process | Requires osaurus installed; if not, prompts install |
| Publish to HuggingFace | upload dialog | See Part 13 |

**Row 2 — File actions (existing):**

| Button | Action |
|---|---|
| Reveal in Finder | `NSWorkspace.activateFileViewerSelecting` |
| Copy Path | Pasteboard |
| Open in MLX Studio | `mlxstudio://open?path=...` (disabled if app missing) |
| Convert Another | reset wizard |
| Finish | close window |

---

## Part 12 — Test Inference chat pane (detailed spec)

Sheet presented over the wizard, not a separate window. Layout:

```
┌──────────────────────────────────────────────────────────────┐
│ Test Inference — <model name>                          [×]   │
├──────────────────────────────────────────────────────────────┤
│ Architecture: qwen3  ·  Profile: JANG_4K  ·  Size: 0.29 GB  │
│ Capabilities: chat · tool-call · reasoning · vision(image)   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [scrolling message view — shows full conversation]          │
│                                                              │
│  [streaming assistant response appears here char-by-char]    │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│ [🖼 image drop zone] [🎬 video drop zone]  [settings ⚙]      │
│ ┌──────────────────────────────────────────────────────┐    │
│ │ Prompt                                                │    │
│ └──────────────────────────────────────────────────────┘    │
│ [Send (⏎)]  tok/s: 42.3  peak RAM: 3.4 GB  [Cancel]         │
└──────────────────────────────────────────────────────────────┘
```

**Behavior:**

- First time opened: spawns `python -m jang_tools inference --model <path> --prompt "..." --json` as a long-lived subprocess with a stdin/stdout pipe protocol (one JSONL message per prompt/response). Streaming tokens arrive as `{"type":"token","text":"Hi"}` events.
- Chat history lives in memory for the session; cleared when sheet closes.
- **Settings gear** opens a popover with sliders for: temperature (0.0-2.0), top-p (0.0-1.0), top-k (0-200), max tokens (8-4096), repetition penalty (1.0-2.0), system prompt text field.
- **Image drop zone** only shown if `detected.isVL` is true.
- **Video drop zone** only shown if `detected.isVideoVL` is true.
- Audio drop zone hidden for v1 (placeholder for future).
- **Pause / Resume** during streaming: SIGSTOP/SIGCONT the generate call.
- **Copy Response** button on each assistant message.
- **Export Conversation** → JSON file with prompts + responses + settings used.

**Failure modes:**

- If bundled Python runtime missing (Debug build without bundle): banner says "Set `JANGSTUDIO_PYTHON_OVERRIDE` to use your own Python" + link to CONTRIBUTING.md.
- If model architecture unsupported: banner says "Inference not yet supported for this architecture; try converting to mlx-lm directly" + link to troubleshooting.
- If OOM during load: banner says "Model requires X GB; system has Y GB" (estimated via existing size estimate).

---

## Part 13 — Usage Examples sheet (capability-aware snippets)

Sheet with 4 tabs. Each tab has: preview (syntax-highlighted), **Copy** button, **Save to file** button.

### Python tab (auto-adapts to model capabilities)

For a dense chat model with tool calls:
```python
from jang_tools.loader import load_model
from mlx_lm import generate

model, tokenizer = load_model("/path/to/JANG-model")

messages = [
    {"role": "system", "content": "You are a helpful assistant with tools."},
    {"role": "user", "content": "What's the weather in Paris?"},
]
tools = [{"type": "function", "function": {"name": "get_weather", ...}}]
prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True)

response = generate(model, tokenizer, prompt=prompt, max_tokens=512)
# If response contains <tool_call>…</tool_call>, parse with jang_tools.tool_parser
```

For a VL model:
```python
from jang_tools.load_jangtq_vlm import load_jangtq_vlm
from mlx_vlm import generate
from PIL import Image

model, processor = load_jangtq_vlm("/path/to/JANGTQ-VL-model")
image = Image.open("photo.jpg")
prompt = "Describe this image in detail."
response = generate(model, processor, image=image, prompt=prompt, max_tokens=200)
```

### Swift tab

```swift
import JANGCore

let modelURL = URL(fileURLWithPath: "/path/to/JANG-model")
let model = try await JANGModel.load(at: modelURL)
let response = try await model.generate(prompt: "Hello", maxTokens: 100)
print(response.text)
```

For JANGTQ:
```swift
let model = try await JANGTQModel.load(at: modelURL)   // same API, different loader
```

### Server tab

```bash
# Osaurus (OpenAI-compatible)
osaurus serve --model /path/to/JANG-model --port 8080 &

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-jang-model",
    "messages": [{"role":"user","content":"Hello"}],
    "stream": true
  }'
```

### HuggingFace tab

Ready-to-paste model card markdown (populates from `jang-tools modelcard --json`):

```markdown
---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B-Base
tags: [jang, mlx, apple-silicon, quantized]
quantization_config:
  family: jang
  profile: JANG_4K
  actual_bits: 4.23
  size_gb: 0.29
---

# Qwen3-0.6B-Base quantized via JANG Studio

## Quick start (Python)

pip install jang[mlx]
...

## Why JANG?

JANG (Jang Adaptive N-bit Grading) uses mixed-precision quantization to
preserve the tensors that matter most…
```

---

## Part 14 — Publish to HuggingFace dialog

- Target repo name: `<org>/<model-name>` — default `{user}/{source-model-name}-JANG-{profile}`
- Visibility: Public / Private radio
- Token source: env var `HF_HUB_TOKEN` (auto-detected) OR "Browse..." to point at a token file
- License: dropdown (apache-2.0 / mit / cc-by-nc-4.0 / other) — defaults from source model if present
- Base model repo: text field (pre-populated from `detected.modelType` + guessed repo)
- Tags: chip editor — pre-populated: `jang`, `mlx`, `apple-silicon`, `quantized`
- Include (checkboxes): model weights (mandatory) · tokenizer (mandatory) · README.md (auto-generate) · chat_template.jinja (mandatory if present) · preprocessor_config.json (mandatory if VL)
- **Estimated upload:** X.Y GB
- **Upload speed:** ~N MB/s (measured from last upload)
- Progress bar during upload + per-file status
- Success: "Published to https://huggingface.co/<repo>" with Copy URL + Open in Browser

---

## Part 15 — Model Card template (`hf-card.md.jinja`)

```jinja
---
license: {{ license | default("apache-2.0") }}
base_model: {{ base_model }}
tags:
  - jang
  - mlx
  - apple-silicon
  - quantized
{% if is_vl %}  - vision-language{% endif %}
{% if is_video_vl %}  - video{% endif %}
{% if has_tool_parser %}  - tool-use{% endif %}
{% if has_reasoning %}  - reasoning{% endif %}
quantization_config:
  family: {{ family }}    # jang or jangtq
  profile: {{ profile }}
  actual_bits: {{ actual_bits }}
  block_size: {{ block_size }}
  size_gb: {{ size_gb }}
---

# {{ model_name }} — JANG {{ profile }}

Quantized from [`{{ base_model }}`]({{ base_model_url }}) via [JANG Studio](https://jangq.ai).

## Quick start

{{ snippet_python }}

## Performance

- Size on disk: **{{ size_gb }} GB** ({{ size_reduction_pct }}% smaller than FP16 source)
- Average bits/weight: **{{ actual_bits }}**
- First-token latency (M3 Ultra): **{{ latency_ms }} ms**
- Steady-state throughput: **{{ tokens_per_sec }} tok/s**

## Capabilities

{% if has_chat_template %}✓ Chat template preserved{% endif %}
{% if has_tool_parser %}✓ Tool-calling supported (parser: `{{ tool_parser }}`){% endif %}
{% if has_reasoning %}✓ Reasoning / thinking supported{% endif %}
{% if is_vl %}✓ Vision (image input){% endif %}
{% if is_video_vl %}✓ Vision (video input){% endif %}

## Why JANG?

JANG uses mixed-precision quantization that gives each tensor class the bits it
needs — attention weights stay at 6-8 bits while expert MLPs compress to 2-4 bits —
so the model stays coherent at drastically smaller sizes. See [FORMAT.md](https://github.com/jjang-ai/jangq/blob/main/FORMAT.md).

## Reproducing this conversion

```
pip install 'jang[mlx]'
python -m jang_tools convert {{ base_model }} -o {{ model_name }} -p {{ profile }}
```

## Model card authored by

[JANG Studio](https://jangq.ai) — auto-generated {{ date }}.
```

---

## Part 16 — Public adoption docs (`docs/adoption/`)

Separate from JANGStudio/docs (which are app-specific). These live at the repo root for ecosystem adoption.

| File | Purpose |
|---|---|
| `docs/adoption/FORMAT.md` | Canonical format spec — supersedes top-level FORMAT.md when v1.0 ships. Extends with JANGTQ. |
| `docs/adoption/PORTING.md` | How to add JANG support to a new inference framework (ollama, llama.cpp, candle, vllm). Covers: format on disk, per-tensor dequant math, block structure, JANGTQ codebook structure. |
| `docs/adoption/EXAMPLES/python.py` | Minimal standalone example. 30 lines. |
| `docs/adoption/EXAMPLES/swift.swift` | Minimal standalone Swift example. |
| `docs/adoption/EXAMPLES/server.md` | How to stand up an OpenAI-compatible JANG server. |
| `docs/adoption/MODEL_CARD_TEMPLATE.md` | Published Jinja template others can reuse. |
| `docs/adoption/README.md` | Entry point: "Here's everything you need to adopt JANG in your project." |

---

## Part 17 — Ralph Loop audit extensions for adoption (A15-A17)

Extend the audit matrix in parent spec Part 3:

| # | Audit | Pass criterion | How |
|---|---|---|---|
| A15 | **Inference works via bundled runtime** | Load converted model with `jang_tools.loader.load_model` or `load_jangtq_vlm`; generate 20 tokens from "Hello, how are you?"; assert no exception + non-empty non-garbage output | Runs remotely on macstudio after each GREEN convert |
| A16 | **Chat template functionally applies** | If `has_chat_template`: apply 2-turn conversation via `tokenizer.apply_chat_template`; generate; assert each role marker appears in response | Skipped if source had no template |
| A17 | **Model card generatable** | `python -m jang_tools modelcard --model <out> --json` returns valid markdown with license + base_model + quantization_config fields | Pure string validation |
| A18 | **Usage examples generatable** | `python -m jang_tools examples --model <out> --lang python --json` returns runnable Python that imports cleanly (`python -c` smoke) | For each lang: python, swift (syntax parse only), server (shell syntax parse), hf |

A15 is the critical one — a conversion that "verifies green" but can't actually infer is the exact failure mode the current 12-row verifier misses.

---

## Part 18 — Implementation phases (addition to parent plan)

### Phase R2.5 (immediate) — A15-A18 audits

- Implement in `ralph_runner/audit.py`
- Extend `runner.py` to call them after every successful convert
- Extend `PROMPT.md` so failed A15-A18 flips combo status back to `failed`
- Smoke: run on Qwen3-0.6B-Base+JANG_4K, confirm all A1-A18 green
- **Must pass before Ralph Loop is autonomous-enabled**

### Phase P1 — New jang-tools CLI commands

- `jang-tools inference` (reuses existing `jang_tools.loader` + `mlx-lm` generate)
- `jang-tools examples` (loads Jinja templates + capability detection)
- `jang-tools modelcard` (calls examples + wraps in HF card frontmatter)
- `jang-tools publish` (huggingface_hub wrapper + auto-modelcard)
- Ship with Python tests — all 4 commands get pytest coverage
- **Ships as `jang-tools v2.5`**

### Phase P2 — Swift runtime bundle

- `build-swift-runtime.sh` compiles JANGCLI + JANGCore.framework
- `build-metal-kernels.sh` compiles .metal → .metallib
- Bundle both into `.app` under `Contents/MacOS/` and `Contents/Frameworks/`
- CI adds these steps before codesign
- **Adds ~30-50 MB to bundle size (from 305 → 335-355 MB)**

### Phase P3 — Step 5 adoption UI

- Test Inference chat pane (SwiftUI sheet with streaming Python subprocess)
- Usage Examples sheet (4 tabs, syntax-highlighted)
- Generate Model Card button
- Publish to HuggingFace dialog
- Serve Locally button
- **Ships as JANG Studio v1.1**

### Phase Q — Public adoption docs

- `docs/adoption/` directory and all 7 files listed in Part 16
- Published alongside the first public v1.0 release

---

## Part 19 — Acceptance criteria for "world adoption ready"

JANG Studio v1.1 is ready to ship publicly when:

- ✅ A user who has never seen JANG before can open the app, point at a HuggingFace model folder, hit Next through all 5 steps, and in under 10 minutes have: a converted model, a working Test Inference pane, a ready-to-paste code snippet, and (optionally) an uploaded HF repo with a proper model card.
- ✅ Test Inference works for: dense chat, MoE chat, image VL, video VL. (JANGTQ-specific archs tested via Ralph tier 3 before release.)
- ✅ Copy-paste from Python tab runs in a clean venv with only `jang[mlx]` installed.
- ✅ Copy-paste from Swift tab compiles in a new Xcode project with only JANG Studio's bundled JANGCore imported.
- ✅ Generate Model Card produces a card that passes HuggingFace's metadata linter.
- ✅ `docs/adoption/PORTING.md` is complete enough that a developer unfamiliar with JANG could add basic support to a new framework in a weekend.
- ✅ Ralph Loop tier-3 runs A15-A18 green for Qwen3.6-35B-A3B (JANG and JANGTQ), MiniMax-M2.7 (JANGTQ), Gemma-4-26B (JANG).

---

## Open questions

1. **Osaurus integration.** Osaurus isn't in `jang-tools` today. Two options: (a) bundle `osaurus` as a Python package in the app, (b) "Serve Locally" button spawns a subprocess pointing at the user's existing osaurus install. Recommendation: (b) — keeps our bundle lean; show an "Install osaurus" button if not found.
2. **Streaming inference UI via subprocess.** The Test Inference pane needs stdout streaming from a long-running Python process. This is similar to `PythonRunner` but the stream is bidirectional (prompts in, tokens out). Worth extracting a new `InferenceRunner` actor, or reuse `PythonRunner` with a different protocol?
3. **Swift JANGCore API surface.** The current `JANGCore` library exports low-level loaders. For the adoption snippets we need a higher-level `JANGModel.load(at:)` + `.generate(prompt:)` API. Do we add that to `JANGCore` (breaks ABI for existing consumers) or a new `JANGCoreHighLevel` module layered on top?
4. **Video example asset.** A real test video is ~MBs. Do we ship a small .mp4 or a pre-decoded numpy array of 16 frames? Recommendation: numpy array (smaller, loads faster).
5. **HF auth token storage.** Storing HF_HUB_TOKEN in Keychain or env var only? Keychain preferred for UX but requires entitlement updates.

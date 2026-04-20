# JANG Studio

**Convert HuggingFace models to JANG — a mixed-precision quantization format for Apple Silicon — with a five-step wizard, live chat preview, and one-click publishing to HuggingFace.**

<!-- TODO: add hero screenshot of the wizard here once captured -->
<!-- ![hero](docs/screenshots/hero.png) -->

## What is JANG?

JANG (Jang Adaptive N-bit Grading) is a mixed-precision quantization format for Apple Silicon. A JANG model is a safetensors directory where each tensor is quantized to the bit count that suits its role — attention weights at 6-8 bits, expert MLPs at 2-4 bits — so the model stays coherent at drastically smaller sizes while running at full MLX speed.

**What JANG Studio does for you:**

- Converts any HuggingFace model to JANG or JANGTQ format (dense, MoE, VL, video-VL, MLA, hybrid-SSM — every architecture).
- Ships a self-contained Python 3.11 + MLX runtime inside the `.app` — no Python install needed.
- Runs a 12-row post-convert audit so you know the output is real before you use it.
- Lets you chat with the converted model inside the app.
- Generates Python/Swift/Server/HuggingFace code snippets for your use case.
- Generates a HuggingFace model card and pushes to Hub in one click.

## Install

### Option 1 — download the signed DMG (recommended)

<!-- TODO: add real release link once CI tags v1.0.0 -->
Download the latest `JANGStudio.dmg` from [Releases](https://github.com/jjang-ai/jangq/releases), drag `JANG Studio.app` into `/Applications`.

**System requirements:**
- macOS 15 (Sequoia) or later
- Apple Silicon (M1, M2, M3, M4 — any Ultra/Max/Pro tier)
- RAM ≥ 1.5× source model size (conversion peaks are high)
- Free disk ≥ output model size × 1.1

### Option 2 — build from source

```bash
git clone https://github.com/jjang-ai/jangq
cd jangq

# Build the bundled Python runtime (one-time, ~3 min, produces 305 MB)
cd JANGStudio
Scripts/build-python-bundle.sh

# Build the .app
xcodegen generate
xcodebuild -project JANGStudio.xcodeproj -scheme JANGStudio -configuration Release build

# Launch
open build/Build/Products/Release/JANGStudio.app
```

## Using the app

<!-- Screenshots captured by Eric on YYYY-MM-DD — replace TODOs with real paths -->

### Step 1 — Source Model
<!-- TODO screenshot: Step 1 wizard view, folder picker -->
Click **Choose Folder…** and pick a HuggingFace model directory (one containing `config.json` and `.safetensors` shards). JANG Studio auto-detects model type, expert count, dtype, and VL/video capability.

### Step 2 — Architecture
<!-- TODO screenshot: Step 2 detected arch card + advanced overrides -->
Review the auto-detected architecture. Use **Advanced overrides** only when detection gets something wrong — rare, usually for brand-new architectures.

### Step 3 — Profile
<!-- TODO screenshot: Step 3 profile picker, JANG/JANGTQ segmented control, preflight panel -->
Pick a **JANG** profile (works on every architecture) or **JANGTQ** profile (Qwen 3.6 + MiniMax today, GLM in v1.1). Pre-flight panel runs 10 checks live as you change options.

### Step 4 — Run
<!-- TODO screenshot: Step 4 running conversion, phase progress bar + fine tensor progress + live logs -->
Live macro + fine progress bars. Streaming log view. Cancel mid-run; partial output stays for inspection.

### Step 5 — Verify & Adopt
<!-- TODO screenshot: Step 5 verifier checklist green + adoption action row -->
12-row checklist proves every config file, tokenizer, chat template, and shard landed. Then the adoption action bar:

- **Test Inference** — chat with the model inside the app
- **View Usage Examples** — copy Python/Swift/Server/HuggingFace snippets tailored to your model's capabilities
- **Generate Model Card** — HF-compatible card auto-written
- **Publish to HuggingFace** — dry-run preview + one-click upload

### Settings (⌘,)
<!-- TODO screenshot: Settings window with 5 tabs -->
Five tabs: General (defaults, naming template), Advanced (Python override, logs, throttling), Performance (thread count, Metal cache), Diagnostics (anonymize, auto-open issues), Updates.

## Using the scripts directly (no app)

The same Python toolkit that powers the app is a standalone pip package:

```bash
pip install 'jang[mlx]'

# Convert a model
python -m jang_tools convert /path/to/HF-model -o /path/to/output -p JANG_4K

# With JSONL progress events (for building your own frontend)
python -m jang_tools --progress=json --quiet-text convert /path/to/HF-model -o /path/to/output -p JANG_4K

# Inspect a source before converting
python -m jang_tools inspect-source --json /path/to/HF-model

# Estimate output size
python -m jang_tools estimate-model --model /path/to/HF-model --profile JANG_4K --json

# List available profiles
python -m jang_tools profiles --json

# After conversion: test inference
python -m jang_tools inference --model /path/to/output --prompt "Hello" --max-tokens 100 --json

# Generate a HuggingFace model card
python -m jang_tools modelcard --model /path/to/output --output README.md

# Generate usage snippets
python -m jang_tools examples --model /path/to/output --lang python --json

# Publish to HuggingFace
export HF_HUB_TOKEN=...
python -m jang_tools publish --model /path/to/output --repo org/model-JANG_4K
```

All 15 subcommands are documented via `python -m jang_tools --help`.

## Using JANG models in your own Swift app

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/jjang-ai/jangq", branch: "main")
],
targets: [
    .executableTarget(
        name: "MyApp",
        dependencies: [.product(name: "JANGKit", package: "jangq")]
    )
]
```

Then:

```swift
import JANGKit

let model = try await JANGKit.Model.load(at: modelURL)
let result = try await model.generate(
    prompt: "Hello",
    config: JANGKit.SamplingConfig(temperature: 0.0, maxTokens: 200)
)
print(result.text)
print("\(result.tokensPerSecond) tok/s, \(result.tokens) tokens")
```

`JANGKit.Model.load(at:)` auto-detects JANG vs JANGTQ from `jang_config.json`.

## Using JANG models in Python

```python
from jang_tools.loader import load_jang_model
from mlx_lm import generate

model, tokenizer = load_jang_model("/path/to/JANG-model")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Hello"}],
    add_generation_prompt=True
)
response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
print(response)
```

For VL models:
```python
from jang_tools.load_jangtq_vlm import load_jangtq_vlm_model
from mlx_vlm import generate
from PIL import Image

model, processor = load_jangtq_vlm_model("/path/to/JANG-VL-model")
image = Image.open("photo.jpg")
response = generate(model, processor, image=image, prompt="Describe.", max_tokens=200)
```

## Serving a JANG model as an OpenAI-compatible server

```bash
pip install osaurus
osaurus serve --model /path/to/JANG-model --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

## Profile cheat sheet

| Bit tier | JANG | JANGTQ |
|---:|:---|:---|
| 1-bit | JANG_1L | — |
| 2-bit | JANG_2S · JANG_2M · JANG_2L | JANGTQ2 |
| 3-bit | JANG_3K · JANG_3S · JANG_3M · JANG_3L | JANGTQ3 |
| 4-bit | **JANG_4K (default)** · JANG_4S · JANG_4M · JANG_4L | JANGTQ4 |
| 5/6-bit | JANG_5K · JANG_6K · JANG_6M | — |

- **JANG** works on every architecture.
- **JANGTQ** (TurboQuant) supports Qwen 3.6 (`qwen3_5_moe`) and MiniMax 2.7 (`minimax_m2`) in v1. GLM JANGTQ is coming in v1.1.
- **K-suffix profiles** (`JANG_3K/4K/5K/6K`) use K-quant style budget-neutral allocation.
- **L-suffix profiles** are the best-quality at a given bit tier.

## Adoption docs for framework authors

Want to add JANG support to your own inference runtime? See:
- [`docs/adoption/README.md`](docs/adoption/README.md) — entry point
- [`docs/adoption/PORTING.md`](docs/adoption/PORTING.md) — on-disk format + dequant math + JANGTQ codebook spec
- [`docs/adoption/EXAMPLES/`](docs/adoption/EXAMPLES/) — runnable Python + Swift examples
- [`FORMAT.md`](FORMAT.md) — canonical format specification

## Publishing guidance

The app's **Publish to HF** button auto-generates a model card with:
- Source model link + license pass-through
- Quantization family, profile, actual bits, size
- Capability tags: vision-language, video, tool-use, reasoning
- Runnable code snippet for Python + Swift
- Link back to JANG Studio

Example published model: <!-- TODO: fill in one real example after first public upload -->

## Getting help

- **Bug report:** Click **Copy Diagnostics** in the app's failure banner. It saves a zip to `~/Desktop/` with plan.json + run.log + events.jsonl + system.json. Open an issue at https://github.com/jjang-ai/jangq/issues and attach the zip.
- **Questions:** open a Discussion on the repo.

## Credits

Created by **Jinho Jang** (`eric@jangq.ai`) · [jangq.ai](https://jangq.ai)

JANG Studio is the native macOS companion to the [`jang`](https://pypi.org/project/jang/) Python package. The format spec is open; every piece of JANG is free to port to any runtime.

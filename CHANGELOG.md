# Changelog

All notable changes to JANG Studio.

## [1.0.1] — Advanced overrides wired through + public JANGTQ runtime

### Fixed

- **JANGTQ runtime import failure on `pip install jang[mlx]`** — 4 turboquant kernels (`tq_kernel`, `gather_tq_kernel`, `fused_gate_up_kernel`, `linear`) were present in the DMG-bundled Python but never tracked in git, so users installing `jang` from PyPI or GitHub hit `ModuleNotFoundError: No module named 'jang_tools.turboquant.tq_kernel'` at `load_jangtq`. Now tracked and shipped publicly.
- **Architecture → Advanced overrides were silent no-ops.** Pre-1.0.1 the "Force dtype" and "Block size" pickers wrote to `plan.overrides` but `CLIArgsBuilder` dropped them on the floor — Python's convert always auto-detected regardless of UI choice. Both now propagate via new `--force-dtype {bf16,fp16,fp8}` and `-b / --block-size N` flags on `jang_tools convert`.

### Added

- **`jang_tools` 2.4.2** CLI flags:
  - `-b / --block-size N` — quantization group size override (0 = auto, the default)
  - `--force-dtype {bf16,fp16,fp8}` — bypass per-tensor safetensors-header sniff. Useful when the header is stripped or mislabeled.

### Notes

- JANGTQ convert scripts (`convert_qwen35_jangtq`, `convert_minimax_jangtq`) still take positional args only — extending them to the new flags is v1.1 territory.
- 13/13 `CLIArgsBuilderTests` pass on the new propagation logic.

---

## [1.0.0] — First public release

The initial signed + notarized `JANGStudio.dmg` — a native macOS wizard that converts HuggingFace models to JANG and JANGTQ formats with zero Python setup.

### Features

- **5-step wizard** — Source → Architecture → Profile → Run → Verify
- **Auto-detects architecture** — llama, qwen3_5_moe, minimax_m2, deepseek_v32, gemma3/4, glm_moe, idefics3, nemotron_h, mistral4, and more (dense, MoE, VL, video-VL, MLA, hybrid-SSM)
- **JANG profiles** — 11 profiles covering 1-bit through 6-bit, every architecture
- **JANGTQ profiles** — TurboQuant codebook (Hadamard + Lloyd-Max) for `qwen3_5_moe` (Qwen 3.6) and `minimax_m2` families at 2 / 3 / 4 bits
- **Bundled Python runtime** — self-contained python-build-standalone 3.11 + MLX + jang-tools inside the `.app`. No system Python required
- **14-row post-convert verifier** — checks `jang_config.json`, tokenizer, chat template (inline / `.jinja` / `.json`), shard/index parity, `generation_config.json`, and capability stamping before you trust the output
- **In-app chat preview** — test the converted model immediately without leaving the wizard
- **One-click HuggingFace publishing** — generates a model card and pushes to Hub
- **Live JSONL progress** — cancellable; each phase reports bytes/tokens/tensor-level progress
- **Diagnostic redaction** — HF tokens, `Bearer` values, URL query secrets, and JSON body secrets are scrubbed from logs + diagnostic bundles

### Requirements

- macOS 15 (Sequoia) or later
- Apple Silicon (M1, M2, M3, M4 — any tier)
- RAM ≥ 1.5× source model size
- Free disk ≥ output model size × 1.1

### Install

Download `JANGStudio.dmg` from [Releases](https://github.com/jjang-ai/jangq/releases), drag `JANG Studio.app` into `/Applications`.

### Build + signing

- Universal binary (arm64 + x86_64)
- Hardened runtime (JIT disabled, `disable-library-validation` enabled for the embedded Python)
- Notarized and stapled by Apple
- Bundle identifier: `ai.jangq.JANGStudio`

### Notes for v1.1

- JANGTQ for `glm_moe_dsa` (GLM-5.1 and siblings) is in progress
- Additional convert-time profiles are being validated before public exposure

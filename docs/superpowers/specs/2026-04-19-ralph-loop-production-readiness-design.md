# Ralph Loop + JANG Studio Production-Readiness — Design Spec

**Date:** 2026-04-19
**Author:** Jinho Jang
**Status:** Draft — awaiting user review

**Goal:** Autonomous test harness that iteratively converts real small-scale models through JANG and JANGTQ on Mac Studio, audits every output dimension, and drives a production-readiness roadmap that eliminates all hardcoded values from the JANG Studio app.

**Strategy:** Start with the smallest models first — fast feedback, tight cycle times, bug discovery before scaling up. Only move to 35B+ real-target models after the small-model matrix is green.

---

## Part 1 — Model inventory (what's available today)

### Locally cached on Eric's MacBook Pro (MacBook-Pro, ~6 GB free in HF cache)

| Source model | Size | Usable for conversion? |
|---|---|---|
| `Qwen/Qwen3.6-35B-A3B` (BF16) | 67 GB | ✅ Large integration target |
| `MiniMaxAI/MiniMax-M2.7` | 12 KB (metadata only) | ❌ Not fully downloaded |
| `mlx-community/Qwen3-0.6B-8bit` | 619 MB | ❌ Already quantized (MLX 8bit is not a JANG source) |
| `mlx-community/Llama-3.2-1B-Instruct-4bit` | 680 MB | ❌ Already quantized |
| `mlx-community/gemma-4-e2b-it-4bit` | small | ❌ Already quantized |

### Cached + local on Mac Studio (erics-mac-studio.local, **97 GB free, tight**)

| Source model | Location | Size | Usable? |
|---|---|---|---|
| `Qwen/Qwen3.6-35B-A3B` (BF16) | HF cache | 67 GB | ✅ Big target |
| `mlx-community/Qwen3-VL-2B-Instruct-4bit` | HF cache | 1.7 GB | ❌ Already quantized |
| `mlx-community/gemma-3n-E2B-it-4bit` | HF cache | 4.2 GB | ❌ Already quantized |
| `/Volumes/EricsLLMDrive/Gemma-4-26B-A4B-it-BF16` | external | 48 GB | ✅ Medium target |
| `/Volumes/EricsLLMDrive/Gemma-4-31B-it-BF16` | external | ~60 GB | ✅ Medium |
| `/Volumes/EricsLLMDrive/MiniMax-M2.7-FP8` | external | 214 GB | ✅ Real JANGTQ target (but huge) |
| `/Volumes/EricsLLMDrive/GLM-5.1-FP8` | external | 704 GB | ❌ Way too big for fast iteration |

### Gap — what's NOT available and needs downloading (small, fast)

| Model | Approx DL size | Why we want it |
|---|---|---|
| `Qwen/Qwen3-0.6B-Base` (BF16) | ~1.2 GB | Smallest dense LLM, fastest JANG convert |
| `meta-llama/Llama-3.2-1B-Instruct` (BF16) | ~2.5 GB | Standard dense instruct path + chat template |
| `Qwen/Qwen3-1.7B-Base` (BF16) | ~3.4 GB | Slightly bigger dense |
| `HuggingFaceTB/SmolVLM-256M-Instruct` | ~500 MB | Image-VL path (`is_vl=True`), smallest VL |
| `Qwen/Qwen2-VL-2B-Instruct` (BF16) | ~4.4 GB | VL + video-preprocessor path (if it ships one) |
| `Qwen/Qwen1.5-MoE-A2.7B-Chat` (BF16) | ~26 GB | **Smallest real MoE** — required to validate MoE path without 67 GB Qwen3.6 |

**Tiered matrix:**

- **Tier 1 — fast (<5 min per convert):** Qwen3-0.6B-Base, Llama-3.2-1B, SmolVLM-256M (<5 GB total download)
- **Tier 2 — medium (5-20 min):** Qwen3-1.7B, Qwen2-VL-2B, Qwen1.5-MoE-A2.7B (~34 GB total)
- **Tier 3 — real (20+ min):** Qwen3.6-35B-A3B (already cached), MiniMax-M2.7-FP8 (already on drive)

Ralph starts at Tier 1. Promotes through tiers only when the prior tier is fully green.

---

## Part 2 — Ralph Loop architecture

### Directory layout

```
jang/
├── ralph-runner/                       # NEW (private — under research/ namespace per memory)
│   ├── README.md
│   ├── models.yaml                     # Source-of-truth test model matrix
│   ├── profiles.yaml                   # Which profiles to exercise per tier
│   ├── runner.py                       # The iteration engine
│   ├── audit.py                        # Audit functions (12 verifier rows + 8 new)
│   ├── inference.py                    # Load converted model + generate + score
│   ├── remote.py                       # SSH/rsync orchestration for macstudio
│   ├── report.py                       # Results aggregation + HTML dashboard
│   ├── results/                        # Per-run JSON + logs (gitignored)
│   │   └── 2026-04-19/
│   │       ├── Qwen3-0.6B-Base__JANG_4K/
│   │       │   ├── plan.json
│   │       │   ├── run.log
│   │       │   ├── events.jsonl
│   │       │   ├── verify.json
│   │       │   ├── audit.json          # New audits (generation, tokenizer RT, speed)
│   │       │   └── sample_output.txt
│   │       └── history.jsonl           # Append-only across all runs
│   └── baselines/                      # Golden outputs to regression-test against
│       └── Qwen3-0.6B-Base__JANG_4K.json
└── docs/superpowers/plans/2026-04-19-ralph-loop.md   # Implementation plan
```

**Language choice:** Pure Python. No Swift. The wizard app consumes a subset of the same signals (JSONL progress), but Ralph talks directly to `jang_tools`.

**Privacy:** Per memory `feedback_no_research_public.md`, `ralph-runner/results/` + `baselines/` are gitignored. The source code (`runner.py`, `audit.py`, etc.) is fine to commit.

### Execution topology

```
  ┌───────────────────────────────────────────────────┐
  │ MacBook Pro (local dev machine)                   │
  │                                                   │
  │  /loop skill fires every 6h                       │
  │  → invokes ralph-runner/runner.py                 │
  │  → runner.py rsync's source tree to macstudio     │
  │  → runner.py SSHes + spawns remote pytest/conv.   │
  │  → streams logs back via SSH                      │
  │  → audits the remote output via SSH               │
  │  → rsync results back to local                    │
  │  → commits to research/ (private gitignored)      │
  │  → optional: open GitHub Issue on failure         │
  └───────────────────────────────────────────────────┘
                       │
                       │ Tailscale SSH (eric@100.76.98.16)
                       ▼
  ┌───────────────────────────────────────────────────┐
  │ Mac Studio (M3 Ultra, 256 GB RAM, 97 GB disk)     │
  │                                                   │
  │  ~/jang-ralph-workspace/                          │
  │    ├── jang/               (rsynced source)       │
  │    ├── out/                (converted models)     │
  │    └── logs/               (run logs)             │
  │                                                   │
  │  Uses existing HF cache in ~/.cache/huggingface/  │
  │  + /Volumes/EricsLLMDrive (read-only)             │
  └───────────────────────────────────────────────────┘
```

### Config files

#### `models.yaml` (no hardcodes — extensible)

```yaml
# ralph-runner/models.yaml — test model registry.
# Tier 1 runs unconditionally every cycle. Tier 2 only after Tier 1 all green.
# Tier 3 only on explicit --tier 3 flag.

tiers:
  - tier: 1
    min_ram_gb: 4
    min_free_gb: 10
    max_convert_minutes: 5
    models:
      - hf_repo: Qwen/Qwen3-0.6B-Base
        family: dense
        archs: [qwen3]
        approx_gb: 1.2
        has_chat_template: false
      - hf_repo: meta-llama/Llama-3.2-1B-Instruct
        family: dense
        archs: [llama]
        approx_gb: 2.5
        has_chat_template: true
        has_tool_parser: true
      - hf_repo: HuggingFaceTB/SmolVLM-256M-Instruct
        family: vl_image
        archs: [idefics3]
        approx_gb: 0.6
        has_chat_template: true
        has_preprocessor_config: true

  - tier: 2
    min_ram_gb: 16
    min_free_gb: 50
    max_convert_minutes: 20
    models:
      - hf_repo: Qwen/Qwen3-1.7B-Base
        family: dense
        archs: [qwen3]
        approx_gb: 3.4
      - hf_repo: Qwen/Qwen2-VL-2B-Instruct
        family: vl_image
        archs: [qwen2_vl]
        approx_gb: 4.4
        has_chat_template: true
        has_preprocessor_config: true
      - hf_repo: Qwen/Qwen1.5-MoE-A2.7B-Chat
        family: moe
        archs: [qwen2_moe]
        approx_gb: 26
        has_chat_template: true

  - tier: 3
    min_ram_gb: 128
    min_free_gb: 150
    max_convert_minutes: 180
    models:
      - local_path: ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/*/
        family: moe_hybrid_ssm
        archs: [qwen3_5_moe]
        approx_gb: 67
        has_chat_template: true
        supports_jangtq: true
      - local_path: /Volumes/EricsLLMDrive/MiniMax-M2.7-FP8
        family: moe_mla
        archs: [minimax_m2]
        approx_gb: 214
        has_chat_template: true
        supports_jangtq: true
        has_custom_py: true                # modeling_*.py + configuration_*.py
```

#### `profiles.yaml` (no hardcodes — one knob = one YAML line)

```yaml
# ralph-runner/profiles.yaml — profiles to exercise per tier.
# Profile list queried from `python -m jang_tools profiles --json` at startup;
# this file only says WHICH of the valid profiles to run, not which ones exist.

tier_1_profiles:
  jang: [JANG_4K, JANG_2S, JANG_6M]     # default, aggressive, near-lossless
  jangtq: []                             # JANGTQ blocked on tier 1 (no whitelisted archs)

tier_2_profiles:
  jang: [JANG_4K, JANG_2S, JANG_4M]
  jangtq: [JANGTQ4]                      # qwen2_moe not whitelisted but near-arch; confirm behavior

tier_3_profiles:
  jang: [JANG_4K, JANG_2L, JANG_4M]
  jangtq: [JANGTQ2, JANGTQ3, JANGTQ4]
```

### Runner lifecycle (single iteration)

```
1. Load models.yaml + profiles.yaml
2. Query `python -m jang_tools profiles --json` → list of valid profile names
3. Build (model, profile) matrix; skip combos that already have a green result within TTL
4. For each (model, profile):
   a. Preflight locally: do we have the model? If not, HF-download to macstudio (only if < DL_BUDGET_GB)
   b. SSH to macstudio, run: python -m jang_tools --progress=json convert <src> -o <out> -p <profile>
   c. Stream stderr (JSONL) back, parse via the same JSONLProgressParser
   d. On successful done: run audit suite (see Part 3) remotely; stream results back
   e. Append to results/<date>/<model>__<profile>/
   f. Delete the output model (out/) to free disk — keep only the audit JSON
   g. Record success/fail in history.jsonl
5. If any failures: open a GitHub issue with diagnostics bundle zip
```

### Disk hygiene

- Mac Studio's 97 GB free is **tight**. Ralph must delete each converted model immediately after audit.
- Pre-flight: `df -g ~` check; if < `min_free_gb` for tier, skip until freed.
- HF-cache-only source models stay cached; external drive models are read-only.

### Ralph cadence (via `/loop` skill)

```
/loop 6h ralph-runner/runner.py --tier 1
```

After Tier 1 is green for 3 consecutive runs, bump to:
```
/loop 12h ralph-runner/runner.py --tier 2
```

Tier 3 runs manually: `ralph-runner/runner.py --tier 3 --one-shot`.

---

## Part 3 — Audit matrix (the actual "full audit")

Beyond the 12 PostConvertVerifier rows (which Ralph also executes), each run also runs:

| # | Audit check | Pass criterion | How |
|---|---|---|---|
| A1 | **Tokenizer round-trip** | `decode(encode(s)) == s` for 20 sample strings | Load tokenizer, roundtrip test strings incl. unicode, special tokens, whitespace edge cases |
| A2 | **Chat template render** | Rendering a 3-turn conversation produces a non-empty string that mentions each role | Use `tokenizer.apply_chat_template` or Jinja2 fallback; regex-assert each role marker appears |
| A3 | **Generation coherence** | "The capital of France is" → completion contains "Paris" within 40 tokens | mlx-lm generate with greedy decode, max 40 tokens |
| A4 | **Tokens/sec throughput** | `tokens_per_sec > prior_run * 0.7` (no >30% regression) | Warm-up 10 tok, measure 100 tok |
| A5 | **Chat-turn end-to-end** | Apply chat template → generate → model responds without infinite-loop | Prompt "Hello, how are you?" — generate 50 tok; assert no repeating substring > 10 chars |
| A6 | **Convert wall time** | `t_convert <= baseline * 1.5` | Measured from first phase event to done |
| A7 | **Size vs estimate** | `|actual_size - predicted_size| / predicted_size <= 0.15` | Compare to PreflightRunner's sizeEstimate |
| A8 | **Tool/reasoning parser preservation** | If source config had `tool_choice_parser`, `tool_call_parser`, `reasoning_parser`, `chat_template_kwargs`, `enable_thinking`, those fields must be present in output `config.json` or `tokenizer_config.json` | Deep diff of the relevant subtrees |
| A9 | **Special tokens preservation** | Every `special_tokens_map.json` key in source also in output, with same value | JSON structural equality |
| A10 | **JANGTQ codebook metadata** | For JANGTQ outputs, `jang_config.quantization.tq_*` fields exist and are non-empty | Parse `jang_config.json`, assert `tq_codebook`, `tq_block_size`, `tq_scales_shape` present |
| A11 | **VL preprocessor functional** | If VL: load `AutoProcessor`, run on a 224x224 test image, assert no errors | `transformers.AutoProcessor.from_pretrained(out); processor(images=test_img)` |
| A12 | **Video preprocessor functional** | If video VL: same but with a test video frame array | Similar to A11, uses `video_processor` if present |
| A13 | **Perplexity regression (tier 2+)** | On a 100-sample slice of C4 or similar, `perplexity <= source_perplexity * 1.15` | Expensive, gated on tier ≥ 2 |
| A14 | **MMLU mini (tier 3 only)** | 50-question MMLU subset: accuracy within 5 pts of source | Uses existing `jang_tools.benchmark_mmlu` |

### Results schema

```json
{
  "run_id": "2026-04-19T18-42-01_Qwen3-0.6B-Base__JANG_4K",
  "model": "Qwen/Qwen3-0.6B-Base",
  "profile": "JANG_4K",
  "host": "erics-mac-studio.local",
  "jang_tools_version": "2.4.1",
  "jang_studio_sha": "94401d6",
  "convert": {
    "wall_time_s": 78.4,
    "peak_rss_mb": 3412,
    "final_output_gb": 0.46,
    "predicted_output_gb": 0.51
  },
  "verifier_12": {"jangConfigExists": "pass", "chatTemplate": "pass", ...},
  "audit_14": {
    "A1_tokenizer_roundtrip": "pass",
    "A3_coherence": {"status": "pass", "output": "...Paris..."},
    "A4_tokens_per_sec": {"status": "pass", "value": 42.3, "baseline": 40.1},
    "A8_parser_preservation": {"status": "n/a", "source_had": false}
  },
  "regression": "none"
}
```

---

## Part 4 — Hardcode elimination roadmap

### Dynamic CLI queries (new `jang-tools` commands)

Three new JSON-output subcommands let Swift (and Ralph, and any future frontend) discover capabilities at runtime instead of hardcoding:

```
python -m jang_tools profiles --json
  # → {"jang": [{"name": "JANG_4K", "bits": [8,4,4], "use": "default", ...}, ...],
  #    "jangtq": [{"name": "JANGTQ2", "bits": 2, "min_source_dtype": ["bf16","fp8"]}, ...]}

python -m jang_tools capabilities --json
  # → {"jangtq_whitelist": ["qwen3_5_moe", "minimax_m2"],
  #    "known_512_expert_types": ["minimax_m2", "glm_moe_dsa"],
  #    "supported_dtypes": ["bfloat16", "float16", "float8_e4m3fn"],
  #    "block_sizes": [32, 64, 128],
  #    "methods": ["mse", "rtn", "mse-all", "awq", "gptq"]}

python -m jang_tools estimate --model <dir> --profile <name> --json
  # → {"predicted_bytes": 493421568, "breakdown": {"attn_bf16": 12M, ...}}
```

### Swift-side consumers

| File | Change |
|---|---|
| `ConversionPlan.swift` | Delete the hardcoded `JANGTQ_V1_WHITELIST` constant. At app startup, fetch via `CapabilitiesService.load()`. Cache to `UserDefaults`. |
| `ProfileStep.swift` | Delete hardcoded `JANG_PROFILES` + `JANGTQ_PROFILES`. Populate from `ProfilesService.current()` at view appear. |
| `PreflightRunner.swift` | Delete hardcoded `KNOWN_512_EXPERT_TYPES` + bit-per-weight estimation. Query `jang-tools estimate` instead (cached per-plan). |
| `ArchitectureStep.swift` | Block-size picker uses `capabilities.block_sizes` instead of `[32, 64, 128]`. |
| `PostConvertVerifier.swift` | Tokenizer class blocklist becomes `capabilities.tokenizer_class_blocklist` (currently only `TokenizersBackend`). |

### Settings pane (currently missing — add entirely)

New `SettingsWindow.swift` opened via `Cmd+,`. Fields:

**General tab:**
- Default output directory parent (NSOpenPanel → URL bookmark persisted)
- Default profile (menu populated from `ProfilesService.current()`)
- Default family (`jang` / `jangtq` if available)
- Default method (menu from `capabilities.methods`)
- Default Hadamard rotation (toggle, warn at 2-bit)
- Default calibration sample count (slider 64…1024, logarithmic)
- Default output naming template (text field, tokens: `{basename}`, `{profile}`, `{family}`, `{date}`, `{time}`, `{user}`)
- Auto-delete partial output on cancel (toggle)
- Reveal output in Finder on finish (toggle)

**Advanced tab:**
- Bundled Python override path (text field; shows current resolution)
- Custom `jang-tools` module path (text field; for `pip install -e` dev mode)
- Log verbosity (segmented: Normal / Verbose / Debug — maps to `--progress=json` + log level)
- JSONL log retention (number of lines to keep in UI, default 10k; slider 1k…50k)
- JSONL log file output dir (defaults `~/Library/Logs/JANGStudio/`)
- Tick throttle ms (slider 50-500ms, controls Python-side `_TICK_MIN_INTERVAL_S` via new CLI flag)
- Max bundle size warning MB (slider 200-1000)

**Performance tab:**
- MLX thread count (slider 1-system-cpu-count)
- Enable Metal pipeline cache (toggle)
- Pre-allocate RAM at start (toggle + size field)
- Convert concurrency (int 1-N; when >1 and queue has N models, run in parallel)

**Diagnostics tab:**
- Copy diagnostics bundle anytime (not just on failure) — toggle that shows the button always
- Anonymize paths in diagnostics (toggle)
- GitHub repo for bug reports (text field, default `https://github.com/jjang-ai/jangq/issues`)
- Auto-open issue tracker on crash (toggle)

**Updates tab (deferred to v1.1 + Sparkle, stubs now):**
- Check for updates (button; wired to Sparkle later)
- Update channel (menu: stable / beta)

### Wizard-step enhancements (beyond Settings)

**Step 1 — Source Model (production)**
- Drag-and-drop folder target
- Recents: last 10 source dirs (UserDefaults)
- "Scan for multiple models" when user selects a parent dir → queue
- HF Hub download dialog (v1.1; stub button + "coming in v1.1")
- Expand detected card: show full `config.json` model_type + detailed arch params

**Step 2 — Architecture (production)**
- Advanced overrides expand to include:
  - Force dtype (already exists)
  - Force block size (already exists)
  - Force group size override (new)
  - Skip tensor patterns (new text area, one pattern per line)
  - Custom calibration dataset (new file picker for `.jsonl`)
  - AWQ alpha slider (0.0-1.0, default 0.5)
  - GPTQ Hessian source path (new dir picker)
  - Quantization method override (dropdown from capabilities)

**Step 3 — Profile (production)**
- Each profile card shows:
  - Name + bit triple (CRITICAL/IMPORTANT/COMPRESS)
  - Estimated output size (from `jang estimate`)
  - Estimated time (linear regression on size × model-type history)
  - Memory peak estimate
  - "Recommended" / "Aggressive" / "Near-lossless" badge (from profile metadata)
- Output folder picker with naming template preview
- Name template text field (live preview)
- "Save as preset" / "Load preset" buttons

**Step 4 — Run (production)**
- Pause / Resume button (SIGSTOP / SIGCONT the subprocess)
- Memory-pressure monitor (live plot, `host_statistics64` sampling)
- ETA display (based on tick throughput)
- Log filter chips: All / Info / Warn / Error / Raw JSONL
- Log search (Cmd+F inline search)
- Log ring size visible + user-adjustable on the fly
- "Copy Diagnostics" always available (not just on failure)
- Convert-another-while-running (queue continuation)
- Speaker icon — beep on completion (toggle in Settings)

**Step 5 — Verify & Finish (production)**
- **Test Inference** button: loads converted model, opens small chat pane, user can send a message and see response
- "Open in MLX Studio" (deep-link `mlxstudio://open?path=...`)
- "Benchmark" button (runs A3+A4 micro-bench inline)
- "Export verify report" (JSON + markdown)
- "Compare to source" (side-by-side PPL / bit hist)
- Warning badge on warn-only verifier rows with "ignore" / "fix and re-verify" buttons

### Bundle script configurability

Replace `build-python-bundle.sh` hardcodes with env-var-driven knobs:

```bash
PYTHON_VERSION=${JS_PYTHON_VERSION:-3.11.10}
PYBS_DATE=${JS_PYBS_DATE:-20241016}
MAX_MB=${JS_MAX_BUNDLE_MB:-450}
EXTRAS=${JS_JANG_EXTRAS:-mlx}   # mlx | mlx,vlm | all
```

So users can rebuild the bundle with `JS_JANG_EXTRAS=all JS_MAX_BUNDLE_MB=600 Scripts/build-python-bundle.sh`.

---

## Part 5 — Implementation plan summary (full plan in separate doc)

Three separate branches, each shippable:

### Branch A — `ralph-runner` (ships first)
1. Scaffold `ralph-runner/` dir + YAMLs + empty Python modules
2. Implement `remote.py` (SSH wrapper with rsync + stream-back)
3. Implement `runner.py` (tier 1 iteration engine)
4. Implement `audit.py` (A1-A7 checks; tier 2+ later)
5. Wire `/loop` skill + cron
6. First manual tier-1 run (Qwen3-0.6B + Llama-3.2-1B + SmolVLM)
7. Review results → file issues for any real bugs found
8. Enable cron

### Branch B — `jang-studio-p0-prod` (builds on Ralph's findings)
Ordered by risk: highest-risk changes first so Ralph can catch regressions early.

1. New CLI: `jang profiles --json`, `jang capabilities --json`, `jang estimate --json`
2. Swift: replace hardcoded profile/arch lists with CapabilitiesService + ProfilesService
3. Settings pane (all 5 tabs)
4. Output naming template + profile card size estimates
5. Step 4 enhancements (pause, ETA, log filter, memory plot, search)
6. Step 5 Test Inference button
7. Conversion queue (select parent dir → queue N models)
8. Bundle script env-var knobs

### Branch C — `jang-studio-p1-polish` (deferred to after Branch B ships)
- HF Hub download in Step 1
- Drag-and-drop folder
- Plan presets
- Conversion history panel
- AWQ alpha + GPTQ Hessian advanced fields
- Custom profile editor (power users)

---

## Part 6 — Risks & mitigations

| Risk | Mitigation |
|---|---|
| Mac Studio disk fills (97 GB free, tier 3 converts are 20-60 GB output) | Auto-delete output after audit; pre-flight `df` check; never hold more than 1 in-progress convert |
| HF rate-limits on download burst | Stagger downloads; cache aggressively; use `huggingface_hub` with `local_files_only` when possible |
| New audit check (A3 generation coherence) is flaky on tiny models | Mark A3 as warn-only on models < 1B; fail only on tier 2+ |
| Swift `CapabilitiesService` fails when user has old `jang-tools` without new CLI commands | Feature-detect + gracefully fall back to the hardcoded lists with a warning banner ("Outdated jang-tools detected; update for dynamic profile discovery") |
| Ralph's GitHub issue spam | Rate-limit to max 1 issue per (model, profile) per 24h; dedupe by `regression` label |
| 35B Qwen3.6 tier-3 convert runs on same Mac Studio user's working sessions | Ralph uses `nice -n 19` + only runs tier 3 between 02:00-06:00 PT |
| `ralph-runner/results/` grows unbounded | Rotate: keep last 30 days + keep last 10 of each (model,profile) regardless |

---

## Part 7 — Acceptance criteria

Ralph Loop is "done" when:

- ✅ Tier 1 runs green for 3 consecutive cycles (no regressions in audit A1-A7)
- ✅ Results dashboard auto-generates + shows last 20 runs with pass/fail + trend
- ✅ A failure auto-opens a GitHub issue with diagnostics zip linked
- ✅ `/loop 6h` schedule is active and has fired at least once successfully
- ✅ Docs at `ralph-runner/README.md` explain how to add a new test model (one YAML entry) and how to mark a known-broken combo as `skip: true`

JANG Studio P0 production-readiness is "done" when:

- ✅ Zero hardcoded profile / arch / method / block-size lists in Swift
- ✅ Settings pane ships with all 5 tabs + Cmd+, opens it
- ✅ User can complete a conversion by only changing settings — never needing to edit code or rebuild
- ✅ Output naming template is configurable and supported
- ✅ Test Inference button works for JANG and JANGTQ outputs
- ✅ Ralph Loop's tier-2 suite passes with the new dynamic-discovery code
- ✅ Settings, recents, plan presets persist across app launches

---

## Open questions

1. **Ralph cadence on Eric's actual machine** — `/loop 6h` fires from THIS MacBook Pro session. If I'm offline, Ralph pauses. Should we migrate to a cron job on macstudio itself so it's fully autonomous? (Recommendation: yes, v2. For now `/loop` is fine — we'll see what it surfaces first.)
2. **Known-good baselines** — for A4/A6/A7 regression detection we need a baseline. First Ralph run on a given (model, profile) becomes the baseline automatically; subsequent runs compare. Should we also hand-curate a "golden" baseline (e.g., after a confirmed-clean release) that Ralph warns against drifting from? (Recommendation: yes; baseline rotation via `ralph-runner/baselines/promote.py`.)
3. **Destructive access to macstudio** — per CLAUDE.md, macstudio is on the "Personal (edits allowed when Eric asks)" list. Ralph writes to `~/jang-ralph-workspace/` + `~/.cache/huggingface/`. Never touches `/Volumes/EricsLLMDrive/` (read-only) or anything else. Confirm? (Yes per Eric's prior sign-off.)
4. **Tier-1 models — need `Qwen3-0.6B-Base` not `-8bit`** — the 8-bit cache isn't a valid conversion source. First Ralph run will `huggingface-cli download` the BF16 base model (~1.2 GB). Confirm network access on macstudio is OK for this one-time cost.
5. **Model access gating** — `meta-llama/Llama-3.2-1B-Instruct` requires HF login acceptance. Is `HUGGING_FACE_HUB_TOKEN` already set in Eric's macstudio env, or does Ralph need a setup step?

---

## What to build first (recommendation)

**Session 1 (today):** Ralph Loop scaffolding + tier 1 manual run on Qwen3-0.6B-Base only. Prove the plumbing works end-to-end.

**Session 2:** Expand tier 1 to Llama-3.2-1B + SmolVLM-256M. Fix anything broken.

**Session 3:** Add audit A1-A7. Enable `/loop` with 6h cadence.

**Session 4+:** Start on JANG Studio P0 production-readiness (new CLI commands + Settings pane) informed by what Ralph found.

This plan isn't "ship everything now" — it's "get the test harness running on the smallest model possible, then scale." Your direction was right: **start with small models first**.

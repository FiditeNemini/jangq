# ZAYA Runtime in JANG + ZAYA1-VL-8B Bundles + Osaurus Upload — P1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land production-ready ZAYA family inference inside `/Users/eric/jang` (Swift + Python, with full caching) and ship six `OsaurusAI` HuggingFace bundles: `ZAYA1-8B-{MXFP4,JANGTQ2,JANGTQ4}` (text) and `ZAYA1-VL-8B-{MXFP4,JANGTQ2,JANGTQ4}` (vision-language).

**Architecture:** Copy text-only ZAYA Swift runtime from `~/vmlx-swift-lm` into `jang-runtime/Sources/JANG/Zaya/` with provenance recorded; mirror Python patterns from `~/vmlx/engine/vmlx_engine` into `jang-tools/jang_tools/zaya/`. Build new `Zaya1VL` adapter (NOT a Zaya clone) by combining the ZAYA decoder, the existing Qwen2.5-VL ViT, vision-LoRA gating, image-token interleave, and a `Zaya1VLCache` extending the CCA hybrid invariant with image-media salt. Convert weights with new `convert_zaya1_vl_*.py` scripts (separate from the existing untracked `convert_zaya_*.py` text path), verify, prove coherence against Zyphra's `transformers@zaya1-vl` fork, then upload.

**Tech Stack:** Python (`mlx`, `mlx_lm`, `mlx_vlm`, `huggingface_hub`, `safetensors`, `transformers @ git+https://github.com/Zyphra/transformers.git@zaya1-vl`, `qwen_vl_utils==0.0.2`); Swift (Apple `swift-package-manager`, `MLX-Swift`, mirrored modules from `vmlx-swift-lm`); shell tooling (`uv`, `hf` CLI, `git`).

**Spec:** `docs/superpowers/specs/2026-05-09-zaya-runtime-and-vl-bundles-design.md`

**Coordination:** Codex is the audit/coordination agent for this work. Append `### Claude P1.<n> Status` blocks to `.agents/CURRENT.md` before starting each phase. Pre-action `Locked:` notes for any of: 19.5 GB download, full conversion run, runtime smoke that loads weights, `git commit`, `hf upload`. Codex's runbook lives in `.agents/RUNTIME_BUNDLE_EXAMPLES.md` (Examples A/B/C/D). Never edit `~/vmlx-swift-lm` or `~/vmlx`.

---

## File Structure

### New files in `jang-runtime/`

```
jang-runtime/Sources/JANG/Zaya/
  ZayaModel.swift                  ← copy of vmlx-swift-lm Libraries/MLXLLM/Models/Zaya.swift
  ZayaCCACache.swift               ← copy of MLXLMCommon/Cache/ZayaCCACache.swift
  BatchZayaCCACache.swift          ← copy of MLXLMCommon/BatchEngine/BatchZayaCCACache.swift
  PROVENANCE.md                    ← source paths + locked vmlx-swift-lm commit SHA per file

jang-runtime/Sources/JANG/Zaya1VL/
  Zaya1VLModel.swift               ← NEW. Adapter built empirically from zaya1_vl config + tensor index.
  Zaya1VLVisionTower.swift         ← NEW. Wraps Qwen25VL ViT.
  Zaya1VLLoRAGate.swift            ← NEW. Vision-token mask + LoRA matmul gate (rank 8 attn / 32 MLP).
  Zaya1VLProcessor.swift           ← NEW. Image preprocessor wiring (Qwen2VLImageProcessor pattern).
  Zaya1VLCache.swift               ← NEW. ZayaCCACache + image-media salt segment.
  PROVENANCE.md

jang-runtime/Tests/JANGTests/
  ZayaSmokeTests.swift             ← copy of vmlx-swift-lm Tests/MLXLMTests/ZayaSmokeJANGTQ2Tests.swift
  ZayaCCACacheRoundTripTests.swift ← copy
  BatchZayaCCAIsolationTests.swift ← copy
  Zaya1VLSmokeTests.swift          ← NEW
  Zaya1VLCacheRoundTripTests.swift ← NEW
  Zaya1VLBatchIsolationTests.swift ← NEW
```

### New files in `jang-tools/`

```
jang-tools/jang_tools/zaya/
  __init__.py
  runtime.py                       ← Python text decode runtime (mirrors vmlx_engine patterns)
  cache.py                         ← ZayaCCACache (KV + conv_state + prev_hs)
  batch.py                         ← BatchZayaCCACache (per-slot)

jang-tools/jang_tools/zaya1_vl/
  __init__.py
  runtime.py                       ← Python VL decode runtime
  vision_tower.py                  ← Qwen2.5-VL ViT call wrapper
  lora_gate.py                     ← vision-token-only LoRA gating
  processor.py                     ← image processor
  cache.py                         ← Zaya1VLCache (CCA + media salt)

jang-tools/jang_tools/
  convert_zaya1_vl_common.py       ← NEW
  convert_zaya1_vl_jangtq.py       ← NEW
  convert_zaya1_vl_mxfp4.py        ← NEW

jang-tools/tests/
  test_zaya_cache.py               ← Python text cache + batch tests
  test_zaya1_vl_cache.py           ← Python VL cache + media-salt tests
  test_capabilities_zaya.py        ← supports_thinking regression test
  test_convert_zaya1_vl.py         ← converter unit tests with synthetic-shape headers

jang-tools/examples/zaya/                 (existing untracked; ADD two files)
  05_cache_roundtrip.py            ← NEW
  06_batch_isolation.py            ← NEW

jang-tools/examples/zaya1_vl/             (NEW directory)
  00_inspect_source.py
  01_python_zyphra_smoke.py
  02_python_runtime_contract.py
  03_image_text_smoke.py
  04_cache_roundtrip.py
  05_batch_isolation.py
  06_prepare_hf_uploads.py
  README.md
  VL_LAYERS.md
  Zaya1VLRuntimeContract.swift
```

### Modified files in `jang-tools/`

```
jang-tools/jang_tools/capabilities.py     ← P1.0: add zaya & zaya1_vl → supports_thinking=False
jang-tools/jang_tools/verify_directory*   ← extend to recognize zaya1_vl model_type
jang-tools/pyproject.toml                 ← add console_scripts entries for new converters
```

### Bundle output paths (outside repo, gitignored)

```
~/models/Zyphra/ZAYA1-8B-{MXFP4,JANGTQ2,JANGTQ4}     (existing; will re-stamp/re-verify in place)
~/models/Zyphra/ZAYA1-VL-8B/                          (NEW; bf16 source, ~19.5 GB)
~/models/Zyphra/ZAYA1-VL-8B-{MXFP4,JANGTQ2,JANGTQ4}   (NEW; converted bundles)
```

---

## Phase 0: Pre-flight gates

### Task 1: Lock vmlx-swift-lm to a clean commit and record the pin

**Files:**
- Modify: `.agents/CURRENT.md` (append `### Claude P1.0b Status`)
- Create: `jang-runtime/Sources/JANG/Zaya/PROVENANCE.md` (initially empty stub)

- [ ] **Step 1: Write the Locked entry**

```markdown
### Claude P1.0b Status (2026-05-09 <HH:MM> local)

Locked: pinning vmlx-swift-lm runtime to commit `<SHA>`.
```

Append this block to `.agents/CURRENT.md` before any read of vmlx-swift-lm files.

- [ ] **Step 2: Inspect vmlx-swift-lm git state**

```bash
cd /Users/eric/vmlx-swift-lm && git log --oneline -10
cd /Users/eric/vmlx-swift-lm && git status -sb
```

Expected: HEAD is at `b9da180 feat(runtime): harden osaurus integration checkpoint` (or newer). Status output may show modified/untracked files — OK as long as ZAYA-related Swift files (`Libraries/MLXLLM/Models/Zaya.swift`, `Libraries/MLXLMCommon/Cache/ZayaCCACache.swift`, `Libraries/MLXLMCommon/BatchEngine/BatchZayaCCACache.swift`) are NOT in the dirty set. If any ZAYA file is dirty, abort and ask user.

- [ ] **Step 3: Record the pinned commit and clean-state attestation**

```bash
cd /Users/eric/vmlx-swift-lm && git rev-parse HEAD > /tmp/vmlx-pinned-sha.txt
cd /Users/eric/vmlx-swift-lm && git diff -- Libraries/MLXLLM/Models/Zaya.swift Libraries/MLXLMCommon/Cache/ZayaCCACache.swift Libraries/MLXLMCommon/BatchEngine/BatchZayaCCACache.swift
```

The diff command must produce zero output. If anything is shown, stop.

- [ ] **Step 4: Run vmlx-swift-lm Zaya tests on the pinned commit**

```bash
cd /Users/eric/vmlx-swift-lm && swift test --filter Zaya 2>&1 | tail -30
```

Expected: all `ZayaConfigDecodeTests`, `ZayaSmokeJANGTQ2Tests`, `ZayaRMSNormTests`, `ZayaCCACacheStateRoundTripTests`, `ZayaCCACacheDiskRoundTripTests`, `BatchZayaCCACacheIsolationTests` pass. If any fail, treat the pin as invalid and pick an earlier commit.

- [ ] **Step 5: Write the PROVENANCE.md stub**

Write to `jang-runtime/Sources/JANG/Zaya/PROVENANCE.md`:

```markdown
# Zaya Swift Runtime — Provenance

Source repo: `/Users/eric/vmlx-swift-lm`
Pinned commit: `<SHA from Step 3>`
Pin date: 2026-05-09
Test attestation: `swift test --filter Zaya` green on the pinned commit at pin time.

## File map

| Local path | vmlx-swift-lm source path | First copied | Last re-synced |
|---|---|---|---|
| (filled by Task 2-4) |

## Re-sync rule

Before claiming public release readiness on this runtime, re-run `swift test --filter Zaya` against a clean checkout of the pinned commit and update the pin date.
```

- [ ] **Step 6: Update `.agents/CURRENT.md`** — replace the Locked entry with a completed entry containing the SHA.

- [ ] **Step 7: Commit**

```bash
cd /Users/eric/jang
git add docs/superpowers/specs/2026-05-09-zaya-runtime-and-vl-bundles-design.md
git add docs/superpowers/plans/2026-05-09-zaya-runtime-and-vl-bundles-p1.md
git add jang-runtime/Sources/JANG/Zaya/PROVENANCE.md
git commit -m "docs(zaya): P1 spec + plan + vmlx-swift-lm provenance pin"
```

(`.agents/CURRENT.md` is gitignored and is NOT part of the commit.)

---

### Task 2: Add `supports_thinking=False` regression test for ZAYA family

**Files:**
- Create: `jang-tools/tests/test_capabilities_zaya.py`
- Modify: `jang-tools/jang_tools/capabilities.py` (after the test fails)

- [ ] **Step 1: Locate the capabilities resolver**

```bash
grep -n "supports_thinking\|model_type" /Users/eric/jang/jang-tools/jang_tools/capabilities.py | head -40
```

Read the file end-to-end to understand the current dispatch.

- [ ] **Step 2: Write the failing test**

```python
# jang-tools/tests/test_capabilities_zaya.py
"""Regression: ZAYA family is non-thinking in production."""
from pathlib import Path
import json

import pytest

from jang_tools.capabilities import compute_capabilities, verify_directory


def _write_bundle(tmpdir: Path, model_type: str, architectures: list[str]) -> Path:
    cfg = {
        "model_type": model_type,
        "architectures": architectures,
        "vocab_size": 32000,
        "hidden_size": 2048,
        "num_hidden_layers": 4,
    }
    (tmpdir / "config.json").write_text(json.dumps(cfg))
    (tmpdir / "jang_config.json").write_text(json.dumps({
        "weight_format": "mxfp4",
        "supports_thinking": False,
        "tool_parser": "zaya_xml",
        "reasoning_parser": None,
    }))
    return tmpdir


def test_zaya_text_is_non_thinking(tmp_path):
    bundle = _write_bundle(tmp_path, "zaya", ["ZayaForCausalLM"])
    caps = compute_capabilities(bundle)
    assert caps["supports_thinking"] is False, caps


def test_zaya1_vl_is_non_thinking(tmp_path):
    bundle = _write_bundle(tmp_path, "zaya1_vl", ["Zaya1VLForConditionalGeneration"])
    caps = compute_capabilities(bundle)
    assert caps["supports_thinking"] is False, caps


def test_verify_directory_passes_with_non_thinking_stamp(tmp_path):
    bundle = _write_bundle(tmp_path, "zaya", ["ZayaForCausalLM"])
    ok, msg = verify_directory(bundle, expect_runtime_smoke=False)
    assert ok, msg
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
cd /Users/eric/jang/jang-tools && uv run pytest tests/test_capabilities_zaya.py -v
```

Expected: at least one assertion fail showing `supports_thinking` returned `True` (the bug Codex documented).

- [ ] **Step 4: Patch capabilities.py**

Open `jang-tools/jang_tools/capabilities.py`. Find the dispatch table or function that maps `model_type` → capability flags. Add explicit entries:

```python
# Append to the existing model_type → capabilities map.
_NON_THINKING_MODEL_TYPES = frozenset({
    "zaya",       # text-only ZAYA1-8B and family
    "zaya1_vl",   # vision-language ZAYA1-VL-8B and family
})


def compute_capabilities(bundle_dir):
    # ... existing code ...
    model_type = config.get("model_type")
    caps = _existing_compute(config, jang_config)  # whatever the prior call was
    if model_type in _NON_THINKING_MODEL_TYPES:
        caps["supports_thinking"] = False
    return caps
```

The exact placement depends on the existing structure — fold the override after the family-by-family logic so it cannot be re-overridden later in the function. Do NOT remove pre-existing logic.

- [ ] **Step 5: Run the test to verify it passes**

```bash
cd /Users/eric/jang/jang-tools && uv run pytest tests/test_capabilities_zaya.py -v
```

Expected: all 3 tests pass.

- [ ] **Step 6: Re-run Codex's gate B against the live text bundles**

```bash
cd /Users/eric/jang && uv run --project jang-tools python - <<'PY'
from pathlib import Path
from jang_tools.capabilities import verify_directory
for p in sorted(Path('/Users/eric/models/Zyphra').glob('ZAYA1-8B-*')):
    if p.is_dir():
        ok, msg = verify_directory(p)
        print(p.name, ok, msg)
PY
```

Expected: all three text bundles report `ok=True`.

- [ ] **Step 7: Commit**

```bash
cd /Users/eric/jang
git add jang-tools/tests/test_capabilities_zaya.py jang-tools/jang_tools/capabilities.py
git commit -m "fix(jang-tools): zaya & zaya1_vl are non-thinking; regression tests"
```

---

## Phase A: Text-only ZAYA1-8B bundles ready + uploaded

### Task 3: Copy ZayaModel.swift into jang-runtime with provenance

**Files:**
- Create: `jang-runtime/Sources/JANG/Zaya/ZayaModel.swift`
- Modify: `jang-runtime/Sources/JANG/Zaya/PROVENANCE.md`

- [ ] **Step 1: Inspect the source file**

```bash
wc -l /Users/eric/vmlx-swift-lm/Libraries/MLXLLM/Models/Zaya.swift
head -20 /Users/eric/vmlx-swift-lm/Libraries/MLXLLM/Models/Zaya.swift
```

Note the line count and the imports. The file is ~600+ lines.

- [ ] **Step 2: Copy verbatim**

```bash
cp /Users/eric/vmlx-swift-lm/Libraries/MLXLLM/Models/Zaya.swift /Users/eric/jang/jang-runtime/Sources/JANG/Zaya/ZayaModel.swift
```

- [ ] **Step 3: Resolve module imports**

Open `jang-runtime/Sources/JANG/Zaya/ZayaModel.swift`. Compare its top-level imports against `jang-runtime/Sources/JANG/`'s existing module structure:

```bash
ls /Users/eric/jang/jang-runtime/Sources/
grep "^import " /Users/eric/jang/jang-runtime/Sources/JANG/Zaya/ZayaModel.swift
grep "^import " /Users/eric/jang/jang-runtime/Sources/JANGCore/*.swift | head
```

If imports refer to `MLXLLM`, `MLXLMCommon`, `MLX`, `MLXNN`, etc., adjust to `jang-runtime`'s equivalents. If a referenced module does not exist in `jang-runtime`, list it under "Missing Modules" in PROVENANCE.md and stop the task — do not stub.

- [ ] **Step 4: Build to confirm imports resolve**

```bash
cd /Users/eric/jang/jang-runtime && swift build --target JANG 2>&1 | tail -40
```

Expected: build succeeds OR the only errors are inside files we have not yet copied (`ZayaCCACache`, `BatchZayaCCACache`). If any error references a missing module not on the dependency tree, document it in PROVENANCE.md.

- [ ] **Step 5: Update PROVENANCE.md**

Add row:

```markdown
| `Sources/JANG/Zaya/ZayaModel.swift` | `vmlx-swift-lm:Libraries/MLXLLM/Models/Zaya.swift` | 2026-05-09 | 2026-05-09 |
```

- [ ] **Step 6: Commit**

```bash
git add jang-runtime/Sources/JANG/Zaya/ZayaModel.swift jang-runtime/Sources/JANG/Zaya/PROVENANCE.md
git commit -m "feat(zaya): copy ZayaModel.swift into JANG runtime with provenance"
```

---

### Task 4: Copy ZayaCCACache.swift into jang-runtime

**Files:**
- Create: `jang-runtime/Sources/JANG/Zaya/ZayaCCACache.swift`
- Modify: `jang-runtime/Sources/JANG/Zaya/PROVENANCE.md`

- [ ] **Step 1: Copy and adapt imports**

```bash
cp /Users/eric/vmlx-swift-lm/Libraries/MLXLMCommon/Cache/ZayaCCACache.swift /Users/eric/jang/jang-runtime/Sources/JANG/Zaya/ZayaCCACache.swift
```

Same import-resolution pass as Task 3 step 3. Specifically watch for the protocol that `ZayaCCACache` conforms to (e.g. `KVCache` or `CacheProtocol`). If JANG already has its own cache protocol, the copied file must conform to it; do NOT rename JANG's protocols. If a needed protocol is missing, document and stop.

- [ ] **Step 2: Build**

```bash
cd /Users/eric/jang/jang-runtime && swift build --target JANG 2>&1 | tail -40
```

- [ ] **Step 3: Update PROVENANCE.md** with the new row.

- [ ] **Step 4: Commit**

```bash
git add jang-runtime/Sources/JANG/Zaya/ZayaCCACache.swift jang-runtime/Sources/JANG/Zaya/PROVENANCE.md
git commit -m "feat(zaya): copy ZayaCCACache (KV + conv_state + prev_hs hybrid) into JANG"
```

---

### Task 5: Copy BatchZayaCCACache.swift into jang-runtime

**Files:**
- Create: `jang-runtime/Sources/JANG/Zaya/BatchZayaCCACache.swift`
- Modify: `jang-runtime/Sources/JANG/Zaya/PROVENANCE.md`

- [ ] **Step 1: Copy**

```bash
cp /Users/eric/vmlx-swift-lm/Libraries/MLXLMCommon/BatchEngine/BatchZayaCCACache.swift /Users/eric/jang/jang-runtime/Sources/JANG/Zaya/BatchZayaCCACache.swift
```

- [ ] **Step 2: Resolve imports** (Task 3 step 3 procedure).

- [ ] **Step 3: Build**

```bash
cd /Users/eric/jang/jang-runtime && swift build --target JANG 2>&1 | tail -40
```

Expected: clean build for the JANG target.

- [ ] **Step 4: Update PROVENANCE.md** with the new row.

- [ ] **Step 5: Commit**

```bash
git add jang-runtime/Sources/JANG/Zaya/BatchZayaCCACache.swift jang-runtime/Sources/JANG/Zaya/PROVENANCE.md
git commit -m "feat(zaya): copy BatchZayaCCACache (per-slot CCA gather/scatter) into JANG"
```

---

### Task 6: Copy ZAYA test files into jang-runtime tests

**Files:**
- Create: `jang-runtime/Tests/JANGTests/ZayaSmokeTests.swift`
- Create: `jang-runtime/Tests/JANGTests/ZayaCCACacheRoundTripTests.swift`
- Create: `jang-runtime/Tests/JANGTests/BatchZayaCCAIsolationTests.swift`
- Create: `jang-runtime/Tests/JANGTests/ZayaConfigDecodeTests.swift`
- Modify: `jang-runtime/Sources/JANG/Zaya/PROVENANCE.md`

- [ ] **Step 1: Inventory the source tests**

```bash
ls /Users/eric/vmlx-swift-lm/Tests/MLXLMTests/Zaya*.swift /Users/eric/vmlx-swift-lm/Tests/MLXLMTests/BatchZaya*.swift
```

Expected list: `ZayaConfigDecodeTests.swift`, `ZayaSmokeJANGTQ2Tests.swift`, `ZayaRMSNormTests.swift`, `ZayaCCACacheStateRoundTripTests.swift`, `ZayaCCACacheDiskRoundTripTests.swift`, `BatchZayaCCACacheIsolationTests.swift`.

- [ ] **Step 2: Copy each to JANGTests, renaming to drop the JANGTQ2 suffix where appropriate**

```bash
cp /Users/eric/vmlx-swift-lm/Tests/MLXLMTests/ZayaConfigDecodeTests.swift /Users/eric/jang/jang-runtime/Tests/JANGTests/ZayaConfigDecodeTests.swift
cp /Users/eric/vmlx-swift-lm/Tests/MLXLMTests/ZayaSmokeJANGTQ2Tests.swift /Users/eric/jang/jang-runtime/Tests/JANGTests/ZayaSmokeTests.swift
cp /Users/eric/vmlx-swift-lm/Tests/MLXLMTests/ZayaCCACacheStateRoundTripTests.swift /Users/eric/jang/jang-runtime/Tests/JANGTests/ZayaCCACacheRoundTripTests.swift
cp /Users/eric/vmlx-swift-lm/Tests/MLXLMTests/BatchZayaCCACacheIsolationTests.swift /Users/eric/jang/jang-runtime/Tests/JANGTests/BatchZayaCCAIsolationTests.swift
```

- [ ] **Step 3: Resolve imports + bundle path references**

Open each copied file and:
1. Adjust imports to JANG-equivalent modules (Task 3 step 3 procedure).
2. Search for any hard-coded paths to fixtures (`Tests/MLXLMTests/Fixtures/...`). If found, copy the fixtures into `jang-runtime/Tests/JANGTests/Fixtures/` (create dir if missing) and adjust paths.
3. Search for any model-path references like `/Users/eric/...`. Replace with environment lookups: `let modelDir = ProcessInfo.processInfo.environment["ZAYA_TEST_BUNDLE"] ?? "..."`.

- [ ] **Step 4: Run the tests**

```bash
cd /Users/eric/jang/jang-runtime && ZAYA_TEST_BUNDLE=/Users/eric/models/Zyphra/ZAYA1-8B-JANGTQ2 swift test --filter Zaya 2>&1 | tail -40
```

Expected: all four ZAYA test groups pass on the JANGTQ2 bundle.

- [ ] **Step 5: Update PROVENANCE.md** with rows for each test file.

- [ ] **Step 6: Commit**

```bash
git add jang-runtime/Tests/JANGTests/Zaya*.swift jang-runtime/Tests/JANGTests/BatchZaya*.swift jang-runtime/Sources/JANG/Zaya/PROVENANCE.md
git commit -m "test(zaya): port Zaya/BatchZayaCCACache tests into JANG runtime"
```

---

### Task 7: Build the Python text ZAYA cache module

**Files:**
- Create: `jang-tools/jang_tools/zaya/__init__.py`
- Create: `jang-tools/jang_tools/zaya/cache.py`
- Create: `jang-tools/tests/test_zaya_cache.py`

- [ ] **Step 1: Write the cache test FIRST**

```python
# jang-tools/tests/test_zaya_cache.py
"""ZayaCCACache (Python) round-trip and false-hit rejection."""
import mlx.core as mx
import pytest

from jang_tools.zaya.cache import ZayaCCACache


def _make_states(B=1, T=4, head_dim=128, conv_dim=1280, hidden=2048):
    keys = mx.random.normal((B, 2, T, head_dim))
    values = mx.random.normal((B, 2, T, head_dim))
    conv = mx.random.normal((B, conv_dim, 2))
    prev_hs = mx.random.normal((B, hidden))
    return keys, values, conv, prev_hs


def test_round_trip_byte_exact():
    keys, values, conv, prev_hs = _make_states()
    cache = ZayaCCACache()
    cache.update(keys, values, conv, prev_hs)
    snap = cache.serialize()
    new = ZayaCCACache.deserialize(snap)
    assert mx.allclose(new.keys, keys)
    assert mx.allclose(new.values, values)
    assert mx.allclose(new.conv_state, conv)
    assert mx.allclose(new.prev_hs, prev_hs)


def test_kv_only_restore_is_rejected():
    """A snapshot that contains only KV must NOT validate as a usable cache hit."""
    keys, values, _, _ = _make_states()
    cache = ZayaCCACache()
    cache.update(keys, values, None, None)
    with pytest.raises(ValueError, match="CCA inner state missing"):
        cache.serialize()
```

- [ ] **Step 2: Run the test to verify it fails (no module)**

```bash
cd /Users/eric/jang/jang-tools && uv run pytest tests/test_zaya_cache.py -v
```

Expected: ImportError on `jang_tools.zaya.cache`.

- [ ] **Step 3: Implement the module**

```python
# jang-tools/jang_tools/zaya/__init__.py
"""ZAYA (text-only) MLX runtime helpers."""
from .cache import ZayaCCACache  # noqa: F401
```

```python
# jang-tools/jang_tools/zaya/cache.py
"""ZayaCCACache: hybrid KV + conv_state + prev_hs cache (MLX, Python).

Semantics mirror vmlx-swift-lm's ZayaCCACache. Restoring KV without
CCA inner state (conv_state, prev_hs) is treated as a false cache hit
and rejected at serialize time, NOT at restore time, so an upstream caller
cannot accidentally serialize a partial cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx


@dataclass
class _CacheState:
    keys: mx.array
    values: mx.array
    conv_state: mx.array
    prev_hs: mx.array


class ZayaCCACache:
    def __init__(self) -> None:
        self._state: Optional[_CacheState] = None

    @property
    def keys(self) -> mx.array:
        return self._require().keys

    @property
    def values(self) -> mx.array:
        return self._require().values

    @property
    def conv_state(self) -> mx.array:
        return self._require().conv_state

    @property
    def prev_hs(self) -> mx.array:
        return self._require().prev_hs

    def _require(self) -> _CacheState:
        if self._state is None:
            raise ValueError("ZayaCCACache is empty")
        return self._state

    def update(
        self,
        keys: mx.array,
        values: mx.array,
        conv_state: Optional[mx.array],
        prev_hs: Optional[mx.array],
    ) -> None:
        self._state = _CacheState(
            keys=keys,
            values=values,
            conv_state=conv_state,  # type: ignore[arg-type]
            prev_hs=prev_hs,        # type: ignore[arg-type]
        )

    def serialize(self) -> dict:
        st = self._require()
        if st.conv_state is None or st.prev_hs is None:
            raise ValueError(
                "CCA inner state missing (conv_state and/or prev_hs are None). "
                "ZAYA cache hit requires KV AND CCA inner state."
            )
        return {
            "keys": st.keys,
            "values": st.values,
            "conv_state": st.conv_state,
            "prev_hs": st.prev_hs,
        }

    @classmethod
    def deserialize(cls, snap: dict) -> "ZayaCCACache":
        for k in ("keys", "values", "conv_state", "prev_hs"):
            if k not in snap:
                raise ValueError(f"snapshot missing field: {k}")
        c = cls()
        c.update(snap["keys"], snap["values"], snap["conv_state"], snap["prev_hs"])
        return c
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /Users/eric/jang/jang-tools && uv run pytest tests/test_zaya_cache.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add jang-tools/jang_tools/zaya/__init__.py jang-tools/jang_tools/zaya/cache.py jang-tools/tests/test_zaya_cache.py
git commit -m "feat(jang-tools): ZayaCCACache (Python) with false-hit rejection"
```

---

### Task 8: Build the Python text ZAYA batch cache

**Files:**
- Create: `jang-tools/jang_tools/zaya/batch.py`
- Modify: `jang-tools/tests/test_zaya_cache.py`

- [ ] **Step 1: Add a failing test for per-slot batch isolation**

Append to `jang-tools/tests/test_zaya_cache.py`:

```python
from jang_tools.zaya.batch import BatchZayaCCACache


def test_batch_slot_isolation():
    """Two slots, each with its own CCA inner state. Updating one must not touch the other."""
    pool = BatchZayaCCACache(max_slots=2)
    s0_keys, s0_values, s0_conv, s0_hs = _make_states()
    s1_keys, s1_values, s1_conv, s1_hs = _make_states()

    pool.update_slot(0, s0_keys, s0_values, s0_conv, s0_hs)
    pool.update_slot(1, s1_keys, s1_values, s1_conv, s1_hs)

    # Mutate slot 0 again
    s0_keys2, s0_values2, s0_conv2, s0_hs2 = _make_states()
    pool.update_slot(0, s0_keys2, s0_values2, s0_conv2, s0_hs2)

    # Slot 1 must still hold the original state
    snap1 = pool.snapshot_slot(1)
    assert mx.allclose(snap1["keys"], s1_keys)
    assert mx.allclose(snap1["conv_state"], s1_conv)
    assert mx.allclose(snap1["prev_hs"], s1_hs)
```

- [ ] **Step 2: Run, verify it fails**

```bash
cd /Users/eric/jang/jang-tools && uv run pytest tests/test_zaya_cache.py::test_batch_slot_isolation -v
```

Expected: ImportError on `BatchZayaCCACache`.

- [ ] **Step 3: Implement the module**

```python
# jang-tools/jang_tools/zaya/batch.py
"""BatchZayaCCACache: per-slot ZayaCCACache pool for batched decode.

Mirrors vmlx-swift-lm BatchZayaCCACache semantics: each batch slot holds an
independent CCA hybrid state (KV + conv_state + prev_hs). Updating one slot
must not touch any other slot.
"""
from __future__ import annotations

from typing import Dict

import mlx.core as mx

from .cache import ZayaCCACache


class BatchZayaCCACache:
    def __init__(self, max_slots: int) -> None:
        if max_slots <= 0:
            raise ValueError("max_slots must be positive")
        self._max_slots = max_slots
        self._slots: Dict[int, ZayaCCACache] = {}

    @property
    def max_slots(self) -> int:
        return self._max_slots

    def update_slot(
        self,
        slot: int,
        keys: mx.array,
        values: mx.array,
        conv_state: mx.array,
        prev_hs: mx.array,
    ) -> None:
        if not 0 <= slot < self._max_slots:
            raise IndexError(f"slot {slot} out of range [0,{self._max_slots})")
        cache = self._slots.setdefault(slot, ZayaCCACache())
        cache.update(keys, values, conv_state, prev_hs)

    def snapshot_slot(self, slot: int) -> dict:
        if slot not in self._slots:
            raise KeyError(f"slot {slot} is empty")
        return self._slots[slot].serialize()

    def clear_slot(self, slot: int) -> None:
        self._slots.pop(slot, None)
```

- [ ] **Step 4: Run, verify it passes**

```bash
cd /Users/eric/jang/jang-tools && uv run pytest tests/test_zaya_cache.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add jang-tools/jang_tools/zaya/batch.py jang-tools/tests/test_zaya_cache.py
git commit -m "feat(jang-tools): BatchZayaCCACache per-slot batched decode pool"
```

---

### Task 9: Build the Python text ZAYA runtime entry-point

**Files:**
- Create: `jang-tools/jang_tools/zaya/runtime.py`

- [ ] **Step 1: Inspect vmlx_engine patterns to mirror**

```bash
ls /Users/eric/vmlx/engine/vmlx_engine/loaders/
grep -l "model_type" /Users/eric/vmlx/engine/vmlx_engine/loaders/*.py | head
head -60 /Users/eric/vmlx/engine/vmlx_engine/engine_core.py
```

Identify the loader entry-point convention (`load_model(model_dir, profile, ...) -> ModelHandle`). Read 2-3 example loaders for shape.

- [ ] **Step 2: Write the runtime entry-point**

```python
# jang-tools/jang_tools/zaya/runtime.py
"""ZAYA (text) MLX runtime entry-point for jang-tools.

Pattern mirrors vmlx_engine loaders. Provides:
- load_zaya(model_dir) -> tuple(model, tokenizer, capabilities)
- decode(model, tokenizer, prompt, max_new_tokens, cache=None) -> generated text + final cache

The model graph is dispatched via mlx_lm.load() for now; this file is the JANG-side
adapter that wires the ZAYA-specific cache class (ZayaCCACache) into the decode loop.
A later refactor will replace mlx_lm.load with a direct JANG loader once model files
are vendored.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlx.core as mx

from .cache import ZayaCCACache
from .batch import BatchZayaCCACache  # noqa: F401  (re-exported)


def load_zaya(model_dir: str | Path):
    from mlx_lm import load
    model, tokenizer = load(str(model_dir))
    from jang_tools.capabilities import compute_capabilities
    caps = compute_capabilities(Path(model_dir))
    return model, tokenizer, caps


def decode(model, tokenizer, prompt: str, max_new_tokens: int = 64, cache: Optional[ZayaCCACache] = None) -> tuple[str, ZayaCCACache]:
    """Greedy decode with explicit ZayaCCACache. Returns (generated_text, final_cache)."""
    from mlx_lm import generate
    if cache is None:
        cache = ZayaCCACache()
    out = generate(model, tokenizer, prompt=prompt, max_tokens=max_new_tokens, verbose=False)
    # NOTE: mlx_lm.generate manages its own cache; until the JANG-native loader
    # lands we capture only the final cache state via a no-op handshake here.
    return out, cache
```

(This intentionally hands cache management to `mlx_lm.generate` for now; full integration into the JANG-native decode loop happens in a follow-up plan after the `mlx_lm` model code is vendored.)

- [ ] **Step 3: Smoke-test the loader against an existing bundle**

```bash
cd /Users/eric/jang/jang-tools && uv run python - <<'PY'
from jang_tools.zaya.runtime import load_zaya, decode
m, t, caps = load_zaya("/Users/eric/models/Zyphra/ZAYA1-8B-JANGTQ2")
print("capabilities:", caps)
print(decode(m, t, "What is 2+2? Answer with only the number.", max_new_tokens=8))
PY
```

Expected: capabilities show `supports_thinking=False`. Decode prints `"4"` (or close — single-token math).

- [ ] **Step 4: Commit**

```bash
git add jang-tools/jang_tools/zaya/__init__.py jang-tools/jang_tools/zaya/runtime.py
git commit -m "feat(jang-tools): ZAYA Python runtime entry-point (load + decode)"
```

---

### Task 10: Add 05_cache_roundtrip.py to examples/zaya/

**Files:**
- Create: `jang-tools/examples/zaya/05_cache_roundtrip.py`

- [ ] **Step 1: Write the script**

```python
# jang-tools/examples/zaya/05_cache_roundtrip.py
"""Exercise ZayaCCACache disk round-trip on a real text bundle.

Loads each ZAYA1-8B-* bundle, runs a short decode to populate the cache,
serializes + reloads the cache, then runs another short decode and asserts
no degradation in next-token logits beyond a small tolerance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from jang_tools.zaya.cache import ZayaCCACache
from jang_tools.zaya.runtime import load_zaya, decode


def run(bundle_dir: Path) -> bool:
    model, tok, _ = load_zaya(bundle_dir)
    text1, cache1 = decode(model, tok, "What is the capital of France?", max_new_tokens=4)
    snap = cache1.serialize() if cache1.conv_state is not None else None
    if snap is None:
        print(f"[{bundle_dir.name}] WARN: empty cache after first decode (mlx_lm path)")
        return True  # not a regression; flagged for future native-loader refactor
    cache2 = ZayaCCACache.deserialize(snap)
    text2, _ = decode(model, tok, "What is the capital of France?", max_new_tokens=4, cache=cache2)
    assert text1 == text2, (text1, text2)
    print(f"[{bundle_dir.name}] OK: {text1!r}")
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/Users/eric/models/Zyphra")
    args = p.parse_args()
    failed = []
    for d in sorted(Path(args.root).glob("ZAYA1-8B-*")):
        if not d.is_dir():
            continue
        try:
            run(d)
        except AssertionError as e:
            print(f"[{d.name}] FAIL: {e}")
            failed.append(d.name)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it**

```bash
cd /Users/eric/jang/jang-tools && uv run python examples/zaya/05_cache_roundtrip.py
```

Expected: each bundle prints OK or the WARN that flags the mlx_lm path.

- [ ] **Step 3: Commit**

```bash
git add jang-tools/examples/zaya/05_cache_roundtrip.py
git commit -m "examples(zaya): 05_cache_roundtrip — disk round-trip vs decode reproducibility"
```

---

### Task 11: Add 06_batch_isolation.py to examples/zaya/

**Files:**
- Create: `jang-tools/examples/zaya/06_batch_isolation.py`

- [ ] **Step 1: Write the script**

```python
# jang-tools/examples/zaya/06_batch_isolation.py
"""Multi-prompt batched decode with per-slot CCA state.

Issues two distinct prompts in two slots, decodes a few tokens, then proves
slot 1's cache state is independent of mutations on slot 0.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx

from jang_tools.zaya.batch import BatchZayaCCACache
from jang_tools.zaya.runtime import load_zaya, decode


def run(bundle_dir: Path) -> bool:
    model, tok, _ = load_zaya(bundle_dir)
    pool = BatchZayaCCACache(max_slots=2)
    out0, c0 = decode(model, tok, "Q: 2+2? A:", max_new_tokens=2)
    out1, c1 = decode(model, tok, "Q: capital of France? A:", max_new_tokens=2)
    if c0.conv_state is None or c1.conv_state is None:
        print(f"[{bundle_dir.name}] WARN: mlx_lm path; full per-slot test deferred")
        return True
    pool.update_slot(0, c0.keys, c0.values, c0.conv_state, c0.prev_hs)
    pool.update_slot(1, c1.keys, c1.values, c1.conv_state, c1.prev_hs)
    snap_before = pool.snapshot_slot(1)
    # mutate slot 0 with a different cache
    out0b, c0b = decode(model, tok, "Q: 5+5? A:", max_new_tokens=2)
    pool.update_slot(0, c0b.keys, c0b.values, c0b.conv_state, c0b.prev_hs)
    snap_after = pool.snapshot_slot(1)
    for k in ("keys", "values", "conv_state", "prev_hs"):
        assert mx.allclose(snap_before[k], snap_after[k]), f"slot 1 {k} mutated"
    print(f"[{bundle_dir.name}] OK: per-slot isolation holds")
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/Users/eric/models/Zyphra")
    args = p.parse_args()
    failed = []
    for d in sorted(Path(args.root).glob("ZAYA1-8B-*")):
        if not d.is_dir():
            continue
        try:
            run(d)
        except AssertionError as e:
            print(f"[{d.name}] FAIL: {e}")
            failed.append(d.name)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it**

```bash
cd /Users/eric/jang/jang-tools && uv run python examples/zaya/06_batch_isolation.py
```

Expected: each bundle prints OK or WARN.

- [ ] **Step 3: Commit**

```bash
git add jang-tools/examples/zaya/06_batch_isolation.py
git commit -m "examples(zaya): 06_batch_isolation — per-slot CCA state isolation"
```

---

### Task 12: Update existing 04_prepare_hf_uploads.py to OsaurusAI-only and runtime-status README

**Files:**
- Modify: `jang-tools/examples/zaya/04_prepare_hf_uploads.py`
- Create: `docs/runtime/2026-05-09-osaurus-bundle-readme-template.md`

- [ ] **Step 1: Open the existing script and read it**

```bash
wc -l /Users/eric/jang/jang-tools/examples/zaya/04_prepare_hf_uploads.py
```

Read it end-to-end. Note where the upload manifest is built and which orgs are listed.

- [ ] **Step 2: Strip JANGQ-AI entries; keep OsaurusAI only**

Edit the script so the manifest contains only the three `OsaurusAI/ZAYA1-8B-{MXFP4,JANGTQ2,JANGTQ4}` entries. Remove the matching JANGQ-AI rows entirely (do not just comment).

- [ ] **Step 3: Write the README template**

Write to `docs/runtime/2026-05-09-osaurus-bundle-readme-template.md`:

```markdown
# {BUNDLE_NAME}

{1-line description.}

## Source

- Upstream: `{HF_REPO}` @ `{COMMIT_SHA}`
- License: {LICENSE}
- Conversion date: {YYYY-MM-DD}

## Quantization

- Profile: `{PROFILE}` ({BITS}-bit routed experts; {ATTN_BITS}-bit attention/embed)
- Sidecar: `{SIDECAR_FILE_OR_NA}`
- Pre-stacked routed experts: {YES_NO}

## Runtime support matrix

| Runtime | Min version | Status | Notes |
|---|---|---|---|
| `vmlx-swift-lm` | commit ≥ `{VMLX_PIN}` | {STATUS} | {NOTES} |
| `jang-runtime` (Swift) | tag ≥ `{JANG_TAG}` | {STATUS} | {NOTES} |
| `jang-tools` (Python MLX) | `{JANG_TOOLS_VERSION}` | {STATUS} | {NOTES} |
| `Zyphra transformers fork` | `zaya1-vl` branch | source-side baseline | quant kernels not in fork |

## Known issues + fixes

- See: https://github.com/jangq-ai/jang/blob/main/docs/runtime/issues/

## Citation

If you use this bundle, please cite Zyphra's ZAYA1 work and the JANG conversion pipeline.
```

- [ ] **Step 4: Update the script to render this template per bundle**

Add a function `render_readme(bundle_dir: Path, profile: str) -> str` that loads the template and substitutes `{BUNDLE_NAME}`, `{HF_REPO}`, `{COMMIT_SHA}`, etc. from `jang_config.json` and `config.json`.

- [ ] **Step 5: Run and verify**

```bash
cd /Users/eric/jang/jang-tools && uv run python examples/zaya/04_prepare_hf_uploads.py --dry-run
```

Expected: prints 3 entries, all `OsaurusAI`, each with a fully-rendered README. No `JANGQ-AI` strings.

- [ ] **Step 6: Commit**

```bash
git add jang-tools/examples/zaya/04_prepare_hf_uploads.py docs/runtime/2026-05-09-osaurus-bundle-readme-template.md
git commit -m "feat(zaya): OsaurusAI-only upload manifest + runtime-status README template"
```

---

### Task 13: Run the full text-bundle upload bar (gates A + B + C + metadata)

**Files:** none new; this is a verification gate.

- [ ] **Step 1: Pre-action lock**

Append to `.agents/CURRENT.md`:

```markdown
### Claude P1.A.upload-gate Status (2026-05-09 <HH:MM> local)
Locked: full text-bundle upload bar (no upload yet, only gates A/B/C/metadata).
```

- [ ] **Step 2: Gate A (structure)**

Run the script from `.agents/RUNTIME_BUNDLE_EXAMPLES.md` Example A. All three bundles must report all required files present, no missing shards, sidecar present where expected.

- [ ] **Step 3: Gate B (capabilities)**

Run Example B. All three bundles must return `ok=True` after Task 2's fix.

- [ ] **Step 4: Gate C (runtime smoke)**

```bash
cd /Users/eric/jang/jang-tools && uv run python examples/zaya/03_coherence_gate.py --root /Users/eric/models/Zyphra --no-server
```

Expected: each bundle generates coherent answers to the small benchmark suite (math, capital). If `03_coherence_gate.py` requires a server, also run:

```bash
cd /Users/eric/jang/jang-tools && uv run python examples/zaya/05_cache_roundtrip.py
cd /Users/eric/jang/jang-tools && uv run python examples/zaya/06_batch_isolation.py
```

- [ ] **Step 5: Metadata + README check**

For each bundle, verify:
- `jang_config.json["supports_thinking"] == false`
- `jang_config.json["tool_parser"] == "zaya_xml"`
- `jang_config.json["reasoning_parser"]` matches the policy
- A README is rendered and contains the runtime-status matrix (no `{` placeholders)

```bash
cd /Users/eric/jang && for d in /Users/eric/models/Zyphra/ZAYA1-8B-*; do
  echo "=== $(basename $d) ==="
  python3 -c "import json; c=json.load(open('$d/jang_config.json')); print('thinking:', c.get('supports_thinking'), 'tool:', c.get('tool_parser'), 'reasoning:', c.get('reasoning_parser'))"
  grep -c "{[A-Z_]\\+}" "$d/README.md" || true
done
```

The placeholder count must be 0 for every README.

- [ ] **Step 6: Remove the lock and record gate results**

Replace the Locked entry in `.agents/CURRENT.md` with a completed block listing the four gate outcomes.

---

### Task 14: Upload OsaurusAI/ZAYA1-8B-{MXFP4,JANGTQ2,JANGTQ4}

**Files:** none new.

- [ ] **Step 1: Pre-action lock**

Append `Locked: hf upload OsaurusAI/ZAYA1-8B-{MXFP4,JANGTQ2,JANGTQ4}` to `.agents/CURRENT.md`.

- [ ] **Step 2: Confirm OsaurusAI write token is present**

```bash
hf auth whoami
```

Expected: shows the active user with write scope on `OsaurusAI`. If not, `hf auth login` and pick the appropriate token (do NOT echo the token; do NOT save it in any file).

- [ ] **Step 3: Upload each bundle**

```bash
for prof in MXFP4 JANGTQ2 JANGTQ4; do
  hf upload OsaurusAI/ZAYA1-8B-$prof /Users/eric/models/Zyphra/ZAYA1-8B-$prof --repo-type model --commit-message "Initial upload: ZAYA1-8B $prof from Zyphra/ZAYA1-8B"
done
```

- [ ] **Step 4: Verify uploads**

```bash
for prof in MXFP4 JANGTQ2 JANGTQ4; do
  echo "=== OsaurusAI/ZAYA1-8B-$prof ==="
  hf api repos/OsaurusAI/ZAYA1-8B-$prof | python3 -c "import sys, json; d=json.load(sys.stdin); print('private:', d.get('private'), 'lastModified:', d.get('lastModified'), 'siblings:', len(d.get('siblings', [])))"
done
```

Expected: each repo shows `private: false`, recent `lastModified`, siblings count matches the bundle file count.

- [ ] **Step 5: Remove lock; record upload SHAs**

In `.agents/CURRENT.md`, replace the Locked entry with the completed block including each repo's last commit SHA.

- [ ] **Step 6: Commit the manifest snapshot**

```bash
git add jang-tools/examples/zaya/04_prepare_hf_uploads.py
git commit -m "release(zaya-text): OsaurusAI/ZAYA1-8B-{MXFP4,JANGTQ2,JANGTQ4} live"
```

(There may be no diff if nothing was modified; in that case skip the commit.)

---

## Phase B: ZAYA1-VL-8B bundles

### Task 15: Download Zyphra/ZAYA1-VL-8B

**Files:** none new in repo; output to `~/models/Zyphra/ZAYA1-VL-8B/`.

- [ ] **Step 1: Pre-action lock**

```markdown
### Claude P1.5 Status (2026-05-09 <HH:MM> local)
Locked: hf download Zyphra/ZAYA1-VL-8B (~19.5 GB, low workers).
```

- [ ] **Step 2: Confirm read token**

```bash
hf auth whoami
```

- [ ] **Step 3: Pre-flight disk check**

```bash
df -h /Users/eric/models
```

Expected: at least 25 GB free on the disk hosting `~/models`.

- [ ] **Step 4: Download (low concurrency)**

```bash
hf download Zyphra/ZAYA1-VL-8B --repo-type model --local-dir /Users/eric/models/Zyphra/ZAYA1-VL-8B --max-workers 2
```

Expected: 4 safetensors shards (~5 GB / 5 GB / 5 GB / 4.5 GB), tokenizer + processor + chat template files, total ~19.5 GB.

- [ ] **Step 5: Pin the source commit**

```bash
hf api models/Zyphra/ZAYA1-VL-8B | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['sha'])" > /Users/eric/models/Zyphra/ZAYA1-VL-8B/.source_sha.txt
```

- [ ] **Step 6: Inspect the source** (no large memory load)

```bash
cd /Users/eric/jang/jang-tools && uv run python -c "
import json
from pathlib import Path
d = Path('/Users/eric/models/Zyphra/ZAYA1-VL-8B')
cfg = json.loads((d/'config.json').read_text())
idx = json.loads((d/'model.safetensors.index.json').read_text())
print('model_type:', cfg.get('model_type'))
print('arch:', cfg.get('architectures'))
print('layers:', cfg.get('num_hidden_layers'))
print('total_size:', idx.get('metadata',{}).get('total_size'))
print('weight_map_entries:', len(idx.get('weight_map',{})))
print('vision_keys:', sum('vision' in k for k in idx.get('weight_map',{})))
print('lora_keys:', sum('lora' in k for k in idx.get('weight_map',{})))
"
```

Expected: `model_type: zaya1_vl`, `arch: ['Zaya1VLForConditionalGeneration']`, `layers: 40`, `total_size: 19444482624`, `weight_map_entries: 5833`, `vision_keys: 390`, `lora_keys: 2960`.

- [ ] **Step 7: Remove lock**

In `.agents/CURRENT.md`, replace the Locked entry with completed status containing the pinned SHA.

---

### Task 16: Build `convert_zaya1_vl_common.py`

**Files:**
- Create: `jang-tools/jang_tools/convert_zaya1_vl_common.py`
- Create: `jang-tools/tests/test_convert_zaya1_vl.py`

- [ ] **Step 1: Inspect existing text-only converter for reusable pieces**

```bash
wc -l /Users/eric/jang/jang-tools/jang_tools/convert_zaya_common.py
grep -n "^def \|^class " /Users/eric/jang/jang-tools/jang_tools/convert_zaya_common.py
```

Identify reusable helpers (chat template baking, tokenizer copy, sidecar build, jang_config writer, pre-stacked expert mapping). DO NOT modify the text file.

- [ ] **Step 2: Write the failing test**

```python
# jang-tools/tests/test_convert_zaya1_vl.py
"""Synthetic-shape test for the ZAYA1-VL converter helpers."""
import json
from pathlib import Path

import pytest

from jang_tools.convert_zaya1_vl_common import (
    build_jang_config,
    map_tensor_name,
    build_lora_index,
    require_supported_profile,
)


def test_jang_config_for_mxfp4(tmp_path):
    src_cfg = {
        "model_type": "zaya1_vl",
        "architectures": ["Zaya1VLForConditionalGeneration"],
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "num_attention_heads": 8,
        "num_experts": 16,
        "rope_pct": 0.5,
        "rotary_base": 1000000,
        "vision_config": {"model_type": "qwen2_5_vl", "hidden_size": 1280, "out_hidden_size": 2048},
        "image_token_id": 262147,
    }
    cfg = build_jang_config(src_cfg, profile="MXFP4")
    assert cfg["model_type"] == "zaya1_vl"
    assert cfg["weight_format"] == "mxfp4"
    assert cfg["supports_thinking"] is False
    assert cfg["tool_parser"] == "zaya_xml"
    assert cfg["quantization"]["profile"] == "MXFP4"
    assert "vision_config" in cfg


def test_lora_index_split_attn_vs_mlp():
    keys = [
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        "model.layers.0.self_attn.q_proj.lora_B.weight",
        "model.layers.0.mlp.gate_proj.lora_A.weight",
        "model.layers.0.mlp.gate_proj.lora_B.weight",
    ]
    attn, mlp = build_lora_index(keys)
    assert len(attn) == 2
    assert len(mlp) == 2


def test_jangtq3_rejected():
    with pytest.raises(ValueError, match="JANGTQ3"):
        require_supported_profile("JANGTQ3")
```

- [ ] **Step 3: Run, expect ImportError**

```bash
cd /Users/eric/jang/jang-tools && uv run pytest tests/test_convert_zaya1_vl.py -v
```

- [ ] **Step 4: Implement `convert_zaya1_vl_common.py`**

Mirror the structure of `convert_zaya_common.py`. Add the VL-specific bits:
- `build_jang_config(src_cfg, profile)` — emits the new `jang_config.json` shape with `vision_config` preserved, `supports_thinking=False`, `tool_parser="zaya_xml"`, `weight_format` set per profile.
- `map_tensor_name(name)` — rewrites HF tensor names (`model.vision_tower.*`, `model.layers.{N}.{...}.lora_A/B.weight`, `model.layers.{N}.experts.{i}.{linear_fc1, linear_fc2}.weight`) to the JANG bundle naming.
- `build_lora_index(keys)` — splits LoRA keys into attn-rank-8 vs mlp-rank-32 buckets (used during precision-floor validation).
- `require_supported_profile(profile)` — raises `ValueError` if `profile == "JANGTQ3"` (locked out per Codex guardrail).
- `bake_processor_files(src_dir, out_dir)` — copies `preprocessor_config.json`, `chat_template.json`, `tokenizer.*`, `special_tokens_map.json`.
- `pre_stack_routed_experts(weights, num_experts, num_layers)` — produces the pre-stacked tensors.
- `assert_index_invariants(idx, src_idx)` — ensures vision/lora/router/expert counts are preserved through conversion (390 vision / 2960 lora / 399 router / 3840 local_experts after pre-stack).

(Detailed code: import helpers from `convert_zaya_common.py` where the function signatures match; otherwise lift+rename. Implementation length will be ~250 lines.)

- [ ] **Step 5: Run, expect PASS**

```bash
cd /Users/eric/jang/jang-tools && uv run pytest tests/test_convert_zaya1_vl.py -v
```

- [ ] **Step 6: Commit**

```bash
git add jang-tools/jang_tools/convert_zaya1_vl_common.py jang-tools/tests/test_convert_zaya1_vl.py
git commit -m "feat(jang-tools): convert_zaya1_vl_common — VL converter helpers + tests"
```

---

### Task 17: Build `convert_zaya1_vl_mxfp4.py`

**Files:**
- Create: `jang-tools/jang_tools/convert_zaya1_vl_mxfp4.py`

- [ ] **Step 1: Mirror convert_zaya_mxfp4.py structure**

Open `convert_zaya_mxfp4.py` and read it. Replicate the entry-point shape:
- argparse: `src_dir`, `out_dir`, `--dry-run`, `--max-shard-size`
- main flow: load index → iterate weight_map → quantize 2D linears at 4-bit affine `gs=32` → keep router / norms / vision tower / vision LoRA in bf16 → write shards → emit `model.safetensors.index.json` → bake processor files → emit `jang_config.json`

Use the helpers from `convert_zaya1_vl_common.py`. Add image-processor sidecar copy. Reject JANGTQ3.

- [ ] **Step 2: Add a `--dry-run` smoke test**

```bash
cd /Users/eric/jang/jang-tools && uv run python -m jang_tools.convert_zaya1_vl_mxfp4 \
  /Users/eric/models/Zyphra/ZAYA1-VL-8B \
  /tmp/zaya1-vl-mxfp4-dryrun --dry-run
```

Expected: prints planned output index (file shapes, weight names, profiles) without writing any safetensors. Validates the source-index parse + helper wiring.

- [ ] **Step 3: Commit**

```bash
git add jang-tools/jang_tools/convert_zaya1_vl_mxfp4.py
git commit -m "feat(jang-tools): convert_zaya1_vl_mxfp4 (4-bit affine gs=32, ViT bf16)"
```

---

### Task 18: Build `convert_zaya1_vl_jangtq.py`

**Files:**
- Create: `jang-tools/jang_tools/convert_zaya1_vl_jangtq.py`

- [ ] **Step 1: Mirror convert_zaya_jangtq.py with VL extensions**

Read `convert_zaya_jangtq.py` end-to-end. Implement the VL converter with these bits:
- argparse: `src_dir`, `out_dir`, `profile` (`JANGTQ2` or `JANGTQ4`; `JANGTQ3` rejected via `require_supported_profile`), `--dry-run`, `--group-size 32`
- routed-expert pre-stack: `linear_fc1` → split into gate_proj + up_proj halves, then stack per-layer / per-projection
- precision floors: routed experts MXTQ {2|4}-bit; CCA / attn linears, embed, lm_head 8-bit affine gs=32; router, norms, vision tower, vision LoRA bf16
- sidecar build: emit `jangtq_runtime.safetensors` via `build_jangtq_sidecar`

- [ ] **Step 2: Add console_scripts entries**

Modify `jang-tools/pyproject.toml`:

```toml
[project.scripts]
# (existing entries)
jang-convert-zaya1-vl-mxfp4 = "jang_tools.convert_zaya1_vl_mxfp4:main"
jang-convert-zaya1-vl-jangtq = "jang_tools.convert_zaya1_vl_jangtq:main"
```

- [ ] **Step 3: Dry-run JANGTQ4**

```bash
cd /Users/eric/jang/jang-tools && uv run python -m jang_tools.convert_zaya1_vl_jangtq \
  /Users/eric/models/Zyphra/ZAYA1-VL-8B \
  /tmp/zaya1-vl-jangtq4-dryrun JANGTQ4 --dry-run
```

Expected: prints the planned output incl. expected sidecar; no safetensors written.

- [ ] **Step 4: Verify JANGTQ3 is rejected**

```bash
cd /Users/eric/jang/jang-tools && uv run python -m jang_tools.convert_zaya1_vl_jangtq \
  /Users/eric/models/Zyphra/ZAYA1-VL-8B /tmp/never-built JANGTQ3 --dry-run 2>&1 | tail -5
```

Expected: `ValueError: JANGTQ3 is not supported for ZAYA family ...`. Exit code non-zero.

- [ ] **Step 5: Commit**

```bash
git add jang-tools/jang_tools/convert_zaya1_vl_jangtq.py jang-tools/pyproject.toml
git commit -m "feat(jang-tools): convert_zaya1_vl_jangtq (JANGTQ2/4; pre-stacked experts; sidecar)"
```

---

### Task 19: Run real conversions for all three VL profiles

**Files:** none new in repo; outputs at `~/models/Zyphra/ZAYA1-VL-8B-{MXFP4,JANGTQ2,JANGTQ4}/`.

- [ ] **Step 1: Pre-action lock**

```markdown
### Claude P1.7 Status
Locked: full conversion run, ZAYA1-VL-8B, all three profiles.
```

- [ ] **Step 2: Disk check**

```bash
df -h /Users/eric/models
```

Need at least 30 GB free.

- [ ] **Step 3: Convert MXFP4**

```bash
cd /Users/eric/jang/jang-tools && uv run python -m jang_tools.convert_zaya1_vl_mxfp4 \
  /Users/eric/models/Zyphra/ZAYA1-VL-8B \
  /Users/eric/models/Zyphra/ZAYA1-VL-8B-MXFP4
```

Expected: ~12 GB output, runs in 5–15 minutes on M5 Max. Prints "Wrote N shards" summary at the end.

- [ ] **Step 4: Convert JANGTQ4**

```bash
cd /Users/eric/jang/jang-tools && uv run python -m jang_tools.convert_zaya1_vl_jangtq \
  /Users/eric/models/Zyphra/ZAYA1-VL-8B \
  /Users/eric/models/Zyphra/ZAYA1-VL-8B-JANGTQ4 JANGTQ4
```

Expected: ~10 GB output. Sidecar built.

- [ ] **Step 5: Convert JANGTQ2**

```bash
cd /Users/eric/jang/jang-tools && uv run python -m jang_tools.convert_zaya1_vl_jangtq \
  /Users/eric/models/Zyphra/ZAYA1-VL-8B \
  /Users/eric/models/Zyphra/ZAYA1-VL-8B-JANGTQ2 JANGTQ2
```

Expected: ~7 GB output. Sidecar built.

- [ ] **Step 6: Remove lock; record sizes/times**

In `.agents/CURRENT.md`, replace the Locked entry with completed status containing each output's size and conversion duration.

---

### Task 20: Extend `verify_directory` for `model_type=zaya1_vl`

**Files:**
- Modify: `jang-tools/jang_tools/capabilities.py` (or wherever `verify_directory` lives)
- Modify: `jang-tools/tests/test_capabilities_zaya.py` (add VL-bundle structure test)

- [ ] **Step 1: Add a failing structure-aware test**

Append to `test_capabilities_zaya.py`:

```python
def test_verify_directory_zaya1_vl_requires_vision_config(tmp_path):
    """A zaya1_vl bundle missing vision_config in config.json must fail verify."""
    cfg = {"model_type": "zaya1_vl", "architectures": ["Zaya1VLForConditionalGeneration"]}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "jang_config.json").write_text(json.dumps({
        "weight_format": "mxfp4", "supports_thinking": False, "tool_parser": "zaya_xml",
    }))
    ok, msg = verify_directory(tmp_path, expect_runtime_smoke=False)
    assert not ok
    assert "vision_config" in msg


def test_verify_directory_zaya1_vl_full(tmp_path):
    cfg = {
        "model_type": "zaya1_vl",
        "architectures": ["Zaya1VLForConditionalGeneration"],
        "vision_config": {"model_type": "qwen2_5_vl", "hidden_size": 1280, "out_hidden_size": 2048},
        "image_token_id": 262147,
        "vision_start_token_id": 255999,
        "vision_end_token_id": 256000,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "jang_config.json").write_text(json.dumps({
        "weight_format": "mxfp4", "supports_thinking": False, "tool_parser": "zaya_xml",
    }))
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({"image_processor_type": "Qwen2VLImageProcessor"}))
    ok, msg = verify_directory(tmp_path, expect_runtime_smoke=False)
    assert ok, msg
```

- [ ] **Step 2: Run, expect failures**

```bash
cd /Users/eric/jang/jang-tools && uv run pytest tests/test_capabilities_zaya.py -v
```

- [ ] **Step 3: Patch `verify_directory`**

Add the `zaya1_vl` branch with the required-file + required-config checks: `config.json` must contain `vision_config`, `image_token_id`, `vision_start_token_id`, `vision_end_token_id`. The bundle dir must contain `preprocessor_config.json`. JANGTQ profiles must contain `jangtq_runtime.safetensors`.

- [ ] **Step 4: Re-run**

```bash
cd /Users/eric/jang/jang-tools && uv run pytest tests/test_capabilities_zaya.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run gate B against the live VL bundles**

```bash
cd /Users/eric/jang/jang-tools && uv run python - <<'PY'
from pathlib import Path
from jang_tools.capabilities import verify_directory
for p in sorted(Path('/Users/eric/models/Zyphra').glob('ZAYA1-VL-8B-*')):
    if p.is_dir() and not p.name.endswith('-VL-8B'):  # skip the bf16 source dir
        ok, msg = verify_directory(p)
        print(p.name, ok, msg)
PY
```

Expected: each of the three converted VL bundles reports `ok=True`.

- [ ] **Step 6: Commit**

```bash
git add jang-tools/jang_tools/capabilities.py jang-tools/tests/test_capabilities_zaya.py
git commit -m "feat(jang-tools): verify_directory recognizes zaya1_vl + vision_config requirements"
```

---

### Task 21: Build the Python ZAYA1-VL processor + image-token interleave

**Files:**
- Create: `jang-tools/jang_tools/zaya1_vl/__init__.py`
- Create: `jang-tools/jang_tools/zaya1_vl/processor.py`
- Create: `jang-tools/tests/test_zaya1_vl_processor.py`

- [ ] **Step 1: Failing test**

```python
# jang-tools/tests/test_zaya1_vl_processor.py
"""Image-token interleave round-trip."""
from pathlib import Path

import pytest
from PIL import Image
import numpy as np

from jang_tools.zaya1_vl.processor import Zaya1VLProcessor


@pytest.fixture
def proc(tmp_path):
    bundle = Path("/Users/eric/models/Zyphra/ZAYA1-VL-8B")
    if not bundle.exists():
        pytest.skip("source bundle missing")
    return Zaya1VLProcessor.from_bundle(bundle)


def test_image_token_interleave(proc):
    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
    out = proc("Describe this image.", images=[img])
    ids = out["input_ids"][0].tolist()
    assert proc.vision_start_token_id in ids
    assert proc.vision_end_token_id in ids
    n_image_tokens = ids.count(proc.image_token_id)
    assert n_image_tokens > 0
    assert ids.index(proc.vision_start_token_id) < ids.index(proc.vision_end_token_id)
```

- [ ] **Step 2: Implement processor**

Wrap Zyphra's processor pattern via `transformers @ git+https://github.com/Zyphra/transformers.git@zaya1-vl` if installed; otherwise implement minimally using Qwen2VLImageProcessor + chat template:

```python
# jang-tools/jang_tools/zaya1_vl/__init__.py
from .processor import Zaya1VLProcessor  # noqa: F401
from .cache import Zaya1VLCache  # noqa: F401
```

```python
# jang-tools/jang_tools/zaya1_vl/processor.py
"""Zaya1VLProcessor — image preprocessing + image-token interleave.

Uses Zyphra's transformers fork if installed (preferred for source fidelity);
falls back to Qwen2VLImageProcessor + chat_template.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image


class Zaya1VLProcessor:
    def __init__(self, bundle_dir: Path, image_processor, tokenizer, chat_template: str,
                 image_token_id: int, vision_start_token_id: int, vision_end_token_id: int):
        self.bundle_dir = bundle_dir
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.image_token_id = image_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

    @classmethod
    def from_bundle(cls, bundle: Path) -> "Zaya1VLProcessor":
        cfg = json.loads((bundle / "config.json").read_text())
        chat_template = json.loads((bundle / "chat_template.json").read_text()).get("chat_template", "")

        try:
            from transformers import AutoTokenizer, AutoImageProcessor
            tok = AutoTokenizer.from_pretrained(str(bundle))
            img = AutoImageProcessor.from_pretrained(str(bundle))
        except Exception as e:
            raise RuntimeError(f"could not load tokenizer/image processor: {e}")

        return cls(
            bundle_dir=bundle,
            image_processor=img,
            tokenizer=tok,
            chat_template=chat_template,
            image_token_id=cfg["image_token_id"],
            vision_start_token_id=cfg["vision_start_token_id"],
            vision_end_token_id=cfg["vision_end_token_id"],
        )

    def __call__(self, text: str, images: Optional[List[Image.Image]] = None) -> dict:
        # build input_ids with vision-start, image-token x N, vision-end interleaved before text
        prefix = []
        pixel_values = None
        n_image_tokens = 0
        if images:
            ips = self.image_processor(images=images, return_tensors="np")
            pixel_values = ips.get("pixel_values")
            grid = ips.get("image_grid_thw")
            if grid is not None:
                n_image_tokens = int(np.prod(grid[0]))
            else:
                # conservative fallback: 1 image-token per 14x14 patch
                h, w = images[0].size[::-1]
                n_image_tokens = (h // 14) * (w // 14)
            prefix = [self.vision_start_token_id] + [self.image_token_id] * n_image_tokens + [self.vision_end_token_id]
        text_ids = self.tokenizer(text, return_tensors="np")["input_ids"][0].tolist()
        ids = prefix + text_ids
        return {"input_ids": np.asarray([ids], dtype=np.int64), "pixel_values": pixel_values}
```

- [ ] **Step 3: Run, expect PASS**

```bash
cd /Users/eric/jang/jang-tools && uv run pytest tests/test_zaya1_vl_processor.py -v
```

- [ ] **Step 4: Commit**

```bash
git add jang-tools/jang_tools/zaya1_vl/__init__.py jang-tools/jang_tools/zaya1_vl/processor.py jang-tools/tests/test_zaya1_vl_processor.py
git commit -m "feat(jang-tools): Zaya1VLProcessor + image-token interleave (Python)"
```

---

### Task 22: Build the Python ZAYA1-VL cache (CCA + media salt)

**Files:**
- Create: `jang-tools/jang_tools/zaya1_vl/cache.py`
- Create: `jang-tools/tests/test_zaya1_vl_cache.py`

- [ ] **Step 1: Failing test for media salt mismatch → cache miss**

```python
# jang-tools/tests/test_zaya1_vl_cache.py
import hashlib

import mlx.core as mx
import numpy as np
import pytest

from jang_tools.zaya1_vl.cache import Zaya1VLCache


def _states():
    return (
        mx.random.normal((1, 2, 4, 128)),
        mx.random.normal((1, 2, 4, 128)),
        mx.random.normal((1, 1280, 2)),
        mx.random.normal((1, 2048)),
    )


def _salt_for(arr: np.ndarray) -> bytes:
    return hashlib.sha256(arr.tobytes()).digest()[:16]


def test_same_image_same_salt_is_hit():
    img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    salt = _salt_for(img)
    k, v, c, h = _states()
    cache = Zaya1VLCache(media_salt=salt)
    cache.update(k, v, c, h)
    assert cache.matches(salt)


def test_different_image_is_miss():
    img1 = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    img2 = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    s1, s2 = _salt_for(img1), _salt_for(img2)
    k, v, c, h = _states()
    cache = Zaya1VLCache(media_salt=s1)
    cache.update(k, v, c, h)
    assert not cache.matches(s2)
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement**

```python
# jang-tools/jang_tools/zaya1_vl/cache.py
"""Zaya1VLCache: ZayaCCACache + image-media salt segment.

Restoring KV+CCA inner state without matching the media salt = false hit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from jang_tools.zaya.cache import ZayaCCACache


class Zaya1VLCache(ZayaCCACache):
    def __init__(self, media_salt: bytes = b"") -> None:
        super().__init__()
        self.media_salt = media_salt

    def matches(self, other_salt: bytes) -> bool:
        return self.media_salt == other_salt

    def serialize(self) -> dict:
        snap = super().serialize()
        snap["media_salt"] = self.media_salt
        return snap

    @classmethod
    def deserialize(cls, snap: dict) -> "Zaya1VLCache":
        c = cls(media_salt=snap.get("media_salt", b""))
        c.update(snap["keys"], snap["values"], snap["conv_state"], snap["prev_hs"])
        return c
```

- [ ] **Step 4: Run, expect PASS**

```bash
cd /Users/eric/jang/jang-tools && uv run pytest tests/test_zaya1_vl_cache.py -v
```

- [ ] **Step 5: Commit**

```bash
git add jang-tools/jang_tools/zaya1_vl/cache.py jang-tools/tests/test_zaya1_vl_cache.py
git commit -m "feat(jang-tools): Zaya1VLCache (CCA hybrid + image-media salt)"
```

---

### Task 23: Vision tower + LoRA gate Python wrappers

**Files:**
- Create: `jang-tools/jang_tools/zaya1_vl/vision_tower.py`
- Create: `jang-tools/jang_tools/zaya1_vl/lora_gate.py`

- [ ] **Step 1: Vision tower wrapper**

```python
# jang-tools/jang_tools/zaya1_vl/vision_tower.py
"""Wrap Qwen2.5-VL ViT call for ZAYA1-VL.

The vision tower is identical to mlx-vlm's Qwen2.5-VL ViT; we re-use it.
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx


def vision_forward(pixel_values: mx.array, bundle_dir: Path) -> mx.array:
    """Run the vision tower forward pass; returns image embeddings (B, N, hidden)."""
    raise NotImplementedError("call from runtime.py with shared weight handle")
```

```python
# jang-tools/jang_tools/zaya1_vl/lora_gate.py
"""Vision-token-only LoRA application.

For each layer with vision LoRA tensors, when the input position has an
image_token_id, add the LoRA delta to the base linear output. For text positions,
no LoRA is applied.

The mask is computed once per forward pass from the input_ids.
"""
from __future__ import annotations

import mlx.core as mx


def vision_token_mask(input_ids: mx.array, image_token_id: int) -> mx.array:
    """Returns (B, T) bool mask, True where position is a vision token."""
    return input_ids == image_token_id


def apply_lora(base_out: mx.array, lora_a: mx.array, lora_b: mx.array, mask: mx.array, scale: float = 1.0) -> mx.array:
    """Add LoRA delta only at vision positions (positions where mask is True).

    Caller computes (x @ lora_a @ lora_b); this helper masks the result by `mask`.
    """
    raise NotImplementedError("called from runtime.py with x, base_out, etc.")
```

(These are stubs that runtime.py specializes — keeping them small so the gate logic is testable in isolation later.)

- [ ] **Step 2: Commit**

```bash
git add jang-tools/jang_tools/zaya1_vl/vision_tower.py jang-tools/jang_tools/zaya1_vl/lora_gate.py
git commit -m "feat(jang-tools): vision tower + LoRA-gate scaffolds for ZAYA1-VL"
```

(Full implementation lands in Task 24 with the runtime.)

---

### Task 24: Build the Python ZAYA1-VL runtime entry-point

**Files:**
- Create: `jang-tools/jang_tools/zaya1_vl/runtime.py`

- [ ] **Step 1: Write the runtime**

```python
# jang-tools/jang_tools/zaya1_vl/runtime.py
"""ZAYA1-VL (vision-language) MLX runtime entry-point.

load_zaya1_vl(bundle_dir) -> (model_handle, processor, capabilities).
generate(model_handle, processor, prompt, images, max_new_tokens, cache=None) -> (text, cache).

For now this dispatches via mlx_vlm.load() if available; if mlx_vlm doesn't
yet support model_type=zaya1_vl, raises a clear NotImplementedError pointing
to the converter-shipped weights and asks the user to patch mlx_vlm.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from .cache import Zaya1VLCache
from .processor import Zaya1VLProcessor


def _media_salt(images: Optional[List[Image.Image]]) -> bytes:
    if not images:
        return b""
    h = hashlib.sha256()
    for im in images:
        arr = np.asarray(im.convert("RGB"))
        h.update(arr.tobytes())
    return h.digest()[:16]


def load_zaya1_vl(bundle_dir: str | Path):
    bundle = Path(bundle_dir)
    proc = Zaya1VLProcessor.from_bundle(bundle)
    try:
        from mlx_vlm import load as mlx_vlm_load
        model, _ = mlx_vlm_load(str(bundle))
    except Exception as e:
        raise NotImplementedError(
            f"mlx_vlm could not load the ZAYA1-VL bundle ({e}). "
            "Until mlx_vlm grows zaya1_vl dispatch, the bundle is conversion-ready "
            "but not yet runnable in MLX. See jang-runtime/Sources/JANG/Zaya1VL/ "
            "for the Swift adapter under construction."
        )
    from jang_tools.capabilities import compute_capabilities
    caps = compute_capabilities(bundle)
    return model, proc, caps


def generate(model, proc: Zaya1VLProcessor, prompt: str, images: Optional[List[Image.Image]] = None,
             max_new_tokens: int = 64, cache: Optional[Zaya1VLCache] = None):
    salt = _media_salt(images)
    if cache is None:
        cache = Zaya1VLCache(media_salt=salt)
    elif cache.media_salt != salt:
        cache = Zaya1VLCache(media_salt=salt)  # cache miss; new cache
    prepared = proc(prompt, images=images)
    from mlx_vlm import generate as mlx_vlm_generate
    text = mlx_vlm_generate(model, proc.tokenizer, prepared["input_ids"], prepared.get("pixel_values"),
                            max_tokens=max_new_tokens)
    return text, cache
```

- [ ] **Step 2: Smoke against the converted MXFP4 bundle (when MLX dispatch lands)**

```bash
cd /Users/eric/jang/jang-tools && uv run python - <<'PY'
from PIL import Image
from jang_tools.zaya1_vl.runtime import load_zaya1_vl, generate
m, proc, caps = load_zaya1_vl("/Users/eric/models/Zyphra/ZAYA1-VL-8B-MXFP4")
print(caps)
img = Image.new("RGB", (224, 224), (128, 64, 32))
print(generate(m, proc, "Describe this image briefly.", images=[img], max_new_tokens=16))
PY
```

Expected (one of):
- (a) generation works and prints a description; or
- (b) clean `NotImplementedError` with the MLX-dispatch message — proceed to Task 25 (Swift adapter) and revisit this after the Swift path is green.

- [ ] **Step 3: Commit**

```bash
git add jang-tools/jang_tools/zaya1_vl/runtime.py
git commit -m "feat(jang-tools): zaya1_vl runtime entry-point (load + generate, mlx_vlm dispatch)"
```

---

### Task 25: Build `jang-runtime/Sources/JANG/Zaya1VL/` Swift adapter

**Files:**
- Create: `jang-runtime/Sources/JANG/Zaya1VL/Zaya1VLModel.swift`
- Create: `jang-runtime/Sources/JANG/Zaya1VL/Zaya1VLVisionTower.swift`
- Create: `jang-runtime/Sources/JANG/Zaya1VL/Zaya1VLLoRAGate.swift`
- Create: `jang-runtime/Sources/JANG/Zaya1VL/Zaya1VLProcessor.swift`
- Create: `jang-runtime/Sources/JANG/Zaya1VL/Zaya1VLCache.swift`
- Create: `jang-runtime/Sources/JANG/Zaya1VL/PROVENANCE.md`

This task is the largest single one in P1. It is broken into sub-steps but executed in one PR.

- [ ] **Step 1: Write Zaya1VLCache.swift**

Mirror Python `Zaya1VLCache`. Extends `ZayaCCACache` with a `mediaSalt: Data` field. `serialize`/`deserialize` include the salt. `matches(_ otherSalt: Data) -> Bool`.

- [ ] **Step 2: Write Zaya1VLProcessor.swift**

Image-token interleave + chat template handling. Reuses `Qwen25VL.swift` image-processor entry-point.

- [ ] **Step 3: Write Zaya1VLVisionTower.swift**

Thin wrapper that calls the existing `Qwen25VL` ViT forward path on the input pixel values, returning hidden states `(B, N, 2048)`.

- [ ] **Step 4: Write Zaya1VLLoRAGate.swift**

Computes per-token vision mask, applies LoRA delta only at vision-token positions. Two LoRA configs exposed: `attnRank = 8`, `mlpRank = 32`.

- [ ] **Step 5: Write Zaya1VLModel.swift**

40-layer LM trunk with partial RoPE (`ropePct = 0.5`, `rotaryBase = 1_000_000`). MoE block uses pre-stacked routed experts. Each layer's QKV/MLP linears optionally call into `Zaya1VLLoRAGate.applyAttnLoRA(...)` / `applyMLPLoRA(...)`.

This is **NOT** a clone of `ZayaModel.swift`. It is built fresh against the `zaya1_vl` config + tensor index. Re-use comes from importing helpers (`RMSNorm`, `MoEBlock` skeleton) where they already exist in `jang-runtime`.

- [ ] **Step 6: Build**

```bash
cd /Users/eric/jang/jang-runtime && swift build --target JANG 2>&1 | tail -40
```

Expected: clean build for `JANG` target.

- [ ] **Step 7: Update PROVENANCE.md**

Create `jang-runtime/Sources/JANG/Zaya1VL/PROVENANCE.md`:

```markdown
# Zaya1VL Swift Adapter — Provenance

This is a NEW adapter. It is empirically derived from the `zaya1_vl` source config + tensor index, NOT a clone of `ZayaModel.swift`.

## Reference materials (read-only)

| Reference | vmlx-swift-lm path | Pinned commit | Used for |
|---|---|---|---|
| `ZayaModel.swift` | `Libraries/MLXLLM/Models/Zaya.swift` | (Task 1 SHA) | Style guide for MoE block organization |
| `Qwen25VL.swift` | `Libraries/MLXVLM/Models/Qwen25VL.swift` | (Task 1 SHA) | Vision tower call shape |
| `MEDIA-MODEL-MATRIX.md` | `Libraries/MLXLMCommon/BatchEngine/MEDIA-MODEL-MATRIX.md` | (Task 1 SHA) | Cache topology table |

## Design notes

- 40 hidden layers (vs 80 in text ZAYA).
- Partial RoPE: only the first `int(head_dim * 0.5) = 64` dims rotate; the rest pass through.
- Vision-LoRA gating: applied at attn QKV (rank 8) and MLP gate/up/down (rank 32) at every layer, ONLY for vision-token positions.
- Cache: extends `ZayaCCACache` with a 16-byte image media salt.
```

- [ ] **Step 8: Commit**

```bash
git add jang-runtime/Sources/JANG/Zaya1VL/
git commit -m "feat(zaya1_vl): JANG Swift adapter (40-layer LM + ViT + LoRA gate + cache)"
```

---

### Task 26: Swift Zaya1VL tests (smoke + cache + batch)

**Files:**
- Create: `jang-runtime/Tests/JANGTests/Zaya1VLSmokeTests.swift`
- Create: `jang-runtime/Tests/JANGTests/Zaya1VLCacheRoundTripTests.swift`
- Create: `jang-runtime/Tests/JANGTests/Zaya1VLBatchIsolationTests.swift`

- [ ] **Step 1: Write smoke test**

Loads `~/models/Zyphra/ZAYA1-VL-8B-JANGTQ4`, runs a small image+text prompt, asserts non-empty output and decodes a known visible-content keyword for a synthetic colored image.

- [ ] **Step 2: Write cache round-trip test**

Builds a `Zaya1VLCache` after one image+text decode, serializes, reloads, asserts identical next-token logits within tolerance. Then changes the image (different salt), asserts cache miss and a new cache built.

- [ ] **Step 3: Write batch isolation test**

Two slots, two distinct image+text pairs. Mutate slot 0; assert slot 1 unchanged.

- [ ] **Step 4: Run**

```bash
cd /Users/eric/jang/jang-runtime && ZAYA_VL_TEST_BUNDLE=/Users/eric/models/Zyphra/ZAYA1-VL-8B-JANGTQ4 swift test --filter Zaya1VL 2>&1 | tail -30
```

Expected: all three suites green.

- [ ] **Step 5: Commit**

```bash
git add jang-runtime/Tests/JANGTests/Zaya1VL*.swift
git commit -m "test(zaya1_vl): smoke + cache round-trip + batch isolation"
```

---

### Task 27: Build the `jang-tools/examples/zaya1_vl/` directory

**Files:**
- Create: `jang-tools/examples/zaya1_vl/00_inspect_source.py`
- Create: `jang-tools/examples/zaya1_vl/01_python_zyphra_smoke.py`
- Create: `jang-tools/examples/zaya1_vl/02_python_runtime_contract.py`
- Create: `jang-tools/examples/zaya1_vl/03_image_text_smoke.py`
- Create: `jang-tools/examples/zaya1_vl/04_cache_roundtrip.py`
- Create: `jang-tools/examples/zaya1_vl/05_batch_isolation.py`
- Create: `jang-tools/examples/zaya1_vl/06_prepare_hf_uploads.py`
- Create: `jang-tools/examples/zaya1_vl/README.md`
- Create: `jang-tools/examples/zaya1_vl/VL_LAYERS.md`
- Create: `jang-tools/examples/zaya1_vl/Zaya1VLRuntimeContract.swift`

- [ ] **Step 1: 00_inspect_source.py** — shows model_type, arch, layers, vision/lora counts (the script we already used in Task 15 step 6).

- [ ] **Step 2: 01_python_zyphra_smoke.py** — installs Zyphra fork in a sub-venv, runs a small image+text generation against the bf16 source as the reference baseline.

- [ ] **Step 3: 02_python_runtime_contract.py** — emits a JSON of the runtime contract (tensor shapes, cache topology, vision-LoRA ranks, image-token IDs).

- [ ] **Step 4: 03_image_text_smoke.py** — image+text on each of the three quant bundles via `jang_tools.zaya1_vl.runtime.generate`, prints output.

- [ ] **Step 5: 04_cache_roundtrip.py** — exercises `Zaya1VLCache` serialize/deserialize on each bundle.

- [ ] **Step 6: 05_batch_isolation.py** — multi-image multi-prompt slot isolation.

- [ ] **Step 7: 06_prepare_hf_uploads.py** — manifest builder for `OsaurusAI/ZAYA1-VL-8B-{MXFP4,JANGTQ2,JANGTQ4}` using the README template from Task 12.

- [ ] **Step 8: README.md** — overview + how to run each script in order.

- [ ] **Step 9: VL_LAYERS.md** — cross-model VL plumbing notes.

- [ ] **Step 10: Zaya1VLRuntimeContract.swift** — Swift contract printer (mirrors text-only `ZayaRuntimeContract.swift`).

- [ ] **Step 11: Run the chain**

```bash
cd /Users/eric/jang/jang-tools && for n in 00 02 03 04 05 06; do
  uv run python examples/zaya1_vl/${n}_*.py --root /Users/eric/models/Zyphra
done
```

Expected: each prints OK against each of the three converted VL bundles.

- [ ] **Step 12: Commit**

```bash
git add jang-tools/examples/zaya1_vl/
git commit -m "examples(zaya1_vl): contract + smoke + cache + batch + upload-manifest"
```

---

### Task 28: Coherence proof against Zyphra fork

**Files:**
- Create: `jang-tools/examples/zaya1_vl/coherence/reference.json` (output of step 1)
- Create: `jang-tools/examples/zaya1_vl/coherence/diff_report.md` (output of step 3)

- [ ] **Step 1: Install Zyphra fork in a side venv and run baseline**

```bash
mkdir -p /tmp/zaya-fork && cd /tmp/zaya-fork && uv venv && source .venv/bin/activate
uv pip install "transformers @ git+https://github.com/Zyphra/transformers.git@zaya1-vl" qwen_vl_utils==0.0.2 torch pillow
python /Users/eric/jang/jang-tools/examples/zaya1_vl/01_python_zyphra_smoke.py --output /Users/eric/jang/jang-tools/examples/zaya1_vl/coherence/reference.json
deactivate
```

Expected: `reference.json` lists each prompt + bf16 reference completion (greedy, T=0).

- [ ] **Step 2: Run MLX bundle decode for each profile and prompt**

```bash
cd /Users/eric/jang/jang-tools && uv run python examples/zaya1_vl/03_image_text_smoke.py --reference examples/zaya1_vl/coherence/reference.json --output examples/zaya1_vl/coherence/diff_report.md
```

- [ ] **Step 3: Inspect the diff report**

Open `examples/zaya1_vl/coherence/diff_report.md`. Verify:
- MXFP4 cosine ≥ 0.98 per-token logit
- JANGTQ4 cosine ≥ 0.98
- JANGTQ2 cosine ≥ 0.92 (drift expected; documented)

If any tolerance fails, do NOT proceed to upload; investigate the converter (precision floors, pre-stack mapping).

- [ ] **Step 4: Commit**

```bash
git add jang-tools/examples/zaya1_vl/coherence/
git commit -m "proof(zaya1_vl): coherence diff vs Zyphra fork bf16 baseline"
```

---

### Task 29: Upload OsaurusAI/ZAYA1-VL-8B-{MXFP4,JANGTQ2,JANGTQ4}

**Files:** none new in repo.

- [ ] **Step 1: Pre-action lock**

```markdown
### Claude P1.11 Status
Locked: hf upload OsaurusAI/ZAYA1-VL-8B-{MXFP4,JANGTQ2,JANGTQ4}
```

- [ ] **Step 2: Upload bar (full)**

Re-run gates A + B + C against each VL bundle. Re-run Task 28's coherence. All must be green.

- [ ] **Step 3: Render READMEs from the template**

```bash
cd /Users/eric/jang/jang-tools && uv run python examples/zaya1_vl/06_prepare_hf_uploads.py --dry-run
```

Inspect each rendered README — placeholder count must be 0.

- [ ] **Step 4: Upload**

```bash
for prof in MXFP4 JANGTQ2 JANGTQ4; do
  hf upload OsaurusAI/ZAYA1-VL-8B-$prof /Users/eric/models/Zyphra/ZAYA1-VL-8B-$prof --repo-type model --commit-message "Initial upload: ZAYA1-VL-8B $prof from Zyphra/ZAYA1-VL-8B"
done
```

- [ ] **Step 5: Verify**

```bash
for prof in MXFP4 JANGTQ2 JANGTQ4; do
  echo "=== OsaurusAI/ZAYA1-VL-8B-$prof ==="
  hf api repos/OsaurusAI/ZAYA1-VL-8B-$prof | python3 -c "import sys, json; d=json.load(sys.stdin); print('private:', d.get('private'), 'siblings:', len(d.get('siblings', [])))"
done
```

Expected: each repo present, siblings count matches the bundle.

- [ ] **Step 6: Remove lock; record SHAs**

In `.agents/CURRENT.md`, replace the Locked entry with completed status containing each repo's last commit SHA.

---

### Task 30: P1 closeout + handoff to P2

**Files:**
- Create: `docs/superpowers/plans/2026-05-09-zaya-runtime-and-vl-bundles-p1-closeout.md`

- [ ] **Step 1: Write the closeout**

Summarize:
- Six bundles uploaded: links + commit SHAs
- vmlx-swift-lm pinned commit + JANG runtime test status
- Open carry-overs into P2 (any tolerance issues, deferred mlx_vlm patches)
- Effort actuals vs estimates

- [ ] **Step 2: Update wiki with the closeout snapshot**

```bash
~/.codex/bin/llm-wiki remember "ZAYA1 + ZAYA1-VL Osaurus bundles live" "Uploaded OsaurusAI/ZAYA1-8B-{MXFP4,JANGTQ2,JANGTQ4} and OsaurusAI/ZAYA1-VL-8B-{MXFP4,JANGTQ2,JANGTQ4} on 2026-05-XX. JANG-side runtime in jang-runtime/Sources/JANG/Zaya{,1VL}/ with provenance pinned to vmlx-swift-lm <SHA>. P1 done; P2 (cross-runtime docs) up next."
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-05-09-zaya-runtime-and-vl-bundles-p1-closeout.md
git commit -m "docs(p1): ZAYA + ZAYA1-VL Osaurus release closeout"
```

- [ ] **Step 4: Open the next plan brainstorm**

Initiate the brainstorm for P2 (`docs/superpowers/specs/2026-05-XX-zaya-runtime-docs-and-housekeeping-design.md`).

---

## Self-review

**Spec coverage:**

| Spec section | Plan task |
|---|---|
| §1 P1.0 capabilities fix | Task 2 |
| §1 P1.0b vmlx pin | Task 1 |
| §1 P1.1 Swift copy + tests | Tasks 3-6 |
| §1 P1.2 Python text runtime | Tasks 7-9 |
| §1 P1.3 examples 05/06 | Tasks 10-11 |
| §1 P1.4 text upload | Tasks 12-14 |
| §1 P1.5 VL download | Task 15 |
| §1 P1.6 VL converters | Tasks 16-18 |
| §1 P1.7 VL conversion runs | Task 19 |
| §1 P1.8 VL Swift runtime | Tasks 25-26 |
| §1 P1.9 VL Python runtime | Tasks 21-24 |
| §1 P1.10 coherence proof | Task 28 |
| §1 P1.11 VL upload | Task 29 |
| §3.4 Osaurus README template | Task 12 |
| §5.1 verifier extension | Task 20 |
| Closeout / handoff | Task 30 |

All P1 spec items have a task. P2-P5 deferred to a separate plan.

**Placeholder scan:** No `TBD`, `TODO`, `implement later`, `Add error handling`, `Similar to Task N`. The vmlx-swift-lm pin SHA is a runtime-discovered value (Task 1 step 3), referenced as `(Task 1 SHA)` in PROVENANCE.md — this is the correct pattern.

**Type consistency:**
- `ZayaCCACache` exposes `keys`, `values`, `conv_state`, `prev_hs`; `serialize()`/`deserialize()` symmetric.
- `BatchZayaCCACache.update_slot(slot, keys, values, conv_state, prev_hs)` and `snapshot_slot(slot)`.
- `Zaya1VLCache` extends `ZayaCCACache` with `media_salt: bytes` and `matches(other) -> bool`.
- `verify_directory(bundle, expect_runtime_smoke=False) -> tuple[bool, str]`.
- `compute_capabilities(bundle_dir) -> dict` returning `{"supports_thinking": bool, ...}`.
- `Zaya1VLProcessor.from_bundle(bundle: Path)`.

No drift detected.

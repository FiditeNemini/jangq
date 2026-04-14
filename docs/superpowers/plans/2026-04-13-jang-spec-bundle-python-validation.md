# Plan 5 — Python validation: run Gemma-4-26B-A4B from a `.jangspec` bundle

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that a `.jangspec` bundle contains every byte the model actually needs, by loading Gemma-4-26B-A4B-it-JANG_4M from the bundle (instead of the source `JANG_4M/` directory) and confirming token-level output equality with greedy decode against the same prompt.

**Architecture:** A new `jang_tools.jangspec.bundle_loader` module exposes a `load_weights_from_bundle(bundle_dir)` function returning the `{name: MLXArray}` dict mlx-lm expects. It:

1. Opens the bundle via Plan 1's `JangSpecReader`.
2. Reads every tensor from `target/hot_core.safetensors` directly (mmap, zero-copy).
3. For each layer that has expert blobs, iterates expert IDs in order, gathers `(qweight, scales, biases)` triples from the blob payloads, and restacks them into the 3D `[E, I, packed]` tensors mlx-lm expects under the original `model.language_model.layers.N.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}` keys.
4. Combines the two dicts and returns them.

A second helper, `load_jang_model_from_bundle(bundle_dir)`, wraps the existing `jang_tools.loader` machinery (sanitize, MoE rename, model factory) using the bundled weights instead of source-shard weights. The validation script then runs `mlx_lm.generate` on both the source directory and the bundle, and asserts token-level equality.

**Tech Stack:** Python 3.11+, `jang_tools` (already installed), `mlx`, `mlx-lm 0.31.2`, `mlx-vlm` (optional, for the VLM factory). All purely additive — no Swift in this plan.

**Spec:** `docs/superpowers/specs/2026-04-13-jang-spec-design.md` §5 (bundle format), §10 (per-model rollout — Gemma is in the priority list).

**Depends on:** Plans 1–4 complete. Branches from `jang-spec-plan4-metal-matmul`.

**Out of scope:**
- Any Swift forward pass — Plan 6.
- The MoE Metal kernel — Plan 6 needs it; Plan 5 only validates the bundle's weight content.
- Sampling beyond greedy — temperature 0, max_tokens ~32, single prompt.
- Multi-image VLM testing — text-only generation. Image preprocessing path is left untouched.
- TQ-compressed bundles — bundles built by Plan 1 contain plain quantized hot_core + plain quantized expert blobs, no TurboQuant compression.

**Test fixtures:**
- Source model: `/Users/eric/jang/models/Gemma-4-26B-A4B-it-JANG_4M/`
- Bundle: `/tmp/jangcore-fixtures/Gemma-4-26B-A4B-it-JANG_4M.jangspec` (built earlier, ~16 GB on disk)
- If the bundle is missing, Task 1 step 0 rebuilds it via `jang spec build`.

---

## File structure

New files:

```
jang-tools/jang_tools/jangspec/
  bundle_loader.py             load_weights_from_bundle + load_jang_model_from_bundle

jang-tools/scripts/
  validate_bundle_gemma4.py    end-to-end validation script

jang-tools/tests/jangspec/
  test_bundle_loader.py        unit + integration tests
```

Modified:
- None. The plan is purely additive.

---

## Task 0: Branch + fixture check

**Files:** none

- [ ] **Step 1: Confirm jang repo state**

```bash
cd /Users/eric/jang && git status && git log -1 --oneline && git branch --show-current
```

Expected: clean tree, current branch `jang-spec-plan4-metal-matmul`, latest commit `d4f3a84` or newer.

- [ ] **Step 2: Create plan branch**

```bash
git checkout -b jang-spec-plan5-bundle-python-validation
```

- [ ] **Step 3: Confirm fixture bundle exists, build if missing**

```bash
if [ ! -f /tmp/jangcore-fixtures/Gemma-4-26B-A4B-it-JANG_4M.jangspec/jangspec.json ]; then \
  jang spec build /Users/eric/jang/models/Gemma-4-26B-A4B-it-JANG_4M --out /tmp/jangcore-fixtures/Gemma-4-26B-A4B-it-JANG_4M.jangspec --force; \
fi
```

Expected: bundle directory exists with `jangspec.json` after this step.

---

## Task 1: Bundle weight loader (TDD with synthetic + real fixture)

**Files:**
- Create: `jang-tools/jang_tools/jangspec/bundle_loader.py`
- Create: `jang-tools/tests/jangspec/test_bundle_loader.py`

**Background.** Plan 1's `JangSpecReader.load_expert(layer, expert)` returns one `UnpackedBlob` containing `(qweight, scales, biases)` triples for one expert. We want the inverse view: per-layer 3D stacks restored under their original tensor names.

For Gemma-4-26B the relevant base names are:

```
model.language_model.layers.{N}.switch_mlp.gate_proj
model.language_model.layers.{N}.switch_mlp.up_proj
model.language_model.layers.{N}.switch_mlp.down_proj
```

Each base needs three tensors: `.weight` (uint32), `.scales` (float16), `.biases` (float16). Shapes:

- `gate_proj.weight` = `[E, I, packed_in]` where E=num_experts, I=moe_intermediate_size, packed_in = ceil(hidden_size * bits / 32)
- `up_proj.weight` = same shape as gate_proj
- `down_proj.weight` = `[E, hidden_size, packed_intermediate]`
- scales/biases shapes follow the same outer dims with `n_groups = in_dim / group_size`

The exact dtype + shape is whatever the source bundle was built from — we don't need to guess, we just stack what the blobs contain.

- [ ] **Step 1: Write the failing test**

Write `jang-tools/tests/jangspec/test_bundle_loader.py`:
```python
"""Round-trip tests for jang_tools.jangspec.bundle_loader."""

from pathlib import Path

import numpy as np
import pytest

from jang_tools.jangspec.bundle_loader import load_weights_from_bundle


def test_bundle_loader_returns_hot_core_tensors(jangspec_fixture_model: Path, tmp_path: Path):
    # Use the conftest fixture (Gemma-4-26B-A4B-it-JANG_4M) to build a
    # bundle, then verify the loader returns at least one hot-core tensor.
    from jang_tools.jangspec.builder import JangSpecBuilder

    out = tmp_path / "fx.jangspec"
    JangSpecBuilder(source_dir=jangspec_fixture_model, out_dir=out).build()

    weights = load_weights_from_bundle(out)
    # Hot core has embeddings, attention, norms, lm_head — pick one we know
    # must be present in any Gemma-4 JANG bundle.
    assert any(k.endswith("embed_tokens.weight") for k in weights), (
        "expected embed_tokens.weight in bundle weights"
    )
    assert any(k.endswith("self_attn.q_proj.weight") for k in weights), (
        "expected self_attn.q_proj.weight in bundle weights"
    )


def test_bundle_loader_reconstructs_expert_3d_stacks(
    jangspec_fixture_model: Path, tmp_path: Path
):
    from jang_tools.jangspec.builder import JangSpecBuilder
    from jang_tools.jangspec.manifest import load_manifest
    from jang_tools.jangspec import format as fmt

    out = tmp_path / "fx.jangspec"
    JangSpecBuilder(source_dir=jangspec_fixture_model, out_dir=out).build()

    weights = load_weights_from_bundle(out)
    manifest = load_manifest(out / fmt.MANIFEST_FILENAME)

    # For every expert base name in the manifest, the loader must emit
    # a 3D stacked tensor with leading dim == n_experts_per_layer.
    for base in manifest.expert_tensor_names:
        weight_key = f"{base}.weight"
        scales_key = f"{base}.scales"
        biases_key = f"{base}.biases"

        assert weight_key in weights, f"missing {weight_key}"
        assert scales_key in weights, f"missing {scales_key}"
        assert biases_key in weights, f"missing {biases_key}"

        wt = weights[weight_key]
        assert wt.ndim == 3, f"{weight_key} should be 3D, got {wt.shape}"
        assert wt.shape[0] == manifest.n_experts_per_layer, (
            f"{weight_key} leading dim {wt.shape[0]} != "
            f"n_experts_per_layer {manifest.n_experts_per_layer}"
        )


def test_bundle_loader_byte_parity_against_source(
    jangspec_fixture_model: Path, tmp_path: Path
):
    """The reconstructed expert 3D stacks should be byte-identical to slicing
    the source safetensors directly."""
    import json
    from safetensors import safe_open
    from jang_tools.jangspec.builder import JangSpecBuilder

    out = tmp_path / "fx.jangspec"
    JangSpecBuilder(source_dir=jangspec_fixture_model, out_dir=out).build()
    weights = load_weights_from_bundle(out)

    # Pick layer 0 and compare the gate_proj 3D tensor end-to-end.
    st_index = json.loads(
        (jangspec_fixture_model / "model.safetensors.index.json").read_text()
    )["weight_map"]
    base = next(
        b for b in (
            "model.language_model.layers.0.switch_mlp.gate_proj",
            "model.layers.0.switch_mlp.gate_proj",
            "language_model.model.layers.0.switch_mlp.gate_proj",
        )
        if f"{b}.weight" in st_index
    )
    src_shard = jangspec_fixture_model / st_index[f"{base}.weight"]
    with safe_open(src_shard, framework="numpy", device="cpu") as f:
        src_qweight = f.get_tensor(f"{base}.weight")
        src_scales = f.get_tensor(f"{base}.scales")
        src_biases = f.get_tensor(f"{base}.biases")

    rec_qweight = weights[f"{base}.weight"]
    rec_scales = weights[f"{base}.scales"]
    rec_biases = weights[f"{base}.biases"]

    # The bundle loader returns mx.array for MLX consumption; convert for
    # numpy equality comparison.
    import mlx.core as mx
    np.testing.assert_array_equal(np.array(rec_qweight, copy=False), src_qweight)
    np.testing.assert_array_equal(np.array(rec_scales, copy=False), src_scales)
    np.testing.assert_array_equal(np.array(rec_biases, copy=False), src_biases)
```

- [ ] **Step 2: Run and confirm failure**

```bash
cd /Users/eric/jang/jang-tools && python3 -m pytest tests/jangspec/test_bundle_loader.py -v
```

Expected: `ModuleNotFoundError: No module named 'jang_tools.jangspec.bundle_loader'`.

- [ ] **Step 3: Implement bundle_loader.py**

Write `jang-tools/jang_tools/jangspec/bundle_loader.py`:
```python
"""
Bundle weight loader: read a .jangspec bundle and produce the
{tensor_name: mx.array} dict that mlx-lm models expect.

This is the inverse of `jang_tools.jangspec.builder.JangSpecBuilder`. The
builder splits a source JANG model into hot_core.safetensors + per-expert
blobs; this module recombines them back into the canonical layout that
mlx-lm's `load_weights_from_safetensors` would produce if pointed at the
original directory.

Used by:
- Plan 5's bundle validation script (Python token-equality check).
- Future Plan 6 Swift port (as a Python reference oracle when debugging).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import mlx.core as mx
import numpy as np
from safetensors.numpy import load_file

from . import format as fmt
from .blob import unpack_expert_blob
from .reader import JangSpecReader

_LAYER_RE = re.compile(r"\.?layers\.(\d+)\.")


def _layer_idx(base_name: str) -> int:
    m = _LAYER_RE.search(base_name)
    if not m:
        raise ValueError(f"cannot parse layer index from {base_name!r}")
    return int(m.group(1))


def load_weights_from_bundle(bundle_dir: Path | str) -> Dict[str, "mx.array"]:
    """Load every tensor a model needs from a `.jangspec` bundle.

    The returned dict has the same keys and dtypes you would get from
    `mx.load("model.safetensors")` on the source JANG_xxx directory: the
    hot-core tensors copied through unchanged, plus per-expert blobs
    restacked into 3D `[E, ...]` tensors under their original
    `switch_mlp.{gate,up,down}_proj.{weight,scales,biases}` names.

    Memory: hot-core tensors are mmap'd (zero-copy). Expert stacks are
    materialized as new mx.array instances because `mx.stack` over many
    blob slices is required to produce a contiguous tensor.
    """

    bundle_dir = Path(bundle_dir)
    reader = JangSpecReader(bundle_dir)

    out: Dict[str, mx.array] = {}

    # 1. Hot core — mmap'd safetensors, copy keys directly.
    hot_core_path = bundle_dir / fmt.HOT_CORE_FILENAME
    if not hot_core_path.exists():
        raise FileNotFoundError(f"missing {hot_core_path}")
    hot_np = load_file(str(hot_core_path))
    for name, arr in hot_np.items():
        out[name] = mx.array(arr)

    # 2. Per-layer expert restack. Iterate every expert tensor base name
    #    in the manifest, group by layer, then for each (layer, base) pair
    #    walk the experts in order and stack their qweight/scales/biases.
    manifest = reader.manifest
    by_layer: dict[int, list[str]] = {}
    for base in manifest.expert_tensor_names:
        by_layer.setdefault(_layer_idx(base), []).append(base)

    for layer_idx, base_names in by_layer.items():
        # Each layer has 3 base names (gate_proj, up_proj, down_proj). We
        # iterate experts once per layer and dispatch to all 3 bases as
        # we go — avoids re-loading the same blob 3 times.

        # Collect per-base buffers as Python lists, then stack at the end.
        buffers: dict[str, dict[str, list[np.ndarray]]] = {
            base: {"weight": [], "scales": [], "biases": []} for base in base_names
        }
        # Map kind name fragment -> the matching base name in this layer.
        # The blob's tensors don't carry the full path, just the kind enum,
        # so we look up by which base name ends in the same suffix.
        kind_to_base: dict[str, str] = {}
        for base in base_names:
            for kind in ("gate_proj", "up_proj", "down_proj"):
                if base.endswith(f".switch_mlp.{kind}"):
                    kind_to_base[kind] = base

        for expert_id in range(reader.n_experts_per_layer):
            blob = reader.load_expert(layer_idx=layer_idx, expert_id=expert_id)
            for tensor in blob.tensors.gate, blob.tensors.up, blob.tensors.down:
                pass  # unused — see below

            # `blob.tensors` is an ExpertTensors dataclass with .gate/.up/.down,
            # each a (qweight, scales, biases) numpy triple.
            for kind_name, triple in (
                ("gate_proj", blob.tensors.gate),
                ("up_proj", blob.tensors.up),
                ("down_proj", blob.tensors.down),
            ):
                base = kind_to_base.get(kind_name)
                if base is None:
                    continue
                qw, sc, bi = triple
                buffers[base]["weight"].append(qw)
                buffers[base]["scales"].append(sc)
                buffers[base]["biases"].append(bi)

        # Stack and emit as mx.array.
        for base in base_names:
            stacked_w = np.stack(buffers[base]["weight"], axis=0)
            stacked_s = np.stack(buffers[base]["scales"], axis=0)
            stacked_b = np.stack(buffers[base]["biases"], axis=0)
            out[f"{base}.weight"] = mx.array(stacked_w)
            out[f"{base}.scales"] = mx.array(stacked_s)
            out[f"{base}.biases"] = mx.array(stacked_b)

    reader.close()
    return out


def load_jang_model_from_bundle(bundle_dir: Path | str):
    """Load an mlx-lm model from a `.jangspec` bundle.

    Mirrors `jang_tools.loader.load_jang_model` / `load_jang_vlm_model` but
    sources weights from the bundle's `bundle_loader.load_weights_from_bundle`
    helper instead of the source JANG_xxx directory's shards. Intended for
    the Plan 5 validation script.

    Returns: (model, tokenizer)
    """
    from transformers import AutoTokenizer

    bundle_dir = Path(bundle_dir)
    target_dir = bundle_dir / "target"

    # The model factory needs config.json + jang_config.json. Both are
    # copied verbatim into target/ at bundle build time (see
    # JangSpecBuilder._copy_tokenizer).
    if not (target_dir / "config.json").exists():
        raise FileNotFoundError(
            f"bundle is missing target/config.json — built with an older builder?"
        )

    # Load weights via the bundle reader.
    weights = load_weights_from_bundle(bundle_dir)

    # Build the model skeleton via mlx-lm's factory using config.json.
    import json
    config = json.loads((target_dir / "config.json").read_text())

    # Pick the right factory: VLM if model has vision/audio fields,
    # plain LLM otherwise. Gemma-4 is multimodal, so VLM.
    is_vlm = any(
        k in config for k in ("vision_config", "vision_tower", "audio_config")
    ) or "Conditional" in str(config.get("architectures", []))

    if is_vlm:
        from mlx_vlm.utils import load_model
        model, _ = load_model(target_dir, lazy=True)
    else:
        from mlx_lm.utils import load_model
        model, _ = load_model(target_dir, lazy=True)

    # Apply the model's sanitize step (renames switch_mlp keys, etc.).
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # Convert the weight dict to (key, value) pairs and load.
    model.load_weights(list(weights.items()), strict=False)

    tokenizer = AutoTokenizer.from_pretrained(str(bundle_dir))
    return model, tokenizer
```

- [ ] **Step 4: Run tests against the Gemma fixture**

```bash
cd /Users/eric/jang/jang-tools && python3 -m pytest tests/jangspec/test_bundle_loader.py -v -s
```

Expected: 3 tests pass (or skip if the Gemma fixture is unavailable on this machine — should be present per the conftest default).

If the byte-parity test fails:
- The blob restack order may not match the original 3D stacking order. Check that experts are iterated `0..E-1` and `np.stack(axis=0)` is used.
- The hot-core copy may have a key with a different prefix. Print `weights.keys()` vs the source's `model.safetensors.index.json` and reconcile.
- STOP and report BLOCKED with the first 5 keys from each side and a `np.array_equal` diff of the first 16 bytes.

- [ ] **Step 5: Commit**

```bash
cd /Users/eric/jang && git add jang-tools/jang_tools/jangspec/bundle_loader.py
git add -f jang-tools/tests/jangspec/test_bundle_loader.py
git commit -m "jang-spec: bundle_loader — reconstruct mlx-lm weight dict from .jangspec"
```

---

## Task 2: End-to-end validation script

**Files:**
- Create: `jang-tools/scripts/validate_bundle_gemma4.py`

**Background.** This is the actual Plan 5 deliverable: load Gemma-4-26B-A4B from both the source `JANG_4M/` directory and the `.jangspec` bundle, generate tokens for the same prompt with greedy decode, and assert the token sequences match. If they do, the bundle is functionally equivalent to the source.

The script is a single-file utility, not a unit test. It expects RAM (the model is ~16 GB resident) and runs end-to-end inference. **Do not run it without RAM headroom.**

- [ ] **Step 1: Implement the validation script**

Write `jang-tools/scripts/validate_bundle_gemma4.py`:
```python
#!/usr/bin/env python3
"""
Plan 5 validation: load Gemma-4-26B-A4B-it-JANG_4M two ways and confirm
greedy decode produces identical token sequences.

  Path A (baseline):  jang_tools.loader.load_jang_vlm_model on the source
                      directory /Users/eric/jang/models/Gemma-4-26B-A4B-it-JANG_4M
  Path B (bundle):    jang_tools.jangspec.bundle_loader.load_jang_model_from_bundle
                      on /tmp/jangcore-fixtures/Gemma-4-26B-A4B-it-JANG_4M.jangspec

Usage:
    python3 jang-tools/scripts/validate_bundle_gemma4.py

Environment:
    GEMMA_BUNDLE         override bundle path
    GEMMA_SOURCE         override source JANG_4M dir
    GEMMA_PROMPT         override prompt (default: "The capital of France is")
    GEMMA_MAX_TOKENS     override max generated tokens (default: 16)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

DEFAULT_SOURCE = "/Users/eric/jang/models/Gemma-4-26B-A4B-it-JANG_4M"
DEFAULT_BUNDLE = "/tmp/jangcore-fixtures/Gemma-4-26B-A4B-it-JANG_4M.jangspec"
DEFAULT_PROMPT = "The capital of France is"
DEFAULT_MAX = 16


def _generate_greedy(model, tokenizer, prompt: str, max_tokens: int) -> list[int]:
    """Run greedy decode and return the generated token IDs (no sampling)."""
    import mlx.core as mx

    input_ids = tokenizer.encode(prompt, return_tensors=None)
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    tokens = list(input_ids)
    cache = None

    for _ in range(max_tokens):
        x = mx.array([tokens[-1]] if cache is not None else tokens)
        x = x.reshape(1, -1)
        logits = model(x, cache=cache)
        if hasattr(logits, "logits"):
            logits = logits.logits
        next_token = int(mx.argmax(logits[0, -1, :]).item())
        tokens.append(next_token)
        # mlx-lm models construct cache on first call; subsequent calls
        # need the same cache instance. Pull it from the model if present.
        if cache is None:
            try:
                from mlx_lm.models.cache import make_prompt_cache
                cache = make_prompt_cache(model)
                # Replay the prompt through the new cache to seed it.
                _ = model(mx.array([tokens[:-1]]), cache=cache)
            except Exception:
                cache = None  # fall through and re-encode each step

    # Return only the newly generated tokens.
    return tokens[len(input_ids):]


def main() -> int:
    source = Path(os.environ.get("GEMMA_SOURCE", DEFAULT_SOURCE))
    bundle = Path(os.environ.get("GEMMA_BUNDLE", DEFAULT_BUNDLE))
    prompt = os.environ.get("GEMMA_PROMPT", DEFAULT_PROMPT)
    max_tokens = int(os.environ.get("GEMMA_MAX_TOKENS", str(DEFAULT_MAX)))

    if not source.exists():
        print(f"  source missing: {source}")
        return 2
    if not (bundle / "jangspec.json").exists():
        print(f"  bundle missing: {bundle}")
        print(f"  build with: jang spec build {source} --out {bundle}")
        return 2

    print("=" * 64)
    print(f"  Plan 5 — Gemma-4-26B-A4B bundle validation")
    print(f"  prompt:     {prompt!r}")
    print(f"  max_tokens: {max_tokens}")
    print("=" * 64)

    # --- Path A: source directory ---
    print("\n[A] Loading source via jang_tools.loader ...")
    t0 = time.time()
    from jang_tools.loader import load_jang_vlm_model
    model_a, tok_a = load_jang_vlm_model(str(source))
    print(f"    loaded in {time.time() - t0:.1f}s")

    print("    generating ...")
    t0 = time.time()
    tokens_a = _generate_greedy(model_a, tok_a, prompt, max_tokens)
    print(f"    generated {len(tokens_a)} tokens in {time.time() - t0:.1f}s")
    decoded_a = tok_a.decode(tokens_a)
    print(f"    text:   {decoded_a!r}")
    print(f"    tokens: {tokens_a}")

    # Free Path A weights aggressively before loading Path B — the model
    # is ~16 GB resident and we don't want both in RAM at once.
    del model_a
    import gc
    gc.collect()

    # --- Path B: .jangspec bundle ---
    print("\n[B] Loading bundle via jangspec.bundle_loader ...")
    t0 = time.time()
    from jang_tools.jangspec.bundle_loader import load_jang_model_from_bundle
    model_b, tok_b = load_jang_model_from_bundle(bundle)
    print(f"    loaded in {time.time() - t0:.1f}s")

    print("    generating ...")
    t0 = time.time()
    tokens_b = _generate_greedy(model_b, tok_b, prompt, max_tokens)
    print(f"    generated {len(tokens_b)} tokens in {time.time() - t0:.1f}s")
    decoded_b = tok_b.decode(tokens_b)
    print(f"    text:   {decoded_b!r}")
    print(f"    tokens: {tokens_b}")

    # --- Compare ---
    print("\n" + "=" * 64)
    if tokens_a == tokens_b:
        print("  ✅ TOKEN-LEVEL MATCH — bundle is functionally equivalent to source")
        return 0
    else:
        print("  ❌ MISMATCH — bundle path produces different tokens")
        print(f"     source tokens: {tokens_a}")
        print(f"     bundle tokens: {tokens_b}")
        # First divergence
        for i, (a, b) in enumerate(zip(tokens_a, tokens_b)):
            if a != b:
                print(f"     first diff at index {i}: source={a} bundle={b}")
                break
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Sanity-check the script imports without running it**

```bash
cd /Users/eric/jang && python3 -c "
import importlib.util
spec = importlib.util.spec_from_file_location('v', 'jang-tools/scripts/validate_bundle_gemma4.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('script imports cleanly, main signature:', mod.main.__name__)
"
```

Expected: `script imports cleanly, main signature: main`. **Do not run `main()` here** — it loads the model, which uses RAM.

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/jang && git add jang-tools/scripts/validate_bundle_gemma4.py
git commit -m "jang-spec: validate_bundle_gemma4 — token-equality validation script"
```

---

## Task 3: STATUS update

**Files:**
- Modify: `docs/superpowers/notes/jang-spec-STATUS.md`

- [ ] **Step 1: Update the status doc**

Edit `docs/superpowers/notes/jang-spec-STATUS.md`:

- Plans table: mark Plan 5 row with status **CODE READY (run pending)** and branch `jang-spec-plan5-bundle-python-validation`. Artifacts: `bundle_loader.py`, `validate_bundle_gemma4.py`, 3 unit tests
- Tests line: bump Python count by 3 (from 14 to 17)
- TL;DR: add a sentence noting Python validation script is ready, awaiting RAM-free run on Gemma-4-26B
- Immediate next: rewrite for Plan 6 (Swift forward pass on Gemma-4-26B, MoE in RAM, no streaming yet — to be designed once Plan 5 validates the bundle is correct)
- Add a "Plan 5 notes" subsection summarizing what `bundle_loader` does and how to invoke the validation script

- [ ] **Step 2: Commit STATUS update**

```bash
cd /Users/eric/jang && git add docs/superpowers/notes/jang-spec-STATUS.md
git commit -m "jang-spec: update STATUS after Plan 5 (code ready, run pending)"
```

---

## Task 4: Final sweep

**Files:** none

- [ ] **Step 1: Re-run jangspec test suite (excluding the run-pending validation script)**

```bash
cd /Users/eric/jang/jang-tools && python3 -m pytest tests/jangspec/ -v 2>&1 | tail -25
```

Expected: 17 tests pass (14 from prior plans + 3 new from `test_bundle_loader.py`).

- [ ] **Step 2: Print plan commit log**

```bash
cd /Users/eric/jang && git log --oneline jang-spec-plan4-metal-matmul..HEAD
```

Expected: 4 commits from this plan.

- [ ] **Step 3: Print expected next manual run**

The validation script is committed but **not executed**. When RAM is free, run:

```bash
cd /Users/eric/jang && python3 jang-tools/scripts/validate_bundle_gemma4.py
```

Expected output:
- "[A] Loading source via jang_tools.loader ..." → loads model, generates 16 tokens
- "[B] Loading bundle via jangspec.bundle_loader ..." → loads model, generates 16 tokens
- Final line: "✅ TOKEN-LEVEL MATCH" if the bundle is correct, "❌ MISMATCH" with first-diff index otherwise

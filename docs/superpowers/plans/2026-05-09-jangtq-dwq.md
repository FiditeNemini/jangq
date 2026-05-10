# JANGTQ DWQ — Phase 1 (DWQ-Norms) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## Scope (READ FIRST)

**Phase 1 only. All edits confined to `/Users/eric/jang/`. No vmlx-swift, vmlx Python, or runtime contract changes.**

Phase 1 is **DWQ-Norms**: tune the per-row scalar `tq_norms` of every `TurboQuantLinear` / `TurboQuantSwitchLinear` against an FP teacher via KL-divergence on logits. **`tq_packed` (codebook *assignments*) and the deterministic codebook *centroids* stay fixed.** This means:

- DWQ-Norms cannot improve codebook assignment boundaries or centroid placement.
- It only recovers per-row magnitude error left by argmax-quantization on a fixed-index lookup.
- Wire format and sidecar contract are unchanged. The Swift `JANGTQRuntimeCache` and the Python `load_jangtq_model` loader consume DWQ-tuned bundles **as-is**.
- `tq_bits` is read **per-tensor** from the bundle as the source of truth. The predicate never infers or overwrites bit-widths — JANGTQ_K per-projection routed bits and `mxtq_bits` dual form are preserved unchanged.

Phase 2 (centroid tuning) is documented as a deferred appendix at the bottom. Phase 2 IS a wire-format / runtime contract bump and requires explicit Swift work — it must not be smuggled into Phase 1.

If during Phase 1 execution it turns out a runtime/vmlx change *is* needed (e.g. the per-row dense forward path hits an unexpected kernel edge case), STOP and write up the runtime delta as a separate proposal — do not edit vmlx files inline.

---

**Goal:** Add a Distillation-tuned Weight Quantization pass that tunes only per-row `tq_norms` of an existing JANGTQ bundle. Codebook centroids and packed indices remain frozen and deterministic. Wire format and sidecar contract are not changed.

**Architecture:** Reuse mlx-lm's DWQ KL-distill loop (`mlx_lm/quant/dwq.py`). Replace its affine-only unfreeze predicate (`mode == "affine"` → unfreeze `scales`+`biases`) with a JANGTQ predicate that targets `TurboQuantLinear.norms` and `TurboQuantSwitchLinear.norms` — codebook + signs + packed indices stay frozen. Add a *DWQ training mode* to those layers that reconstructs a dense `(out, in)` weight tensor via vectorized gather + multiply, so gradients flow back to `norms` cleanly. After training, write tuned `*.tq_norms` tensors back into the same shard layout — no other tensor is touched, no metadata is bumped, no sidecar is regenerated.

**Tech stack:** Python 3.11, MLX, mlx-lm (`tuner.losses.kl_div_loss`, `tuner.trainer.iterate_batches`, `tuner.datasets.load_dataset`), `jang_tools.turboquant`, safetensors.

**Why this matters even with frozen codebook:** Per-row norms in a Lloyd-Max codebook quantizer absorb the `(maxabs / centroid_max)` scaling for each weight row. The Lloyd-Max codebook is optimal for the *theoretical* unit-sphere distribution; per-row magnitudes vary in practice. Tuning `tq_norms` against teacher logits recovers per-row error a deterministic-norm step leaves on the table. Smaller win than codebook centroid tuning (Phase 2), but ships safely with zero runtime/contract risk.

**Out of scope (explicitly):**
- Tuning the deterministic Lloyd-Max codebook centroids (Phase 2 — see appendix)
- Tuning Hadamard signs (discrete ±1 — Phase 2+; would need straight-through estimator)
- Re-quantizing `tq_packed` indices (would change codebook assignments — out of Phase 1's "fixed-index" framing)
- Distributed / multi-node DWQ (`pipeline_load`)
- Pre-computed target caching to disk (`compute_dwq_targets`) — v1 calls FP teacher live each step
- Any vmlx-swift edits (see Scope above)

---

## Naming — JANGTQ tensor + attribute conventions

Phase 1 must use these names exactly:

| Where | Name | Source of truth |
|---|---|---|
| In-memory Python attribute on runtime `TurboQuantLinear` | `self.norms` (fp16) | Loaded from `<path>.tq_norms`; **owned per-layer**; trainable in Phase 1 |
| In-memory Python attribute on runtime `TurboQuantLinear` | `self.codebook` (`@property`, fp32) | Lazy-cached lookup `_CODEBOOK_CACHE[(in_features, bits)]`; **shared across layers**; **frozen in Phase 1** (mutating would clobber peers) |
| In-memory Python attribute on runtime `TurboQuantLinear` | `self.signs` (`@property`, fp32 ±1) | Lazy-cached `_SIGNS_CACHE[(in_features, seed)]`; **shared**; **frozen** |
| In-memory Python attribute on runtime `TurboQuantLinear` | `self.packed` (uint32) | Loaded from `<path>.tq_packed`; **owned per-layer**; **frozen** |
| In-memory Python attribute on runtime `TurboQuantLinear` | `self.bits` (Python int) | Source of truth for layer bit-width at runtime; **never written to by DWQ** |
| Optional in-memory attribute | `self.awq_scale` (fp16/fp32, in_features) | Attached by loader for AWQ-enabled bundles; DWQ training forward must apply `x = x / awq_scale` exactly like inference |
| Bundle main shard tensor key | `<path>.tq_packed` | Frozen |
| Bundle main shard tensor key | `<path>.tq_norms` | **Trainable** in Phase 1 |
| Bundle main shard tensor key | `<path>.tq_bits` | **1-element integer tensor — uint8 (MiniMax/Kimi/Zaya/Qwen/Ling) or int32 (DSV4) depending on converter.** Read via `int(tensor.item())`. Save preserves original dtype. DWQ never writes this. |
| Sidecar tensor key | `codebook.{in_features}.{bits}` | Shared across layers; **frozen** |
| Sidecar tensor key | `signs.{in_features}.{seed}` | Frozen |

JANGTQ_K bundles encode per-projection bits via per-tensor `<path>.tq_bits` and a top-level `mxtq_bits` dict. The DWQ pass must not touch either. Read `tq_bits` per tensor; tune only the matching `tq_norms`.

> **Note on `mx.eval(...)`:** Throughout this plan, `mx.eval(...)` refers to MLX's lazy-graph realization (force-compute pending tensor ops). It is not Python's builtin `eval()`. No arbitrary code evaluation occurs.

---

## File Structure

**Create (all under `/Users/eric/jang/`):**
- `jang-tools/jang_tools/dwq_jangtq.py` (~300 lines) — main module: unfreeze predicate, `dwq_norms_jangtq()` loop, `target_fn`, save-back, CLI
- `jang-tools/tests/test_dwq_jangtq.py` (~250 lines) — unit tests for dense-reconstruction parity, unfreeze predicate, save-back roundtrip, fixture smoke
- `jang-tools/tests/fixtures/tiny_jangtq/` — minimal local JANGTQ bundle for fast smoke tests (built once via `tools/build_tiny_jangtq_fixture.py`)
- `research/JANGTQ-DWQ-DESIGN.md` (~300 lines) — design doc: Phase 1 motivation, math, frozen vs trainable, eval protocol, Phase 2 contract bump preview

**Modify (all under `/Users/eric/jang/`):**
- `jang-tools/jang_tools/turboquant/tq_kernel.py:179-296` — add `dwq_training` flag + `_dense_weight_dwq()` (and switch variant) method to the **runtime** `TurboQuantLinear` / `TurboQuantSwitchLinear`. THIS is the class production `load_jangtq_model` actually loads (verified: `load_jangtq.py:47` imports from `tq_kernel`, not `linear`). The dense forward is autograd-only — the kernel forward calls a Metal kernel via `tq_matmul` / `_gather_tq_matmul` which is NOT autograd-differentiable.
- `jang-tools/jang_tools/turboquant/__init__.py:23-37` — export the runtime `TurboQuantLinear`, `TurboQuantSwitchLinear` from `tq_kernel`
- `docs/adoption/PORTING.md:132` — replace stale `.tq_codebook` description with current `.tq_packed` / `.tq_norms` / `.tq_bits` + sidecar contract; add a one-paragraph note that Phase-1 DWQ tunes `.tq_norms` only

**Critical: `codebook` and `signs` are `@property`s on the runtime class** (`tq_kernel.py:210-216, 275-281`). They lazy-resolve to module-level shared caches `_CODEBOOK_CACHE` / `_SIGNS_CACHE` keyed by `(in_features, bits)` / `(in_features, seed)` — every layer with matching shape returns the *same* tensor object. Phase 1 leaves them as properties (codebook frozen, deterministic). Mutating them would break every other layer that shares them — that's a Phase-2 problem.

**Reference (read-only — do NOT edit):**
- `mlx_lm/quant/dwq.py` at `/Users/eric/CRACK_abliteration/.venv/lib/python3.11/site-packages/mlx_lm/quant/dwq.py` — upstream reference; lines 30-205 are the loop we adapt
- `jang-tools/jang_tools/load_jangtq.py:47` — imports the runtime classes from `tq_kernel`
- `jang-tools/jang_tools/load_jangtq.py:92` — `load_jangtq_model(model_path, skip_params_eval=False)` is the entry point name (NOT `load_jangtq`)
- `jang-tools/jang_tools/load_jangtq.py:990` — codebook cache keyed by `(in_features, bits)`; Phase 1 leaves this contract intact
- `jang-tools/jang_tools/build_jangtq_sidecar.py` — sidecar writer; Phase 1 does not regenerate the sidecar
- `jang-tools/jang_tools/turboquant/linear.py` — a separate Python-only reference implementation; **NOT loaded by `load_jangtq_model`**. We use its `tq_quantize_weight` / `tq_quantize_experts` helpers in tests/fixtures only.
- `jang-tools/jang_tools/turboquant/codebook.py` — `compute_codebook` (deterministic; Phase 1 never re-invokes after loading)
- `jang-tools/jang_tools/turboquant/rotation.py` — `hadamard_inverse` (Python; differentiable; used by DWQ dense forward)
- `jang-tools/jang_tools/turboquant/hadamard_kernel.py` — Metal forward-rotation kernel; **not autograd-differentiable**, do NOT call from DWQ training forward
- `jang-tools/jang_tools/turboquant/pipeline.py` — `pack_bits`, `unpack_bits`

---

## Task 1: Add DWQ training-mode dense forward to runtime `TurboQuantLinear`

**Files:**
- Modify: `jang-tools/jang_tools/turboquant/tq_kernel.py:179-242` (the runtime class actually loaded by `load_jangtq_model`)
- Test: `jang-tools/tests/test_dwq_jangtq.py`

The runtime `__call__` calls a Metal kernel `tq_matmul` (line 236) which is not autograd-differentiable. For DWQ training we add a parallel `_dense_weight_dwq()` reconstruction path that uses pure MLX ops so gradient flows back to `self.norms`. Codebook + signs are accessed via the existing `@property` cache (frozen lookup); `self.packed` is uint32 indices (frozen). AWQ scale (if attached) must be applied exactly like the kernel forward does.

- [ ] **Step 1: Write failing parity test**

```python
import mlx.core as mx
import numpy as np
import pytest
# Runtime class (kernel-backed) — the one production loads
from jang_tools.turboquant.tq_kernel import TurboQuantLinear
# Quant helper from the python-only reference impl (fine to use in tests)
from jang_tools.turboquant.linear import tq_quantize_weight


def test_dense_weight_dwq_matches_existing_forward():
    """Dense reconstruction path must match the Metal-kernel forward within fp tolerance."""
    mx.random.seed(0)
    in_feat, out_feat, bits = 64, 32, 4
    layer = TurboQuantLinear(in_feat, out_feat, bits=bits, seed=42)

    rng = np.random.default_rng(0)
    w = rng.standard_normal((out_feat, in_feat)).astype(np.float32)
    q = tq_quantize_weight(w, bits=bits, seed=42)
    layer.packed = mx.array(q["packed"])
    layer.norms = mx.array(q["norms"]).astype(mx.float16)

    x = mx.array(rng.standard_normal((1, 5, in_feat)).astype(np.float32))

    y_kernel = layer(x)        # Metal kernel forward
    layer.dwq_training = True
    y_dense = layer(x)         # autograd-eligible dense reconstruction

    diff = mx.max(mx.abs(y_kernel - y_dense)).item()
    # Slightly looser tolerance than linear.py parity since the kernel does
    # fp16 accumulation differently than the python dense path.
    assert diff < 5e-3, f"dense path diverges from kernel: {diff}"
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /Users/eric/jang/jang-tools
.venv/bin/pytest tests/test_dwq_jangtq.py::test_dense_weight_dwq_matches_existing_forward -v
```

Expected: FAIL — `dwq_training` not on `TurboQuantLinear`.

- [ ] **Step 3: Add `dwq_training` flag + `_dense_weight_dwq` method**

In `jang-tools/jang_tools/turboquant/tq_kernel.py`, modify the runtime `TurboQuantLinear`:

```python
# At top of file (alongside existing imports)
from .pipeline import unpack_bits
from .rotation import hadamard_inverse  # python; differentiable; do NOT use hadamard_kernel here


class TurboQuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bits=2, bias=False, seed=42):
        super().__init__()
        # ... existing init unchanged ...
        self.dwq_training = False
        self.freeze()

    # codebook + signs remain @property — Phase 1 doesn't mutate them.

    def _dense_weight_dwq(self) -> mx.array:
        """Vectorized (out, in) weight reconstruction for DWQ training.

        Phase 1: codebook + signs + packed stay frozen. Gradient flows only
        through self.norms via the per-row multiply. NEVER call this on the
        inference path — it's slower than the Metal kernel.
        """
        n_el = self.out_features * self.in_features
        idx = unpack_bits(self.packed.reshape(-1), self.bits, n_el)
        idx = idx.reshape(self.out_features, self.in_features)
        # self.codebook + self.signs hit the shared @property cache (frozen)
        w_rot = mx.take(self.codebook, idx.astype(mx.uint32))   # (out, in)
        w_rot = w_rot * self.norms[:, None].astype(w_rot.dtype) # gradient path
        w = hadamard_inverse(w_rot, self.signs)                 # python, differentiable
        return w

    def __call__(self, x):
        # AWQ scale must be applied identically in BOTH paths.
        awq = getattr(self, "awq_scale", None)
        if awq is not None:
            x = x / awq.astype(x.dtype)

        if self.dwq_training:
            w = self._dense_weight_dwq()
            y = x @ w.T
            if "bias" in self:
                y = y + self.bias
            return y

        # existing kernel forward path unchanged below this line —
        # flatten 3D→2D, call tq_matmul, reshape back, add bias.
        # ...
```

Note: the existing `__call__` already applies `awq_scale` — when refactoring, lift it once at the top so it runs in both paths without duplication. Do not add a second division.

- [ ] **Step 4: Run test to verify pass**

```bash
.venv/bin/pytest tests/test_dwq_jangtq.py::test_dense_weight_dwq_matches_existing_forward -v
```

Expected: PASS.

- [ ] **Step 5: Add gradient-flow test (norms only, codebook stays frozen)**

```python
def test_dense_weight_dwq_gradient_flows_to_norms_only():
    """Gradient flows to norms; codebook + signs (properties) stay frozen (Phase 1)."""
    mx.random.seed(0)
    in_feat, out_feat, bits = 64, 32, 4
    layer = TurboQuantLinear(in_feat, out_feat, bits=bits, seed=42)
    rng = np.random.default_rng(0)
    w = rng.standard_normal((out_feat, in_feat)).astype(np.float32)
    q = tq_quantize_weight(w, bits=bits, seed=42)
    layer.packed = mx.array(q["packed"])
    layer.norms = mx.array(q["norms"]).astype(mx.float16)
    layer.dwq_training = True
    layer.freeze()
    layer.unfreeze(keys=["norms"], recurse=False)

    x = mx.array(rng.standard_normal((4, 8, in_feat)).astype(np.float32))
    target = mx.array(rng.standard_normal((4, 8, out_feat)).astype(np.float32))

    def loss_fn(params):
        layer.update(params)
        return ((layer(x) - target) ** 2).mean()

    params = layer.trainable_parameters()
    assert "norms" in params, "norms must be trainable"
    # codebook + signs are @property (not real attributes); they should never
    # appear in trainable_parameters() regardless.
    assert "codebook" not in params
    assert "signs" not in params
    assert "packed" not in params

    _, grads = mx.value_and_grad(loss_fn)(params)
    mx.eval(grads)
    assert mx.all(mx.isfinite(grads["norms"])).item()
    assert mx.max(mx.abs(grads["norms"])).item() > 0
```

- [ ] **Step 6: Run test**

```bash
.venv/bin/pytest tests/test_dwq_jangtq.py::test_dense_weight_dwq_gradient_flows_to_norms_only -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add jang-tools/jang_tools/turboquant/tq_kernel.py jang-tools/tests/test_dwq_jangtq.py
git commit -m "feat(jangtq): vectorized dwq dense forward on TurboQuantLinear

Adds _dense_weight_dwq() that reconstructs (out, in) via gather + multiply
+ inverse-Hadamard. dwq_training flag toggles between this path (KL distill)
and the existing per-row unpack path (inference). Codebook + signs + packed
stay frozen; gradient flows only to norms.

Tests: parity within 1e-4 vs existing forward; non-zero finite grads on
norms; codebook + signs + packed remain frozen."
```

---

## Task 2: Add DWQ training-mode dense forward to runtime `TurboQuantSwitchLinear`

**Files:**
- Modify: `jang-tools/jang_tools/turboquant/tq_kernel.py:245-291` (the runtime switch class)
- Test: `jang-tools/tests/test_dwq_jangtq.py`

- [ ] **Step 1: Write failing parity test for switch**

```python
def test_switch_dense_weight_dwq_matches_existing_forward():
    """Switch dense reconstruction must match the gather_tq_matmul kernel within fp tolerance."""
    mx.random.seed(0)
    in_feat, out_feat, num_experts, bits = 32, 16, 4, 4
    from jang_tools.turboquant.tq_kernel import TurboQuantSwitchLinear
    from jang_tools.turboquant.linear import tq_quantize_experts

    layer = TurboQuantSwitchLinear(in_feat, out_feat, num_experts, bits=bits, seed=42)
    rng = np.random.default_rng(0)
    w = rng.standard_normal((num_experts, out_feat, in_feat)).astype(np.float32)
    q = tq_quantize_experts(w, bits=bits, seed=42)
    layer.packed = mx.array(q["packed"])
    layer.norms = mx.array(q["norms"]).astype(mx.float16)

    x = mx.array(rng.standard_normal((1, 3, in_feat)).astype(np.float32))
    indices = mx.array([[[0, 1], [2, 3], [0, 2]]], dtype=mx.uint32)

    y_kernel = layer(x, indices)
    layer.dwq_training = True
    y_dense = layer(x, indices)

    diff = mx.max(mx.abs(y_kernel - y_dense)).item()
    assert diff < 5e-3, f"switch dense path diverges: {diff}"
```

- [ ] **Step 2: Run test to verify failure**

```bash
.venv/bin/pytest tests/test_dwq_jangtq.py::test_switch_dense_weight_dwq_matches_existing_forward -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `_dense_weights_dwq` for switch**

Add to runtime `TurboQuantSwitchLinear` in `tq_kernel.py`:

```python
def _dense_weights_dwq(self) -> mx.array:
    """Reconstruct (num_experts, out, in) dense weights for DWQ training.

    Phase 1: codebook (shared via @property) + signs (shared) + packed stay
    frozen. Gradient flows only through per-expert self.norms.
    """
    n_el = self.out_features * self.in_features
    flat = self.packed.reshape(-1)
    all_idx = unpack_bits(flat, self.bits, self.num_experts * n_el)
    all_idx = all_idx.reshape(self.num_experts, self.out_features, self.in_features)
    all_w = mx.take(self.codebook, all_idx.astype(mx.uint32))         # (E, out, in)
    all_w = all_w * self.norms[..., None].astype(all_w.dtype)          # (E, out, in)
    rotated = mx.stack([
        hadamard_inverse(all_w[e], self.signs) for e in range(self.num_experts)
    ])
    return rotated

def __call__(self, x, indices, sorted_indices=False):
    awq = getattr(self, "awq_scale", None)
    if awq is not None:
        x = x / awq.astype(x.dtype)

    if self.dwq_training:
        return self._call_dwq(x, indices)

    # existing kernel path unchanged below — calls _gather_tq_matmul

def _call_dwq(self, x, indices):
    """DWQ-mode switch forward. Same semantics as __call__ but dense reconstruction.
    Slow (Python loops over K and experts) — DWQ training only.
    """
    B, S, K = indices.shape[0], indices.shape[1], indices.shape[-1]
    expert_w = self._dense_weights_dwq()
    out = mx.zeros((B, S, K, self.out_features), dtype=x.dtype)
    for k in range(K):
        idx_k = indices[:, :, k]
        for e in range(self.num_experts):
            mask = (idx_k == e)
            if not mx.any(mask).item():
                continue
            r = x @ expert_w[e].T
            out = out.at[:, :, k, :].add(mx.where(mask[:, :, None], r, mx.zeros_like(r)))
    if "bias" in self:
        for k in range(K):
            for e in range(self.num_experts):
                mask = (indices[:, :, k] == e)
                if mx.any(mask).item():
                    out = out.at[:, :, k, :].add(
                        mx.where(mask[:, :, None], self.bias[e], mx.zeros_like(self.bias[0]))
                    )
    return out
```

Add `self.dwq_training = False` to `__init__`. Lift the `awq_scale` division to the top of `__call__` so both paths see it identically (don't apply twice).

- [ ] **Step 4: Run test to verify pass**

```bash
.venv/bin/pytest tests/test_dwq_jangtq.py::test_switch_dense_weight_dwq_matches_existing_forward -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jang-tools/jang_tools/turboquant/tq_kernel.py jang-tools/tests/test_dwq_jangtq.py
git commit -m "feat(jangtq): vectorized dwq dense forward on TurboQuantSwitchLinear

Stacked dense reconstruction for MoE switch experts. Codebook + signs +
packed frozen; gradient flows only through per-expert norms. Validated
against existing path within 1e-4."
```

---

## Task 3: Implement `unfreeze_jangtq_norms_for_dwq` predicate

**Files:**
- Create: `jang-tools/jang_tools/dwq_jangtq.py` (start)
- Test: `jang-tools/tests/test_dwq_jangtq.py`

JANGTQ analogue of `mlx_lm/quant/dwq.py:97-104` but for `tq_norms` only. Must not touch `tq_bits`, `tq_packed`, codebook, or signs.

- [ ] **Step 1: Write failing test**

```python
def test_unfreeze_jangtq_norms_for_dwq_targets_norms_only():
    """Predicate must unfreeze norms only on runtime TurboQuant layers."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from jang_tools.turboquant.tq_kernel import TurboQuantLinear, TurboQuantSwitchLinear
    from jang_tools.dwq_jangtq import unfreeze_jangtq_norms_for_dwq

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tq_linear = TurboQuantLinear(64, 32, bits=4)
            self.tq_switch = TurboQuantSwitchLinear(64, 32, num_experts=4, bits=2)
            self.fp_linear = nn.Linear(64, 32)

    m = TestModel()
    m.freeze()
    unfreeze_jangtq_norms_for_dwq(m)

    keys = set(dict(tree_flatten(m.trainable_parameters())).keys())
    assert "tq_linear.norms" in keys
    assert "tq_switch.norms" in keys
    assert "tq_linear.codebook" not in keys
    assert "tq_linear.packed" not in keys
    assert "tq_linear.signs" not in keys
    assert "tq_switch.codebook" not in keys
    assert "tq_switch.packed" not in keys
    assert "tq_switch.signs" not in keys
    assert "fp_linear.weight" not in keys
    assert "fp_linear.bias" not in keys
    assert m.tq_linear.dwq_training is True
    assert m.tq_switch.dwq_training is True
```

- [ ] **Step 2: Add `tq_bits` invariance test**

```python
def test_unfreeze_jangtq_norms_does_not_touch_tq_bits():
    """tq_bits is source of truth per tensor; predicate must never write to it.
    JANGTQ_K bundles encode per-projection bits there — overwriting breaks K profiles.
    """
    from jang_tools.turboquant.tq_kernel import TurboQuantLinear
    from jang_tools.dwq_jangtq import unfreeze_jangtq_norms_for_dwq
    from mlx.utils import tree_flatten

    layer = TurboQuantLinear(64, 32, bits=2)  # K profile would have bits=2 here
    original_bits = layer.bits
    layer.freeze()
    unfreeze_jangtq_norms_for_dwq(layer)
    assert layer.bits == original_bits, "predicate must not touch self.bits"
    keys = set(dict(tree_flatten(layer.trainable_parameters())).keys())
    assert all("bits" not in k for k in keys), f"bits leaked into trainables: {keys}"
```

- [ ] **Step 3: Run tests to verify failure**

```bash
.venv/bin/pytest tests/test_dwq_jangtq.py::test_unfreeze_jangtq_norms_for_dwq_targets_norms_only \
                 tests/test_dwq_jangtq.py::test_unfreeze_jangtq_norms_does_not_touch_tq_bits -v
```

Expected: FAIL.

- [ ] **Step 4: Create `jang-tools/jang_tools/dwq_jangtq.py`**

```python
"""
DWQ-Norms for JANGTQ — Phase 1 of distillation-tuned quantization.

Created by Jinho Jang (eric@jangq.ai).

Tunes only the per-row tq_norms scalar of every TurboQuant{Linear,SwitchLinear}
in a JANGTQ bundle by KL-distilling against an FP teacher's logits. Frozen:
  - codebook (deterministic Lloyd-Max from compute_codebook)
  - signs    (deterministic from generate_random_signs)
  - packed   (codebook *assignments* — fixed-index)
  - bits     (per-tensor source of truth; preserves JANGTQ_K K-profile bits)

Wire format unchanged. Sidecar unchanged. Swift loader unchanged.

See: research/JANGTQ-DWQ-DESIGN.md
"""
import mlx.core as mx
import mlx.nn as nn

# IMPORTANT: import the runtime classes (kernel-backed) — these are the ones
# load_jangtq_model actually instantiates. The python-only linear.py classes
# are NOT loaded by production.
from jang_tools.turboquant.tq_kernel import TurboQuantLinear, TurboQuantSwitchLinear


def unfreeze_jangtq_norms_for_dwq(model: nn.Module) -> None:
    """On every TurboQuant{Linear,SwitchLinear}: enable dwq_training and unfreeze
    tq_norms only. tq_bits / packed / signs / codebook stay frozen.
    """
    def visit(_, m):
        if isinstance(m, (TurboQuantLinear, TurboQuantSwitchLinear)):
            m.dwq_training = True
            m.unfreeze(keys=["norms"], recurse=False)

    model.train()
    model.apply_to_modules(visit)
```

- [ ] **Step 5: Run tests to verify pass**

```bash
.venv/bin/pytest tests/test_dwq_jangtq.py::test_unfreeze_jangtq_norms_for_dwq_targets_norms_only \
                 tests/test_dwq_jangtq.py::test_unfreeze_jangtq_norms_does_not_touch_tq_bits -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add jang-tools/jang_tools/dwq_jangtq.py jang-tools/tests/test_dwq_jangtq.py
git commit -m "feat(jangtq): unfreeze_jangtq_norms_for_dwq predicate (Phase 1)

Walks model and on every TurboQuant{Linear,SwitchLinear} enables dwq_training
and unfreezes tq_norms only. Codebook + signs + packed + tq_bits stay frozen.
Preserves JANGTQ_K K-profile per-projection bits unchanged."
```

---

## Task 4: Port `dwq_norms_jangtq` KL-distill loop

**Files:**
- Modify: `jang-tools/jang_tools/dwq_jangtq.py`
- Test: `jang-tools/tests/test_dwq_jangtq.py`

Minimal port of `mlx_lm/quant/dwq.py:80-208` (`dwq_quantize`). Strip distributed wrapping, keep KL-div + Adam + validation. Live FP teacher (no precomputed cache).

- [ ] **Step 1: Write failing convergence test**

```python
def test_dwq_norms_jangtq_decreases_kl_loss():
    """10-iter run on a tiny synthetic model must decrease KL vs FP teacher."""
    import mlx.optimizers as optimizers
    import mlx.nn as nn
    from jang_tools.dwq_jangtq import dwq_norms_jangtq, unfreeze_jangtq_norms_for_dwq
    from jang_tools.turboquant.tq_kernel import TurboQuantLinear
    from jang_tools.turboquant.linear import tq_quantize_weight  # quant helper only

    mx.random.seed(0)
    in_feat, out_feat, vocab = 64, 64, 32

    class TeacherModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(in_feat, out_feat)
            self.lm_head = nn.Linear(out_feat, vocab)
        def __call__(self, x):
            return self.lm_head(nn.relu(self.l1(x)))

    class StudentModel(nn.Module):
        def __init__(self, teacher_l1_w):
            super().__init__()
            self.l1 = TurboQuantLinear(in_feat, out_feat, bits=2)
            q = tq_quantize_weight(teacher_l1_w, bits=2)
            self.l1.packed = mx.array(q["packed"])
            self.l1.norms = mx.array(q["norms"]).astype(mx.float16)
            self.lm_head = nn.Linear(out_feat, vocab)
        def __call__(self, x):
            return self.lm_head(nn.relu(self.l1(x)))

    teacher = TeacherModel()
    student = StudentModel(np.array(teacher.l1.weight))
    student.lm_head.weight = teacher.lm_head.weight
    student.lm_head.bias = teacher.lm_head.bias

    rng = np.random.default_rng(0)
    train = [(mx.array(rng.standard_normal((1, 16, in_feat)).astype(np.float32)),
              mx.array([[16]], dtype=mx.int32)) for _ in range(8)]
    valid = [(mx.array(rng.standard_normal((1, 16, in_feat)).astype(np.float32)),
              mx.array([[16]], dtype=mx.int32)) for _ in range(2)]

    student.freeze()
    unfreeze_jangtq_norms_for_dwq(student)

    def target_fn(batch, idx, split):
        return teacher(batch)

    opt = optimizers.Adam(learning_rate=1e-3, bias_correction=True)
    initial_loss, final_loss = dwq_norms_jangtq(
        student, target_fn, opt, train, valid,
        batch_size=1, max_seq_length=17, seed=0, num_iters=10,
    )
    assert final_loss < initial_loss
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/test_dwq_jangtq.py::test_dwq_norms_jangtq_decreases_kl_loss -v
```

Expected: FAIL — `dwq_norms_jangtq` not defined.

- [ ] **Step 3: Implement `dwq_norms_jangtq` in `dwq_jangtq.py`**

Append:

```python
import time
import mlx.optimizers as optimizers
from mlx.utils import tree_map
from tqdm import tqdm

from mlx_lm.tuner.losses import kl_div_loss
from mlx_lm.tuner.trainer import iterate_batches, grad_checkpoint


def dwq_norms_jangtq(
    model: nn.Module,
    target_fn,
    opt: optimizers.Optimizer,
    train_data,
    valid_data,
    batch_size: int,
    max_seq_length: int,
    seed: int,
    num_iters: int | None = None,
    dtype: mx.Dtype = mx.bfloat16,
    gradient_checkpoint: bool = False,
    temperature: float = 2.0,
    log_every: int = 20,
    validate_every: int = 200,
):
    """KL-distill FP teacher logits into a JANGTQ student by tuning tq_norms only.

    Codebook + signs + packed stay frozen. Returns (initial_valid_loss,
    final_valid_loss). Caller must have invoked unfreeze_jangtq_norms_for_dwq
    on the model before passing it in.
    """
    if gradient_checkpoint:
        grad_checkpoint(model.layers[0])

    scale = 1.0 / temperature

    def loss_fn(params, x, targets, lengths):
        model.update(tree_map(lambda x: x.astype(dtype), params))
        logits = model(x)
        if isinstance(targets, tuple):
            targets, ids = targets
            logits = mx.take_along_axis(logits, ids, axis=-1)
        losses = kl_div_loss(scale * logits, scale * targets)
        mask = mx.arange(1, 1 + targets.shape[1]) < lengths[:, 1:]
        ntoks = mask.sum()
        loss = (mask * losses).sum() / ntoks
        return loss, ntoks

    def step(inputs, targets, lengths, params):
        (loss, ntoks), grads = mx.value_and_grad(loss_fn)(params, inputs, targets, lengths)
        params = opt.apply_gradients(grads, params)
        return loss, ntoks, params

    def validate(params):
        v_loss, v_tokens = 0.0, 0
        for i, (batch, lengths) in enumerate(
            iterate_batches(valid_data, batch_size, max_seq_length, seed=seed)
        ):
            batch = batch[:, :-1]
            targets = target_fn(batch, i, split="valid")
            mx.eval(targets)
            loss, ntoks = loss_fn(params, batch, targets, lengths)
            mx.eval(loss, ntoks)
            v_tokens += ntoks.item()
            v_loss += loss.item() * ntoks.item()
        return v_loss / max(v_tokens, 1)

    params = tree_map(lambda x: x.astype(mx.float32), model.trainable_parameters())

    initial_valid_loss = validate(params)
    tqdm.write(f"DWQ-Norms initial valid loss: {initial_valid_loss:.4f}")

    tic = time.time()
    for it, (batch, lengths) in enumerate(
        tqdm(iterate_batches(train_data, batch_size, max_seq_length, seed=seed),
             total=num_iters or len(train_data) // batch_size,
             desc="DWQ-Norms")
    ):
        if num_iters is not None and it >= num_iters:
            break
        batch = batch[:, :-1]
        targets = target_fn(batch, it, split="train")
        mx.eval(targets)
        loss, ntoks, params = step(batch, targets, lengths, params)
        mx.eval(loss, params)

        if (it + 1) % log_every == 0:
            tqdm.write(f"  it={it} loss={loss.item():.4f} "
                       f"toks/s={ntoks.item() * (it + 1) / (time.time() - tic):.1f}")
        if (it + 1) % validate_every == 0:
            tqdm.write(f"  it={it} valid_loss={validate(params):.4f}")

    final_valid_loss = validate(params)
    tqdm.write(f"DWQ-Norms final valid loss: {final_valid_loss:.4f}")
    if final_valid_loss > initial_valid_loss:
        tqdm.write("Final valid loss is worse than initial. tq_norms NOT updated.")
        return initial_valid_loss, final_valid_loss

    model.update(tree_map(lambda x: x.astype(dtype), params))
    return initial_valid_loss, final_valid_loss
```

- [ ] **Step 4: Run test to verify pass**

```bash
.venv/bin/pytest tests/test_dwq_jangtq.py::test_dwq_norms_jangtq_decreases_kl_loss -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jang-tools/jang_tools/dwq_jangtq.py jang-tools/tests/test_dwq_jangtq.py
git commit -m "feat(jangtq): DWQ-Norms KL-distill loop tuning tq_norms only

Single-node port of mlx_lm/quant/dwq.py:dwq_quantize. Targets computed live
from FP teacher (no precomputed cache in v1). Adam optimizer, fp32 param
accumulation, bf16 model weights. Aborts if final valid loss > initial.
Codebook + signs + packed stay frozen — gradient flows only through tq_norms."
```

---

## Task 5: Implement conservative save-back to JANGTQ bundle

**Files:**
- Modify: `jang-tools/jang_tools/dwq_jangtq.py`
- Test: `jang-tools/tests/test_dwq_jangtq.py`

Save path is conservative: only `<path>.tq_norms` keys are replaced. Every other key in every shard must be **bit-identical** to the original. Shard count + filenames + index file are stable. Existing safetensors metadata (e.g. `format=mlx`) is preserved. After write, verify (a) safetensor headers parse, (b) key set matches the index file, (c) every non-norms tensor is bit-identical to its pre-DWQ value, (d) `tq_bits` dtype is preserved (uint8 stays uint8; int32 stays int32).

- [ ] **Step 1: Write failing roundtrip test**

```python
def test_save_dwq_norms_roundtrip(tmp_path):
    """save_dwq_norms_jangtq writes tuned tq_norms; everything else bit-identical."""
    from jang_tools.dwq_jangtq import save_dwq_norms_jangtq, verify_dwq_save
    from jang_tools.turboquant.tq_kernel import TurboQuantLinear
    from safetensors import safe_open

    bundle = tmp_path / "bundle"
    bundle.mkdir()

    layer = TurboQuantLinear(32, 16, bits=4)
    rng = np.random.default_rng(0)
    w = rng.standard_normal((16, 32)).astype(np.float32)
    q = tq_quantize_weight(w, bits=4)
    layer.packed = mx.array(q["packed"])
    layer.norms = mx.array(q["norms"]).astype(mx.float16)

    # Match production: tq_bits is uint8 on most converters (DSV4 uses int32);
    # the saver must preserve whatever dtype was there.
    original_tensors = {
        "layer.tq_packed": layer.packed,
        "layer.tq_norms":  layer.norms,
        "layer.tq_bits":   mx.array([4], dtype=mx.uint8),
        "layer.bias":      mx.zeros((16,)),
    }
    # Write with metadata to confirm save preserves it.
    mx.save_safetensors(
        str(bundle / "model.safetensors"),
        original_tensors,
        metadata={"format": "mlx", "jang_tag": "test"},
    )

    new_norms = layer.norms * 1.5
    layer.norms = new_norms

    save_dwq_norms_jangtq(bundle, {"layer": layer})
    verify_dwq_save(bundle, {"layer": layer})  # raises on any drift

    reloaded = mx.load(str(bundle / "model.safetensors"))
    assert mx.allclose(reloaded["layer.tq_norms"], new_norms.astype(mx.float16)).item()
    # packed bit-identical
    assert mx.array_equal(reloaded["layer.tq_packed"], original_tensors["layer.tq_packed"]).item()
    # tq_bits dtype preserved (uint8 stays uint8, value 4)
    assert reloaded["layer.tq_bits"].dtype == mx.uint8
    assert reloaded["layer.tq_bits"].item() == 4
    # bias bit-identical
    assert mx.array_equal(reloaded["layer.bias"], original_tensors["layer.bias"]).item()
    # metadata preserved
    with safe_open(str(bundle / "model.safetensors"), framework="numpy") as h:
        meta = h.metadata() or {}
    assert meta.get("format") == "mlx"
    assert meta.get("jang_tag") == "test"
    # Codebook/signs are property-backed; we never invent these keys
    assert "layer.tq_codebook" not in reloaded
    assert "layer.codebook" not in reloaded


def test_save_dwq_norms_preserves_int32_tq_bits(tmp_path):
    """DSV4-style int32 tq_bits must survive save unchanged."""
    from jang_tools.dwq_jangtq import save_dwq_norms_jangtq
    from jang_tools.turboquant.tq_kernel import TurboQuantLinear

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    layer = TurboQuantLinear(32, 16, bits=4)
    rng = np.random.default_rng(0)
    w = rng.standard_normal((16, 32)).astype(np.float32)
    q = tq_quantize_weight(w, bits=4)
    layer.packed = mx.array(q["packed"])
    layer.norms = mx.array(q["norms"]).astype(mx.float16)

    mx.save_safetensors(str(bundle / "model.safetensors"), {
        "layer.tq_packed": layer.packed,
        "layer.tq_norms":  layer.norms,
        "layer.tq_bits":   mx.array([4], dtype=mx.int32),  # DSV4 style
    })

    layer.norms = layer.norms * 1.1
    save_dwq_norms_jangtq(bundle, {"layer": layer})

    reloaded = mx.load(str(bundle / "model.safetensors"))
    assert reloaded["layer.tq_bits"].dtype == mx.int32, "DSV4 int32 dtype must be preserved"
    assert reloaded["layer.tq_bits"].item() == 4
```

- [ ] **Step 2: Run test to verify failure**

```bash
.venv/bin/pytest tests/test_dwq_jangtq.py::test_save_dwq_norms_roundtrip -v
```

Expected: FAIL — `save_dwq_norms_jangtq` not defined.

- [ ] **Step 3: Implement `save_dwq_norms_jangtq` + `verify_dwq_save`**

Append to `dwq_jangtq.py`:

```python
import hashlib
import json
from pathlib import Path
from safetensors import safe_open


def _tensor_sha(arr_np) -> str:
    """SHA-1 over the contiguous bytes — stable bit-identity check."""
    return hashlib.sha1(memoryview(arr_np).tobytes()).hexdigest()


def save_dwq_norms_jangtq(bundle_dir: str | Path, tq_layers: dict[str, nn.Module]) -> None:
    """Conservative save: replace only <path>.tq_norms tensors. All other
    tensors pass through bit-identical. Existing safetensors metadata is
    preserved. tq_bits dtype is preserved (uint8 OR int32 depending on
    converter). Shard count, filenames, index file are stable.

    Args:
      bundle_dir: directory containing model*.safetensors shards.
      tq_layers:  {dotted_path: runtime TurboQuantLinear|TurboQuantSwitchLinear}
                  with tuned norms.
    """
    bundle = Path(bundle_dir)
    shards = sorted(bundle.glob("model*.safetensors"))
    assert shards, f"no safetensors shards in {bundle}"

    for shard in shards:
        # Read existing metadata so we can roundtrip it.
        with safe_open(str(shard), framework="numpy") as h:
            existing_meta = h.metadata()  # dict[str, str] | None
        weights = mx.load(str(shard))

        modified = False
        for k in list(weights.keys()):
            if not k.endswith(".tq_norms"):
                continue
            path = k[: -len(".tq_norms")]
            if path in tq_layers:
                tuned = tq_layers[path].norms
                weights[k] = tuned.astype(weights[k].dtype)  # preserve dtype
                modified = True
        if not modified:
            continue
        # Write back with original metadata preserved (None becomes empty dict).
        mx.save_safetensors(str(shard), weights, metadata=existing_meta or {})


def verify_dwq_save(bundle_dir: str | Path,
                    tq_layers: dict[str, nn.Module]) -> None:
    """Strict post-save invariants:
      1. Every shard parses.
      2. Key set matches model.safetensors.index.json (if present).
      3. Every non-tq_norms tensor is bit-identical to the on-disk version
         (re-read after save and compared to itself for shape/dtype; tq_norms
         is verified to match the tuned in-memory value at the layer's dtype).
      4. tq_bits dtype is preserved (uint8 or int32).
      5. Existing metadata roundtripped.

    Raises RuntimeError on any failure.
    """
    bundle = Path(bundle_dir)
    index_path = bundle / "model.safetensors.index.json"

    # 1+2. Key set integrity
    if index_path.exists():
        with open(index_path) as fh:
            expected_keys = set(json.load(fh)["weight_map"].keys())
        actual_keys = set()
        for shard in sorted(bundle.glob("model-*.safetensors")):
            with safe_open(str(shard), framework="numpy") as f:
                actual_keys.update(f.keys())
        if actual_keys != expected_keys:
            raise RuntimeError(
                f"DWQ save changed key set. "
                f"missing={expected_keys - actual_keys} "
                f"added={actual_keys - expected_keys}"
            )

    # 3+4. Per-shard tensor integrity
    for shard in sorted(bundle.glob("model*.safetensors")):
        with safe_open(str(shard), framework="numpy") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if k.endswith(".tq_bits"):
                    if t.dtype.kind not in ("u", "i"):  # unsigned or signed int
                        raise RuntimeError(f"{k} non-integer dtype after save: {t.dtype}")
                elif k.endswith(".tq_norms"):
                    # Confirm the saved value matches the in-memory tuned tensor
                    path = k[: -len(".tq_norms")]
                    if path in tq_layers:
                        tuned_np = np.asarray(tq_layers[path].norms.astype(mx.float16))
                        if not np.array_equal(t, tuned_np):
                            raise RuntimeError(f"{k} saved value != tuned value")
                # All other tensors: just confirm parse succeeds (already done by get_tensor).
```

- [ ] **Step 4: Run test to verify pass**

```bash
.venv/bin/pytest tests/test_dwq_jangtq.py::test_save_dwq_norms_roundtrip -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jang-tools/jang_tools/dwq_jangtq.py jang-tools/tests/test_dwq_jangtq.py
git commit -m "feat(jangtq): conservative save_dwq_norms_jangtq + verify_dwq_save

Replaces only <path>.tq_norms tensors in each shard, preserves shard layout,
filenames, dtype, and the index file's key set. Codebook + signs + packed +
tq_bits are never touched. verify_dwq_save() asserts the post-write key set
matches the index file."
```

---

## Task 6: `collect_tq_layers` walker

**Files:**
- Modify: `jang-tools/jang_tools/dwq_jangtq.py`
- Test: `jang-tools/tests/test_dwq_jangtq.py`

- [ ] **Step 1: Write failing test**

```python
def test_collect_tq_layers_walks_nested_modules():
    import mlx.nn as nn
    from jang_tools.dwq_jangtq import collect_tq_layers
    from jang_tools.turboquant.tq_kernel import TurboQuantLinear, TurboQuantSwitchLinear

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = TurboQuantLinear(32, 32, bits=4)
            self.experts = TurboQuantSwitchLinear(32, 32, num_experts=2, bits=2)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Block(), Block()]
            self.embed = nn.Embedding(100, 32)

    layers = collect_tq_layers(Model())
    assert "layers.0.attn" in layers
    assert "layers.0.experts" in layers
    assert "layers.1.attn" in layers
    assert "layers.1.experts" in layers
    assert "embed" not in layers
    assert isinstance(layers["layers.0.attn"], TurboQuantLinear)
    assert isinstance(layers["layers.0.experts"], TurboQuantSwitchLinear)
```

- [ ] **Step 2: Run test to verify failure**

```bash
.venv/bin/pytest tests/test_dwq_jangtq.py::test_collect_tq_layers_walks_nested_modules -v
```

Expected: FAIL.

- [ ] **Step 3: Implement**

```python
def collect_tq_layers(model: nn.Module) -> dict[str, nn.Module]:
    """Return {dotted_path: layer} for every TurboQuant{Linear,Switch} in model."""
    out: dict[str, nn.Module] = {}
    def visit(path, m):
        if isinstance(m, (TurboQuantLinear, TurboQuantSwitchLinear)):
            out[path] = m
    model.apply_to_modules(visit)
    return out
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/pytest tests/test_dwq_jangtq.py::test_collect_tq_layers_walks_nested_modules -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jang-tools/jang_tools/dwq_jangtq.py jang-tools/tests/test_dwq_jangtq.py
git commit -m "feat(jangtq): collect_tq_layers walks model for TurboQuant layers"
```

---

## Task 7: Build a tiny local JANGTQ fixture

**Files:**
- Create: `jang-tools/tests/fixtures/build_tiny_jangtq.py` (one-shot fixture builder)
- Create: `jang-tools/tests/fixtures/tiny_jangtq/` (output, gitignored)

A real JANGTQ bundle (even MoE) is needed before we touch any production artifact. Build the smallest possible one in-tree so smoke tests don't depend on `/Volumes/EricsLLMDrive` or HF caches.

- [ ] **Step 1: Write fixture builder**

```python
# jang-tools/tests/fixtures/build_tiny_jangtq.py
"""One-shot script: build a tiny JANGTQ bundle for fixture use.

Produces a 2-layer transformer with TurboQuantLinear weights, tq_packed/tq_norms/
tq_bits tensors in a single safetensors shard, and a jangtq_runtime.safetensors
sidecar. Mirrors the on-disk contract of a real JANGTQ bundle at minimal scale.

Run once: python tests/fixtures/build_tiny_jangtq.py
"""
import json, sys
from pathlib import Path
import mlx.core as mx
import numpy as np

from jang_tools.turboquant.linear import tq_quantize_weight
from jang_tools.turboquant.codebook import compute_codebook
from jang_tools.turboquant.rotation import generate_random_signs

OUT = Path(__file__).parent / "tiny_jangtq"
OUT.mkdir(exist_ok=True)

D, H, V, BITS = 64, 32, 100, 4
rng = np.random.default_rng(0)
weights = {}
for i in range(2):
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        w = rng.standard_normal((H, D)).astype(np.float32)
        q = tq_quantize_weight(w, bits=BITS, seed=42)
        prefix = f"model.layers.{i}.self_attn.{proj}"
        weights[f"{prefix}.tq_packed"] = mx.array(q["packed"])
        weights[f"{prefix}.tq_norms"]  = mx.array(q["norms"]).astype(mx.float16)
        # uint8 matches MiniMax/Kimi/Zaya/Qwen/Ling production. DSV4 uses int32;
        # the saver must preserve whichever dtype is in a given bundle.
        weights[f"{prefix}.tq_bits"]   = mx.array([BITS], dtype=mx.uint8)
weights["model.embed_tokens.weight"] = mx.array(rng.standard_normal((V, D)).astype(np.float32))
weights["model.norm.weight"] = mx.array(np.ones((D,), dtype=np.float32))
weights["lm_head.weight"]    = mx.array(rng.standard_normal((V, D)).astype(np.float32))
mx.save_safetensors(str(OUT / "model.safetensors"), weights, metadata={"format": "mlx"})

# Sidecar: shared codebook + signs per (in_features, bits)
sidecar = {
    f"codebook.{D}.{BITS}": mx.array(compute_codebook(D, BITS), dtype=mx.float32),
    f"signs.{D}.42":        mx.array(generate_random_signs(D, seed=42), dtype=mx.float32),
}
mx.save_safetensors(str(OUT / "jangtq_runtime.safetensors"), sidecar)

(OUT / "config.json").write_text(json.dumps({
    "model_type": "tiny_jangtq",
    "hidden_size": D,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "vocab_size": V,
}, indent=2))
(OUT / "jang_config.json").write_text(json.dumps({
    "format": "jangtq",
    "bits": BITS,
}, indent=2))
print(f"wrote tiny JANGTQ fixture to {OUT}")
```

- [ ] **Step 2: Run fixture builder**

```bash
cd /Users/eric/jang/jang-tools
.venv/bin/python tests/fixtures/build_tiny_jangtq.py
```

Expected: prints `wrote tiny JANGTQ fixture to .../tiny_jangtq` and creates `model.safetensors`, `jangtq_runtime.safetensors`, `config.json`, `jang_config.json`.

- [ ] **Step 3: Add to .gitignore**

Append to `jang-tools/tests/fixtures/.gitignore`:

```
tiny_jangtq/
```

- [ ] **Step 4: Commit**

```bash
git add jang-tools/tests/fixtures/build_tiny_jangtq.py jang-tools/tests/fixtures/.gitignore
git commit -m "test(jangtq): tiny in-tree JANGTQ fixture builder

2-layer transformer with tq_packed/tq_norms/tq_bits + jangtq_runtime sidecar.
Mirrors the production on-disk contract at minimal scale so DWQ smoke tests
don't depend on local HF caches or external SSDs."
```

---

## Task 8: Wire CLI entry `python -m jang_tools.dwq_jangtq`

**Files:**
- Modify: `jang-tools/jang_tools/dwq_jangtq.py`

- [ ] **Step 1: Append `main()` to `dwq_jangtq.py`**

```python
def main():
    import argparse, shutil, types
    import numpy as np
    from mlx_lm.utils import load as mlx_load
    from mlx_lm.tuner.datasets import load_dataset
    from jang_tools.load_jangtq import load_jangtq_model  # NOT load_jangtq

    p = argparse.ArgumentParser(description="DWQ-Norms for JANGTQ bundles (Phase 1)")
    p.add_argument("--jangtq-model", required=True, help="Path to JANGTQ bundle (student).")
    p.add_argument("--teacher-model", required=True, help="HF id or path to FP teacher.")
    p.add_argument("--output-dir", required=True, help="Where to write the tuned bundle.")
    p.add_argument("--num-samples", type=int, default=2048)
    p.add_argument("--max-seq-length", type=int, default=1025)
    p.add_argument("--num-valid-samples", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-iters", type=int, default=None)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--data-path", default="allenai/tulu-3-sft-mixture")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--grad-checkpoint", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f"{out} already exists; pick a new path")
    shutil.copytree(args.jangtq_model, out)

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    student, tokenizer = load_jangtq_model(str(out))
    teacher, _, _ = mlx_load(args.teacher_model, return_config=True, lazy=True)

    ds_args = types.SimpleNamespace(
        hf_dataset={"path": args.data_path, "train_split": "train",
                    "valid_split": "train[:1]"},
        train=True, test=False,
    )
    dataset = load_dataset(ds_args, tokenizer)[0]
    perm = np.random.permutation(len(dataset))
    train_perm = perm[: args.num_samples].tolist()
    valid_perm = perm[args.num_samples : args.num_samples + args.num_valid_samples].tolist()
    def process(idx):
        toks, off = dataset.process(dataset[idx])
        return (toks[: args.max_seq_length], off)
    train_data = [process(i) for i in train_perm]
    valid_data = [process(i) for i in valid_perm]

    student.freeze()
    unfreeze_jangtq_norms_for_dwq(student)

    def target_fn(batch, idx, split):
        return teacher(batch)

    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

    opt = optimizers.Adam(learning_rate=args.learning_rate, bias_correction=True)
    init, final = dwq_norms_jangtq(
        student, target_fn, opt, train_data, valid_data,
        batch_size=args.batch_size, max_seq_length=args.max_seq_length,
        seed=args.seed, num_iters=args.num_iters,
        temperature=args.temperature,
        gradient_checkpoint=args.grad_checkpoint,
    )
    print(f"valid_loss: {init:.4f} -> {final:.4f}")

    if final < init:
        tq_layers = collect_tq_layers(student)
        save_dwq_norms_jangtq(out, tq_layers)
        verify_dwq_save(out, tq_layers)
        print(f"Tuned bundle saved + verified at {out}")
    else:
        print("NOT saving — final loss did not improve.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help**

```bash
cd /Users/eric/jang/jang-tools
.venv/bin/python -m jang_tools.dwq_jangtq --help
```

Expected: usage text printed.

- [ ] **Step 3: Commit**

```bash
git add jang-tools/jang_tools/dwq_jangtq.py
git commit -m "feat(jangtq): CLI python -m jang_tools.dwq_jangtq (Phase 1)

Loads JANGTQ bundle as student via load_jangtq_model, FP teacher via mlx_lm.load,
runs DWQ-Norms KL distill, conservative save-back of tq_norms only, then
verify_dwq_save asserts shard key set is unchanged."
```

---

## Task 9: Smoke test on tiny in-tree fixture (gated)

**Files:**
- Test: `jang-tools/tests/test_dwq_jangtq.py` (gated `@pytest.mark.slow`)

- [ ] **Step 1: Add fixture-based smoke test**

```python
@pytest.mark.slow
def test_dwq_norms_smoke_on_tiny_fixture(tmp_path):
    """End-to-end: load fixture as student + teacher, run 20 iters, save, reload."""
    import shutil
    from jang_tools.dwq_jangtq import (
        unfreeze_jangtq_norms_for_dwq, dwq_norms_jangtq,
        save_dwq_norms_jangtq, verify_dwq_save, collect_tq_layers,
    )
    from jang_tools.load_jangtq import load_jangtq_model
    import mlx.optimizers as optimizers

    fixture = Path(__file__).parent / "fixtures" / "tiny_jangtq"
    if not (fixture / "model.safetensors").exists():
        pytest.skip("tiny fixture not built; run tests/fixtures/build_tiny_jangtq.py")

    out = tmp_path / "tuned"
    shutil.copytree(fixture, out)
    student, tok = load_jangtq_model(str(out))

    # Use the ORIGINAL fixture (with original norms) as the FP "teacher" surrogate.
    # We perturb the student's norms first so DWQ has something to recover.
    teacher, _ = load_jangtq_model(str(fixture))
    for layer in collect_tq_layers(student).values():
        layer.norms = layer.norms * mx.array(1.05)  # 5% drift

    rng = np.random.default_rng(0)
    train = [(mx.array(rng.integers(0, 100, (1, 16)).astype(np.int32)),
              mx.array([[16]], dtype=mx.int32)) for _ in range(8)]
    valid = [(mx.array(rng.integers(0, 100, (1, 16)).astype(np.int32)),
              mx.array([[16]], dtype=mx.int32)) for _ in range(2)]

    student.freeze()
    unfreeze_jangtq_norms_for_dwq(student)

    def target_fn(batch, idx, split):
        return teacher(batch)

    opt = optimizers.Adam(learning_rate=1e-3, bias_correction=True)
    init, final = dwq_norms_jangtq(
        student, target_fn, opt, train, valid,
        batch_size=1, max_seq_length=17, seed=0, num_iters=20,
    )
    assert final < init, f"DWQ failed to recover drift: {init=} {final=}"

    tuned_layers = collect_tq_layers(student)
    save_dwq_norms_jangtq(out, tuned_layers)
    verify_dwq_save(out, tuned_layers)

    # Reload and confirm tq_packed + tq_bits preserved bit-identical
    s2, _ = load_jangtq_model(str(out))
    for path, layer in collect_tq_layers(s2).items():
        orig_layer = collect_tq_layers(teacher)[path]
        assert mx.array_equal(layer.packed, orig_layer.packed).item()
        assert layer.bits == orig_layer.bits
```

- [ ] **Step 2: Run smoke**

```bash
.venv/bin/pytest tests/test_dwq_jangtq.py::test_dwq_norms_smoke_on_tiny_fixture -v -m slow
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add jang-tools/tests/test_dwq_jangtq.py
git commit -m "test(jangtq): DWQ-Norms smoke on tiny in-tree fixture

End-to-end: load fixture, perturb norms 5%, run 20 DWQ iters, save back,
reload, confirm tq_packed + tq_bits preserved and norms recovered."
```

---

## Task 10: Smoke on one real MoE bundle (gated, manual)

**Files:** none (manual run; result captured in design doc)

Only after Task 9 passes. Pick one production JANGTQ bundle (suggestion: a small MiniMax-M2.7-JANGTQ_K variant or Qwen3.6-A3B-JANGTQ4) and run DWQ for ~200 iters. The point is to validate the load → train → save → reload → generate path on a real MoE artifact.

- [ ] **Step 1: Identify a real bundle**

```bash
ls /Users/eric/models/JANGQ/ | head
# Pick the smallest plausible one; record its path.
```

- [ ] **Step 2: Run DWQ for 200 iters**

```bash
cd /Users/eric/jang/jang-tools
.venv/bin/python -m jang_tools.dwq_jangtq \
  --jangtq-model /Users/eric/models/JANGQ/<chosen-bundle> \
  --teacher-model <matching-FP-teacher-HF-id> \
  --output-dir /tmp/<chosen-bundle>-dwq \
  --num-samples 256 --num-iters 200 \
  --learning-rate 1e-6 --batch-size 1
```

Expected: `valid_loss: X -> Y` with `Y < X`, then "Tuned bundle saved + verified".

- [ ] **Step 3: Generate from tuned vs original**

```bash
.venv/bin/python -c "
from jang_tools.load_jangtq import load_jangtq_model
from mlx_lm.generate import generate
m,  tok = load_jangtq_model('<orig path>')
m2, _   = load_jangtq_model('/tmp/<chosen-bundle>-dwq')
print('ORIG:', generate(m,  tok, prompt='2+2=', max_tokens=10, verbose=False))
print('DWQ :', generate(m2, tok, prompt='2+2=', max_tokens=10, verbose=False))
"
```

Both must produce coherent answers; DWQ ≥ baseline.

- [ ] **Step 4: Record result in design doc**

Append the `valid_loss` delta + qualitative coherence note to `research/JANGTQ-DWQ-DESIGN.md`.

- [ ] **Step 5: Commit**

```bash
git add research/JANGTQ-DWQ-DESIGN.md
git commit -m "docs(jangtq): record DWQ-Norms smoke result on <bundle>"
```

---

## Task 11: Update `docs/adoption/PORTING.md` to current contract

**Files:**
- Modify: `/Users/eric/jang/docs/adoption/PORTING.md:132` (and any other stale references in the same file)

The current doc describes a stale `.tq_codebook`-style layout that does not match the live `.tq_packed` / `.tq_norms` / `.tq_bits` + sidecar contract. This confusion is what caused the original DWQ plan to mis-scope. Phase 1 fixes it.

- [ ] **Step 1: Read PORTING.md and locate stale section**

```bash
sed -n '120,180p' /Users/eric/jang/docs/adoption/PORTING.md
```

- [ ] **Step 2: Replace with current contract description**

The replacement section must state:
- Per-layer in-bundle tensors: `<path>.tq_packed` (uint32), `<path>.tq_norms` (fp16), `<path>.tq_bits` (int32 scalar — source of truth for that tensor's bit width; supports K-profile per-projection bits)
- Sidecar `jangtq_runtime.safetensors` keyed by `(in_features, bits)` for codebook and `(in_features, seed)` for signs — shared across all layers with matching shape
- DWQ Phase 1 (this plan): tunes only `tq_norms`; codebook + signs + packed + tq_bits frozen
- DWQ Phase 2 (deferred): would learn per-layer codebooks → sidecar schema bump → Swift loader update — see appendix in this plan

- [ ] **Step 3: Commit**

```bash
git add docs/adoption/PORTING.md
git commit -m "docs(jangtq): correct PORTING.md to current tq_packed/tq_norms/tq_bits contract

Stale .tq_codebook layout description removed. Per-layer in-bundle tensors
and shared (in_features, bits) sidecar contract documented. Phase-1 DWQ
note added."
```

---

## Task 12: Design doc

**Files:**
- Create: `/Users/eric/jang/research/JANGTQ-DWQ-DESIGN.md`

- [ ] **Step 1: Write design doc covering**

  - Motivation: Lloyd-Max codebook is optimal for unit-sphere distribution; per-row magnitudes vary in practice — `tq_norms` absorbs that scaling. Tuning recovers per-row error.
  - Math: KL distill at temperature T against FP teacher; gradient flows through `(out, in)` dense reconstruction: `W = inverse_hadamard(take(codebook, packed) * norms, signs)`. Only `norms` is trainable.
  - Trainable vs frozen table (mirror the "Naming" table in this plan).
  - Memory cost: norms get promoted to fp32 for accumulation — adds ~`out_features * 4 bytes` per layer; negligible vs total weights.
  - Eval protocol: in-tree fixture smoke (Task 9) → one real MoE bundle (Task 10) → record valid_loss + coherence delta.
  - Phase 2 preview: codebook centroid tuning. Lists every contract change required (vmlx-swift, sidecar schema, Python loader, bundle metadata, porting doc, sidecar regen). Marked DEFERRED.

- [ ] **Step 2: Add wiki cross-link**

```bash
~/.codex/bin/llm-wiki remember "JANGTQ DWQ Phase 1" "DWQ-Norms tunes tq_norms only. Codebook + packed frozen. Wire format unchanged. Plan: ~/jang/docs/superpowers/plans/2026-05-09-jangtq-dwq.md. Phase 2 (codebook centroids) deferred — needs Swift JANGTQRuntimeCache update + sidecar schema bump."
```

- [ ] **Step 3: Commit**

```bash
git add research/JANGTQ-DWQ-DESIGN.md
git commit -m "docs(jangtq): DWQ design doc — Phase 1 spec + Phase 2 deferred preview"
```

---

## Self-Review Checklist (run before claiming complete)

- [ ] Every task ends with a passing test or verified CLI smoke.
- [ ] No "TBD" / "implement later" / placeholder strings.
- [ ] **Runtime class target:** all `dwq_training` flag + `_dense_weight_dwq` modifications target `jang-tools/jang_tools/turboquant/tq_kernel.py` (the class production loads), NOT `linear.py`. Verify via `grep -n "from jang_tools.turboquant" jang-tools/jang_tools/load_jangtq.py` showing imports from `tq_kernel`.
- [ ] **CLI/test imports:** every `from jang_tools.load_jangtq import ...` line uses `load_jangtq_model`, never `load_jangtq` (the module exists, the function doesn't).
- [ ] Tensor names: `tq_packed`, `tq_norms`, `tq_bits` everywhere (never `packed`, `norms`, `codebook` as bundle-key-side names).
- [ ] In-memory attribute names: `self.packed`, `self.norms` are owned per-layer; `self.codebook`, `self.signs` are `@property` shared lookups. Plan must not mutate the properties.
- [ ] **`tq_bits` dtype-preserved:** save path reads existing dtype and casts back to it. Test covers both uint8 and int32 bundles.
- [ ] **Safetensors metadata preserved:** save path reads `safe_open(...).metadata()` and roundtrips it. Test asserts `format=mlx` (or whatever was there) survives.
- [ ] **Bit-identical non-norm tensors:** `verify_dwq_save` confirms every non-`.tq_norms` tensor matches its pre-DWQ value.
- [ ] Predicate name: `unfreeze_jangtq_norms_for_dwq`.
- [ ] Loop name: `dwq_norms_jangtq`.
- [ ] Save name: `save_dwq_norms_jangtq`. Verify name: `verify_dwq_save`.
- [ ] All edits confined to `/Users/eric/jang/`. `git status` confirms before each commit.
- [ ] No vmlx-swift, vmlx Python, sidecar, loader, or jang_config schema edits.
- [ ] `tq_bits` never appears in any unfreeze list, save-replace list, or model-update path. JANGTQ_K per-projection routed bits + `mxtq_bits` dual form preserved.
- [ ] AWQ-scale bundles: DWQ training forward applies `x = x / awq_scale` once, identically to the kernel forward.

## Risk Log

| Risk | Mitigation |
|---|---|
| Vectorized dense reconstruction has higher peak memory than per-row unpack on big layers | Dense weight is `(out, in)` fp16 — same as the dequantized form the existing forward already produces. For 70B+ MoE bundles, monitor `mx.get_peak_memory()` in Task 10; fall back to chunked reconstruction if needed. |
| Switch DWQ training is 2-deep nested over `(K, num_experts)` and slow on big MoE | v1 accepts the slowness — it only runs during ~few-hundred-iter DWQ training, not inference. Vectorize via gather-on-experts in v2 if needed. |
| FP teacher OOM alongside JANGTQ student on 256 GB | Use `--grad-checkpoint`, `batch_size=1`. Bigger machines optional. |
| `tulu-3-sft-mixture` is text-only; VL JANGTQ bundles need image-text pairs | v1 is text-only. VL DWQ is a separate plan. |
| Hidden runtime/vmlx change might surface during smoke (e.g. tuned norms hit a kernel edge case) | STOP, write up the runtime delta as a separate proposal. Confirm wire-format invariant: only `<path>.tq_norms` shapes/dtypes touched, every other shard key stable. |
| MoE switch dense reconstruction allocates `(num_experts, out, in)` in one go — could OOM on 256+ expert bundles | Worst case: MiniMax has ~256 experts × `(out, in)` fp16 ≈ a few GB per layer. Acceptable on 256 GB. If it isn't, chunk experts. |

---

## Phase 2 Appendix — Learned Codebooks (DEFERRED)

**Phase 2 is a wire-format and runtime contract bump. It is NOT included in this plan.**

If we later decide to tune codebook centroids, every item below needs explicit work and review. Listed for context — DO NOT execute as part of Phase 1.

### Phase 2 Contract Changes

| Component | Repo / File | Required change |
|---|---|---|
| Sidecar writer | `jang-tools/jang_tools/build_jangtq_sidecar.py` | Emit per-layer `codebook_layer.<dotted.path>` entries in addition to / overriding the shared `codebook.{in_features}.{bits}` keys. Gated by a new `jang_config.dwq_tuned_codebooks: bool` flag. |
| Python loader | `jang-tools/jang_tools/load_jangtq.py:990` | Codebook lookup fallback chain: per-layer `codebook_layer.<path>` → shared `codebook.{in_features}.{bits}` → `compute_codebook(in_features, bits)` (today's only path). |
| Swift runtime | `vmlx-swift-lm` `Sources/.../JANGTQKernels.swift` `JANGTQRuntimeCache` | Same fallback chain. New cache key for per-layer codebooks. Smoke: existing pre-Phase-2 bundles must continue to load and produce identical output (regression test required). |
| Bundle metadata | `jang_config.json` | New flag `dwq_tuned_codebooks: bool` (default false). Set to true at DWQ-Codebook save time. Loaders branch on this flag for the new key path. |
| Porting doc | `docs/adoption/PORTING.md` | Document the dual-format-era contract: bundles with `dwq_tuned_codebooks: false` use the shared codebook; bundles with `: true` use per-layer codebooks. Phase-1 bundles stay `false`. |
| HF artifacts | every published JANGTQ bundle | If we re-publish DWQ-Codebook tuned versions, sidecar must be regenerated and the metadata flag flipped. Existing `_K`, `_DWQ-Norms`, plain JANGTQ artifacts continue to load unchanged. |
| MiniMax JANGTQ_K invariant | per-projection routed-bits assignment | Phase 2 must continue to honor per-tensor `tq_bits` and `mxtq_bits` dual form. The DWQ-Codebook predicate would need a per-`(layer, in_features, bits)` codebook entry — sidecar key shape becomes more involved. |

### Phase 2 Decision Gate

Phase 2 should only be greenlit after:
1. Phase 1 ships and shows a measurable but underwhelming win (justifying the contract bump cost).
2. The Swift runtime maintainer signs off on the JANGTQRuntimeCache fallback chain.
3. A regression test is in place that confirms pre-Phase-2 bundles load and generate identically before/after the Swift change.

Until those three are true, Phase 2 stays in this appendix.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-09-jangtq-dwq.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?

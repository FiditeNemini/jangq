# Swift JANGTQ runtime extended to GLM 5.1 + Qwen 3.6 (2026-04-16)

**Status:** Two new model variants land in the vMLX-swift JANGTQ runtime —
`Qwen35JANGTQModel` (covers Qwen 3.5 + Qwen 3.6 MoE family,
`model_type: qwen3_5_moe`) and `GLM4MoEJANGTQModel` (covers GLM-4 MoE +
GLM 5.1, `model_type: glm4_moe`). Both build clean, factory dispatch
auto-routes on `weight_format: "mxtq"`, and the existing JANGTQ sidecar
loader (`jangtq_runtime.safetensors` → `JANGTQRuntimeCache`) handles
codebook + Hadamard signs without changes. **No runtime validation has
run yet** — the converted JANGTQ models for these architectures don't
exist on disk and need to be produced via the Python converter on the
M3 Ultra first.

## What landed this session

### `Sources/vMLXLLM/Models/Qwen35JANGTQ.swift` (NEW, ~520 LOC)

JANGTQ variant of Qwen 3.5 / 3.6 MoE family. Mirrors the affine
`Qwen35.swift` structure exactly, with these differences:

- **MoE block (`Qwen35JANGTQSparseMoeBlock`)** — uses `TurboQuantSwitchGLU`
  instead of `SwitchGLU` for the routed-expert projections. Router
  math (softmax + topk + optional norm) and shared-expert path
  (`routed + sigmoid(shared_gate(x)) * shared_expert(x)`) are byte-
  identical to the affine version.
- **Hybrid attention** — reuses internal classes `Qwen35GatedDeltaNet`
  (linear-attn) and `Qwen35Attention` (full-attn) from `Qwen35.swift`
  via the same access path. Layer toggles between them via
  `(layerIdx + 1) % full_attention_interval`. No duplication.
- **Shared expert + dense MLP fallback** — `Qwen3NextMLP` reused from
  `Qwen3Next.swift`. Stays affine.
- **Configuration (`Qwen35JANGTQTextConfiguration`)** — copy of
  `Qwen35TextConfiguration` shape plus three JANGTQ fields:
  `weight_format`, `mxtq_bits`, `mxtq_seed`. Has an `asAffine()`
  helper that round-trips through JSON to produce a
  `Qwen35TextConfiguration` for the internal class initialisers.
- **`Qwen35JANGTQConfiguration`** — top-level wrapper matching the
  affine `Qwen35Configuration` shape: `model_type` at root, optional
  nested `text_config`. Falls back to flat decode if unnested.
- **Sanitize** — three-stage pipeline matching MiniMaxJANGTQ:
  1. Strip MTP heads + apply (1+w) shift on the 5 norm types
     (`input_layernorm`, `post_attention_layernorm`, `model.norm`,
     `q_norm`, `k_norm`) when MTP weights or unsanitized conv1d are
     detected
  2. Drop `.tq_bits` metadata tensors anywhere in the tree (not
     module parameters; they're per-tensor bit-width hints the
     converter writes for sidecar bookkeeping)
  3. Stack per-expert `experts.{E}.{w1,w2,w3}.{tq_packed,tq_norms}`
     into 3D `mlp.switch_mlp.{gate_proj,up_proj,down_proj}.{tq_packed,tq_norms}`
     for `TurboQuantSwitchGLU`'s expected layout

### `Sources/vMLXLLM/Models/GLM4MoEJANGTQ.swift` (NEW, ~370 LOC)

JANGTQ variant of GLM-4 MoE / GLM 5.1. Same pattern as Qwen35JANGTQ:

- **Routed-MoE block (`GLM4MoEJANGTQ`)** — uses `TurboQuantSwitchGLU`
  for routed experts; reuses `GLM4MoEGate` (the noaux_tc grouped
  router with sigmoid OR softmax + `e_score_correction_bias`) and
  `GLM4MoEMLP` (shared experts) from the affine path.
- **Attention** — reuses `GLM4MoEAttention` from `GLM4MOE.swift`.
  This is **standard GQA, not MLA**, so the deepseek_v32 L==1 SDPA
  bf16 absorb-bug fix from `project_mla_absorb_bug` does NOT apply
  here. Documented inline.
- **Dense MLP fallback** — first `first_k_dense_replace` layers stay
  as standard `GLM4MoEMLP` (affine).
- **Configuration (`GLM4MoEJANGTQConfiguration`)** — same shape as
  `GLM4MoEConfiguration` plus the three JANGTQ fields. Has
  `asAffine()` round-trip helper.
- **Sanitize** — strips `.tq_bits` metadata, stacks per-expert
  `experts.{E}.{gate,up,down}_proj.{tq_packed,tq_norms}` into
  `mlp.switch_mlp.*` layout. Skips layers below `firstKDenseReplace`
  (those are dense — no MoE tensors to stack).

### `Sources/vMLXLLM/LLMModelFactory.swift` (modified)

Added two `weight_format` sniffer dispatch closures:

- `qwen3_5_moe`: if root or `text_config.weight_format == "mxtq"` →
  `Qwen35JANGTQModel`, else `Qwen35MoEModel` (affine path).
- `glm4_moe`: if `weight_format == "mxtq"` → `GLM4MoEJANGTQModel`,
  else `GLM4MoEModel` (affine path).

This matches the existing `minimax_m2` JANGTQ dispatch pattern at
`LLMModelFactory.swift:75-87`. The sidecar loader at `Load.swift:33`
already detects `jangtq_runtime.safetensors` for any model and feeds
it into `JANGTQRuntimeCache.shared.loadSidecar` — works for the new
model types automatically.

## What stays affine (in both new models)

Per the JANGTQ design contract:
- All attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- All shared experts (`shared_expert`, `shared_experts.*`)
- All dense MLP layers (Qwen 3.6: layers with no MoE; GLM 5.1: first
  `first_k_dense_replace` layers)
- Routers (`gate.weight` and `e_score_correction_bias`)
- Linear-attn weights in Qwen 3.5/3.6 hybrid path
  (`linear_attn.in_proj_*`, `linear_attn.out_proj`, `conv1d.weight`)
- LM head, embeddings, all RMSNorm gamma weights

Only the routed-expert SwitchGLU projections go through the codebook
JANGTQ kernels. This matches MiniMax M2.7-JANGTQ_2L's quantization
budget exactly.

## Build state

```
$ cd /Users/eric/vmlx/swift && swift build --target vMLXLLM
Build of target: 'vMLXLLM' complete!
$ swift build      # full tree
[182/187] Linking vmlxctl
```

Both new files compile clean. `vmlxctl` rebuilt successfully (121 MB
binary at `.build/arm64-apple-macosx/debug/vmlxctl`). Pre-existing
actor-isolation warnings in `vMLXEngine` are unrelated to this work.

## What is NOT validated

1. **No JANGTQ-converted Qwen 3.6 model on disk**. Per the inventory
   sweep, `/Users/eric/jang/models/` has affine JANG profiles
   (JANG_2S/4K/4S) of Qwen3.5-{4B,9B,35B-A3B} but no JANGTQ variant.
   Need the Python converter to produce one before runtime testing.
   The 35B-A3B-JANGTQ_4M variant per `QWEN36-ANALYSIS.md:208-243`
   would be ~25-30 GB and fits on this MacBook.

2. **No JANGTQ-converted GLM 5.1 on disk**. `GLM-5.1-JANGTQ_1L`
   convert was in flight per `GLM-5.1-RUNTIME-AUDIT.md:282`; status
   unknown. The model is 233 GB JANG_1L per memory
   `project_glm51_jang1l_working` — won't fit on a 128 GB MacBook
   even at 1L.

3. **Python loader (`load_jangtq.py`) NOT updated this session**. The
   user explicitly scoped this session as "Swift JANGTQ scripts" — the
   Python loader generalization (Slice 1 of `JANGTQ-UPDATE-PLAN.md`)
   is a separate stack and was deferred.

4. **Runtime decode never executed**. End-to-end smoke requires a
   converted JANGTQ model + RAM headroom. Both new model files
   compile and instantiate cleanly via the factory dispatch, but no
   forward pass has touched real JANGTQ weights.

## Risk register (will-not-know-until-runtime items)

- **Sanitize key paths**: the per-expert tensor key format
  `model.layers.{L}.mlp.experts.{E}.{w1,w2,w3}.{tq_packed,tq_norms}` is
  inferred from MiniMax's pattern + the affine sanitize. If the
  Python converter for Qwen 3.6 emits keys at a different prefix
  (e.g. `model.language_model.layers.{L}.mlp.experts.gate_up_proj.tq_packed`
  with `gate_up_proj` pre-stacked), the stack loop will silently
  fail to find the probe and the model will load with random
  weights for the routed experts.
- **GLM 5.1 JANGTQ wire format**: the converter side hasn't been
  audited for shared-expert routing per `JANGTQ-UPDATE-PLAN.md:152`.
  If shared experts get accidentally converted to TQ format, the
  affine `GLM4MoEMLP` will fail to bind their parameters.
- **Weight format detection**: the factory sniffer reads
  `weight_format` at the top of `config.json`. If the converter
  writes it to `jang_config.json` instead, the dispatcher falls
  through to the affine path. Confirm via `cat config.json | grep
  weight_format` after conversion.
- **`asAffine()` round-trip**: encodes via JSON to produce a
  config the internal class initialisers accept. Any future field
  added to `Qwen35TextConfiguration` or `GLM4MoEConfiguration` that
  the JANGTQ projection doesn't carry will silently default — fine
  for backwards compat, but worth a unit test if anyone adds new
  required fields.

## Next actionable

In priority order:

1. **Convert Qwen 3.6-35B-A3B-JANGTQ_4M on M3 Ultra** via the Python
   converter. Per `QWEN36-ANALYSIS.md:208-243` build plan. Follow the
   three converter patches (P-PY-1/2/3) in
   `JANGTQ-UPDATE-PLAN.md:97-152` if not yet applied.
2. **rsync to MacBook + smoke test**:
   ```bash
   /Users/eric/vmlx/swift/.build/arm64-apple-macosx/debug/vmlxctl \
     chat -m /Users/eric/models/Qwen3.5-35B-A3B-JANGTQ_4M
   ```
   Expected: factory routes through `Qwen35JANGTQModel`, sidecar
   loader picks up `jangtq_runtime.safetensors`, model decodes coherent
   output. First failure mode is sanitize-key-mismatch (silent: model
   loads but routed experts produce zero output → garbled tokens).
3. **Finish GLM-5.1-JANGTQ_1L conversion** on M3 Ultra. Then test on
   M3 Ultra (won't fit on MacBook).
4. **Speed bench**: target is ~45 tok/s on M3 Ultra (matching MiniMax
   M2.7-JANGTQ_2L's 44 tok/s baseline) for both Qwen 3.6-35B-A3B and
   GLM 5.1.

## Companion DFlash Phase 1 status (parked from prior sessions)

For context — last session's work was on the JANG-DFlash spec-dec
runtime. That's all committed and code-complete on this MacBook:

- 13/13 Phase 1 code tasks done
- 28/28 unit tests passing
- `vmlxctl dflash-smoke` subcommand wired with `--cached` (v2 KV
  cache path), `--max-new-tokens`, `--loop` (interactive stdin mode)
- Python distillation stack ready (`distill_data.py`, `train.py`,
  `convert_to_mlx.py`)
- Final docs at `2026-04-14-jang-dflash-loop-and-python-stack.md`

Five DFlash commits on `jang-spec-plan5-bundle-python-validation`:
- `c22c293`, `2876d54`, `6183b5b`, `807cada`, `6d4afbc`

Both projects are in the same parked state — code complete, runtime
validation pending hardware/RAM availability.

## File map (quick reference)

```
vMLX Swift JANGTQ runtime (in /Users/eric/vmlx/swift/):
  Sources/vMLXLMCommon/JANGTQKernels.swift            # Metal kernels
  Sources/vMLXLMCommon/TurboQuantSwitchLinear.swift   # SwitchGLU shim
  Sources/vMLXLMCommon/Load.swift                     # sidecar loader
  Sources/vMLXLLM/Models/MiniMaxJANGTQ.swift          # MiniMax (existing, ~44 tok/s on M3 Ultra)
  Sources/vMLXLLM/Models/Qwen35JANGTQ.swift           # NEW — Qwen 3.5/3.6
  Sources/vMLXLLM/Models/GLM4MoEJANGTQ.swift          # NEW — GLM 4 / 5.1
  Sources/vMLXLLM/LLMModelFactory.swift               # dispatch (modified)

JANG project (in /Users/eric/jang/):
  jang-tools/jang_tools/load_jangtq.py                # Python loader (NOT updated this session)
  jang-tools/jang_tools/convert.py                    # Python converter
  research/JANGTQ-UPDATE-PLAN.md                      # canonical update plan
  research/QWEN36-ANALYSIS.md                         # Qwen 3.6 architecture deep-dive
  research/GLM-5.1-RUNTIME-AUDIT.md                   # GLM 5.1 + MLA SDPA bug ref
  research/JANGTQ-VMLX-SWIFT-PLAN.md                  # original vMLX-swift wiring plan
```

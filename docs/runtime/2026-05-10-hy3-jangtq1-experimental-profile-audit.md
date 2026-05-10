# Hy3 JANGTQ1 Experimental Profile Audit

Created 2026-05-10.

## Decision

`JANGTQ1` is mechanically valid for the Hy3 converter and Python JANGTQ
runtime path, but it is experimental. Treat it as a size-floor experiment, not
as a production-quality replacement for `JANGTQ2`.

## Quantization Policy

| Tensor family | Policy |
|---|---|
| Routed expert `gate_proj`, `up_proj`, `down_proj` | MXTQ 1-bit |
| Attention q/k/v/o | affine 8-bit, group size 64 |
| Shared expert | affine 8-bit, group size 64 |
| Dense layer-0 FFN | affine 8-bit, group size 64 |
| Embeddings and lm_head | affine 8-bit, group size 64 |
| MTP matmuls | affine 8-bit, `mtp_mode=preserved_disabled` |
| Norms, router gate, expert bias | fp16 passthrough |

## Mechanical Checks

- `tq_quantize_weight` accepts `bits=1`.
- `pack_bits` and all Python Metal kernels derive `vals_per_u32 = 32 // bits`,
  so 1-bit uses 32 packed values per uint32.
- Hy3 routed dimensions align with 1-bit vector packing:
  - gate/up input width: `4096 % 32 == 0`
  - down input width: `1536 % 32 == 0`
- Sidecar generation reads every `.tq_bits` tensor and writes
  `codebook.{in_features}.1` with two entries.
- `JANGTQ_TOPK_OVERRIDE` is separate from this profile. Top-K changes runtime
  expert count; `JANGTQ1` changes routed expert storage precision.

## Size Estimate

Planning estimator on the local source:

| Profile | Estimated bundle | 4K KV cache | Runtime total with 12 GB reserve |
|---|---:|---:|---:|
| `JANGTQ1` | 51.3 GB | 1.34 GB | 64.6 GB |
| `JANGTQ2` | 88.5 GB | 1.34 GB | 101.8 GB |
| `JANGTQ_K` | 113.3 GB | 1.34 GB | 126.6 GB |

The previous rough 44 GB number omitted part of the non-routed 8-bit affine
core and sidecar overhead. Use the measured finished bundle size once the
conversion completes.

## Quality Boundary

1-bit MXTQ has a two-entry codebook after Hadamard rotation. Because Hy3 is a
295B/21B-active MoE and routed experts dominate its behavior, significant
coherence loss versus `JANGTQ2` is expected.

Before publishing any positive quality claim:

- verify directory/capabilities/sidecar/index integrity
- inspect `mxtq_bits` and a sample of `.tq_bits` for value `1`
- load through `load_jangtq_model`
- run short prompt smoke against `JANGTQ2`
- run at least one longer reasoning/code prompt
- keep MTP wording as `preserved_disabled`

## Runtime Boundary

Python JANGTQ kernels appear 1-bit-capable. Swift/vmlx runtime support still
needs a direct load/generation proof before public claims. Do not assume every
Swift JANGTQ fast path accepts 1-bit merely because the Python path does.

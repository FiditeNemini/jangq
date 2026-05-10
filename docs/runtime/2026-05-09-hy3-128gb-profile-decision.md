# Hy3 128 GB Fit And JANGTQ Profile Decision

Created 2026-05-09.

## Decision

Do not advertise `Hy3-preview-JANGTQ_K` as comfortably fitting 128 GB devices until the finished bundle is measured and a real runtime load proves memory headroom.

Current recommendation:

| Target | Profile | Intended device class | Reason |
|---|---|---|---|
| First release candidate | `JANGTQ2` | 128 GB | Smaller routed expert footprint; higher quality risk on a 295B top-8 MoE, but this is the only current profile with reasonable 128 GB headroom. |
| Quality candidate | `JANGTQ_K` | 192 GB+ preferred; 128 GB only if measured runtime headroom is acceptable at short context | Better quality margin: routed `gate_proj/up_proj` at 2-bit, routed `down_proj` at 4-bit, non-routed core at 8-bit. |
| Reference fallback | `JANGTQ4` | Not a 128 GB target | Useful for quality comparison if disk/time allow. |

## Why `JANGTQ_K` Is Tight On 128 GB

Hy3 has 79 sparse MoE layers after `first_k_dense_replace=1`.

Approximate routed expert parameters:

```text
79 layers * 192 experts * 3 projections * 4096 hidden * 1536 expert hidden
= 286.3B routed expert parameters
```

Estimated routed expert storage:

| Profile | Routed bits | Routed expert storage |
|---|---:|---:|
| `JANGTQ2` | 2.0 avg | ~72 GB before sidecar/metadata overhead |
| `JANGTQ_K` | 2.67 avg | ~95 GB before sidecar/metadata overhead |
| `JANGTQ4` | 4.0 avg | ~143 GB before sidecar/metadata overhead |

The non-routed core still includes attention, shared experts, dense layer 0, embeddings, lm head, router, norms, and the MTP layer. With 8-bit affine storage this likely adds roughly 14-20 GB depending on final tensor census and sidecar overhead.

That puts `JANGTQ_K` in the ~110-120 GB finished-bundle range before runtime allocator pressure, KV cache, tokenizer/model metadata, Metal buffers, and OS/app memory. On a 128 GB unified-memory Mac, that is a tight target, not a comfortable one.

Low-RAM estimator run on the partial local source config:

| Profile | Estimated bundle | 4K KV cache | Runtime total with 12 GB headroom | 128 GB verdict |
|---|---:|---:|---:|---|
| `JANGTQ_K` | 113.3 GB | 1.34 GB | 126.6 GB | not comfortable |
| `JANGTQ2` | 88.5 GB | 1.34 GB | 101.8 GB | tight |

Hy3 KV cache is also non-trivial:

```text
80 layers * 2 K/V * 8 KV heads * 128 head_dim * fp16
= 327,680 bytes per token
~= 1.25 GB at 4K accepted tokens for batch 1
~= 5.0 GB at 16K accepted tokens for batch 1
```

Speculative MTP draft cache must be separate from accepted base KV and cannot be counted as free.

## Release Wording

Use this wording unless a measured load proves otherwise:

```text
Hy3-preview-JANGTQ_K is the quality-first mixed-bit bundle. It is expected to be tight on 128 GB devices; use short context and treat 192 GB+ as the preferred class until measured runtime proofs are published.
```

For this 128 GB-focused release, build and test `JANGTQ2` explicitly:

```text
Hy3-preview-JANGTQ2 is the 128 GB comfort candidate. It trades quality margin for memory headroom.
```

## Required Proof Before Upload Claims

- Finished bundle byte size from `model.safetensors.index.json`.
- `jangtq_runtime.safetensors` present and byte size included.
- First load RSS / unified memory pressure measurement.
- Fresh-cache generation at 512, 2048, and 4096 token windows.
- Two-turn cache continuation proof.
- MTP status stated as `preserved_disabled` or `enabled`.

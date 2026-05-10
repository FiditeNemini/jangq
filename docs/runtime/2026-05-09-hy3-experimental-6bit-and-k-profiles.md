# Hy3 Experimental 6-Bit And K Profiles

Created 2026-05-09.

This doc records Eric's follow-up idea: after the first `JANGTQ2` baseline, test a profile where the larger non-routed matmuls use 6-bit affine instead of 8-bit, and optionally combine that with the mixed routed-expert `K` profile (`4/2/2`, meaning `down/gate/up`).

## Current Baseline

`Hy3-preview-JANGTQ2` remains the active first conversion:

| Tensor family | Policy |
|---|---|
| Routed experts `gate/up/down` | MXTQ 2-bit |
| Attention, shared expert, dense FFN, embeddings/lm_head, MTP matmuls | 8-bit affine |
| Norms, router gate, expert bias | fp16 passthrough |

This is the 128 GB candidate.

## Feasible Experimental Profiles

MLX `mx.quantize(..., bits=6, group_size=64)` works on a local smoke test, so 6-bit affine is mechanically feasible.

| Profile name | Routed experts | Non-routed matmuls | Estimated bundle | 4K runtime total with 12 GB headroom | Role |
|---|---|---|---:|---:|---|
| `JANGTQ2` | 2/2/2 | 8-bit affine | ~88.5 GB | ~101.8 GB | First 128 GB baseline |
| `JANGTQ2_6` | 2/2/2 | 6-bit affine | ~85.2 GB | ~98.5 GB | Smaller 128 GB variant |
| `JANGTQ_K` | 4/2/2 (`down/gate/up`) | 8-bit affine | ~113.3 GB | ~126.6 GB | Quality-first, 192 GB preferred |
| `JANGTQ_K6` | 4/2/2 (`down/gate/up`) | 6-bit affine | ~110.0 GB | ~123.3 GB | Quality experiment, still tight on 128 GB |

The 6-bit change saves only about 3.3 GB because the routed experts dominate model size. It is useful, but it does not make `K` comfortable on 128 GB.

## Naming

Avoid calling this `JANGTQ1` in public metadata unless the runtime contract defines what the `1` means. There is no actual 1-bit path here.

Recommended internal names:

- `JANGTQ2_6`: routed experts 2-bit, non-routed matmuls 6-bit.
- `JANGTQ_K6`: routed down 4-bit, routed gate/up 2-bit, non-routed matmuls 6-bit.

## Converter Changes Needed

Add profile metadata without changing the first `JANGTQ2` run:

```text
JANGTQ2_6:
  routed expert gate/up/down -> MXTQ 2-bit
  attention/shared/dense/embed/lm_head/MTP matmuls -> affine 6-bit
  norms/router/expert_bias -> passthrough

JANGTQ_K6:
  routed expert down -> MXTQ 4-bit
  routed expert gate/up -> MXTQ 2-bit
  attention/shared/dense/embed/lm_head/MTP matmuls -> affine 6-bit
  norms/router/expert_bias -> passthrough
```

`jang_config.json` should explicitly stamp:

```json
{
  "profile": "JANGTQ2_6",
  "mxtq_bits": {
    "routed_expert": 2,
    "attention": 6,
    "shared_expert": 6,
    "dense_ffn": 6,
    "mtp": 6,
    "embed_tokens": 6,
    "lm_head": 6,
    "norms_router_biases": 16
  }
}
```

For `JANGTQ_K6`, `mxtq_bits.routed_expert` should be:

```json
{"down_proj": 4, "gate_proj": 2, "up_proj": 2}
```

## Runtime Checks

Before publishing either 6-bit profile:

- Confirm the Swift/Python affine runtime accepts `bits=6` for all non-routed linear roles.
- Confirm sidecar/build tools preserve `mxtq_bits` as a role dictionary and do not assume only 2/4/8.
- Run short coherence against the `JANGTQ2` baseline.
- Recompute measured output size; do not rely on estimates.
- Keep MTP status as `preserved_disabled` unless accept/reject speculative decode exists.

## Conversion Coordination Hazard

On 2026-05-09, a duplicate Hy3 conversion was started by another agent to the same output path while the first conversion was already running. This is unsafe because both processes can write `model-XXXXX-of-XXXXX.safetensors` shards and corrupt resume state.

Rule: only one active writer per output directory. Check with:

```sh
ps -axo pid,lstart,etime,%cpu,rss,command | rg 'convert_hy3_jangtq|Hy3-preview-JANGTQ'
```

If a duplicate starts, stop the newer writer and keep only one locked conversion. If the output may already be suspect, stop all writers and delete the partial output directory before restarting.

Cleanup performed 2026-05-09:

```text
stopped remaining Codex-started converter
deleted /Users/eric/models/JANGQ/Hy3-preview-JANGTQ2
verified no convert_hy3_jangtq writer remained
```

# MTP Retrofit Strategy - 2026-05-15

## Question

Can existing local JANG/JANGTQ/MXFP4 models be updated to include MTP layers,
or do they need full rebuilds?

## Short answer

There are three cases:

1. Existing artifact already contains `mtp.*` tensors.
   - No rebuild is needed to preserve MTP.
   - Bundle metadata should expose `mtp_mode=preserved_enabled`.
   - Runtime work is still required before speculative draft tokens can be
     accepted into the base cache.
   - First local example found: `/Volumes/EricsLLMDrive/jangq-ai/Qwen3.5-35B-A3B-JANG_4K`.

2. Existing artifact config advertises MTP, but no `mtp.*` tensors exist.
   - Runtime cannot activate MTP from this artifact.
   - A source recovery or rebuild is required unless the missing MTP tensors can
     be extracted from the exact same source checkpoint and appended with a
     compatible quant policy.

3. Existing artifact and source config both have no MTP.
   - No native MTP path exists for that model.
   - Keep `mtp_mode=none`; use it as a negative-space regression row.

## Patch vs rebuild decision

### Affine JANG artifacts

Affine JANG artifacts can sometimes be patched in place if all of these are true:

- the exact source checkpoint is available;
- source contains the missing `mtp.*` tensors;
- the existing artifact format is standard MLX-native JANG v2 safetensors;
- the MTP tensors can use the same affine quantization writer as normal tensors;
- `model.safetensors.index.json`, `config.json`, and `jang_config.json` are
  updated atomically;
- runtime can still load the base model with `mtp_mode=preserved_enabled` and
  fall back to plain autoregressive decode if speculative MTP is not wired.

This is a possible path for `Qwen3.6-27B-JANG_4M-MTP` once the BF16 source
download is available. The current local `Qwen3.6-27B-JANG_4M-CRACK` has
`text_config.mtp_num_hidden_layers=1` but no `mtp.*` tensors, so it cannot be
enabled as-is.

### JANGTQ / MXTQ artifacts

JANGTQ/MXTQ artifacts should normally be rebuilt, not patched, because MTP
touches more than a plain safetensors append:

- routed/expert tensors may be pre-stacked;
- `jangtq_runtime.safetensors` sidecars must stay consistent;
- `mxtq_bits` / profile metadata must describe MTP policy explicitly;
- loader hydration can hard-fail on missing or unexpected quantized modules;
- MTP cache/runtime state must be separated from accepted base cache state.

For JANGTQ, treat patching as unsafe unless the converter has a dedicated
`--preserve-mtp` path for that family.

### MXFP4 artifacts

MXFP4 artifacts should be rebuilt unless the missing MTP tensors are pure
passthrough sidecars. In practice, use a rebuild so all model config, index,
quant metadata, and runtime flags agree.

## Current local Qwen facts

### Qwen3.6 27B

Current artifact:

`/Users/eric/models/dealign.ai/Qwen3.6-27B-JANG_4M-CRACK`

Evidence:

- `model_type=qwen3_5`
- `text_config.mtp_num_hidden_layers=1`
- `text_config.mtp_use_dedicated_embeddings=false`
- `mtp.*` tensor keys: `0`
- source mirror `/Users/eric/models/Sources/Qwen/Qwen3.6-27B` has `15`
  `mtp.*` tensors and `333` `model.visual.*` tensors
- layer range: `0..63`

Status:

`config_claims_mtp_shape_but_no_tensors`.

Decision:

Download/recover source, then build a new `Qwen3.6-27B-JANG_4M-MTP` artifact.
Do not try to enable MTP on the current artifact.

### Qwen3.5 35B A3B historical artifact

Current artifact:

`/Volumes/EricsLLMDrive/jangq-ai/Qwen3.5-35B-A3B-JANG_4K`

Evidence:

- `model_type=qwen3_5_moe`
- `format=jang`
- `mtp.*` tensor keys: `46`
- sample keys:
  - `mtp.fc.weight`
  - `mtp.fc.scales`
  - `mtp.fc.biases`
  - `mtp.layers.0.self_attn.q_proj.weight`
  - `mtp.layers.0.self_attn.k_proj.weight`
  - `mtp.layers.0.self_attn.v_proj.weight`
  - `mtp.layers.0.self_attn.o_proj.weight`
  - `mtp.layers.0.mlp.shared_expert.up_proj.weight`
  - `mtp.layers.0.mlp.shared_expert.gate_proj.weight`
  - `mtp.layers.0.mlp.shared_expert.down_proj.weight`

Status:

`present_in_weights`.

Decision:

Do not use this as a target. It is old Qwen3.5 evidence only. The active MTP
workstream is Qwen3.6-only.

The only useful lesson from this artifact is shape evidence for the likely
Qwen-family MTP namespace:

- `mtp.fc.*`
- `mtp.layers.0.self_attn.*`
- `mtp.layers.0.mlp.*`

Reconfirm those names on the Qwen3.6 27B source before implementing converter
or runtime code.

### Qwen3.6 35B A3B JANGTQ

Current artifacts inspected:

- `/Users/eric/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK`
- `/Volumes/EricsLLMDrive/jangq-ai/Qwen3.6-35B-A3B-JANGTQ4`

Evidence:

- `model_type=qwen3_5_moe`
- `text_config.mtp_num_hidden_layers=1`
- `mtp.*` tensor keys: `0`

Decision:

Rebuild with a Qwen3.6 MoE JANG/JANGTQ converter that preserves MTP. Do not
enable MTP on the existing JANGTQ artifacts.

## Active scope

Only Qwen3.6 is in scope:

1. First target: `Qwen3.6-27B-JANG_4M-MTP`.
2. Second target after 27B works: Qwen3.6 35B A3B with an explicit MTP-preserve
   profile.
3. Qwen3.5 artifacts are historical references only and must not be used as
   release/runtime proof for Qwen3.6.

## Runtime activation plan

Every MTP-bearing artifact starts as:

```json
{
  "runtime": {
    "bundle_has_mtp": true,
    "mtp_layers": 1,
    "mtp_mode": "preserved_enabled"
  },
  "mtp": {
    "kept": true,
    "enabled": true,
    "num_layers": 1
  }
}
```

Only after an accept/reject proof script passes should runtime dispatch use:

```json
{
  "runtime": {
    "bundle_has_mtp": true,
    "mtp_layers": 1,
    "mtp_mode": "speculative_verified",
    "speculative_tokens": 1
  }
}
```

The runtime script must keep:

- base verifier cache: accepted tokens only;
- MTP draft cache: temporary and discardable;
- accepted draft token path: verified by the base model before commit;
- rejected draft token path: draft cache discarded;
- cancellation path: draft cache discarded;
- prefix/cache key: includes `mtp_mode`, model revision, quant profile, parser
  mode, chat-template salt, and media salt if applicable.

## Current download

Qwen3.6 27B source download target:

`/Volumes/EricsLLMDrive/Sources/Qwen/Qwen3.6-27B`

Local working mirror:

`/Users/eric/models/Sources/Qwen/Qwen3.6-27B`

Expected source repo:

`Qwen/Qwen3.6-27B`

Once complete, the first check is a static source census:

```sh
python3 - <<'PY'
from pathlib import Path
import json
p = Path("/Volumes/EricsLLMDrive/Sources/Qwen/Qwen3.6-27B")
idx = json.load(open(p / "model.safetensors.index.json"))
keys = idx["weight_map"].keys()
print([k for k in keys if k.startswith("mtp.")][:50])
PY
```

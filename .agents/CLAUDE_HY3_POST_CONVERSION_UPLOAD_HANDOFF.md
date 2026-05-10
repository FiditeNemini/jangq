# Claude Handoff: Hy3-preview-JANGTQ2 Post-Conversion

Status timestamp: 2026-05-09 local.

Audience: Claude / companion agents working on Hy3 packaging, README rendering, verification, and upload.

## Current State

Bundle path:

```text
/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2
```

Codex quick filesystem check confirms:

```text
config.json present
jang_config.json present
model.safetensors.index.json present
```

Claude-reported conversion facts to preserve in model card / notes:

```text
bundle size: ~79 GB
shards: 85
wallclock: ~22m25s
tensor census: 45504 MXTQ 2-bit routed experts + 1146 affine 8-bit + 488 fp16 passthrough = 47138 total
sidecar: jangtq_runtime.safetensors
sidecar domains: in_features {1536, 4096}
capabilities: family=hy_v3, reasoning=qwen3, tool=hunyuan, modality=text, cache_type=kv
MTP: bundle_has_mtp=true, mtp_layers=1, mtp_mode=preserved_disabled
```

## Do Not Overstate

Do not claim:

- MTP speculative decoding is enabled.
- Swift `vmlx-swift-lm` can already run Hy3.
- Python `../vmlx` can already run Hy3 locally.
- The bundle is quality-proven.
- Long-context 128 GB runtime is proven.

Correct wording:

```text
MTP status: tensors are preserved in the bundle, but current JANG/vmlx runtimes should decode normally until Hy3 accept/reject speculative decoding is implemented and tested.
```

```text
Runtime status: bundle is converted and structurally verified. Hy3-specific Python/Swift runtime implementation is pending; reference handoff docs and skeletons are included in the jang repo.
```

```text
128 GB status: bundle size is ~79 GB. This is the 128 GB candidate, but runtime load/headroom still needs measured proof on the intended engine.
```

## Architecture Facts For README

Hy3 is text-only:

```text
model_type=hy_v3
architectures=["HYV3ForCausalLM"]
no vision_config
no image_token_id
no preprocessor requirement
```

Attention:

```text
dense causal GQA KV cache
num_hidden_layers=80
hidden_size=4096
num_attention_heads=64
num_key_value_heads=8
head_dim=128
qk_norm=true
rope_parameters.rope_type=default
rope_parameters.rope_theta=11158840.0
max_position_embeddings=262144
```

Not Hy3:

```text
not MLA
not SSM/Mamba
not CCA
not sliding-window
not VLM/VL
not Qwen alias
not MiniMax alias
```

MoE:

```text
192 routed experts
top-8
sigmoid router
expert correction bias
route_norm=true
router_scaling_factor=2.826
num_shared_experts=1
first_k_dense_replace=1
```

Reasoning/tool surface:

```text
reasoning tags: <think>...</think>
reasoning_effort: no_think | low | high
tool parser: hunyuan / Tencent XML-like tags
tool call tags: <tool_calls>, <tool_call>, <tool_sep>, <arg_key>, <arg_value>
```

## Bit Policy

For `Hy3-preview-JANGTQ2`:

| Tensor family | Policy |
|---|---|
| Routed expert `gate/up/down` | MXTQ 2-bit |
| Attention q/k/v/o | affine 8-bit, group size 64 |
| Shared expert | affine 8-bit, group size 64 |
| Dense layer-0 MLP | affine 8-bit, group size 64 |
| Embeddings/lm_head | affine 8-bit, group size 64 |
| MTP matmuls | affine 8-bit, group size 64 |
| Q/K norms, RMSNorms | fp16 passthrough |
| Router gate, expert bias | fp16 passthrough |

Do not describe this as `JANGTQ_K`.

Do not mention MXFP4 for Hy3 unless Eric explicitly reopens that lane.

## Required Gates Before Upload

Run or verify equivalent output before upload:

```sh
uv run --project /Users/eric/jang/jang-tools python - <<'PY'
from pathlib import Path
from jang_tools.capabilities import verify_directory
p = Path('/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2')
ok, msg = verify_directory(p)
print(ok, msg)
raise SystemExit(0 if ok else 1)
PY
```

Check index integrity:

```sh
python3 - <<'PY'
import json, pathlib, sys
p = pathlib.Path('/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2')
idx = json.load(open(p / 'model.safetensors.index.json'))
missing = sorted({s for s in idx['weight_map'].values() if not (p / s).exists()})
print('tensors=', len(idx['weight_map']), 'shards=', len(set(idx['weight_map'].values())), 'missing=', len(missing))
if missing:
    print('first_missing=', missing[0])
    sys.exit(1)
PY
```

Check sidecar:

```sh
ls -lh /Users/eric/models/JANGQ/Hy3-preview-JANGTQ2/jangtq_runtime.safetensors
```

Check tokenizer/template:

```sh
ls /Users/eric/models/JANGQ/Hy3-preview-JANGTQ2/{tokenizer.json,tokenizer_config.json,chat_template.jinja,generation_config.json}
```

Optional low-RAM MTP/readiness inspect:

```sh
python3 /Users/eric/jang/jang-tools/examples/mtp/inspect_mtp_bundle.py /Users/eric/models/JANGQ/Hy3-preview-JANGTQ2
```

## README / Model Card Required Sections

Include:

- `What this is`: Hy3-preview JANGTQ2, text-only, 295B/21B active.
- `Quantization`: exact bit table above.
- `Runtime support matrix`: current support is converter/metadata complete; Hy3 runtime pending in Python/Swift.
- `MTP status`: `preserved_disabled`.
- `Reasoning/tool support`: qwen3 reasoning tags + Hunyuan/Tencent tool tags.
- `128 GB note`: bundle size is ~79 GB; runtime load proof still pending.
- `Known limitations`: no MTP decode yet; no Hy3 Swift/Python runtime yet; no quality benchmark yet.
- `Implementation handoff`: point to the docs/scripts below.

Reference docs/scripts already added by Codex:

```text
docs/runtime/2026-05-09-hy3-runtime-handoff-vmlx-python-swift.md
docs/runtime/2026-05-09-hy3-jangtq2-layer-bit-audit.md
docs/runtime/2026-05-09-mtp-runtime-integration-plan.md
docs/runtime/2026-05-09-mtp-spec-decoding.md
jang-tools/examples/hy3/python_runtime/
jang-tools/examples/hy3/swift_runtime/
jang-tools/examples/mtp/
```

## Upload Rule

Before upload:

1. Confirm no active writer:

```sh
ps -axo pid,etime,%cpu,rss,command | rg 'convert_hy3_jangtq|Hy3-preview-JANGTQ2' || true
```

2. Confirm final local path is stable and no `model-*-of-XXXXX.safetensors` temp names remain.
3. Render README/model card with no AI attribution.
4. Upload only to:

```text
OsaurusAI/Hy3-preview-JANGTQ2
```

5. Do not mirror to JANGQ-AI unless Eric explicitly says so.

## Coordination

If any upload starts, update `.agents/CURRENT.md` with:

```text
Locked: Hy3 upload to OsaurusAI/Hy3-preview-JANGTQ2
```

Remove the lock only after upload completes or is aborted.

If a runtime implementation pass starts later, do not edit `../vmlx` or `../vmlx-swift-lm` casually from this repo. Open a dedicated runtime task and use the handoff docs as source material.


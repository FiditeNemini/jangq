# Hy3 MTP Compatibility

Hy3 has `num_nextn_predict_layers=1`. Treat that as a runtime feature, not just
extra tensors in the bundle.

## Reference Server Startup

Tencent documents MTP-enabled serving for vLLM:

```sh
vllm serve tencent/Hy3-preview \
  --tensor-parallel-size 8 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 1 \
  --tool-call-parser hy_v3 \
  --reasoning-parser hy_v3 \
  --enable-auto-tool-choice \
  --served-model-name hy3-preview
```

And SGLang:

```sh
python3 -m sglang.launch_server \
  --model tencent/Hy3-preview \
  --tp 8 \
  --tool-call-parser hunyuan \
  --reasoning-parser hunyuan \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --speculative-algorithm EAGLE \
  --served-model-name hy3-preview
```

The local JANG smoke client in this folder assumes an OpenAI-compatible server:

```sh
python3 01_python_reference_smoke.py \
  --base-url http://127.0.0.1:8010 \
  --model hy3-preview
```

## JANG Runtime Contract

Runtime must expose an explicit MTP mode:

```text
mtp_mode = none | preserved_disabled | enabled
```

`none` means config and tensor census show no MTP.

`preserved_disabled` means normal autoregressive decode using only the 80 base
decoder layers. This is acceptable for a first runtime proof if model cards say
MTP is present in the bundle but speculative MTP decode is not enabled yet.

`enabled` means the runtime uses the `model.layers.80.*` MTP tensors to draft
one speculative token and verifies it through the base model before accepting
it. Do not call a runtime "MTP compatible" until this accept/reject path is
tested.

## Cache Rules

Hy3 base decode uses standard causal KV cache:

```text
base_kv[layer=0..79] = K/V tensors for accepted tokens
```

MTP speculative decode must keep draft state separate:

```text
mtp_draft_state != accepted base_kv
```

Acceptance rule:

- accepted draft token: append/update normal base KV
- rejected draft token: discard MTP draft state

Never persist speculative draft KV as accepted cache state.

## Bundle Metadata

JANGTQ bundles should include explicit MTP metadata:

```json
{
  "mxtq_bits": {
    "routed_expert": {"gate_proj": 2, "up_proj": 2, "down_proj": 4},
    "attention": 8,
    "shared_expert": 8,
    "dense_ffn": 8,
    "mtp": 8,
    "embed_tokens": 8,
    "lm_head": 8,
    "norms_router_biases": 16
  },
  "runtime": {
    "mtp_mode": "preserved_disabled|enabled",
    "mtp_layers": 1
  }
}
```

The converter should not silently drop `model.layers.80.*`. If the final tensor
census shows different MTP names, update this doc and the converter before
running the full conversion.

## Model Card Wording

Use one of these exact support lines:

```text
MTP status: bundle includes Hy3 MTP tensors, but this runtime currently uses normal autoregressive decode only.
```

```text
MTP status: enabled. Runtime uses Hy3's 1-layer MTP speculative path and verifies drafted tokens before accepting them into the base KV cache.
```

Do not use generic "MTP supported" wording without saying which of those two
states applies.

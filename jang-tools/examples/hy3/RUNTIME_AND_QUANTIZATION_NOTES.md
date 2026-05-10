# Hy3 Runtime And Quantization Notes

## Source Status

Source repo: `tencent/Hy3-preview`

Current local target:

```text
/Users/eric/models/Tencent/Hy3-preview
```

The source download is expected to be about 598 GB across 112 shards. Do not
start duplicate downloads. Use `ps` and `du -sh` to check the active transfer.

## Runtime Shape

Hy3 is text-only. It has no VL processor, no image-token path, and no media
salt requirement. Runtime work is still non-trivial because it combines:

- dense GQA attention
- q/k RMSNorm before RoPE
- sigmoid top-k MoE routing with expert-bias correction
- one shared expert per sparse layer
- one initial dense FFN layer
- one native MTP layer for speculative decoding

Cache topology is standard causal KV. Do not classify this as MLA, SSM hybrid,
SWA hybrid, CCA, or VL.

## MoE Rule

The runtime must match the upstream router semantics:

1. compute router logits in fp32
2. apply sigmoid
3. add `e_score_correction_bias` for expert selection
4. choose top-8 experts
5. gather the original sigmoid weights for the selected experts
6. normalize selected weights by their sum
7. multiply by `router_scaling_factor=2.826`
8. add the always-active shared expert output

The router gate and expert correction bias are precision-sensitive. They should
stay passthrough in initial bundles.

## Attention Rule

Attention is dense causal GQA:

- Q heads: 64
- KV heads: 8
- head dim: 128
- RoPE theta: 11158840
- context length: 262144
- `q_norm` and `k_norm` run per head before RoPE

First runtime proof should validate:

- short fresh-cache generation
- two-turn continuation using cache
- same prompt deterministic replay
- q/k norm present and not silently skipped

## MTP Policy

The config has `num_nextn_predict_layers=1`. The first release has two honest
options:

- implement MTP speculative decode and test it
- explicitly ignore/drop MTP tensors and state that runtime uses normal decode

The JANGTQ bundle should preserve MTP tensors with an explicit `mtp=8` policy
when the source index exposes them. The runtime may still run normal decode at
first, but the model card and runtime status must say whether MTP speculative
decode is enabled.

Do not silently include MTP weights without a runtime status line.

## Converter Policy

The final converter should be source-index driven. Do not assume tensor names
from MiniMax, DeepSeek, or Ling until `model.safetensors.index.json` is local
and scanned.

Active quantization policy:

| Tensor family | `JANGTQ2` first 128 GB target |
|---|---|
| attention q/k/v/o | affine 8-bit |
| q_norm/k_norm/RMSNorm | passthrough |
| routed expert gate/up/down | MXTQ 2-bit |
| shared expert | affine 8-bit |
| dense layer-0 MLP | affine 8-bit |
| router gate/bias | passthrough |
| lm_head | 8-bit or passthrough first |
| MTP | affine 8-bit where present, with runtime status documented |

`JANGTQ2` is the first 128 GB release candidate. It should not be called the
best quality profile until it has coherence proof against `JANGTQ_K`.

## Swift Work Items

- `Hy3Config` decode for `hy_v3`
- `Hy3Attention` with q/k RMSNorm and GQA KV cache
- `Hy3MoE` with sigmoid+bias top-k routing and shared expert
- `Hy3MTP` policy module
- model factory dispatch
- JANGTQ switch expert decode for stacked routed experts
- tests for fresh-cache, cache continuation, router top-k, q/k norm, and MTP policy

## Python Work Items

- source inspector and tensor census
- converter common helpers after index is complete
- `convert_hy3_jangtq.py`
- reference smoke client against vLLM/SGLang
- local load/smoke once JANG runtime exists

## Upload Gate

Do not upload Osaurus bundles until:

- full source download verifies
- converter has explicit tensor-name coverage
- `JANGTQ2` output passes `verify_directory`
- JANGTQ sidecar exists
- runtime proof records real generated text
- cache proof records fresh and continuation behavior
- model card clearly states MTP and runtime support status
- model card does not call `JANGTQ_K` comfortable on 128 GB unless measured
  runtime load proves it

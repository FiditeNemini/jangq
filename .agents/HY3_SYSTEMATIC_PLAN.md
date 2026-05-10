# Hy3-preview Systematic Coordination Plan

Status timestamp: 2026-05-09 local.

Private coordination file. `.agents/` is gitignored.

## Current Download

Source: `tencent/Hy3-preview`

Target:

```text
/Users/eric/models/Tencent/Hy3-preview
```

Active downloader observed:

```text
uvx --from huggingface-hub hf download tencent/Hy3-preview --repo-type model --local-dir /Users/eric/models/Tencent/Hy3-preview --max-workers 4
```

Do not start a second downloader. Check `ps` and `du -sh` first.

Latest Codex monitor:

- active downloader PID was still running
- local size reached 317 GB during this pass
- `model.safetensors.index.json` was still absent, so full tensor coverage is not final yet

## Verified Metadata

Current local `config.json` and Hugging Face docs/model card agree:

- `model_type=hy_v3`
- `architectures=["HYV3ForCausalLM"]`
- text-only, no VL sidecar expected
- 295B total / 21B active
- 80 base decoder layers plus 1 MTP layer
- hidden size 4096
- GQA: 64 Q heads, 8 KV heads, head dim 128
- q/k RMSNorm before RoPE
- context length 262144
- RoPE theta 11158840
- MoE: 192 routed experts, top-8
- sigmoid router with expert-bias correction
- one shared expert per sparse MoE layer
- first layer dense (`first_k_dense_replace=1`)
- `enable_lm_head_fp32=true`
- reasoning effort levels in card/template: `no_think`, `low`, `high`
- recommended serving parsers: vLLM `hy_v3`, SGLang `hunyuan`

## Local Artifacts Added

- `jang-tools/examples/hy3/README.md`
- `jang-tools/examples/hy3/00_inspect_source.py`
- `jang-tools/examples/hy3/01_python_reference_smoke.py`
- `jang-tools/examples/hy3/02_python_runtime_contract.py`
- `jang-tools/examples/hy3/Hy3RuntimeContract.swift`
- `jang-tools/examples/hy3/RUNTIME_AND_QUANTIZATION_NOTES.md`

## Immediate Gates

1. Let download finish.
2. Verify checksums with `hf cache verify` or equivalent `hf` local-dir check.
3. Run `00_inspect_source.py` again once `model.safetensors.index.json` exists.
4. Build tensor-name census:
   - dense layer 0
   - sparse MoE layers 1..79
   - router gate and `e_score_correction_bias`
   - shared expert
   - q/k norms
   - MTP layer tensors
   - embed/lm_head
5. Only then finalize converter mappings.

## Converter Plan

Active profile priority:

- **First 128 GB target: `JANGTQ2`** — routed expert gate/up/down at 2-bit, attention/shared/dense/MTP at 8-bit affine, router/norms passthrough.
- `JANGTQ_K` remains the later quality-first target, but is likely tight (~110-120 GB bundle range before runtime/KV/OS headroom). Do not market it as comfortable on 128 GB until measured load proof exists.
- `JANGTQ4` remains the quality/reference fallback if `JANGTQ2` coherence is not good enough.
- MXFP4 is out of the active lane unless Eric asks for it again.

Initial JANGTQ2 policy:

- routed experts -> MXTQ 2-bit
- attention -> 8-bit affine for JANGTQ first pass
- shared expert -> 8-bit affine first pass
- dense layer-0 MLP -> 8-bit affine first pass
- q/k norms, RMSNorms, router gate, expert bias -> passthrough
- lm_head -> fp16 or 8-bit until coherence proof
- MTP -> explicit support/inclusion policy, never silent drop. Best first pass is 8-bit affine MTP tensors plus runtime docs saying whether speculative decode is enabled or normal decode is used.

Do not assume MiniMax or Ling tensor names until full index scan completes.

Dry-run evidence on partial local shards:

- `convert_hy3_jangtq --dry-run ... JANGTQ2` runs without writing output.
- Latest partial-shard dry-run found 32360 tensors: 31979 MXTQ, 304 affine, 77 passthrough.
- Current partial shards expose base layers 0..79 only; no MTP-like tensor names have appeared yet. This is not final until `model.safetensors.index.json` is present.
- Partial-shard dry-run saw routed expert tensors under:
  - `model.layers.N.mlp.experts.E.{gate_proj,up_proj,down_proj}.weight`
- Router and expert-bias tensors seen:
  - `model.layers.N.mlp.router.gate.weight`
  - `model.layers.N.mlp.expert_bias`
- Shared expert tensors seen:
  - `model.layers.N.mlp.shared_mlp.{gate_proj,up_proj,down_proj}.weight`
- q/k norm tensors seen:
  - `model.layers.N.self_attn.{q_norm,k_norm}.weight`
- Full MTP tensor naming still needs source-index confirmation after download completes.
- 128 GB fit doc: `docs/runtime/2026-05-09-hy3-128gb-profile-decision.md`.
- Layer/bit audit doc: `docs/runtime/2026-05-09-hy3-jangtq2-layer-bit-audit.md`.
- Experimental 6-bit/K profile doc: `docs/runtime/2026-05-09-hy3-experimental-6bit-and-k-profiles.md`.
- vmlx/vmlx-swift-lm runtime handoff doc: `docs/runtime/2026-05-09-hy3-runtime-handoff-vmlx-python-swift.md`.
- Runtime skeletons:
  - `jang-tools/examples/hy3/python_runtime/`
  - `jang-tools/examples/hy3/swift_runtime/`
- Fit estimator: `jang-tools/examples/mtp/estimate_jangtq_fit.py`.

## Runtime Plan

Python:

- reference smoke against vLLM or SGLang server
- future local runtime wrapper once JANG Hy3 runtime exists
- parser handling for reasoning effort and tool parser (`hunyuan`/`hy_v3`)

Swift:

- `Hy3Config`
- dense GQA attention with q/k norm
- sigmoid+bias top-k router
- shared expert path
- MTP policy and tests:
  - normal decode path must be correct with MTP disabled
  - MTP speculative path must be gated by runtime support
  - cache must distinguish normal decode state from speculative draft state
- standard KV cache tests
- JANGTQ expert dispatch

## Upload Gate

No Osaurus upload until:

- source download complete and verified
- converter tensor coverage complete
- generated bundles pass `verify_directory`
- JANGTQ sidecar exists
- runtime generation proof exists
- cache continuation proof exists
- model card states exact runtime support and MTP status

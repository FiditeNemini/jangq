# DSV4 Flash vMLX 1.5.47 Live Quality Blocker

Date: 2026-05-22 00:20 PDT

Update: 2026-05-22 00:55 PDT

Source evidence from vMLX Python/Electron worktree:

- Repo: `/Users/eric/mlx/vllm-mlx-finite-launch-guard`
- Commit tested: `55dbd3bd`
- Latest vMLX release-hardening commit after suite work:
  `af141b30 test: require mcp policy gate markers`
- Bundled runtime: `panel/release/mac-arm64/vMLX.app/.../python3`, `vmlx_engine 1.5.47`
- Model tested: `/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANG`
- vMLX live artifact: `build/current-production-family-audit-live-dsv4-jang-local-20260522.json`
- Raw probe artifacts:
  - `/tmp/vmlx_family_audit/dsv4_jang_local_dsv4_threejs_identifier_integrity_1779434265.json`
  - `/tmp/vmlx_family_audit/dsv4_jang_local_dsv4_long_vc_project_plan_1779434384.json`

## Current Result

Do not treat the current local DSV4 Flash JANG artifact as release-cleared for long-output/code/file-generation quality.

The live vMLX row still fails:

- `dsv4_threejs_identifier_integrity`
  - corrupt identifiers: `THREE.PPerspectiveCamera`, `THREE.MMeshBasicMaterial`
  - also emits markdown fences when asked for exact code only
- `dsv4_thinking_mode_max`
  - finish reason `length`
  - no visible content
  - 4145 reasoning characters
- `dsv4_long_context_full_output_vc_project_plan`
  - finish reason `length`
  - no visible content
  - repetitive reasoning tail
- `responses_tool_history_continuation`
  - returned `READEOM.md`
- static audit still reports the output-head/final-norm precision boundary:
  - `head=U32`
  - `norm=F16`
  - source-vs-quant or rebuilt-artifact clearance is still required before long-output production claims

## Important Non-Cause

This specific failing row is not cleared by changing vMLX UI wording, max-output/context wiring, tool parser selection, or generic cache toggles.

Recent vMLX no-heavy gates are green for:

- max output vs max context separation
- parser registry / MiniMax alias coverage
- generation defaults without hidden sampler forcing
- DSV4 DSML tool parser and default-cache multi-tool rows
- VL/media/cache/tool-followup named rows
- API surfaces for Chat, Responses, Anthropic, and Ollama
- MCP autodiscovery, policy filtering, secret redaction, gateway routing, and
  built-in-tool separation marker rows

The remaining DSV4 blocker is output quality on the tested artifact/runtime path.

Latest vMLX proof after the MCP marker hardening:

- `build/current-mcp-policy-contract-20260522-marker-hardening.json`
  - pass, `missing_markers=[]`
  - engine MCP/security: 76 passed
  - panel MCP/gateway: 13 passed, 3 optional live rows skipped
- `build/current-regression-suite-20260522-mcp-marker-hardening.json`
  - pass, `failed_steps=[]`
  - still lists only:
    `DSV4 long-output/code/file-generation quality is release-cleared`
- `build/current-release-surface-contract-20260522-post-mcp-marker-hardening-push.json`
  - pass
  - public updater and PyPI remain at `1.5.46`
  - GitHub `jjang-ai/vmlx` release `v1.5.47` is still not published

## vMLX Harness Fix Made During Diagnosis

One stale audit-harness issue was found and fixed in vMLX:

- `tests/cross_matrix/run_production_family_audit.py` had launched DSV4 with generic `--enable-prefix-cache`.
- DSV4 native composite cache needs `--dsv4-enable-prefix-cache`.
- vMLX commit `55dbd3bd` updates the production-family audit command builder to emit the DSV4-native flag for `deepseek_v4` rows while leaving non-DSV4 rows on generic prefix cache.

That harness fix prevents future cache false negatives. It does not fix the identifier/long-output quality failure above.

## Model-Side Next Work

Before uploading/replacing the DSV4 model artifact, the JANG side needs a fresh source-vs-quant or rebuilt-body clearance:

1. Run the exact identifier canary on source or a newly rebuilt DSV4 body:

   ```text
   const scene = new THREE.Scene();
   const renderer = new THREE.WebGLRenderer();
   const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
   const mesh = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial());
   ```

2. Require all of these before claiming release-ready:
   - no markdown fence
   - exact identifiers preserved
   - no duplicated identifier pieces such as `PPerspective`, `MMesh`, `BBox`, `WebWeb`
   - finish reason is not `length`
   - visible content is present when the prompt asks for visible output

3. Re-run the vMLX live row after the rebuilt artifact is in place.

The likely repair lane is model artifact / quantization boundary work, especially around output head, final norm, routed-bit plan, group sizes, and source-vs-quant parity. Do not hide this with sampler defaults or hard output constraints.

Important 2026-05-22 follow-up:

- The earlier `DeepSeek-V4-Flash-JANGTQ-K-HeadBF16-Probe-20260520` overlay is
  only a head/norm overlay on top of the existing quantized body. It is not
  equivalent to a full high-precision DSV4 rebuild.
- `jang_tools.dsv4.convert_dsv4_jangtq` has a full candidate lane:
  `DSV4_HIGH_PRECISION=1`.
- That lane preserves the whole every-token non-routed path as passthrough:
  attention, shared experts, compressor/indexer, embed, and head. Routed
  experts remain on the selected MXTQ profile.
- The guard is now pinned in
  `jang-tools/tests/test_dsv4_converter_contract.py` as
  `test_dsv4_jangtq_high_precision_keeps_full_nonrouted_path_passthrough`.
- This still does not clear release quality by itself. A new full
  high-precision or otherwise rebuilt artifact must pass the vMLX identifier
  canary and long-output gates before upload/release claims.

Current vMLX evidence makes the failure narrower than "long context only":

- Single identifier copy can pass:
  `THREE.WebGLRenderer`
- Multi-identifier exact-copy probes fail with duplicated API-name fragments:
  - `THREE.WebWebGLRenderer`
  - `THREE.PPerspectiveCamera`
  - `THREE.MMeshBasicMaterial`
  - `THREE.BBoxGeometry`
  - `THREE.ScScene`
- The identifier-count ablation was run with:
  - `DSV4_LONG_CTX=1`
  - `DSV4_POOL_QUANT=0`
  - `VMLINUX_DSV4_ENABLE_PREFIX_CACHE=0`
- That means the current blocker is not proven to be caused by prefix cache,
  L2 cache, MCP, Responses assembly, UI settings, or tool parsing.

Release implication:

- vMLX can continue hardening and testing other family rows.
- A v1.5.47 public release should not claim DSV4 long-output/code quality as
  cleared until a rebuilt/source-equivalent DSV4 body passes the identifier and
  long-output gates above.

## vMLX Update - 2026-05-22 07:25 PDT

Latest vMLX Python/Electron checkpoint:

- repo/worktree: `/Users/eric/mlx/vllm-mlx-finite-launch-guard`
- branch: `codex/pr-intake-manifest`
- commit: `29286344 test: pin qwen nemotron hybrid cache rows`
- pushed branch: `origin/codex/pr-intake-manifest`

Fresh vMLX proof after the Qwen/Nemotron family-gate checkpoint:

- `build/current-model-family-detection-contract-20260522-qwen-nemotron-hybrid-cache.json`
  - `status=pass`
  - `missing_rows=[]`
  - engine `41 passed / 111 deselected`
  - panel `41 passed / 12 skipped`
- `build/current-regression-suite-20260522-qwen-nemotron-hybrid-cache.json`
  - `status=pass`
  - `failed_steps=[]`
  - still open: `DSV4 long-output/code/file-generation quality is release-cleared`
- `build/current-release-surface-contract-20260522-post-qwen-nemotron-hybrid-cache.json`
  - `status=pass`

The current objective digest still keeps the DSV4 quality row open because:

- exact-code identifier checks are false:
  - `identifier_integrity=false`
  - `threejs_single_file=false`
  - `no_markdown_fence=false`
  - `no_corrupt_identifiers=false`
  - `non_length_stop=false`
  - `source_or_rebuilt_body_clearance=false`
- missing clearance artifacts include:
  - `build/dsv4-source-full-output/result.json`
  - `build/dsv4-chat-prompt-ablation-20260520101331/result.json`
- the current identifier ablation still shows corrupted multi-identifier API
  names such as `THREE.WebWebGLRenderer`, `THREE.PPerspectiveCamera`,
  `THREE.MMeshBasicMaterial`, `THREE.BBoxGeometry`, and `THREE.ScScene`.

JANG-side converter guard rechecked from vMLX at 2026-05-22 07:25 PDT:

```bash
PYTHONPATH=/Users/eric/jang/jang-tools .venv/bin/python -m pytest -q \
  /Users/eric/jang/jang-tools/tests/test_dsv4_converter_contract.py \
  -k "high_precision or rope_scaling or f32_control or metadata_declares"
```

Result: `6 passed, 21 deselected`.

This only proves the converter guard for the next rebuilt candidate. It does
not clear the current DSV4 model artifact. The rebuilt/source-equivalent DSV4
body still has to pass the vMLX identifier and long-output live gates before
vMLX can honestly release-claim DSV4 long-output/code/file generation.

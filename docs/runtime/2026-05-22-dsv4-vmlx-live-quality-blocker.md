# DSV4 Flash vMLX 1.5.47 Live Quality Blocker

Date: 2026-05-22 00:20 PDT

Source evidence from vMLX Python/Electron worktree:

- Repo: `/Users/eric/mlx/vllm-mlx-finite-launch-guard`
- Commit tested: `55dbd3bd`
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

The remaining DSV4 blocker is output quality on the tested artifact/runtime path.

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

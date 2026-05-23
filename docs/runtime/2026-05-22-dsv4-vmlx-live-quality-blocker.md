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

## vMLX Update - 2026-05-22 08:00 PDT

Latest vMLX Python/Electron checkpoint:

- repo/worktree: `/Users/eric/mlx/vllm-mlx-finite-launch-guard`
- branch: `codex/pr-intake-manifest`
- commit: `6de1134e test: pin persisted chat output cap isolation`
- pushed branch: `origin/codex/pr-intake-manifest`
- release gate version triple: `1.5.47`

Fresh vMLX proof after the persisted Chat Max Output Tokens isolation guard:

- `build/current-max-output-context-contract-20260522-persisted-chat-output-cap-final.json`
  - `status=pass`
  - `failed=[]`
  - `missing_markers=[]`
  - engine `20 passed`
  - panel `39 passed / 292 skipped`
- `build/current-release-regression-manifest-20260522-persisted-chat-output-cap-final.json`
  - 18 release-regression rows
- `build/current-release-surface-contract-20260522-post-persisted-chat-output-cap.json`
  - `status=pass`
- `build/current-regression-suite-20260522-post-persisted-chat-output-cap.json`
  - `status=pass`
  - `failed_steps=[]`
  - still open: `DSV4 long-output/code/file-generation quality is release-cleared`
- direct vMLX release gate:
  - `panel/scripts/release-gate-python-app.py --skip-app --skip-gui`
  - rc=1 with `[FAIL] objective proof digest: DSV4 long-output/code/file-generation quality is release-cleared`

Fresh JANG-side guard recheck from the vMLX worktree:

```bash
PYTHONPATH=/Users/eric/jang/jang-tools .venv/bin/python -m pytest -q \
  /Users/eric/jang/jang-tools/tests/test_dsv4_converter_contract.py \
  -k "high_precision or rope_scaling or f32_control or metadata_declares"
```

Result: `6 passed, 21 deselected, 2 warnings`.

Current local DSV4 artifact inventory:

- `/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANG`
  - size: `97G`
  - `weight_format=affine`
  - `profile=JANG_2L_GS64_ProjLayerBits_Ggs32-Dgs32-Ugs64_bk4_Tok8g64_NoMTP`
  - `rope_scaling` is present and matches the required YaRN block.
- `/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K`
  - size: `80G`
  - `weight_format=mxtq`
  - `profile=JANGTQ_K`
  - `rope_scaling` is present and matches the required YaRN block.
- `/Users/eric/models/Sources/DeepSeek-V4-Flash`
  - size: `149G`
  - available locally, but not a realistic live vMLX clearance target on the
    128G local machine without a different loading strategy or remote machine.

## vMLX Update - 2026-05-23 15:55 PDT

Latest vMLX Python/Electron checkpoint:

- repo/worktree: `/Users/eric/mlx/vllm-mlx-finite-launch-guard`
- branch: `codex/pr-intake-manifest`
- release gate version triple: `1.5.48`

Fresh vMLX proof after the Qwen native-MTP/VLM norm-format fix and bundled
runtime rebuild:

- Qwen VLM+native-MTP default live artifact:
  `build/current-decode-speed-live-qwen27-jang4m-mtp-default-after-norm-shift-20260523.json`
  - `status=pass`
  - prompt-processing rows around `865`, `817`, and `760 tok/s`
  - deterministic counting output, no loopish output
  - native MTP active for `text+vl`, depth 3
- bundled Python was rebuilt from the vMLX checkout and verified with
  `npm --prefix panel run verify-bundled`;
  source-vs-bundled `vmlx_engine` and `jang_tools` hash parity passed.
- canonical vMLX umbrella suite:
  `build/current-regression-suite-20260523-profile-chat-cap-clean-jang.json`
  - `status=pass`
  - `failed_steps=[]`
  - only open requirement:
    `DSV4 long-output/code/file-generation quality is release-cleared`
- release-regression manifest proof sweep:
  `build/current-release-regression-manifest-20260521.json`
  - `current_proof_sweep=pass`

This narrows the release blocker again. Do not reopen Qwen based on older
pre-norm-shift artifacts unless a new no-env default live artifact regresses.
The current vMLX release gate has exactly one known open requirement: this DSV4
long-output/code/file-generation row.

The DSV4 quality failure must not be papered over with hidden parameters:

- do not force temperature, top-p, top-k, repetition penalty, reasoning, or
  output caps outside the model metadata / request / explicit UI setting path;
- do not call DSV4 cleared because max output vs max context wiring is now
  guarded;
- do not call DSV4 cleared because DSML tools, Responses assembly, prefix,
  paged cache, block-disk L2, or pool-quant rows pass;
- do not call DSV4 cleared because a single exact identifier can pass.

The remaining failure evidence is model/runtime output quality on the current
artifact. The current diagnostic evidence includes an installed-tokenizer
roundtrip pass for identifiers and logprob/context probes where the model path
can rank corrupted identifier continuations above the exact continuation in
multi-identifier/code contexts. That points the next JANG-side work back to
artifact/runtime parity: source-vs-quant, full high-precision or otherwise
rebuilt DSV4 body, final norm/output head/non-routed path precision, routed-bit
plan, group sizes, and exact source-body clearance.

Next clearance action:

1. Build or stage a rebuilt/source-equivalent DSV4 candidate. The existing
   `DSV4_HIGH_PRECISION=1` converter lane is only a candidate path until a full
   artifact exists.
2. Re-run the exact vMLX identifier, long-output, and tool-history live gates
   against that candidate.
3. Only update the vMLX objective proof row from open to pass after the live
   artifacts prove exact identifiers, no corrupt duplicated fragments, visible
   requested content, non-`length` finish for bounded prompts, and no hidden
   sampler/config forcing.

Interpretation:

- vMLX max-output/context wiring, persisted chat output caps, parser registry,
  DSV4 default native cache/tool loops, API/cache surfaces, and release-surface
  checks are green in the current suite.
- The release gate still intentionally blocks because the current DSV4 local
  model body has not passed exact multi-identifier/code and long-output
  generation.
- The next real DSV4 release-clearance path remains a rebuilt/source-equivalent
  body, likely using the guarded `DSV4_HIGH_PRECISION=1` lane or another
  source-vs-quant parity-proven rebuild. Do not clear this by changing sampler
  defaults, hiding max-token caps, or documenting around the failure.

## vMLX Update - 2026-05-22 08:21 PDT

Latest vMLX checkpoint:

- repo/worktree: `/Users/eric/mlx/vllm-mlx-finite-launch-guard`
- branch: `codex/pr-intake-manifest`
- commit: `79f14837 fix: block stale dsv4 native mtp args`
- pushed branch: `origin/codex/pr-intake-manifest`
- release gate version triple: `1.5.47`

Fresh vMLX release-path state:

- post-push release surface:
  `build/current-release-surface-contract-20260522-post-dsv4-additional-args.json`
  -> `status=pass`
- post-push umbrella:
  `build/current-regression-suite-20260522-post-dsv4-additional-args.json`
  -> `status=pass`, `failed_steps=[]`, still open:
  `DSV4 long-output/code/file-generation quality is release-cleared`
- direct release gate after rebuilding bundled Python:
  `docs/internal/release-gates/20260522_081735/SUMMARY.md`
  -> all non-app checks pass except objective digest, which fails only on:
  `DSV4 long-output/code/file-generation quality is release-cleared`

Bundled-runtime parity was repaired locally:

- initial direct release gate failed because bundled
  `jang_tools/convert_hy3_jangtq.py` drifted from clean JANG source.
- reran vMLX `./panel/scripts/bundle-python.sh` with clean JANG source:
  `/Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools`
- `npm --prefix panel run verify-bundled` now passes all critical imports and
  source-vs-bundle hash parity for `vmlx_engine` and `jang_tools`.

Current DSV4 artifact header-only checks:

```bash
PYTHONPATH=/Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools:$PWD \
  .venv/bin/python - <<'PY'
from pathlib import Path
from vmlx_engine.loaders.load_jangtq_dsv4 import (
    _audit_dsv4_control_tensor_dtypes,
    _dsv4_nested_routed_bit_plan,
    _dsv4_routed_default_bits,
    _read_json,
)
for p in [
    Path('/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K'),
    Path('/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANG'),
    Path('/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K-HeadBF16-Probe-20260520'),
]:
    cfg=_read_json(p/'config.json'); jang=_read_json(p/'jang_config.json')
    print(p, _audit_dsv4_control_tensor_dtypes(p))
    print('default_bits', _dsv4_routed_default_bits(cfg,jang))
    print('routed_plan', _dsv4_nested_routed_bit_plan(cfg,jang))
PY
```

Result:

- all three local DSV4 artifacts have `critical_count=344`,
  `non_f32_count=0`;
- all three pass `scripts/validate_dsv4_flash_rope_scaling.py`;
- DSV4-K routed plan is still:
  `{'0': 2, '1': 2, '2': 2, '23': 4, '25': 4, '28': 4, '34': 4, '36': 4}`.

This means the current blocker is not missing YaRN config or downcast mHC/
router/sink controls in the local artifacts. The existing failing live
artifacts remain failing evidence, not clearance evidence:

- `/Users/eric/mlx/vllm-mlx/build/dsv4-source-full-output/result.json`
  -> `status=open`
- `/Users/eric/mlx/vllm-mlx/build/dsv4-chat-prompt-ablation-20260520101331/result.json`
  -> still shows `THREE.WebWebGLRenderer`, `THREE.ScScene`,
  `THREE.PPerspectiveCamera`, `THREE.BBoxGeometry`,
  `THREE.MMeshBasicMaterial`
- `/Users/eric/mlx/vllm-mlx/docs/internal/release-gates/20260520_sisyphus_dsv4_identifier_gate_jang_affine_current/result.json`
  -> failing identifier evidence, not a pass artifact

JANG worktree nuance:

- The clean JANG release worktree used for bundling does not contain tracked
  `jang-tools/_internal/jang_v3/*`.
- The dirty main JANG checkout has those files only as untracked local files.
- Therefore this JANG-side test batch currently fails in the clean worktree:

```bash
PYTHONPATH=/Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools:$PWD \
  .venv/bin/python -m pytest -q \
  /Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools/tests/test_dsv4_converter_contract.py \
  /Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools/tests/test_jang_v3_dsv4_contract.py \
  /Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools/tests/test_dsv4_rope_reference.py \
  /Users/eric/jang/.worktrees/vmlx-release-clean-7f643ed/jang-tools/tests/test_dsv4_hc_sinkhorn.py
```

Result: `29 passed, 3 failed`; the three failures are all missing
`_internal.jang_v3` / missing `jang-tools/_internal/jang_v3/encode.py`.

Interpretation:

- vMLX release-path parity is now clean except the DSV4 objective row.
- Current local DSV4-K/JANG artifacts have correct rope metadata and F32
  critical controls, but still do not have live exact-code/identifier
  clearance.
- The remaining real path is a rebuilt/source-equivalent DSV4 body plus live
  vMLX identifier/full-output gate, not a vMLX UI/API/cache/parser setting and
  not hidden sampler forcing.

## JANG Update - 2026-05-22 08:32 PDT

Fixed the clean-worktree DSV4 V3 helper hygiene issue noted above:

- `.gitignore` now keeps `jang-tools/_internal/` ignored by default but
  explicitly unignores the narrow helper files required by the DSV4 V3 safety
  tests:
  - `jang-tools/_internal/jang_v3/__init__.py`
  - `jang-tools/_internal/jang_v3/budget_solver.py`
  - `jang-tools/_internal/jang_v3/encode.py`
- The rest of `jang-tools/_internal/jang_v3/` remains ignored because it is
  calibration/scratch pipeline code and not needed for the release safety
  contract.

Fresh focused JANG verification:

```bash
PYTHONPATH=/Users/eric/jang/jang-tools \
  /Users/eric/mlx/vllm-mlx-finite-launch-guard/.venv/bin/python \
  -m pytest -q jang-tools/tests/test_jang_v3_dsv4_contract.py
```

Result: `3 passed`.

```bash
PYTHONPATH=/Users/eric/jang/jang-tools \
  /Users/eric/mlx/vllm-mlx-finite-launch-guard/.venv/bin/python \
  -m pytest -q \
  jang-tools/tests/test_dsv4_converter_contract.py \
  jang-tools/tests/test_dsv4_rope_reference.py \
  jang-tools/tests/test_dsv4_hc_sinkhorn.py
```

Result: `30 passed, 2 warnings`.

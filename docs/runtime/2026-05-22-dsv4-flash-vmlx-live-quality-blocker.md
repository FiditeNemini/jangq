# DSV4 Flash vMLX Live Quality Blocker - 2026-05-22

This note mirrors the vMLX release gate state for DeepSeek-V4-Flash JANG.

Do not claim DSV4 Flash long-output/code/file-generation production clearance
from the current local affine artifact:

- model: `/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANG`
- vMLX repo/worktree:
  `/Users/eric/mlx/vllm-mlx-finite-launch-guard`
- fresh live artifact:
  `build/current-production-family-audit-live-dsv4-jang-local-20260522-after-stream-cache-detail.json`
- command:
  `uv run --extra dev python tests/cross_matrix/run_production_family_audit.py --rows dsv4_jang_local --live --out build/current-production-family-audit-live-dsv4-jang-local-20260522-after-stream-cache-detail.json`

Result: live `FAIL`, 5 failed rows.

Passed in the same live audit:

- DSV4 native/paged composite cache enabled;
- canonical DSV4 encoder shim;
- multi-EOS;
- cache/model capability endpoints;
- runtime cache layout logging;
- basic thinking-off chat;
- thinking-on recall/toggle;
- structured Responses auto-tool choice;
- Anthropic/Ollama basics;
- stream disconnect/done;
- second-turn cache coherence;
- no blocking runtime log findings.

Failed rows:

- `dsv4_thinking_mode_max`
  - `finish="length"`, empty visible content, reasoning chars 4145.
- `dsv4_threejs_identifier_integrity`
  - deterministic request: thinking off, `temperature=0.0`, `top_p=1.0`,
    `repetition_penalty=1.0`;
  - still emitted markdown fences and corrupt identifiers:
    `THREE.PPerspectiveCamera`, `THREE.MMeshBasicMaterial`.
- `dsv4_long_context_full_output_vc_project_plan`
  - `finish="length"`, incomplete tail.
- `dsv4_long_context_full_output_game_design_long_context`
  - skipped because identifier gate failed.
- `responses_tool_history_continuation`
  - returned `READEOM.md`, consistent with exact-token/code-ish corruption.

The vMLX audit reported this static issue:

`DSV4 output-head/final-norm precision boundary requires source-vs-quant or rebuilt-artifact clearance before long-output production claims (head=U32, norm=F16)`

Interpretation:

- This is not cleared by max-output/context wiring.
- This is not cleared by Responses API routing.
- This is not cleared by DSV4 prefix/paged/L2 cache proof.
- This is not cleared by the DSML/tool-parser rows.
- This reproduces with explicit deterministic request parameters, so do not
  paper over it with hidden sampler forcing or fake defaults.

Release implication:

- vMLX can only build/release with DSV4 long-output/code quality descoped, or
  after a rebuilt/source-body DSV4 artifact passes the live gate.
- The next JANG-side path is source-vs-quant or rebuilt-artifact clearance for
  the output-head/final-norm precision boundary, followed by the same vMLX live
  gate.

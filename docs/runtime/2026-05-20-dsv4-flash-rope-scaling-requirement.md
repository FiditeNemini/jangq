# DeepSeek-V4-Flash Rope Scaling Requirement

Date: 2026-05-20

## Release-Critical Rule

Do not upload or publish a DeepSeek-V4-Flash JANG/JANGTQ artifact whose
`config.json` has missing or null `rope_scaling`.

This field is load-bearing for the local DSV4 MLX runtime. `rope_parameters`
is only a compatibility mirror for newer Transformers config handling; it does
not replace `rope_scaling` for `jang_tools.dsv4.mlx_model`.

Required block:

```json
"rope_scaling": {
  "type": "yarn",
  "factor": 16,
  "original_max_position_embeddings": 65536,
  "beta_fast": 32,
  "beta_slow": 1
}
```

The converter may also emit:

```json
"rope_parameters": {
  "rope_type": "yarn",
  "factor": 16.0,
  "original_max_position_embeddings": 65536,
  "beta_fast": 32.0,
  "beta_slow": 1.0,
  "rope_theta": 10000.0
}
```

Both keys are acceptable. `rope_parameters` alone is not acceptable.

## Why It Matters

DeepSeek-V4-Flash uses compressed-context attention layers. The source config
sets `compress_rope_theta=160000` and YaRN `rope_scaling` with factor 16 and
original context 65536.

`jang_tools.dsv4.mlx_model.ModelArgs` reads `rope_scaling` directly. If a
converter removes that key while leaving `compress_rope_theta` and
`compress_ratios`, compressed layers are built without the source YaRN scaling.
That can corrupt long/full-output behavior without an obvious load-time error.

## Patched Local Artifacts

These local artifacts were corrected before upload:

- `/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K/config.json`
- `/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANG/config.json`

The DSV4 JANG and JANGTQ converters were also corrected to preserve
`rope_scaling` and add `rope_parameters` only as a mirror.

## Required Pre-Upload Check

Run this before any JANGQ upload or release claim for DSV4 Flash:

```sh
PYTHONPATH=/Users/eric/jang/jang-tools \
python3 /Users/eric/jang/jang-tools/scripts/validate_dsv4_flash_rope_scaling.py \
  /Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K \
  /Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANG
```

Equivalent one-off config check:

```sh
jq -e '
  .model_type == "deepseek_v4" and
  .compress_rope_theta == 160000 and
  .rope_scaling.type == "yarn" and
  .rope_scaling.factor == 16 and
  .rope_scaling.original_max_position_embeddings == 65536 and
  .rope_scaling.beta_fast == 32 and
  .rope_scaling.beta_slow == 1
' config.json
```

## Still Required

This config fix removes a real artifact/runtime mismatch. It does not by
itself clear DSV4 for production. The vMLX release gate still needs live
full-output DSV4 proof across the exact chat/API/cache paths, including exact
identifier/code output and no parser/reasoning/tool leakage.

## 2026-05-20 vMLX Identifier Ablation

After restoring `rope_scaling`, vMLX still reproduced the DSV4 exact-code
identifier blocker on `DeepSeek-V4-Flash-JANGTQ-K`.

Saved proof:

- `/Users/eric/mlx/vllm-mlx/build/dsv4-identifier-count-ablation-20260520120601/result.json`

Key result:

- A one-line exact-copy prompt for
  `const renderer = new THREE.WebGLRenderer();` passed.
- Adding a preceding `THREE.Scene` line immediately corrupted both identifiers:
  `THREE.Scene` became `THREE.ScScene`, and `THREE.WebGLRenderer` became
  `THREE.WebWebGLRenderer`.
- Putting the renderer line first preserved `THREE.WebGLRenderer`, but then
  `THREE.Scene` degraded to `THREE.Sc`.

Do not treat the YaRN metadata repair as a release fix for DSV4 full-output or
exact-code generation. It is a required model/config hygiene fix. Production
clearance still needs source-vs-quant, broader rebuilt body/runtime, or another
live proof that exact identifiers and long file output pass without prompt,
parser, cache, or response-assembly leakage.

## 2026-05-20 Affine JANG Cross-Check

The same vMLX identifier-count ablation was rerun on the local affine DSV4 JANG
artifact:

- `/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANG`

Saved proof:

- `/Users/eric/mlx/vllm-mlx/build/dsv4-identifier-count-ablation-20260520122117/result.json`

Result:

- The affine JANG artifact also failed the short exact-code gate.
- `THREE.WebGLRenderer` survived in isolation, but multi-identifier prompts
  still produced corruptions such as `THREE.Sc`, `THREE.WebWebGLRenderer`,
  `THREE.PPerspectiveCamera`, `THREE.BBoxGeometry`, and
  `THREE.MMeshBasicMaterial`.
- The affine run also wrapped answers in markdown fences despite the exact-copy
  prompt.

This rules out a narrow "JANGTQ-only matmul/cache/parser" explanation for the
remaining release blocker. Treat the issue as broader DSV4 local artifact /
runtime exact-code fidelity until a source-vs-local or rebuilt-artifact proof
passes the identifier gate.

## 2026-05-20 12:34 PDT Verification Update

Focused verification from the vMLX release-hardening session:

```sh
PYTHONPATH=/Users/eric/jang/jang-tools \
/Users/eric/mlx/vllm-mlx/.venv/bin/python -m pytest \
  /Users/eric/jang/jang-tools/tests/test_dsv4_converter_contract.py -q
# 26 passed, 2 warnings

PYTHONPATH=/Users/eric/jang/jang-tools \
python3 /Users/eric/jang/jang-tools/scripts/validate_dsv4_flash_rope_scaling.py \
  /Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K \
  /Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANG
# PASS for both config.json files

cd /Users/eric/mlx/vllm-mlx
.venv/bin/python -m pytest \
  tests/test_dsv4_contract_hardening.py::test_dsv4_runtime_config_reinjects_source_yarn_rope_scaling \
  tests/test_dsv4_contract_hardening.py::test_dsv4_runtime_config_preserves_existing_rope_scaling \
  tests/test_dsv4_contract_hardening.py::test_dsv4_normalized_load_config_is_scoped_and_restored \
  tests/test_cross_matrix_audit_runner.py::test_dsv4_rope_scaling_contract_flags_null_flash_yarn_metadata \
  tests/test_cross_matrix_audit_runner.py::test_dsv4_static_audit_exposes_missing_source_rope_scaling -q
# 5 passed
```

Current interpretation:

- The converter change should stay: future DSV4 Flash artifacts must preserve
  `rope_scaling` and may add `rope_parameters` only as a mirror.
- The current local DSV4 JANGTQ-K and affine JANG artifacts already pass this
  RoPE metadata validator, so this is not enough to clear the vMLX exact-code
  release blocker.
- Requantization is not proven necessary by this RoPE/config evidence alone.
  The next useful DSV4 release gate remains source-vs-local comparison or a
  rebuilt/reselected artifact that passes the short identifier gate.

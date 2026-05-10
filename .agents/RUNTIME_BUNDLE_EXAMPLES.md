# Runtime And Bundle Examples (Local, Cache-Aware)

Status timestamp: 2026-05-09 local.

Private coordination runbook for this repo. `.agents/` is gitignored.

## Scope

- Keep examples local to this folder first.
- Keep RAM pressure low by default.
- Do not run heavy conversion or long decode unless explicitly approved.
- Use this as the shared Codex/Claude verification checklist.

## Latest execution snapshot (2026-05-09 local)

- Executed all six target bundles:
  - `ZAYA1-8B-*` (text): JANGTQ2 / JANGTQ4 / MXFP4
  - `ZAYA1-VL-8B-*` (vision): MXFP4 / JANGTQ2 / JANGTQ4
- Structure/capability gates pass for all six.
- VL bundles preserve `model_type: zaya1_vl` and image token IDs:
  - `image_token_id=262147`
  - `vision_start_token_id=255999`
  - `vision_end_token_id=256000`
- Runtime pending note:
  - Text-only ZAYA is in runtime/queue once coherence smoke is accepted.
  - ZAYA1-VL remains runtime-pending until `zaya1_vl` dispatch + image+text proof exists.

## Example A: Text ZAYA Bundle Structure Gate (Low RAM)

Target bundles:

- `/Users/eric/models/Zyphra/ZAYA1-8B-JANGTQ2`
- `/Users/eric/models/Zyphra/ZAYA1-8B-JANGTQ4`
- `/Users/eric/models/Zyphra/ZAYA1-8B-MXFP4`

Checks:

1. Structure files exist:
   - `config.json`
   - `jang_config.json`
   - `model.safetensors.index.json`
   - tokenizer/template files (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`)
   - chat template file (`chat_template.jinja` preferred; `chat_template.json` accepted when source provides it)
2. JANGTQ sidecar presence:
   - `jangtq_runtime.safetensors` required for JANGTQ bundles.
3. Index integrity:
   - all listed shards exist.
   - no accidental text/VL naming mix.
4. Capabilities contract:
   - non-thinking policy should be explicit.
   - `tool_parser` and `reasoning_parser` must match intended runtime policy.

Suggested command skeleton:

```bash
uv run --project jang-tools python - <<'PY'
from pathlib import Path
import json

bundles = [
    Path('/Users/eric/models/Zyphra/ZAYA1-8B-JANGTQ2'),
    Path('/Users/eric/models/Zyphra/ZAYA1-8B-JANGTQ4'),
    Path('/Users/eric/models/Zyphra/ZAYA1-8B-MXFP4'),
]

for b in bundles:
    print('\\n==', b.name)
    required = ['config.json', 'jang_config.json', 'model.safetensors.index.json']
    for r in required:
        p = b / r
        print(r, 'OK' if p.exists() else 'MISSING')
    for r in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
        p = b / r
        print(r, 'OK' if p.exists() else 'MISSING')
    tpl_jinja = (b / 'chat_template.jinja').exists()
    tpl_json = (b / 'chat_template.json').exists()
    print('chat_template', 'OK' if (tpl_jinja or tpl_json) else 'MISSING')
    idx = json.loads((b / 'model.safetensors.index.json').read_text())
    missing = [s for s in set(idx.get('weight_map', {}).values()) if not (b / s).exists()]
    print('missing_shards', len(missing))
    sidecar = (b / 'jangtq_runtime.safetensors').exists()
    print('sidecar', sidecar)
PY
```

## Example B: Capabilities Verification Gate

Goal: ensure generated bundle capability stamps and recomputed capabilities agree before upload.

Suggested command:

```bash
uv run --project jang-tools python - <<'PY'
from pathlib import Path
from jang_tools.capabilities import verify_directory

for p in sorted(Path('/Users/eric/models/Zyphra').glob('ZAYA1-8B-*')):
    if p.is_dir():
        ok, msg = verify_directory(p)
        print(p.name, ok, msg)
PY
```

Pass criteria:

- All target bundles return `ok=True`.
- Any mismatch is a release blocker.

## Example C: Text Runtime Smoke Gate (Cache-Sensitive)

Goal: minimal proof that target runtime can load the bundle and run a short generation path with correct cache behavior.

Requirements:

- Use short prompt and low token count.
- Single-bundle smoke first.
- Avoid batch stress until single smoke passes.

What to capture:

- model id/path
- runtime commit/path used
- one short generated output
- any cache warning or fallback messages

## Example D: ZAYA1-VL Metadata Gate (No Weight Download)

Goal: confirm VL identity without high RAM or long downloads.

Checks from lightweight metadata/config fetch:

- `model_type = zaya1_vl`
- `architectures` includes `Zaya1VLForConditionalGeneration`
- vision processor config present
- image token ids present
- index contains vision and lora tensor families

Pass criteria:

- If any of these are missing, block VL conversion/upload planning and re-verify source.

Target path map for this pass:

- `~/models/Zyphra/ZAYA1-VL-8B-MXFP4`
- `~/models/Zyphra/ZAYA1-VL-8B-JANGTQ2`
- `~/models/Zyphra/ZAYA1-VL-8B-JANGTQ4`

## Example E: Cache Topology Documentation Gate

Before public docs/readmes:

1. State the cache topology per bundle family:
   - dense KV
   - SSM/gated hybrid
   - SWA hybrid
   - ZAYA CCA hybrid (`KV + conv_state + prev_hs`)
2. State prefix/paged-cache boundary for ZAYA CCA clearly:
   - no KV-only restore claims.
3. State media-salt expectations for VL paths.

## Example F: Osaurus Runtime Support Matrix Gate

Per bundle family, document:

- runtime(s) currently supported
- required runtime pin/commit
- known unsupported paths and expected failure mode
- pending fixes required before enabling in catalog

Minimum rows:

- ZAYA1-8B text bundles
- ZAYA1-VL-8B bundles
- Qwen2.5-VL reference

## Stop Conditions

Stop and escalate before continuing if any occur:

- Capability verification mismatch.
- Missing sidecar on JANGTQ bundle.
- Runtime claim cannot be tied to a concrete source path/commit.
- Text/VL naming ambiguity in output bundle directories.
- Attempted public `-VL` upload from text-only `ZAYA1-8B-*` bundles.

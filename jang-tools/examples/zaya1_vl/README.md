# ZAYA1-VL-8B Runtime Prep

This folder tracks the VL workflow for `Zyphra/ZAYA1-VL-8B` and keeps the
local handoff explicit before any Osaurus publish.

## Source

Expected source directory:

```sh
/Users/eric/models/Zyphra/ZAYA1-VL-8B
```

## Runbook

Inspect source metadata without loading weights:

```sh
python3 00_inspect_source.py
```

Confirm local source-to-bundle mapping:

```sh
python3 - <<'PY'
from pathlib import Path
from jang_tools.capabilities import verify_directory

for name in [
    "ZAYA1-VL-8B-MXFP4",
    "ZAYA1-VL-8B-JANGTQ2",
    "ZAYA1-VL-8B-JANGTQ4",
]:
    p = Path("/Users/eric/models/Zyphra") / name
    ok, msg = verify_directory(p)
    print(name, ok, msg)
PY
```

Check JANGTQ sidecars:

```sh
test -f /Users/eric/models/Zyphra/ZAYA1-VL-8B-MXFP4/jangtq_runtime.safetensors || true
test -f /Users/eric/models/Zyphra/ZAYA1-VL-8B-JANGTQ2/jangtq_runtime.safetensors && echo "VL JANGTQ2 sidecar ok"
test -f /Users/eric/models/Zyphra/ZAYA1-VL-8B-JANGTQ4/jangtq_runtime.safetensors && echo "VL JANGTQ4 sidecar ok"
```

## Conversion Paths

Converter entry points:

- `jang-convert-zaya1-vl-mxfp4`
- `jang-convert-zaya1-vl-jangtq`

Safe dry runs:

```sh
jang-convert-zaya1-vl-mxfp4 /Users/eric/models/Zyphra/ZAYA1-VL-8B /tmp/ZAYA1-VL-8B-MXFP4 --dry-run
jang-convert-zaya1-vl-jangtq /Users/eric/models/Zyphra/ZAYA1-VL-8B /tmp/ZAYA1-VL-8B-JANGTQ2 JANGTQ2 --dry-run
```

`JANGTQ3` is intentionally unsupported for this family until the packing,
group-size, and runtime-contract constraints are solved and independently
validated.

## Output Contract

Generated bundles use naming to avoid confusion:

- `ZAYA1-VL-8B-MXFP4`
- `ZAYA1-VL-8B-JANGTQ2`
- `ZAYA1-VL-8B-JANGTQ4`

Do not repurpose text-only `ZAYA1-8B-*` directories for VL.

## Runtime Status

Source identity:

- `model_type=zaya1_vl`
- `architectures=["Zaya1VLForConditionalGeneration"]`
- `image_token_id=262147`
- `vision_start_token_id=255999`
- `vision_end_token_id=256000`

Current release posture:

- local MXFP4/JANGTQ2/JANGTQ4 bundle structure and capabilities pass
- `supports_thinking=false` is the current ZAYA family policy
- JANGTQ sidecar is required only for JANGTQ variants
- Swift/Python runtime support remains pending until `zaya1_vl` dispatch,
  image-token handling, media salt, CCA cache state, and image+text smoke all
  pass

Osaurus-facing readmes must state that pending status clearly. A VL upload must
not claim production support until the runtime proof exists.

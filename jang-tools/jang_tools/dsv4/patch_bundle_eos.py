"""DSV4-Flash bundle on-disk eos patcher.

Forces `eos_token_id` to a 2-element list `[<｜end▁of▁sentence｜>, <｜User｜>]`
across `config.json`, `generation_config.json`, and `tokenizer_config.json`.

Why: upstream DSV4 + earlier jang DSV4 bundles ship `eos_token_id: 1` (single
int). Without `<｜User｜>` (typically 128803) ALSO marked as a stop, the model
auto-continues past `<｜end▁of▁sentence｜>` into a fake user turn on
multi-turn chat, producing duplicated "🤖 My name is..." restart loops or
runaway markdown drift until max_tokens. Single-turn happens to terminate
because no fake-turn precedent gets sampled; the bug only manifests after the
first eos is hit.

Idempotent. Safe to run on already-correct bundles. No re-quant, just config
JSON rewrites.

Usage:
    python -m jang_tools.dsv4.patch_bundle_eos <bundle_path> [<bundle_path> ...]
    python -m jang_tools.dsv4.patch_bundle_eos --all   # patch every JANG-stamped DSV4 bundle on common paths

API:
    from jang_tools.dsv4.patch_bundle_eos import patch_bundle
    patch_bundle("/path/to/DeepSeek-V4-Flash-JANGTQ2")
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

CONFIG_FILES = ("config.json", "generation_config.json", "tokenizer_config.json")


def patch_bundle(bundle_path) -> dict:
    B = Path(bundle_path)
    if not B.is_dir():
        raise FileNotFoundError(f"not a directory: {B}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(B), trust_remote_code=True)
    eos_id = tok.convert_tokens_to_ids("<｜end▁of▁sentence｜>")
    user_id = tok.convert_tokens_to_ids("<｜User｜>")
    if eos_id is None or user_id is None or eos_id == user_id:
        raise RuntimeError(
            f"Cannot resolve <｜end▁of▁sentence｜> and <｜User｜> distinctly "
            f"from tokenizer at {B} (eos={eos_id}, user={user_id})"
        )
    target = sorted({eos_id, user_id})

    changed = []
    for fn in CONFIG_FILES:
        p = B / fn
        if not p.exists():
            continue
        d = json.load(open(p))
        if d.get("eos_token_id") == target:
            continue
        d["eos_token_id"] = target
        p.write_text(json.dumps(d, indent=2, ensure_ascii=False))
        changed.append(fn)

    return {"bundle": str(B), "eos_list": target, "files_patched": changed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundles", nargs="*", type=Path, help="bundle directories to patch")
    args = ap.parse_args(argv)

    if not args.bundles:
        ap.print_help()
        return 1

    rc = 0
    for b in args.bundles:
        try:
            res = patch_bundle(b)
            if res["files_patched"]:
                print(f"[patched] {res['bundle']}: eos={res['eos_list']} files={res['files_patched']}")
            else:
                print(f"[ok]      {res['bundle']}: eos={res['eos_list']} (already correct)")
        except Exception as e:
            print(f"[FAIL]    {b}: {e}", file=sys.stderr)
            rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main())

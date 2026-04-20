"""Publish a converted JANG/JANGTQ model to HuggingFace Hub.

Wraps huggingface_hub.upload_folder with auto-generated model card.
Requires HF_HUB_TOKEN env var or a token file path via --token.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any

from .modelcard import generate_card


def cmd_publish(args) -> None:
    model_dir = Path(args.model)
    if not model_dir.exists():
        print(f"ERROR: model dir not found: {model_dir}", file=sys.stderr)
        sys.exit(2)

    # Resolve HF token
    token = None
    if args.token:
        token_path = Path(args.token)
        if token_path.exists():
            token = token_path.read_text().strip()
        else:
            token = args.token   # assume literal token string
    else:
        token = os.environ.get("HF_HUB_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: set HF_HUB_TOKEN env var or pass --token", file=sys.stderr)
        sys.exit(2)

    # Ensure README.md exists with a generated card
    readme = model_dir / "README.md"
    if not readme.exists() or args.regenerate_card:
        try:
            card = generate_card(model_dir)
            readme.write_text(card)
        except Exception as e:
            print(f"ERROR: failed to generate model card: {e}", file=sys.stderr)
            sys.exit(3)

    if args.dry_run:
        result = {
            "dry_run": True,
            "repo": args.repo,
            "private": args.private,
            "model_card_path": str(readme),
            "files_count": sum(1 for _ in model_dir.rglob("*") if _.is_file()),
            "total_size_bytes": sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file()),
        }
    else:
        try:
            from huggingface_hub import HfApi, create_repo
        except ImportError:
            print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
            sys.exit(2)

        try:
            create_repo(repo_id=args.repo, token=token, private=args.private, exist_ok=True)
            api = HfApi(token=token)
            info = api.upload_folder(
                folder_path=str(model_dir),
                repo_id=args.repo,
                token=token,
                commit_message=f"Upload {model_dir.name} via jang-tools",
            )
            result = {
                "dry_run": False,
                "repo": args.repo,
                "url": f"https://huggingface.co/{args.repo}",
                "commit_url": str(info),
            }
        except Exception as e:
            print(f"ERROR: publish failed: {type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(3)

    if args.json:
        print(json.dumps(result, indent=None))
    else:
        if result.get("dry_run"):
            print(f"DRY RUN — would publish {result['files_count']} files ({result['total_size_bytes']/1e9:.2f} GB) to {result['repo']}")
        else:
            print(f"Published to: {result['url']}")


def register(subparsers) -> None:
    p = subparsers.add_parser("publish", help="Publish converted model to HuggingFace Hub")
    p.add_argument("--model", required=True, help="Path to converted model dir")
    p.add_argument("--repo", required=True, help="Target HF repo id (e.g., my-org/my-model-JANG_4K)")
    p.add_argument("--private", action="store_true", help="Create as private repo")
    p.add_argument("--token", help="HF token (env HF_HUB_TOKEN is used if not passed)")
    p.add_argument("--regenerate-card", action="store_true",
                   help="Overwrite existing README.md with freshly generated card")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan but don't upload — returns file count + size")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_publish)

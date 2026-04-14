"""`jang spec ...` subcommand implementations."""

from __future__ import annotations

import argparse
from pathlib import Path

from .builder import JangSpecBuilder
from .reader import JangSpecReader


def cmd_build(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    if out.exists() and not args.force:
        print(f"error: output directory {out} already exists (use --force to overwrite)")
        return 1
    builder = JangSpecBuilder(
        source_dir=source, out_dir=out, write_streaming=args.streaming)
    builder.build()
    print(f"  built bundle: {out}")
    print(f"    layers:       {builder.n_layers}")
    print(f"    experts/layer:{builder.n_experts_per_layer}")
    print(f"    hot_core:     {builder.hot_core_bytes / 1e9:.2f} GB")
    print(f"    expert_bytes: {builder.expert_bytes / 1e9:.2f} GB")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    reader = JangSpecReader(Path(args.bundle))
    m = reader.manifest
    print(f"  bundle:        {args.bundle}")
    print(f"  source jang:   {m.source_jang}")
    print(f"  arch:          {m.target_arch}")
    print(f"  n_layers:      {m.n_layers}")
    print(f"  experts/layer: {m.n_experts_per_layer}")
    print(f"  top_k:         {m.target_top_k}")
    print(f"  hot_core:      {m.hot_core_bytes / 1e9:.2f} GB")
    print(f"  expert_bytes:  {m.expert_bytes / 1e9:.2f} GB")
    print(f"  draft:         {m.has_draft}")
    print(f"  router_prior:  {m.has_router_prior}")
    return 0


def register_subparsers(spec_parser: argparse.ArgumentParser) -> None:
    sub = spec_parser.add_subparsers(dest="spec_cmd", required=True)

    build = sub.add_parser("build", help="Build a .jangspec bundle from a source JANG MoE model")
    build.add_argument("source", help="Path to source JANG model directory")
    build.add_argument("--out", required=True, help="Path to output .jangspec directory")
    build.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    build.add_argument(
        "--streaming", action="store_true",
        help="Also emit the per-expert blob format (experts.jsidx + experts-*.bin). "
             "Used by future SSD streaming runtimes; doubles bundle size on disk. "
             "The default Swift loader never reads it.")
    build.set_defaults(func=cmd_build)

    inspect = sub.add_parser("inspect", help="Inspect a .jangspec bundle")
    inspect.add_argument("bundle", help="Path to .jangspec directory")
    inspect.set_defaults(func=cmd_inspect)

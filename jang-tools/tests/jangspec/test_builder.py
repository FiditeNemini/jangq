"""Integration tests for jang_tools.jangspec.builder against a real JANG model."""

from pathlib import Path

from jang_tools.jangspec import format as fmt
from jang_tools.jangspec.builder import JangSpecBuilder
from jang_tools.jangspec.index import read_index
from jang_tools.jangspec.manifest import load_manifest


def test_build_creates_all_bundle_files(jangspec_fixture_model: Path, tmp_path: Path):
    out = tmp_path / "fixture.jangspec"
    builder = JangSpecBuilder(source_dir=jangspec_fixture_model, out_dir=out)
    builder.build()

    # Required files exist
    assert (out / fmt.MANIFEST_FILENAME).exists()
    assert (out / fmt.INDEX_FILENAME).exists()
    assert (out / fmt.HOT_CORE_FILENAME).exists()
    assert (out / "tokenizer.json").exists()
    # At least one experts-NNNNN.bin was emitted
    shards = sorted(out.glob("target/experts-*.bin"))
    assert len(shards) >= 1

    # Manifest is loadable and reports an MoE model
    manifest = load_manifest(out / fmt.MANIFEST_FILENAME)
    assert manifest.n_experts_per_layer > 1
    assert manifest.n_layers > 0
    assert manifest.n_experts_total == manifest.n_layers * manifest.n_experts_per_layer
    assert manifest.has_draft is False
    assert manifest.has_router_prior is False

    # Index is loadable and has one entry per (layer, expert)
    idx = read_index(out / fmt.INDEX_FILENAME)
    assert idx.n_layers == manifest.n_layers
    assert idx.n_experts_per_layer == manifest.n_experts_per_layer
    assert len(idx.entries) == manifest.n_experts_total

    # Every blob offset is 4 KB-aligned and fits inside its file
    for e in idx.entries:
        assert e.offset % fmt.BLOB_ALIGNMENT == 0, f"unaligned blob at {e}"
        shard = out / f"target/experts-{e.file_id:05d}.bin"
        assert shard.exists()
        assert e.offset + e.nbytes <= shard.stat().st_size
